"""Extração das características da v4 (uma passada pelo modelo congelado), retomável e com orçamento.

Os áudios pendentes do plano são lidos por UM DataLoader (workers não persistentes: vivem só durante a
passada), em lotes de leitura que nunca misturam dois shards. Dentro de cada shard, os áudios são lidos
ordenados por duração.

Lotes da GPU com formato fixo ("baldes"; execução, não método): cada clipe vai para o balde do seu
tamanho arredondado para cima a um múltiplo de BALDE_S segundos (ex.: 2, 4, 6, 8 s); cada balde tem um nº
fixo de linhas. O balde
vai à GPU quando enche; no fim do shard, os baldes incompletos são completados com linhas vazias, que são
descartadas. Motivo medido: no Windows, cada formato novo de lote faz a GPU carregar mais variantes de
kernels na memória da CPU; com formatos variados o processo chegou a 13-14 GB reservados e ficou ~2x mais
lento; com 1 formato, 4,5 GB; com 25, 10,5 GB (estável). O enchimento é mascarado pela attention_mask, e
média e desvio usam só os quadros válidos, então o resultado de cada clipe não depende do balde (dentro da
tolerância do float16, conferida no teste QUICK).
"""
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from src.cache_v4 import COLS_MANIFEST, gravar_shard, shard_pronto
from src.preproc import COLS_ESTAT, SR, ClipesV4, juntar_clipes
from src.ssl import DIM, N_SAIDAS, representar


class LotesFixos(Sampler):
    def __init__(self, lotes):
        self.lotes = lotes

    def __iter__(self):
        return iter(self.lotes)

    def __len__(self):
        return len(self.lotes)


def montar_lotes(dur_s, max_amostras, max_clipes, max_s=8.0, sr=SR):
    """Lotes de LEITURA: posições consecutivas (já ordenadas por duração) com (nº de clipes × maior clipe)
    até `max_amostras` e no máximo `max_clipes`. A duração do manifest (antes do corte) é um teto do tamanho."""
    lotes, atual, maior = [], [], 0
    for i, d in enumerate(dur_s):
        n = int(min(d, max_s) * sr) + 1
        if atual and (max(maior, n) * (len(atual) + 1) > max_amostras or len(atual) >= max_clipes):
            lotes.append(atual)
            atual, maior = [], 0
        atual.append(i)
        maior = max(maior, n)
    if atual:
        lotes.append(atual)
    return lotes


class Baldes:
    """Junta clipes por tamanho (1, 2, ..., `max_s` s) e manda cada balde cheio à GPU com formato fixo."""

    def __init__(self, model, hp, device="cuda", log=print):
        self.model, self.hp, self.device, self.log = model, hp, device, log
        self.max_s = int(math.ceil(hp["MAX_S"]))
        self.largura = int(hp.get("BALDE_S", 1))           # largura de cada balde, em segundos
        self.linhas = {s: int(min(hp["MAX_CLIPES_LOTE"], hp["MAX_AMOSTRAS_LOTE"] // (s * SR)))
                       for s in range(self.largura, self.max_s + 1, self.largura)}
        self.fila = {s: [] for s in self.linhas}
        self.refeitos_bf16 = 0
        self.lotes_gpu = 0
        self.linhas_vazias = 0

    def balde(self, n_amostras):
        s = self.largura * math.ceil(n_amostras / (self.largura * SR))
        return int(min(max(self.linhas), max(self.largura, s)))

    def _rodar(self, s, itens):
        clipes, pos = [c for c, _ in itens], np.array([p for _, p in itens])
        forma = (self.linhas[s], s * SR)
        mu, sd, nq = representar(self.model, clipes, self.device, self.hp["DTYPE"], forma=forma)
        dt = self.hp["DTYPE"]
        if not (torch.isfinite(mu).all() and torch.isfinite(sd).all()):
            # fp16 deu NaN/infinito neste lote: refaz em bfloat16 (mesma faixa do float32) e registra
            self.model.to(torch.bfloat16)
            mu, sd, nq = representar(self.model, clipes, self.device, "bfloat16", forma=forma)
            self.model.to(torch.float16 if self.hp["DTYPE"] == "float16" else torch.float32)
            self.refeitos_bf16 += len(clipes)
            dt = "bfloat16"
            self.log(f"  aviso: lote com NaN/infinito em {self.hp['DTYPE']}; refeito em bfloat16 ({len(clipes)} clipes)")
        mu16, sd16 = mu.half().cpu().numpy(), sd.half().cpu().numpy()
        if not (np.isfinite(mu16).all() and np.isfinite(sd16).all()):
            raise FloatingPointError("valor fora da faixa do float16 ao gravar")
        self.lotes_gpu += 1
        self.linhas_vazias += self.linhas[s] - len(clipes)
        return pos, mu16, sd16, nq.numpy(), dt

    def adicionar(self, clipes, posicoes):
        """Põe os clipes nos baldes; devolve a lista de resultados dos baldes que encheram."""
        prontos = []
        for c, p in zip(clipes, posicoes):
            s = self.balde(c.numel())
            self.fila[s].append((c, int(p)))
            if len(self.fila[s]) == self.linhas[s]:
                prontos.append(self._rodar(s, self.fila[s]))
                self.fila[s] = []
        return prontos

    def esvaziar(self):
        """Roda os baldes incompletos (fim de shard)."""
        prontos = [self._rodar(s, f) for s, f in self.fila.items() if f]
        self.fila = {s: [] for s in self.linhas}
        return prontos


def extrair(df, plano, model, hp, cache_dir, orcamento, cfg_extracao, log=print, device="cuda",
            num_workers=4, so_conjuntos=None, max_shards=None):
    """Extrai os shards pendentes do `plano` na ordem, até acabar o plano ou o `orcamento`.

    Antes de começar cada shard, confere se o tempo restante cabe nele (pela vazão medida até agora,
    com 20% de folga); se não couber, para. Devolve um dicionário com o que foi feito.
    """
    pend = []
    plano = plano.sort_values("ordem")
    grupos = plano.groupby(["conjunto", "shard"], sort=False)
    for conjunto, shard in plano.drop_duplicates(["conjunto", "shard"])[["conjunto", "shard"]].itertuples(index=False):
        if so_conjuntos and conjunto not in so_conjuntos:
            continue
        if not shard_pronto(cache_dir, conjunto, shard):
            pend.append((conjunto, int(shard), grupos.get_group((conjunto, shard))))
    if max_shards is not None:
        pend = pend[:max_shards]
    if not pend:
        return {"shards_feitos": 0, "parou_por": "plano completo"}

    # um Dataset com todos os áudios pendentes; lotes de leitura por shard
    blocos, lotes, inicio = [], [], 0
    for conjunto, shard, g in pend:
        g = g.assign(duration_s=df.loc[g.idx_manifest.values, "duration_s"].values)
        g = g.assign(posicao=np.arange(len(g))).sort_values(["duration_s", "posicao"], kind="stable")
        blocos.append(g)
        for b in montar_lotes(g.duration_s.values, hp["MAX_AMOSTRAS_LOTE"], hp["MAX_CLIPES_LOTE"]):
            lotes.append([inicio + i for i in b])
        inicio += len(g)
    todos = pd.concat(blocos, ignore_index=True)
    # o corte de silêncio é por conjunto (os *_semcorte não cortam); o Dataset recebe um valor por áudio
    ds = ClipesV4(df.loc[todos.idx_manifest.values, "path"].values, todos.posicao.values,
                  cortar_silencio=todos.cortar_silencio.values.astype(bool), top_db=hp["TOP_DB"],
                  min_s=hp["MIN_S"], max_s=hp["MAX_S"])
    dl = DataLoader(ds, batch_sampler=LotesFixos(lotes), num_workers=num_workers, collate_fn=juntar_clipes,
                    persistent_workers=False, prefetch_factor=4 if num_workers else None)

    # quantos lotes de leitura tem cada shard
    shard_do_item = np.repeat(np.arange(len(pend)), [len(b) for b in blocos])
    lotes_por_shard = np.bincount([shard_do_item[b[0]] for b in lotes], minlength=len(pend))

    baldes = Baldes(model, hp, device, log)
    feitos, n_clipes, t_ini = 0, 0, time.time()
    parou_por = "plano completo"
    atual, buf, lotes_vistos = -1, None, 0
    it = iter(dl)
    k = 0

    def guardar(resultados):
        for pos, mu16, sd16, nq, dt in resultados:
            buf["media"][pos], buf["desvio"][pos] = mu16, sd16
            buf["nq"][pos], buf["dtype"][pos], buf["ok"][pos] = nq, dt, True

    while k < len(pend):
        conjunto, shard, g = pend[k]
        if atual != k:
            # antes de começar um shard: o orçamento cabe?
            vazao = n_clipes / max(time.time() - t_ini, 1e-9) if n_clipes else hp.get("VAZAO_ESTIMADA", 20.0)
            precisa = 1.2 * len(g) / max(vazao, 1e-3)
            if orcamento.resta() < precisa:
                parou_por = (f"orçamento: restam {orcamento.resta() / 60:.1f} min, o shard {conjunto}/{shard} "
                             f"precisa de ~{precisa / 60:.1f} min")
                break
            atual, lotes_vistos, t_shard = k, 0, time.time()
            n = len(g)
            buf = {"media": np.zeros((n, N_SAIDAS, DIM), np.float16), "desvio": np.zeros((n, N_SAIDAS, DIM), np.float16),
                   "est": np.full((n, len(COLS_ESTAT)), np.nan), "nq": np.zeros(n, np.int64),
                   "dtype": np.array([hp["DTYPE"]] * n, dtype=object), "ok": np.zeros(n, bool)}
        clipes, est, pos = next(it)
        buf["est"][pos] = est
        guardar(baldes.adicionar(clipes, pos))
        lotes_vistos += 1
        if lotes_vistos == lotes_por_shard[k]:
            guardar(baldes.esvaziar())
            assert buf["ok"].all(), "faltou algum áudio no shard"
            ordem_orig = g.reset_index(drop=True)
            meta = df.loc[ordem_orig.idx_manifest.values, COLS_MANIFEST].reset_index(drop=True)
            meta.insert(0, "idx_manifest", ordem_orig.idx_manifest.values)
            meta["conjunto"], meta["cortar_silencio"] = conjunto, bool(ordem_orig.cortar_silencio.iloc[0])
            for j, c in enumerate(COLS_ESTAT):
                meta[c] = buf["est"][:, j]
            meta["n_quadros"], meta["dtype"] = buf["nq"], buf["dtype"].astype(str)
            seg = time.time() - t_shard
            info = {"conjunto": conjunto, "shard": shard, "n": n, "segundos": round(seg, 1),
                    "clipes_por_s": round(n / max(seg, 1e-9), 2), "extracao": cfg_extracao,
                    "num_workers": num_workers, "concluido_em": datetime.now().isoformat(timespec="seconds")}
            gravar_shard(cache_dir, conjunto, shard, buf["media"], buf["desvio"], meta, info)
            feitos += 1
            n_clipes += n
            orcamento.gravar(ultimo_shard=f"{conjunto}/{shard}")
            log(f"  shard {conjunto}/{shard:05d}: {n} clipes em {seg:.0f} s ({n / max(seg, 1e-9):.1f} clipes/s) | "
                f"extração acumulada {orcamento.gasto() / 3600:.2f} h de {orcamento.limite_s / 3600:.1f} h")
            k += 1
    del it, dl
    seg = time.time() - t_ini
    return {"shards_feitos": feitos, "clipes": n_clipes, "segundos": round(seg, 1),
            "clipes_por_s": round(n_clipes / max(seg, 1e-9), 2), "refeitos_bf16": baldes.refeitos_bf16,
            "lotes_gpu": baldes.lotes_gpu, "linhas_vazias": baldes.linhas_vazias, "parou_por": parou_por}
