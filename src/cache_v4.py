"""Plano de extração e cache de características da v4, em shards retomáveis.

Cada shard é uma pasta `<cache>/<conjunto>/shard_NNNNN/` com:
- `media.npy` e `desvio.npy`: float16, (n, 25, 1024): média e desvio no tempo de cada saída do modelo;
- `meta.parquet`: uma linha por áudio (índice do manifest, rótulo, fonte, estatísticas da carga...);
- `concluido.json`: marcador, gravado POR ÚLTIMO.
A pasta é escrita como `shard_NNNNN.tmp` e só é renomeada quando está completa (atômico). Pastas `.tmp`
que sobraram de uma queda são apagadas na próxima execução e o shard é refeito.

Conjuntos (ordem de extração):
1. `dev_shift`: o dev_shift inteiro (16 locutores do In-the-Wild);
2. `val`: subamostra da validação, estratificada por fonte e rótulo, com TODOS os áudios das fontes
   pequenas;
3. `treino`: shards sucessivos, cada um um sorteio balanceado 50/50 real/fake com teto por fonte e por
   gerador (as funções de src/sampling.py), sem repetir áudio entre shards. Qualquer prefixo de shards
   concluídos é um conjunto de treino válido;
4. opcional (`*_semcorte`): os mesmos áudios de um subconjunto pequeno, SEM corte de silêncio.
"""
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.arquivos import gravar_atomico, salvar_json, salvar_parquet
from src.sampling import calcular_cotas, montar_grupos

COLS_MANIFEST = ["path", "source_dataset", "label", "generator", "fake_type", "speaker_id", "duration_s",
                 "sample_rate"]


# ---------------------------------------------------------------- plano
def amostra_val(df, maximo, limite_pequena, seed):
    """Subamostra da validação: fontes com até `limite_pequena` áudios entram inteiras; nas outras, a
    mesma fração em cada (fonte, rótulo), para o total dar `maximo`."""
    val = df[df.split == "val"]
    tam = val.groupby("source_dataset").size()
    pequenas = tam[tam <= limite_pequena].index
    fixas = val[val.source_dataset.isin(pequenas)]
    grandes = val[~val.source_dataset.isin(pequenas)]
    if maximo is None or len(val) <= maximo:
        return val
    frac = max(0.0, (maximo - len(fixas)) / max(len(grandes), 1))
    partes = [fixas] + [g.sample(int(round(frac * len(g))), random_state=seed)
                        for _, g in grandes.groupby(["source_dataset", "label"])]
    return pd.concat(partes)


def plano_treino(pool, por_shard, max_shards, seed):
    """Lista de shards de treino: cada um com `por_shard/2` reais e `por_shard/2` fakes.

    Usa as regras de src/sampling.py (cotas iguais por fonte e, dentro da fonte, por gerador; sobra de
    um grupo pequeno vai para os outros), mas SEM repetir áudio: a cota de cada shard é calculada sobre
    o que ainda não foi sorteado. Para quando uma das classes não tem mais áudio para completar um shard.
    `pool` tem o índice do manifest; devolve uma lista de arrays de índices do manifest.
    """
    p = pool.reset_index().rename(columns={"index": "idx_manifest"})
    grupos = montar_grupos(p)
    rng = np.random.default_rng(seed)
    filas = {k: rng.permutation(v) for k, v in sorted(grupos.items())}
    pos = {k: 0 for k in filas}
    shards = []
    while len(shards) < max_shards:
        restantes = {k: f[pos[k]:] for k, f in filas.items()}
        cotas = calcular_cotas(restantes, por_shard // 2)
        if any(sum(c for (l, _, _), c in cotas.items() if l == lab) < por_shard // 2 for lab in (0, 1)):
            break
        idx = []
        for k, c in cotas.items():
            idx.append(filas[k][pos[k]:pos[k] + c])
            pos[k] += c
        shards.append(p.idx_manifest.values[rng.permutation(np.concatenate(idx))])
    return shards


def montar_plano(df, hp):
    """DataFrame (idx_manifest, conjunto, shard, cortar_silencio) na ordem de extração."""
    seed, tam = hp["SEED"], hp["TAM_SHARD"]
    linhas = []

    def em_shards(idx, conjunto, cortar=True, tamanho=tam):
        for s, i in enumerate(range(0, len(idx), tamanho)):
            parte = idx[i:i + tamanho]
            linhas.append(pd.DataFrame({"idx_manifest": parte, "conjunto": conjunto, "shard": s,
                                        "cortar_silencio": cortar}))

    rng = np.random.default_rng(seed)
    dev = df[df.eval_subset == "dev_shift"]
    if hp.get("DEV_MAX"):
        dev = dev.sample(min(len(dev), hp["DEV_MAX"]), random_state=seed)
    dev_idx = rng.permutation(dev.index.values)
    em_shards(dev_idx, "dev_shift")
    val = amostra_val(df, hp["VAL_MAX"], hp["VAL_FONTE_PEQUENA"], seed)
    val_idx = rng.permutation(val.index.values)       # embaralhado: cada shard é ~estratificado
    em_shards(val_idx, "val")
    tr = plano_treino(df[df.split == "train"], tam, hp["TREINO_MAX_SHARDS"], seed)
    for s, idx in enumerate(tr):
        linhas.append(pd.DataFrame({"idx_manifest": idx, "conjunto": "treino", "shard": s,
                                    "cortar_silencio": True}))
    if hp.get("SEMCORTE"):
        sc = hp["SEMCORTE"]
        tr_sc = np.concatenate(tr[: sc["treino_shards"]]) if tr else np.array([], dtype=int)
        em_shards(tr_sc, "treino_semcorte", cortar=False)
        em_shards(val_idx[: sc["val"]], "val_semcorte", cortar=False)
        em_shards(dev_idx[: sc["dev_shift"]], "dev_shift_semcorte", cortar=False)
    plano = pd.concat(linhas, ignore_index=True)
    plano["ordem"] = np.arange(len(plano))
    return plano


def carregar_ou_criar_plano(df, hp, cache_dir, cfg_extracao, manifest_sha):
    """Cria o plano na 1ª vez; nas seguintes, confere que nada mudou (senão para, sem apagar nada)."""
    cache_dir = Path(cache_dir)
    arq, arq_cfg = cache_dir / "plano.parquet", cache_dir / "plano.json"
    cfg = {"extracao": cfg_extracao, "manifest_sha256": manifest_sha,
           "plano": {k: hp[k] for k in ("SEED", "TAM_SHARD", "VAL_MAX", "VAL_FONTE_PEQUENA", "TREINO_MAX_SHARDS",
                                        "DEV_MAX", "SEMCORTE")}}
    if arq.exists():
        salvo = json.loads(arq_cfg.read_text(encoding="utf-8"))
        if salvo != json.loads(json.dumps(cfg, default=str)):
            raise RuntimeError(f"{arq_cfg} é de outra configuração. Nada foi apagado; confira à mão.\n"
                               f"salvo: {salvo}\nagora: {cfg}")
        return pd.read_parquet(arq)
    plano = montar_plano(df, hp)
    cache_dir.mkdir(parents=True, exist_ok=True)
    salvar_parquet(plano, arq)
    salvar_json(cfg, arq_cfg)
    return plano


# ---------------------------------------------------------------- shards
def pasta_shard(cache_dir, conjunto, shard):
    return Path(cache_dir) / conjunto / f"shard_{int(shard):05d}"


def shard_pronto(cache_dir, conjunto, shard):
    return (pasta_shard(cache_dir, conjunto, shard) / "concluido.json").exists()


def limpar_temporarios(cache_dir):
    """Apaga pastas `shard_*.tmp` que sobraram de uma execução interrompida (só dentro do cache)."""
    n = 0
    for d in Path(cache_dir).glob("*/shard_*.tmp"):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


def gravar_shard(cache_dir, conjunto, shard, media, desvio, meta, info):
    final = pasta_shard(cache_dir, conjunto, shard)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "media.npy", media)
    np.save(tmp / "desvio.npy", desvio)
    meta.to_parquet(tmp / "meta.parquet", index=False)
    (tmp / "concluido.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str),
                                        encoding="utf-8")
    if final.exists():          # pasta final sem marcador (não deveria acontecer): substitui
        shutil.rmtree(final)
    os.replace(tmp, final)


# ---------------------------------------------------------------- orçamento cumulativo
class Orcamento:
    """Tempo de extração ACUMULADO entre tentativas, gravado em `arq` (json) a cada `intervalo` s.

    Se o processo cair, perde-se no máximo `intervalo` s de contagem. O tempo conta do início do processo
    de extração (inclui carregar o modelo), então é conservador.
    """

    def __init__(self, arq, limite_h, intervalo=30):
        self.arq, self.limite_s, self.intervalo = Path(arq), float(limite_h) * 3600, intervalo
        self.estado = json.loads(self.arq.read_text(encoding="utf-8")) if self.arq.exists() else {}
        self.base = float(self.estado.get("segundos_extracao", 0.0))
        self.t0 = time.time()
        self._parar = threading.Event()
        self._lock = threading.Lock()
        self.estado.setdefault("tentativas_extracao", 0)
        self.estado["tentativas_extracao"] += 1
        self.gravar()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def gasto(self):
        return self.base + (time.time() - self.t0)

    def resta(self):
        return self.limite_s - self.gasto()

    def gravar(self, **extra):
        with self._lock:
            self.estado.update(extra)
            self.estado["segundos_extracao"] = round(self.gasto(), 1)
            self.estado["horas_extracao"] = round(self.gasto() / 3600, 3)
            self.estado["orcamento_h"] = self.limite_s / 3600
            self.estado["atualizado_em"] = datetime.now().isoformat(timespec="seconds")
            txt = json.dumps(self.estado, ensure_ascii=False, indent=2, default=str)
            gravar_atomico(self.arq, lambda t: Path(t).write_text(txt, encoding="utf-8"))

    def _loop(self):
        while not self._parar.wait(self.intervalo):
            try:
                self.gravar()
            except OSError:
                pass

    def fechar(self, **extra):
        self._parar.set()
        self.gravar(**extra)


# ---------------------------------------------------------------- leitura do cache
class Conjunto:
    """Os shards concluídos de um conjunto, lidos por mmap (não ocupam memória reservada).

    `meta`: uma linha por áudio, na ordem dos shards. `camada(l)`: matriz (n, 1024) float32 da saída `l`
    (média no tempo). `linhas(idx)`: (k, 25, 1024) float16 das linhas pedidas.
    """

    def __init__(self, cache_dir, conjunto, shards=None):
        base = Path(cache_dir) / conjunto
        prontos = sorted(int(d.name.split("_")[1]) for d in base.glob("shard_?????")
                         if (d / "concluido.json").exists()) if base.exists() else []
        if shards is not None:
            prontos = [s for s in prontos if s in set(shards)]
        self.shards = prontos
        self.nome = conjunto
        metas, self._mm, self._dp = [], [], []
        for s in prontos:
            d = pasta_shard(cache_dir, conjunto, s)
            m = pd.read_parquet(d / "meta.parquet").assign(shard=s)
            metas.append(m)
            self._mm.append(np.load(d / "media.npy", mmap_mode="r"))
            self._dp.append(np.load(d / "desvio.npy", mmap_mode="r"))
        self.meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
        tam = [len(m) for m in self._mm]
        self.offsets = np.concatenate([[0], np.cumsum(tam)]).astype(np.int64)

    def __len__(self):
        return int(self.offsets[-1])

    def camada(self, l, tipo="media"):
        fonte = self._mm if tipo == "media" else self._dp
        if not fonte:
            return np.zeros((0, 1024), dtype=np.float32)
        return np.concatenate([np.asarray(m[:, l, :], dtype=np.float32) for m in fonte])

    def linhas(self, idx):
        """(len(idx), 25, 1024) float16, na ordem de `idx` (lê agrupado por shard, em ordem crescente)."""
        idx = np.asarray(idx, dtype=np.int64)
        ordem = np.argsort(idx, kind="stable")
        ord_idx = idx[ordem]
        sh = np.searchsorted(self.offsets, ord_idx, side="right") - 1
        out = np.empty((len(idx), *self._mm[0].shape[1:]), dtype=np.float16)
        for s in np.unique(sh):
            sel = np.nonzero(sh == s)[0]
            out[ordem[sel]] = self._mm[s][ord_idx[sel] - self.offsets[s]]
        return out
