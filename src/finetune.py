"""v8: fine-tuning PARCIAL do XLS-R a partir da T3 da v7 (protocolo: docs/PROTOCOLO_V8.md).

A T3 é: XLS-R 300M congelado -> média no tempo das saídas 5, 6 e 7 (3.072 números) -> padronização
(média e desvio do treino da T3) -> regressão logística (C = 0,01). Aqui o MESMO caminho vira uma rede só,
e uma parte dela volta a aprender:

- o XLS-R é cortado depois da camada 7 (as saídas 8 a 24 não são usadas pela T3: cortar só economiza
  tempo e memória, não muda nada no que a T3 vê);
- as camadas escolhidas (5, 6 e 7, ou só a 7) ficam treináveis; o resto continua congelado;
- a padronização da T3 fica fixa; a regressão logística vira uma camada linear (3.072 -> 1), que pode
  começar com os pesos da T3 (no passo zero o modelo É a T3) ou aleatória (ablação).

Precisão: a parte congelada roda em float16 puro, exatamente como na extração dos vetores da T3 (assim
o passo zero reproduz a T3: medido, diferença mediana de 0,005 de logit; em bfloat16 eram 0,08). As
camadas treináveis guardam os pesos em float32 e fazem as contas em float16 (precisão mista com
"GradScaler", que multiplica a perda para os gradientes pequenos não virarem zero em float16).

⚠️ Fora da trilha: fine-tuning de transformer (descongelar camadas de um modelo pré-treinado), AdamW,
precisão mista float16, aquecimento + queda linear da taxa e corte da norma do gradiente. Explicação em
docs/PROTOCOLO_V8.md, seção 4.

O modelo fica SEMPRE no modo `eval()` (sem dropout, sem LayerDrop e sem o mascaramento SpecAugment do
pré-treino): assim, no passo zero, a saída é a mesma da T3, e a única fonte de mudança é o gradiente.

Este arquivo não depende do Windows nem dos caminhos do projeto: roda igual no Google Colab.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from src.preproc import COLS_ESTAT, SR, ClipesV4, juntar_clipes

CAMADAS = (5, 6, 7)              # saídas usadas pela T3 (índice 0 = entrada do transformer)
LARGURA_S = 2                    # lotes com formato fixo: 2, 4, 6 ou 8 s (ver src/extracao_v4.py)
MAX_S = 8
I_ERRO = COLS_ESTAT.index("erro_leitura")


class CamadaMista(nn.Module):
    """Camada treinável: pesos em float32, contas em float16 (autocast); entra e sai em float16, como as
    camadas congeladas em volta dela."""

    def __init__(self, camada):
        super().__init__()
        self.camada = camada.float()

    def forward(self, hidden_states, *args, **kw):
        dt = hidden_states.dtype
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.camada(hidden_states, *args, **kw)
        if isinstance(out, tuple):
            return (out[0].to(dt),) + tuple(out[1:])
        return out.to(dt)


def normalizar(x):
    """Média 0 e variância 1 no clipe inteiro (o mesmo de src/ssl.py)."""
    x = x.float()
    return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)


class DetectorFT(nn.Module):
    """XLS-R cortado na camada 7 + média no tempo das saídas 5, 6, 7 + padronização fixa + camada linear."""

    def __init__(self, ssl, mu, sd, coef=None, b=None, treinaveis=(5, 6, 7), seed=0):
        super().__init__()
        cfg = ssl.config
        assert cfg.do_stable_layer_norm, "esperado o XLS-R (encoder com layer norm 'estável')"
        n = max(CAMADAS)
        ssl.encoder.layers = ssl.encoder.layers[:n]
        # o layer norm final do encoder só é aplicado à ÚLTIMA saída; com o corte, ele cairia na saída 7,
        # que na T3 (modelo inteiro) não tem esse layer norm. Identidade = a saída 7 fica igual à da T3.
        ssl.encoder.layer_norm = nn.Identity()
        cfg.layerdrop, cfg.mask_time_prob, cfg.mask_feature_prob = 0.0, 0.0, 0.0
        ssl = ssl.half()                           # parte congelada: float16 puro (como na extração)
        for p in ssl.parameters():
            p.requires_grad_(False)
        for k in treinaveis:                       # saída k = resultado da camada de índice k-1
            ssl.encoder.layers[k - 1] = CamadaMista(ssl.encoder.layers[k - 1])
            for p in ssl.encoder.layers[k - 1].parameters():
                p.requires_grad_(True)
        self.ssl = ssl
        self.treinaveis = tuple(treinaveis)
        self.register_buffer("mu", torch.as_tensor(np.asarray(mu, np.float32)))
        self.register_buffer("sd", torch.as_tensor(np.asarray(sd, np.float32)))
        self.cabeca = nn.Linear(len(CAMADAS) * cfg.hidden_size, 1)
        if coef is not None:
            with torch.no_grad():
                self.cabeca.weight.copy_(torch.as_tensor(np.asarray(coef, np.float32)).view(1, -1))
                self.cabeca.bias.fill_(float(b))
        else:
            g = torch.Generator().manual_seed(int(seed))
            lim = 1 / math.sqrt(self.cabeca.in_features)          # o mesmo intervalo do nn.Linear padrão
            with torch.no_grad():
                self.cabeca.weight.uniform_(-lim, lim, generator=g)
                self.cabeca.bias.uniform_(-lim, lim, generator=g)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.ssl.eval()                            # sempre eval (ver docstring do módulo)
        return self

    def parametros(self):
        """(camadas treináveis do XLS-R, cabeça): dois grupos com taxas de aprendizado diferentes."""
        camadas = [p for p in self.ssl.parameters() if p.requires_grad]
        return camadas, list(self.cabeca.parameters())

    def estado_treinavel(self):
        """Só o que muda no treino (camadas treináveis + cabeça): o checkpoint fica pequeno."""
        nomes = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in nomes}

    def forward(self, x, mask, n_amostras):
        out = self.ssl(x.half(), attention_mask=mask, output_hidden_states=True)
        nq = self.ssl._get_feat_extract_output_lengths(n_amostras.to(x.device))
        T = out.hidden_states[0].shape[1]
        valido = (torch.arange(T, device=x.device)[None, :] < nq[:, None]).float()[:, :, None]
        cont = valido.sum(dim=1)
        f = torch.cat([(out.hidden_states[k].float() * valido).sum(dim=1) / cont for k in CAMADAS], dim=1)
        return self.cabeca((f - self.mu) / self.sd).squeeze(-1)


def montar_lote(clipes, linhas=None, device="cuda"):
    """Tensor (linhas, amostras) com formato fixo: amostras = múltiplo de 2 s (até 8 s); cada clipe é
    normalizado sozinho ANTES do enchimento; linhas extras = 25 ms de zeros (descartadas)."""
    n = torch.tensor([c.numel() for c in clipes])
    passo = LARGURA_S * SR
    amostras = int(min(MAX_S * SR, max(passo, passo * math.ceil(int(n.max()) / passo))))
    amostras = max(amostras, int(n.max()))
    linhas = len(clipes) if linhas is None else linhas
    lote = torch.zeros(linhas, amostras)
    mask = torch.zeros(linhas, amostras, dtype=torch.long)
    mask[len(clipes):, :400] = 1
    for i, c in enumerate(clipes):
        lote[i, : c.numel()] = normalizar(c)
        mask[i, : c.numel()] = 1
    n = torch.cat([n, torch.full((linhas - len(clipes),), 400)])
    return lote.to(device, non_blocking=True), mask.to(device, non_blocking=True), n


class LotesFixos(Sampler):
    def __init__(self, lotes):
        self.lotes = lotes

    def __iter__(self):
        return iter(self.lotes)

    def __len__(self):
        return len(self.lotes)


def balde(dur_s):
    """Balde de cada clipe pela duração do manifesto (teto do tamanho depois do corte de silêncio)."""
    d = np.minimum(np.nan_to_num(np.asarray(dur_s, float), nan=MAX_S), MAX_S)
    return np.clip(np.ceil(d / LARGURA_S) * LARGURA_S, LARGURA_S, MAX_S).astype(int)


def lotes_treino(dur_s, por_lote, rng):
    """Lotes de treino: clipes do mesmo balde, embaralhados; sobra de cada balde descartada; ordem dos
    lotes embaralhada. Devolve lista de listas de posições."""
    b = balde(dur_s)
    lotes = []
    for s in np.unique(b):
        idx = rng.permutation(np.flatnonzero(b == s))
        lotes += [idx[i:i + por_lote].tolist() for i in range(0, len(idx) - por_lote + 1, por_lote)]
    return [lotes[i] for i in rng.permutation(len(lotes))]


MAX_AMOSTRAS_AVAL = 2_048_000     # medido (RTX 4050 6 GB, Windows): com 4,1 M a GPU passa de 5,9 GB e o
                                  # driver usa a RAM como extensão (memória do processo > 13 GB); com 2 M, 3,2 GB


def linhas_aval(s, por_lote):
    return int(min(por_lote, MAX_AMOSTRAS_AVAL // (s * SR)))


def lotes_avaliacao(dur_s, por_lote):
    """Lotes de avaliação: todos os clipes, por balde de duração (ordenados), até `por_lote` clipes e
    MAX_AMOSTRAS_AVAL amostras por lote. Devolve [(linhas do formato fixo, posições)]."""
    dur_s = np.asarray(dur_s, float)
    b = balde(dur_s)
    lotes = []
    for s in np.unique(b):
        idx = np.flatnonzero(b == s)
        idx = idx[np.argsort(dur_s[idx], kind="stable")]
        k = linhas_aval(s, por_lote)
        lotes += [(k, idx[i:i + k].tolist()) for i in range(0, len(idx), k)]
    return lotes


# ------------------------------------------------------------------ áudios já preparados (execução, não método)
# Ler e decodificar (FLAC/WAV/MP3), reamostrar e cortar o silêncio custa mais que a GPU nesta máquina (medido:
# ~20 clipes/s com 2 workers, contra ~45 da GPU). Como a v8 passa várias vezes pelos mesmos áudios, cada clipe
# é preparado UMA vez (o mesmo src/preproc.py da extração) e guardado em float16, em shards.


class ClipesPreparados(torch.utils.data.Dataset):
    """Clipe já preparado (float16 no disco -> float32), mesmo formato de saída do ClipesV4."""

    def __init__(self, arq, ini, n):
        self.arq, self.ini, self.n = list(arq), np.asarray(ini, np.int64), np.asarray(n, np.int64)
        self._abertos = {}

    def __len__(self):
        return len(self.arq)

    def __getitem__(self, i):
        a = self.arq[i]
        if a not in self._abertos:
            self._abertos[a] = np.load(a, mmap_mode="r")
        x = torch.from_numpy(np.asarray(self._abertos[a][self.ini[i]:self.ini[i] + self.n[i]], dtype=np.float32))
        est = np.zeros(len(COLS_ESTAT))
        return x, est, i


def preparar_audios(paths, chaves, destino, hp, num_workers=6, por_shard=5000, log=print):
    """Prepara os clipes (src/preproc.py, igual à extração) e grava em `destino`/shard_XXXXX.npy (float16,
    um atrás do outro) + shard_XXXXX.parquet (chave, ini, n, erro_leitura). Retomável por shard."""
    import time
    from pathlib import Path

    import pandas as pd
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    paths = np.asarray(paths)
    chaves = np.asarray(chaves)
    t0, feitos = time.time(), 0
    for k, i0 in enumerate(range(0, len(paths), por_shard)):
        arq, ind = destino / f"shard_{k:05d}.npy", destino / f"shard_{k:05d}.parquet"
        if ind.exists():
            continue
        sl = slice(i0, i0 + por_shard)
        ds = ClipesV4(paths[sl], np.arange(len(paths[sl])), cortar_silencio=True, top_db=hp["TOP_DB"],
                      min_s=hp["MIN_S"], max_s=hp["MAX_S"])
        dl = DataLoader(ds, batch_size=32, num_workers=num_workers, collate_fn=juntar_clipes,
                        persistent_workers=False, prefetch_factor=4 if num_workers else None)
        partes, n, erro = [None] * len(ds), np.zeros(len(ds), np.int64), np.zeros(len(ds), bool)
        for clipes, est, pos in dl:
            for c, e, p in zip(clipes, est, pos):
                partes[p] = c.numpy().astype(np.float16)
                n[p] = c.numel()
                erro[p] = e[I_ERRO] == 1
        ini = np.concatenate([[0], np.cumsum(n)[:-1]])
        tmp = arq.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate(partes))
        tmp.replace(arq)
        pd.DataFrame({"chave": chaves[sl], "arq": str(arq), "ini": ini, "n": n, "erro_leitura": erro}).to_parquet(ind, index=False)
        feitos += len(ds)
        dt = time.time() - t0
        log(f"  preparar: shard {k} ({len(ds)} clipes, {int(erro.sum())} ilegíveis) | {feitos / dt:.1f} clipes/s")


def carregador(fonte, lotes, hp, num_workers):
    """`fonte`: array de caminhos (lê e prepara na hora) ou DataFrame (arq, ini, n) de áudios preparados."""
    if hasattr(fonte, "columns"):
        ds = ClipesPreparados(fonte.arq.values, fonte.ini.values, fonte.n.values)
    else:
        ds = ClipesV4(fonte, np.arange(len(fonte)), cortar_silencio=True, top_db=hp["TOP_DB"],
                      min_s=hp["MIN_S"], max_s=hp["MAX_S"])
    return DataLoader(ds, batch_sampler=LotesFixos(lotes), num_workers=num_workers, collate_fn=juntar_clipes,
                      persistent_workers=False, prefetch_factor=4 if num_workers else None, pin_memory=True)


@torch.no_grad()
def pontuar(model, paths, dur_s, hp, num_workers=2, por_lote=32, device="cuda", log=print, rotulo=""):
    """Logit (maior = mais falso) de cada áudio. Devolve (z, erro_leitura)."""
    model.eval()
    n = len(paths)
    z, erro = np.full(n, np.nan), np.zeros(n, bool)
    plano = lotes_avaliacao(dur_s, por_lote)
    t0, feitos = __import__("time").time(), 0
    for k, (clipes, est, pos) in enumerate(carregador(paths, [l_ for _, l_ in plano], hp, num_workers)):
        x, mask, nn_ = montar_lote(clipes, plano[k][0], device)
        out = model(x, mask, nn_)
        z[pos] = out[: len(clipes)].float().cpu().numpy()
        erro[pos] = est[:, I_ERRO] == 1
        feitos += len(clipes)
        if (k + 1) % 200 == 0:
            dt = __import__("time").time() - t0
            log(f"  {rotulo} notas: {feitos}/{n} clipes ({feitos / dt:.1f} clipes/s)")
    return z, erro


def agenda(otimizador, passos_total, frac_aquecimento=0.1):
    """Taxa sobe linearmente de 0 ao valor cheio nos primeiros 10% dos passos e desce linearmente a 0."""
    aq = max(1, int(round(frac_aquecimento * passos_total)))

    def f(passo):
        if passo < aq:
            return (passo + 1) / aq
        return max(0.0, (passos_total - passo) / max(1, passos_total - aq))
    return torch.optim.lr_scheduler.LambdaLR(otimizador, f)


def treinar_epoca(model, otim, sched, paths, y, dur_s, lotes, hp, num_workers=2, device="cuda", log=print,
                  max_norma=1.0, rotulo=""):
    """Uma passada pelos `lotes` (posições em paths/y). Devolve estatísticas da época."""
    import time
    model.train()
    camadas, cabeca = model.parametros()
    perda_f = nn.BCEWithLogitsLoss()
    escala = torch.amp.GradScaler("cuda")
    t0, soma, n_lotes, n_clipes, pulados = time.time(), 0.0, 0, 0, 0
    y_t = torch.as_tensor(np.asarray(y, np.float32))
    for k, (clipes, est, pos) in enumerate(carregador(paths, lotes, hp, num_workers)):
        ok = torch.as_tensor(est[:, I_ERRO] != 1)
        if not bool(ok.all()):
            pulados += int((~ok).sum())
        x, mask, nn_ = montar_lote(clipes, None, device)
        alvo = y_t[pos].to(device)
        logit = model(x, mask, nn_)
        okd = ok.to(device)
        perda = perda_f(logit.float()[okd], alvo[okd])
        otim.zero_grad(set_to_none=True)
        escala.scale(perda).backward()
        escala.unscale_(otim)
        torch.nn.utils.clip_grad_norm_(camadas + cabeca, max_norma)
        escala.step(otim)
        escala.update()
        sched.step()
        soma += float(perda.detach())
        n_lotes += 1
        n_clipes += len(clipes)
        if (k + 1) % 100 == 0:
            dt = time.time() - t0
            log(f"  {rotulo} passo {k + 1}/{len(lotes)} | perda média {soma / n_lotes:.4f} | "
                f"{n_clipes / dt:.1f} clipes/s | GPU pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    dt = time.time() - t0
    return {"passos": n_lotes, "clipes": n_clipes, "perda_media": soma / max(1, n_lotes),
            "segundos": round(dt, 1), "clipes_por_s": round(n_clipes / max(dt, 1e-9), 2), "ilegiveis": pulados}
