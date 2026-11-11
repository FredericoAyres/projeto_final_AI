"""Cabeças e fusões da v6 sobre os vetores em cache (média e desvio no tempo das 25 saídas de cada modelo).

- `Modelo`: os conjuntos em cache de UM modelo congelado (XLS-R da Meta ou WavLM da Microsoft), com a
  padronização (média e desvio de cada saída e dimensão, calculados no TREINO).
- Regressão logística (RL, scikit-learn, lbfgs). Ela é uma camada linear com sigmoide (Aula 4); o C é o
  inverso da força da regularização L2 (C pequeno = pesos menores = menos overfitting).
- MLP (1 camada escondida) e CNN ao longo das 25 camadas (PyTorch). ⚠️ Fora da trilha: a CNN anda pelas
  camadas do modelo (não pelo tempo) e o AdamW.
- Fusão dos dois modelos: média das notas, stacking (RL de 2 entradas, com cross-fitting por fonte) e
  vetores colados. ⚠️ Fora da trilha: o stacking e o cross-fitting.

Todas as notas são logits (positivo = fake).
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.cache_v4 import Conjunto, pasta_shard
from src.metrics import eer_media_por_fonte

N_SAIDAS, DIM = 25, 1024


# ---------------------------------------------------------------- dados de um modelo
def _media_desvio(mmaps):
    s1 = s2 = None
    n = 0
    for mm in mmaps:
        x = np.asarray(mm, dtype=np.float64)
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x ** 2).sum(0) if s2 is None else s2 + (x ** 2).sum(0)
        n += len(x)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    return mu.astype(np.float32), np.maximum(sd, 1e-6).astype(np.float32)


class Modelo:
    """Conjuntos de um modelo: `conj[nome]` (Conjunto do cache), `ok[nome]` (clipes usados) e a padronização.

    `extras` = {nome: (cache_dir, conjunto)} para conjuntos de outro cache (os bancos de teste, na E6).
    """

    def __init__(self, nome, cache_dir, arq_pad, extras=None):
        self.nome = nome
        self.conj = {n: Conjunto(cache_dir, n) for n in ("treino", "val", "dev_shift")}
        for n, (cd, c) in (extras or {}).items():
            self.conj[n] = Conjunto(cd, c)
        for n, c in self.conj.items():
            if len(c) == 0:
                raise RuntimeError(f"{nome}: conjunto '{n}' vazio")
        self.ok = {n: c.meta.erro_leitura.fillna(0).values == 0 for n, c in self.conj.items()}
        self.arq_pad = arq_pad
        self._pad = None

    def pad(self, tipo="media"):
        """(mu, sd) (25, 1024) do treino, para a média ("media") ou o desvio ("desvio") no tempo."""
        if self._pad is None:
            from pathlib import Path
            p = Path(self.arq_pad)
            tr = self.conj["treino"]
            if p.exists() and int(np.load(p)["n"]) == len(tr):
                z = np.load(p)
                self._pad = {k: z[k] for k in ("mu", "sd", "mu_d", "sd_d")}
            else:
                mu, sd = _media_desvio(tr._mm)
                mu_d, sd_d = _media_desvio(tr._dp)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, mu=mu, sd=sd, mu_d=mu_d, sd_d=sd_d, n=len(tr))
                self._pad = {"mu": mu, "sd": sd, "mu_d": mu_d, "sd_d": sd_d}
        return (self._pad["mu"], self._pad["sd"]) if tipo == "media" else (self._pad["mu_d"], self._pad["sd_d"])

    def meta(self, n):
        return self.conj[n].meta[self.ok[n]].reset_index(drop=True)

    def X(self, n, camadas):
        """Média no tempo das `camadas`, padronizada com o treino, lado a lado: (clipes usados, 1024 × k)."""
        mu, sd = self.pad("media")
        c = self.conj[n]
        return np.hstack([(c.camada(l)[self.ok[n]] - mu[l]) / sd[l] for l in camadas]).astype(np.float32)

    def X25(self, n, idx=None):
        """(clipes, 25, 2048) float16: média e desvio de todas as saídas, padronizados com o treino."""
        c = self.conj[n]
        pos = np.nonzero(self.ok[n])[0] if idx is None else np.nonzero(self.ok[n])[0][idx]
        mu, sd = self.pad("media")
        mu_d, sd_d = self.pad("desvio")
        out = np.empty((len(pos), N_SAIDAS, 2 * DIM), dtype=np.float16)
        sh = np.searchsorted(c.offsets, pos, side="right") - 1
        for s in np.unique(sh):
            sel = np.nonzero(sh == s)[0]
            loc = pos[sel] - c.offsets[s]
            out[sel, :, :DIM] = ((np.asarray(c._mm[s][loc], dtype=np.float32) - mu) / sd).astype(np.float16)
            out[sel, :, DIM:] = ((np.asarray(c._dp[s][loc], dtype=np.float32) - mu_d) / sd_d).astype(np.float16)
        return out


def conferir_mesmos_clipes(modelos, nomes):
    """Confere que os modelos têm os mesmos clipes (idx_manifest e caminho) na mesma ordem e junta os `ok`
    (um clipe com erro de leitura em qualquer modelo sai de todos). Devolve um relatório."""
    rel = {}
    ms = list(modelos.values())
    for n in nomes:
        a = ms[0].conj[n].meta
        iguais = all(len(m.conj[n].meta) == len(a) and (m.conj[n].meta.idx_manifest.values == a.idx_manifest.values).all()
                     and (m.conj[n].meta.path.values == a.path.values).all() for m in ms[1:])
        if not iguais:
            raise RuntimeError(f"conjunto {n}: os modelos não têm os mesmos clipes na mesma ordem")
        ok = np.logical_and.reduce([m.ok[n] for m in ms])
        for m in ms:
            m.ok[n] = ok
        rel[n] = {"clipes": int(len(a)), "mesmos_idx_e_caminhos": True, "excluidos_erro_leitura": int((~ok).sum())}
    return rel


# ---------------------------------------------------------------- regressão logística
def pesos_por_fonte(meta):
    """sample_weight com o mesmo peso total por fonte de treino (1 / nº de clipes da fonte), reescalado para
    reais e fakes somarem o mesmo e para a média ser 1 (assim o C continua na mesma escala)."""
    f, y = meta.source_dataset.values, meta.label.values.astype(int)
    tam = pd.Series(f).value_counts()
    w = 1.0 / tam.loc[f].values
    for c in (0, 1):
        m = y == c
        w[m] *= 0.5 / w[m].sum()
    return w * len(w)


def ajustar_rl(X, y, C, max_iter=1000, sample_weight=None):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings(record=True) as av:
        warnings.simplefilter("always", ConvergenceWarning)
        rl = LogisticRegression(C=C, max_iter=max_iter, tol=1e-4).fit(X, y, sample_weight=sample_weight)
    rl.convergiu_ = not any(issubclass(a.category, ConvergenceWarning) for a in av)
    return rl


def rl_notas(Xtr, ytr, alvos, C, max_iter=1000, sample_weight=None):
    """Ajusta a RL e devolve ({alvo: logits}, modelo, segundos)."""
    t = time.time()
    rl = ajustar_rl(Xtr, ytr, C, max_iter, sample_weight)
    return {k: rl.decision_function(X).astype(np.float64) for k, X in alvos.items()}, rl, time.time() - t


# ---------------------------------------------------------------- fusão
def padronizar_pela_val(z, zv):
    m, s = float(np.mean(zv)), float(np.std(zv) + 1e-9)
    return (np.asarray(z) - m) / s


def fusao_media(notas):
    """`notas` = [{alvo: z} de cada modelo]. Cada logit é padronizado com a média e o desvio da VALIDAÇÃO
    do próprio modelo; depois, média simples."""
    return {k: np.mean([padronizar_pela_val(n[k], n["val"]) for n in notas], axis=0) for k in notas[0]}


def fusao_stacking(notas, meta_val, C=1.0):
    """RL de 2 entradas (os logits padronizados pela validação) com cross-fitting AGRUPADO POR FONTE dentro
    da validação: a nota de cada fonte da validação vem de uma RL treinada nas outras fontes (fora da
    amostra). Os outros alvos (dev_shift, testes) usam a RL treinada na validação inteira.
    Devolve ({alvo: z}, coeficientes)."""
    from sklearn.linear_model import LogisticRegression

    Z = {k: np.column_stack([padronizar_pela_val(n[k], n["val"]) for n in notas]) for k in notas[0]}
    yv, fv = meta_val.label.values.astype(int), meta_val.source_dataset.values
    oof = np.full(len(yv), np.nan)
    coefs = []
    for s in sorted(set(fv)):
        m = fv == s
        if len(np.unique(yv[~m])) < 2:
            continue
        rl = LogisticRegression(C=C, max_iter=1000).fit(Z["val"][~m], yv[~m])
        oof[m] = rl.decision_function(Z["val"][m])
        coefs.append({"dobra (fonte fora)": s, "intercepto": float(rl.intercept_[0]),
                      **{f"peso modelo {i + 1}": float(c) for i, c in enumerate(rl.coef_[0])}})
    rl = LogisticRegression(C=C, max_iter=1000).fit(Z["val"], yv)
    coefs.append({"dobra (fonte fora)": "validação inteira (usada no dev_shift e no teste)",
                  "intercepto": float(rl.intercept_[0]), **{f"peso modelo {i + 1}": float(c) for i, c in enumerate(rl.coef_[0])}})
    if np.isnan(oof).any():          # dobra sem as duas classes no treino (só no QUICK): usa a RL inteira
        oof[np.isnan(oof)] = rl.decision_function(Z["val"][np.isnan(oof)])
    out = {k: rl.decision_function(v) for k, v in Z.items() if k != "val"}
    out["val"] = oof
    return out, coefs


def X_colados(modelos_camadas, n):
    """Vetores dos dois modelos lado a lado; cada bloco dividido pela raiz do seu nº de dimensões."""
    blocos = []
    for M, cams in modelos_camadas:
        x = M.X(n, cams)
        blocos.append(x / np.sqrt(x.shape[1]))
    return np.hstack(blocos).astype(np.float32)


# ---------------------------------------------------------------- MLP e CNN (PyTorch)
class MLP(nn.Module):
    def __init__(self, d_in, oculta=256, dropout=0.3):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(d_in, oculta), nn.ReLU(), nn.Dropout(dropout), nn.Linear(oculta, 1))

    def forward(self, x):
        return self.rede(x).squeeze(1)


class CNNCamadas(nn.Module):
    """Entrada (lote, 25, C): as 25 saídas do modelo como uma "sequência" de 25 posições com C números.
    Conv1d(C→64, k=3) + BatchNorm + ReLU, Conv1d(64→64, k=3) + ReLU, média nas posições, Linear(64→1).
    ⚠️ Fora da trilha: a convolução anda ao longo das CAMADAS do modelo, não do tempo."""

    def __init__(self, c_in=2 * DIM, canais=64, k=3, dropout=0.0):
        super().__init__()
        self.c1, self.bn = nn.Conv1d(c_in, canais, k), nn.BatchNorm1d(canais)
        self.c2 = nn.Conv1d(canais, canais, k)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lin = nn.Linear(canais, 1)

    def forward(self, x):
        h = x.transpose(1, 2).float()
        h = torch.relu(self.bn(self.c1(h)))
        h = torch.relu(self.c2(h))
        return self.lin(self.drop(h.mean(-1))).squeeze(1)


def n_parametros(model):
    return int(sum(p.numel() for p in model.parameters()))


class NaGPU:
    """Matriz inteira já na GPU (MLP; CNN de um modelo só). `lote(idx)` devolve as linhas pedidas."""

    def __init__(self, X, device):
        self.X = X if torch.is_tensor(X) else torch.from_numpy(np.ascontiguousarray(X))
        self.X = self.X.to(device)

    def __len__(self):
        return len(self.X)

    def lote(self, idx):
        return self.X[torch.as_tensor(idx, device=self.X.device)]


class Lido25:
    """(clipes, 25, 2048 x nº de modelos) lido do cache a cada lote (quando não cabe na GPU: modelos colados)."""

    def __init__(self, partes, n, device):
        self.partes, self.n, self.device = partes, n, device
        self.tam = int(partes[0].ok[n].sum())

    def __len__(self):
        return self.tam

    def lote(self, idx):
        idx = np.asarray(idx)
        x = np.concatenate([M.X25(self.n, idx) for M in self.partes], axis=2)
        return torch.from_numpy(x).to(self.device)


def x25_na_gpu(M, n, device, bloco=4000):
    """Monta a matriz (clipes, 25, 2048) float16 direto na GPU, aos pedaços (pouca memória na CPU)."""
    tam = int(M.ok[n].sum())
    X = torch.empty((tam, N_SAIDAS, 2 * DIM), dtype=torch.float16, device=device)
    for i in range(0, tam, bloco):
        X[i:i + bloco] = torch.from_numpy(M.X25(n, np.arange(i, min(i + bloco, tam)))).to(device)
    return X


@torch.no_grad()
def prever(model, fonte, device, bloco=2048):
    """Logits de todas as linhas de `fonte` (NaGPU ou Lido25)."""
    model.eval()
    out = [model(fonte.lote(np.arange(i, min(i + bloco, len(fonte)))).float()).float().cpu().numpy()
           for i in range(0, len(fonte), bloco)]
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def treinar_nn(model, fonte_tr, ytr, fonte_v, meta_val, hp, seed, device, log=print):
    """AdamW + BCE, lote `hp["BATCH"]`, até `hp["MAX_EPOCAS"]` épocas, early stopping pela EER média por
    fonte da VALIDAÇÃO (fontes com real e fake; paciência `hp["PACIENCIA"]`).
    Devolve (modelo com o melhor estado, histórico, melhor época, segundos)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["LR"], weight_decay=hp["WD"])
    crit = nn.BCEWithLogitsLoss()
    y = torch.from_numpy(np.asarray(ytr, dtype=np.float32)).to(device)
    yv, fv = meta_val.label.values, meta_val.source_dataset.values
    melhor, sem, hist, estado, ep_melhor = np.inf, 0, [], None, 0
    t0 = time.time()
    for ep in range(1, hp["MAX_EPOCAS"] + 1):
        t = time.time()
        model.train()
        perm = rng.permutation(len(y))
        soma, n = 0.0, 0
        for i in range(0, len(perm), hp["BATCH"]):
            j = perm[i:i + hp["BATCH"]]
            if len(j) < 2:
                continue
            perda = crit(model(fonte_tr.lote(j).float()), y[torch.as_tensor(j, device=device)])
            opt.zero_grad()
            perda.backward()
            opt.step()
            soma += perda.item() * len(j)
            n += len(j)
        zv = prever(model, fonte_v, device)
        eer_m = eer_media_por_fonte(yv, zv, fv)[0]
        hist.append({"época": ep, "perda treino": soma / max(n, 1), "val EER média por fonte": eer_m,
                     "segundos": round(time.time() - t, 1)})
        marca = ""
        if eer_m < melhor:
            melhor, sem, ep_melhor, marca = eer_m, 0, ep, "  <- melhor"
            estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sem += 1
        log(f"      época {ep:2d} | perda {soma / max(n, 1):.4f} | val EER média/fonte {100 * eer_m:5.2f}% | "
            f"{time.time() - t:.0f} s{marca}")
        if sem >= hp["PACIENCIA"]:
            log(f"      early stopping ({hp['PACIENCIA']} épocas sem melhorar)")
            break
    model.load_state_dict(estado)
    return model, pd.DataFrame(hist), ep_melhor, time.time() - t0
