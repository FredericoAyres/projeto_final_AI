"""Cabeças leves sobre as características em cache (v4) e a avaliação padrão (Partes E e F).

Cabeças:
1. Regressão logística por camada (média no tempo de UMA das 25 saídas). ⚠️ Fora da trilha: a regressão
   logística em si é uma camada linear com sigmoide (Aula 4), mas aqui é ajustada pelo scikit-learn,
   uma vez por camada, para descobrir qual camada carrega mais informação ("probing").
2. Concatenação das melhores camadas (pela validação) + regressão logística.
3. Soma ponderada das 25 saídas (pesos por softmax, aprendidos) + camada linear; e a mesma soma + MLP de
   uma camada escondida. Treinadas em PyTorch (AdamW, early stopping pela EER média por fonte).
   ⚠️ Fora da trilha: a soma ponderada aprendida de camadas (padrão do benchmark SUPERB) e o AdamW.

Todas as entradas são padronizadas com a média e o desvio do TREINO (por saída e por dimensão).
Notas são logits (positivo = fake); limiar 0,5 de probabilidade = logit 0.
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.analysis import FAIXAS_DURACAO, faixa_duracao
from src.metrics import eer_e_limiar, eer_media_por_fonte, metricas_completas

N_SAIDAS = 25


# ---------------------------------------------------------------- padronização
def media_desvio_treino(conj):
    """Média e desvio (25, 1024) de cada saída e dimensão no treino, calculados shard a shard em float64."""
    s1 = s2 = None
    n = 0
    for mm in conj._mm:
        x = np.asarray(mm, dtype=np.float64)
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x ** 2).sum(0) if s2 is None else s2 + (x ** 2).sum(0)
        n += len(x)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    return mu.astype(np.float32), np.maximum(sd, 1e-6).astype(np.float32)


# ---------------------------------------------------------------- avaliação padrão
def avaliar(nota_val, meta_val, nota_dev, meta_dev):
    """Métricas padrão da v4 para uma cabeça. Limiar = ponto de EER da validação inteira (nunca o dev_shift).

    Validação: EER média por fonte (só fontes com real e fake), EER global, EER de cada fonte, AUC.
    dev_shift: AUC, EER, acurácia e acurácia balanceada (no limiar da validação e em 0,5), acurácia
    trivial, recall de fake e falso alarme de real; e AUC/EER por faixa de duração original.
    """
    yv, yd = meta_val.label.values.astype(int), meta_dev.label.values.astype(int)
    eer_m, por_fonte = eer_media_por_fonte(yv, nota_val, meta_val.source_dataset.values)
    eer_g, limiar = eer_e_limiar(yv, nota_val)
    mv = metricas_completas(yv, nota_val, limiar)
    md = metricas_completas(yd, nota_dev, limiar)
    r = {"val EER média por fonte": eer_m, "val EER global": eer_g, "val AUC": mv["AUC"],
         "val acurácia balanceada": mv["acurácia balanceada"],
         **{f"val EER {s}": v for s, v in por_fonte.items()},
         "limiar (logit)": limiar, "limiar (prob.)": float(1 / (1 + np.exp(-limiar))),
         "dev AUC": md["AUC"], "dev EER": md["EER"], "dev acurácia": md["acurácia"],
         "dev acurácia balanceada": md["acurácia balanceada"], "dev acurácia (0,5)": md["acurácia (0,5)"],
         "dev acurácia balanceada (0,5)": md["acurácia balanceada (0,5)"],
         "dev acurácia trivial": md["acurácia trivial"], "dev recall fake": md["recall fake"],
         "dev falso alarme real": md["falso alarme"], "n val": len(yv), "n dev": len(yd)}
    faixa = faixa_duracao(meta_dev.duration_s.values)
    for f in FAIXAS_DURACAO:
        m = np.asarray(faixa == f)
        duas = m.any() and len(np.unique(yd[m])) == 2
        r[f"dev AUC {f}"] = metricas_completas(yd[m], nota_dev[m], limiar)["AUC"] if duas else np.nan
        r[f"dev EER {f}"] = eer_e_limiar(yd[m], nota_dev[m])[0] if duas else np.nan
        r[f"dev n {f}"] = int(m.sum())
    return r


# ---------------------------------------------------------------- regressão logística (scikit-learn)
def ajustar_rl(Xtr, ytr, Xv, meta_val, grade_c, max_iter, log=None):
    """Ajusta uma regressão logística para cada C da grade e fica com a de MENOR EER média por fonte na
    validação. Devolve (modelo, C escolhido, tabela da grade)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    melhor, linhas = None, []
    for C in grade_c:
        t = time.time()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            rl = LogisticRegression(C=C, max_iter=max_iter, tol=1e-4).fit(Xtr, ytr)
        zv = rl.decision_function(Xv)
        eer_m = eer_media_por_fonte(meta_val.label.values, zv, meta_val.source_dataset.values)[0]
        conv = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
        linhas.append({"C": C, "val EER média por fonte": eer_m, "iterações": int(rl.n_iter_[0]),
                       "convergiu": conv, "segundos": round(time.time() - t, 1)})
        if log:
            log(f"    C={C:g}: EER média/fonte {100 * eer_m:.2f}% | {int(rl.n_iter_[0])} iterações"
                f"{'' if conv else ' (não convergiu)'} | {time.time() - t:.0f} s")
        if melhor is None or eer_m < melhor[2]:
            melhor = (rl, C, eer_m)
    return melhor[0], melhor[1], pd.DataFrame(linhas)


# ---------------------------------------------------------------- soma ponderada das camadas (PyTorch)
class SomaPonderada(nn.Module):
    """x (lote, 25, 1024) padronizado -> soma das 25 saídas com pesos softmax(w) -> linear ou MLP -> logit."""

    def __init__(self, n_saidas=N_SAIDAS, dim=1024, oculta=None, dropout=0.3):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_saidas))      # começa com pesos iguais (1/25 cada)
        if oculta:
            self.cab = nn.Sequential(nn.Linear(dim, oculta), nn.ReLU(), nn.Dropout(dropout), nn.Linear(oculta, 1))
        else:
            self.cab = nn.Linear(dim, 1)

    def pesos(self):
        return torch.softmax(self.w, 0)

    def forward(self, x):
        h = (self.pesos()[None, :, None] * x).sum(1)
        return self.cab(h).squeeze(1)


def _blocos(conj, idx, mu, sd, device, bloco):
    """Gera (posições, tensor padronizado float32 na GPU) em blocos de `bloco` linhas de `idx`."""
    for i in range(0, len(idx), bloco):
        parte = idx[i:i + bloco]
        x = torch.from_numpy(conj.linhas(parte)).to(device).float()
        yield i, (x - mu) / sd


@torch.no_grad()
def prever_soma(model, conj, mu, sd, device, bloco=4096):
    model.eval()
    idx = np.arange(len(conj))
    return np.concatenate([model(x).float().cpu().numpy() for _, x in _blocos(conj, idx, mu, sd, device, bloco)])


def treinar_soma(tr, val, mu_np, sd_np, hp, seed, oculta, device, log=print):
    """Treina a soma ponderada (+ linear se `oculta` é None, + MLP senão) com early stopping pela EER média
    por fonte na validação. Devolve (melhor estado, histórico, melhor época, pesos das camadas)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = torch.from_numpy(mu_np).to(device), torch.from_numpy(sd_np).to(device)
    model = SomaPonderada(oculta=oculta, dropout=hp["SOMA_DROPOUT"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["SOMA_LR"], weight_decay=hp["SOMA_WD"])
    crit = nn.BCEWithLogitsLoss()
    ytr = torch.from_numpy(tr.meta.label.values.astype(np.float32)).to(device)
    yv, fv = val.meta.label.values, val.meta.source_dataset.values
    melhor, sem, hist, estado, ep_melhor = np.inf, 0, [], None, 0
    for ep in range(1, hp["SOMA_MAX_EPOCAS"] + 1):
        t = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        soma, n = 0.0, 0
        for i, x in _blocos(tr, perm, mu, sd, device, hp["SOMA_BLOCO"]):
            y_bloco = ytr[torch.from_numpy(perm[i:i + len(x)]).to(device)]
            for j in range(0, len(x), hp["SOMA_BATCH"]):
                xb, yb = x[j:j + hp["SOMA_BATCH"]], y_bloco[j:j + hp["SOMA_BATCH"]]
                perda = crit(model(xb), yb)
                opt.zero_grad()
                perda.backward()
                opt.step()
                soma += perda.item() * len(yb)
                n += len(yb)
        zv = prever_soma(model, val, mu, sd, device)
        eer_m = eer_media_por_fonte(yv, zv, fv)[0]
        eer_g = eer_e_limiar(yv, zv)[0]
        hist.append({"época": ep, "perda treino": soma / n, "EER média por fonte (val)": eer_m,
                     "EER global (val)": eer_g, "segundos": time.time() - t})
        marca = ""
        if eer_m < melhor:
            melhor, sem, ep_melhor, marca = eer_m, 0, ep, "  <- melhor"
            estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sem += 1
        log(f"    época {ep:2d} | perda {soma / n:.4f} | val EER média/fonte {100 * eer_m:5.2f}% | "
            f"global {100 * eer_g:5.2f}% | {time.time() - t:.0f} s{marca}")
        if sem >= hp["SOMA_PACIENCIA"]:
            log(f"    early stopping: {hp['SOMA_PACIENCIA']} épocas sem melhorar")
            break
    model.load_state_dict(estado)
    return model, pd.DataFrame(hist), ep_melhor, model.pesos().detach().cpu().numpy()


# ---------------------------------------------------------------- atalhos (Parte F)
def matriz_estat(meta, cols):
    X = meta[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0)


def classificador_fonte(Xtr, fonte_tr, Xv, fonte_v, C=1.0, max_iter=2000):
    """Regressão logística multinomial que prevê a FONTE (só áudios reais). Devolve métricas e a matriz
    de confusão (linhas = fonte verdadeira na validação, colunas = fonte prevista)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    esc = StandardScaler().fit(Xtr)
    rl = LogisticRegression(C=C, max_iter=max_iter).fit(esc.transform(Xtr), fonte_tr)
    pred = rl.predict(esc.transform(Xv))
    conf = pd.crosstab(pd.Series(fonte_v, name="verdadeira"), pd.Series(pred, name="prevista"))
    maj = pd.Series(fonte_v).value_counts(normalize=True).iloc[0]
    return {"acurácia": float((pred == fonte_v).mean()), "acurácia balanceada": float(balanced_accuracy_score(fonte_v, pred)),
            "acaso (1/nº de fontes da validação)": 1 / pd.Series(fonte_v).nunique(),
            "acurácia trivial (fonte majoritária)": float(maj), "n treino": len(fonte_tr), "n val": len(fonte_v),
            "fontes treino": sorted(pd.unique(fonte_tr).tolist()),
            "fontes val": sorted(pd.unique(fonte_v).tolist())}, conf
