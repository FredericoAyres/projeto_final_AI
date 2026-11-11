"""Métricas da v6: AUC da PIOR fonte, bootstrap pareado por locutor, regra do desafiante e regras de limiar.

Pré-registro: docs/HISTORICO_VERSOES.md, seção "v6 - pré-registro".

"Fontes de decisão" (G): as fontes da validação com real E fake (dfadd, shiftyspeech, codecfake_plus) e o
dev_shift. A métrica primária é a menor AUC entre elas ("pior fonte"). Notas são logits (positivo = fake).

⚠️ Fora da trilha: o bootstrap (reamostrar com reposição para medir a incerteza de uma diferença). Aqui ele
sorteia LOCUTORES (todos os clipes de cada um), porque clipes do mesmo locutor não são independentes; no
CodecFake+, que não tem locutor, sorteia clipes.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metrics import acuracia_trivial, acuracias, eer_e_limiar

FONTES_VAL = ("dfadd", "shiftyspeech", "codecfake_plus")
DEV = "dev_shift"


# ---------------------------------------------------------------- fontes de decisão
def grupos(meta_val, meta_dev, fontes_val=FONTES_VAL):
    """[(nome, "val" | "dev", índices)] das fontes de G que têm as duas classes (na ordem fixa)."""
    out = []
    fv, yv = meta_val.source_dataset.values, meta_val.label.values
    for f in fontes_val:
        idx = np.nonzero(fv == f)[0]
        if len(idx) and len(np.unique(yv[idx])) == 2:
            out.append((f, "val", idx))
    if len(meta_dev) and len(np.unique(meta_dev.label.values)) == 2:
        out.append((DEV, "dev", np.arange(len(meta_dev))))
    return out


def avaliar(zv, mv, zd, md, limiar=None):
    """Métricas da v6 de um sistema (notas `zv` na validação e `zd` no dev_shift).

    Limiar A = ponto de EER da validação inteira (ou `limiar`, se dado). Devolve um dict plano:
    AUC / EER / acurácia balanceada (no limiar A) de cada fonte de G, a pior AUC (e qual fonte),
    as médias nas fontes de G, a EER média por fonte só da validação (a métrica da v4/v5) e o dev_shift.
    """
    zv, zd = np.asarray(zv, dtype=np.float64), np.asarray(zd, dtype=np.float64)
    yv, yd = mv.label.values.astype(int), md.label.values.astype(int)
    lim_a = eer_e_limiar(yv, zv)[1] if limiar is None else float(limiar)
    r = {"limiar A (logit)": lim_a}
    aucs, eers, bals = {}, {}, {}
    for nome, orig, idx in grupos(mv, md):
        y, z = (yv, zv) if orig == "val" else (yd, zd)
        y, z = y[idx], z[idx]
        aucs[nome] = float(roc_auc_score(y, z))
        eers[nome] = eer_e_limiar(y, z)[0]
        a = acuracias(y, z, lim_a)
        bals[nome] = a["acurácia balanceada"]
        pred = z >= lim_a
        r[f"AUC {nome}"], r[f"EER {nome}"], r[f"acc bal {nome}"] = aucs[nome], eers[nome], bals[nome]
        r[f"recall fake {nome}"] = float(pred[y == 1].mean())
        r[f"falso alarme {nome}"] = float(pred[y == 0].mean())
    if aucs:
        pior = min(aucs, key=aucs.get)
        r["AUC pior fonte"], r["fonte pior"] = aucs[pior], pior
        r["AUC média (G)"] = float(np.mean(list(aucs.values())))
        r["acc bal média (G)"] = float(np.mean(list(bals.values())))
        r["EER média (G)"] = float(np.mean(list(eers.values())))
    else:
        r["AUC pior fonte"], r["fonte pior"] = np.nan, "-"
    ev = [eers[f] for f in FONTES_VAL if f in eers]
    r["val EER média por fonte (3)"] = float(np.mean(ev)) if ev else np.nan
    r["val EER global"] = eer_e_limiar(yv, zv)[0] if len(np.unique(yv)) == 2 else np.nan
    if len(np.unique(yd)) == 2:
        r["dev AUC"] = float(roc_auc_score(yd, zd))
        r["dev EER"], r["limiar B (logit)"] = eer_e_limiar(yd, zd)
        a = acuracias(yd, zd, lim_a)
        r["dev acurácia"], r["dev acc bal"] = a["acurácia"], a["acurácia balanceada"]
        r["dev acurácia trivial"] = acuracia_trivial(yd)
    return r


# ---------------------------------------------------------------- bootstrap pareado da pior fonte
def _auc_ponderada(y, z, W):
    """AUC de cada linha de pesos W (B, n) (reamostragem = nº de vezes que cada clipe foi sorteado).

    Ordena as notas uma vez; empates contam meio (como a AUC usual). Devolve (B,), NaN sem as duas classes.
    """
    ordem = np.argsort(z, kind="mergesort")
    zs, ys = z[ordem], y[ordem]
    inicio = np.concatenate([[0], np.nonzero(np.diff(zs) != 0)[0] + 1])
    Ws = W[:, ordem]
    Gp = np.add.reduceat(Ws * (ys == 1), inicio, axis=1)
    Gn = np.add.reduceat(Ws * (ys == 0), inicio, axis=1)
    neg_antes = np.cumsum(Gn, axis=1) - Gn
    num = (Gp * (neg_antes + 0.5 * Gn)).sum(1)
    den = Gp.sum(1) * Gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


class BootPior:
    """Reamostragens FIXAS (seed) das fontes de G, reutilizadas em todas as comparações (pareado).

    Em cada fonte, sorteia com reposição os grupos (locutores; clipes quando não há locutor) e cada clipe
    recebe o peso = nº de vezes que o seu grupo saiu. `pior(zv, zd)` devolve a pior AUC de cada reamostragem.
    """

    def __init__(self, mv, md, n=2000, seed=42):
        self.n, self.seed = n, seed
        self.fontes = []
        rng = np.random.default_rng(seed)
        for nome, orig, idx in grupos(mv, md):
            meta = (mv if orig == "val" else md).iloc[idx]
            spk = meta.speaker_id
            if spk.notna().all():
                cod, uniq = pd.factorize(spk.astype(str))
                unidade = "locutor"
            else:
                cod, uniq = np.arange(len(meta)), np.arange(len(meta))
                unidade = "clipe"
            k = len(uniq)
            sorteio = rng.integers(0, k, size=(n, k))
            cont = np.stack([np.bincount(s, minlength=k) for s in sorteio]).astype(np.float32)
            W = cont[:, cod]
            self.fontes.append({"nome": nome, "orig": orig, "idx": idx, "W": W, "y": meta.label.values.astype(int),
                                "unidade": unidade, "n_grupos": k})

    def descricao(self):
        return {f["nome"]: {"unidade": f["unidade"], "n_grupos": int(f["n_grupos"]), "n_clipes": int(len(f["idx"]))}
                for f in self.fontes}

    def aucs(self, zv, zd):
        """{fonte: (B,) AUC de cada reamostragem}."""
        zv, zd = np.asarray(zv, dtype=np.float64), np.asarray(zd, dtype=np.float64)
        out = {}
        for f in self.fontes:
            z = (zv if f["orig"] == "val" else zd)[f["idx"]]
            out[f["nome"]] = _auc_ponderada(f["y"], z, f["W"])
        return out

    def pior(self, zv, zd):
        a = self.aucs(zv, zd)
        return np.min(np.stack(list(a.values())), axis=0)

    def comparar(self, atual, desafiante, dif_min=0.005, mv=None, md=None):
        """Diferença (desafiante − atual) da AUC da pior fonte. `atual`/`desafiante` = (zv, zd).

        Relevante = IC95 sem o zero E diferença na amostra inteira >= `dif_min` a favor do desafiante.
        """
        pa, pb = self.pior(*atual), self.pior(*desafiante)
        d = pb - pa
        ok = np.isfinite(d)
        d = d[ok]
        lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (np.nan, np.nan)
        cheio = np.nan
        if mv is not None:
            cheio = (avaliar(desafiante[0], mv, desafiante[1], md)["AUC pior fonte"]
                     - avaliar(atual[0], mv, atual[1], md)["AUC pior fonte"])
        exclui = bool(lo > 0 or hi < 0)
        return {"dif AUC pior (desafiante - atual)": float(cheio), "IC95": [lo, hi], "IC95 exclui zero": exclui,
                "média bootstrap": float(d.mean()) if len(d) else np.nan,
                "reamostragens válidas": int(ok.sum()), "reamostragens descartadas": int((~ok).sum()),
                "relevante": bool(exclui and lo > 0 and cheio >= dif_min)}


def regra_desafiante(boot, candidatos, mv, md, dif_min=0.005, log=print):
    """Ordem do mais simples ao mais complexo; cada um só toma o lugar se a diferença sobre o atual for
    relevante. `candidatos` = [(nome, (zv, zd)) ou (nome, [(zv, zd), ...]) para várias seeds].

    Com várias seeds, o desafiante precisa ser relevante EM CADA seed. Devolve (vencedor, comparações).
    """
    def seeds(x):
        return x if isinstance(x, list) else [x]

    atual_nome, atual = candidatos[0]
    comps = []
    for nome, notas in candidatos[1:]:
        at, ns = seeds(atual), seeds(notas)
        # mesma seed contra mesma seed quando os dois têm seeds; senão, cada seed contra o sistema atual
        r_seeds = [boot.comparar(at[i] if len(at) == len(ns) else at[0], s, dif_min, mv, md)
                   for i, s in enumerate(ns)]
        relevante = all(r["relevante"] for r in r_seeds)
        comps.append({"atual": atual_nome, "desafiante": nome, "relevante em todas as seeds": relevante,
                      "por seed": r_seeds})
        log(f"  {nome} × {atual_nome}: " + " | ".join(
            f"dif {r['dif AUC pior (desafiante - atual)']:+.4f} IC95 [{r['IC95'][0]:+.4f}, {r['IC95'][1]:+.4f}]"
            f"{' RELEVANTE' if r['relevante'] else ''}" for r in r_seeds)
            + f" -> {'toma o lugar' if relevante else 'empate/pior: fica ' + atual_nome}")
        if relevante:
            atual_nome, atual = nome, notas
    return atual_nome, comps


# ---------------------------------------------------------------- regras de limiar (E5, passo 4)
def limiar_regra(regra, z, meta, fontes_duas=FONTES_VAL, fpr=0.05):
    """Limiar (logit) pela regra, calculado com as notas `z` e o meta dado.

    A: ponto de EER do conjunto inteiro; C: mediana dos limiares de EER de cada fonte com as duas classes;
    D: em cada fonte com reais, o percentil (1 − fpr) das notas dos reais (falso alarme = fpr); a mediana.
    """
    z, y, f = np.asarray(z, dtype=np.float64), meta.label.values.astype(int), meta.source_dataset.values
    if regra == "A":
        return eer_e_limiar(y, z)[1]
    if regra == "C":
        ts = [eer_e_limiar(y[f == s], z[f == s])[1] for s in sorted(set(f))
              if s in fontes_duas and len(np.unique(y[f == s])) == 2]
        return float(np.median(ts))
    if regra == "D":
        ts = [float(np.quantile(z[(f == s) & (y == 0)], 1 - fpr)) for s in sorted(set(f)) if ((f == s) & (y == 0)).any()]
        return float(np.median(ts))
    raise ValueError(regra)


def loso_limiar(z, meta, regras=("A", "C", "D"), fontes_duas=FONTES_VAL, fpr=0.05):
    """Validação cruzada deixando uma fonte de fora: para cada fonte com as duas classes, o limiar vem das
    OUTRAS fontes da validação e é aplicado nela. Devolve a tabela (regra × fonte) da acurácia balanceada."""
    z, f = np.asarray(z, dtype=np.float64), meta.source_dataset.values
    linhas = []
    for s in fontes_duas:
        dentro = f == s
        if not dentro.any() or len(np.unique(meta.label.values[dentro])) < 2:
            continue
        resto = meta[~dentro]
        for r in regras:
            try:
                t = limiar_regra(r, z[~dentro], resto, fontes_duas, fpr)
            except (ValueError, IndexError):
                t = np.nan
            a = acuracias(meta.label.values[dentro], z[dentro], t) if np.isfinite(t) else {"acurácia balanceada": np.nan}
            linhas.append({"regra": r, "fonte deixada de fora": s, "limiar (logit)": t,
                           "acurácia balanceada": a["acurácia balanceada"]})
    return pd.DataFrame(linhas)
