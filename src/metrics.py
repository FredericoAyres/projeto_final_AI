"""Métricas de detecção. "Positivo" = fake. As notas podem ser logits ou probabilidades.

AUC e EER só dependem da ORDEM das notas, então dão o mesmo valor com logit ou probabilidade.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoide(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def eer_e_limiar(y, nota):
    """EER e o limiar (na mesma escala de `nota`) em que falso alarme = fakes que escapam."""
    fpr, tpr, thr = roc_curve(y, nota)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    t = thr[i] if np.isfinite(thr[i]) else np.max(nota)
    return float((fpr[i] + fnr[i]) / 2), float(t)


def eer_media_por_fonte(y, nota, fontes):
    """Média das EER de cada fonte que tem real E fake. Devolve (média, {fonte: EER})."""
    y, nota, fontes = np.asarray(y), np.asarray(nota), np.asarray(fontes)
    por_fonte = {}
    for s in sorted(set(fontes)):
        m = fontes == s
        if len(np.unique(y[m])) == 2:
            por_fonte[s] = eer_e_limiar(y[m], nota[m])[0]
    return float(np.mean(list(por_fonte.values()))), por_fonte


def acuracias(y, nota, limiar):
    """Acurácia e acurácia balanceada (média do recall de real e do recall de fake) no `limiar`.

    Atenção: num banco desbalanceado a acurácia sozinha engana. No dev_shift (81% real), um modelo
    que chama tudo de real acerta 81%. Por isso vai junto a acurácia trivial ("sempre a classe
    majoritária") e a balanceada (que vale 0,5 para qualquer regra que ignore o áudio).
    """
    y, nota = np.asarray(y).astype(int), np.asarray(nota)
    pred = (nota >= limiar).astype(int)
    rec_fake = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    rec_real = float((pred[y == 0] == 0).mean()) if (y == 0).any() else np.nan
    return {"acurácia": float((pred == y).mean()),
            "acurácia balanceada": float(np.nanmean([rec_real, rec_fake]))}


def acuracia_trivial(y):
    """Acurácia de "sempre a classe majoritária" (a referência mínima para a acurácia)."""
    y = np.asarray(y).astype(int)
    return float(max((y == 0).mean(), (y == 1).mean()))


def metricas_completas(y, nota, limiar, limiar_05=0.0):
    """`metricas` + acurácia e acurácia balanceada no `limiar` (fixado na validação) e em `limiar_05`
    (nota 0,5 de probabilidade; com notas em logit, é 0), mais a acurácia trivial.

    AUC e EER continuam sendo as métricas principais (não dependem de limiar nem da proporção).
    """
    m = metricas(y, nota, limiar)
    a, a05 = acuracias(y, nota, limiar), acuracias(y, nota, limiar_05)
    return {**m, "acurácia": a["acurácia"], "acurácia balanceada": a["acurácia balanceada"],
            "acurácia (0,5)": a05["acurácia"], "acurácia balanceada (0,5)": a05["acurácia balanceada"],
            "acurácia trivial": acuracia_trivial(y)}


def metricas(y, nota, limiar):
    """AUC, EER (sem limiar) e recall de fake / falso alarme no `limiar` (mesma escala de `nota`)."""
    y, nota = np.asarray(y).astype(int), np.asarray(nota)
    pred = (nota >= limiar).astype(int)
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    duas = n_real > 0 and n_fake > 0
    return {"n real": n_real, "n fake": n_fake,
            "AUC": roc_auc_score(y, nota) if duas else np.nan,
            "EER": eer_e_limiar(y, nota)[0] if duas else np.nan,
            "recall fake": float(pred[y == 1].mean()) if n_fake else np.nan,
            "falso alarme": float(pred[y == 0].mean()) if n_real else np.nan}
