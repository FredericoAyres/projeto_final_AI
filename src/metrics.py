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
