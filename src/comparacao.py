"""Comparação de dois modelos nos MESMOS clipes (v5: XLS-R × WavLM): bootstrap pareado e métricas por grupo.

⚠️ Fora da trilha: o bootstrap (reamostragem com reposição) para medir a incerteza de uma diferença.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis import FAIXAS_DURACAO, faixa_duracao
from src.metrics import acuracias, eer_e_limiar


def juntar_notas(a, b, nomes=("a", "b")):
    """Junta as notas de dois modelos pelo `idx_manifest` (confere que os clipes são os mesmos)."""
    ka, kb = set(a.idx_manifest), set(b.idx_manifest)
    if ka != kb:
        raise ValueError(f"clipes diferentes: {len(ka - kb)} só em {nomes[0]}, {len(kb - ka)} só em {nomes[1]}")
    m = a.merge(b[["idx_manifest", "z"]], on="idx_manifest", suffixes=("", "_b"))
    return m.rename(columns={"z": f"z_{nomes[0]}", "z_b": f"z_{nomes[1]}"})


def bootstrap_pareado_por_locutor(y, za, zb, locutor, n=1000, seed=42):
    """Diferença (b − a) de AUC e de EER com reamostragem POR LOCUTOR (os mesmos sorteios para os dois).

    Sorteia os locutores com reposição e leva todos os clipes de cada um. Reamostrar clipes soltos daria
    um intervalo estreito demais, porque clipes do mesmo locutor não são independentes.
    Devolve um dict com o valor no conjunto inteiro, a média e o intervalo de 2,5% a 97,5% de cada diferença.
    """
    y, za, zb, locutor = (np.asarray(v) for v in (y, za, zb, locutor))
    grupos = [np.where(locutor == s)[0] for s in np.unique(locutor)]
    rng = np.random.default_rng(seed)
    d_auc, d_eer, pulados = [], [], 0
    for _ in range(n):
        idx = np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            pulados += 1
            continue
        d_auc.append(roc_auc_score(yy, zb[idx]) - roc_auc_score(yy, za[idx]))
        d_eer.append(eer_e_limiar(yy, zb[idx])[0] - eer_e_limiar(yy, za[idx])[0])
    out = {"n_reamostragens": len(d_auc), "puladas_sem_duas_classes": pulados, "n_locutores": len(grupos)}
    for nome, d, cheio in (("AUC", d_auc, roc_auc_score(y, zb) - roc_auc_score(y, za)),
                           ("EER", d_eer, eer_e_limiar(y, zb)[0] - eer_e_limiar(y, za)[0])):
        d = np.asarray(d)
        out[f"dif {nome} (b - a)"] = float(cheio)
        out[f"dif {nome} média bootstrap"] = float(d.mean())
        out[f"dif {nome} IC95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out[f"dif {nome} IC95 inclui zero"] = bool(np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5))
    return out


def metricas_por_grupo(notas, colunas_z, grupo, limiares=None):
    """AUC e EER de cada modelo (`colunas_z`) por valor de `grupo` (só grupos com real e fake).

    Com `limiares` ({coluna: limiar}), inclui também o recall de fake e o falso alarme de real nesse limiar.
    """
    linhas = []
    for g, d in notas.groupby(grupo, observed=True):
        y = d.label.values.astype(int)
        lin = {grupo: g, "n real": int((y == 0).sum()), "n fake": int((y == 1).sum())}
        for c in colunas_z:
            if len(np.unique(y)) == 2:
                lin[f"AUC {c}"] = roc_auc_score(y, d[c].values)
                lin[f"EER {c}"] = eer_e_limiar(y, d[c].values)[0]
            if limiares:
                pred = d[c].values >= limiares[c]
                lin[f"recall fake {c}"] = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
                lin[f"falso alarme {c}"] = float(pred[y == 0].mean()) if (y == 0).any() else np.nan
        linhas.append(lin)
    return pd.DataFrame(linhas)


def eer_por_gerador(notas, colunas_z):
    """EER de cada gerador: os fakes daquele gerador contra TODOS os reais do conjunto."""
    reais = notas[notas.label == 0]
    linhas = []
    for gen, f in notas[notas.label == 1].groupby("generator"):
        d = pd.concat([reais, f])
        y = d.label.values.astype(int)
        lin = {"gerador": gen, "fonte": f.source_dataset.mode().iloc[0], "n fake": len(f)}
        for c in colunas_z:
            lin[f"EER {c}"] = eer_e_limiar(y, d[c].values)[0]
        linhas.append(lin)
    return pd.DataFrame(linhas)


def por_duracao(notas, colunas_z, limiares):
    d = notas.assign(faixa=pd.Categorical(faixa_duracao(notas.duration_s), FAIXAS_DURACAO, ordered=True))
    return metricas_por_grupo(d, colunas_z, "faixa", limiares)


def acuracias_no_limiar(y, z, limiar):
    a, a05 = acuracias(y, z, limiar), acuracias(y, z, 0.0)
    return {"acurácia": a["acurácia"], "acurácia balanceada": a["acurácia balanceada"],
            "acurácia (0,5)": a05["acurácia"], "acurácia balanceada (0,5)": a05["acurácia balanceada"]}
