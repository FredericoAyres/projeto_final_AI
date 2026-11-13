"""v7, seção PÓS-HOC (feita depois de ver o teste mestre; não muda nenhuma receita, regra ou escolha).

Pergunta: o teste mestre do ML-ITW tem 55 locutores de uma classe só. A AUC dele mistura "separar locutores"
com "separar real de falso dentro do mesmo locutor". Aqui:
  (a) só os locutores do teste mestre que têm as duas classes (IC por locutor, 2.000 repetições, seed 42);
  (b) AUC dentro de cada locutor (com >= 5 clipes por classe), média simples entre locutores.
Saída: results/v7_poshoc.csv.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import v7  # noqa: E402
from src.arquivos import salvar_csv  # noqa: E402

notas = pd.read_parquet("data/processed/v7_notas.parquet")
esc = pd.read_json("results/v7_escolha.json", typ="series")
ml = notas[(notas.bloco == "ml_itw") & (notas.split == "teste_mestre")]
ncl = ml.groupby("locutor").label.nunique()
ambos = ml[ml.locutor.isin(ncl[ncl == 2].index)]
linhas = []
for rc in v7.RECEITAS:
    e = esc["por_receita"][rc]
    cortes = {r: e["cortes"][r] for r in v7.REGRAS}
    L, _ = v7.medir(ambos, rc, cortes, "locutor", v7.N_BOOT)
    linhas.append({"análise": "(a) ML-ITW teste, só locutores com as duas classes", "receita": rc, **L})
    aucs = []
    for loc, g in ambos.groupby("locutor"):
        if (g.label == 0).sum() >= 5 and (g.label == 1).sum() >= 5:
            aucs.append(roc_auc_score(g.label, g[f"z_{rc}"]))
    linhas.append({"análise": "(b) ML-ITW teste, AUC dentro de cada locutor (>= 5 clipes por classe), média entre locutores",
                   "receita": rc, "AUC": float(np.mean(aucs)), "locutores_real": len(aucs), "locutores_falso": len(aucs),
                   "AUC mediana entre locutores": float(np.median(aucs)),
                   "AUC min": float(np.min(aucs)), "AUC max": float(np.max(aucs))})
t = pd.DataFrame(linhas)
salvar_csv(t, "results/v7_poshoc.csv")
pd.set_option("display.width", 220)
print(t[["análise", "receita", "n_real", "n_falso", "locutores_real", "locutores_falso", "AUC", "AUC IC2.5", "AUC IC97.5",
         "AUC mediana entre locutores", "AUC min", "AUC max"]].round(3).to_string())
