#!/usr/bin/env python3
"""Parte A da v4: acurácia e acurácia balanceada RETROATIVAS das seis execuções da v3.

Usa só as notas já salvas (results/v3_*_notas_val.parquet e results/v3_*_notas_dev_shift.parquet);
nada é retreinado e nenhum teste é consultado. Avaliação por janelas (z_janelas), como no fechamento
da v3. Limiar 1 = ponto de EER da validação inteira (o mesmo procedimento da v3); limiar 2 = 0,5 de
probabilidade (logit 0). Grava results/v3_acuracias.csv.

    python scripts/acuracias_v3.py
"""
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from src.arquivos import salvar_csv  # noqa: E402
from src.metrics import eer_e_limiar, metricas_completas, sigmoide  # noqa: E402


def main():
    linhas = []
    for seed in (42, 123, 2026):
        for cond in ("repeat", "zero"):
            exp = f"v3_{cond}_s{seed}"
            v = pd.read_parquet(f"results/{exp}_notas_val.parquet")
            d = pd.read_parquet(f"results/{exp}_notas_dev_shift.parquet")
            lim = eer_e_limiar(v.label.values, v.z_janelas.values)[1]
            for conjunto, g in [("validação", v), ("dev_shift", d)]:
                m = metricas_completas(g.label.values, g.z_janelas.values, lim)
                linhas.append({"experimento": exp, "condição": cond, "seed": seed, "conjunto": conjunto,
                               "avaliação": "janelas", "limiar (prob.)": float(sigmoide(lim)),
                               "limiar (logit)": lim, **m})
    t = pd.DataFrame(linhas)
    salvar_csv(t, "results/v3_acuracias.csv")
    pd.set_option("display.width", 250)
    cols = ["experimento", "conjunto", "limiar (prob.)", "AUC", "EER", "acurácia", "acurácia balanceada",
            "acurácia (0,5)", "acurácia balanceada (0,5)", "acurácia trivial"]
    print(t[cols].round(3).to_string(index=False))
    print("\nmédia por condição e conjunto:")
    print(t.groupby(["conjunto", "condição"])[cols[3:]].agg(["mean", "min", "max"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
