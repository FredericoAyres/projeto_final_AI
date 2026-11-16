#!/usr/bin/env python3
"""Figura results/slides/ctxw_01_acuracia_antes_depois.png (le so results/diagnostico_whisper_idioma_metricas.csv; sem nomes, sem texto)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "results/diagnostico_whisper_idioma_metricas.csv")
m = m[m.recorte == "todos"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
CORES = {"15/11 (pt forcado)": "#8a8f98", "corrigido": "#2a6fb0"}
ROT = {"15/11 (pt forcado)": "15/11 (Whisper forçado a pt)", "corrigido": "corrigido (Whisper auto)"}
METS = [("acc", "acurácia"), ("tpr", "recall\n(golpes alertados)"), ("fpr", "falso alarme\n(normais alertados)")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, conj, tit in zip(axs, ["RESERVADO (principal)", "DEV (usado no diagnostico)"], ["RESERVADO (15 pessoas, 60 clipes)\nprincipal", "DEV (8 pessoas, 40 clipes)\nusado no diagnóstico"]):
    d = m[m.conjunto == conj]
    for j, v in enumerate(["15/11 (pt forcado)", "corrigido"]):
        r = d[d.versao == v].iloc[0]
        x = np.arange(3) + (j - 0.5) * 0.36
        y = np.array([r[k] for k, _ in METS]) * 100
        lo = np.array([r[k + "_lo"] for k, _ in METS]) * 100
        hi = np.array([r[k + "_hi"] for k, _ in METS]) * 100
        ax.bar(x, y, 0.34, color=CORES[v], label=ROT[v])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333", capsize=3, lw=1)
        for xi, yi, hi_i in zip(x, y, hi):
            ax.text(xi, hi_i + 2, f"{yi:.0f}%", ha="center", fontsize=10)
    triv = d.acc_trivial.iloc[0] * 100
    ax.hlines(triv, -0.5, 0.0, color="#b03a48", lw=1.6, ls="--")
    ax.text(-0.48, triv + 2, f"trivial {triv:.0f}%", color="#b03a48", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in METS]); ax.set_title(tit, fontsize=11)
    ax.set_ylim(0, 112)
axs[0].set_ylabel("%  (alerta = risco médio ou alto)")
axs[0].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Módulo de contexto: antes e depois de corrigir o idioma do Whisper", y=1.0)
fig.text(0.5, -0.02, "Barras de erro: IC95 por bootstrap de pessoas (2.000). Teste de 23 pessoas, um gerador; textos de pedido repetidos entre pessoas.", ha="center", fontsize=9, color="#555")
fig.tight_layout()
fig.savefig(ROOT / "results/slides/ctxw_01_acuracia_antes_depois.png", dpi=160, bbox_inches="tight")
