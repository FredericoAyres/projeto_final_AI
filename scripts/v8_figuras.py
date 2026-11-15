#!/usr/bin/env python3
"""Figuras da v8 (fine-tuning parcial do XLS-R a partir da T3). Só LÊ de results/v8_*.

Saída: results/v8_fig_*.png. O estilo segue o de scripts/v7_slides.py (fundo creme, grade clara). Cada figura
usa uma cor de destaque só (a receita vencedora pela validação, R1) e cinza para o resto, com rótulos diretos.

    python scripts/v8_figuras.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FUNDO, TEXTO, GRADE, CINZA, DESTAQUE, REF = "#FBFAF6", "#22272E", "#E8E5DC", "#7D848C", "#B0306A", "#3A3F45"
plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO, "font.size": 11,
    "axes.edgecolor": "#9AA0A6", "xtick.color": "#4A5058", "ytick.color": "#4A5058", "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": TEXTO, "axes.labelcolor": TEXTO, "axes.titleweight": "bold", "axes.titlesize": 12,
})
DPI = 110
ROTULO = {"R0": "R0 = T3 (sem fine-tuning)", "R1_s42": "R1 seed 42", "R1_s123": "R1 seed 123",
          "R1 (média das seeds)": "R1 (média das 2 seeds)", "R2_s42": "R2 (só camada 7)", "R3_s42": "R3 (cabeça aleatória)"}
FONTES = {"ml_itw": "ML-ITW (internet real)", "in_the_wild": "In-the-Wild (38 locutores)", "radar_en": "RADAR en"}
CONTROLES = {"asvspoof2019_la": "ASVspoof 2019 LA", "asvspoof2021_df": "ASVspoof 2021 DF",
             "llamapartialspoof": "LlamaPartialSpoof", "brspeech_df": "BRSpeech-DF (pt-BR)"}


def cor(c):
    return DESTAQUE if c.startswith("R1") else CINZA


def salvar(fig, nome):
    p = R / f"v8_fig_{nome}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("gravado", p.relative_to(ROOT))


def fig_diferenca(pr):
    """Diferença da média da AUC (3 fontes) para a R0, com IC95 pareado."""
    ordem = ["R1_s42", "R1_s123", "R1 (média das seeds)", "R2_s42", "R3_s42"]
    d = pr.set_index("candidato").loc[[c for c in ordem if c in set(pr.candidato)]]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, (c, x) in enumerate(d.iterrows()):
        y = len(d) - 1 - i
        v = x["dif. média vs R0 (T3)"]
        lo, hi = x["dif. média vs R0 IC2.5"], x["dif. média vs R0 IC97.5"]
        if pd.notna(lo):
            ax.plot([lo, hi], [y, y], color=cor(c), lw=2, solid_capstyle="round")
        ax.plot(v, y, "o", ms=9, color=cor(c), mec=FUNDO, mew=2, zorder=3)
        txt = f"{v:+.4f}".replace(".", ",") + (f"  [{lo:+.4f}; {hi:+.4f}]".replace(".", ",") if pd.notna(lo) else "")
        ax.annotate(txt, (max(v, hi if pd.notna(hi) else v), y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=TEXTO)
    ax.axvline(0, color=REF, lw=1.2, ls="--")
    ax.text(0.0003, -0.45, "igual à T3", ha="left", fontsize=9, color=REF)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([ROTULO[c] for c in d.index][::-1])
    ax.set_xlim(-0.016, 0.016)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("diferença na AUC média do teste mestre (ML-ITW, In-the-Wild, RADAR en) em relação à T3")
    ax.set_title("Fine-tuning × T3 no teste mestre (IC95 pareado; a R1 foi escolhida pela validação)", loc="left")
    ax.grid(axis="y", visible=False)
    salvar(fig, "1_diferenca_vs_t3")


def fig_fontes(tm):
    """AUC por fonte do teste mestre, com IC95, por candidato."""
    ordem = ["R0", "R1_s42", "R1_s123", "R2_s42", "R3_s42"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (rec, nome) in zip(axs, FONTES.items()):
        t = tm[tm.recorte == rec].set_index("candidato").loc[ordem]
        r0 = t.loc["R0", "AUC"]
        ax.axvline(r0, color=REF, lw=1, ls="--")
        for i, (c, x) in enumerate(t.iterrows()):
            y = len(t) - 1 - i
            ax.plot([x["AUC IC2.5"], x["AUC IC97.5"]], [y, y], color=cor(c) if c != "R0" else REF, lw=2,
                    solid_capstyle="round")
            ax.plot(x.AUC, y, "o", ms=8, color=cor(c) if c != "R0" else REF, mec=FUNDO, mew=2, zorder=3)
            ax.annotate(f"{x.AUC:.3f}".replace(".", ","), (x["AUC IC97.5"], y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=9)
        ax.set_title(nome, loc="left")
        ax.set_xlabel("AUC (IC95)")
        lo, hi = t["AUC IC2.5"].min(), t["AUC IC97.5"].max()
        ax.set_xlim(lo - 0.005, hi + (hi - lo) * 0.35)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(ordem)))
    axs[0].set_yticklabels([ROTULO[c] for c in ordem][::-1])
    fig.suptitle("Teste mestre por fonte (linha tracejada = T3). No RADAR, o IC é por clipe (otimista).",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    salvar(fig, "2_teste_por_fonte")


def fig_val_teste(val, pr):
    """Validação (época escolhida) × teste: a ordem da validação se manteve no teste?"""
    v = val[val["época escolhida"]].set_index("candidato")["AUC média por fonte"]
    t = pr.set_index("candidato")["média AUC (3 fontes)"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    desl = {"R1_s42": (-10, 8), "R1_s123": (8, -4), "R0": (8, -4), "R2_s42": (8, -10), "R3_s42": (8, -4)}
    for c in v.index:
        ax.plot(v[c], t[c], "o", ms=10, color=cor(c) if c != "R0" else REF, mec=FUNDO, mew=2, zorder=3)
        ax.annotate(ROTULO[c], (v[c], t[c]), xytext=desl.get(c, (8, 0)), textcoords="offset points",
                    fontsize=9.5, ha="right" if desl.get(c, (8, 0))[0] < 0 else "left", va="center")
    ax.set_xlabel("validação: média da AUC por fonte (6 fontes, época escolhida)")
    ax.set_ylabel("teste mestre: média da AUC (3 fontes)")
    ax.set_title("A ordem da validação se repetiu no teste", loc="left")
    ax.margins(x=0.25, y=0.15)
    salvar(fig, "3_validacao_x_teste")


def fig_controles(ct):
    """Controles de estúdio (esquecimento): AUC por candidato."""
    ordem = ["R0", "R1_s42", "R1_s123", "R2_s42", "R3_s42"]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (rec, nome) in zip(axs, CONTROLES.items()):
        t = ct[ct.recorte == rec].set_index("candidato").loc[ordem]
        ax.axvline(t.loc["R0", "AUC"], color=REF, lw=1, ls="--")
        for i, (c, x) in enumerate(t.iterrows()):
            y = len(t) - 1 - i
            if pd.notna(x["AUC IC2.5"]):
                ax.plot([x["AUC IC2.5"], x["AUC IC97.5"]], [y, y], color=cor(c) if c != "R0" else REF, lw=2,
                        solid_capstyle="round")
            ax.plot(x.AUC, y, "o", ms=8, color=cor(c) if c != "R0" else REF, mec=FUNDO, mew=2, zorder=3)
        ax.set_title(nome, loc="left", fontsize=11)
        ax.set_xlabel("AUC")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(ordem)))
    axs[0].set_yticklabels([ROTULO[c] for c in ordem][::-1])
    fig.suptitle("Controles de estúdio (esquecimento): a R1 manteve ou melhorou a T3 (linha tracejada)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    salvar(fig, "4_controles")


def fig_perda():
    """Perda média de treino por época, por rodada."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    finais = {}
    for c in ("R1_s42", "R1_s123", "R2_s42", "R3_s42"):
        h = json.loads((R / f"v8_treino_{c}.json").read_text(encoding="utf-8"))["historico"]
        x = [e["epoca"] for e in h]
        y = [e["perda_media"] for e in h]
        ax.plot(x, y, "-o", lw=2, ms=8, color=cor(c), mec=FUNDO, mew=2)
        finais[c] = y[-1]
    # rótulos em ordem da perda final, com um espaço mínimo entre eles (para não se sobreporem)
    pos, ult = {}, None
    for c in sorted(finais, key=finais.get):
        pos[c] = finais[c] if ult is None else max(finais[c], ult + 0.013)
        ult = pos[c]
    for c, yl in pos.items():
        ax.annotate(f"{ROTULO[c]}: {finais[c]:.3f}".replace(".", ","), (2, finais[c]), xytext=(2.08, yl),
                    textcoords="data", fontsize=9.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRADE, lw=1))
    ax.set_xticks([1, 2])
    ax.set_xlim(0.8, 3.0)
    ax.set_xlabel("época")
    ax.set_ylabel("perda média de treino (BCE)")
    ax.set_title("Perda de treino: a cabeça aleatória (R3) começa muito pior", loc="left")
    salvar(fig, "5_perda_treino")


def main():
    pr = pd.read_csv(R / "v8_principal.csv")
    tm = pd.read_csv(R / "v8_teste_mestre.csv")
    ct = pd.read_csv(R / "v8_controles.csv")
    val = pd.read_csv(R / "v8_validacao.csv")
    fig_diferenca(pr)
    fig_fontes(tm)
    fig_val_teste(val, pr)
    fig_controles(ct)
    fig_perda()


if __name__ == "__main__":
    main()
