#!/usr/bin/env python3
"""Figuras do teste "WhatsApp" (results/slides/tw_01..tw_05.png). Le so results/teste_whatsapp_*.csv (sem nomes, sem texto)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
S = R / "slides"
S.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
AZUL, LARANJA, CINZA, VERDE = "#2a6fb0", "#d9792b", "#8a8f98", "#3a9d5d"
GRUPOS = ["real_sem_golpe", "real_com_golpe", "ia_pedido_dinheiro", "ia_neutro"]
ROT = {"real_sem_golpe": "real\nsem golpe", "real_com_golpe": "real\ncom golpe", "ia_pedido_dinheiro": "IA\npede dinheiro", "ia_neutro": "IA\nneutra"}
COR = {"real_sem_golpe": VERDE, "real_com_golpe": LARANJA, "ia_pedido_dinheiro": "#b03a48", "ia_neutro": AZUL}

det = pd.read_csv(R / "teste_whatsapp_detector.csv")
grp = pd.read_csv(R / "teste_whatsapp_grupos.csv")
ctx = pd.read_csv(R / "teste_whatsapp_contexto.csv")
cgr = pd.read_csv(R / "teste_whatsapp_contexto_grupos.csv")
notas = pd.read_csv(R / "teste_whatsapp_notas.csv")
comb = pd.read_csv(R / "teste_whatsapp_combinado.csv")
NOTA = "100 áudios, 23 pessoas, um gerador. IC95 por bootstrap de pessoas (2.000). Teste feito uma vez."


def barras_ic(ax, x, v, lo, hi, cor, rot=None):
    ax.bar(x, v, color=cor, width=0.62)
    ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#222", capsize=4, lw=1.3)
    for xi, vi in zip(x, v):
        ax.text(xi, 1.02, f"{100 * vi:.0f}%", ha="center", fontsize=11, fontweight="bold")


# ---- tw_01: acuracia dos dois modelos
t3 = det[(det.modelo == "T3") & (det.corte_regra == "M") & (det.condicao == "A")].iloc[0]
f0 = det[(det.modelo == "F0") & (det.corte_regra == "A") & (det.condicao == "A")].iloc[0]
cx = ctx[(ctx.alerta == "medio+ (principal)") & (ctx.recorte == "todos")].iloc[0]
fig, ax = plt.subplots(figsize=(9, 5))
itens = [("Detector de voz\nT3, corte M", t3, "acc", AZUL), ("Detector de voz\nT3, corte M", t3, "bacc", AZUL),
         ("Contexto de golpe\nalerta médio+", cx, "acc", LARANJA), ("Contexto de golpe\nalerta médio+", cx, "bacc", LARANJA)]
x = np.array([0, 1, 2.8, 3.8])
v = np.array([i[1][i[2]] for i in itens])
lo = np.array([i[1].get(i[2] + "_lo", np.nan) for i in itens])
hi = np.array([i[1].get(i[2] + "_hi", np.nan) for i in itens])
for k in range(4):
    ax.bar(x[k], v[k], color=itens[k][3], width=0.8, alpha=1 if k % 2 == 0 else 0.55)
    if not np.isnan(lo[k]):
        ax.errorbar(x[k], v[k], yerr=[[v[k] - lo[k]], [hi[k] - v[k]]], fmt="none", ecolor="#222", capsize=4, lw=1.3)
    ax.text(x[k], min(hi[k] if not np.isnan(hi[k]) else v[k], 1.0) + 0.02, f"{100 * v[k]:.0f}%", ha="center", fontweight="bold")
ax.axhline(t3["acc_trivial"], xmin=0.03, xmax=0.45, color=CINZA, ls="--", lw=1)
ax.axhline(cx["acc_trivial"], xmin=0.55, xmax=0.97, color=CINZA, ls="--", lw=1)
ax.text(0, t3["acc_trivial"] - 0.05, f"trivial {100 * t3['acc_trivial']:.0f}%", ha="center", color="#fff", fontsize=9)
ax.text(3.3, cx["acc_trivial"] + 0.012, f"trivial {100 * cx['acc_trivial']:.0f}%", ha="center", color=CINZA, fontsize=9)
ax.set_xticks(x, ["acurácia", "acurácia\nbalanceada", "acurácia", "acurácia\nbalanceada"])
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.01, .2), [f"{int(i * 100)}%" for i in np.arange(0, 1.01, .2)])
ax.text(0.5, -0.2, "Detector de voz (T3, corte M)", ha="center", transform=ax.get_xaxis_transform(), fontweight="bold", color=AZUL)
ax.text(3.3, -0.2, "Contexto de golpe (alerta médio+)", ha="center", transform=ax.get_xaxis_transform(), fontweight="bold", color=LARANJA)
ax.set_title("Teste WhatsApp: acurácia dos dois modelos", loc="left", fontweight="bold")
fig.text(0.01, 0.01, NOTA + " Linha tracejada = acurácia trivial.", fontsize=8, color=CINZA)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(S / "tw_01_acuracia_dois_modelos.png", dpi=160)
plt.close(fig)

# ---- tw_02: taxas por grupo
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
g1 = grp[(grp.modelo == "T3") & (grp.corte_regra == "M") & (grp.condicao == "A")].set_index("grupo").loc[GRUPOS]
g2 = cgr[cgr.alerta == "medio+"].set_index("grupo").loc[GRUPOS]
for ax, g, kcol, tit in ((axs[0], g1, "acusados_como_ia", "Detector de voz: % acusado como IA (T3, corte M)"), (axs[1], g2, "com_alerta", "Contexto: % com alerta (médio ou alto)")):
    xs = np.arange(4)
    ax.bar(xs, g.taxa, color=[COR[k] for k in GRUPOS], width=0.65)
    ax.errorbar(xs, g.taxa, yerr=[g.taxa - g.wilson_lo, g.wilson_hi - g.taxa], fmt="none", ecolor="#222", capsize=4, lw=1.2)
    for i, (t, n, k) in enumerate(zip(g.taxa, g.n, g[kcol])):
        ax.text(i, min(g.wilson_hi.iloc[i], 1) + 0.02, f"{100 * t:.0f}%\n({int(k)}/{int(n)})", ha="center", fontsize=9)
    ax.set_xticks(xs, [ROT[k] for k in GRUPOS])
    ax.set_title(tit, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.2)
axs[0].set_yticks(np.arange(0, 1.01, .25), [f"{int(i * 100)}%" for i in np.arange(0, 1.01, .25)])
fig.text(0.01, 0.01, "Barras de erro: Wilson (ignora o agrupamento por pessoa; otimista). " + NOTA, fontsize=8, color=CINZA)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(S / "tw_02_taxas_por_grupo.png", dpi=160)
plt.close(fig)

# ---- tw_03: histograma das notas
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, col, tit, cortes in ((axs[0], "z_T3_A", "T3 (condição A)", [("M", 0.2326)]), (axs[1], "z_F0_A", "F0 (condição A)", [("A", -0.0395)])):
    bins = np.linspace(notas[col].min() - 1, notas[col].max() + 1, 28)
    ax.hist(notas[notas.autenticidade == "real"][col], bins=bins, color=VERDE, alpha=.75, label="real (44)")
    ax.hist(notas[notas.autenticidade == "ia"][col], bins=bins, color="#b03a48", alpha=.75, label="IA (56)")
    for n, c in cortes:
        ax.axvline(c, color="#222", ls="--", lw=1.2)
        ax.text(c, ax.get_ylim()[1] * 0.95, f" corte {n}", fontsize=9)
    ax.set_xlabel("nota (logit; maior = mais parecido com IA)")
    ax.set_title(tit, loc="left", fontweight="bold")
    ax.legend(frameon=False)
fig.text(0.01, 0.01, NOTA, fontsize=8, color=CINZA)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(S / "tw_03_histograma_notas.png", dpi=160)
plt.close(fig)

# ---- tw_04: matriz de confusao do contexto
yv = (notas.conteudo == "golpe").values
al = notas.alerta_medio.values == 1
M = np.array([[np.sum(yv & al), np.sum(yv & ~al)], [np.sum(~yv & al), np.sum(~yv & ~al)]])
fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.4)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{M[i, j]}\n({100 * M[i, j] / M[i].sum():.0f}% da linha)", ha="center", va="center", fontsize=12,
                color="#fff" if M[i, j] > M.max() * .6 else "#222")
ax.set_xticks([0, 1], ["com alerta\n(médio ou alto)", "sem alerta"])
ax.set_yticks([0, 1], ["golpe (73)", "normal (27)"])
ax.xaxis.tick_top()
ax.set_title("Contexto: verdade × alerta", loc="left", pad=36, fontweight="bold")
fig.text(0.01, 0.01, "Golpe = real com golpe + IA que pede dinheiro. Indeterminado conta como sem alerta.", fontsize=8, color=CINZA)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(S / "tw_04_matriz_contexto.png", dpi=160)
plt.close(fig)

# ---- tw_05: tabela grupo x nivel
c = comb.set_index("grupo").loc[GRUPOS]
hip = {"real_sem_golpe": "baixo", "real_com_golpe": "atencao", "ia_pedido_dinheiro": "alto", "ia_neutro": "atencao"}
fig, ax = plt.subplots(figsize=(9.6, 4.2))
ax.axis("off")
cel = [[f"{int(c.loc[g, n])}/{int(c.loc[g, 'n'])}  ({100 * c.loc[g, n] / c.loc[g, 'n']:.0f}%)" for n in ("baixo", "atencao", "alto")] for g in GRUPOS]
tb = ax.table(cellText=cel, rowLabels=[ROT[g].replace("\n", " ") for g in GRUPOS], colLabels=["baixo", "atenção", "alto"],
              cellLoc="center", loc="center", colWidths=[.22, .22, .22])
tb.auto_set_font_size(False)
tb.set_fontsize(12)
tb.scale(1, 2.2)
for i, g in enumerate(GRUPOS):
    j = ["baixo", "atencao", "alto"].index(hip[g])
    tb[(i + 1, j)].set_facecolor("#dfeaf6")
ax.set_title("Resultado combinado: grupo × nível de alerta", loc="left", fontweight="bold")
fig.text(0.01, 0.02, "Voz = IA se T3 ≥ corte M; conteúdo = alerta se risco ≥ médio. Alto = os dois; atenção = um; baixo = nenhum.\n"
         "Célula sombreada = hipótese do protocolo (não promessa). Não é acurácia.", fontsize=8, color=CINZA)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(S / "tw_05_grupo_por_nivel.png", dpi=160)
plt.close(fig)
print("figuras gravadas em", S)
