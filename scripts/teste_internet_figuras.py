#!/usr/bin/env python3
"""Figuras da avaliação do F0 nos bancos da internet (RADAR eval e ML-ITW). Só lê do disco:
results/teste_internet_*.csv e data/processed/teste_internet_notas.parquet. Nada é recalculado
além do desenho (a ROC e os histogramas usam as notas por clipe).

    python scripts/teste_internet_figuras.py [--quick]

Saída: results/slides/ti_0N_*.png (13,33 x 7,5 pol., dpi 180). Real = azul, IA = laranja, neutro cinza.
Os títulos (a conclusão em linguagem simples) estão em TITULOS, escritos depois de ver os números.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

REAL_C, IA_C, NEUTRO = "#2B63C9", "#E0662C", "#5E6670"
FUNDO, TEXTO, GRADE, CLARO = "#FBFAF6", "#22272E", "#E8E5DC", "#B4BAC1"
TAM, DPI = (13.33, 7.5), 180
CORTE_A, CORTE_B = -0.0395, 8.2184
_fontes = {f.name for f in font_manager.fontManager.ttflist}
FONTE = "IBM Plex Sans" if "IBM Plex Sans" in _fontes else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONTE, "font.size": 14, "text.color": TEXTO, "axes.labelcolor": TEXTO,
    "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 14, "legend.frameon": False, "figure.facecolor": FUNDO, "axes.facecolor": FUNDO,
    "savefig.facecolor": FUNDO, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA0A6", "xtick.color": "#4A5058", "ytick.color": "#4A5058", "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.7, "axes.axisbelow": True})

NOME_FONTE = {"radar_eval": "RADAR eval", "ml_itw": "ML-ITW"}
IDIOMA = {"en": "inglês", "en-sg": "inglês (Singapura)", "zh": "chinês", "zh-tw": "chinês (Taiwan)", "ja": "japonês",
          "vi": "vietnamita", "chinese": "chinês", "english": "inglês", "japanese": "japonês", "portuguese": "português",
          "turkish": "turco", "korean": "coreano", "german": "alemão", "spanish": "espanhol", "russian": "russo",
          "hindi": "híndi", "hungarian": "húngaro", "french": "francês", "ukrainian": "ucraniano", "hebrew": "hebraico"}

# títulos = conclusão (escritos depois dos números); subtítulos = o que olhar
TITULOS = {
    "ti_01": ("Fora dos bancos já vistos, a AUC cai de 0,98 para cerca de 0,7", "AUC com intervalo de 95%; ponto cheio = estimativa, vazado = indício (< 30 locutores por classe)"),
    "ti_02": ("Nos dois bancos, as notas de real e de IA se sobrepõem muito", "Nota do sistema (logit) por clipe; linhas: cortes A e B (fixados antes) e o corte de EER da própria fonte"),
    "ti_03": ("A duração não explica a queda: a AUC é parecida em todas as faixas", "AUC por faixa de duração original e no RADAR com durações pareadas entre real e IA"),
    "ti_04": ("Com o corte B, 3 de cada 4 falsos passam como reais", "Curva ROC por fonte; os pontos marcam onde os cortes A e B caem"),
    "ti_05": ("Só com metadados chega-se perto do sistema, sem passar de 0,7", "AUC do sistema × AUC de um classificador que só vê metadados (duração, silêncio, volume)"),
}


def br(v, d=2):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def mil(n):
    return f"{int(n):,}".replace(",", ".")


def virgula(ax, eixo="x", d=2):
    f = FuncFormatter(lambda v, _: br(v, d))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(f)


def quadro(chave, nrows=1, ncols=1, top=0.76, bottom=0.11, left=0.07, right=0.98, wspace=0.25, hspace=0.4, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=TAM, squeeze=False, **kw)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
    t, s = TITULOS[chave]
    fig.text(0.012, 0.975, t, fontsize=23, fontweight="bold", va="top", color=TEXTO)
    fig.text(0.012, 0.905, s, fontsize=15, va="top", color=NEUTRO)
    return fig, axs


def legenda(fig, handles, y=0.855, ncol=None):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, y), ncol=ncol or len(handles), fontsize=14,
               frameon=False, handlelength=1.6, columnspacing=1.6, borderaxespad=0)


def salvar(fig, nome, destino):
    p = destino / f"{nome}.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"{p} ({p.stat().st_size / 1024:.0f} KB)", flush=True)


def painel_auc(ax, linhas, titulo):
    """Pontos de AUC com IC95 (uma linha por recorte), de cima para baixo; o valor vai junto do nome."""
    ys = np.arange(len(linhas))[::-1]
    for y, (_, r) in zip(ys, linhas.iterrows()):
        tem = np.isfinite(r.get("AUC", np.nan))
        rot = f"{r['rótulo']}  {br(r.AUC, 2)}" if tem else f"{r['rótulo']}  (só contagem)"
        if tem:
            cheio = r.marca == "estimativa" or str(r.marca).startswith("IC otimista")
            ax.plot([r["AUC IC2.5"], r["AUC IC97.5"]], [y, y], color=TEXTO if cheio else NEUTRO, lw=2.2)
            ax.plot(r.AUC, y, "o", ms=10, mfc=TEXTO if cheio else FUNDO, mec=TEXTO if cheio else NEUTRO, mew=2)
        ax.text(-0.02, y, rot, va="center", ha="right", fontsize=13, transform=ax.get_yaxis_transform(),
                fontweight="bold" if r.recorte == "inteiro" else "normal", color=TEXTO if tem else NEUTRO)
    ax.axvline(0.5, color=NEUTRO, ls="--", lw=1.2)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(linhas) - 0.3)
    ax.set_xlim(0.1, 1.0)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0])
    virgula(ax, d=1)
    ax.set_title(titulo, loc="left", fontsize=16, fontweight="bold", pad=8)
    ax.set_xlabel("AUC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    R = Path("results/teste_internet_quick") if a.quick else Path("results")
    destino = (R / "slides") if a.quick else Path("results/slides")
    destino.mkdir(parents=True, exist_ok=True)
    notas = pd.read_parquet(Path("data/processed") / ("teste_internet_notas_quick.parquet" if a.quick
                                                      else "teste_internet_notas.parquet"))
    notas = notas[~notas.ilegivel]
    tf = pd.read_csv(R / "teste_internet_por_fonte.csv")
    ti = pd.read_csv(R / "teste_internet_por_idioma.csv")
    tp = pd.read_csv(R / "teste_internet_por_plataforma.csv")
    ts = pd.read_csv(R / "teste_internet_sensibilidade.csv")
    ta = pd.read_csv(R / "teste_internet_auditoria_atalho.csv")

    # ---------------- 1. AUC com IC95 por fonte, idioma e plataforma
    fig, axs = quadro("ti_01", 1, 3, left=0.17, right=0.985, wspace=0.95, bottom=0.12, top=0.73)

    def com_rotulo(df, fonte, nome_inteiro):
        d = df.copy()
        d["rótulo"] = [nome_inteiro if r.recorte == "inteiro" else IDIOMA.get(r.valor, r.valor) for _, r in d.iterrows()]
        return d
    r_rad = pd.concat([tf[tf.fonte == "radar_eval"], ti[ti.fonte == "radar_eval"].sort_values("n_real", ascending=False)])
    painel_auc(axs[0, 0], com_rotulo(r_rad, "radar_eval", "RADAR inteiro"), "RADAR: por idioma")
    r_pl = pd.concat([tf[tf.fonte == "ml_itw"], tp.sort_values("n_real", ascending=False)])
    painel_auc(axs[0, 1], com_rotulo(r_pl, "ml_itw", "ML-ITW inteiro"), "ML-ITW: por plataforma")
    r_id = ti[ti.fonte == "ml_itw"].sort_values("n_real", ascending=False)
    painel_auc(axs[0, 2], com_rotulo(r_id, "ml_itw", ""), "ML-ITW: idioma")
    for ax, txt in zip(axs[0], ("IC por clipe: otimista (sem locutor)", "IC por locutor", "IC por locutor")):
        ax.text(0, -0.15, txt, transform=ax.transAxes, fontsize=13, color=NEUTRO)
    legenda(fig, [Line2D([], [], ls="-", marker="o", ms=10, color=TEXTO, label="estimativa (AUC e IC95)"),
                  Line2D([], [], ls="-", marker="o", ms=10, mfc=FUNDO, mec=NEUTRO, mew=2, color=NEUTRO,
                         label="indício: < 30 locutores por classe"),
                  Line2D([], [], ls="--", color=NEUTRO, label="0,5 = acaso")], y=0.855)
    salvar(fig, "ti_01_auc_recortes", destino)

    # ---------------- 2. histogramas real × IA por fonte
    fig, axs = quadro("ti_02", 1, 2, wspace=0.18, top=0.74)
    for ax, f in zip(axs[0], ("radar_eval", "ml_itw")):
        g = notas[notas.fonte == f]
        lo, hi = np.quantile(g.z_F0, [0.005, 0.995])
        bins = np.linspace(lo, hi, 70)
        for lab, cor in ((0, REAL_C), (1, IA_C)):
            ax.hist(g[g.label == lab].z_F0.clip(lo, hi), bins=bins, density=True, color=cor, alpha=0.55)
        r = tf[tf.fonte == f].iloc[0]
        for t, est in ((CORTE_A, dict(color=TEXTO, ls="--", lw=2.2)), (CORTE_B, dict(color=TEXTO, ls=":", lw=2.6)),
                       (r.corte_EER_da_fonte, dict(color=NEUTRO, ls="-.", lw=2.2))):
            if lo <= t <= hi:
                ax.axvline(t, **est)
        ax.set_title(f"{NOME_FONTE[f]} ({mil(r.n_real)} reais, {mil(r.n_falso)} IA)", loc="left", fontsize=16,
                     fontweight="bold")
        ax.set_xlabel("nota do sistema (logit; maior = mais cara de IA)")
        ax.set_yticks([])
        virgula(ax, d=0)
        ax.text(0.01, 0.97, "corte de EER\nda fonte: " + br(r.corte_EER_da_fonte), transform=ax.transAxes, ha="left",
                va="top", fontsize=13, color=NEUTRO)
    legenda(fig, [Patch(facecolor=REAL_C, alpha=0.55, label="real"), Patch(facecolor=IA_C, alpha=0.55, label="IA"),
                  Line2D([], [], color=TEXTO, ls="--", lw=2.2, label=f"corte A = {br(CORTE_A)}"),
                  Line2D([], [], color=TEXTO, ls=":", lw=2.6, label=f"corte B = {br(CORTE_B)}"),
                  Line2D([], [], color=NEUTRO, ls="-.", lw=2.2, label="corte de EER da fonte (diagnóstico)")], y=0.85)
    salvar(fig, "ti_02_histogramas", destino)

    # ---------------- 3. duração
    fig, axs = quadro("ti_03", 1, 2, wspace=0.3, top=0.74, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0, 0]
    faixas = ["< 2 s", "2 a 4 s", ">= 4 s"]
    fx = ts[ts["análise"] == "a) AUC por faixa de duração original"]
    for i, (f, cor, dx) in enumerate((("radar_eval", TEXTO, -0.15), ("ml_itw", NEUTRO, 0.15))):
        for j, b in enumerate(faixas):
            r = fx[(fx.fonte == f) & (fx.valor == b)]
            if len(r) and np.isfinite(r.AUC.iloc[0]):
                r = r.iloc[0]
                ax.plot([j + dx] * 2, [r["AUC IC2.5"], r["AUC IC97.5"]], color=cor, lw=2.2)
                ax.plot(j + dx, r.AUC, "o" if f == "radar_eval" else "s", ms=11, color=cor)
                txt = br(r.AUC, 3) + "\n" + f"{mil(r.n_real)} / {mil(r.n_falso)}"
                if f == "radar_eval":   # acima do ponto
                    ax.text(j + dx, r["AUC IC97.5"] + 0.012, txt, fontsize=13, ha="center", va="bottom", color=cor)
                else:                   # à direita, abaixo do ponto
                    ax.text(j + dx + 0.05, r.AUC - 0.02, txt, fontsize=13, ha="left", va="top", color=cor)
            elif j == 0:
                ax.text(j + dx, 0.52, f"{NOME_FONTE[f]}:\nsem clipes\n< 2 s", ha="center", fontsize=13, color=cor)
    ax.set_xticks(range(3), ["< 2 s", "2 a 4 s", "≥ 4 s"])
    ax.set_xlim(-0.95, 2.95)
    ax.set_ylim(0.5, 1.0)
    virgula(ax, "y")
    ax.set_ylabel("AUC (IC95)")
    ax.set_title("AUC por faixa de duração original (n real / IA)", loc="left", fontsize=16, fontweight="bold")
    ax = axs[0, 1]
    orig = tf[tf.fonte == "radar_eval"].iloc[0]
    par = ts[ts["análise"] == "a) RADAR com durações pareadas"].iloc[0]
    ax.bar([0], [orig.AUC], color=CLARO, width=0.6)
    ax.plot([0, 0], [orig["AUC IC2.5"], orig["AUC IC97.5"]], color=TEXTO, lw=2)
    ax.bar([1], [par.AUC], color=CLARO, width=0.6, hatch="//", edgecolor=NEUTRO)
    ax.plot([1, 1], [par["AUC min"], par["AUC max"]], color=TEXTO, lw=2)
    for x, v in ((0, orig.AUC), (1, par.AUC)):
        ax.text(x, 0.505, br(v, 3), ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1], [f"original\n{mil(orig.n_real)} / {mil(orig.n_falso)}",
                           f"pareado por duração\n{mil(par.n_real)} / {mil(par.n_falso)}"])
    ax.set_ylim(0.5, 1.0)
    virgula(ax, "y")
    ax.set_title("RADAR: original × pareado", loc="left", fontsize=16, fontweight="bold")
    ax.text(0, -0.25, "barra: IC95 (original) e mín.–máx. de 20 sorteios (pareado)", transform=ax.transAxes,
            fontsize=13, color=NEUTRO)
    legenda(fig, [Line2D([], [], ls="-", marker="o", ms=10, color=TEXTO, label="RADAR (IC por clipe, otimista)"),
                  Line2D([], [], ls="-", marker="s", ms=10, color=NEUTRO, label="ML-ITW (IC por locutor)")], y=0.85)
    salvar(fig, "ti_03_duracao", destino)

    # ---------------- 4. ROC por fonte
    fig, axs = quadro("ti_04", 1, 2, wspace=0.22, top=0.74, left=0.08)
    for ax, f in zip(axs[0], ("radar_eval", "ml_itw")):
        g = notas[notas.fonte == f]
        fpr, tpr, _ = roc_curve(g.label.values, g.z_F0.values)
        ax.plot(fpr, tpr, color=TEXTO, lw=2.6)
        ax.plot([0, 1], [0, 1], color=NEUTRO, ls="--", lw=1.2)
        r = tf[tf.fonte == f].iloc[0]
        for L, mk in (("A", "o"), ("B", "s")):
            ax.plot(r[f"FPR {L}"], r[f"TPR {L}"], mk, ms=13, mfc=IA_C if L == "A" else FUNDO, mec=IA_C, mew=2.5)
            ax.annotate(f"corte {L}: pega {br(100 * r[f'TPR {L}'], 0)}% da IA,\nacusa {br(100 * r[f'FPR {L}'], 0)}% dos reais",
                        (r[f"FPR {L}"], r[f"TPR {L}"]), xytext=(0.36, 0.30 if L == "A" else 0.12),
                        textcoords="axes fraction", fontsize=13, arrowprops=dict(arrowstyle="-", color=NEUTRO, lw=1))
        ax.set_title(f"{NOME_FONTE[f]}: AUC {br(r.AUC, 3)}", loc="left", fontsize=16, fontweight="bold")
        ax.set_xlabel("reais acusados como IA (FPR)")
        ax.set_ylabel("IA detectada (TPR)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        virgula(ax, "x", 1)
        virgula(ax, "y", 1)
    legenda(fig, [Line2D([], [], color=TEXTO, lw=2.6, label="curva ROC"),
                  Line2D([], [], ls="", marker="o", ms=12, mfc=IA_C, mec=IA_C, label="corte A"),
                  Line2D([], [], ls="", marker="s", ms=12, mfc=FUNDO, mec=IA_C, mew=2.5, label="corte B"),
                  Line2D([], [], color=NEUTRO, ls="--", label="acaso")], y=0.85)
    salvar(fig, "ti_04_roc", destino)

    # ---------------- 5. auditoria de atalho
    fig, axs = quadro("ti_05", 1, 1, top=0.74, left=0.08, right=0.7)
    ax = axs[0, 0]
    cl = ta[(ta.tipo == "classificador só com metadados") & (ta.modelo == "regressão logística")]
    barras = [("sistema F0 (voz)", None, TEXTO, None),
              ("metadados (pedido)", "pedido: duração, taxa, canais, fração de silêncio, RMS, pico", NEUTRO, None),
              ("metadados + tamanho do arquivo", "protocolo: + tamanho do arquivo", CLARO, None),
              ("só a duração", "só duração", CLARO, "//")]
    w = 0.2
    for i, f in enumerate(("radar_eval", "ml_itw")):
        g = cl[cl.fonte == f]
        for j, (rot, chave, cor, hatch) in enumerate(barras):
            v = g["AUC F0"].iloc[0] if chave is None else g[g.atributos == chave]["AUC metadados (fora da dobra)"].iloc[0]
            x = i + (j - 1.5) * w
            ax.bar(x, v - 0.5, w * 0.92, bottom=0.5, color=cor, hatch=hatch, edgecolor=NEUTRO if hatch else "none")
            ax.text(x, max(v, 0.5) + 0.01, br(v, 2), ha="center", fontsize=13)
    ax.axhline(0.7, color=IA_C, ls="--", lw=1.6)
    ax.text(1.02, 0.7, "0,7: acima disso,\n'confundida pelo canal\nou pela duração'", transform=ax.get_yaxis_transform(),
            fontsize=13, color=IA_C, va="center")
    ax.set_xticks([0, 1], ["RADAR eval\n(validação cruzada por clipe)", "ML-ITW\n(validação cruzada por locutor)"])
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color=NEUTRO, lw=1)
    virgula(ax, "y")
    ax.set_ylabel("AUC")
    legenda(fig, [Patch(facecolor=c, hatch=h, edgecolor=NEUTRO if h else "none", label=r) for r, _, c, h in barras],
            y=0.85)
    salvar(fig, "ti_05_atalho", destino)
    print("FIGURAS OK", flush=True)


if __name__ == "__main__":
    main()
