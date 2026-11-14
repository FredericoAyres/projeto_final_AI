#!/usr/bin/env python3
"""Figuras para os slides da história completa (F0 -> prova externa -> v7 -> T3). Só LÊ do disco.

Lê: results/v7_*.csv/.json, results/teste_internet_*.csv/.json, results/v4final_teste_por_banco.csv,
results/v4final_reproducao.csv, results/v7_t3_modelo.json e data/processed/splits_v7.parquet (só para contar
locutores por lado; nada é treinado nem recalculado além de contagens).

Saída: results/slides/f_00_*.png a f_09_*.png (13,33 x 7,5 pol., dpi 180, fundo #FBFAF6, sem moldura) e
results/slides/resumo_v7_para_slides.csv.

    python scripts/v7_slides.py
"""
import json
import os
import sys
import textwrap
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
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

R = Path("results")
OUT = R / "slides"
REAL, IA, NEUTRO = "#2B63C9", "#E0662C", "#5E6670"
FUNDO, TEXTO, GRADE, CLARO = "#FBFAF6", "#22272E", "#E8E5DC", "#B4BAC1"
VERDE = "#2E7D5B"          # só para "resultado OK" no quadro anti-vazamento
TAM, DPI = (13.33, 7.5), 180
_fontes = {f.name for f in font_manager.fontManager.ttflist}
FONTE = "IBM Plex Sans" if "IBM Plex Sans" in _fontes else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONTE, "font.size": 14, "text.color": TEXTO, "axes.labelcolor": TEXTO,
    "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "legend.frameon": False, "figure.facecolor": FUNDO, "axes.facecolor": FUNDO,
    "savefig.facecolor": FUNDO, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA0A6", "xtick.color": "#4A5058", "ytick.color": "#4A5058", "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.7, "axes.axisbelow": True})

# receitas: cinza para o sistema anterior, tons próprios (sem azul nem laranja, que são real e IA)
COR_REC = {"T0": "#7D848C", "T1": "#3E9C8F", "T2": "#8C6BC2", "T3": "#B0306A", "T4": "#A8862E"}
CLARO_REC = {"T0": "#C9CDD2", "T1": "#B5DCD6", "T2": "#D6C8EC", "T3": "#E7B3CB", "T4": "#E3D3A8"}
REC = ("T0", "T1", "T2", "T3", "T4")
DESC_REC = {"T0": "T0 = F0, sistema anterior", "T1": "T1 = + internet real", "T2": "T2 = + IA moderna em canal ruim",
            "T3": "T3 = + os dois (sistema atual)", "T4": "T4 = T3 com peso igual por fonte"}
# rótulos pelo que o banco representa (nome técnico pequeno entre parênteses)
ROT = {
    "ml_itw": ("Internet real: vídeos de redes sociais", "ML-ITW, teste mestre"),
    "radar_en": ("IA comercial moderna em canal degradado, inglês", "RADAR en"),
    "radar_en_pareado": ("A mesma, com durações pareadas", "RADAR en pareado"),
    "in_the_wild": ("Entrevistas e discursos na internet", "In-the-Wild, 38 locutores"),
    "ml_itw_ingles": ("Internet real, só inglês", "ML-ITW inglês; indício"),
    "ml_itw_portugues": ("Internet real, só português", "ML-ITW pt; indício"),
    "llamapartialspoof": ("Estúdio: fala lida com trechos de IA", "LlamaPartialSpoof"),
    "brspeech_df": ("Português de audiolivro × TTS", "BRSpeech-DF"),
    "coraa": ("Português real de entrevistas e aulas", "CORAA, só reais"),
    "asvspoof2019_la": ("Estúdio, geradores antigos", "ASVspoof 2019; secundário"),
    "asvspoof2021_df": ("Estúdio com compressão", "ASVspoof 2021 DF; secundário"),
}


CURTO = {"ml_itw": "Internet real (ML-ITW)", "in_the_wild": "Entrevistas na internet (In-the-Wild)",
         "radar_en": "IA moderna, canal degradado (RADAR en)", "ml_itw_portugues": "Internet real, português (ML-ITW)",
         "llamapartialspoof": "Estúdio, trechos de IA (LlamaPartialSpoof)",
         "brspeech_df": "Português de audiolivro (BRSpeech-DF)", "coraa": "Português real, só reais (CORAA)",
         "asvspoof2019_la": "Estúdio, geradores antigos (ASVspoof 2019)",
         "asvspoof2021_df": "Estúdio com compressão (ASVspoof 2021)"}


def br(v, d=2):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def mil(n):
    return f"{int(n):,}".replace(",", ".")


def rot(k, quebra=True):
    a, b = ROT[k]
    return f"{a}\n({b})" if quebra else f"{a} ({b})"


def cabecalho(fig, titulo, subtitulo):
    """Título (a conclusão) e subtítulo (o que olhar), com quebra de linha; devolve onde o cabeçalho termina."""
    t = textwrap.fill(titulo, 74)
    st = textwrap.fill(subtitulo, 124)
    nt, ns = t.count("\n") + 1, st.count("\n") + 1
    fig.text(0.012, 0.975, t, fontsize=21, fontweight="bold", va="top", color=TEXTO, linespacing=1.1)
    y_s = 0.975 - nt * 0.047 - 0.012
    fig.text(0.012, y_s, st, fontsize=14, va="top", color=NEUTRO, linespacing=1.15)
    return y_s - ns * 0.033


def quadro(titulo, subtitulo, nrows=1, ncols=1, folga=0.05, bottom=0.12, left=0.07, right=0.98, wspace=0.25,
           hspace=0.4, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=TAM, squeeze=False, **kw)
    fim = cabecalho(fig, titulo, subtitulo)
    fig.subplots_adjust(top=fim - folga, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
    return fig, axs


def rodape(fig, texto):
    fig.text(0.012, 0.010, textwrap.fill(texto, 136), fontsize=13, color=NEUTRO, va="bottom", linespacing=1.1)


def salvar(fig, nome):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nome}.png", dpi=DPI, facecolor=FUNDO)
    plt.close(fig)
    print("ok", nome)


# ------------------------------------------------------------------------------------------------ dados
def carregar():
    d = {}
    d["tm"] = pd.read_csv(R / "v7_teste_mestre.csv")
    d["pr"] = pd.read_csv(R / "v7_principal.csv").set_index("receita")
    d["val"] = pd.read_csv(R / "v7_validacao.csv").set_index("receita")
    d["cont"] = pd.read_csv(R / "v7_splits_contagens.csv")
    d["sj"] = json.loads((R / "v7_splits.json").read_text(encoding="utf-8"))
    d["esc"] = json.loads((R / "v7_escolha.json").read_text(encoding="utf-8"))
    d["ti"] = pd.read_csv(R / "teste_internet_por_fonte.csv").set_index("fonte")
    d["v4"] = pd.read_csv(R / "v4final_teste_por_banco.csv")
    d["rep"] = pd.read_csv(R / "v4final_reproducao.csv").set_index("conjunto")
    d["poshoc"] = pd.read_csv(R / "v7_poshoc.csv")
    sp = pd.read_parquet("data/processed/splits_v7.parquet", columns=["bloco", "fonte", "split", "label", "locutor", "idioma"])
    d["sp"] = sp
    return d


def linha(tm, rc, recorte):
    return tm[(tm.receita == rc) & (tm.recorte == recorte)].iloc[0]


def visto(rc, recorte):
    """'sim', 'parcial' ou 'nao': se a receita viu aquele tipo de áudio no treino (pelo protocolo, seção 5)."""
    if recorte.startswith("ml_itw") and rc in ("T1", "T3", "T4"):
        return "sim"
    if recorte.startswith("radar") and rc in ("T2", "T3", "T4"):
        return "parcial"
    return "nao"


# ------------------------------------------------------------------------------------------------ f_00
def f00(d):
    v4 = d["v4"][d["v4"].sistema.str.startswith("F0") & (d["v4"].limiar == "A")].set_index("banco")
    ti = d["ti"]

    def ic(s):
        a, b = s.split(" a ")
        return float(a), float(b)

    itens = [  # (rótulo, nome técnico, AUC, lo, hi, novo?)
        ("Estúdio (a validação do treino)", "validação da v4; sem IC gravado", d["rep"].loc["val", "AUC reajuste"], None, None, False),
        ("Entrevistas e discursos na internet", "In-the-Wild, teste da v4", v4.loc["in_the_wild", "AUC"], *ic(v4.loc["in_the_wild", "AUC IC95"]), False),
        ("Estúdio: fala lida com trechos de IA", "LlamaPartialSpoof", v4.loc["llamapartialspoof", "AUC"], *ic(v4.loc["llamapartialspoof", "AUC IC95"]), False),
        ("Português de audiolivro × TTS", "BRSpeech-DF", v4.loc["brspeech_df", "AUC"], *ic(v4.loc["brspeech_df", "AUC IC95"]), False),
        ("IA comercial moderna em canal degradado", "RADAR eval inteiro; IC otimista", ti.loc["radar_eval", "AUC"],
         ti.loc["radar_eval", "AUC IC2.5"], ti.loc["radar_eval", "AUC IC97.5"], True),
        ("Internet real: vídeos de redes sociais", "ML-ITW inteiro; IC por locutor", ti.loc["ml_itw", "AUC"],
         ti.loc["ml_itw", "AUC IC2.5"], ti.loc["ml_itw", "AUC IC97.5"], True),
    ]
    novos = [x[2] for x in itens if x[5]]
    fig, axs = quadro(f"O sistema anterior (F0) ia muito bem no estúdio e nas entrevistas da internet, "
                      f"mas caiu nos dois bancos novos",
                      f"AUC do F0 (v4) por tipo de áudio, com IC95. Nos bancos novos ficou entre {br(min(novos))} e "
                      f"{br(max(novos))}; 0,5 = cara ou coroa", left=0.33, right=0.95, bottom=0.17, folga=0.08)
    ax = axs[0, 0]
    y = np.arange(len(itens))[::-1]
    for yy, (a, b, v, lo, hi, novo) in zip(y, itens):
        cor = TEXTO if novo else CLARO
        ax.barh(yy, v - 0.5, left=0.5, height=0.62, color=cor)
        if lo is not None:
            ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=NEUTRO if not novo else "#000000",
                        capsize=5, lw=1.6)
        ax.text((hi if hi is not None else v) + 0.008, yy, br(v, 3), va="center", fontsize=15, fontweight="bold",
                color=TEXTO)
    ax.set_yticks(y, [f"{a}\n({b})" for a, b, *_ in itens], fontsize=14)
    for t, it in zip(ax.get_yticklabels(), itens):
        t.set_color(TEXTO if it[5] else NEUTRO)
    ax.set_xlim(0.5, 1.04)
    ax.set_xlabel("AUC (chance de o áudio de IA receber nota maior que o real)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 1)))
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=CLARO, label="bancos já usados no projeto (v4)"),
                        Patch(color=TEXTO, label="bancos novos (avaliados uma vez, 13/11)")],
               loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=13, bbox_transform=ax.transAxes)
    rodape(fig, "Cortes e pesos do F0 sem mudança. Bancos inteiros: RADAR eval 102.725 clipes, ML-ITW 24.529.")
    salvar(fig, "f_00_queda_externa")


# ------------------------------------------------------------------------------------------------ f_01
def halteres(ax, itens, y, cor_a, cor_b, rc_b="T3", casas=2):
    """itens: (recorte, linha_a, linha_b). Ponto A (sistema anterior) e B (receita), com IC95."""
    for yy, (rcorte, a, b) in zip(y, itens):
        ax.plot([a.AUC, b.AUC], [yy, yy], color="#9AA0A6", lw=3, zorder=1, solid_capstyle="round")
        for r_, cor, dy in ((a, cor_a, 0.0), (b, cor_b, 0.0)):
            if np.isfinite(r_["AUC IC2.5"]):
                ax.errorbar(r_.AUC, yy + dy, xerr=[[r_.AUC - r_["AUC IC2.5"]], [r_["AUC IC97.5"] - r_.AUC]], fmt="none",
                            ecolor=NEUTRO, capsize=4, lw=1.3, zorder=2)
        ax.plot(a.AUC, yy, "o", ms=13, color=cor_a, mec="white", mew=1.5, zorder=3)
        v = visto(rc_b, rcorte)
        if v == "nao":
            ax.plot(b.AUC, yy, "o", ms=15, color=cor_b, mec="white", mew=1.5, zorder=4)
        else:
            ax.plot(b.AUC, yy, "o", ms=15, color=CLARO_REC[rc_b], mec=cor_b, mew=2.5, zorder=4)
        esq = a.AUC <= b.AUC
        ax.text(a.AUC + (-1 if esq else 1) * 0.004, yy + 0.30, br(a.AUC, casas), ha="right" if esq else "left",
                va="center", fontsize=13, color=NEUTRO)
        ax.text(b.AUC + (1 if esq else -1) * 0.004, yy + 0.30, br(b.AUC, casas), ha="left" if esq else "right",
                va="center", fontsize=13, color=cor_b, fontweight="bold")


def f01(d):
    tm = d["tm"]
    rec = ["ml_itw", "radar_en", "in_the_wild", "llamapartialspoof", "brspeech_df", "ml_itw_portugues"]
    itens = [(r, linha(tm, "T0", r), linha(tm, "T3", r)) for r in rec]
    pr = d["pr"]
    fig, axs = quadro(f"Com a T3, a AUC média no teste mestre foi de {br(pr.loc['T0', 'média AUC (3 fontes)'])} para "
                      f"{br(pr.loc['T3', 'média AUC (3 fontes)'])}; o que nunca foi visto não piorou",
                      "AUC no teste mestre (IC95), F0 × T3. Ponto cheio = tipo de áudio nunca visto no treino; "
                      "claro com contorno = visto (outros locutores ou outro idioma)", left=0.37, right=0.97, bottom=0.20)
    ax = axs[0, 0]
    y = np.arange(len(rec))[::-1].astype(float)
    halteres(ax, itens, y, COR_REC["T0"], COR_REC["T3"])
    marca = {"sim": "visto: mesmo banco, outros locutores", "parcial": "visto: mesma fonte e geradores, idioma novo",
             "nao": "nunca visto"}
    marca = {"sim": "visto: outros locutores", "parcial": "visto: idioma novo", "nao": "nunca visto"}
    lab = []
    for r in rec:
        v = visto("T3", r)
        extra = "indício; português nunca visto" if r == "ml_itw_portugues" else marca[v]
        lab.append(f"{ROT[r][0]}\n({ROT[r][1].split(';')[0]}; {extra})")
    ax.set_yticks(y, lab, fontsize=13)
    ax.set_xlim(0.40, 1.03)
    ax.axvline(0.5, color=NEUTRO, ls="--", lw=1)
    ax.set_xlabel("AUC no teste mestre")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 1)))
    ax.grid(axis="y", visible=False)
    hs = [Line2D([], [], marker="o", ls="", ms=13, color=COR_REC["T0"], label="F0 (sistema anterior)"),
          Line2D([], [], marker="o", ls="", ms=14, color=COR_REC["T3"], label="T3, tipo nunca visto no treino"),
          Line2D([], [], marker="o", ls="", ms=14, color=CLARO_REC["T3"], mec=COR_REC["T3"], mew=2.5,
                 label="T3, tipo visto no treino")]
    fig.legend(handles=hs, loc="lower right", bbox_to_anchor=(0.985, 0.075), ncol=3, fontsize=13)
    rodape(fig, "O F0 aqui é medido no teste mestre (subconjunto fixo), por isso o ML-ITW dá 0,59 e não 0,69 (banco inteiro). "
                "IC: por locutor (internet), por clipe (RADAR, otimista).")
    salvar(fig, "f_01_antes_depois")


# ------------------------------------------------------------------------------------------------ f_02
def f02(d):
    c = d["cont"]
    sp = d["sp"]
    locs = sp.groupby(["bloco", "split", "label"]).locutor.nunique()

    def cel(bloco, split):
        g = c[(c.bloco == bloco) & (c.split == split)].set_index("classe")
        if g.empty or g.clipes.sum() == 0:
            return None
        n_r, n_f = int(g.loc["real", "clipes"]), int(g.loc["falso", "clipes"])
        if bloco == "radar_eval":
            idi = sorted(sp[(sp.bloco == bloco) & (sp.split == split)].idioma.unique())
            loc = "sem locutor no arquivo; idioma: " + ", ".join(idi)
            if split == "treino":
                loc += f"; usados {mil(g.loc['real', 'clipes_pareados_treino'])} / " \
                       f"{mil(g.loc['falso', 'clipes_pareados_treino'])} (durações pareadas)"
        elif bloco == "estudio":
            loc = f"{locs.get((bloco, split, 0), 0)} / {locs.get((bloco, split, 1), 0)} locutores (o CodecFake+ não tem)"
        else:
            loc = f"{locs.get((bloco, split, 0), 0)} / {locs.get((bloco, split, 1), 0)} locutores"
        return n_r, n_f, textwrap.fill(loc, 33)

    linhas = [("estudio", "Estúdio, inglês\n(os dados da v4)"),
              ("ml_itw", "Internet real:\nvídeos de redes\nsociais (ML-ITW)"),
              ("radar_eval", "IA comercial moderna\nem canal degradado\n(RADAR eval)"),
              ("in_the_wild", "Entrevistas e\ndiscursos na internet\n(In-the-Wild)")]
    lados = [("treino", "TREINO"), ("validacao", "VALIDAÇÃO\n(escolhe C, corte e receita)"),
             ("teste_mestre", "TESTE MESTRE\n(fixo, igual para todas)")]
    fig = plt.figure(figsize=TAM)
    fim = cabecalho(fig, "Cada pessoa fica de um lado só: treino, validação ou teste mestre",
                    "Clipes reais / de IA e locutores por lado (barra: azul = real, laranja = IA). "
                    "Português e In-the-Wild nunca entram no treino")
    ax = fig.add_axes([0.0, 0.11, 1.0, fim - 0.12])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    x0, w = 19.0, 27.0
    for j, (_, t) in enumerate(lados):
        ax.text(x0 + j * w + w / 2, 100, t, ha="center", va="top", fontsize=14, fontweight="bold", linespacing=1.1)
    h = 21.5
    for i, (bloco, nome) in enumerate(linhas):
        yc = 89.5 - i * (h + 0.8)
        ax.text(0.8, yc - h / 2, nome, ha="left", va="center", fontsize=14, linespacing=1.1)
        for j, (sl, _) in enumerate(lados):
            xx = x0 + j * w + 0.5
            v = cel(bloco, sl)
            box = FancyBboxPatch((xx, yc - h), w - 1.0, h - 0.8, boxstyle="round,pad=0,rounding_size=1.0",
                                 fc="white" if v else FUNDO, ec=CLARO, lw=1.2)
            ax.add_patch(box)
            if not v:
                ax.text(xx + (w - 1.0) / 2, yc - h / 2, "não entra", ha="center", va="center", fontsize=14,
                        color=NEUTRO, style="italic")
                continue
            n_r, n_f, loc = v
            tot = n_r + n_f
            bw = w - 3.0
            ax.add_patch(Rectangle((xx + 1.0, yc - 4.2), bw * n_r / tot, 2.4, color=REAL, lw=0))
            ax.add_patch(Rectangle((xx + 1.0 + bw * n_r / tot, yc - 4.2), bw * n_f / tot, 2.4, color=IA, lw=0))
            ax.text(xx + 1.0, yc - 5.6, f"{mil(n_r)} / {mil(n_f)} clipes", ha="left", va="top", fontsize=14,
                    fontweight="bold")
            ax.text(xx + 1.0, yc - 11.0, loc, ha="left", va="top", fontsize=13, color=NEUTRO, linespacing=1.1)
    sj = d["sj"]["salvaguardas"]
    msg = (f"Nenhuma pessoa em dois lados: {sj['locutores_em_comum']} locutores em comum, {sj['caminhos_em_comum']} "
           f"arquivos em comum, {sj['duplicatas_exatas_entre_lados']} áudios idênticos entre lados")
    fig.text(0.5, 0.072, msg, ha="center", fontsize=15, fontweight="bold", color=VERDE)
    rodape(fig, "Fora da conta acima: 41.320 clipes de controle (testes antigos de estúdio e português), "
                "só para medir esquecimento; 13 clipes excluídos (duplicatas).")
    salvar(fig, "f_02_desenho_dos_dados")


# ------------------------------------------------------------------------------------------------ f_03
def barras_receitas(ax, tm, recortes, fs_val=13):
    w = 0.16
    x = np.arange(len(recortes))
    for i, rc in enumerate(REC):
        for k, r in enumerate(recortes):
            g = linha(tm, rc, r)
            v = visto(rc, r)
            pos = x[k] + (i - 2) * w
            cor = COR_REC[rc] if v == "nao" else CLARO_REC[rc]
            ax.bar(pos, g.AUC - 0.4, w * 0.92, bottom=0.4, color=cor, edgecolor=COR_REC[rc] if v != "nao" else "white",
                   hatch="///" if v != "nao" else None, linewidth=1.2)
            if np.isfinite(g["AUC IC2.5"]):
                ax.errorbar(pos, g.AUC, yerr=[[g.AUC - g["AUC IC2.5"]], [g["AUC IC97.5"] - g.AUC]], fmt="none",
                            ecolor="#3A3F45", capsize=2.5, lw=1)
            ax.text(pos, 0.415, br(g.AUC), ha="center", va="bottom", rotation=90, fontsize=fs_val,
                    color="white" if v == "nao" else TEXTO, fontweight="bold" if rc == "T3" else None)
    ax.set_xticks(x, [textwrap.fill(ROT[r][0], 22) + f"\n({ROT[r][1]})" for r in recortes], fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color=NEUTRO, ls="--", lw=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 1)))
    ax.grid(axis="x", visible=False)


def legenda_receitas(fig, y=0.0, extra=True):
    hs = [Patch(facecolor=COR_REC[r], label=DESC_REC[r]) for r in REC]
    if extra:
        hs.append(Patch(facecolor="white", edgecolor=NEUTRO, hatch="///", label="claro/hachurado = visto no treino"))
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, fontsize=13, columnspacing=1.5)


def f03(d):
    tm = d["tm"]
    pr = d["pr"]
    rec = ["ml_itw", "radar_en", "in_the_wild", "llamapartialspoof", "brspeech_df"]
    fig, axs = quadro(f"Juntar os dois bancos novos (T3, T4) é melhor que cada um sozinho; a T3 foi escolhida antes do teste",
                      f"AUC no teste mestre por receita (IC95). Média principal: T0 {br(pr.loc['T0', 'média AUC (3 fontes)'], 3)}, "
                      f"T1 {br(pr.loc['T1', 'média AUC (3 fontes)'], 3)}, T2 {br(pr.loc['T2', 'média AUC (3 fontes)'], 3)}, "
                      f"T3 {br(pr.loc['T3', 'média AUC (3 fontes)'], 3)}, T4 {br(pr.loc['T4', 'média AUC (3 fontes)'], 3)}",
                      left=0.07, right=0.99, bottom=0.36)
    ax = axs[0, 0]
    barras_receitas(ax, tm, rec)
    ax.set_ylabel("AUC no teste mestre")
    ax.axvline(2.5, color=CLARO, lw=1.5)
    ax.text(1.0, 1.005, "média principal (3 fontes)", ha="center", va="bottom", fontsize=13, color=NEUTRO,
            transform=ax.get_xaxis_transform())
    ax.text(3.5, 1.005, "controles (nunca vistos)", ha="center", va="bottom", fontsize=13, color=NEUTRO,
            transform=ax.get_xaxis_transform())
    legenda_receitas(fig, y=0.045)
    rodape(fig, "A receita foi escolhida pela validação (T3: 0,975; T4: 0,973), gravada antes de abrir o teste.")
    salvar(fig, "f_03_receitas")


# ------------------------------------------------------------------------------------------------ f_04
def f04(d):
    tm = d["tm"]
    pares = [("T1", r) for r in ("radar_en", "in_the_wild", "llamapartialspoof", "brspeech_df")] + \
            [("T2", r) for r in ("ml_itw", "in_the_wild", "llamapartialspoof", "brspeech_df")]
    g1 = linha(tm, "T1", "radar_en").AUC - linha(tm, "T0", "radar_en").AUC
    g2 = linha(tm, "T2", "ml_itw").AUC - linha(tm, "T0", "ml_itw").AUC
    fig, axs = quadro("Aprender um banco novo melhorou o outro banco novo, que a receita nunca viu",
                      f"Só tipos de áudio que a receita NÃO viu no treino. Treinar com internet real (T1) sobe a IA em canal "
                      f"degradado em +{br(g1)}; treinar com ela (T2) sobe a internet real em +{br(g2)}",
                      left=0.335, right=0.97, bottom=0.14)
    ax = axs[0, 0]
    y = []
    pos = 0.0
    yt, lab = [], []
    for k, (rc, r) in enumerate(pares):
        if k == 4:
            pos -= 0.8
        a, b = linha(tm, "T0", r), linha(tm, rc, r)
        ax.annotate("", xy=(b.AUC, pos), xytext=(a.AUC, pos),
                    arrowprops=dict(arrowstyle="-|>", color=COR_REC[rc], lw=3, mutation_scale=22))
        for r_, cor in ((a, COR_REC["T0"]), (b, COR_REC[rc])):
            ax.errorbar(r_.AUC, pos, xerr=[[r_.AUC - r_["AUC IC2.5"]], [r_["AUC IC97.5"] - r_.AUC]], fmt="none",
                        ecolor=cor, capsize=4, lw=1.3, alpha=0.8)
        ax.plot(a.AUC, pos, "o", ms=11, color=COR_REC["T0"], mec="white")
        dlt = b.AUC - a.AUC
        ax.text(max(a.AUC, b.AUC) + 0.015, pos, f"{br(a.AUC)} → {br(b.AUC)}  ({'+' if dlt >= 0 else ''}{br(dlt)})",
                va="center", fontsize=13, fontweight="bold" if r in ("radar_en", "ml_itw") else None, color=TEXTO)
        yt.append(pos)
        lab.append(CURTO[r])
        y.append(pos)
        pos -= 1.0
    ax.set_yticks(yt, lab, fontsize=13)
    for t, (rc, r) in zip(ax.get_yticklabels(), pares):
        t.set_color(COR_REC[rc])
        if r in ("radar_en", "ml_itw"):
            t.set_fontweight("bold")
    ax.set_xlim(0.5, 1.17)
    ax.set_xlabel("AUC no teste mestre (seta: do F0 para a receita)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 1)))
    ax.grid(axis="y", visible=False)
    ax.text(0.505, yt[0] + 0.55, "T1 = estúdio + internet real (não viu a IA em canal degradado)", fontsize=13,
            color=COR_REC["T1"], fontweight="bold")
    ax.text(0.505, yt[4] + 0.55, "T2 = estúdio + IA em canal degradado (não viu a internet real)", fontsize=13,
            color=COR_REC["T2"], fontweight="bold")
    ax.set_ylim(yt[-1] - 0.6, yt[0] + 0.9)
    rodape(fig, "Por que um ajuda o outro (canal degradado? geradores modernos em comum?) é hipótese, não testada. "
                "IC por clipe no RADAR (otimista).")
    salvar(fig, "f_04_sinal_de_generalizacao")


# ------------------------------------------------------------------------------------------------ f_05
def f05(d):
    tm = d["tm"]
    rec = ["llamapartialspoof", "brspeech_df", "asvspoof2019_la", "asvspoof2021_df"]
    dl = {r: linha(tm, "T3", r).AUC - linha(tm, "T0", r).AUC for r in rec}
    pior = min(dl, key=dl.get)
    co0, co3 = linha(tm, "T0", "coraa"), linha(tm, "T3", "coraa")
    f0 = co0[f"FPR {co0['regra escolhida (validação)']}"]
    f3 = co3[f"FPR {co3['regra escolhida (validação)']}"]
    fig, axs = quadro(f"O estúdio não piorou: a maior queda de AUC da T3 foi {br(dl[pior], 3)}",
                      "Testes antigos de estúdio (controles de esquecimento), F0 × T3. À direita: % de reais do "
                      "português (CORAA) acusados como IA, na regra de corte escolhida", ncols=2, left=0.30, right=0.98,
                      bottom=0.20, wspace=0.35, gridspec_kw={"width_ratios": [3.0, 1.0]})
    ax, ax2 = axs[0]
    y = np.arange(len(rec))[::-1].astype(float)
    itens = [(r, linha(tm, "T0", r), linha(tm, "T3", r)) for r in rec]
    halteres(ax, itens, y, COR_REC["T0"], COR_REC["T3"], casas=3)
    ax.set_yticks(y, [rot(r) for r in rec], fontsize=13)
    ax.set_xlim(0.80, 1.02)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("AUC")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 2)))
    ax.grid(axis="y", visible=False)
    ax2.bar([0, 1], [100 * f0, 100 * f3], 0.6, color=[COR_REC["T0"], COR_REC["T3"]])
    for i, (v, rg) in enumerate(((f0, co0["regra escolhida (validação)"]), (f3, co3["regra escolhida (validação)"]))):
        ax2.text(i, 100 * v + 0.4, f"{br(100 * v, 1)}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax2.set_xticks([0, 1], [f"F0\n(regra {co0['regra escolhida (validação)']})",
                            f"T3\n(regra {co3['regra escolhida (validação)']})"])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 0)))
    ax2.set_ylim(0, 100 * max(f0, f3) * 1.3)
    ax2.set_ylabel("% de reais acusados")
    ax2.set_title("Português real\n(CORAA, só reais)", fontsize=14)
    ax2.grid(axis="x", visible=False)
    hs = [Line2D([], [], marker="o", ls="", ms=13, color=COR_REC["T0"], label="F0 (sistema anterior)"),
          Line2D([], [], marker="o", ls="", ms=14, color=COR_REC["T3"], label="T3 (sistema atual); nenhum destes foi visto no treino")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.30, 0.075), ncol=2, fontsize=13)
    rodape(fig, "ASVspoof: secundários (podem ter locutores em comum com o treino). IC por clipe. "
                "O BRSpeech-DF sobe em AUC, mas a calibração dele piora (figura f_07).")
    salvar(fig, "f_05_estudio_nao_piorou")


# ------------------------------------------------------------------------------------------------ f_06
def f06(d):
    tm = d["tm"]
    rec = ["radar_en", "radar_en_pareado", "ml_itw_ingles", "in_the_wild", "llamapartialspoof", "asvspoof2019_la",
           "asvspoof2021_df"]
    M = np.array([[linha(tm, rc, r).AUC for rc in REC] for r in rec])
    li = linha(tm, "T3", "ml_itw_ingles")
    fig, axs = quadro(f"Em inglês, a T3 sobe onde o F0 falhava e não perde onde ele já ia bem",
                      "AUC por receita no bloco inglês. Célula hachurada = tipo de áudio visto no treino daquela receita; "
                      "cor mais escura = AUC maior", left=0.355, right=0.97, bottom=0.10, folga=0.10)
    ax = axs[0, 0]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("n", ["#FFFFFF", "#7D848C", "#22272E"])
    ax.imshow(np.clip((M - 0.55) / 0.45, 0, 1), cmap=cmap, aspect="auto", vmin=0, vmax=1.15)
    for i, r in enumerate(rec):
        for j, rc in enumerate(REC):
            v = M[i, j]
            if visto(rc, r) != "nao":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//",
                                       ec=(1, 1, 1, 0.30) if v > 0.85 else (0.37, 0.4, 0.44, 0.35), lw=0))
            escuro = (v - 0.55) / 0.45 > 0.55
            ax.text(j, i, br(v, 3), ha="center", va="center", fontsize=15, color="white" if escuro else TEXTO,
                    fontweight="bold" if rc == "T3" else None,
                    bbox=dict(boxstyle="round,pad=0.25", fc=(0.2, 0.22, 0.25, 0.85) if escuro else (1, 1, 1, 0.85),
                              ec="none") if visto(rc, r) != "nao" else None)
    cab = {"T0": "T0\n(F0, anterior)", "T1": "T1\n(+ internet real)", "T2": "T2\n(+ IA canal ruim)",
           "T3": "T3\n(os dois; atual)", "T4": "T4\n(T3, peso igual)"}
    ax.set_xticks(range(5), [cab[r] for r in REC], fontsize=13)
    ax.xaxis.tick_top()
    labs = []
    for r in rec:
        a, b = ROT[r]
        if r == "ml_itw_ingles":
            b = f"ML-ITW inglês; indício: {int(li.locutores_real)} / {int(li.locutores_falso)} locutores"
        labs.append(f"{a}\n({b})")
    ax.set_yticks(range(len(rec)), labs, fontsize=13)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(4.5, color=FUNDO, lw=6)
    ax.add_patch(Rectangle((2.5, -0.5), 1, len(rec), fill=False, ec=COR_REC["T3"], lw=3))
    rodape(fig, "Linhas de baixo: controles de estúdio (ASVspoof secundários). Pareado = mesmo nº de reais e de IA por faixa de "
                "duração: a duração não explica o resultado.")
    salvar(fig, "f_06_ingles")


# ------------------------------------------------------------------------------------------------ f_07
def f07(d):
    tm = d["tm"]
    esc = d["esc"]["por_receita"]["T3"]
    cortes = esc["cortes"]
    regra = esc["regra"]
    rec = ["ml_itw", "in_the_wild", "radar_en", "ml_itw_portugues", "llamapartialspoof", "brspeech_df", "coraa",
           "asvspoof2019_la", "asvspoof2021_df"]
    pt = {"ml_itw_portugues", "brspeech_df", "coraa"}
    br_ = linha(tm, "T3", "brspeech_df")
    fig, axs = quadro(f"A escala da nota muda entre ambientes: com o mesmo corte, a T3 acusa de "
                      f"{br(100 * min(linha(tm, 'T3', r)['FPR ' + regra] for r in rec), 0)}% a "
                      f"{br(100 * br_['FPR ' + regra], 0)}% dos reais",
                      "Esquerda: nota média de reais e de IA por fonte (T3) e os três cortes da validação. "
                      "Direita: % de reais acusados (FPR) e de IA detectada (TPR) em cada regra", ncols=2,
                      left=0.335, right=0.985, bottom=0.25, wspace=0.06, sharey=True,
                      gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax, ax2 = axs[0]
    y = np.arange(len(rec))[::-1].astype(float)
    for yy, r in zip(y, rec):
        g = linha(tm, "T3", r)
        if r in pt:
            for a_ in (ax, ax2):
                a_.axhspan(yy - 0.5, yy + 0.5, color="#F1EDE2", zorder=0, lw=0)
        mr, mf = g["nota média real"], g["nota média falso"]
        if np.isfinite(mf):
            ax.plot([mr, mf], [yy, yy], color=CLARO, lw=2.5, zorder=1)
            ax.plot(mf, yy, "o", ms=12, color=IA, zorder=3)
        ax.plot(mr, yy, "o", ms=12, color=REAL, zorder=3)
    est = {"A'": ":", "M": "-", "F5": "--"}
    for k, v in cortes.items():
        ax.axvline(v, color=TEXTO, ls=est[k], lw=1.6 if k == regra else 1.1)
    c_a, c_m, c_f = (br(cortes[k], 2) for k in ("A'", "M", "F5"))
    ax.text(0.02, 0.995, f"cortes da validação: A' {c_a} · M {c_m} · F5 {c_f}", ha="left", va="top", fontsize=13,
            transform=ax.transAxes, bbox=dict(fc=FUNDO, ec="none", pad=2))
    ax.set_yticks(y, [CURTO[r] for r in rec], fontsize=13)
    for t, r in zip(ax.get_yticklabels(), rec):
        if r in pt:
            t.set_fontweight("bold")
    ax.set_xlabel("nota média (logit; maior = mais IA)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(y[-1] - 0.6, y[0] + 1.6)
    mk = {"A'": "s", "M": "o", "F5": "^"}
    off = {"A'": -0.22, "M": 0.0, "F5": 0.22}
    for yy, r in zip(y, rec):
        g = linha(tm, "T3", r)
        for k in ("A'", "M", "F5"):
            ax2.plot(100 * g[f"FPR {k}"], yy + off[k], mk[k], ms=10 if k == regra else 8, color=REAL,
                     mec="black" if k == regra else REAL)
            if np.isfinite(g[f"TPR {k}"]):
                ax2.plot(100 * g[f"TPR {k}"], yy + off[k], mk[k], ms=10 if k == regra else 8, color=IA,
                         mec="black" if k == regra else IA)
        ax2.text(108, yy, f"{br(100 * g['FPR ' + regra], 0)}%", va="center", fontsize=13, color=REAL,
                 fontweight="bold" if r in pt else None)
    ax2.set_xlim(-2, 124)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_xlabel("% (à direita: reais acusados, regra M)")
    ax2.grid(axis="y", visible=False)
    hs = [Line2D([], [], marker="o", ls="", color=REAL, ms=11), Line2D([], [], marker="o", ls="", color=IA, ms=11),
          (Line2D([], [], color=TEXTO, ls=":"), Line2D([], [], marker="s", ls="", color=NEUTRO)),
          (Line2D([], [], color=TEXTO, ls="-", lw=1.6), Line2D([], [], marker="o", ls="", color=NEUTRO, mec="black")),
          (Line2D([], [], color=TEXTO, ls="--"), Line2D([], [], marker="^", ls="", color=NEUTRO))]
    nomes = ["real", "IA", "A' = EER da validação agrupada", "M = mediana dos EER por fonte (marcada pelo protocolo)",
             "F5 = no máximo 5% de reais acusados"]
    fig.legend(hs, nomes, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, fontsize=13, columnspacing=1.6,
               handlelength=2.2)
    rodape(fig, "Faixa bege = português. A regra do sistema final ainda NÃO foi escolhida: será decidida só pela validação.")
    salvar(fig, "f_07_cortes_e_calibracao")


# ------------------------------------------------------------------------------------------------ f_08
def f08(d):
    sj = d["sj"]["salvaguardas"]
    n_exc = len(sj["duplicatas_exatas_excluidas_da_avaliacao"])
    regras = pd.read_csv(R / "v7_splits_mlitw_regras.csv")
    lados_3 = regras[regras.regra.str.startswith("(3)")].groupby("split").size().to_dict()
    linhas = [
        ("Mesmo arquivo em dois lados", "o caminho de cada clipe aparece em um lado só",
         f"{sj['caminhos_em_comum']} em comum", True),
        ("Mesma pessoa em dois lados", "internet real e entrevistas: todo o áudio de um locutor fica do mesmo lado; "
         "nomes iguais entre os dois bancos = mesma pessoa",
         f"{sj['locutores_em_comum']} em comum; {len(sj['coincidentes'])} pessoas nos dois bancos, postas no lado delas "
         f"({lados_3.get('validacao', 0)} validação, {lados_3.get('teste_mestre', 0)} teste)", True),
        ("Áudio idêntico em dois lados", "vetor exatamente igual (hash) entre treino, validação e teste",
         f"PAROU a 1ª execução: 12 grupos de reais de estúdio (CodecFake+), herança da v4; {n_exc} clipes da "
         f"validação excluídos → {sj['duplicatas_exatas_entre_lados']}", None),
        ("Áudio quase idêntico", "cosseno ≥ 0,9995 entre treino/validação e teste mestre (internet real e entrevistas)",
         f"{len(sj['quase_duplicatas_pares'])} pares; máximo {br(sj['cosseno_maximo_entre_lados'], 3)}", True),
        ("Banco sem locutor no arquivo", "a IA em canal degradado (RADAR) foi dividida por idioma; o dev do RADAR "
         "nunca é usado", "treino zh, zh-tw, ja · validação en-sg, vi · teste en; dev: 0 clipes", True),
        ("Regras antes dos números", "protocolo escrito e congelado antes do treino; escolha gravada antes do teste",
         "congelado 19:36 · escolha 19:41:00 · teste avaliado 1 vez", True),
    ]
    fig = plt.figure(figsize=TAM)
    fim = cabecalho(fig, "Seis travas automáticas contra vazamento; uma delas parou a execução e o problema foi corrigido",
                    "Cada trava é um teste no código: se falha, a divisão para e grava um aviso de bloqueio")
    ax = fig.add_axes([0.0, 0.07, 1.0, fim - 0.08])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    cols = [(1.2, "trava"), (25, "o que verifica"), (62, "resultado")]
    for x, t in cols:
        ax.text(x, 98, t.upper(), fontsize=13, fontweight="bold", color=NEUTRO, va="top")
    hh = 15.2
    for i, (a, b, c, ok) in enumerate(linhas):
        yt = 92 - i * hh
        ax.add_patch(FancyBboxPatch((0.6, yt - hh + 1.2), 98.8, hh - 1.6, boxstyle="round,pad=0,rounding_size=1",
                                    fc="white", ec=CLARO, lw=1))
        ax.text(1.8, yt - hh / 2 + 0.6, textwrap.fill(a, 24), fontsize=14, fontweight="bold", va="center")
        ax.text(25, yt - hh / 2 + 0.6, textwrap.fill(b, 46), fontsize=13, va="center", color=TEXTO)
        cor = VERDE if ok else IA
        ax.text(62, yt - hh / 2 + 0.6, textwrap.fill(c, 43), fontsize=13, va="center", color=cor, fontweight="bold")
    rodape(fig, "Limite: o cruzamento de pessoas entre bancos foi feito pelo nome (pode haver a mesma pessoa com nomes diferentes). "
                "O teste mestre não é virgem.")
    salvar(fig, "f_08_anti_vazamento")


# ------------------------------------------------------------------------------------------------ f_09
def f09(d):
    tm = d["tm"]
    pr = d["pr"]
    li = linha(tm, "T3", "ml_itw_ingles")
    lp = linha(tm, "T3", "ml_itw_portugues")
    mostra = [
        f"No teste mestre fixo, a AUC média foi de {br(pr.loc['T0', 'média AUC (3 fontes)'])} (F0) para "
        f"{br(pr.loc['T3', 'média AUC (3 fontes)'])} (T3), com outras pessoas e outro idioma",
        "A receita T3 foi escolhida pela validação, antes de abrir o teste",
        "O que nunca foi visto (entrevistas na internet, estúdio) não piorou",
        "Treinar com um banco novo melhorou o outro banco novo, que a receita não viu",
        "Os ganhos não são explicados por duração, volume ou silêncio: um classificador só com esses dados fica abaixo de 0,7",
        "Na internet real, o ganho se mantém dentro do mesmo locutor (análise feita depois)",
    ]
    nao = [
        "Que o sistema pega geradores novos: no RADAR, o idioma é novo, mas geradores e canal são os do treino",
        "Uma prova virgem: o F0 já tinha sido medido nesses clipes, e as entrevistas da internet foram vistas da v4 à v6",
        f"Uma estimativa firme em inglês da internet real: só {int(li.locutores_real)} / {int(li.locutores_falso)} "
        f"locutores (real / IA)",
        f"Nada firme em português: {int(lp.locutores_real)} / {int(lp.locutores_falso)} locutores na internet; "
        f"o outro banco em português é audiolivro limpo",
        "Que um corte fixo serve em todo lugar: a escala da nota muda entre ambientes",
        "Nada sobre ligações ou WhatsApp reais: nenhum banco é ligação de verdade",
        "Por que um banco ajuda o outro (canal degradado ou geradores modernos): hipótese",
    ]
    fig = plt.figure(figsize=TAM)
    cabecalho(fig, "O que o teste mostra, e o que ele ainda não mostra", "Leia os números da v7 com estas ressalvas ao lado")
    for x0, titulo, itens, cor, simb in ((0.012, "O TESTE MOSTRA", mostra, VERDE, "✓"),
                                         (0.51, "O TESTE NÃO MOSTRA", nao, IA, "✗")):
        fig.text(x0, 0.85, titulo, fontsize=16, fontweight="bold", color=cor, va="top")
        y = 0.79
        for it in itens:
            txt = textwrap.fill(it, 56)
            fig.text(x0, y, simb, fontsize=16, color=cor, va="top", fontweight="bold")
            fig.text(x0 + 0.025, y, txt, fontsize=14, va="top", color=TEXTO, linespacing=1.15)
            y -= 0.036 * (txt.count("\n") + 1) + 0.022
    fig.add_artist(Line2D([0.495, 0.495], [0.07, 0.86], color=CLARO, lw=1.5))
    rodape(fig, "Detalhes: docs/RESULTADO_V7.md (seções 6 e 7) e docs/PROTOCOLO_V7.md (seção 0).")
    salvar(fig, "f_09_o_que_nao_afirmamos")


# ------------------------------------------------------------------------------------------------ resumo CSV
def resumo_csv(d):
    tm = d["tm"]
    sp = d["sp"]
    locs = {}
    m_tm = sp.split == "teste_mestre"
    locs["ml_itw"] = sp[m_tm & (sp.bloco == "ml_itw")]
    locs["in_the_wild"] = sp[m_tm & (sp.bloco == "in_the_wild")]

    def nloc(r):
        g = linha(tm, "T3", r)
        if np.isfinite(g.locutores_real):
            return f"{int(g.locutores_real)} reais / {int(g.locutores_falso)} IA"
        if r.startswith("radar"):
            return "sem locutor no arquivo"
        f = sp[(sp.split == "controle") & (sp.fonte == r)]
        lr, lf = f[f.label == 0].locutor.nunique(), f[f.label == 1].locutor.nunique()
        if lr == 0 and lf == 0:
            return "sem locutor no arquivo"
        return f"{lr} reais / {lf} IA" if lf else f"{lr} reais"

    vis = {"sim": "sim", "parcial": "parcial", "nao": "nao"}
    obs = {
        "ml_itw": "teste mestre (locutores fora do treino); F0 no banco inteiro: 0,689",
        "radar_en": "idioma novo, mas mesma fonte, geradores e canal do treino; IC por clipe (otimista); F0 no banco inteiro: 0,721",
        "radar_en_pareado": "mesmo nº de reais e IA por faixa de duração; igual ao natural",
        "in_the_wild": "nunca entra no treino; amostra de 10.000 clipes da v4",
        "ml_itw_ingles": "indício (< 30 locutores por classe)",
        "ml_itw_portugues": "indício; banco visto, mas português nunca entrou no treino",
        "llamapartialspoof": "controle de esquecimento",
        "brspeech_df": "controle; AUC sobe, mas {fpr}% dos reais acusados na regra M (T3)",
        "coraa": "só reais: reais acusados {f0}% (F0, regra A') → {f3}% (T3, regra M)",
        "asvspoof2019_la": "secundário: pode ter locutores em comum com o treino",
        "asvspoof2021_df": "secundário: pode ter locutores em comum com o treino",
    }
    rows = []
    for r in ["ml_itw", "radar_en", "radar_en_pareado", "in_the_wild", "ml_itw_ingles", "ml_itw_portugues",
              "llamapartialspoof", "brspeech_df", "coraa", "asvspoof2019_la", "asvspoof2021_df"]:
        a, b = linha(tm, "T0", r), linha(tm, "T3", r)
        o = obs[r]
        if r == "brspeech_df":
            o = o.format(fpr=br(100 * b["FPR M"], 1))
        if r == "coraa":
            o = o.format(f0=br(100 * a["FPR A'"], 1), f3=br(100 * b["FPR M"], 1))
        v = visto("T3", r)
        if r == "ml_itw_portugues":
            v = "parcial"
        rows.append({"o_que_representa": ROT[r][0], "fonte_tecnica": ROT[r][1],
                     "auc_f0": round(a.AUC, 3) if np.isfinite(a.AUC) else None,
                     "auc_t3": round(b.AUC, 3) if np.isfinite(b.AUC) else None,
                     "visto_no_treino": vis[v], "n_clipes": int(a.n_real + a.n_falso), "n_locutores": nloc(r),
                     "observacao": o})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "resumo_v7_para_slides.csv", index=False, encoding="utf-8")
    print(t.to_string())


def main():
    d = carregar()
    for f in (f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, resumo_csv):
        f(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
