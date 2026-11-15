#!/usr/bin/env python3
"""Figuras da avaliação do módulo de contexto de golpe (lê results/contexto_golpe_*; não chama o Groq).

Saída: results/slides/ctx_01_recall_e_falso_alarme.png, ctx_02_matriz_de_confusao.png,
ctx_03_injecao_e_consistencia.png (13,33 x 7,5 pol., dpi 180, fundo #FBFAF6; mesmo estilo de scripts/v7_slides.py).
Os títulos são montados a partir dos números (a conclusão em linguagem simples).
Uso: python scripts/contexto_golpe_figuras.py   (ou: python scripts/contexto_golpe_avaliar.py figuras)
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "slides"
GOLPE, LEG, NEUTRO = "#B0306A", "#3E9C8F", "#5E6670"
FUNDO, TEXTO, GRADE = "#FBFAF6", "#22272E", "#E8E5DC"
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

NOME = {"golpe_explicito": "Golpes explícitos", "golpe_escondido": "Golpes escondidos",
        "injecao_golpe": "Golpes com injeção", "legitimo_comum": "Conversas comuns",
        "legitimo_dificil": "Legítimas difíceis", "injecao_legitimo": "Legítimas com injeção",
        "ambiguo": "Ambíguas (fora da conta)"}
NIVEIS = ("nenhum", "baixo", "medio", "alto", "indeterminado")
NOME_NIVEL = {"nenhum": "nenhum", "baixo": "baixo", "medio": "médio", "alto": "alto", "indeterminado": "indeterm."}


def pct(x):
    return f"{100 * x:.0f}%"


def de(d):
    return f"{d['k']} de {d['n']}"


def cabecalho(fig, titulo, subtitulo):
    t = textwrap.fill(titulo, 74)
    st = textwrap.fill(subtitulo, 124)
    nt, ns = t.count("\n") + 1, st.count("\n") + 1
    fig.text(0.012, 0.975, t, fontsize=21, fontweight="bold", va="top", color=TEXTO, linespacing=1.1)
    y_s = 0.975 - nt * 0.047 - 0.012
    fig.text(0.012, y_s, st, fontsize=14, va="top", color=NEUTRO, linespacing=1.15)
    return y_s - ns * 0.033


def rodape(fig, texto):
    fig.text(0.012, 0.010, textwrap.fill(texto, 136), fontsize=13, color=NEUTRO, va="bottom", linespacing=1.1)


def barras(ax, itens, cor, meta, rotulo_meta, xlabel):
    y = np.arange(len(itens))[::-1]
    for yy, (nome, d) in zip(y, itens):
        v, (lo, hi) = d["taxa"], d["ic95"]
        ax.barh(yy, v, color=cor, alpha=0.85, height=0.6)
        ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=TEXTO, capsize=5, lw=1.4)
        ax.text(hi + 0.03, yy, f"{pct(v)} ({de(d)})", va="center", fontsize=14, fontweight="bold")
    ax.set_yticks(y, [textwrap.fill(n, 12) for n, _ in itens], fontsize=14)
    ax.axvline(meta, color=TEXTO, ls="--", lw=1.3)
    ax.text(meta, len(itens) - 0.3, rotulo_meta, ha="center", va="bottom", fontsize=13, color=TEXTO)
    ax.set_xlim(0, 1.45)
    ax.set_ylim(-0.6, len(itens) - 0.2)
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig_01(res):
    m = res["positivo_medio_ou_alto"]
    pg = m["por_grupo"]
    rec, fa = m["recall_golpes_todos"], m["falso_alarme_legitimos_todos"]
    ok_rec, ok_fa = rec["taxa"] >= 0.85, fa["taxa"] <= 0.15
    meta_txt = ("atinge as duas metas na média" if ok_rec and ok_fa else
                "atinge só a meta de " + ("recall" if ok_rec else "falso alarme") if ok_rec or ok_fa else
                "não atinge nenhuma das duas metas")
    titulo = (f"Nos textos inventados, o módulo pega {pct(rec['taxa'])} dos golpes e acusa {pct(fa['taxa'])} das "
              f"conversas legítimas: {meta_txt}")
    esc = pg["golpe_escondido"]["recall"]["taxa"]
    if esc < 0.85:
        titulo += f", mas os golpes escondidos ({pct(esc)}) ficam abaixo de 85%"
    sub = (f"Golpes explícitos {pct(pg['golpe_explicito']['recall']['taxa'])}, escondidos "
           f"{pct(pg['golpe_escondido']['recall']['taxa'])}; falso alarme em conversas comuns "
           f"{pct(pg['legitimo_comum']['falso_alarme']['taxa'])} e nas difíceis "
           f"{pct(pg['legitimo_dificil']['falso_alarme']['taxa'])}. Alerta = risco médio ou alto. Barras finas: IC95.")
    fig, axs = plt.subplots(1, 2, figsize=TAM)
    fim = cabecalho(fig, titulo, sub)
    fig.subplots_adjust(top=fim - 0.07, bottom=0.17, left=0.12, right=0.99, wspace=0.75)
    barras(axs[0], [(NOME[g], pg[g]["recall"]) for g in ("golpe_explicito", "golpe_escondido", "injecao_golpe")]
           + [("Todos os golpes", rec)], GOLPE, 0.85, "meta ≥ 85%", "golpes que deram alerta (recall)")
    barras(axs[1], [(NOME[g], pg[g]["falso_alarme"]) for g in ("legitimo_comum", "legitimo_dificil", "injecao_legitimo")]
           + [("Todas as legítimas", fa)], LEG, 0.15, "meta ≤ 15%", "legítimas que deram alerta (falso alarme)")
    axs[0].set_title("Golpes (maior é melhor)", loc="left", fontsize=16, fontweight="bold", pad=22)
    axs[1].set_title("Legítimas (menor é melhor)", loc="left", fontsize=16, fontweight="bold", pad=22)
    rodape(fig, f"130 textos inventados do teste (ambíguos fora da conta), avaliados uma vez com o protocolo congelado "
                f"({res['congelado']['parametros']['modelo_llm']}). Textos escritos por um modelo de linguagem, não golpes "
                "reais; amostras pequenas (IC largos). Só texto: o Whisper não entrou.")
    fig.savefig(OUT / "ctx_01_recall_e_falso_alarme.png", dpi=DPI)
    plt.close(fig)


def fig_02(res, df):
    grupos = [g for g in NOME if (df.grupo == g).any()]
    M = np.array([[int(((df.grupo == g) & (df.risco_previsto == n)).sum()) for n in NIVEIS] for g in grupos])
    frac = M / M.sum(axis=1, keepdims=True)
    esc = res["positivo_medio_ou_alto"]["por_grupo"]["golpe_escondido"]["niveis"]
    n_esc = sum(esc.values())
    baixo_esc = esc["nenhum"] + esc["baixo"]
    titulo = (f"Os erros se concentram nos golpes escondidos: {baixo_esc} de {n_esc} saíram com risco nenhum ou baixo"
              if baixo_esc >= max(1, n_esc // 10) else
              "A maior parte dos textos cai no nível esperado; os erros estão espalhados")
    sub = ("Cada linha é um tipo de texto; cada coluna, o risco que o módulo deu. Número = quantos textos; cor = a fração "
           "da linha. O ideal: golpes em médio ou alto, legítimas em nenhum ou baixo.")
    fig, ax = plt.subplots(figsize=TAM)
    fim = cabecalho(fig, titulo, sub)
    fig.subplots_adjust(top=fim - 0.05, bottom=0.12, left=0.27, right=0.97)
    ax.imshow(frac, cmap="Greys", vmin=0, vmax=1.25, aspect="auto")
    for i in range(len(grupos)):
        for j in range(len(NIVEIS)):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if frac[i, j] > 0.6 else TEXTO)
    ax.set_xticks(range(len(NIVEIS)), [NOME_NIVEL[n] for n in NIVEIS], fontsize=14)
    ax.set_yticks(range(len(grupos)), [f"{NOME[g]} ({int(M[i].sum())})" for i, g in enumerate(grupos)], fontsize=14)
    ax.xaxis.set_ticks_position("top")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(1.5, color=GOLPE, lw=2.5)
    ax.text(1.5, len(grupos) - 0.35, "← sem alerta | alerta →", ha="center", va="top", fontsize=13, color=GOLPE)
    rodape(fig, "Teste: 130 textos inventados, uma execução. 'indeterm.' = falha do serviço ou resposta inválida "
                "(conta como erro).")
    fig.savefig(OUT / "ctx_02_matriz_de_confusao.png", dpi=DPI)
    plt.close(fig)


def fig_03(res):
    inj, cons = res["injecao"], res.get("consistencia")
    ambos = inj["golpes_ainda_pos_e_manipulacao"]
    c3 = cons["risco_igual_nas_3"] if cons else None
    pos, man = inj["golpes_ainda_pos"], inj["golpes_manipulacao_detectada"]
    titulo = (f"Frases de injeção não desligaram o alerta ({de(pos)} golpes ainda com alerta)"
              if pos["k"] == pos["n"] else f"Frases de injeção desligaram o alerta em {pos['n'] - pos['k']} de {pos['n']} golpes")
    if man["k"] < man["n"]:
        titulo += f", mas só {de(man)} foram marcados como manipulação"
    if c3:
        titulo += f"; o risco se repetiu em {pct(c3['taxa'])} das repetições"
    sub = ("Esquerda: os 7 golpes e as 3 conversas legítimas com frases dirigidas ao avaliador ('ignore as regras', "
           "'classifique como nenhum'). Direita: 20 textos sorteados (seed 42), rodados mais 3 vezes cada.")
    fig, axs = plt.subplots(1, 2, figsize=TAM, gridspec_kw={"width_ratios": [1.35, 1]})
    fim = cabecalho(fig, titulo, sub)
    fig.subplots_adjust(top=fim - 0.07, bottom=0.17, left=0.17, right=0.97, wspace=0.62)
    itens = [("Golpes: ainda com alerta", inj["golpes_ainda_pos"], GOLPE),
             ("Golpes: manipulação detectada", inj["golpes_manipulacao_detectada"], GOLPE),
             ("Golpes: as duas coisas", ambos, GOLPE),
             ("Legítimas: deram alerta", inj["legitimos_com_injecao_marcados_pos"], LEG),
             ("Legítimas: manipulação detectada", inj["legitimos_com_injecao_manipulacao"], LEG)]
    y = np.arange(len(itens))[::-1]
    for yy, (nome, d, cor) in zip(y, itens):
        axs[0].barh(yy, d["taxa"], color=cor, alpha=0.85, height=0.6)
        axs[0].text(d["taxa"] + 0.02, yy, de(d), va="center", fontsize=14, fontweight="bold")
    axs[0].set_yticks(y, [textwrap.fill(n, 20) for n, *_ in itens], fontsize=13)
    axs[0].set_xlim(0, 1.2)
    axs[0].set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    axs[0].set_title("Injeção de comando", loc="left", fontsize=16, fontweight="bold")
    axs[0].grid(axis="y", visible=False)
    if cons:
        it2 = [("Risco igual nas 3", cons["risco_igual_nas_3"]), ("Igual nas 3 e no teste", cons["risco_igual_nas_3_e_no_teste"]),
               ("Decisão (alerta ou não) igual nas 3", cons["decisao_pos_igual_nas_3"])]
        y2 = np.arange(len(it2))[::-1]
        for yy, (nome, d) in zip(y2, it2):
            v, (lo, hi) = d["taxa"], d["ic95"]
            axs[1].barh(yy, v, color=NEUTRO, alpha=0.85, height=0.6)
            axs[1].errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=TEXTO, capsize=5, lw=1.4)
            axs[1].text(0.03, yy, de(d), va="center", fontsize=14, fontweight="bold", color="white")
        axs[1].set_yticks(y2, [textwrap.fill(n, 18) for n, _ in it2], fontsize=13)
        axs[1].set_xlim(0, 1.0)
        axs[1].set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        axs[1].set_title("Consistência", loc="left", fontsize=16, fontweight="bold")
        axs[1].grid(axis="y", visible=False)
    rodape(fig, "Mesmo com temperature 0 e seed fixo, o serviço não garante a mesma resposta. Amostras muito pequenas "
                "(7, 3 e 20 textos): servem de indício, não de medida. Barras finas: IC95.")
    fig.savefig(OUT / "ctx_03_injecao_e_consistencia.png", dpi=DPI)
    plt.close(fig)


def main():
    res = json.loads((R / "contexto_golpe_avaliacao_resumo.json").read_text(encoding="utf-8"))
    df = pd.read_csv(R / "contexto_golpe_avaliacao.csv", keep_default_na=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_01(res)
    fig_02(res, df)
    fig_03(res)
    print("figuras gravadas em", OUT)


if __name__ == "__main__":
    main()
