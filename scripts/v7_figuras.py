"""Figuras da v7 (results/slides/v7_*.png): 13,33 x 7,5 pol., dpi 180; real azul #2B63C9, IA laranja #E0662C.

Lê só as tabelas da etapa 'avaliar' (results/v7_*.csv) e as notas (data/processed/v7_notas.parquet).
O título de cada figura é a conclusão, montada a partir dos números (sem nome de pessoa nem de máquina).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("results")
OUT = R / "slides"
REAL, IA = "#2B63C9", "#E0662C"
COR = {"T0": "#7a7974", "T1": "#1baf7a", "T2": "#eda100", "T3": "#e87ba4", "T4": "#7c4dc4"}
REC = ("T0", "T1", "T2", "T3", "T4")
NOME = {"ml_itw": "ML-ITW (teste)", "in_the_wild": "In-the-Wild (38 loc.)", "radar_en": "RADAR en",
        "radar_en_pareado": "RADAR en (pareado)", "ml_itw_ingles": "ML-ITW só inglês",
        "ml_itw_portugues": "ML-ITW português", "llamapartialspoof": "LlamaPartialSpoof",
        "brspeech_df": "BRSpeech-DF (pt-BR)", "coraa": "CORAA (só reais)", "asvspoof2019_la": "ASVspoof 2019*",
        "asvspoof2021_df": "ASVspoof 2021 DF*"}
DESCR = {"T0": "T0: F0 sem retreinar", "T1": "T1: + ML-ITW", "T2": "T2: + RADAR", "T3": "T3: + ML-ITW + RADAR",
         "T4": "T4: T3 com peso igual por fonte"}


def fmt(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def estilo():
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                         "axes.axisbelow": True, "axes.titlesize": 14, "figure.facecolor": "white"})


def salvar(fig, nome, sfx):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nome}{sfx}.png", dpi=180)
    plt.close(fig)


def barras_auc(ax, tab, recortes, hachura_visto=True):
    w = 0.16
    x = np.arange(len(recortes))
    for i, rc in enumerate(REC):
        g = tab[tab.receita == rc].set_index("recorte").reindex(recortes)
        lo = (g.AUC - g["AUC IC2.5"]).clip(lower=0).fillna(0)
        hi = (g["AUC IC97.5"] - g.AUC).clip(lower=0).fillna(0)
        pos = x + (i - 2) * w
        bars = ax.bar(pos, g.AUC, w * 0.9, color=COR[rc], label=DESCR[rc], yerr=[lo, hi],
                      error_kw={"lw": 1.2, "capsize": 3, "ecolor": "#3d3d3a"}, edgecolor="white", linewidth=1)
        if hachura_visto:
            for b, dom in zip(bars, g["domínio"].fillna("")):
                if dom.startswith("visto") or dom.startswith("idioma novo"):
                    b.set_hatch("//")
                    b.set_edgecolor("white")
        for p, v in zip(pos, g.AUC):
            if np.isfinite(v):
                ax.text(p, 0.415, fmt(v), ha="center", va="bottom", fontsize=8.5, rotation=90, color="#1f1f1d")
    ax.axhline(0.5, color="#52514e", ls="--", lw=1)
    ax.set_xticks(x, [NOME[r] for r in recortes])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC (IC95)")


def main(quick=False):
    sfx = "_quick" if quick else ""
    estilo()
    tab = pd.read_csv(R / f"v7_teste_mestre{sfx}.csv")
    pr = pd.read_csv(R / f"v7_principal{sfx}.csv")
    esc = json.loads((R / f"v7_escolha{sfx}.json").read_text(encoding="utf-8"))
    venc = esc["receita_vencedora_pela_validacao"]
    t0 = pr.set_index("receita").loc["T0"]
    best = pr.set_index("receita")

    # (1) AUC por receita e fonte, visto x não visto
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    princ = ["ml_itw", "in_the_wild", "radar_en"]
    barras_auc(ax, tab, princ)
    sub = {s: float((best[f"AUC {s}"] - t0[f"AUC {s}"]).max()) for s in princ}      # maior ganho sobre o T0
    maior = max(sub, key=sub.get)
    titulo = (f"Treinar com dados da internet muda pouco: o maior ganho sobre o T0 é {fmt(sub[maior], 3)} ({NOME[maior]})"
              if sub[maior] < 0.05 else
              f"A AUC sobe onde o domínio foi visto (ML-ITW +{fmt(sub['ml_itw'], 2)}, RADAR en +{fmt(sub['radar_en'], 2)}); "
              f"no In-the-Wild, nunca visto, +{fmt(sub['in_the_wild'], 3)} no máximo")
    ax.set_title(titulo, loc="left", fontsize=15, fontweight="bold")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="#bdbcb6", hatch="//", edgecolor="white"))
    l.append("hachurado: domínio visto no treino (outros locutores / outro idioma)")
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=10, frameon=False, ncol=3)
    fig.text(0.01, 0.01, "Teste mestre fixo (igual para todas as receitas). IC95 por bootstrap: por locutor no ML-ITW e "
             "no In-the-Wild; por clipe no RADAR (otimista). In-the-Wild nunca entra no treino.", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    salvar(fig, "v7_01_auc_por_fonte", sfx)

    # (2) média principal por receita
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(REC))
    g = pr.set_index("receita").loc[list(REC)]
    ax.bar(x, g["média AUC (3 fontes)"], 0.55, color=[COR[r] for r in REC], edgecolor="white",
           yerr=[g["média AUC (3 fontes)"] - g["média IC2.5"], g["média IC97.5"] - g["média AUC (3 fontes)"]],
           error_kw={"lw": 1.4, "capsize": 5}, label="média da AUC (3 fontes)")
    ax.errorbar(x + 0.33, g["AUC pior fonte"], yerr=[g["AUC pior fonte"] - g["pior IC2.5"], g["pior IC97.5"] - g["AUC pior fonte"]],
                fmt="D", color="#1f1f1d", ms=8, capsize=4, label="pior fonte")
    for i, rc in enumerate(REC):
        ax.text(i - 0.26, g.loc[rc, "média AUC (3 fontes)"] + 0.005, fmt(g.loc[rc, "média AUC (3 fontes)"], 3), ha="left",
                va="bottom", fontsize=12)
        ax.text(i + 0.40, g.loc[rc, "AUC pior fonte"], f"{fmt(g.loc[rc, 'AUC pior fonte'], 3)}\n{NOME[g.loc[rc, 'pior fonte']]}",
                ha="left", va="center", fontsize=8.5)
    ax.set_xticks(x, [DESCR[r].replace(": ", ":\n") + ("\n(vencedora pela validação)" if r == venc else "") for r in REC],
                  fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="#52514e", ls="--", lw=1)
    ax.set_ylabel("AUC no teste mestre")
    d = g.loc[venc, "média AUC (3 fontes)"] - g.loc["T0", "média AUC (3 fontes)"]
    ax.set_title(f"A receita escolhida pela validação ({venc}) fica em {fmt(g.loc[venc, 'média AUC (3 fontes)'], 3)} de AUC média "
                 f"({'+' if d >= 0 else ''}{fmt(d, 3)} sobre o T0)", loc="left", fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_xlim(-0.5, 4.95)
    fig.text(0.01, 0.01, "Média da AUC por fonte em ML-ITW teste, In-the-Wild (38) e RADAR en; IC95 por bootstrap (mesmos sorteios em todas as receitas).",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    salvar(fig, "v7_02_media_principal", sfx)

    # (3) esquecimento nos controles
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [3.2, 1]})
    ctr = ["llamapartialspoof", "brspeech_df", "asvspoof2019_la", "asvspoof2021_df"]
    barras_auc(ax, tab, ctr, hachura_visto=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=False, ncol=3)
    co = tab[tab.recorte == "coraa"].set_index("receita").loc[list(REC)]
    fpr = [co.loc[r, f"FPR {co.loc[r, 'regra escolhida (validação)']}"] for r in REC]
    ax2.bar(np.arange(5), np.array(fpr) * 100, 0.6, color=[COR[r] for r in REC], edgecolor="white")
    for i, v in enumerate(fpr):
        ax2.text(i, v * 100 + 1, fmt(v * 100, 1), ha="center", fontsize=10)
    ax2.set_ylim(0, max(fpr) * 100 * 1.25)
    ax2.set_xticks(np.arange(5), REC)
    ax2.set_ylabel("% de reais acusados (regra de corte escolhida)")
    ax2.set_title("CORAA (pt-BR, só reais)", fontsize=12)
    da = {c: (tab[(tab.recorte == c)].set_index("receita").AUC.loc[["T1", "T2", "T3", "T4"]] -
              tab[(tab.recorte == c) & (tab.receita == "T0")].AUC.iloc[0]).min() for c in ctr}
    pior = min(da, key=da.get)
    bra = tab[tab.recorte == "brspeech_df"].set_index("receita").AUC
    fig.suptitle(f"Esquecimento nos testes antigos de estúdio: a maior queda de AUC é {fmt(da[pior], 3)} ({NOME[pior]})"
                 if da[pior] < -0.01 else
                 f"Sem esquecimento: a maior queda de AUC é {fmt(da[pior], 3)}, e o BRSpeech-DF sobe de "
                 f"{fmt(bra['T0'], 2)} para {fmt(bra['T3'], 2)} (T3)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.01, "* ASVspoof: secundários (podem ter locutores em comum com o treino via VCTK). IC95 por clipe.",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    salvar(fig, "v7_03_esquecimento_controles", sfx)

    # (4) regras de corte: TPR e FPR
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5), sharey=True)
    mk = {"A'": "o", "M": "s", "F5": "^"}
    for ax, s in zip(axes, princ):
        for i, rc in enumerate(REC):
            r_ = tab[(tab.receita == rc) & (tab.recorte == s)].iloc[0]
            for j, rg in enumerate(("A'", "M", "F5")):
                xx = i + (j - 1) * 0.22
                esc_ = r_["regra escolhida (validação)"] == rg
                ax.plot(xx, 100 * r_[f"TPR {rg}"], mk[rg], color=IA, ms=9 if esc_ else 6, mec="black" if esc_ else IA)
                ax.plot(xx, 100 * r_[f"FPR {rg}"], mk[rg], color=REAL, ms=9 if esc_ else 6, mec="black" if esc_ else REAL)
        ax.set_xticks(range(5), REC)
        ax.set_title(NOME[s], fontsize=12)
        ax.set_ylim(-2, 102)
    axes[0].set_ylabel("%")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=IA, marker="o", ls="", label="TPR: falsos detectados"),
          Line2D([], [], color=REAL, marker="o", ls="", label="FPR: reais acusados")] + \
         [Line2D([], [], color="#52514e", marker=m, ls="", label=f"regra {k}") for k, m in mk.items()] + \
         [Line2D([], [], color="white", marker="o", mec="black", ms=9, ls="", label="contorno: regra escolhida na validação")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=6, frameon=False, fontsize=10)
    v = tab[(tab.receita == venc) & tab.recorte.isin(princ)]
    rv = v["regra escolhida (validação)"].iloc[0]
    fig.suptitle(f"Na receita {venc}, a regra {rv} pega de {fmt(100 * v[f'TPR {rv}'].min(), 0)}% a {fmt(100 * v[f'TPR {rv}'].max(), 0)}% "
                 f"dos falsos e acusa de {fmt(100 * v[f'FPR {rv}'].min(), 0)}% a {fmt(100 * v[f'FPR {rv}'].max(), 0)}% dos reais",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, "A' = ponto de EER da validação agrupada; M = mediana dos EER por fonte; F5 = no máximo 5% de reais acusados (mediana entre fontes). Cortes da validação.",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    salvar(fig, "v7_04_regras_corte", sfx)

    # (5) deslocamento da nota por fonte: T0 x melhor receita (a vencedora pela validação)
    notas = pd.read_parquet(Path("data/processed") / f"v7_notas{sfx}.parquet")
    nt = notas[notas.split.isin(["teste_mestre", "controle"])].copy()
    nt["rec"] = np.where(nt.bloco == "radar_eval", "radar_en", np.where(nt.split == "controle", nt.fonte, nt.bloco))
    fontes = ["ml_itw", "in_the_wild", "radar_en", "llamapartialspoof", "brspeech_df", "asvspoof2019_la", "asvspoof2021_df"]
    comp = venc if venc != "T0" else pr.set_index("receita")["média AUC (3 fontes)"].drop("T0").idxmax()
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharey=True)
    for ax, rc in zip(axes, ("T0", comp)):
        for i, f in enumerate(fontes):
            g = nt[nt.rec == f]
            for lab, cor, dx in ((0, REAL, -0.17), (1, IA, 0.17)):
                z = g[g.label == lab][f"z_{rc}"].values
                if len(z):
                    q = np.percentile(z, [25, 50, 75])
                    ax.plot([i + dx, i + dx], [q[0], q[2]], color=cor, lw=6, solid_capstyle="round", alpha=0.85)
                    ax.plot(i + dx, q[1], "o", color="white", mec=cor, ms=7, mew=2)
        t = esc["por_receita"][rc]["cortes"][esc["por_receita"][rc]["regra"]]
        ax.axhline(t, color="#1f1f1d", ls="--", lw=1.2)
        ax.text(len(fontes) - 0.5, t, f"corte {esc['por_receita'][rc]['regra']}", va="bottom", ha="right", fontsize=9)
        ax.set_xticks(range(len(fontes)), [NOME[f] for f in fontes], rotation=30, ha="right", fontsize=10)
        ax.set_title(DESCR[rc], fontsize=13)
    axes[0].set_ylabel("nota (logit; maior = mais falso)")
    hs = [Line2D([], [], color=REAL, lw=6, label="reais (25% a 75%, bolinha = mediana)"),
          Line2D([], [], color=IA, lw=6, label="IA")]
    axes[1].legend(handles=hs, loc="upper right", frameon=False, fontsize=10)
    gap = {}
    for rc in ("T0", comp):
        g = nt[nt.rec.isin(["ml_itw", "radar_en"])]
        gap[rc] = g[g.label == 1][f"z_{rc}"].median() - g[g.label == 0][f"z_{rc}"].median()
    br = nt[(nt.rec == "brspeech_df") & (nt.label == 0)][f"z_{comp}"].median()
    fig.suptitle(f"Com o {comp}, os falsos do ML-ITW e do RADAR sobem acima do corte (mediana {fmt(gap[comp], 1)} acima dos reais,\n"
                 f"contra {fmt(gap['T0'], 1)} no T0); os reais do BRSpeech-DF continuam acima do corte (mediana {fmt(br, 1)})",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    salvar(fig, "v7_05_deslocamento_nota", sfx)

    # (6) bloco inglês
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ing = ["radar_en", "radar_en_pareado", "ml_itw_ingles", "in_the_wild", "llamapartialspoof", "asvspoof2019_la", "asvspoof2021_df"]
    barras_auc(ax, tab, ing)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9, frameon=False, ncol=5)
    bi = tab[tab.recorte.isin(["radar_en", "ml_itw_ingles", "in_the_wild"])].pivot(index="receita", columns="recorte", values="AUC")
    m_ = bi.mean(1)
    ax.set_title(f"Inglês: média (RADAR en, ML-ITW inglês, In-the-Wild) vai de {fmt(m_['T0'], 3)} no T0 a {fmt(m_.max(), 3)} "
                 f"no melhor ({m_.idxmax()})", loc="left", fontsize=15, fontweight="bold")
    mli = tab[(tab.recorte == "ml_itw_ingles") & (tab.receita == "T0")].iloc[0]
    fig.text(0.01, 0.01, f"ML-ITW inglês: {int(mli.locutores_real)} / {int(mli.locutores_falso)} locutores real / IA "
             f"- {mli.marca}. * ASVspoof: secundários (possível sobreposição de locutores via VCTK). Hachurado: domínio visto no treino.",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    salvar(fig, "v7_06_bloco_ingles", sfx)
    print("figuras prontas em", OUT)
    return 0
