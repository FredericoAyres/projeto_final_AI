"""Consolida as medições do modelo leve (results/modelo_leve_parcial/*) em tabelas, json e figuras.
Chamado por `python scripts/modelo_leve.py resumo`. Não mede nada: só lê e agrega."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R, PARC = Path("results"), Path("results/modelo_leve_parcial")
DURS = (3, 5, 8)
LIM_MEDIA, LIM_FONTE, TOL_NOTA = 0.003, 0.01, 1e-3
APARELHOS = {"gpu": "placa de vídeo do notebook", "cpu4": "processador do notebook, 4 threads",
             "cpu1": "processador do notebook, 1 thread"}
VARS_AP = {"gpu": ("V0", "V1", "V2"), "cpu4": ("V0", "V1", "V3"), "cpu1": ("V0", "V1", "V3")}
LARANJA, CINZA, CINZA_E = "#E0662C", "#8A8A85", "#C9C7BF"


def ler(p):
    p = PARC / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def latencias():
    linhas = []
    for ap, vs in VARS_AP.items():
        A = ler(f"lat_A_{'cpu4' if ap == 'gpu' else ap}.json")
        for v in vs:
            L = ler(f"lat_{v}_{ap}.json")
            if L is None or A is None:
                continue
            for D in DURS:
                d, a = L["duracoes"][str(D)], np.array(A["duracoes"][str(D)]["a"])
                b, c = np.array(d["b"]), np.array(d["c"])
                partes = {"a_leitura_e_preparo": a, "b_modelo_ate_os_vetores": b, "c_media_padroniza_regressao": c,
                          "total": a + b + c}
                for nome, x in partes.items():
                    ms = x * 1000
                    linhas.append({"variante": v, "aparelho": APARELHOS[ap], "ap": ap, "duracao_s": D, "parte": nome,
                                   "n": len(ms), "mediana_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)),
                                   "media_ms": float(ms.mean())})
    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    ref = df[df.variante == "V0"].set_index(["ap", "duracao_s", "parte"]).mediana_ms
    df["aceleracao_vs_V0"] = [ref.get((r.ap, r.duracao_s, r.parte), np.nan) / r.mediana_ms for r in df.itertuples()]
    return df


def equivalencia():
    res = [pd.read_csv(PARC / f) for f in ("equiv_resumo.csv", "equiv_v3_resumo.csv") if (PARC / f).exists()]
    por = [pd.read_csv(PARC / f) for f in ("equiv_por_fonte.csv", "equiv_v3_por_fonte.csv") if (PARC / f).exists()]
    r = pd.concat(res, ignore_index=True)
    r.insert(1, "fonte", "(média das 6 fontes)")
    p = pd.concat(por, ignore_index=True).rename(columns={"auc_ref": "auc_media_ref", "auc_var": "auc_media_var",
                                                         "queda": "queda_auc_media"})
    p["fonte"] = p["fonte"]
    return pd.concat([r, p], ignore_index=True), r, pd.concat(por, ignore_index=True)


def main():
    lat = latencias()
    eq, r, por = equivalencia()
    tam = pd.read_csv(PARC / "tamanho.csv")
    v3err = ler("v3_erro.json") or {}
    # ---- regra de aceitação
    comp = {"V1": "V1 fp32 x V0 fp32", "V2": "V2 fp16 x V0 fp16", "V3": "V3 int8 x V0 fp32"}
    aceit = {}
    for v, nome in comp.items():
        x = r[r.comparacao == nome]
        if x.empty:
            aceit[v] = {"aceita": None, "motivo": "não medida" if v != "V3" else f"quantização não funcionou: {v3err.get('erro', '')}"}
            continue
        x = x.iloc[0]
        ok = (x.queda_auc_media <= LIM_MEDIA) and (x.maior_queda_auc_fonte <= LIM_FONTE)
        aceit[v] = {"aceita": bool(ok), "comparacao": nome, "queda_auc_media": float(x.queda_auc_media),
                    "maior_queda_auc_fonte": float(x.maior_queda_auc_fonte), "dif_nota_max": float(x.dif_nota_max),
                    "dif_nota_mediana": float(x.dif_nota_mediana)}
        if v == "V1":
            aceit[v]["equivalente"] = bool(x.dif_nota_max <= TOL_NOTA)
    # ---- variantes
    linhas = []
    for t in tam.itertuples():
        d = {"variante": t.variante, "descricao": t.descricao, "parametros_total": t.total,
             "param_extrator_convolucoes": t.extrator_convolucoes, "param_projecao": t.projecao_caracteristicas,
             "param_convolucao_posicional": t.convolucao_posicional, "param_camadas_transformer": t.camadas_transformer,
             "param_layer_norm_final": t.layer_norm_final, "param_outros": t.outros, "param_regressao": 3073,
             "mb_modelo": t.mb_modelo, "mb_regressao": t.mb_regressao, "mb_total": t.mb_modelo + t.mb_regressao}
        for ap in ("gpu", "cpu4", "cpu1"):
            L = ler(f"lat_{t.variante}_{ap}.json")
            if L:
                d[f"pico_ram_mb_{ap}"] = L["pico_ram_processo_mb"]
                d[f"camadas_executadas_{ap}"] = L["camadas_executadas_por_audio"]
                if ap == "gpu":
                    d["pico_gpu_mb"] = L["pico_gpu_mb"]
        a = aceit.get(t.variante)
        d["aceita_pela_regra"] = None if a is None else a["aceita"]
        linhas.append(d)
    var = pd.DataFrame(linhas)
    lat.drop(columns="ap").to_csv(R / "modelo_leve_latencia.csv", index=False)
    eq.to_csv(R / "modelo_leve_equivalencia.csv", index=False)
    var.to_csv(R / "modelo_leve_variantes.csv", index=False)
    resumo = {"regra": {"queda_max_auc_media": LIM_MEDIA, "queda_max_auc_por_fonte": LIM_FONTE,
                        "dif_max_nota_V1_logit": TOL_NOTA},
              "aceitacao": aceit, "v3_quantizacao": v3err,
              "total_ms_mediana": {f"{r_.variante}|{r_.ap}|{r_.duracao_s}s": round(r_.mediana_ms, 2)
                                   for r_ in lat[lat.parte == "total"].itertuples()} if not lat.empty else {}}
    (R / "modelo_leve_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    figuras(lat, var, r, por, aceit)
    print("tabelas e figuras gravadas")


def figuras(lat, var, r, por, aceit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 13, "axes.facecolor": "#FBFAF6",
                         "figure.facecolor": "#FBFAF6", "axes.spines.top": False, "axes.spines.right": False})
    cor = lambda v: CINZA if v == "V0" else (LARANJA if aceit.get(v, {}).get("aceita") else CINZA_E)  # noqa: E731
    nomes = {"V0": "V0\ncompleto", "V1": "V1\ncortado", "V2": "V2\ncortado\nfp16", "V3": "V3\ncortado\nint8"}
    tot = lambda v, ap, D: lat[(lat.variante == v) & (lat.ap == ap) & (lat.duracao_s == D) & (lat.parte == "total")].mediana_ms.iloc[0]  # noqa: E731
    # ---------- figura 1: tamanho e tempo
    fig, ax = plt.subplots(2, 2, figsize=(13.33, 7.5), dpi=180)
    fig.subplots_adjust(top=0.74, bottom=0.09, left=0.07, right=0.98, hspace=0.75, wspace=0.18)
    v0 = var[var.variante == "V0"].iloc[0]
    vs = var.variante.tolist()
    ax[0, 0].bar([nomes[v] for v in vs], var.parametros_total / 1e6, color=[cor(v) for v in vs])
    ax[0, 0].set_title("Parâmetros (milhões)", loc="left")
    ax[0, 0].set_ylim(0, 380)
    for i, v in enumerate(var.parametros_total / 1e6):
        ax[0, 0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=13)
    ax[0, 1].bar([nomes[v] for v in vs], var.mb_total, color=[cor(v) for v in vs])
    ax[0, 1].set_title("Tamanho em disco (MB, modelo + regressão)", loc="left")
    ax[0, 1].set_ylim(0, 1450)
    for i, v in enumerate(var.mb_total):
        ax[0, 1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=13)
    for k, (ap, vv) in enumerate((("gpu", ("V0", "V1", "V2")), ("cpu4", ("V0", "V1", "V3")))):
        a = ax[1, k]
        w = 0.26
        for j, v in enumerate(vv):
            if lat[(lat.variante == v) & (lat.ap == ap)].empty:
                continue
            y = [tot(v, ap, D) for D in DURS]
            a.bar(np.arange(3) + (j - 1) * w, y, w, color=cor(v), label=nomes[v].replace("\n", " "))
            for x, yy in zip(np.arange(3) + (j - 1) * w, y):
                a.text(x, yy, f"{yy:.0f}", ha="center", va="bottom", fontsize=13)
        a.set_xticks(range(3), [f"{D} s de áudio" for D in DURS])
        a.set_title("Tempo por áudio (ms) – " + ("placa de vídeo" if ap == "gpu" else "processador, 4 threads"), loc="left")
        a.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1)
        a.set_ylim(0, a.get_ylim()[1] * 1.45)
    ac = [v for v in ("V1", "V2", "V3") if aceit.get(v, {}).get("aceita")]
    f = 1 - v0.mb_total / var[var.variante == "V1"].mb_total.iloc[0] ** -1 if False else None  # noqa: F841
    mb1 = var[var.variante == "V1"].mb_total.iloc[0]
    sp = tot("V0", "cpu4", 5) / tot("V1", "cpu4", 5)
    fig.text(0.07, 0.955, f"Cortar o modelo deixa o arquivo {v0.mb_total / mb1:.1f}x menor e a resposta\n"
                          f"{sp:.1f}x mais rápida no processador", fontsize=22, fontweight="bold", va="top", linespacing=1.15)
    fig.text(0.07, 0.855, "Exemplo do título: áudio de 5 s, V1 contra V0 (processador, 4 threads). Tempos do V0 como vem (24 camadas).\n"
                          "Valem para este notebook e variam em outros aparelhos. Laranja = aceita pela regra.",
             fontsize=14, va="top", color="#444")
    fig.savefig(R / "slides" / "ml_01_tamanho_e_tempo.png", dpi=180, facecolor="#FBFAF6")
    plt.close(fig)
    fmt = lambda v: "0" if abs(v) < 1e-9 else f"{v:.4f}"  # noqa: E731
    # ---------- figura 2: equivalência
    fig, ax = plt.subplots(1, 2, figsize=(13.33, 7.5), dpi=180)
    fig.subplots_adjust(top=0.68, bottom=0.17, left=0.08, right=0.98, wspace=0.28)
    cmp_ = [("V1", "V1 fp32 x V0 fp32"), ("V2", "V2 fp16 x V0 fp16"), ("V3", "V3 int8 x V0 fp32")]
    cmp_ = [(v, n) for v, n in cmp_ if (r.comparacao == n).any()]
    rx = r.set_index("comparacao")
    x = np.arange(len(cmp_))
    ax[0].bar(x, [rx.loc[n, "dif_nota_max"] for v, n in cmp_], color=[cor(v) for v, _ in cmp_])
    for i, (v, n) in enumerate(cmp_):
        ax[0].text(i, rx.loc[n, "dif_nota_max"], f"{rx.loc[n, 'dif_nota_max']:.3g}", ha="center", va="bottom", fontsize=13)
    ax[0].axhline(TOL_NOTA, color="#444", ls="--", lw=1.2)
    ax[0].text(-0.4, TOL_NOTA + 0.25, "limite da V1: 0,001", ha="left", va="bottom", fontsize=13, color="#444")
    ax[0].set_xticks(x, [f"{v}\nvs V0 {'fp16' if v == 'V2' else 'fp32'}" for v, _ in cmp_])
    ax[0].set_title("Maior diferença de nota (logit)\nem 2.000 clipes", loc="left")
    ax[1].bar(x - 0.18, [max(rx.loc[n, "queda_auc_media"], 0) for v, n in cmp_], 0.36, color=[cor(v) for v, _ in cmp_],
              label="queda na AUC média")
    ax[1].bar(x + 0.18, [max(rx.loc[n, "maior_queda_auc_fonte"], 0) for v, n in cmp_], 0.36, color=[cor(v) for v, _ in cmp_],
              alpha=0.55, label="maior queda numa fonte")
    for i, (v, n) in enumerate(cmp_):
        ax[1].text(i - 0.18, max(rx.loc[n, "queda_auc_media"], 0), fmt(rx.loc[n, 'queda_auc_media']), ha="center", va="bottom", fontsize=13)
        ax[1].text(i + 0.18, max(rx.loc[n, "maior_queda_auc_fonte"], 0), fmt(rx.loc[n, 'maior_queda_auc_fonte']), ha="center", va="bottom", fontsize=13)
    ax[1].axhline(LIM_MEDIA, color="#444", ls="--", lw=1.2)
    ax[1].axhline(LIM_FONTE, color="#444", ls=":", lw=1.2)
    ax[1].text(len(cmp_) - 0.5, LIM_MEDIA, "limite da média (0,003) ", ha="right", va="bottom", fontsize=13, color="#444")
    ax[1].text(len(cmp_) - 0.5, LIM_FONTE, "limite por fonte (0,01) ", ha="right", va="bottom", fontsize=13, color="#444")
    ax[1].set_xticks(x, [f"{v}" for v, _ in cmp_])
    ax[1].set_title("Queda da AUC contra a V0\n(AUC por fonte, 6 fontes)", loc="left")
    ax[1].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=13)
    ok = [v for v, _ in cmp_ if aceit[v]["aceita"]]
    d1 = rx.loc["V1 fp32 x V0 fp32", "dif_nota_max"]
    d3 = rx.loc["V3 int8 x V0 fp32", "dif_nota_max"] if "V3 int8 x V0 fp32" in rx.index else None
    t1 = "Cortado e em fp16: mesma nota do modelo inteiro." if d1 == 0 else "Cortado: quase a mesma nota do modelo inteiro."
    t2 = " Em int8, a nota muda, mas a AUC quase não cai" if d3 is not None else ""
    import textwrap
    fig.text(0.08, 0.955, "\n".join(textwrap.wrap(t1 + t2, 46)), fontsize=22, fontweight="bold", va="top", linespacing=1.15)
    med3 = rx.loc["V3 int8 x V0 fp32", "dif_nota_mediana"] if d3 is not None else 0
    sub = (f"Cortar camadas que não entram na conta não muda a nota. O int8 muda a nota em até {d3:.1f} logit (mediana {med3:.1f}):\n"
           f"vale para a ordem dos clipes (AUC), não para o corte de decisão da T3. "
           f"Aceitas pela regra: {', '.join(ok) if ok else 'nenhuma'} (laranja).") if d3 is not None else ""
    fig.text(0.08, 0.84, sub, fontsize=14, va="top", color="#444")
    fig.savefig(R / "slides" / "ml_02_equivalencia.png", dpi=180, facecolor="#FBFAF6")
    plt.close(fig)
