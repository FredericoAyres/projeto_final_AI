"""Relatório da v6: tabela do funil, gráficos e projeção 2D, a partir dos arquivos em disco (não treina nada).

Também escreve logs/v6_resumo.txt e logs/v6_PROGRESSO.md (status de cada etapa), lidos pelo vigia.
⚠️ Fora da trilha: PCA e t-SNE (projeções em 2D de vetores de 1.024 números).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ETAPAS_TODAS = ("medir", "extracao", "e1", "e2", "e3", "e4", "e5", "finalistas", "teste_extracao", "teste",
                "relatorio", "notebook")
NOME_ETAPA = {"medir": "vazão do WavLM", "extracao": "extração do WavLM (treino, val, dev_shift)",
              "e1": "E1 Meta × Microsoft", "e2": "E2 fusão", "e3": "E3 front-end", "e4": "E4 cabeças básicas",
              "e5": "E5 maximização", "finalistas": "finalistas gravados", "teste_extracao": "E6 extração do teste",
              "teste": "E6 teste (uma vez)", "relatorio": "relatório e gráficos", "notebook": "notebook"}
NOME_FE = {"xlsr": "só Meta (XLS-R)", "wavlm": "só Microsoft (WavLM)", "fusao_media": "fusão: média das notas",
           "fusao_stacking": "fusão: stacking", "fusao_colados": "fusão: vetores colados"}
COR = {"xlsr": "#eb6834", "wavlm": "#2a78d6", "fusao": "#1baf7a", "outro": "#9aa0a6", "venc": "#222222"}


def _j(p, padrao=None):
    p = Path(p)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao
    except (OSError, ValueError):
        return padrao


def _f(x, d=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ---------------------------------------------------------------- tabela do funil
def funil(ns):
    """Uma linha por etapa: candidatos, métrica, vencedor, por quê, alternativa descartada e, para cada
    candidato, a AUC da pior fonte e a acurácia balanceada média (fontes de decisão)."""
    A = ns["arq"]
    linhas = []

    def cand_txt(d):
        return "; ".join(f"{k}: {_f(v[0])}" for k, v in d.items()), "; ".join(f"{k}: {_f(v[1], 3)}" for k, v in d.items())

    e1 = _j(A("e1_resumo", "json"))
    if e1:
        mm = e1["melhor_metricas"]
        d = {f"Meta ({mm['xlsr']['cabeça']})": (mm["xlsr"]["AUC pior fonte"], mm["xlsr"].get("acc bal média (G)")),
             f"Microsoft ({mm['wavlm']['cabeça']})": (mm["wavlm"]["AUC pior fonte"], mm["wavlm"].get("acc bal média (G)")),
             "v4 (referência)": (e1["v4_ref"]["AUC pior fonte"], e1["v4_ref"].get("acc bal média (G)"))}
        c = e1["bootstrap_wavlm_menos_xlsr"]
        dif, (lo, hi) = c["dif AUC pior (desafiante - atual)"], c["IC95"]
        if c["relevante"]:
            v, alt = "Microsoft", "Meta"
        elif hi < 0 and dif <= -ns["HP"]["DIF_MIN"]:
            v, alt = "Meta", "Microsoft"
        else:
            v, alt = "empate (a E3 decide: fica o mais simples)", "—"
        a_, b_ = cand_txt(d)
        linhas.append({"etapa": "E1 Meta × Microsoft", "candidatos": " | ".join(d), "métrica": "AUC da pior fonte (bootstrap pareado por locutor)",
                       "vencedor": v, "por quê": f"Microsoft − Meta = {dif:+.4f} (IC95 {lo:+.4f} a {hi:+.4f})",
                       "alternativa descartada": alt, "AUC pior fonte (cada candidato)": a_, "acc. balanceada média (cada candidato)": b_})
    e2 = _j(A("e2_resumo", "json"))
    if e2:
        d = {NOME_FE[k]: (v["AUC pior fonte"], v.get("acc bal média (G)")) for k, v in e2["metricas"].items()}
        a_, b_ = cand_txt(d)
        v = e2["vencedor_fusao"]
        linhas.append({"etapa": "E2 fusão", "candidatos": " | ".join(d), "métrica": "AUC da pior fonte; regra média → stacking → colados",
                       "vencedor": NOME_FE[v], "por quê": _razao(e2["comparacoes"], v),
                       "alternativa descartada": ", ".join(NOME_FE[k] for k in e2["metricas"] if k != v),
                       "AUC pior fonte (cada candidato)": a_, "acc. balanceada média (cada candidato)": b_})
    e3 = _j(A("e3_frontend", "json"))
    if e3:
        d = {NOME_FE[k]: (v["AUC pior fonte"], v.get("acc bal média (G)")) for k, v in e3["metricas"].items()}
        a_, b_ = cand_txt(d)
        linhas.append({"etapa": "E3 front-end", "candidatos": " | ".join(d), "métrica": "AUC da pior fonte; regra Meta → Microsoft → fusão",
                       "vencedor": NOME_FE[e3["vencedor"]], "por quê": _razao(e3["comparacoes"], e3["vencedor"]),
                       "alternativa descartada": ", ".join(NOME_FE[k] for k in e3["metricas"] if k != e3["vencedor"]),
                       "AUC pior fonte (cada candidato)": a_, "acc. balanceada média (cada candidato)": b_})
    e4 = _j(A("e4_resumo", "json"))
    if e4 and A("e4_cabecas", "csv").exists():
        t = pd.read_csv(A("e4_cabecas", "csv"))
        d = {}
        for _, r in t.iterrows():
            if str(r["seed"]) in ("-",) or "média" in str(r["seed"]):
                d[r["sistema"]] = (r["AUC pior fonte"], r.get("acc bal média (G)"))
            else:
                d[r["sistema"]] = (r["AUC pior fonte"], r.get("acc bal média (G)"))
        a_, b_ = cand_txt(d)
        linhas.append({"etapa": "E4 cabeças básicas", "candidatos": " | ".join(d), "métrica": "AUC da pior fonte; regra RL → MLP → CNN (MLP/CNN precisam ganhar nas 3 seeds)",
                       "vencedor": e4["vencedor"], "por quê": _razao(e4["comparacoes"], e4["vencedor"]),
                       "alternativa descartada": ", ".join(x for x in ("RL", "MLP", "CNN") if x != e4["vencedor"]),
                       "AUC pior fonte (cada candidato)": a_, "acc. balanceada média (cada candidato)": b_})
    e5 = _j(A("e5_resumo", "json"))
    if e5:
        ps = e5["passos"]
        d = {f"passo {p['passo']}: {p['modificação']}": (p["AUC pior depois"], p.get("acc bal média depois")) for p in ps}
        a_, b_ = cand_txt(d)
        mant = [p["modificação"] for p in ps if p["mantida"]]
        desc = [p["modificação"] for p in ps if not p["mantida"]]
        linhas.append({"etapa": "E5 maximização", "candidatos": " | ".join(d),
                       "métrica": "manter se pior fonte +0,003 e dev_shift sem cair > 0,01; limiar por validação cruzada deixando uma fonte de fora",
                       "vencedor": "; ".join(mant) if mant else "nenhuma modificação (fica a básica, limiar A)",
                       "por quê": f"pior fonte final {_f(e5['metricas_final']['AUC pior fonte'])}; regra de limiar {e5['regra_limiar']}",
                       "alternativa descartada": "; ".join(desc) if desc else "—",
                       "AUC pior fonte (cada candidato)": a_, "acc. balanceada média (cada candidato)": b_})
    te = _j(A("teste_resumo", "json"))
    fi = _j(A("finalistas", "json"))
    if te and fi:
        d = {}
        for k, f in fi["finalistas"].items():
            ac = te["resumo_finalistas"].get(k, {})
            d[f"{k}: {f['descrição']}"] = (f["metricas_val_dev"]["AUC pior fonte"], ac.get("média acurácia balanceada (3 principais)"))
        a_, b_ = cand_txt(d)
        linhas.append({"etapa": "E6 teste (uma vez, só mostra)", "candidatos": " | ".join(d),
                       "métrica": "acurácia e acurácia balanceada por banco (nada é escolhido aqui)", "vencedor": "— (o teste não escolhe)",
                       "por quê": "; ".join(f"{k}: média da acurácia nos 3 principais {_f(v.get('média acurácia (3 principais)'), 3)} "
                                            f"(trivial {_f(v.get('média acurácia trivial (3 principais)'), 3)})"
                                            for k, v in te["resumo_finalistas"].items()),
                       "alternativa descartada": "—", "AUC pior fonte (cada candidato)": a_ + " (val + dev_shift)",
                       "acc. balanceada média (cada candidato)": b_ + " (média dos 3 bancos principais, limiar principal)"})
    return pd.DataFrame(linhas)


def _razao(comps, vencedor):
    if not comps:
        return "único candidato"
    partes = []
    for c in comps:
        rs = c["por seed"]
        if len(rs) == 1:
            r = rs[0]
            partes.append(f"{c['desafiante']} − {c['atual']}: {r['dif AUC pior (desafiante - atual)']:+.4f} "
                          f"(IC95 {r['IC95'][0]:+.4f} a {r['IC95'][1]:+.4f}) → {'relevante' if c['relevante em todas as seeds'] else 'não relevante'}")
        else:
            partes.append(f"{c['desafiante']} − {c['atual']} por seed: " + ", ".join(
                f"{r['dif AUC pior (desafiante - atual)']:+.4f}{'*' if r['relevante'] else ''}" for r in rs)
                + f" → {'relevante nas 3' if c['relevante em todas as seeds'] else 'não relevante nas 3'}")
    return f"fica {vencedor}: " + "; ".join(partes)


# ---------------------------------------------------------------- gráficos
def _salvar(fig, destino):
    from src.arquivos import gravar_atomico
    gravar_atomico(destino, lambda t: fig.savefig(t, format="png"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def grafico_funil(ns):
    import matplotlib.pyplot as plt
    A = ns["arq"]
    paineis = []
    e1 = _j(A("e1_resumo", "json"))
    if e1:
        mm = e1["melhor_metricas"]
        paineis.append(("E1", [("Meta", mm["xlsr"]["AUC pior fonte"], COR["xlsr"]), ("Microsoft", mm["wavlm"]["AUC pior fonte"], COR["wavlm"]),
                               ("v4 ref.", e1["v4_ref"]["AUC pior fonte"], COR["outro"])], None))
    e2 = _j(A("e2_resumo", "json"))
    if e2:
        paineis.append(("E2 fusão", [(NOME_FE[k].replace("fusão: ", ""), v["AUC pior fonte"], COR["fusao"]) for k, v in e2["metricas"].items()],
                        NOME_FE[e2["vencedor_fusao"]].replace("fusão: ", "")))
    e3 = _j(A("e3_frontend", "json"))
    if e3:
        cores = {"xlsr": COR["xlsr"], "wavlm": COR["wavlm"]}
        paineis.append(("E3 front-end", [(NOME_FE[k], v["AUC pior fonte"], cores.get(k, COR["fusao"])) for k, v in e3["metricas"].items()],
                        NOME_FE[e3["vencedor"]]))
    if A("e4_cabecas", "csv").exists() and _j(A("e4_resumo", "json")):
        t = pd.read_csv(A("e4_cabecas", "csv"))
        t = t[~t.sistema.str.contains("média")]
        v4 = _j(A("e4_resumo", "json"))["vencedor"]
        paineis.append(("E4 cabeças", [(r.sistema.replace(" básico", "").replace(" básica", "")[:34], r["AUC pior fonte"],
                                        COR["outro"] if "v4" in r.sistema else COR["fusao"]) for _, r in t.iterrows()],
                        {"RL": "RL", "MLP": "MLP", "CNN": "CNN"}[v4]))
    e5 = _j(A("e5_resumo", "json"))
    if e5:
        bars = [("básica (início)", e5["passos"][0]["AUC pior antes"], COR["outro"])]
        for p in e5["passos"]:
            if "limiar" in p["modificação"]:
                continue
            bars.append((f"{p['passo']}: {p['modificação'][:30]}" + (" ✓" if p["mantida"] else " ✗"), p["AUC pior depois"],
                         COR["fusao"] if p["mantida"] else COR["outro"]))
        paineis.append(("E5 passos", bars, None))
    if not paineis:
        return None
    fig, axes = plt.subplots(1, len(paineis), figsize=(3.4 * len(paineis), 3.8), dpi=80, sharey=True)
    axes = np.atleast_1d(axes)
    vals = [b[1] for _, bs, _ in paineis for b in bs if b[1] is not None and np.isfinite(b[1])]
    lo = max(0.5, min(vals) - 0.02) if vals else 0.5
    for ax, (tit, bs, venc) in zip(axes, paineis):
        x = np.arange(len(bs))
        ax.bar(x, [b[1] for b in bs], color=[b[2] for b in bs],
               edgecolor=[COR["venc"] if venc and venc in b[0] else "none" for b in bs], linewidth=2)
        for i, b in enumerate(bs):
            ax.text(i, b[1] + 0.002, f"{b[1]:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x, [b[0] for b in bs], rotation=35, ha="right", fontsize=7)
        ax.set_title(tit, fontsize=10)
        ax.set_ylim(lo, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("AUC da pior fonte (val + dev_shift)")
    fig.suptitle("Funil da v6 (borda preta = vencedor da etapa pela regra pré-registrada)", fontsize=10)
    fig.tight_layout()
    _salvar(fig, A("funil", "png"))
    return A("funil", "png")


def grafico_camadas(ns):
    import matplotlib.pyplot as plt
    A = ns["arq"]
    if not A("e1_camadas", "csv").exists():
        return None
    t = pd.read_csv(A("e1_camadas", "csv"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), dpi=80)
    for m, cor in (("xlsr", COR["xlsr"]), ("wavlm", COR["wavlm"])):
        g = t[t.modelo == m].sort_values("camada")
        ax[0].plot(g.camada, g["AUC pior fonte"], "o-", ms=3, color=cor, label=ns["NOME_MODELO"][m])
        if "dev AUC" in g:
            ax[1].plot(g.camada, g["dev AUC"], "o-", ms=3, color=cor, label=ns["NOME_MODELO"][m])
    ax[0].set(xlabel="saída do modelo", ylabel="AUC", title="AUC da pior fonte (val + dev_shift)")
    ax[1].set(xlabel="saída do modelo", ylabel="AUC", title="AUC do dev_shift (vozes de internet)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    _salvar(fig, A("camadas_meta_microsoft", "png"))
    return A("camadas_meta_microsoft", "png")


def grafico_2x2(ns):
    import matplotlib.pyplot as plt
    A = ns["arq"]
    e2 = _j(A("e2_resumo", "json"))
    if not e2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=80)
    for ax, nome in zip(axes, ("validação", "dev_shift")):
        d = e2["erros_2x2"][nome]
        M = np.array([[d["os dois acertam"], d["só a Meta acerta"]], [d["só a Microsoft acerta"], d["os dois erram"]]])
        ax.imshow(M, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:,}\n({100 * M[i, j] / d['n']:.1f}%)".replace(",", "."), ha="center", va="center",
                        fontsize=9, color="white" if M[i, j] > M.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Microsoft acerta", "Microsoft erra"])
        ax.set_yticks([0, 1], ["Meta acerta", "Meta erra"])
        ax.set_title(f"{nome} (n = {d['n']:,})".replace(",", "."), fontsize=10)
    fig.suptitle("Erros em comum (cada modelo no seu limiar A)", fontsize=10)
    fig.tight_layout()
    _salvar(fig, A("erros_2x2", "png"))
    return A("erros_2x2", "png")


def grafico_limiares_val(ns):
    """(f) falso alarme e fake que passa por fonte (validação e dev_shift) do F2, nos limiares A, B e E5."""
    import matplotlib.pyplot as plt
    A = ns["arq"]
    fi = _j(A("finalistas", "json"))
    if not fi or not A("notas_e5_val", "parquet").exists():
        return None
    nv, nd = pd.read_parquet(A("notas_e5_val", "parquet")), pd.read_parquet(A("notas_e5_dev_shift", "parquet"))
    lims = fi["finalistas"]["F2"]["limiares_logit"]
    grupos = [(f, nv[nv.source_dataset == f]) for f in sorted(nv.source_dataset.unique())] + [("dev_shift", nd)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), dpi=80)
    w = 0.8 / len(lims)
    for i, (L, t) in enumerate(lims.items()):
        fa, fp = [], []
        for _, g in grupos:
            y, z = g.label.values, g.z_final.values
            fa.append(100 * (z[y == 0] >= t).mean() if (y == 0).any() else np.nan)
            fp.append(100 * (z[y == 1] < t).mean() if (y == 1).any() else np.nan)
        x = np.arange(len(grupos)) + (i - (len(lims) - 1) / 2) * w
        axes[0].bar(x, fa, w * 0.9, label=f"limiar {L} ({t:+.2f})")
        axes[1].bar(x, fp, w * 0.9, label=f"limiar {L} ({t:+.2f})")
    for ax, tit in zip(axes, ("falso alarme (% dos reais acusados)", "fake que passa (% dos fakes aceitos)")):
        ax.set_xticks(np.arange(len(grupos)), [g[0] for g in grupos], fontsize=8)
        ax.set_title(tit, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("F2 (sistema final) na validação e no dev_shift, por fonte e por limiar", fontsize=10)
    fig.tight_layout()
    _salvar(fig, A("limiares_por_fonte_val", "png"))
    return A("limiares_por_fonte_val", "png")


def projecao(ns, log=print):
    """(g) PCA e t-SNE dos vetores da melhor camada única (E4) e da última camada (24), real × fake e por fonte,
    numa amostra de até PROJ_N clipes da validação + dev_shift (mesmo nº por fonte, quando dá)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from src.v6_cabecas import Modelo
    A = ns["arq"]
    e4 = _j(A("e4_resumo", "json"))
    if not e4:
        return None
    alvo = A("projecao", "png")
    modelos = e4["modelos"]
    linhas_fig, info = [], {}
    rng = np.random.default_rng(42)
    for m in modelos:
        M = Modelo(m, ns["CACHE"][m], ns["PARCIAL"] / f"{m}_padronizacao.npz")
        mv, md = M.meta("val"), M.meta("dev_shift")
        meta = pd.concat([mv.assign(_c="val", _i=np.arange(len(mv))), md.assign(_c="dev_shift", _i=np.arange(len(md)),
                                                                                  source_dataset="dev_shift (In-the-Wild)")])
        fontes = meta.source_dataset.value_counts()
        cota, resto, n = {}, ns["PROJ_N"], len(fontes)
        for f, c in fontes.sort_values().items():
            cota[f] = min(c, resto // n)
            resto -= cota[f]
            n -= 1
        sel = pd.concat([g.iloc[rng.permutation(len(g))[:cota[f]]] for f, g in meta.groupby("source_dataset")])
        for l in (e4["camada_basica"][m][0], 24):
            Xs = []
            for c in ("val", "dev_shift"):
                s = sel[sel._c == c]
                Xs.append(M.X(c, [l])[s._i.values])
            X = np.vstack(Xs)
            mt = pd.concat([sel[sel._c == "val"], sel[sel._c == "dev_shift"]])
            p2 = PCA(2, random_state=42).fit(X)
            P = p2.transform(X)
            X50 = PCA(min(50, X.shape[1], len(X) - 1), random_state=42).fit_transform(X)
            perp = float(min(30, max(5, (len(X) - 1) / 3)))
            T = TSNE(2, perplexity=perp, init="pca", random_state=42).fit_transform(X50)
            linhas_fig.append((m, l, P, T, mt, p2.explained_variance_ratio_))
            log(f"  projeção {m} camada {l}: {len(X)} pontos")
        info[m] = {"n": int(len(sel)), "por_fonte": {k: int(v) for k, v in cota.items()}}
    cores_f = {"dfadd": "#1baf7a", "shiftyspeech": "#eda100", "codecfake_plus": "#d6336c", "librispeech": "#7a5af8",
               "dev_shift (In-the-Wild)": "#2a78d6"}
    fig, axes = plt.subplots(len(linhas_fig), 4, figsize=(15, 3.4 * len(linhas_fig)), dpi=70, squeeze=False)
    for i, (m, l, P, T, mt, ev) in enumerate(linhas_fig):
        y = mt.label.values
        for j, (E, nome) in enumerate(((P, f"PCA ({100 * ev.sum():.0f}% da variância)"), (T, "t-SNE"))):
            a1, a2 = axes[i, 2 * j], axes[i, 2 * j + 1]
            for c, rot, cor in ((0, "real", "#2a78d6"), (1, "fake", "#d6336c")):
                k = y == c
                a1.scatter(E[k, 0], E[k, 1], s=2, alpha=0.4, color=cor, label=rot)
            for f, cor in cores_f.items():
                k = mt.source_dataset.values == f
                if k.any():
                    a2.scatter(E[k, 0], E[k, 1], s=2, alpha=0.4, color=cor, label=f)
            a1.set_title(f"{ns['NOME_MODELO'][m]}, camada {l}: {nome}\nreal × fake", fontsize=8)
            a2.set_title(f"camada {l}: {nome}\npor fonte", fontsize=8)
            for a in (a1, a2):
                a.set_xticks([])
                a.set_yticks([])
            if i == 0:
                a1.legend(fontsize=7, markerscale=4)
                a2.legend(fontsize=6, markerscale=4)
    fig.tight_layout()
    _salvar(fig, alvo)
    return info


# ---------------------------------------------------------------- gerar tudo
def gerar(ns, log=print):
    from src.arquivos import salvar_csv, salvar_json
    out = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "figuras": {}}
    t = funil(ns)
    if len(t):
        salvar_csv(t, ns["arq"]("funil", "csv"))
        log(t[["etapa", "vencedor", "por quê"]].to_string(index=False))
    for nome, fn in (("funil", grafico_funil), ("camadas", grafico_camadas), ("erros_2x2", grafico_2x2),
                     ("limiares_val", grafico_limiares_val)):
        try:
            p = fn(ns)
            out["figuras"][nome] = str(p) if p else None
        except Exception as e:  # noqa: BLE001 (um gráfico com problema não derruba os outros)
            log(f"aviso: gráfico {nome} falhou: {e!r}")
            out["figuras"][nome] = f"falhou: {e!r}"
    try:
        out["projecao"] = projecao(ns, log)
    except Exception as e:  # noqa: BLE001
        log(f"aviso: projeção falhou: {e!r}")
        out["projecao"] = f"falhou: {e!r}"
    salvar_json(out, ns["arq"]("relatorio", "json"))
    return out


# ---------------------------------------------------------------- status (lido pelo vigia)
def escrever_status(ns):
    from src.arquivos import gravar_atomico
    A, suf = ns["arq"], ns["SUFIXO"]
    agora = datetime.now()
    L = [f"Resumo v6{' (QUICK)' if ns['QUICK'] else ''} (atualizado em {agora:%Y-%m-%d %H:%M:%S})",
         "Decisões só com validação + dev_shift. Teste (E6) só depois dos finalistas gravados.", ""]
    md = [f"# Progresso da v6{' (QUICK)' if ns['QUICK'] else ''}", "", f"Atualizado em {agora:%Y-%m-%d %H:%M:%S}.", "",
          "| etapa | status | min | pico GB | observação |", "|:--|:--|--:|--:|:--|"]
    for e in ETAPAS_TODAS:
        m = _j(ns["marcador"](e))
        obs = ""
        if m:
            if e == "medir":
                v = m.get("vazao", {})
                obs = f"{v.get('clipes_por_s')} clipes/s; estimativa {v.get('estimativa_extracao_min (treino+val+dev, 70.613 clipes)')} min"
            elif e == "extracao":
                obs = f"{m.get('horas_extracao_acumuladas')} h; erros {m.get('erros_leitura')}"
            elif e == "e1":
                obs = (f"melhor: {json.dumps(m.get('melhor_cabeca'))}; pior fonte {json.dumps({k: round(v, 4) for k, v in m.get('pior_fonte', {}).items()})}; "
                       f"reprodução v4: {m.get('reproducao_v4', {}).get('passou')}")
            elif e == "e2":
                obs = f"fusão vencedora: {m.get('vencedor_fusao')}"
            elif e in ("e3", "e4"):
                obs = f"vencedor: {m.get('vencedor')}"
            elif e == "e5":
                obs = f"final: pior fonte {_f(m.get('pior_fonte_final'))}; limiar {m.get('regra_limiar')}"
            elif e == "finalistas":
                obs = f"extrair teste do WavLM: {m.get('precisa_extrair_teste_wavlm')}"
            elif e == "teste_extracao":
                obs = m.get("observacao", "")
            st = "concluída"
            mins, pico = m.get("minutos_neste_processo", 0), m.get("pico_memoria_gb_processo_e_workers", 0)
        else:
            st, mins, pico = "pendente", None, None
        L.append(f"{e:<15} {st:<10} {('%.1f' % mins) if mins is not None else '-':>6} {('%.2f' % pico) if pico is not None else '-':>7}  {obs}")
        md.append(f"| {NOME_ETAPA[e]} | {st} | {('%.1f' % mins) if mins is not None else '-'} | "
                  f"{('%.2f' % pico) if pico is not None else '-'} | {obs.replace('|', '/')} |")
    est = _j(A("wavlm_estado", "json"), {})
    if est:
        L += ["", f"extração WavLM: {est.get('horas_extracao', 0):.2f} h de {ns['ORCAMENTO_EXTRACAO_H']} h | último shard {est.get('ultimo_shard', '-')}"]
        md += ["", f"Extração do WavLM: {est.get('horas_extracao', 0):.2f} h de {ns['ORCAMENTO_EXTRACAO_H']} h; último shard `{est.get('ultimo_shard', '-')}`."]
    if A("funil", "csv").exists():
        t = pd.read_csv(A("funil", "csv"))
        L += ["", "FUNIL:"] + [f"  {r.etapa}: {r.vencedor} | {r['por quê']}" for _, r in t.iterrows()]
        md += ["", "## Funil", ""] + [f"- **{r.etapa}:** {r.vencedor}. {r['por quê']}" for _, r in t.iterrows()]
    te = _j(A("teste_resumo", "json"))
    if te:
        L += ["", "TESTE (E6):"] + [f"  {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in te.get("resumo_finalistas", {}).items()]
        md += ["", "## Teste (E6)", ""] + [f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in te.get("resumo_finalistas", {}).items()]
    L += ["", f"concluído: {'SIM' if A('concluido', 'json').exists() else 'não'}"]
    txt = "\n".join(L) + "\n"
    gravar_atomico(ns["LOGS_DIR"] / f"v6_resumo{suf}.txt", lambda t_: Path(t_).write_text(txt, encoding="utf-8-sig"))
    gravar_atomico(ns["LOGS_DIR"] / f"v6_PROGRESSO{suf}.md", lambda t_: Path(t_).write_text("\n".join(md) + "\n", encoding="utf-8"))
    print(txt, flush=True)
