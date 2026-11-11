#!/usr/bin/env python3
"""E6 da v6: teste final, avaliado UMA vez, só para mostrar os finalistas declarados antes (F0, F1, F2).

Pré-registro: docs/HISTORICO_VERSOES.md, seção "v6 - pré-registro". Este script só roda depois de E1 a E5 e
dos finalistas gravados (results/v6_finalistas.json); ele NÃO escolhe nada: camadas, C, pesos e limiares
vêm desse arquivo (calculados na validação e no dev_shift). Reaproveita src/teste_v4.py (amostra, métricas,
bootstrap) e src/extracao_v4.py (extração).

    python scripts/v6_teste.py extrair  [--quick]   # teste do WavLM, só se algum finalista usa o WavLM
    python scripts/v6_teste.py avaliar  [--quick]   # notas, métricas, bootstrap, gráficos (b), (c), (f)
    opções: --num-workers N | --log arquivo

QUICK: bancos de MENTIRA tirados da validação (os mesmos do teste QUICK da v4); nenhum áudio de teste é lido.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("v6", ROOT / "scripts" / "v6.py")
V6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V6)          # funções e utilidades da v6 (o main dele não roda)

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

from src.execucao import Tee, celula_config, executar_config  # noqa: E402

log = V6.log
ler_json = V6.ler_json
PRINCIPAIS_ROTULO = "3 principais"


def bancos(ns):
    from src.teste_v4 import BANCOS, BANCOS_QUICK
    return BANCOS_QUICK if ns["QUICK"] else BANCOS


def exigir_antes(ns):
    for e in ("e1", "e2", "e3", "e4", "e5", "finalistas"):
        if not ns["marcador"](e).exists():
            raise RuntimeError(f"a E6 só roda depois de E1 a E5 e dos finalistas: falta {e}")
    fi = ler_json(ns["arq"]("finalistas", "json"))
    if not fi:
        raise RuntimeError("results/v6_finalistas.json não existe")
    return fi


# ============================================================ extrair (WavLM, se preciso)
def etapa_extrair(a, ns, t0):
    from src.cache_v4 import Conjunto, Orcamento, limpar_temporarios, shard_pronto
    from src.extracao_v4 import extrair
    from src.ssl import carregar_ssl

    if V6.ja_feita(ns, "teste_extracao"):
        return 0
    fi = exigir_antes(ns)
    if not fi["precisa_extrair_teste_wavlm"]:
        V6.gravar_marcador(ns, "teste_extracao", t0, {
            "observacao": "não necessário: nenhum finalista usa o WavLM (números do WavLM no teste = os do colega, results/v5_*)",
            "extraido": False})
        log("teste do WavLM: não necessário (nenhum finalista usa o WavLM)\nSTATUS=CONCLUIDO")
        return 0
    import pandas as pd
    from src.arquivos import salvar_json, salvar_parquet
    hp = dict(ns["HP"])
    vz = ler_json(ns["arq"]("wavlm_vazao", "json"))
    if vz:
        hp["VAZAO_ESTIMADA"] = vz["clipes_por_s"]
    df, sha = V6.manifest_e_sha()
    base, destino = ns["CACHE_TESTE"]["xlsr"], ns["CACHE_TESTE"]["wavlm"]
    cfg_x = ler_json(base / "plano.json")
    if cfg_x["manifest_sha256"] != sha:
        raise RuntimeError("o manifest mudou desde o teste da v4")
    plano = pd.read_parquet(base / "plano.parquet")
    cfg = {"extracao": ns["CFG_EXTRACAO"], "manifest_sha256": sha, "copiado_de": str(base).replace("\\", "/"),
           "amostra": cfg_x.get("amostra"), "revisao_modelo": ns["REVISAO_WAVLM"]}
    if (destino / "plano.parquet").exists():
        if ler_json(destino / "plano.json") != json.loads(json.dumps(cfg, default=str)):
            raise RuntimeError(f"{destino}/plano.json é de outra configuração. Nada foi apagado.")
    else:
        destino.mkdir(parents=True, exist_ok=True)
        salvar_parquet(plano, destino / "plano.parquet")
        salvar_json(cfg, destino / "plano.json")
    n_tmp = limpar_temporarios(destino)
    if n_tmp:
        log(f"{n_tmp} shard(s) incompleto(s) descartados")
    log(f"teste do WavLM: {len(plano)} clipes (o plano do teste da v4)")
    orc = Orcamento(ns["arq"]("wavlm_teste_estado", "json"), ns["ORCAMENTO_TESTE_H"])
    nw = V6.num_workers(a, ns)
    r = {"parou_por": "orçamento esgotado antes de começar"}
    try:
        if orc.resta() > 0:
            model = carregar_ssl(ns["MODELOS"]["wavlm"], "cuda", hp["DTYPE"], ns["REVISAO_WAVLM"])
            r = extrair(df, plano, model, hp, destino, orc, ns["CFG_EXTRACAO"], log=log, num_workers=nw)
    finally:
        orc.fechar()
    feitos = {}
    for conj, g in plano.groupby("conjunto", sort=False):
        shards = g.shard.unique()
        prontos = [int(s) for s in shards if shard_pronto(destino, conj, s)]
        feitos[conj] = {"shards_prontos": len(prontos), "shards_planejados": int(len(shards))}
    if not all(v["shards_prontos"] == v["shards_planejados"] for v in feitos.values()):
        if orc.resta() <= 0 or r["parou_por"].startswith("orçamento"):
            raise RuntimeError(f"orçamento do teste do WavLM esgotado ({r['parou_por']})")
        log(f"extração incompleta ({r['parou_por']}); rode de novo.\nSTATUS=INCOMPLETO")
        return 2
    erros = {c: int(Conjunto(destino, c).meta.erro_leitura.fillna(0).astype(bool).sum()) for c in feitos}
    V6.gravar_marcador(ns, "teste_extracao", t0, {"observacao": f"extraído: {len(plano)} clipes, {orc.gasto() / 3600:.2f} h",
                                                  "extraido": True, "conjuntos": feitos, "erros_leitura": erros,
                                                  "horas": round(orc.gasto() / 3600, 3), "ultima_passada": r})
    log("teste_extracao CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ avaliar
def notas_finalista(M, f, alvos, ns):
    """Notas de um finalista em todos os `alvos` (recalculadas de forma determinística; NN: checkpoint)."""
    import numpy as np
    import torch
    spec = f["spec"]
    if spec["tipo"] == "rl":
        z, _ = V6.notas_rl_sistema(M, spec, alvos, ns["HP"])
        return z
    zs = [V6.treinar_nn_sistema(M, spec, spec["hp_nn"], s, alvos, torch.device("cuda"), ns, spec["ckpt_nome"])[0]
          for s in spec["seeds"]]
    return {k: np.mean([z[k] for z in zs], 0) for k in alvos}


def etapa_avaliar(a, ns, t0):
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve

    from src.arquivos import gravar_atomico, salvar_csv, salvar_json, salvar_parquet
    from src.teste_v4 import NOME_CURTO, bootstrap_banco, ic, metricas_banco, reamostras

    if V6.ja_feita(ns, "teste"):
        return 0
    fi = exigir_antes(ns)
    if not ns["marcador"]("teste_extracao").exists():
        raise RuntimeError("falta a etapa teste_extracao")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    hp = ns["HP"]
    BANC = bancos(ns)
    TIPO = dict(BANC)
    usa = fi["modelos_usados"]
    extras = {m: {b: (ns["CACHE_TESTE"][m], f"teste_{b}") for b, _ in BANC} for m in usa}
    M, rel = V6.carregar_modelos(ns, tuple(usa), extras=extras)
    m0 = usa[0]
    metas = {"val": M[m0].meta("val"), "dev_shift": M[m0].meta("dev_shift"), **{b: M[m0].meta(b) for b, _ in BANC}}
    excl = {b: int((~M[m0].ok[b]).sum()) for b, _ in BANC}
    log(" | ".join(f"{k} {len(v)}" for k, v in metas.items()) + f" | excluídos por erro de leitura: {excl}")
    if not ns["QUICK"]:      # conferência: os mesmos clipes por banco do teste da v4
        v4 = pd.read_csv(ns["RESULTS_DIR"] / "v4_teste_final.csv")
        v4 = v4[(v4.conjunto == "teste") & (v4.sistema == "S2") & (v4.limiar == "A") & (v4["métrica"] == "AUC")]
        for _, r in v4.iterrows():
            assert len(metas[r.banco]) + excl[r.banco] == r["n real"] + r["n fake"], f"{r.banco}: amostra diferente da v4"
    alvos = {"val": "val", "dev": "dev_shift", **{b: b for b, _ in BANC}}

    notas, checagem = {}, {}
    ref = {"F0": ("notas_e1", "z_v4ref"), "F1": ("notas_e4", "z_RL"), "F2": ("notas_e5", "z_final")}
    for k, f in fi["finalistas"].items():
        t = time.time()
        z = notas_finalista(M, f, alvos, ns)
        z["dev_shift"] = z.pop("dev")
        notas[k] = z
        arqs = ref[k]
        g = {c: pd.read_parquet(ns["arq"](f"{arqs[0]}_{c}", "parquet"))[arqs[1]].values for c in ("val", "dev_shift")}
        checagem[k] = {c: float(np.abs(g[c] - z[c]).max()) if len(g[c]) == len(z[c]) else None for c in g}
        checagem[k]["segundos"] = round(time.time() - t, 1)
        log(f"{k} ({f['descrição']}): dif. máx. contra as notas gravadas antes da E6: {checagem[k]}")

    # ---- limiares: os gravados em results/v6_finalistas.json (nenhum é recalculado aqui)
    lim = {k: f["limiares_logit"] for k, f in fi["finalistas"].items()}
    principal = {k: next(L for L in f["limiares_logit"] if L == f["limiar_principal"] or L.startswith(f"E5 ({f['limiar_principal']})"))
                 if f["limiar_principal"] != "A" else "A" for k, f in fi["finalistas"].items()}
    linhas, boots = [], {}
    S = list(fi["finalistas"])
    for b, m_ in metas.items():
        y = m_.label.values.astype(int)
        conjunto = {"val": "validação", "dev_shift": "dev_shift"}.get(b, "teste")
        tipo = {"val": "validação", "dev_shift": "decisão"}.get(b, TIPO.get(b))
        if conjunto == "teste":
            boots[b] = bootstrap_banco(y, {s: notas[s][b] for s in S}, lim, reamostras(y, hp["TESTE_N_BOOT"], hp["TESTE_SEED_BOOT"]))
        for s in S:
            for L, t_ in lim[s].items():
                mm = metricas_banco(y, notas[s][b], t_)
                for met in ("acurácia", "acurácia balanceada", "acurácia trivial", "AUC", "EER", "recall fake", "falso alarme real"):
                    lo = hi = np.nan
                    if conjunto == "teste":
                        ch = "AUC" if met == "AUC" else f"{met}|{L}"
                        if ch in boots[b][s]:
                            lo, hi = ic(boots[b][s][ch])
                    linhas.append({"sistema": s, "descrição": fi["finalistas"][s]["descrição"], "conjunto": conjunto, "banco": b,
                                   "tipo": tipo, "limiar": L, "limiar principal": L == principal[s], "limiar (logit)": t_,
                                   "métrica": met, "valor": mm[met], "IC 2,5%": lo, "IC 97,5%": hi,
                                   "n real": mm["n real"], "n fake": mm["n fake"]})
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, ns["arq"]("teste_final", "csv"))

    # ---- leitura pré-definida (F1 − F0, F2 − F0), cada sistema no seu limiar principal
    principais = [b for b, t_ in BANC if t_ == "principal"]
    dif, regra = [], {}
    for sb in ("F1", "F2"):
        for met in ("AUC", "acurácia balanceada"):
            d_ = []
            for b in [b for b, t_ in BANC if t_ != "só real"]:
                ka = "AUC" if met == "AUC" else f"{met}|{principal['F0']}"
                kb = "AUC" if met == "AUC" else f"{met}|{principal[sb]}"
                va = tab[(tab.sistema == "F0") & (tab.banco == b) & (tab["métrica"] == met) & (tab.limiar == principal["F0"])].valor.iloc[0]
                vb = tab[(tab.sistema == sb) & (tab.banco == b) & (tab["métrica"] == met) & (tab.limiar == principal[sb])].valor.iloc[0]
                lo, hi = ic(boots[b][sb][kb] - boots[b]["F0"][ka])
                linha = {"leitura": f"{sb} − F0", "métrica": met, "banco": b, "tipo": TIPO[b], "diferença": vb - va,
                         "IC 2,5%": lo, "IC 97,5%": hi, "IC exclui zero": bool(lo > 0 or hi < 0)}
                dif.append(linha)
                if b in principais:
                    d_.append(linha)
            sinais = {np.sign(r["diferença"]) for r in d_}
            ok = len(sinais) == 1 and 0 not in sinais and all(r["IC exclui zero"] for r in d_)
            regra[f"{sb} − F0 | {met}"] = ("relevante: " + ("melhor" if sinais == {1.0} else "pior")) if ok else "não demonstrada"
    tdif = pd.DataFrame(dif)
    salvar_csv(tdif, ns["arq"]("teste_diferencas", "csv"))
    log(tdif.round(4).to_string(index=False))
    log(json.dumps(regra, ensure_ascii=False, indent=1))

    # ---- resumo por finalista
    so_real = [b for b, t_ in BANC if t_ == "só real"]
    resumo = {}
    for s in S:
        for L in lim[s]:
            sel = tab[(tab.sistema == s) & (tab.limiar == L) & tab.banco.isin(principais)]
            fa = tab[(tab.sistema == s) & (tab.limiar == L) & tab.banco.isin(so_real) & (tab["métrica"] == "falso alarme real")]
            r = {"média acurácia (3 principais)": float(sel[sel["métrica"] == "acurácia"].valor.mean()),
                 "média acurácia balanceada (3 principais)": float(sel[sel["métrica"] == "acurácia balanceada"].valor.mean()),
                 "média acurácia trivial (3 principais)": float(sel[sel["métrica"] == "acurácia trivial"].valor.mean()),
                 "média AUC (3 principais)": float(sel[sel["métrica"] == "AUC"].valor.mean()),
                 **({f"{so_real[0]} falso alarme": float(fa.valor.iloc[0])} if len(fa) else {})}
            resumo[f"{s} | limiar {L}"] = r
    resumo_fin = {s: {**resumo[f"{s} | limiar {principal[s]}"], "limiar": principal[s]} for s in S}
    log(pd.DataFrame(resumo).T.round(4).to_string())

    # ---- números do WavLM no teste do colega (só se o WavLM não foi extraído aqui)
    colega = None
    if "wavlm" not in usa and not ns["QUICK"] and (ns["RESULTS_DIR"] / "v5_teste_final.csv").exists():
        c5 = pd.read_csv(ns["RESULTS_DIR"] / "v5_teste_final.csv")
        c5 = c5[(c5.conjunto == "teste") & c5.sistema.isin(["W1", "W2", "X2"]) & c5["métrica"].isin(
            ["AUC", "acurácia", "acurácia balanceada", "falso alarme real"])]
        colega = c5[["sistema", "descrição", "banco", "limiar", "métrica", "valor"]].assign(
            origem="do colega (v5), sem notas por clipe").to_dict("records")

    # ---- notas por clipe (fora do git)
    cols = ["idx_manifest", "path", "source_dataset", "label", "generator", "fake_type", "speaker_id", "duration_s"]
    partes = []
    for b, m_ in metas.items():
        p = m_[[c for c in cols if c in m_.columns]].copy()
        p.insert(0, "banco", b)
        for s in S:
            p[f"z_{s}"] = notas[s][b]
        partes.append(p)
    salvar_parquet(pd.concat(partes, ignore_index=True), ns["arq"]("teste_notas", "parquet"))

    # ---- (b) acurácia por banco dos finalistas, ao lado da trivial (limiar principal de cada um)
    cores = {"F0": "#9aa0a6", "F1": "#eb6834", "F2": "#1baf7a"}
    duas = [b for b, t_ in BANC if t_ != "só real"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=80, gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(duas))
    w = 0.8 / (len(S) + 1)
    for i, s in enumerate(S):
        g = tab[(tab.sistema == s) & (tab.limiar == principal[s]) & (tab["métrica"] == "acurácia") & tab.banco.isin(duas)].set_index("banco").loc[duas]
        axes[0].bar(x + (i - len(S) / 2) * w, g.valor, w * 0.92, color=cores.get(s), label=f"{s} (limiar {principal[s]})",
                    yerr=[g.valor - g["IC 2,5%"], g["IC 97,5%"] - g.valor], error_kw={"lw": 1, "capsize": 2})
    triv = tab[(tab.sistema == "F0") & (tab.limiar == "A") & (tab["métrica"] == "acurácia trivial") & tab.banco.isin(duas)].set_index("banco").loc[duas]
    axes[0].bar(x + (len(S) / 2) * w, triv.valor, w * 0.92, color="white", edgecolor="black", hatch="//", label="trivial (sempre a classe majoritária)")
    axes[0].set_xticks(x, [NOME_CURTO.get(b, b) + f"\n({TIPO[b]})" for b in duas], fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("acurácia")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].set_title("acurácia por banco (barras: IC95 do bootstrap, só da amostra de teste)", fontsize=9)
    if so_real:
        for i, s in enumerate(S):
            g = tab[(tab.sistema == s) & (tab.limiar == principal[s]) & (tab.banco == so_real[0]) & (tab["métrica"] == "falso alarme real")]
            axes[1].bar(i, 100 * g.valor.iloc[0], 0.7, color=cores.get(s))
            axes[1].text(i, 100 * g.valor.iloc[0] + 1, f"{100 * g.valor.iloc[0]:.1f}%", ha="center", fontsize=8)
        axes[1].set_xticks(range(len(S)), S)
        axes[1].set_ylabel("% de reais acusados")
        axes[1].set_title(f"{so_real[0]}: falso alarme (só reais)", fontsize=9)
    fig.suptitle("E6: finalistas no teste (avaliado uma vez; os bancos já tinham sido vistos na v4 e na v5)", fontsize=10)
    fig.tight_layout()
    gravar_atomico(ns["arq"]("teste_acuracia", "png"), lambda t_: fig.savefig(t_, format="png"))
    plt.close(fig)

    # ---- (c) curvas ROC e matrizes de confusão do sistema final (F2)
    fig, axes = plt.subplots(2, len(duas), figsize=(3.2 * len(duas), 6.4), dpi=75, squeeze=False)
    for j, b in enumerate(duas):
        y = metas[b].label.values.astype(int)
        for s in S:
            fpr, tpr, _ = roc_curve(y, notas[s][b])
            axes[0, j].plot(fpr, tpr, color=cores.get(s), lw=1.4 if s == "F2" else 0.9, label=s)
        axes[0, j].plot([0, 1], [0, 1], "k:", lw=0.8)
        axes[0, j].set_title(NOME_CURTO.get(b, b), fontsize=9)
        axes[0, j].set_xlabel("falso alarme", fontsize=8)
        if j == 0:
            axes[0, j].set_ylabel("fakes detectados", fontsize=8)
            axes[0, j].legend(fontsize=7)
        t_ = lim["F2"][principal["F2"]]
        pred = notas["F2"][b] >= t_
        cm = np.array([[(~pred[y == 0]).sum(), pred[y == 0].sum()], [(~pred[y == 1]).sum(), pred[y == 1].sum()]])
        axes[1, j].imshow(cm / cm.sum(1, keepdims=True), cmap="Greens", vmin=0, vmax=1)
        for i_ in range(2):
            for k_ in range(2):
                axes[1, j].text(k_, i_, f"{cm[i_, k_]}\n({100 * cm[i_, k_] / max(cm[i_].sum(), 1):.1f}%)", ha="center", va="center", fontsize=8)
        axes[1, j].set_xticks([0, 1], ["diz real", "diz fake"], fontsize=7)
        axes[1, j].set_yticks([0, 1], ["é real", "é fake"], fontsize=7)
    fig.suptitle(f"E6: curvas ROC (F0, F1, F2) e matriz de confusão do F2 (limiar {principal['F2']})", fontsize=10)
    fig.tight_layout()
    gravar_atomico(ns["arq"]("teste_roc_confusao", "png"), lambda t_: fig.savefig(t_, format="png"))
    plt.close(fig)

    # ---- (f) falso alarme e fake que passa, por banco, nos limiares A, B e E5 (F2)
    grupos = [b for b in metas]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), dpi=80)
    w = 0.8 / len(lim["F2"])
    for i, (L, t_) in enumerate(lim["F2"].items()):
        fa, fp = [], []
        for b in grupos:
            y, z = metas[b].label.values, notas["F2"][b]
            fa.append(100 * (z[y == 0] >= t_).mean() if (y == 0).any() else np.nan)
            fp.append(100 * (z[y == 1] < t_).mean() if (y == 1).any() else np.nan)
        xx = np.arange(len(grupos)) + (i - (len(lim["F2"]) - 1) / 2) * w
        axes[0].bar(xx, fa, w * 0.9, label=f"limiar {L} ({t_:+.2f})")
        axes[1].bar(xx, fp, w * 0.9, label=f"limiar {L} ({t_:+.2f})")
    for ax, tit in zip(axes, ("falso alarme (% dos reais acusados)", "fake que passa (% dos fakes aceitos)")):
        ax.set_xticks(np.arange(len(grupos)), [NOME_CURTO.get(g, g) for g in grupos], fontsize=7, rotation=20)
        ax.set_title(tit, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("E6: F2 por banco e por limiar (A = EER da validação; B = EER do dev_shift; E5 = o escolhido)", fontsize=10)
    fig.tight_layout()
    gravar_atomico(ns["arq"]("teste_limiares", "png"), lambda t_: fig.savefig(t_, format="png"))
    plt.close(fig)

    out = {"finalistas": {k: f["descrição"] for k, f in fi["finalistas"].items()}, "limiares_logit": lim,
           "limiar_principal": principal, "checagem_notas": checagem, "regra_leitura": regra,
           "resumo_finalistas": resumo_fin, "resumo_todos_limiares": resumo, "excluidos_erro_leitura": excl,
           "wavlm_no_teste": ("extraído nesta rodada" if "wavlm" in usa else "não extraído (nenhum finalista usa); números do colega"),
           "colega_v5_teste": colega, "n_boot": hp["TESTE_N_BOOT"], "bancos": BANC, "clipes": rel,
           "aviso": "os bancos de teste já tinham sido vistos na v4 e na v5; o bootstrap mede só a amostra de teste"}
    salvar_json(out, ns["arq"]("teste_resumo", "json"))
    V6.gravar_marcador(ns, "teste", t0, {"regra_leitura": regra, "resumo_finalistas": resumo_fin})
    log("teste CONCLUÍDO\nSTATUS=CONCLUIDO")
    return 0


ETAPAS_FN = {"extrair": etapa_extrair, "avaliar": etapa_avaliar}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=list(ETAPAS_FN))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--notebook", default="notebook.ipynb")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        arq = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, arq)
        sys.stderr = Tee(sys.__stderr__, arq)
    log(f"\n######## {datetime.now().isoformat(timespec='seconds')} | {' '.join(sys.argv)} | pid {os.getpid()}")
    from src.memoria import Amostrador
    V6.AMOSTRADOR = Amostrador()
    V6.AMOSTRADOR.start()
    t0 = time.time()
    V6.sem_suspensao(True)
    try:
        src_cfg = celula_config(a.notebook)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns = executar_config(src_cfg, a.quick, a.notebook)
        ns["_SRC_CONFIG"] = src_cfg
        if ns.get("VERSAO") != "v6":
            raise RuntimeError("a célula config do notebook não é a da v6")
        return ETAPAS_FN[a.etapa](a, ns, t0)
    except BaseException:
        traceback.print_exc()
        log("STATUS=FALHOU")
        return 1
    finally:
        V6.AMOSTRADOR.parar = True
        V6.sem_suspensao(False)
        log(f"PICO_MEMORIA_GB={V6.AMOSTRADOR.pico_proc / 1024:.2f} (este processo + workers) | pico do sistema "
            f"{V6.AMOSTRADOR.pico_sis:.1f} GB | {(time.time() - t0) / 60:.1f} min neste processo")


if __name__ == "__main__":
    sys.exit(main())
