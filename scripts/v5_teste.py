#!/usr/bin/env python3
"""Teste final da v5 (XLS-R × WavLM; avaliado UMA vez, só para reportar). Uma etapa por processo; retomável.

Pré-registro: docs/HISTORICO_VERSOES.md, seção "v5 - teste final: pré-registro" (escrito ANTES).
A decisão da v5 já foi tomada na validação + dev_shift (scripts/comparar_v5.py); este teste não a muda.
Reaproveita src/teste_v4.py (amostra, plano, métricas, bootstrap) e src/extracao_v4.py (extração).

    python scripts/v5_teste.py extrair  --modelo xlsr   [--quick]   # características da amostra de teste
    python scripts/v5_teste.py extrair  --modelo wavlm  [--quick]
    python scripts/v5_teste.py avaliar  [--quick]                   # X1, X2, W1, W2; métricas, bootstrap
    python scripts/v5_teste.py status   [--quick]                   # reescreve logs/v5_teste_resumo.txt
    python scripts/v5_teste.py concluir [--quick]                   # grava results/v5_teste_final_concluido.json
    opções: --num-workers N | --log arquivo

QUICK (--quick): usa os caches QUICK da v5 e bancos de MENTIRA tirados da validação (nunca do teste).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.execucao import Tee, celula_config, executar_config  # noqa: E402

# ---------------- constantes do pré-registro do teste final da v5 ----------------
TESTE_MAX_POR_CLASSE = 5_000     # a mesma regra da v4 (mesmos 51.320 clipes)
TESTE_N_BOOT = 1_000
TESTE_SEED_BOOT = 42
TESTE_ORCAMENTO_H = 3.0          # teto cumulativo da extração do teste, por modelo
QUICK_MAX_POR_CLASSE = 20
MODELOS = ("xlsr", "wavlm")
# sistema: (modelo, cabeça). As camadas e o C vêm dos arquivos da v5 (escolhas já feitas na validação).
SISTEMAS_V5 = {"X1": ("xlsr", "camada_24"), "X2": ("xlsr", "final"),
               "W1": ("wavlm", "camada_24"), "W2": ("wavlm", "final")}
DESCRICAO_V5 = {"X1": "XLS-R (Meta), só a camada 24", "X2": "XLS-R (Meta), cabeça final",
                "W1": "WavLM (Microsoft), só a camada 24", "W2": "WavLM (Microsoft), cabeça final"}
CORES_V5 = {"X1": "#f2a77f", "X2": "#eb6834", "W1": "#8fbbe8", "W2": "#2a78d6"}
PARES = [("W2", "X2", "WavLM − XLS-R, cabeça final (W2 − X2)"), ("W1", "X1", "WavLM − XLS-R, só camada 24 (W1 − X1)")]
ETAPAS_TESTE = ("extracao_xlsr", "extracao_wavlm", "avaliar")

AMOSTRADOR = None


def ler_json(p, padrao=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao


def caminhos(ns, modelo=None):
    suf, R = ns["SUFIXO"], ns["RESULTS_DIR"]
    C = {"suf": suf, "marcador": lambda e: R / f"v5_teste_{e}_concluido{suf}.json",
         "final": R / f"v5_teste_final_concluido{suf}.json",
         "res": lambda nome, ext: R / f"v5_teste_{nome}{suf}.{ext}",
         "resumo_txt": ns["LOGS_DIR"] / f"v5_teste_resumo{suf}.txt"}
    if modelo:
        P = f"v5_{modelo}"
        C.update({"cache": Path("data/processed") / f"{P}_cache{suf}",
                  "cache_teste": Path("data/processed") / f"{P}_cache_teste{suf}",
                  "estado": R / f"{P}_teste_estado{suf}.json",
                  "r": lambda nome, ext, P=P: R / f"{P}_{nome}{suf}.{ext}",
                  "parcial": R / f"{P}_parcial{suf}"})
    return C


def bancos(ns):
    from src.teste_v4 import BANCOS, BANCOS_QUICK
    return BANCOS_QUICK if ns["QUICK"] else BANCOS


def manifest_e_sha():
    from src.arquivos import sha256_arquivo
    from src.experimento import ler_manifest
    arq = ROOT / "data/processed/manifest.parquet"
    return ler_manifest(arq), sha256_arquivo(arq)


def amostra(ns, df):
    from src.cache_v4 import Conjunto
    from src.teste_v4 import amostra_teste_idx
    excluir = ()
    if ns["QUICK"]:     # os bancos de mentira não repetem clipes da validação QUICK
        excluir = Conjunto(caminhos(ns, "xlsr")["cache"], "val").meta.idx_manifest.values
    return amostra_teste_idx(df, QUICK_MAX_POR_CLASSE if ns["QUICK"] else TESTE_MAX_POR_CLASSE, ns["SEED"],
                             quick=ns["QUICK"], excluir=excluir)


def gravar_marcador(ns, C, etapa, t0, extra):
    from src.arquivos import identificar_codigo, salvar_json, sha256_texto
    codigo = identificar_codigo(ROOT, extras=["scripts/v5_teste.py", "src/teste_v4.py"])
    codigo["celula_config_sha256"] = sha256_texto(ns["_SRC_CONFIG"])
    info = {"etapa": etapa, "versao": "v5 (teste final)", "quick": ns["QUICK"], "status": "concluido",
            "concluido_em": datetime.now().isoformat(timespec="seconds"),
            "minutos_neste_processo": round((time.time() - t0) / 60, 2),
            "pico_memoria_gb_processo_e_workers": round(AMOSTRADOR.pico_proc / 1024, 2),
            "pico_memoria_gb_sistema": round(AMOSTRADOR.pico_sis, 2), "codigo": codigo, **extra}
    salvar_json(info, C["marcador"](etapa))
    return info


def sem_suspensao(ligar=True):
    """Impede o Windows de suspender durante a extração (como o rodar_v4.ps1)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if ligar else 0x80000000)
    except Exception:  # noqa: BLE001
        pass


# ============================================================ extrair (um modelo)
def etapa_extrair(a, ns, t0):
    from src.cache_v4 import Conjunto, Orcamento, limpar_temporarios, shard_pronto
    from src.extracao_v4 import extrair
    from src.ssl import carregar_ssl
    from src.teste_v4 import carregar_ou_criar_plano_teste

    m = a.modelo
    if not m or ns["MODELO_V5"] != m:
        raise RuntimeError("use --modelo xlsr ou --modelo wavlm")
    C, hp = caminhos(ns, m), dict(ns["HP"])
    etapa = f"extracao_{m}"
    if C["marcador"](etapa).exists():
        print(f"{etapa}: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    df, sha = manifest_e_sha()
    am = amostra(ns, df)
    cfg = {"extracao": ns["CFG_EXTRACAO"], "manifest_sha256": sha,
           "amostra": {"max_por_classe": QUICK_MAX_POR_CLASSE if ns["QUICK"] else TESTE_MAX_POR_CLASSE,
                       "seed": ns["SEED"], "bancos": [b for b, _ in bancos(ns)],
                       "regra": "src/teste_v4.py: amostra_teste_idx (split test, eval_subset test_final)"},
           "tam_shard": hp["TAM_SHARD"]}
    plano = carregar_ou_criar_plano_teste(am, hp["TAM_SHARD"], C["cache_teste"], cfg)
    n_tmp = limpar_temporarios(C["cache_teste"])
    if n_tmp:
        print(f"extração: {n_tmp} shard(s) incompleto(s) descartados", flush=True)
    print("plano:\n" + plano.groupby("conjunto", sort=False).agg(clipes=("idx_manifest", "size"),
                                                                 shards=("shard", "nunique")).to_string(), flush=True)
    print(f"total: {len(plano)} clipes | modelo {hp['MODELO_SSL']}", flush=True)
    orc = Orcamento(C["estado"], 0.5 if ns["QUICK"] else TESTE_ORCAMENTO_H)
    nw = a.num_workers if a.num_workers is not None else int(os.environ.get("V4_NUM_WORKERS", ns["NUM_WORKERS"]))
    r = {"parou_por": "orçamento já esgotado antes de começar", "shards_feitos": 0}
    sem_suspensao(True)
    try:
        if orc.resta() > 0:
            model = carregar_ssl(hp["MODELO_SSL"], "cuda", hp["DTYPE"])
            r = extrair(df, plano, model, hp, C["cache_teste"], orc, ns["CFG_EXTRACAO"],
                        log=lambda s: print(s, flush=True), num_workers=nw)
    finally:
        orc.fechar()
        sem_suspensao(False)
    feitos = {}
    for conj, g in plano.groupby("conjunto", sort=False):
        shards = g.shard.unique()
        prontos = [int(s) for s in shards if shard_pronto(C["cache_teste"], conj, s)]
        feitos[conj] = {"shards_prontos": len(prontos), "shards_planejados": int(len(shards)),
                        "clipes_prontos": int(g[g.shard.isin(prontos)].shape[0]), "clipes_planejados": int(len(g))}
    print("feito até agora:\n" + json.dumps(feitos, ensure_ascii=False, indent=1), flush=True)
    if not all(v["shards_prontos"] == v["shards_planejados"] for v in feitos.values()):
        print(f"extração incompleta ({r['parou_por']}); rode de novo.\nSTATUS=INCOMPLETO", flush=True)
        return 2
    erros = {conj: int(Conjunto(C["cache_teste"], conj).meta.erro_leitura.fillna(0).astype(bool).sum()) for conj in feitos}
    info = gravar_marcador(ns, C, etapa, t0, {
        "modelo": hp["MODELO_SSL"], "n_clipes": int(len(plano)), "conjuntos": feitos, "erros_leitura": erros,
        "ultima_passada": r, "horas_extracao_acumuladas": round(orc.gasto() / 3600, 3), "num_workers": nw})
    print(f"{etapa} CONCLUÍDA: {info['n_clipes']} clipes, {info['horas_extracao_acumuladas']:.2f} h; "
          f"erros de leitura {erros}\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ avaliar (os dois modelos)
def etapa_avaliar(a, ns, t0):
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from src.arquivos import gravar_atomico, salvar_csv, salvar_json, salvar_parquet
    from src.cache_v4 import Conjunto
    from src.teste_v4 import LIMIARES, bootstrap_banco, grafico, ic, limiares_de, metricas_banco, reamostras

    C0 = caminhos(ns)
    if C0["marcador"]("avaliar").exists():
        print("avaliar: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    for m in MODELOS:
        if not C0["marcador"](f"extracao_{m}").exists():
            raise RuntimeError(f"falta a extração do teste do {m} (python scripts/v5_teste.py extrair --modelo {m})")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    BANC = bancos(ns)
    TIPO = dict(BANC)
    hp = ns["HP"]

    def abrir(cache, nome):
        c = Conjunto(cache, nome)
        if len(c) == 0:
            raise RuntimeError(f"conjunto {nome} vazio em {cache}")
        c.ok = c.meta.erro_leitura.fillna(0).values == 0
        return c

    # ---- conjuntos de cada modelo; um clipe com erro de leitura em QUALQUER modelo sai dos dois
    conj = {}
    for m in MODELOS:
        Cm = caminhos(ns, m)
        conj[m] = {"treino": abrir(Cm["cache"], "treino"), "val": abrir(Cm["cache"], "val"),
                   "dev_shift": abrir(Cm["cache"], "dev_shift"),
                   **{b: abrir(Cm["cache_teste"], f"teste_{b}") for b, _ in BANC}}
    nomes = ["val", "dev_shift"] + [b for b, _ in BANC]
    for k in ["treino"] + nomes:
        ia, ib = (conj[m][k].meta.idx_manifest.values for m in MODELOS)
        assert (ia == ib).all(), f"{k}: os dois modelos não têm os mesmos clipes na mesma ordem"
        ok = conj["xlsr"][k].ok & conj["wavlm"][k].ok
        for m in MODELOS:
            conj[m][k].ok = ok
    assert conj["xlsr"]["treino"].ok.all(), "treino com erro de leitura"
    metas = {k: conj["xlsr"][k].meta[conj["xlsr"][k].ok].reset_index(drop=True) for k in nomes}
    excluidos = {k: int((~conj["xlsr"][k].ok).sum()) for k in nomes}
    print(" | ".join(f"{k} {len(v)}" for k, v in metas.items()) + f" | excluídos: {excluidos}", flush=True)
    if not ns["QUICK"]:      # conferência: os mesmos tamanhos por banco do teste da v4
        v4 = pd.read_csv(ns["RESULTS_DIR"] / "v4_teste_final.csv")
        v4 = v4[(v4.conjunto == "teste") & (v4.sistema == "S2") & (v4.limiar == "A") & (v4["métrica"] == "AUC")]
        for _, r in v4.iterrows():
            yb = metas[r.banco].label.values
            assert (int((yb == 0).sum()) + excluidos[r.banco] >= r["n real"]
                    and len(yb) + excluidos[r.banco] == r["n real"] + r["n fake"]), f"{r.banco}: amostra diferente da v4"

    # ---- reajuste de cada sistema (escolhas já feitas na validação; nada é reescolhido)
    notas = {k: {} for k in nomes}
    checagem = {}
    for s, (m, cab) in SISTEMAS_V5.items():
        Cm = caminhos(ns, m)
        z = np.load(Cm["parcial"] / "padronizacao.npz")
        mu, sd = z["mu"], z["sd"]
        tr = conj[m]["treino"]
        assert int(z["n"]) == len(tr)
        nome_cab = ler_json(Cm["r"]("resumo", "json"))["cabeça_final_pela_validação"] if cab == "final" else "rl_camada_24"
        if nome_cab.startswith("concat_"):
            info = [r for r in ler_json(Cm["r"]("camadas_concluido", "json"))["concat"] if r["cabeça"] == nome_cab][0]
            cams, c_ = info["camadas"], info["C"]
            ref = {k: pd.read_parquet(Cm["r"](f"notas_{nome_cab}_{k}", "parquet")).z.values for k in ("val", "dev_shift")}
        elif nome_cab.startswith("rl_camada_"):      # uma camada só (o "só camada 24"; no QUICK, às vezes a final)
            l = int(nome_cab.rsplit("_", 1)[1])
            cams = [l]
            tabc = pd.read_csv(Cm["r"]("camadas", "csv"))
            c_ = float(tabc.loc[tabc.camada == l, "C"].iloc[0])
            ref = {k: pd.read_parquet(Cm["r"](f"notas_camadas_{k}", "parquet"))[f"z_L{l:02d}"].values
                   for k in ("val", "dev_shift")}
        else:
            raise RuntimeError(f"{m}: cabeça final {nome_cab} (soma ponderada) não prevista no pré-registro do teste")

        def X(c, cams=cams, mu=mu, sd=sd):
            return np.hstack([(c.camada(l)[c.ok] - mu[l]) / sd[l] for l in cams])
        t = time.time()
        rl = LogisticRegression(C=c_, max_iter=hp["RL_MAX_ITER"], tol=1e-4).fit(X(tr), tr.meta.label.values.astype(int))
        for k in nomes:
            notas[k][s] = rl.decision_function(X(conj[m][k]))
        def dif(k):       # conferência contra as notas gravadas pela v5 (só se nenhum clipe saiu por erro)
            return float(np.abs(notas[k][s] - ref[k]).max()) if len(ref[k]) == len(notas[k][s]) else None
        checagem[s] = {"modelo": m, "camadas": cams, "C": c_, "iterações": int(rl.n_iter_[0]),
                       "dif_max_val_vs_gravado": dif("val"), "dif_max_dev_vs_gravado": dif("dev_shift"),
                       "segundos": round(time.time() - t, 1)}
        print(f"{s} ({DESCRICAO_V5[s]}): camadas {cams} C={c_:g} | dif. máx. contra as notas gravadas: "
              f"val {checagem[s]['dif_max_val_vs_gravado']} dev {checagem[s]['dif_max_dev_vs_gravado']}", flush=True)
        del rl

    # ---- limiares, métricas e bootstrap (pareado: as mesmas reamostragens para os 4 sistemas)
    S = list(SISTEMAS_V5)
    yv, ydev = metas["val"].label.values.astype(int), metas["dev_shift"].label.values.astype(int)
    lim = {s: limiares_de(notas["val"][s], yv, notas["dev_shift"][s], ydev) for s in S}
    print("limiares (logit):", json.dumps(lim), flush=True)
    linhas, boots = [], {}
    for k, m_ in metas.items():
        y = m_.label.values.astype(int)
        conjunto = {"val": "validação", "dev_shift": "dev_shift"}.get(k, "teste")
        tipo = {"val": "validação", "dev_shift": "decisão entre versões"}.get(k, TIPO.get(k))
        banco = "in_the_wild (dev_shift)" if k == "dev_shift" else ("validação (todas)" if k == "val" else k)
        if conjunto == "teste":
            boots[k] = bootstrap_banco(y, {s: notas[k][s] for s in S}, lim, reamostras(y, TESTE_N_BOOT, TESTE_SEED_BOOT))
        for s in S:
            for L, t_ in lim[s].items():
                mm = metricas_banco(y, notas[k][s], t_)
                for met in ("acurácia", "acurácia balanceada", "acurácia trivial", "AUC", "EER", "recall fake",
                            "falso alarme real"):
                    lo = hi = np.nan
                    if conjunto == "teste":
                        chave = "AUC" if met == "AUC" else f"{met}|{L}"
                        if chave in boots[k][s]:
                            lo, hi = ic(boots[k][s][chave])
                    linhas.append({"sistema": s, "descrição": DESCRICAO_V5[s], "conjunto": conjunto, "banco": banco,
                                   "tipo": tipo, "limiar": L, "limiar (como)": LIMIARES[L], "limiar (logit)": t_,
                                   "métrica": met, "valor": mm[met], "IC 2,5%": lo, "IC 97,5%": hi,
                                   "n real": mm["n real"], "n fake": mm["n fake"], "excluídos (erro de leitura)": excluidos[k]})
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, C0["res"]("final", "csv"))

    # ---- leitura pré-definida: WavLM − XLS-R
    principais = [b for b, t_ in BANC if t_ == "principal"]
    dif, regra = [], {}
    for sa, sb, nome in PARES:
        for met in ("AUC", "acurácia balanceada|A", "acurácia balanceada|B"):
            for k in [b for b, t_ in BANC if t_ != "só real"]:
                m0, L = met.split("|")[0], (met.split("|")[1] if "|" in met else "A")
                va_ = tab[(tab.sistema == sa) & (tab.banco == k) & (tab["métrica"] == m0) & (tab.limiar == L)].valor.iloc[0]
                vb_ = tab[(tab.sistema == sb) & (tab.banco == k) & (tab["métrica"] == m0) & (tab.limiar == L)].valor.iloc[0]
                lo, hi = ic(boots[k][sa][met] - boots[k][sb][met])
                dif.append({"leitura": nome, "métrica": met.replace("|", ", limiar "), "banco": k, "tipo": TIPO[k],
                            "diferença": va_ - vb_, "IC 2,5%": lo, "IC 97,5%": hi, "IC exclui zero": bool(lo > 0 or hi < 0)})
            if met != "acurácia balanceada|B":
                d = [r for r in dif if r["leitura"] == nome and r["métrica"] == met.replace("|", ", limiar ")
                     and r["banco"] in principais]
                sinais = {np.sign(r["diferença"]) for r in d}
                ok = len(sinais) == 1 and 0 not in sinais and all(r["IC exclui zero"] for r in d)
                regra[f"{nome} | {met.replace('|', ', limiar ')}"] = (
                    ("relevante: " + ("WavLM melhor" if sinais == {1.0} else "WavLM pior")) if ok else "não demonstrada")
    tdif = pd.DataFrame(dif)
    salvar_csv(tdif, C0["res"]("diferencas", "csv"))
    print(tdif.round(4).to_string(index=False), flush=True)
    print(json.dumps(regra, ensure_ascii=False, indent=1), flush=True)

    # ---- resumo e conferência com o S2 da v4
    so_real = [b for b, t_ in BANC if t_ == "só real"]
    resumo = []
    for s in S:
        for L in ("A", "B"):
            sel = tab[(tab.sistema == s) & (tab.limiar == L) & tab.banco.isin(principais)]
            fa = tab[(tab.sistema == s) & (tab.limiar == L) & tab.banco.isin(so_real) & (tab["métrica"] == "falso alarme real")]
            resumo.append({"sistema": s, "limiar": L,
                           "média acurácia (3 principais)": float(sel[sel["métrica"] == "acurácia"].valor.mean()),
                           "média acurácia balanceada (3 principais)": float(sel[sel["métrica"] == "acurácia balanceada"].valor.mean()),
                           "média acurácia trivial (3 principais)": float(sel[sel["métrica"] == "acurácia trivial"].valor.mean()),
                           "média AUC (3 principais)": float(sel[sel["métrica"] == "AUC"].valor.mean()),
                           **({f"{so_real[0]} falso alarme": float(fa.valor.iloc[0])} if len(fa) else {})})
    tres = pd.DataFrame(resumo)
    print(tres.round(4).to_string(index=False), flush=True)
    repro = None
    if not ns["QUICK"]:
        v4 = pd.read_csv(ns["RESULTS_DIR"] / "v4_teste_final.csv")
        a4 = v4[(v4.conjunto == "teste") & (v4.sistema == "S2") & (v4.limiar == "A") & (v4["métrica"] == "AUC")].set_index("banco").valor
        a5 = tab[(tab.conjunto == "teste") & (tab.sistema == "X2") & (tab.limiar == "A") & (tab["métrica"] == "AUC")].set_index("banco").valor
        repro = {b: {"AUC v4 S2": float(a4[b]), "AUC v5 X2": float(a5[b]), "dif": float(a5[b] - a4[b])}
                 for b in a4.index if np.isfinite(a4[b])}
        print("reprodução (X2 da v5 × S2 da v4):", json.dumps(repro, ensure_ascii=False), flush=True)

    # ---- notas por clipe (fora do git), gráfico e resumo
    cols = ["idx_manifest", "path", "source_dataset", "label", "generator", "fake_type", "speaker_id", "duration_s"]
    partes = []
    for k, m_ in metas.items():
        p = m_[cols].copy()
        p.insert(0, "conjunto", {"val": "val", "dev_shift": "dev_shift"}.get(k, "teste"))
        p.insert(1, "banco", k)
        for s in S:
            p[f"z_{s}"] = notas[k][s]
        partes.append(p)
    salvar_parquet(pd.concat(partes, ignore_index=True), C0["res"]("notas", "parquet"))
    fig = grafico(tab, BANC, sistemas=S, descricao=DESCRICAO_V5, cores=CORES_V5,
                  titulo="Teste final da v5: XLS-R × WavLM (barras de erro: intervalo bootstrap de 95%, só da amostra de teste)")
    gravar_atomico(C0["res"]("final", "png"), lambda t_: fig.savefig(t_, format="png"))
    plt.close(fig)
    out = {"sistemas": DESCRICAO_V5, "limiares_logit": lim, "checagem": checagem, "regra_leitura": regra,
           "resumo": tres.to_dict("records"), "reproducao_v4": repro, "excluidos_erro_leitura": excluidos,
           "n_boot": TESTE_N_BOOT, "bancos": BANC}
    salvar_json(out, C0["res"]("final_resumo", "json"))
    gravar_marcador(ns, C0, "avaliar", t0, {"regra_leitura": regra, "resumo": tres.to_dict("records"),
                                            "reproducao_v4": repro})
    print("avaliar CONCLUÍDA\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ status e concluir
def etapa_status(a, ns, t0):
    import pandas as pd

    from src.arquivos import gravar_atomico
    C = caminhos(ns)
    L = [f"Teste final v5{' (QUICK)' if ns['QUICK'] else ''} (atualizado em {datetime.now():%Y-%m-%d %H:%M:%S})",
         "Avaliado UMA vez, só para reportar; a decisão da v5 já foi tomada na validação + dev_shift.", ""]
    for e in ETAPAS_TESTE:
        m = ler_json(C["marcador"](e))
        L.append(f"{e:<15} " + (f"concluída ({m['minutos_neste_processo']:.1f} min)" if m else "pendente"))
    arq = C["res"]("final", "csv")
    if C["marcador"]("avaliar").exists() and arq.exists():
        t = pd.read_csv(arq)
        t = t[t.conjunto == "teste"]
        for L_ in ("A", "B"):
            p = t[(t.limiar == L_) & t["métrica"].isin(["acurácia balanceada", "AUC", "EER", "falso alarme real"])]
            pv = p.pivot_table(index="banco", columns=["métrica", "sistema"], values="valor", sort=False).round(3)
            L += ["", f"limiar {L_}:", pv.to_string()]
        m = ler_json(C["marcador"]("avaliar"))
        L += ["", "resumo:", pd.DataFrame(m["resumo"]).round(3).to_string(index=False), "",
              "leitura pré-definida:"] + [f"  {k}: {v}" for k, v in m["regra_leitura"].items()]
    L += ["", f"concluído: {'SIM' if C['final'].exists() else 'não'} ({C['final']})"]
    txt = "\n".join(L) + "\n"
    gravar_atomico(C["resumo_txt"], lambda t_: Path(t_).write_text(txt, encoding="utf-8-sig"))
    print(txt, flush=True)
    return 0


def etapa_concluir(a, ns, t0):
    from src.arquivos import salvar_json
    C = caminhos(ns)
    faltam = [e for e in ETAPAS_TESTE if not C["marcador"](e).exists()]
    if faltam:
        print(f"ainda faltam: {faltam}\nSTATUS=INCOMPLETO", flush=True)
        return 3
    if not C["final"].exists():
        ms = {e: ler_json(C["marcador"](e)) for e in ETAPAS_TESTE}
        salvar_json({"versao": "v5", "etapa": "teste final", "quick": ns["QUICK"],
                     "concluido_em": datetime.now().isoformat(timespec="seconds"),
                     "etapas": {e: {"concluido_em": m["concluido_em"], "minutos": m["minutos_neste_processo"],
                                    "pico_memoria_gb": m["pico_memoria_gb_processo_e_workers"]} for e, m in ms.items()},
                     "regra_leitura": ms["avaliar"]["regra_leitura"], "resumo": ms["avaliar"]["resumo"]}, C["final"])
    print(f"teste final da v5 CONCLUÍDO ({C['final']})\nSTATUS=CONCLUIDO", flush=True)
    return 0


ETAPAS_FN = {"extrair": etapa_extrair, "avaliar": etapa_avaliar, "status": etapa_status, "concluir": etapa_concluir}


def main():
    global AMOSTRADOR
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=list(ETAPAS_FN))
    ap.add_argument("--modelo", choices=MODELOS, default=None, help="extrair: de qual modelo")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--notebook", default="notebook.ipynb")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    if a.modelo:
        os.environ["V5_MODELO"] = a.modelo          # a célula config escolhe o modelo por esta variável
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        arq = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, arq)
        sys.stderr = Tee(sys.__stderr__, arq)
    leve = a.etapa in ("status", "concluir")
    if not leve:
        print(f"\n######## {datetime.now().isoformat(timespec='seconds')} | {' '.join(sys.argv)} | pid {os.getpid()}", flush=True)
    from src.memoria import Amostrador
    AMOSTRADOR = Amostrador()
    if not leve:
        AMOSTRADOR.start()
    t0 = time.time()
    try:
        src_cfg = celula_config(a.notebook)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns = executar_config(src_cfg, a.quick, a.notebook)
        ns["_SRC_CONFIG"] = src_cfg
        if ns.get("VERSAO") != "v5":
            raise RuntimeError("a célula config do notebook não é a da v5")
        return ETAPAS_FN[a.etapa](a, ns, t0)
    except BaseException:
        traceback.print_exc()
        print("STATUS=FALHOU", flush=True)
        return 1
    finally:
        AMOSTRADOR.parar = True
        if not leve:
            print(f"PICO_MEMORIA_GB={AMOSTRADOR.pico_proc / 1024:.2f} (este processo + workers) | pico do sistema "
                  f"{AMOSTRADOR.pico_sis:.1f} GB | {(time.time() - t0) / 60:.1f} min neste processo", flush=True)


if __name__ == "__main__":
    sys.exit(main())
