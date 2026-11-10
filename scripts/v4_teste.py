#!/usr/bin/env python3
"""Teste final da v4 (avaliado UMA vez). Uma etapa por processo; retomável e atômico.

Pré-registro: docs/HISTORICO_VERSOES.md, seção "v4 - teste final: pré-registro" (escrito ANTES).
Os hiperparâmetros da v4 vêm da célula "config" do notebook.ipynb (não são copiados aqui); as
constantes do teste final, abaixo, são as do pré-registro.

    python scripts/v4_teste.py extrair  [--quick]   # características XLS-R da amostra de teste (shards, retomável)
    python scripts/v4_teste.py cnn      [--quick]   # notas do S0 (CNN da v3) na validação, dev_shift e teste
    python scripts/v4_teste.py avaliar  [--quick]   # S1, S2 (reajuste), S3; métricas, bootstrap, tabela, gráfico
    python scripts/v4_teste.py status   [--quick]   # reescreve logs/v4_teste_resumo.txt (leve)
    python scripts/v4_teste.py concluir [--quick]   # grava results/v4_teste_final_concluido.json (por último)
    opções: --num-workers N | --log arquivo

QUICK (--quick, com V4_SUFIXO_EXTRA=_t2): usa o cache QUICK da v4 e bancos de MENTIRA tirados da
validação (nunca do teste), só para testar o código. Os arquivos levam o sufixo do QUICK.
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

# ---------------- constantes do pré-registro do teste final ----------------
TESTE_MAX_POR_CLASSE = 5_000     # até 5.000 clipes por classe e banco (seed 42), a regra das versões anteriores
TESTE_N_BOOT = 1_000             # reamostragens do bootstrap por banco
TESTE_SEED_BOOT = 42
TESTE_ORCAMENTO_H = 3.0          # teto cumulativo da extração do teste (~51 mil clipes; estimativa ~25 min)
CAMADA_S1 = 24                   # S1: só a última camada
S3_C = 1.0                       # S3: regressão de 2 entradas (C padrão; 2 entradas e 20 mil clipes)
CABECA_S2 = "concat_top3"        # S2: a cabeça final da v4, escolhida pela validação (não é reescolhida)
QUICK_MAX_POR_CLASSE = 20
ETAPAS_TESTE = ("extracao", "cnn", "avaliar")

AMOSTRADOR = None


def ler_json(p, padrao=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao


def caminhos(ns):
    suf = ns["SUFIXO"]
    R = ns["RESULTS_DIR"]
    return {"suf": suf, "cache": Path("data/processed") / f"v4_cache_teste{suf}",
            "estado": R / f"v4_teste_estado{suf}.json",
            "marcador": lambda e: R / f"v4_teste_{e}_concluido{suf}.json",
            "final": R / f"v4_teste_final_concluido{suf}.json",
            "res": lambda nome, ext: R / f"v4_teste_{nome}{suf}.{ext}",
            "resumo_txt": ns["LOGS_DIR"] / f"v4_teste_resumo{suf}.txt"}


def bancos(ns):
    from src.teste_v4 import BANCOS, BANCOS_QUICK
    return BANCOS_QUICK if ns["QUICK"] else BANCOS


def max_por_classe(ns):
    return QUICK_MAX_POR_CLASSE if ns["QUICK"] else TESTE_MAX_POR_CLASSE


def manifest_e_sha():
    from src.arquivos import sha256_arquivo
    from src.experimento import ler_manifest
    arq = ROOT / "data/processed/manifest.parquet"
    return ler_manifest(arq), sha256_arquivo(arq)


def amostra(ns, df):
    from src.cache_v4 import Conjunto
    from src.teste_v4 import amostra_teste_idx
    excluir = ()
    if ns["QUICK"]:     # os bancos de mentira não podem repetir clipes da validação QUICK (onde o S3 treina)
        excluir = Conjunto(ns["CACHE_DIR"], "val").meta.idx_manifest.values
    return amostra_teste_idx(df, max_por_classe(ns), ns["SEED"], quick=ns["QUICK"], excluir=excluir)


def gravar_marcador(ns, C, etapa, t0, extra):
    from src.arquivos import identificar_codigo, salvar_json, sha256_texto
    codigo = identificar_codigo(ROOT, extras=["scripts/v4_teste.py", "scripts/rodar_v4_teste.ps1",
                                              "scripts/vigia_v4_teste.ps1"])
    codigo["celula_config_sha256"] = sha256_texto(ns["_SRC_CONFIG"])
    info = {"etapa": etapa, "versao": "v4 (teste final)", "quick": ns["QUICK"], "status": "concluido",
            "concluido_em": datetime.now().isoformat(timespec="seconds"),
            "minutos_neste_processo": round((time.time() - t0) / 60, 2),
            "pico_memoria_gb_processo_e_workers": round(AMOSTRADOR.pico_proc / 1024, 2),
            "pico_memoria_gb_sistema": round(AMOSTRADOR.pico_sis, 2), "codigo": codigo, **extra}
    salvar_json(info, C["marcador"](etapa))
    return info


def exigir(C, etapa):
    if not C["marcador"](etapa).exists():
        raise RuntimeError(f"a etapa '{etapa}' ainda não terminou ({C['marcador'](etapa)} não existe)")


def num_workers(a, ns):
    if a.num_workers is not None:
        return a.num_workers
    return int(os.environ.get("V4_NUM_WORKERS", ns["NUM_WORKERS"]))


# ============================================================ extrair
def etapa_extrair(a, ns, t0):
    from src.cache_v4 import Orcamento, limpar_temporarios, shard_pronto
    from src.extracao_v4 import extrair
    from src.ssl import carregar_ssl
    from src.teste_v4 import carregar_ou_criar_plano_teste

    C, hp = caminhos(ns), dict(ns["HP"])
    if C["marcador"]("extracao").exists():
        print("extração do teste: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    df, sha = manifest_e_sha()
    am = amostra(ns, df)
    cfg = {"extracao": ns["CFG_EXTRACAO"], "manifest_sha256": sha,
           "amostra": {"max_por_classe": max_por_classe(ns), "seed": ns["SEED"], "bancos": [b for b, _ in bancos(ns)],
                       "regra": "src/experimento.py: amostra_teste (split test, eval_subset test_final)"},
           "tam_shard": hp["TAM_SHARD"]}
    plano = carregar_ou_criar_plano_teste(am, hp["TAM_SHARD"], C["cache"], cfg)
    n_tmp = limpar_temporarios(C["cache"])
    if n_tmp:
        print(f"extração: {n_tmp} shard(s) incompleto(s) de uma execução interrompida foram descartados", flush=True)
    print("plano:\n" + plano.groupby("conjunto", sort=False).agg(clipes=("idx_manifest", "size"),
                                                                 shards=("shard", "nunique")).to_string(), flush=True)
    print(f"total: {len(plano)} clipes", flush=True)
    orc = Orcamento(C["estado"], 0.5 if ns["QUICK"] else TESTE_ORCAMENTO_H)
    nw = num_workers(a, ns)
    r = {"parou_por": "orçamento já esgotado antes de começar", "shards_feitos": 0}
    try:
        if orc.resta() > 0:
            model = carregar_ssl(hp["MODELO_SSL"], "cuda", hp["DTYPE"])
            r = extrair(df, plano, model, hp, C["cache"], orc, ns["CFG_EXTRACAO"],
                        log=lambda s: print(s, flush=True), num_workers=nw)
    finally:
        orc.fechar()
    feitos = {}
    for conj, g in plano.groupby("conjunto", sort=False):
        shards = g.shard.unique()
        prontos = [int(s) for s in shards if shard_pronto(C["cache"], conj, s)]
        feitos[conj] = {"shards_prontos": len(prontos), "shards_planejados": int(len(shards)),
                        "clipes_prontos": int(g[g.shard.isin(prontos)].shape[0]), "clipes_planejados": int(len(g))}
    print("feito até agora:\n" + json.dumps(feitos, ensure_ascii=False, indent=1), flush=True)
    if not all(v["shards_prontos"] == v["shards_planejados"] for v in feitos.values()):
        print(f"extração incompleta ({r['parou_por']}); rode de novo.\nSTATUS=INCOMPLETO", flush=True)
        return 2
    from src.cache_v4 import Conjunto
    erros = {conj: int(Conjunto(C["cache"], conj).meta.erro_leitura.fillna(0).astype(bool).sum()) for conj in feitos}
    info = gravar_marcador(ns, C, "extracao", t0, {
        "n_clipes": int(len(plano)), "conjuntos": feitos, "erros_leitura": erros, "ultima_passada": r,
        "horas_extracao_acumuladas": round(orc.gasto() / 3600, 3), "num_workers": nw, "extracao": ns["CFG_EXTRACAO"]})
    print(f"extração do teste CONCLUÍDA: {info['n_clipes']} clipes, {info['horas_extracao_acumuladas']:.2f} h; "
          f"erros de leitura {erros}\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ cnn (S0)
def etapa_cnn(a, ns, t0):
    import numpy as np
    import pandas as pd
    import torch

    from src.arquivos import salvar_json, salvar_parquet
    from src.cache_v4 import Conjunto
    from src.dataset import limitar_threads_dos_workers
    from src.experimento import calcular_notas, carregar_modelo
    from src.model import LogMelPadronizado
    from src.teste_v4 import escolher_s0

    C = caminhos(ns)
    if C["marcador"]("cnn").exists():
        print("cnn: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    exigir(C, "extracao")          # a mesma amostra do teste já está fixada no plano
    limitar_threads_dos_workers()
    suf_v3 = "_quick" if ns["QUICK"] else ""
    nome, tab = escolher_s0(ns["RESULTS_DIR"], suf_v3)
    mk = ler_json(ns["RESULTS_DIR"] / f"{nome}_concluido{suf_v3}.json")
    print("S0: checkpoints da v3 pela EER média por fonte na validação:\n" + tab.to_string(index=False), flush=True)
    print(f"S0 = {nome} ({mk['arquivos']['checkpoint']})", flush=True)
    device = torch.device("cuda")
    model, ck, modo = carregar_modelo(mk["arquivos"]["checkpoint"], device)
    hcnn = mk["metodo"]["hp"]
    logmel = LogMelPadronizado(hcnn["SAMPLE_RATE"], hcnn["N_FFT"], hcnn["WIN_LENGTH"], hcnn["HOP_LENGTH"],
                               hcnn["N_MELS"]).to(device)
    df, _ = manifest_e_sha()
    plano = pd.read_parquet(C["cache"] / "plano.parquet")
    conjuntos = {"val": Conjunto(ns["CACHE_DIR"], "val").meta.idx_manifest.values,
                 "dev_shift": Conjunto(ns["CACHE_DIR"], "dev_shift").meta.idx_manifest.values,
                 "teste": plano.idx_manifest.values}
    nw = num_workers(a, ns)
    tempos, confere = {}, {}
    for conj, idx in conjuntos.items():
        arq = C["res"](f"notas_s0_{conj}", "parquet")
        if arq.exists() and pd.read_parquet(arq).idx_manifest.tolist() == list(idx):
            print(f"{conj}: notas do S0 já calculadas ({arq})", flush=True)
            continue
        t = time.time()
        am = df.loc[idx]
        out = calcular_notas(model, logmel, am, hcnn, modo, device, num_workers=nw)
        out.insert(0, "idx_manifest", np.asarray(idx))
        salvar_parquet(out, arq)
        tempos[conj] = round((time.time() - t) / 60, 2)
        print(f"{conj}: {len(out)} áudios em {tempos[conj]:.1f} min -> {arq}", flush=True)
    for conj in ("val", "dev_shift"):          # conferência contra as notas gravadas pela v3 (mesmo modelo)
        p = ns["RESULTS_DIR"] / f"{nome}_notas_{conj}{suf_v3}.parquet"
        if p.exists():
            novo = pd.read_parquet(C["res"](f"notas_s0_{conj}", "parquet"))
            j = novo.merge(pd.read_parquet(p)[["path", "z_janelas"]], on="path", suffixes=("", "_v3"))
            confere[conj] = {"n_em_comum": int(len(j)),
                             "dif_max_logit": float((j.z_janelas - j.z_janelas_v3).abs().max()) if len(j) else None}
            print(f"conferência {conj}: {confere[conj]}", flush=True)
    gravar_marcador(ns, C, "cnn", t0, {"s0": nome, "s0_checkpoint": mk["arquivos"]["checkpoint"],
                                       "s0_short_audio_mode": modo, "s0_tabela_escolha": tab.to_dict("records"),
                                       "minutos": tempos, "conferencia_v3": confere, "num_workers": nw})
    salvar_json(tab.to_dict("records"), C["res"]("escolha_s0", "json"))
    print("cnn CONCLUÍDA\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ avaliar
def etapa_avaliar(a, ns, t0):
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from src.arquivos import gravar_atomico, salvar_csv, salvar_json, salvar_parquet
    from src.cabecas import ajustar_rl
    from src.cache_v4 import Conjunto
    from src.teste_v4 import (DESCRICAO, LIMIARES, SISTEMAS, bootstrap_banco, grafico, ic, limiares_de,
                              metricas_banco, reamostras)

    C, hp = caminhos(ns), ns["HP"]
    if C["marcador"]("avaliar").exists():
        print("avaliar: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    exigir(C, "extracao")
    exigir(C, "cnn")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    BANC = bancos(ns)
    TIPO = dict(BANC)
    suf = ns["SUFIXO"]
    R = ns["RESULTS_DIR"]

    # ---- conjuntos (só clipes lidos sem erro)
    def abrir(cache, nome):
        c = Conjunto(cache, nome)
        if len(c) == 0:
            raise RuntimeError(f"conjunto {nome} vazio em {cache}")
        c.ok = c.meta.erro_leitura.fillna(0).values == 0
        return c
    tr, va, dv = (abrir(ns["CACHE_DIR"], n) for n in ("treino", "val", "dev_shift"))
    te = {b: abrir(C["cache"], f"teste_{b}") for b, _ in BANC}
    conjs = {"val": va, "dev_shift": dv, **te}
    metas = {k: c.meta[c.ok].reset_index(drop=True) for k, c in conjs.items()}
    excluidos = {k: int((~c.ok).sum()) for k, c in conjs.items()}
    print(f"treino {len(tr)} | " + " | ".join(f"{k} {len(m)}" for k, m in metas.items())
          + f" | excluídos por erro de leitura: {excluidos}", flush=True)
    z = np.load(R / f"v4_parcial{suf}" / "padronizacao.npz")
    mu, sd = z["mu"], z["sd"]
    assert int(z["n"]) == len(tr) and tr.ok.all()
    ytr = tr.meta.label.values.astype(int)

    def X(c, cams):
        return np.hstack([(c.camada(l)[c.ok] - mu[l]) / sd[l] for l in cams])

    notas = {k: {} for k in conjs}
    checagem = {}

    # ---- S1: só a camada 24, C pela validação (EER média por fonte)
    t = time.time()
    rl1, c1, grade1 = ajustar_rl(X(tr, [CAMADA_S1]), ytr, X(va, [CAMADA_S1]), metas["val"], hp["GRADE_C"],
                                 hp["RL_MAX_ITER"], log=lambda s: print(s, flush=True))
    for k, c in conjs.items():
        notas[k]["S1"] = rl1.decision_function(X(c, [CAMADA_S1]))
    ref = ler_json(R / f"v4_parcial{suf}" / f"camada_{CAMADA_S1:02d}.json")
    nv = pd.read_parquet(ns["arq_result"]("notas_camadas_val", "parquet"))
    nd = pd.read_parquet(ns["arq_result"]("notas_camadas_dev_shift", "parquet"))
    checagem["S1"] = {"C": c1, "C_na_v4": ref["C"], "grade": grade1.to_dict("records"),
                      "dif_max_val_vs_v4": float(np.abs(notas["val"]["S1"] - nv[f"z_L{CAMADA_S1:02d}"].values).max()),
                      "dif_max_dev_vs_v4": float(np.abs(notas["dev_shift"]["S1"] - nd[f"z_L{CAMADA_S1:02d}"].values).max()),
                      "segundos": round(time.time() - t, 1)}
    print(f"S1: C={c1:g} (v4: {ref['C']:g}) | dif. máx. val {checagem['S1']['dif_max_val_vs_v4']:.2e} "
          f"dev {checagem['S1']['dif_max_dev_vs_v4']:.2e}", flush=True)
    del rl1

    # ---- S2: concat_top3, MESMO C e mesmas camadas da v4 (reajuste determinístico, sem grade)
    t = time.time()
    mc = ler_json(ns["marcador"]("camadas"))
    cab = [r for r in mc["concat"] if r["cabeça"] == CABECA_S2][0]
    if not ns["QUICK"]:
        final = ler_json(ns["arq_result"]("resumo", "json"))["cabeça_final_pela_validação"]
        assert final == CABECA_S2, f"a cabeça final da v4 é {final}, não {CABECA_S2}"
    rl2 = LogisticRegression(C=cab["C"], max_iter=hp["RL_MAX_ITER"], tol=1e-4).fit(X(tr, cab["camadas"]), ytr)
    for k, c in conjs.items():
        notas[k]["S2"] = rl2.decision_function(X(c, cab["camadas"]))
    sv = pd.read_parquet(ns["arq_result"](f"notas_{CABECA_S2}_val", "parquet"))
    sdv = pd.read_parquet(ns["arq_result"](f"notas_{CABECA_S2}_dev_shift", "parquet"))
    assert (sv.idx_manifest.values == metas["val"].idx_manifest.values).all()
    assert (sdv.idx_manifest.values == metas["dev_shift"].idx_manifest.values).all()
    checagem["S2"] = {"camadas": cab["camadas"], "C": cab["C"], "iterações": int(rl2.n_iter_[0]),
                      "dif_max_val_vs_v4": float(np.abs(notas["val"]["S2"] - sv.z.values).max()),
                      "dif_max_dev_vs_v4": float(np.abs(notas["dev_shift"]["S2"] - sdv.z.values).max()),
                      "segundos": round(time.time() - t, 1)}
    print(f"S2: camadas {cab['camadas']} C={cab['C']:g} | dif. máx. val {checagem['S2']['dif_max_val_vs_v4']:.2e} "
          f"dev {checagem['S2']['dif_max_dev_vs_v4']:.2e}", flush=True)
    del rl2

    # ---- S0: notas da CNN, alinhadas por idx_manifest
    s0n = {"val": pd.read_parquet(C["res"]("notas_s0_val", "parquet")),
           "dev_shift": pd.read_parquet(C["res"]("notas_s0_dev_shift", "parquet")),
           "teste": pd.read_parquet(C["res"]("notas_s0_teste", "parquet"))}
    for k, m in metas.items():
        fonte = s0n[k] if k in ("val", "dev_shift") else s0n["teste"]
        s = fonte.set_index("idx_manifest").z_janelas
        notas[k]["S0"] = s.loc[m.idx_manifest.values].values.astype(np.float64)
        assert np.isfinite(notas[k]["S0"]).all()

    # ---- S3: 2 entradas (S2 e S0 padronizados pela validação), treinada só na validação
    pad = {s: (float(notas["val"][s].mean()), float(notas["val"][s].std())) for s in ("S2", "S0")}

    def X3(k):
        return np.column_stack([(notas[k][s] - pad[s][0]) / pad[s][1] for s in ("S2", "S0")])
    yv = metas["val"].label.values.astype(int)
    rl3 = LogisticRegression(C=S3_C, max_iter=1000).fit(X3("val"), yv)
    for k in conjs:
        notas[k]["S3"] = rl3.decision_function(X3(k))
    coef = {"intercepto": float(rl3.intercept_[0]), "coef_S2_padronizado": float(rl3.coef_[0][0]),
            "coef_S0_padronizado": float(rl3.coef_[0][1]),
            "padronização (média, desvio) na validação": {s: list(v) for s, v in pad.items()},
            "coef_S2_logit_original": float(rl3.coef_[0][0] / pad["S2"][1]),
            "coef_S0_logit_original": float(rl3.coef_[0][1] / pad["S0"][1]), "C": S3_C, "n_val": int(len(yv))}
    checagem["S3"] = coef
    print(f"S3: {coef}", flush=True)

    # ---- limiares (A: EER da validação; B: EER do dev_shift), cada sistema com os seus
    ydev = metas["dev_shift"].label.values.astype(int)
    lim = {s: limiares_de(notas["val"][s], yv, notas["dev_shift"][s], ydev) for s in SISTEMAS}
    print("limiares (logit):", json.dumps(lim), flush=True)

    # ---- métricas e bootstrap
    linhas, boots = [], {}
    tipo_conj = {"val": ("validação", "validação"), "dev_shift": ("dev_shift", "decisão entre versões")}
    for k, m in metas.items():
        y = m.label.values.astype(int)
        conjunto, tipo = tipo_conj.get(k, ("teste", TIPO.get(k)))
        banco = "in_the_wild (dev_shift)" if k == "dev_shift" else ("validação (todas)" if k == "val" else k)
        if conjunto == "teste":
            idx = reamostras(y, TESTE_N_BOOT, TESTE_SEED_BOOT)
            boots[k] = bootstrap_banco(y, {s: notas[k][s] for s in SISTEMAS}, lim, idx)
        for s in SISTEMAS:
            for L, t_ in lim[s].items():
                mm = metricas_banco(y, notas[k][s], t_)
                for met in ("acurácia", "acurácia balanceada", "acurácia trivial", "AUC", "EER", "recall fake",
                            "falso alarme real"):
                    lo = hi = np.nan
                    if conjunto == "teste":
                        chave = "AUC" if met == "AUC" else f"{met}|{L}"
                        if chave in boots[k][s]:
                            lo, hi = ic(boots[k][s][chave])
                    linhas.append({"sistema": s, "descrição": DESCRICAO[s], "conjunto": conjunto, "banco": banco,
                                   "tipo": tipo, "limiar": L, "limiar (como)": LIMIARES[L], "limiar (logit)": t_,
                                   "métrica": met, "valor": mm[met], "IC 2,5%": lo, "IC 97,5%": hi,
                                   "n real": mm["n real"], "n fake": mm["n fake"],
                                   "excluídos (erro de leitura)": excluidos[k]})
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, C["res"]("final", "csv").with_name(f"v4_teste_final{suf}.csv"))

    # ---- diferenças pareadas (leitura pré-definida)
    principais = [b for b, t_ in BANC if t_ == "principal"]
    pares = [("S2", "S1", "contribuição das nossas escolhas (S2 − S1)"), ("S3", "S2", "contribuição da CNN (S3 − S2)"),
             ("S1", None, "transfer learning sozinho (S1)")]
    dif, regra = [], {}
    for a_, b_, nome in pares[:2]:
        for met in ("AUC", "acurácia balanceada|A", "acurácia balanceada|B"):
            for k in [b for b, t_ in BANC if t_ != "só real"]:
                ba, bb = boots[k][a_][met], boots[k][b_][met]
                m_ = met.split("|")[0]
                L = met.split("|")[1] if "|" in met else "A"
                va_ = tab[(tab.sistema == a_) & (tab.banco == k) & (tab.métrica == m_) & (tab.limiar == L)].valor.iloc[0]
                vb_ = tab[(tab.sistema == b_) & (tab.banco == k) & (tab.métrica == m_) & (tab.limiar == L)].valor.iloc[0]
                lo, hi = ic(ba - bb)
                dif.append({"leitura": nome, "métrica": met.replace("|", ", limiar "), "banco": k, "tipo": TIPO[k],
                            "diferença": va_ - vb_, "IC 2,5%": lo, "IC 97,5%": hi, "IC exclui zero": bool(lo > 0 or hi < 0)})
            if met != "acurácia balanceada|B":
                d = [r for r in dif if r["leitura"] == nome and r["métrica"] == met.replace("|", ", limiar ")
                     and r["banco"] in principais]
                sinais = {np.sign(r["diferença"]) for r in d}
                ok = len(sinais) == 1 and 0 not in sinais and all(r["IC exclui zero"] for r in d)
                regra[f"{nome} | {met.replace('|', ', limiar ')}"] = (
                    ("relevante: " + ("positiva" if sinais == {1.0} else "negativa")) if ok else "não demonstrada")
    tdif = pd.DataFrame(dif)
    salvar_csv(tdif, C["res"]("diferencas", "csv"))
    print(tdif.round(4).to_string(index=False), flush=True)
    print(json.dumps(regra, ensure_ascii=False, indent=1), flush=True)

    # ---- resumo: média da acurácia dos 3 principais, CORAA à parte
    resumo = []
    so_real = [b for b, t_ in BANC if t_ == "só real"]
    for s in SISTEMAS:
        for L in ("A", "B"):
            sel = tab[(tab.sistema == s) & (tab.limiar == L) & tab.banco.isin(principais)]
            fa = tab[(tab.sistema == s) & (tab.limiar == L) & tab.banco.isin(so_real) & (tab.métrica == "falso alarme real")]
            resumo.append({"sistema": s, "limiar": L,
                           "média acurácia (3 principais)": float(sel[sel.métrica == "acurácia"].valor.mean()),
                           "média acurácia balanceada (3 principais)": float(sel[sel.métrica == "acurácia balanceada"].valor.mean()),
                           "média acurácia trivial (3 principais)": float(sel[sel.métrica == "acurácia trivial"].valor.mean()),
                           "média AUC (3 principais)": float(sel[sel.métrica == "AUC"].valor.mean()),
                           **({f"{so_real[0]} falso alarme": float(fa.valor.iloc[0]),
                               f"{so_real[0]} falso alarme IC": [float(fa["IC 2,5%"].iloc[0]), float(fa["IC 97,5%"].iloc[0])]}
                              if len(fa) else {})})
    tres = pd.DataFrame(resumo)
    print(tres.round(4).to_string(index=False), flush=True)

    # ---- notas por clipe (fora do git)
    cols = ["idx_manifest", "path", "source_dataset", "label", "generator", "fake_type", "speaker_id", "duration_s"]
    partes = []
    for k, m in metas.items():
        p = m[cols].copy()
        p.insert(0, "conjunto", "teste" if k in te else k)
        p.insert(1, "banco", k)
        for s in SISTEMAS:
            p[f"z_{s}"] = notas[k][s]
        partes.append(p)
    salvar_parquet(pd.concat(partes, ignore_index=True), C["res"]("notas", "parquet"))

    # ---- gráfico: acurácia balanceada por banco e sistema (limiares A e B) + falso alarme do banco só real
    fig = grafico(tab, BANC)
    png = R / f"v4_teste_final{suf}.png"
    gravar_atomico(png, lambda t_: fig.savefig(t_, format="png"))
    plt.close(fig)

    out = {"sistemas": DESCRICAO, "limiares_logit": lim, "checagem": checagem, "regra_leitura": regra,
           "resumo": tres.to_dict("records"), "excluidos_erro_leitura": excluidos, "n_boot": TESTE_N_BOOT,
           "bancos": BANC, "s0": ler_json(C["marcador"]("cnn"))["s0"]}
    salvar_json(out, C["res"]("final_resumo", "json").with_name(f"v4_teste_final_resumo{suf}.json"))
    gravar_marcador(ns, C, "avaliar", t0, {"regra_leitura": regra, "resumo": tres.to_dict("records")})
    print("avaliar CONCLUÍDA\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ status e concluir
def etapa_status(a, ns, t0):
    import pandas as pd

    from src.arquivos import gravar_atomico
    C = caminhos(ns)
    L = [f"Teste final v4{' (QUICK)' if ns['QUICK'] else ''} (atualizado em {datetime.now():%Y-%m-%d %H:%M:%S})",
         "Avaliado UMA vez; pré-registro em docs/HISTORICO_VERSOES.md. Nada é reescolhido depois.", ""]
    est = ler_json(C["estado"], {})
    L.append(f"extração: {est.get('horas_extracao', 0):.2f} h | último shard: {est.get('ultimo_shard', '-')} | "
             f"tentativas: {est.get('tentativas_extracao', 0)}")
    L += ["", f"{'etapa':<10} {'status':<11} {'min':>6} {'pico GB':>8}  observação"]
    for e in ETAPAS_TESTE:
        m = ler_json(C["marcador"](e))
        if m:
            obs = ""
            if e == "extracao":
                obs = f"{m['n_clipes']} clipes; erros de leitura {sum(m['erros_leitura'].values())}"
            elif e == "cnn":
                obs = f"S0 = {m['s0']}; conferência com a v3: {m['conferencia_v3']}"
            L.append(f"{e:<10} {'concluída':<11} {m['minutos_neste_processo']:>6.1f} "
                     f"{m['pico_memoria_gb_processo_e_workers']:>8.2f}  {obs}")
        else:
            L.append(f"{e:<10} {'pendente':<11} {'-':>6} {'-':>8}")
    arq = ns["RESULTS_DIR"] / f"v4_teste_final{ns['SUFIXO']}.csv"
    if C["marcador"]("avaliar").exists() and arq.exists():
        t = pd.read_csv(arq)
        t = t[t.conjunto == "teste"]
        for L_ in ("A", "B"):
            p = t[(t.limiar == L_) & t.métrica.isin(["acurácia", "acurácia balanceada", "AUC", "EER"])]
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
        salvar_json({"versao": "v4", "etapa": "teste final", "quick": ns["QUICK"],
                     "concluido_em": datetime.now().isoformat(timespec="seconds"),
                     "etapas": {e: {"concluido_em": m["concluido_em"], "minutos": m["minutos_neste_processo"],
                                    "pico_memoria_gb": m["pico_memoria_gb_processo_e_workers"]} for e, m in ms.items()},
                     "regra_leitura": ms["avaliar"]["regra_leitura"], "resumo": ms["avaliar"]["resumo"]}, C["final"])
    print(f"teste final da v4 CONCLUÍDO ({C['final']})\nSTATUS=CONCLUIDO", flush=True)
    return 0


ETAPAS_FN = {"extrair": etapa_extrair, "cnn": etapa_cnn, "avaliar": etapa_avaliar, "status": etapa_status,
             "concluir": etapa_concluir}


def main():
    global AMOSTRADOR
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
