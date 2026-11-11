#!/usr/bin/env python3
"""v6: funil de seis etapas sobre os dois front-ends congelados (XLS-R da Meta e WavLM da Microsoft).
Uma etapa por processo; cada uma grava um MARCADOR por último (results/v6_<etapa>_concluido.json) e, com o
marcador presente, não faz nada. Sem ele, continua de onde parou (shards, camadas e seeds prontos são pulados).

Os hiperparâmetros NÃO são copiados aqui: o script executa a célula com a tag "config" do notebook.ipynb.
O teste final NUNCA é lido aqui (só train, val e dev_shift); a E6 é o scripts/v6_teste.py, separado.

    python scripts/v6.py medir       [--quick]   # vazão do WavLM (500 clipes)
    python scripts/v6.py extrair     [--quick]   # WavLM nos mesmos clipes do v4_cache (treino, val, dev_shift)
    python scripts/v6.py e1          [--quick]   # Meta × Microsoft: RL por camada, concat_top3/4, bootstrap
    python scripts/v6.py e2          [--quick]   # fusão: média das notas, vetores colados, stacking
    python scripts/v6.py e3          [--quick]   # decisão do front-end
    python scripts/v6.py e4          [--quick]   # cabeças básicas: RL, MLP, CNN
    python scripts/v6.py e5          [--quick]   # maximização da cabeça vencedora + regra de limiar
    python scripts/v6.py finalistas  [--quick]   # grava F0, F1, F2 e os limiares (antes da E6)
    python scripts/v6.py relatorio   [--quick]   # funil, gráficos, projeção 2D (lê o que existir)
    python scripts/v6.py status      [--quick]   # reescreve logs/v6_resumo.txt e logs/v6_PROGRESSO.md
    opções: --num-workers N | --log arquivo
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

AMOSTRADOR = None
MODELOS_ORDEM = ("xlsr", "wavlm")
COLS_META = ["idx_manifest", "path", "source_dataset", "label", "generator", "speaker_id", "duration_s"]


# ============================================================ utilidades
def ler_json(p, padrao=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao


def log(s):
    print(s, flush=True)


def sem_suspensao(ligar=True):
    """Impede o Windows de suspender enquanto a etapa roda (execução, não método)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if ligar else 0x80000000)
    except Exception:  # noqa: BLE001
        pass


def manifest_e_sha():
    from src.arquivos import sha256_arquivo
    from src.experimento import ler_manifest
    arq = ROOT / "data/processed/manifest.parquet"
    return ler_manifest(arq), sha256_arquivo(arq)


def gravar_marcador(ns, etapa, t0, extra):
    from src.arquivos import identificar_codigo, salvar_json, sha256_texto
    codigo = identificar_codigo(ROOT, extras=["scripts/v6.py", "scripts/v6_teste.py", "scripts/rodar_v6.ps1",
                                              "scripts/vigia_v6.ps1"])
    codigo["celula_config_sha256"] = sha256_texto(ns["_SRC_CONFIG"])
    info = {"etapa": etapa, "versao": "v6", "quick": ns["QUICK"], "status": "concluido",
            "concluido_em": datetime.now().isoformat(timespec="seconds"),
            "minutos_neste_processo": round((time.time() - t0) / 60, 2),
            "pico_memoria_gb_processo_e_workers": round(AMOSTRADOR.pico_proc / 1024, 2),
            "pico_memoria_gb_sistema": round(AMOSTRADOR.pico_sis, 2), "codigo": codigo, **extra}
    salvar_json(info, ns["marcador"](etapa))
    return info


def ja_feita(ns, etapa):
    if ns["marcador"](etapa).exists():
        log(f"{etapa}: já concluída (marcador presente); nada a fazer.\nSTATUS=JA_CONCLUIDO")
        return True
    return False


def exigir(ns, *etapas):
    for e in etapas:
        if not ns["marcador"](e).exists():
            raise RuntimeError(f"a etapa '{e}' ainda não terminou ({ns['marcador'](e)} não existe)")


def num_workers(a, ns):
    if a.num_workers is not None:
        return a.num_workers
    return int(os.environ.get("V6_NUM_WORKERS", ns["NUM_WORKERS"]))


def carregar_modelos(ns, modelos=MODELOS_ORDEM, extras=None):
    """{modelo: Modelo} com os mesmos clipes e a padronização do treino de cada um.

    Treino, validação e dev_shift são SEMPRE abertos nos dois modelos e conferidos juntos (mesmo
    idx_manifest e caminho, na mesma ordem; um clipe com erro de leitura em qualquer modelo sai dos dois),
    para todas as etapas usarem exatamente os mesmos clipes, mesmo quando só um modelo é usado.
    `extras` = {modelo: {nome: (cache, conjunto)}} (testes, na E6): conferidos entre os modelos que os têm.
    """
    from src.v6_cabecas import Modelo, conferir_mesmos_clipes
    extras = extras or {}
    M = {m: Modelo(m, ns["CACHE"][m], ns["PARCIAL"] / f"{m}_padronizacao.npz", extras=extras.get(m))
         for m in MODELOS_ORDEM}
    rel = conferir_mesmos_clipes(M, ["treino", "val", "dev_shift"])
    com_extra = {m: M[m] for m in MODELOS_ORDEM if m in extras}
    if len(com_extra) > 1:
        rel.update(conferir_mesmos_clipes(com_extra, list(extras[list(com_extra)[0]].keys())))
    if not M["xlsr"].ok["treino"].all():
        raise RuntimeError("treino com erro de leitura (não previsto)")
    return {m: M[m] for m in MODELOS_ORDEM if m in modelos}, rel


def notas_parquet(meta, colunas):
    import pandas as pd
    df = meta[COLS_META].copy()
    for k, v in colunas.items():
        df[k] = v
    return df


def resumo_linha(nome, r, extra=None):
    """Linha de tabela com as métricas da v6 de um sistema."""
    out = {"sistema": nome, **(extra or {})}
    out.update({k: v for k, v in r.items()})
    return out


# ============================================================ medir (vazão do WavLM)
def etapa_medir(a, ns, t0):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.arquivos import salvar_json
    from src.cache_v4 import plano_treino
    from src.extracao_v4 import Baldes, LotesFixos, montar_lotes
    from src.preproc import ClipesV4, juntar_clipes
    from src.ssl import carregar_ssl

    if ja_feita(ns, "medir"):
        return 0
    hp = ns["HP"]
    df, _ = manifest_e_sha()
    idx = plano_treino(df[df.split == "train"], max(2, ns["N_MEDIR"] - ns["N_MEDIR"] % 2), 1, hp["SEED"] + 1)[0]
    g = df.loc[idx].sort_values("duration_s", kind="stable")
    nw = num_workers(a, ns)
    t = time.time()
    model = carregar_ssl(ns["MODELOS"]["wavlm"], "cuda", hp["DTYPE"], ns["REVISAO_WAVLM"])
    t_carga = time.time() - t
    ds = ClipesV4(g.path.values, np.arange(len(g)), cortar_silencio=True, top_db=hp["TOP_DB"],
                  min_s=hp["MIN_S"], max_s=hp["MAX_S"])
    lotes = montar_lotes(g.duration_s.values, hp["MAX_AMOSTRAS_LOTE"], hp["MAX_CLIPES_LOTE"])
    dl = DataLoader(ds, batch_sampler=LotesFixos(lotes), num_workers=nw, collate_fn=juntar_clipes,
                    persistent_workers=False, prefetch_factor=4 if nw else None)
    torch.cuda.reset_peak_memory_stats()
    t, n, t_primeiro, nan = time.time(), 0, None, 0
    baldes = Baldes(model, hp, "cuda")
    for k, (clipes, est, pos) in enumerate(dl):
        if t_primeiro is None:
            t_primeiro = time.time() - t
        res = baldes.adicionar(clipes, pos) + (baldes.esvaziar() if k == len(lotes) - 1 else [])
        nan += sum(int((~np.isfinite(r[1])).any(axis=(1, 2)).sum()) for r in res)
        n += sum(len(r[0]) for r in res)
    torch.cuda.synchronize()
    total = time.time() - t
    r = {"n": n, "num_workers": nw, "segundos_total": round(total, 1), "clipes_por_s": round(n / total, 2),
         "clipes_por_s_sem_partida": round(n / max(total - t_primeiro, 1e-9), 2), "carga_modelo_s": round(t_carga, 1),
         "pico_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "clipes_com_nan": nan,
         "refeitos_bf16": baldes.refeitos_bf16, "medido_em": datetime.now().isoformat(timespec="seconds"),
         "pico_memoria_gb_processo_e_workers": round(AMOSTRADOR.pico_proc / 1024, 2),
         "gpu": torch.cuda.get_device_name(0)}
    plano_total = 70_613
    r["estimativa_extracao_min (treino+val+dev, 70.613 clipes)"] = round(plano_total / max(r["clipes_por_s"], 1e-3) / 60, 1)
    r["estimativa_teste_min (51.320 clipes)"] = round(51_320 / max(r["clipes_por_s"], 1e-3) / 60, 1)
    salvar_json(r, ns["arq"]("wavlm_vazao", "json"))
    log(json.dumps(r, ensure_ascii=False, indent=1))
    gravar_marcador(ns, "medir", t0, {"vazao": r})
    log("medir CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ extrair (WavLM, plano do v4_cache)
def plano_wavlm(ns, df, sha):
    """O plano do v4_cache (XLS-R) só com treino, validação e dev_shift (sem os *_semcorte), copiado."""
    import pandas as pd
    from src.arquivos import salvar_json, salvar_parquet
    base = ns["CACHE"]["xlsr"]
    cfg_x = ler_json(base / "plano.json")
    if cfg_x["manifest_sha256"] != sha:
        raise RuntimeError(f"o manifest mudou desde o v4_cache ({cfg_x['manifest_sha256']} × {sha})")
    p = pd.read_parquet(base / "plano.parquet")
    p = p[p.conjunto.isin(["dev_shift", "val", "treino"])].sort_values("ordem").reset_index(drop=True)
    p["ordem"] = range(len(p))
    destino = ns["CACHE"]["wavlm"]
    cfg = {"extracao": ns["CFG_EXTRACAO"], "manifest_sha256": sha,
           "plano": {"copiado_de": str(base).replace("\\", "/"), "conjuntos": ["dev_shift", "val", "treino"],
                     "plano_original": cfg_x["plano"]}, "revisao_modelo": ns["REVISAO_WAVLM"]}
    arq, arq_cfg = destino / "plano.parquet", destino / "plano.json"
    if arq.exists():
        salvo = ler_json(arq_cfg)
        if salvo != json.loads(json.dumps(cfg, default=str)):
            raise RuntimeError(f"{arq_cfg} é de outra configuração. Nada foi apagado.\nsalvo: {salvo}\nagora: {cfg}")
        antigo = pd.read_parquet(arq)
        if not antigo[["idx_manifest", "conjunto", "shard"]].equals(p[["idx_manifest", "conjunto", "shard"]]):
            raise RuntimeError(f"{arq} não bate com o plano do v4_cache. Nada foi apagado.")
        return antigo
    destino.mkdir(parents=True, exist_ok=True)
    salvar_parquet(p, arq)
    salvar_json(cfg, arq_cfg)
    return p


def etapa_extrair(a, ns, t0):
    from src.cache_v4 import Orcamento, limpar_temporarios, shard_pronto
    from src.extracao_v4 import extrair
    from src.ssl import carregar_ssl

    if ja_feita(ns, "extracao"):
        return 0
    hp = dict(ns["HP"])
    vz = ler_json(ns["arq"]("wavlm_vazao", "json"))
    if vz:
        hp["VAZAO_ESTIMADA"] = vz["clipes_por_s"]
    df, sha = manifest_e_sha()
    plano = plano_wavlm(ns, df, sha)
    cache = ns["CACHE"]["wavlm"]
    n_tmp = limpar_temporarios(cache)
    if n_tmp:
        log(f"extração: {n_tmp} shard(s) incompleto(s) de uma execução interrompida foram descartados")
    log("plano (copiado do v4_cache):\n" + plano.groupby("conjunto", sort=False).agg(
        clipes=("idx_manifest", "size"), shards=("shard", "nunique")).to_string())
    orc = Orcamento(ns["arq"]("wavlm_estado", "json"), ns["ORCAMENTO_EXTRACAO_H"])
    log(f"orçamento: {orc.base / 3600:.2f} h já gastas de {ns['ORCAMENTO_EXTRACAO_H']} h (tentativa {orc.estado['tentativas_extracao']})")
    nw = num_workers(a, ns)
    r = {"parou_por": "orçamento já esgotado antes de começar", "shards_feitos": 0}
    try:
        if orc.resta() > 0:
            model = carregar_ssl(ns["MODELOS"]["wavlm"], "cuda", hp["DTYPE"], ns["REVISAO_WAVLM"])
            r = extrair(df, plano, model, hp, cache, orc, ns["CFG_EXTRACAO"], log=log, num_workers=nw)
    finally:
        orc.fechar()
    feitos = {}
    for conj, g in plano.groupby("conjunto", sort=False):
        shards = g.shard.unique()
        prontos = [int(s) for s in shards if shard_pronto(cache, conj, s)]
        feitos[conj] = {"shards_prontos": len(prontos), "shards_planejados": int(len(shards)),
                        "clipes_prontos": int(g[g.shard.isin(prontos)].shape[0]), "clipes_planejados": int(len(g))}
    log("feito até agora:\n" + json.dumps(feitos, ensure_ascii=False, indent=1))
    completo = all(v["shards_prontos"] == v["shards_planejados"] for v in feitos.values())
    if not completo:
        if orc.resta() <= 0 or r["parou_por"].startswith("orçamento"):
            raise RuntimeError(f"orçamento de extração esgotado sem completar o plano ({r['parou_por']}); "
                               "a E1 precisa dos mesmos clipes nos dois modelos")
        log(f"extração incompleta ({r['parou_por']}); rode de novo.\nSTATUS=INCOMPLETO")
        return 2
    from src.cache_v4 import Conjunto
    erros = {c: int(Conjunto(cache, c).meta.erro_leitura.fillna(0).astype(bool).sum()) for c in feitos}
    info = gravar_marcador(ns, "extracao", t0, {
        "conjuntos": feitos, "erros_leitura": erros, "horas_extracao_acumuladas": round(orc.gasto() / 3600, 3),
        "orcamento_h": ns["ORCAMENTO_EXTRACAO_H"], "ultima_passada": r, "num_workers": nw,
        "modelo": ns["MODELOS"]["wavlm"], "revisao": ns["REVISAO_WAVLM"]})
    log(f"extração CONCLUÍDA; {info['horas_extracao_acumuladas']:.2f} h acumuladas; erros {erros}\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ E1
def etapa_e1(a, ns, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.arquivos import gravar_atomico, salvar_csv, salvar_json, salvar_parquet
    from src.v6_cabecas import N_SAIDAS, rl_notas
    from src.v6_metricas import BootPior, avaliar, grupos

    if ja_feita(ns, "e1"):
        return 0
    exigir(ns, "extracao")
    hp = ns["HP"]
    M, rel = carregar_modelos(ns)
    mv, md = M["xlsr"].meta("val"), M["xlsr"].meta("dev_shift")
    ytr = M["xlsr"].meta("treino").label.values.astype(int)
    log(f"clipes: {json.dumps(rel)}")
    log(f"fontes de decisão: {[g[0] for g in grupos(mv, md)]}")
    boot = BootPior(mv, md, hp["N_BOOT"], hp["SEED_BOOT"])
    ns["PARCIAL"].mkdir(parents=True, exist_ok=True)
    grade = hp["GRADE_C"]
    linhas, notas_v, notas_d = [], {}, {}

    def chave_pior(r):
        return (r["AUC pior fonte"], r.get("AUC média (G)", 0))

    for m in MODELOS_ORDEM:
        for l in range(N_SAIDAS):
            arq_z = ns["PARCIAL"] / f"e1_{m}_L{l:02d}.npz"
            arq_j = ns["PARCIAL"] / f"e1_{m}_L{l:02d}.json"
            if arq_z.exists() and arq_j.exists():
                z = np.load(arq_z)
                zv_c, zd_c = {C: z[f"val_{i}"] for i, C in enumerate(grade)}, {C: z[f"dev_{i}"] for i, C in enumerate(grade)}
                info = ler_json(arq_j)
            else:
                t = time.time()
                Xtr = M[m].X("treino", [l])
                alvos = {"val": M[m].X("val", [l]), "dev": M[m].X("dev_shift", [l])}
                zv_c, zd_c, info = {}, {}, {"conv": {}, "seg": {}}
                for C in grade:
                    z, rl, seg = rl_notas(Xtr, ytr, alvos, C, hp["RL_MAX_ITER"])
                    zv_c[C], zd_c[C] = z["val"], z["dev"]
                    info["conv"][str(C)], info["seg"][str(C)] = bool(rl.convergiu_), round(seg, 1)
                np.savez(arq_z, **{f"val_{i}": zv_c[C] for i, C in enumerate(grade)},
                         **{f"dev_{i}": zd_c[C] for i, C in enumerate(grade)})
                salvar_json(info, arq_j)
                del Xtr, alvos
                log(f"  {m} camada {l:2d}: {time.time() - t:.0f} s")
            res = {C: avaliar(zv_c[C], mv, zd_c[C], md) for C in grade}
            c6 = max(grade, key=lambda C: chave_pior(res[C]))
            c4 = min(grade, key=lambda C: res[C]["val EER média por fonte (3)"])
            r = res[c6]
            c1 = 1.0 if 1.0 in grade else grade[-1]
            linhas.append({"modelo": m, "camada": l, "C (pior fonte)": c6, **r,
                           "C (regra v4)": c4, "val EER média por fonte (3), regra v4": res[c4]["val EER média por fonte (3)"],
                           "AUC pior fonte (C=1)": res[c1]["AUC pior fonte"], "fonte pior (C=1)": res[c1]["fonte pior"],
                           "AUC média (G) (C=1)": res[c1].get("AUC média (G)"),
                           "convergiu": info["conv"][str(c6)]})
            notas_v[f"z_{m}_L{l:02d}"], notas_d[f"z_{m}_L{l:02d}"] = zv_c[c6], zd_c[c6]
            notas_v[f"z_{m}_L{l:02d}_C1"], notas_d[f"z_{m}_L{l:02d}_C1"] = zv_c[c1], zd_c[c1]
            log(f"{m} camada {l:2d}: C={c6:g} | pior fonte {r['AUC pior fonte']:.4f} ({r['fonte pior']}) | "
                f"média {r.get('AUC média (G)', np.nan):.4f} | dev AUC {r.get('dev AUC', np.nan):.4f} | "
                f"val EER média (v4) {100 * res[c4]['val EER média por fonte (3)']:.2f}% (C={c4:g})")
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, ns["arq"]("e1_camadas", "csv"))

    # ---- concatenações e melhor cabeça de cada modelo
    cabecas, melhor = [], {}
    for m in MODELOS_ORDEM:
        t_m = tab[tab.modelo == m].sort_values(["AUC pior fonte", "AUC média (G)"], ascending=False)
        ordem = t_m.camada.tolist()
        cands = [{"modelo": m, "cabeça": f"rl_camada_{int(r.camada)}", "camadas": [int(r.camada)],
                  "C": float(r["C (pior fonte)"]),
                  **{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if k not in ("modelo", "camada")}}
                 for _, r in t_m.iterrows()]
        for k in hp["TOP_K"]:
            cams = sorted(ordem[:k])
            nome = f"concat_top{k}"
            arq_z = ns["PARCIAL"] / f"e1_{m}_{nome}.npz"
            if arq_z.exists() and ler_json(ns["PARCIAL"] / f"e1_{m}_{nome}.json", {}).get("camadas") == cams:
                z = np.load(arq_z)
                zv_c, zd_c = {C: z[f"val_{i}"] for i, C in enumerate(grade)}, {C: z[f"dev_{i}"] for i, C in enumerate(grade)}
            else:
                t = time.time()
                Xtr = M[m].X("treino", cams)
                alvos = {"val": M[m].X("val", cams), "dev": M[m].X("dev_shift", cams)}
                zv_c, zd_c = {}, {}
                for C in grade:
                    z, rl, seg = rl_notas(Xtr, ytr, alvos, C, hp["RL_MAX_ITER"])
                    zv_c[C], zd_c[C] = z["val"], z["dev"]
                np.savez(arq_z, **{f"val_{i}": zv_c[C] for i, C in enumerate(grade)},
                         **{f"dev_{i}": zd_c[C] for i, C in enumerate(grade)})
                salvar_json({"camadas": cams}, ns["PARCIAL"] / f"e1_{m}_{nome}.json")
                del Xtr, alvos
                log(f"  {m} {nome} {cams}: {time.time() - t:.0f} s")
            res = {C: avaliar(zv_c[C], mv, zd_c[C], md) for C in grade}
            c6 = max(grade, key=lambda C: chave_pior(res[C]))
            cands.append({"modelo": m, "cabeça": nome, "camadas": cams, "C": c6, **res[c6]})
            notas_v[f"z_{m}_{nome}"], notas_d[f"z_{m}_{nome}"] = zv_c[c6], zd_c[c6]
            log(f"{m} {nome} {cams} C={c6:g}: pior fonte {res[c6]['AUC pior fonte']:.4f} ({res[c6]['fonte pior']}) | "
                f"dev AUC {res[c6].get('dev AUC', np.nan):.4f}")
        best = max(cands, key=chave_pior)
        melhor[m] = {k: best[k] for k in ("cabeça", "camadas", "C")}
        nome_col = f"z_{m}_L{best['camadas'][0]:02d}" if best["cabeça"].startswith("rl_camada") else f"z_{m}_{best['cabeça']}"
        notas_v[f"z_{m}_melhor"], notas_d[f"z_{m}_melhor"] = notas_v[nome_col], notas_d[nome_col]
        for c in cands:
            c["é a melhor do modelo"] = c is best
        cabecas += [c for c in cands if not c["cabeça"].startswith("rl_camada") or c is best]
        log(f"==> melhor cabeça {m}: {best['cabeça']} {best['camadas']} C={best['C']:g} | pior fonte {best['AUC pior fonte']:.4f}")

    # ---- referência v4 (F0): XLS-R camadas 5, 6, 7, C = 0,1
    ref = hp["V4_REF"]
    z, _, _ = rl_notas(M["xlsr"].X("treino", ref["camadas"]), ytr,
                       {"val": M["xlsr"].X("val", ref["camadas"]), "dev": M["xlsr"].X("dev_shift", ref["camadas"])},
                       ref["C"], hp["RL_MAX_ITER"])
    notas_v["z_v4ref"], notas_d["z_v4ref"] = z["val"], z["dev"]
    r_ref = avaliar(z["val"], mv, z["dev"], md)
    cabecas.append({"modelo": "xlsr", "cabeça": "v4_ref (concat 5,6,7; C=0,1)", "camadas": ref["camadas"], "C": ref["C"], **r_ref})
    conf_ref = {}
    for nome, zz, mm in (("val", z["val"], mv), ("dev_shift", z["dev"], md)):
        arq_v4 = ns["RESULTS_DIR"] / f"v4_notas_concat_top3_{nome}{'_quick_t2' if ns['QUICK'] else ''}.parquet"
        if arq_v4.exists():
            g = pd.read_parquet(arq_v4)
            if len(g) == len(mm) and (g.idx_manifest.values == mm.idx_manifest.values).all():
                conf_ref[nome] = float(np.abs(g.z.values - zz).max())
    log(f"v4_ref: pior fonte {r_ref['AUC pior fonte']:.4f} | dif. máx. contra as notas gravadas da v4: {conf_ref}")

    # ---- reprodução da v4 (XLS-R): EER média por fonte por camada, C pela regra da v4
    repro = {"feita": False}
    arq_v4c = ns["RESULTS_DIR"] / f"v4_camadas{'_quick_t2' if ns['QUICK'] else ''}.csv"
    if arq_v4c.exists():
        v4c = pd.read_csv(arq_v4c).set_index("camada")["val EER média por fonte"]
        tx = tab[tab.modelo == "xlsr"].set_index("camada")["val EER média por fonte (3), regra v4"]
        dif = (tx - v4c.loc[tx.index]).abs()
        repro = {"feita": True, "dif_max_pontos": round(100 * float(dif.max()), 3), "camada_dif_max": int(dif.idxmax()),
                 "limite_pontos": 0.5, "passou": bool(100 * dif.max() <= 0.5),
                 "dif_por_camada_pontos": {int(k): round(100 * float(v), 3) for k, v in dif.items()}}
        log(f"reprodução XLS-R × v4: dif. máx. {repro['dif_max_pontos']} ponto (camada {repro['camada_dif_max']}) -> "
            f"{'PASSOU' if repro['passou'] else 'NÃO PASSOU'}")

    # ---- bootstrap: melhor WavLM − melhor XLS-R
    comp = boot.comparar((notas_v["z_xlsr_melhor"], notas_d["z_xlsr_melhor"]),
                         (notas_v["z_wavlm_melhor"], notas_d["z_wavlm_melhor"]), hp["DIF_MIN"], mv, md)
    log(f"bootstrap (WavLM − XLS-R, melhores cabeças): {json.dumps(comp)}")
    tcab = pd.DataFrame(cabecas)
    tcab["camadas"] = tcab.camadas.astype(str)
    salvar_csv(tcab, ns["arq"]("e1_cabecas", "csv"))
    salvar_parquet(notas_parquet(mv, notas_v), ns["arq"]("notas_e1_val", "parquet"))
    salvar_parquet(notas_parquet(md, notas_d), ns["arq"]("notas_e1_dev_shift", "parquet"))

    # ---- gráfico: AUC por camada e por fonte, nos dois modelos
    fontes = [g[0] for g in grupos(mv, md)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=80, sharey=True)
    cores = {"dfadd": "#1baf7a", "shiftyspeech": "#eda100", "codecfake_plus": "#d6336c", "dev_shift": "#2a78d6"}
    for ax, m in zip(axes, MODELOS_ORDEM):
        t_m = tab[tab.modelo == m].sort_values("camada")
        for f in fontes:
            ax.plot(t_m.camada, t_m[f"AUC {f}"], "-", lw=1.2, color=cores.get(f, "gray"), label=f)
        ax.plot(t_m.camada, t_m["AUC pior fonte"], "k-", lw=2.4, label="pior fonte")
        ax.axvline(melhor[m]["camadas"][0] if len(melhor[m]["camadas"]) == 1 else np.mean(melhor[m]["camadas"]),
                   color="gray", ls=":", lw=1)
        ax.set(title=ns["NOME_MODELO"][m], xlabel="saída do modelo (0 = entrada do transformer)", ylim=(0.5, 1.0))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUC (validação por fonte; dev_shift)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("E1: AUC por camada (RL, C pela pior fonte)", fontsize=10)
    fig.tight_layout()
    gravar_atomico(ns["arq"]("e1_camadas", "png"), lambda t_: fig.savefig(t_, format="png"))
    plt.close(fig)

    resumo = {"identidade_clipes": rel, "fontes_decisao": fontes, "bootstrap": boot.descricao(),
              "melhor_cabeca": melhor, "melhor_metricas": {m: [c for c in cabecas if c["modelo"] == m and c.get("é a melhor do modelo")][0]
                                                           for m in MODELOS_ORDEM},
              "v4_ref": r_ref, "v4_ref_dif_max_notas_gravadas": conf_ref, "reproducao_v4": repro,
              "bootstrap_wavlm_menos_xlsr": comp}
    salvar_json(resumo, ns["arq"]("e1_resumo", "json"))
    gravar_marcador(ns, "e1", t0, {"melhor_cabeca": melhor, "reproducao_v4": {k: v for k, v in repro.items() if k != "dif_por_camada_pontos"},
                                   "bootstrap_wavlm_menos_xlsr": comp,
                                   "pior_fonte": {m: resumo["melhor_metricas"][m]["AUC pior fonte"] for m in MODELOS_ORDEM},
                                   "v4_ref_pior_fonte": r_ref["AUC pior fonte"]})
    log("e1 CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ E2
def tabela_2x2(y, za, zb, la, lb):
    import numpy as np
    ca, cb = (za >= la) == (y == 1), (zb >= lb) == (y == 1)
    n = len(y)
    d = {"os dois acertam": int((ca & cb).sum()), "só a Meta acerta": int((ca & ~cb).sum()),
         "só a Microsoft acerta": int((~ca & cb).sum()), "os dois erram": int((~ca & ~cb).sum())}
    return {**d, **{f"{k} (%)": round(100 * v / n, 2) for k, v in list(d.items())}, "n": int(n)}


def etapa_e2(a, ns, t0):
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    from src.arquivos import salvar_csv, salvar_json, salvar_parquet
    from src.metrics import eer_e_limiar
    from src.v6_cabecas import X_colados, fusao_media, fusao_stacking, rl_notas
    from src.v6_metricas import BootPior, avaliar, regra_desafiante

    if ja_feita(ns, "e2"):
        return 0
    exigir(ns, "e1")
    hp = ns["HP"]
    e1 = ler_json(ns["arq"]("e1_resumo", "json"))
    nv, nd = pd.read_parquet(ns["arq"]("notas_e1_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e1_dev_shift", "parquet"))
    mv, md = nv, nd
    boot = BootPior(mv, md, hp["N_BOOT"], hp["SEED_BOOT"])
    zx = {"val": nv.z_xlsr_melhor.values, "dev": nd.z_xlsr_melhor.values}
    zw = {"val": nv.z_wavlm_melhor.values, "dev": nd.z_wavlm_melhor.values}

    # ---- correlação e erros em comum (antes de comparar)
    corr = []
    for nome, meta, k in (("validação", nv, "val"), ("dev_shift", nd, "dev")):
        for c, rot in ((0, "real"), (1, "fake")):
            mk = meta.label.values == c
            if mk.sum() > 2:
                corr.append({"conjunto": nome, "classe": rot, "n": int(mk.sum()),
                             "Pearson": float(pearsonr(zx[k][mk], zw[k][mk])[0]),
                             "Spearman": float(spearmanr(zx[k][mk], zw[k][mk])[0])})
    lim = {m: eer_e_limiar(nv.label.values, z["val"])[1] for m, z in (("xlsr", zx), ("wavlm", zw))}
    t2 = {"limiar A (logit)": lim}
    for nome, meta, k in (("validação", nv, "val"), ("dev_shift", nd, "dev")):
        t2[nome] = tabela_2x2(meta.label.values.astype(int), zx[k], zw[k], lim["xlsr"], lim["wavlm"])
        for c, rot in ((0, "reais"), (1, "fakes")):
            mk = meta.label.values == c
            t2[f"{nome} ({rot})"] = tabela_2x2(meta.label.values[mk].astype(int), zx[k][mk], zw[k][mk], lim["xlsr"], lim["wavlm"])
    log("correlação:\n" + pd.DataFrame(corr).round(3).to_string(index=False))
    log("erros em comum: " + json.dumps({k: v for k, v in t2.items() if k in ("validação", "dev_shift")}, ensure_ascii=False))

    # ---- (a) média das notas
    sist, linhas = {}, []
    fa = fusao_media([{"val": zx["val"], "dev": zx["dev"]}, {"val": zw["val"], "dev": zw["dev"]}])
    sist["fusao_media"] = (fa["val"], fa["dev"])
    # ---- (c) stacking com cross-fitting por fonte
    fc, coefs = fusao_stacking([{"val": zx["val"], "dev": zx["dev"]}, {"val": zw["val"], "dev": zw["dev"]}], nv, hp["STACK_C"])
    sist["fusao_stacking"] = (fc["val"], fc["dev"])
    log("stacking (pesos: modelo 1 = Meta, modelo 2 = Microsoft):\n" + pd.DataFrame(coefs).round(3).to_string(index=False))
    # ---- (b) vetores colados
    M, _ = carregar_modelos(ns)
    ytr = M["xlsr"].meta("treino").label.values.astype(int)
    mc = [(M[m], e1["melhor_cabeca"][m]["camadas"]) for m in MODELOS_ORDEM]
    arq_z = ns["PARCIAL"] / "e2_colados.npz"
    grade = hp["GRADE_C"]
    if arq_z.exists():
        z = np.load(arq_z)
        zv_c, zd_c = {C: z[f"val_{i}"] for i, C in enumerate(grade)}, {C: z[f"dev_{i}"] for i, C in enumerate(grade)}
    else:
        Xtr = X_colados(mc, "treino")
        alvos = {"val": X_colados(mc, "val"), "dev": X_colados(mc, "dev_shift")}
        log(f"vetores colados: {Xtr.shape[1]} dimensões ({[(m.nome, c) for m, c in mc]})")
        zv_c, zd_c = {}, {}
        for C in grade:
            z, rl, seg = rl_notas(Xtr, ytr, alvos, C, hp["RL_MAX_ITER"])
            zv_c[C], zd_c[C] = z["val"], z["dev"]
            log(f"  C={C:g}: {seg:.0f} s, convergiu {rl.convergiu_}")
        np.savez(arq_z, **{f"val_{i}": zv_c[C] for i, C in enumerate(grade)}, **{f"dev_{i}": zd_c[C] for i, C in enumerate(grade)})
        del Xtr, alvos
    res = {C: avaliar(zv_c[C], mv, zd_c[C], md) for C in grade}
    c_b = max(grade, key=lambda C: (res[C]["AUC pior fonte"], res[C].get("AUC média (G)", 0)))
    sist["fusao_colados"] = (zv_c[c_b], zd_c[c_b])
    for C in grade:
        linhas.append(resumo_linha(f"fusao_colados (C={C:g})", res[C], {"escolhido": C == c_b}))

    base = {"xlsr_melhor": (zx["val"], zx["dev"]), "wavlm_melhor": (zw["val"], zw["dev"])}
    for nome, (v, d) in {**base, **sist}.items():
        linhas.append(resumo_linha(nome, avaliar(v, mv, d, md), {"escolhido": True}))
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, ns["arq"]("e2_fusao", "csv"))
    log(tab[["sistema", "AUC pior fonte", "fonte pior", "AUC média (G)", "dev AUC", "acc bal média (G)"]].round(4).to_string(index=False))

    log("regra do desafiante dentro da fusão (média → stacking → colados):")
    venc, comps = regra_desafiante(boot, [(k, sist[k]) for k in ("fusao_media", "fusao_stacking", "fusao_colados")],
                                   mv, md, hp["DIF_MIN"], log=log)
    salvar_parquet(notas_parquet(mv, {f"z_{k}": v[0] for k, v in sist.items()}), ns["arq"]("notas_e2_val", "parquet"))
    salvar_parquet(notas_parquet(md, {f"z_{k}": v[1] for k, v in sist.items()}), ns["arq"]("notas_e2_dev_shift", "parquet"))
    out = {"correlacao": corr, "erros_2x2": t2, "stacking_coeficientes": coefs, "colados_C": c_b,
           "colados_camadas": {m: e1["melhor_cabeca"][m]["camadas"] for m in MODELOS_ORDEM},
           "comparacoes": comps, "vencedor_fusao": venc,
           "metricas": {k: avaliar(v[0], mv, v[1], md) for k, v in sist.items()}}
    salvar_json(out, ns["arq"]("e2_resumo", "json"))
    gravar_marcador(ns, "e2", t0, {"vencedor_fusao": venc, "pior_fonte": {k: out["metricas"][k]["AUC pior fonte"] for k in sist}})
    log(f"==> vencedor da fusão: {venc}\ne2 CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ E3
FRONTEND_MODELOS = {"xlsr": ["xlsr"], "wavlm": ["wavlm"], "fusao_media": ["xlsr", "wavlm"],
                    "fusao_stacking": ["xlsr", "wavlm"], "fusao_colados": ["xlsr", "wavlm"]}


def etapa_e3(a, ns, t0):
    import pandas as pd

    from src.arquivos import salvar_json
    from src.v6_metricas import BootPior, avaliar, regra_desafiante

    if ja_feita(ns, "e3"):
        return 0
    exigir(ns, "e2")
    hp = ns["HP"]
    e2 = ler_json(ns["arq"]("e2_resumo", "json"))
    nv, nd = pd.read_parquet(ns["arq"]("notas_e1_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e1_dev_shift", "parquet"))
    fv, fd = pd.read_parquet(ns["arq"]("notas_e2_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e2_dev_shift", "parquet"))
    boot = BootPior(nv, nd, hp["N_BOOT"], hp["SEED_BOOT"])
    vf = e2["vencedor_fusao"]
    cands = [("xlsr", (nv.z_xlsr_melhor.values, nd.z_xlsr_melhor.values)),
             ("wavlm", (nv.z_wavlm_melhor.values, nd.z_wavlm_melhor.values)),
             (vf, (fv[f"z_{vf}"].values, fd[f"z_{vf}"].values))]
    log("regra do desafiante (só Meta → só Microsoft → fusão):")
    venc, comps = regra_desafiante(boot, cands, nv, nd, hp["DIF_MIN"], log=log)
    metr = {n: avaliar(v[0], nv, v[1], nd) for n, v in cands}
    razao = (f"{venc} ficou: " + "; ".join(
        f"{c['desafiante']} × {c['atual']}: dif {c['por seed'][0]['dif AUC pior (desafiante - atual)']:+.4f}, IC95 "
        f"[{c['por seed'][0]['IC95'][0]:+.4f}, {c['por seed'][0]['IC95'][1]:+.4f}] -> "
        f"{'relevante' if c['relevante em todas as seeds'] else 'não relevante'}" for c in comps))
    out = {"vencedor": venc, "modelos": FRONTEND_MODELOS[venc], "candidatos": [n for n, _ in cands],
           "comparacoes": comps, "metricas": metr, "razao": razao}
    salvar_json(out, ns["arq"]("e3_frontend", "json"))
    gravar_marcador(ns, "e3", t0, {"vencedor": venc, "razao": razao})
    log(f"==> front-end: {venc}\n{razao}\ne3 CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ sistemas (usados na E4, na E5 e na E6)
def combinar(frontend, notas_por_modelo, meta_val, hp):
    """Junta as notas de cada modelo conforme o front-end. `notas_por_modelo` = {m: {alvo: z}}."""
    from src.v6_cabecas import fusao_media, fusao_stacking
    ms = FRONTEND_MODELOS[frontend]
    if len(ms) == 1:
        return notas_por_modelo[ms[0]], None
    lst = [notas_por_modelo[m] for m in ms]
    if frontend == "fusao_media":
        return fusao_media(lst), None
    if frontend == "fusao_stacking":
        return fusao_stacking(lst, meta_val, hp["STACK_C"])
    raise ValueError(frontend)


def notas_rl_sistema(M, spec, alvos, hp):
    """Notas de um sistema RL `spec` = {frontend, camadas{m}, C{m} (ou C), pesos} nos `alvos`
    ({nome curto: nome do conjunto}); devolve ({alvo: z}, coeficientes do stacking ou None)."""
    from src.v6_cabecas import X_colados, pesos_por_fonte, rl_notas
    ms = FRONTEND_MODELOS[spec["frontend"]]
    mtr = M[ms[0]].meta("treino")
    ytr = mtr.label.values.astype(int)
    sw = pesos_por_fonte(mtr) if spec.get("pesos") == "fonte" else None
    if spec["frontend"] == "fusao_colados":
        mc = [(M[m], spec["camadas"][m]) for m in ms]
        z, _, _ = rl_notas(X_colados(mc, "treino"), ytr, {k: X_colados(mc, c) for k, c in alvos.items()},
                           spec["C"]["colados"], hp["RL_MAX_ITER"], sw)
        return z, None
    por_m = {}
    for m in ms:
        cams = spec["camadas"][m]
        por_m[m], _, _ = rl_notas(M[m].X("treino", cams), ytr, {k: M[m].X(c, cams) for k, c in alvos.items()},
                                  spec["C"][m], hp["RL_MAX_ITER"], sw)
    return combinar(spec["frontend"], por_m, M[ms[0]].meta("val"), hp)


def treinar_nn_sistema(M, spec, hp_nn, seed, alvos, device, ns, nome_ckpt, log_=log):
    """Treina (ou carrega) a MLP/CNN de `spec` em cada modelo do front-end (ou nos colados) e devolve
    ({alvo: z} já combinadas, info). `spec["tipo"]` = "mlp" | "cnn"."""
    import numpy as np
    import torch

    from src.arquivos import salvar_torch
    from src.v6_cabecas import CNNCamadas, MLP, Lido25, NaGPU, X_colados, n_parametros, prever, treinar_nn, x25_na_gpu
    hp = ns["HP"]
    ms = FRONTEND_MODELOS[spec["frontend"]]
    mval = M[ms[0]].meta("val")
    ytr = M[ms[0]].meta("treino").label.values.astype(int)
    grupos = [ms] if spec["frontend"] == "fusao_colados" else [[m] for m in ms]
    por_m, info = {}, {"parametros": 0, "segundos": 0.0, "melhor_época": {}, "épocas": {}}
    for g in grupos:
        chave = "+".join(g)
        ckpt = ns["MODELS_DIR"] / f"v6_{nome_ckpt}_{chave}_s{seed}{ns['SUFIXO']}.pt"

        def fonte(n):
            if spec["tipo"] == "mlp":
                if len(g) == 1:
                    return NaGPU(M[g[0]].X(n, spec["camadas"][g[0]]), device)
                return NaGPU(X_colados([(M[m], spec["camadas"][m]) for m in g], n), device)
            if len(g) == 1:
                return NaGPU(x25_na_gpu(M[g[0]], n, device), device)
            return Lido25([M[m] for m in g], n, device)

        def novo():
            if spec["tipo"] == "mlp":
                d = sum(1024 * len(spec["camadas"][m]) for m in g)
                return MLP(d, hp_nn["OCULTA"], hp_nn["DROPOUT"])
            return CNNCamadas(2048 * len(g), hp["CNN_CANAIS"], hp["CNN_K"], hp_nn.get("DROPOUT_CNN", 0.0))
        torch.manual_seed(seed)
        model = novo()
        if ckpt.exists():
            st = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(st["state"])
            model.to(device)
            info["melhor_época"][chave], info["épocas"][chave] = st["melhor_época"], st["épocas"]
            info["segundos"] += st["segundos"]
            log_(f"    {nome_ckpt} {chave} seed {seed}: checkpoint já treinado ({ckpt.name})")
        else:
            log_(f"    {nome_ckpt} {chave} seed {seed}: treinando ({n_parametros(model):,} parâmetros)")
            ftr, fv = fonte("treino"), fonte("val")
            model, hist, ep, seg = treinar_nn(model, ftr, ytr, fv, mval, hp_nn, seed, device, log=log_)
            del ftr, fv
            torch.cuda.empty_cache()
            salvar_torch({"state": model.state_dict(), "melhor_época": ep, "épocas": len(hist), "segundos": seg,
                          "spec": spec, "hp_nn": hp_nn, "seed": seed, "historico": hist.to_dict("records")}, ckpt)
            info["melhor_época"][chave], info["épocas"][chave] = ep, len(hist)
            info["segundos"] += seg
        info["parametros"] += n_parametros(model)
        z = {}
        for k, c in alvos.items():
            f = fonte(c)
            z[k] = prever(model, f, device)
            del f
            torch.cuda.empty_cache()
        por_m[chave] = z
        del model
    if spec["frontend"] == "fusao_colados" or len(ms) == 1:
        return list(por_m.values())[0], info
    zc, coefs = combinar(spec["frontend"], {m: por_m[m] for m in ms}, mval, hp)
    info["stacking"] = coefs
    return zc, info


# ============================================================ E4
def etapa_e4(a, ns, t0):
    import numpy as np
    import pandas as pd
    import torch

    from src.arquivos import salvar_csv, salvar_json, salvar_parquet
    from src.v6_metricas import BootPior, avaliar, regra_desafiante

    if ja_feita(ns, "e4"):
        return 0
    exigir(ns, "e3")
    hp = ns["HP"]
    e3 = ler_json(ns["arq"]("e3_frontend", "json"))
    fe, ms = e3["vencedor"], e3["modelos"]
    tab1 = pd.read_csv(ns["arq"]("e1_camadas", "csv"))
    camada_basica = {}
    for m in ms:
        t_m = tab1[tab1.modelo == m].sort_values(["AUC pior fonte (C=1)", "AUC média (G) (C=1)"], ascending=False)
        camada_basica[m] = [int(t_m.camada.iloc[0])]
    log(f"front-end {fe}; melhor camada única (C=1) de cada modelo: {camada_basica}")
    M, _ = carregar_modelos(ns, tuple(ms) if len(ms) > 1 else (ms[0],))
    mv, md = M[ms[0]].meta("val"), M[ms[0]].meta("dev_shift")
    boot = BootPior(mv, md, hp["N_BOOT"], hp["SEED_BOOT"])
    alvos = {"val": "val", "dev": "dev_shift"}
    linhas, notas = [], {}
    device = torch.device("cuda")

    # ---- RL básica
    spec_rl = {"tipo": "rl", "frontend": fe, "camadas": camada_basica, "pesos": "clipe",
               "C": ({"colados": hp["C_BASICO"]} if fe == "fusao_colados" else {m: hp["C_BASICO"] for m in ms})}
    t = time.time()
    z, coefs = notas_rl_sistema(M, spec_rl, alvos, hp)
    seg = time.time() - t
    notas["RL"] = (z["val"], z["dev"])
    n_par = sum(1024 * len(c) + 1 for c in camada_basica.values())
    linhas.append(resumo_linha("RL básica", avaliar(z["val"], mv, z["dev"], md),
                               {"cabeça": "RL", "seed": "-", "parâmetros": n_par, "segundos": round(seg, 1)}))
    log(f"RL básica: pior fonte {linhas[-1]['AUC pior fonte']:.4f} ({linhas[-1]['fonte pior']}) | {seg:.0f} s")

    # ---- MLP e CNN básicos (3 seeds)
    specs_nn = {"MLP": {"tipo": "mlp", "frontend": fe, "camadas": camada_basica},
                "CNN": {"tipo": "cnn", "frontend": fe, "camadas": {m: list(range(25)) for m in ms}}}
    infos = {}
    for nome, spec in specs_nn.items():
        notas[nome] = []
        for seed in hp["SEEDS"]:
            arq_p = ns["PARCIAL"] / f"e4_{nome}_s{seed}.npz"
            if arq_p.exists():
                zz = np.load(arq_p)
                zv, zd = zz["val"], zz["dev"]
                inf = ler_json(arq_p.with_suffix(".json"))
            else:
                log(f"{nome} básico, seed {seed}:")
                z, inf = treinar_nn_sistema(M, spec, hp["NN"], seed, alvos, device, ns, f"e4_{nome.lower()}")
                zv, zd = z["val"], z["dev"]
                np.savez(arq_p, val=zv, dev=zd)
                salvar_json(inf, arq_p.with_suffix(".json"))
            infos[f"{nome} s{seed}"] = inf
            notas[nome].append((zv, zd))
            linhas.append(resumo_linha(f"{nome} básico s{seed}", avaliar(zv, mv, zd, md),
                                       {"cabeça": nome, "seed": seed, "parâmetros": inf["parametros"],
                                        "segundos": round(inf["segundos"], 1),
                                        "melhor época": json.dumps(inf["melhor_época"])}))
            log(f"{nome} s{seed}: pior fonte {linhas[-1]['AUC pior fonte']:.4f} ({linhas[-1]['fonte pior']})")
        zv = np.mean([n[0] for n in notas[nome]], 0)
        zd = np.mean([n[1] for n in notas[nome]], 0)
        r_seeds = [l_ for l_ in linhas if l_["cabeça"] == nome]
        linhas.append({"sistema": f"{nome} básico (média das 3 seeds, só reportado)", "cabeça": nome, "seed": "média",
                       **{k: float(np.mean([r[k] for r in r_seeds])) for k in r_seeds[0]
                          if isinstance(r_seeds[0][k], (int, float)) and k not in ("seed",)}})
    e1n = pd.read_parquet(ns["arq"]("notas_e1_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e1_dev_shift", "parquet"))
    linhas.append(resumo_linha("v4 (referência: XLS-R concat 5,6,7 + RL, C=0,1)",
                               avaliar(e1n[0].z_v4ref.values, mv, e1n[1].z_v4ref.values, md), {"cabeça": "RL", "seed": "-"}))
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, ns["arq"]("e4_cabecas", "csv"))
    log(tab[["sistema", "AUC pior fonte", "fonte pior", "AUC média (G)", "dev AUC", "acc bal média (G)"]].round(4).to_string(index=False))
    log("regra do desafiante (RL → MLP → CNN; MLP e CNN precisam ganhar nas 3 seeds):")
    venc, comps = regra_desafiante(boot, [("RL", notas["RL"]), ("MLP", notas["MLP"]), ("CNN", notas["CNN"])],
                                   mv, md, hp["DIF_MIN"], log=log)
    cols = {"z_RL": notas["RL"][0]}
    cols_d = {"z_RL": notas["RL"][1]}
    for nome in ("MLP", "CNN"):
        for s, (zv, zd) in zip(hp["SEEDS"], notas[nome]):
            cols[f"z_{nome}_s{s}"], cols_d[f"z_{nome}_s{s}"] = zv, zd
    salvar_parquet(notas_parquet(mv, cols), ns["arq"]("notas_e4_val", "parquet"))
    salvar_parquet(notas_parquet(md, cols_d), ns["arq"]("notas_e4_dev_shift", "parquet"))
    out = {"frontend": fe, "modelos": ms, "camada_basica": camada_basica, "spec_rl_basica": spec_rl,
           "specs_nn": specs_nn, "infos_nn": infos, "comparacoes": comps, "vencedor": venc}
    salvar_json(out, ns["arq"]("e4_resumo", "json"))
    gravar_marcador(ns, "e4", t0, {"vencedor": venc, "camada_basica": camada_basica})
    log(f"==> cabeça vencedora: {venc}\ne4 CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ E5
def etapa_e5(a, ns, t0):
    import numpy as np
    import pandas as pd
    import torch

    from src.arquivos import salvar_csv, salvar_json, salvar_parquet
    from src.v6_cabecas import rl_notas
    from src.v6_metricas import BootPior, avaliar, limiar_regra, loso_limiar

    if ja_feita(ns, "e5"):
        return 0
    exigir(ns, "e4")
    hp = ns["HP"]
    e4 = ler_json(ns["arq"]("e4_resumo", "json"))
    fe, ms, venc = e4["frontend"], e4["modelos"], e4["vencedor"]
    M, _ = carregar_modelos(ns, tuple(ms))
    mv, md = M[ms[0]].meta("val"), M[ms[0]].meta("dev_shift")
    ytr = M[ms[0]].meta("treino").label.values.astype(int)
    boot = BootPior(mv, md, hp["N_BOOT"], hp["SEED_BOOT"])
    alvos = {"val": "val", "dev": "dev_shift"}
    passos = []

    def manter(antes, depois):
        ganho = depois["AUC pior fonte"] - antes["AUC pior fonte"]
        queda = antes.get("dev AUC", 0) - depois.get("dev AUC", 0)
        return bool(ganho >= hp["E5_GANHO_MIN"] and queda <= hp["E5_QUEDA_DEV_MAX"]), ganho, queda

    def registrar(passo, descricao, antes, depois, ficou, ganho, queda, extra=None):
        passos.append({"passo": passo, "modificação": descricao, "AUC pior antes": antes["AUC pior fonte"],
                       "AUC pior depois": depois["AUC pior fonte"], "ganho": ganho,
                       "fonte pior depois": depois["fonte pior"], "dev AUC antes": antes.get("dev AUC"),
                       "dev AUC depois": depois.get("dev AUC"), "queda dev": queda,
                       "acc bal média antes": antes.get("acc bal média (G)"), "acc bal média depois": depois.get("acc bal média (G)"),
                       "mantida": ficou, **(extra or {})})
        log(f"  passo {passo} ({descricao}): pior fonte {antes['AUC pior fonte']:.4f} -> {depois['AUC pior fonte']:.4f} "
            f"(ganho {ganho:+.4f}); dev AUC {antes.get('dev AUC', np.nan):.4f} -> {depois.get('dev AUC', np.nan):.4f} -> "
            f"{'MANTIDA' if ficou else 'descartada'}")

    if venc == "RL":
        spec = json.loads(json.dumps(e4["spec_rl_basica"]))
        z, _ = notas_rl_sistema(M, spec, alvos, hp)
        atual_z, atual = (z["val"], z["dev"]), avaliar(z["val"], mv, z["dev"], md)
        base_z = atual_z
        log(f"base (RL básica): pior fonte {atual['AUC pior fonte']:.4f}")

        def cand_rl(m, cams, C, pesos):
            """Notas de uma RL de um modelo só (com cache em PARCIAL)."""
            chave = f"e5_{m}_{'-'.join(map(str, cams))}_C{C:.6g}_{pesos}"
            arq_p = ns["PARCIAL"] / f"{chave}.npz"
            if arq_p.exists():
                zz = np.load(arq_p)
                return zz["val"], zz["dev"]
            from src.v6_cabecas import pesos_por_fonte
            sw = pesos_por_fonte(M[m].meta("treino")) if pesos == "fonte" else None
            z, _, _ = rl_notas(M[m].X("treino", cams), ytr, {"val": M[m].X("val", cams), "dev": M[m].X("dev_shift", cams)},
                               C, hp["RL_MAX_ITER"], sw)
            np.savez(arq_p, val=z["val"], dev=z["dev"])
            return z["val"], z["dev"]

        # (1) zona de camadas: camada única ou 3 vizinhas, por modelo, pela pior fonte do próprio modelo
        nova = json.loads(json.dumps(spec))
        zonas_tab = []
        for m in ms:
            Cm = spec["C"].get(m, spec["C"].get("colados", hp["C_BASICO"]))
            opc = [[l] for l in range(25)] + [[l - 1, l, l + 1] for l in range(1, 24)]
            melhores = []
            for cams in opc:
                zv, zd = cand_rl(m, cams, Cm, spec["pesos"])
                r = avaliar(zv, mv, zd, md)
                zonas_tab.append({"modelo": m, "zona": str(cams), "AUC pior fonte": r["AUC pior fonte"],
                                  "fonte pior": r["fonte pior"], "AUC média (G)": r.get("AUC média (G)"), "dev AUC": r.get("dev AUC")})
                melhores.append((r["AUC pior fonte"], r.get("AUC média (G)", 0), cams))
            melhores.sort(key=lambda x: (x[0], x[1]), reverse=True)
            nova["camadas"][m] = melhores[0][2]
            log(f"  zona de {m}: melhor {melhores[0][2]} (pior fonte {melhores[0][0]:.4f})")
        salvar_csv(pd.DataFrame(zonas_tab), ns["arq"]("e5_zonas", "csv"))
        z, _ = notas_rl_sistema(M, nova, alvos, hp)
        dep = avaliar(z["val"], mv, z["dev"], md)
        ficou, g, q = manter(atual, dep)
        registrar(1, f"zona de camadas: {nova['camadas']}", atual, dep, ficou, g, q, {"zona": json.dumps(nova["camadas"])})
        if ficou:
            spec, atual, atual_z = nova, dep, (z["val"], z["dev"])

        # (2) grade ampla de C, pela pior fonte (por modelo; nos colados, a RL única)
        nova = json.loads(json.dumps(spec))
        grade_tab = []
        chaves = ["colados"] if fe == "fusao_colados" else ms
        for m in chaves:
            res = []
            for C in hp["GRADE_C_AMPLA"]:
                if fe == "fusao_colados":
                    s2 = json.loads(json.dumps(spec))
                    s2["C"]["colados"] = C
                    zz, _ = notas_rl_sistema(M, s2, alvos, hp)
                    zv, zd = zz["val"], zz["dev"]
                else:
                    zv, zd = cand_rl(m, spec["camadas"][m], C, spec["pesos"])
                r = avaliar(zv, mv, zd, md)
                grade_tab.append({"modelo": m, "C": C, "AUC pior fonte": r["AUC pior fonte"], "fonte pior": r["fonte pior"],
                                  "dev AUC": r.get("dev AUC")})
                res.append((r["AUC pior fonte"], r.get("AUC média (G)", 0), C))
            res.sort(key=lambda x: (x[0], x[1]), reverse=True)
            nova["C"][m] = res[0][2]
            log(f"  C de {m}: melhor {res[0][2]:g} (pior fonte {res[0][0]:.4f})")
        salvar_csv(pd.DataFrame(grade_tab), ns["arq"]("e5_grade_c", "csv"))
        z, _ = notas_rl_sistema(M, nova, alvos, hp)
        dep = avaliar(z["val"], mv, z["dev"], md)
        ficou, g, q = manter(atual, dep)
        registrar(2, f"grade ampla de C: {nova['C']}", atual, dep, ficou, g, q, {"C": json.dumps(nova["C"])})
        if ficou:
            spec, atual, atual_z = nova, dep, (z["val"], z["dev"])

        # (3) pesos iguais por fonte
        nova = json.loads(json.dumps(spec))
        nova["pesos"] = "fonte"
        z, _ = notas_rl_sistema(M, nova, alvos, hp)
        dep = avaliar(z["val"], mv, z["dev"], md)
        ficou, g, q = manter(atual, dep)
        registrar(3, "pesos iguais por fonte (sample_weight)", atual, dep, ficou, g, q)
        if ficou:
            spec, atual, atual_z = nova, dep, (z["val"], z["dev"])
        spec_final = spec
    else:
        # MLP ou CNN: wd, dropout, épocas, ensemble (média das 3 seeds na regra)
        spec = json.loads(json.dumps(e4["specs_nn"][venc]))
        hpn = dict(hp["NN"])
        device = torch.device("cuda")

        def treinar_todas(hpn_, nome):
            res = []
            for seed in hp["SEEDS"]:
                arq_p = ns["PARCIAL"] / f"e5_{nome}_s{seed}.npz"
                if arq_p.exists():
                    zz = np.load(arq_p)
                    res.append((zz["val"], zz["dev"]))
                    continue
                z, _ = treinar_nn_sistema(M, spec, hpn_, seed, alvos, device, ns, f"e5_{nome}")
                np.savez(arq_p, val=z["val"], dev=z["dev"])
                res.append((z["val"], z["dev"]))
            return res

        def media_seeds(res):
            rr = [avaliar(v, mv, d, md) for v, d in res]
            out = dict(rr[0])
            for k in ("AUC pior fonte", "AUC média (G)", "dev AUC", "acc bal média (G)"):
                out[k] = float(np.mean([r[k] for r in rr]))
            return out
        # a base é a da E4 (mesmas seeds): lê as notas gravadas
        atual_res = [tuple(np.load(ns["PARCIAL"] / f"e4_{venc}_s{s}.npz")[k] for k in ("val", "dev")) for s in hp["SEEDS"]]
        atual = media_seeds(atual_res)
        base_z = atual_res[0]
        ckpt_atual = f"e4_{venc.lower()}"
        mods = [("weight decay 1e-2", {"WD": hp["E5_NN"]["WD"]}), ("dropout 0,5", {"DROPOUT": hp["E5_NN"]["DROPOUT"], "DROPOUT_CNN": hp["E5_NN"]["DROPOUT"]}),
                ("até 60 épocas, paciência 10", {"MAX_EPOCAS": hp["E5_NN"]["MAX_EPOCAS"], "PACIENCIA": hp["E5_NN"]["PACIENCIA"]})]
        for i, (desc, mud) in enumerate(mods, 1):
            hp2 = {**hpn, **mud}
            res = treinar_todas(hp2, f"{venc.lower()}_p{i}")
            dep = media_seeds(res)
            ficou, g, q = manter(atual, dep)
            registrar(i, desc, atual, dep, ficou, g, q)
            if ficou:
                hpn, atual, atual_res, ckpt_atual = hp2, dep, res, f"e5_{venc.lower()}_p{i}"
        ens = (np.mean([r[0] for r in atual_res], 0), np.mean([r[1] for r in atual_res], 0))
        dep = avaliar(*ens, mv, md)
        ficou, g, q = manter(atual, dep)
        registrar(4, "ensemble das 3 seeds (média dos logits)", atual, dep, ficou, g, q)
        atual_z = ens if ficou else atual_res[0]
        atual = dep if ficou else avaliar(*atual_res[0], mv, md)
        spec_final = {**spec, "hp_nn": hpn, "ensemble": ficou, "seed_sem_ensemble": hp["SEEDS"][0],
                      "ckpt_nome": ckpt_atual, "seeds": list(hp["SEEDS"]) if ficou else [hp["SEEDS"][0]]}

    # (último) regra de limiar: A, C ou D, por validação cruzada deixando uma fonte de fora
    zv, zd = atual_z
    lo = loso_limiar(zv, mv, ("A", "C", "D"), hp["FONTES_VAL"], hp["E5_FPR_D"])
    med = lo.groupby("regra")["acurácia balanceada"].mean()
    melhor_r = med.idxmax()
    regra = melhor_r if (melhor_r == "A" or med[melhor_r] - med.get("A", -1) >= hp["E5_LIMIAR_DIF_MIN"]) else "A"
    lims = {r: limiar_regra(r, zv, mv, hp["FONTES_VAL"], hp["E5_FPR_D"]) for r in ("A", "C", "D")}
    from src.metrics import acuracias
    conf_dev = {r: acuracias(md.label.values, zd, t)["acurácia balanceada"] for r, t in lims.items()}
    log("regra de limiar (acurácia balanceada média, deixando uma fonte de fora):\n" + med.round(4).to_string()
        + f"\n  -> regra {regra}; limiares {json.dumps({k: round(v, 3) for k, v in lims.items()})}; "
          f"dev_shift (só confirma): {json.dumps({k: round(v, 4) for k, v in conf_dev.items()})}")
    passos.append({"passo": len(passos) + 1, "modificação": f"regra de limiar: {regra}", "mantida": regra != "A",
                   "AUC pior antes": atual["AUC pior fonte"], "AUC pior depois": atual["AUC pior fonte"], "ganho": 0.0,
                   "acc bal LOSO A": float(med.get("A", np.nan)), "acc bal LOSO C": float(med.get("C", np.nan)),
                   "acc bal LOSO D": float(med.get("D", np.nan)), "dev acc bal A": conf_dev["A"],
                   "dev acc bal C": conf_dev["C"], "dev acc bal D": conf_dev["D"]})
    comp_base = boot.comparar(base_z, atual_z, hp["DIF_MIN"], mv, md)
    tab = pd.DataFrame(passos)
    salvar_csv(tab, ns["arq"]("e5_passos", "csv"))
    salvar_csv(lo, ns["arq"]("e5_limiar_loso", "csv"))
    salvar_parquet(notas_parquet(mv, {"z_final": zv, "z_base": base_z[0]}), ns["arq"]("notas_e5_val", "parquet"))
    salvar_parquet(notas_parquet(md, {"z_final": zd, "z_base": base_z[1]}), ns["arq"]("notas_e5_dev_shift", "parquet"))
    out = {"cabeca": venc, "frontend": fe, "spec_final": spec_final, "regra_limiar": regra, "limiares": lims,
           "loso_media": med.to_dict(), "dev_acc_bal_por_regra": conf_dev, "metricas_final": atual,
           "final_menos_base_bootstrap": comp_base, "passos": passos}
    salvar_json(out, ns["arq"]("e5_resumo", "json"))
    gravar_marcador(ns, "e5", t0, {"spec_final": spec_final, "regra_limiar": regra,
                                   "pior_fonte_final": atual["AUC pior fonte"]})
    log(f"==> sistema final: {json.dumps(spec_final)} | limiar {regra}\ne5 CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ finalistas (antes da E6)
def etapa_finalistas(a, ns, t0):
    import pandas as pd

    from src.arquivos import salvar_json
    from src.v6_metricas import avaliar, limiar_regra

    if ja_feita(ns, "finalistas"):
        return 0
    exigir(ns, "e1", "e2", "e3", "e4", "e5")
    hp = ns["HP"]
    e4, e5 = ler_json(ns["arq"]("e4_resumo", "json")), ler_json(ns["arq"]("e5_resumo", "json"))
    n1v, n1d = pd.read_parquet(ns["arq"]("notas_e1_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e1_dev_shift", "parquet"))
    n4v, n4d = pd.read_parquet(ns["arq"]("notas_e4_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e4_dev_shift", "parquet"))
    n5v, n5d = pd.read_parquet(ns["arq"]("notas_e5_val", "parquet")), pd.read_parquet(ns["arq"]("notas_e5_dev_shift", "parquet"))
    ref = hp["V4_REF"]
    regra = e5["regra_limiar"]
    F = {"F0": {"descrição": "v4: XLS-R (Meta) concat_top3 (camadas 5, 6, 7) + RL, C = 0,1",
                "spec": {"tipo": "rl", "frontend": "xlsr", "camadas": {"xlsr": ref["camadas"]}, "C": {"xlsr": ref["C"]},
                         "pesos": "clipe"}, "limiar_principal": "A", "notas": (n1v.z_v4ref.values, n1d.z_v4ref.values)},
         "F1": {"descrição": f"front-end vencedor ({e4['frontend']}) + RL básica (camada {e4['camada_basica']}, C = 1)",
                "spec": e4["spec_rl_basica"], "limiar_principal": "A", "notas": (n4v.z_RL.values, n4d.z_RL.values)},
         "F2": {"descrição": f"sistema final da E5 ({e5['cabeca']}, {e5['frontend']}), limiar {regra}",
                "spec": {"tipo": e5["cabeca"].lower(), **e5["spec_final"]} if e5["cabeca"] != "RL" else e5["spec_final"],
                "limiar_principal": regra, "notas": (n5v.z_final.values, n5d.z_final.values)}}
    out = {"gravado_em": datetime.now().isoformat(timespec="seconds"),
           "aviso": "gravado ANTES da E6; nada aqui pode mudar depois de ver o teste",
           "regra_limiar_e5": regra, "finalistas": {}}
    for k, f in F.items():
        zv, zd = f["notas"]
        lims = {"A": limiar_regra("A", zv, n1v), "B": avaliar(zv, n1v, zd, n1d)["limiar B (logit)"],
                f"E5 ({regra})": limiar_regra(regra, zv, n1v, hp["FONTES_VAL"], hp["E5_FPR_D"])}
        out["finalistas"][k] = {"descrição": f["descrição"], "spec": f["spec"], "limiar_principal": f["limiar_principal"],
                                "limiares_logit": lims, "metricas_val_dev": avaliar(zv, n1v, zd, n1d)}
    usa = sorted({m for f in out["finalistas"].values() for m in FRONTEND_MODELOS[f["spec"]["frontend"]]})
    out["modelos_usados"] = usa
    out["precisa_extrair_teste_wavlm"] = "wavlm" in usa
    salvar_json(out, ns["arq"]("finalistas", "json"))
    gravar_marcador(ns, "finalistas", t0, {"precisa_extrair_teste_wavlm": out["precisa_extrair_teste_wavlm"],
                                           "finalistas": {k: v["descrição"] for k, v in out["finalistas"].items()}})
    log(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:3000])
    log("finalistas CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ relatório (funil, gráficos, projeção)
def etapa_relatorio(a, ns, t0):
    from src.v6_relatorio import gerar
    gerar(ns, log)
    gravar_marcador(ns, "relatorio", t0, {})
    log("relatorio CONCLUÍDO\nSTATUS=CONCLUIDO")
    return 0


# ============================================================ status
def etapa_status(a, ns, t0):
    from src.v6_relatorio import escrever_status
    escrever_status(ns)
    return 0


def etapa_concluir(a, ns, t0):
    from src.arquivos import salvar_json
    from src.v6_relatorio import ETAPAS_TODAS
    obrig = [e for e in ETAPAS_TODAS if e != "medir"]
    faltam = [e for e in obrig if not ns["marcador"](e).exists()]
    if faltam:
        log(f"ainda faltam: {faltam}\nSTATUS=INCOMPLETO")
        return 3
    salvar_json({"versao": "v6", "quick": ns["QUICK"], "concluido_em": datetime.now().isoformat(timespec="seconds"),
                 "etapas": {e: (ler_json(ns["marcador"](e)) or {}).get("concluido_em") for e in ETAPAS_TODAS}},
                ns["arq"]("concluido", "json"))
    log("v6 CONCLUÍDA\nSTATUS=CONCLUIDO")
    return 0


ETAPAS_FN = {"concluir": etapa_concluir, "medir": etapa_medir, "extrair": etapa_extrair, "e1": etapa_e1, "e2": etapa_e2, "e3": etapa_e3,
             "e4": etapa_e4, "e5": etapa_e5, "finalistas": etapa_finalistas, "relatorio": etapa_relatorio,
             "status": etapa_status}


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
        log(f"\n######## {datetime.now().isoformat(timespec='seconds')} | {' '.join(sys.argv)} | pid {os.getpid()}")
    from src.memoria import Amostrador
    AMOSTRADOR = Amostrador()
    if not leve:
        AMOSTRADOR.start()
    t0 = time.time()
    sem_suspensao(not leve)
    try:
        src_cfg = celula_config(a.notebook)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns = executar_config(src_cfg, a.quick, a.notebook)
        ns["_SRC_CONFIG"] = src_cfg
        if ns.get("VERSAO") != "v6":
            raise RuntimeError("a célula config do notebook não é a da v6")
        assert ns["AVALIAR_TESTE_FINAL"] is False, "este script não avalia o teste final"
        ns["PARCIAL"].mkdir(parents=True, exist_ok=True)
        return ETAPAS_FN[a.etapa](a, ns, t0)
    except BaseException:
        traceback.print_exc()
        log("STATUS=FALHOU")
        return 1
    finally:
        AMOSTRADOR.parar = True
        sem_suspensao(False)
        if not leve:
            log(f"PICO_MEMORIA_GB={AMOSTRADOR.pico_proc / 1024:.2f} (este processo + workers) | pico do sistema "
                f"{AMOSTRADOR.pico_sis:.1f} GB | {(time.time() - t0) / 60:.1f} min neste processo")


if __name__ == "__main__":
    sys.exit(main())
