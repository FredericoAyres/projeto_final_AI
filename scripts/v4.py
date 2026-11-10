#!/usr/bin/env python3
"""v4/v5: modelo auto-supervisionado congelado (v4: XLS-R 300M; v5: XLS-R 300M e WavLM Large) + cabeças leves.
Uma etapa por processo. Os arquivos levam o prefixo da configuração (`PREFIXO`; sem ele, "v4").

Os hiperparâmetros NÃO são copiados aqui: o script executa a célula com a tag "config" do notebook.ipynb.
Uso (na raiz do repositório, com o env ativo; scripts\\rodar_v4.ps1 roda tudo na ordem):

    python scripts/v4.py medir      [--quick] [--n 500]   # vazão (clipes/s), não grava cache
    python scripts/v4.py extrair    [--quick]             # extração em shards, orçamento cumulativo
    python scripts/v4.py camadas    [--quick]             # regressão logística por camada + concatenação + ablação do corte
    python scripts/v4.py somas      [--quick]             # soma ponderada (linear e MLP), 3 seeds, + ensembles
    python scripts/v4.py atalhos    [--quick]             # Parte F (diagnóstico de atalhos)
    python scripts/v4.py resumo     [--quick]             # tabelas finais, gráfico, escolha pela validação
    python scripts/v4.py status     [--quick]             # só reescreve logs/v4_resumo.txt (leve)
    python scripts/v4.py concluir   [--quick]             # grava results/v4_concluido.json se tudo terminou
    opções: --num-workers N (padrão: V4_NUM_WORKERS ou NUM_WORKERS da configuração) | --log arquivo

Cada etapa grava os seus arquivos de forma atômica e um MARCADOR por último
(results/v4_<etapa>_concluido.json); com o marcador presente, a etapa não faz nada. Sem ele, rodar de novo
continua de onde parou (shards prontos, camadas prontas e seeds prontas são pulados).
O teste final NUNCA é lido aqui (só train, val e dev_shift).
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


# ============================================================ utilidades
def ler_json(p, padrao=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao


def prefixo(ns):
    """Prefixo dos arquivos de resultado: "v4" (padrão, a v4 como foi rodada) ou o PREFIXO da configuração
    (v5: "v5_xlsr" ou "v5_wavlm"). Mudança de execução, não de método."""
    return ns.get("PREFIXO", "v4")


def manifest_e_sha():
    from src.arquivos import sha256_arquivo
    from src.experimento import ler_manifest
    arq = ROOT / "data/processed/manifest.parquet"
    df = ler_manifest(arq)
    assert isinstance(df.index, __import__("pandas").RangeIndex)
    return df, sha256_arquivo(arq)


def gravar_marcador(ns, etapa, t0, extra):
    from src.arquivos import identificar_codigo, salvar_json, sha256_texto
    codigo = identificar_codigo(ROOT, extras=["scripts/v4.py", "scripts/rodar_v4.ps1", "scripts/vigia_v4.ps1"])
    codigo["celula_config_sha256"] = sha256_texto(ns["_SRC_CONFIG"])
    info = {"etapa": etapa, "versao": prefixo(ns), "quick": ns["QUICK"], "status": "concluido",
            "concluido_em": datetime.now().isoformat(timespec="seconds"),
            "minutos_neste_processo": round((time.time() - t0) / 60, 2),
            "pico_memoria_gb_processo_e_workers": round(AMOSTRADOR.pico_proc / 1024, 2),
            "pico_memoria_gb_sistema": round(AMOSTRADOR.pico_sis, 2), "codigo": codigo, **extra}
    salvar_json(info, ns["marcador"](etapa))
    return info


def exigir(ns, etapa):
    if not ns["marcador"](etapa).exists():
        raise RuntimeError(f"a etapa '{etapa}' ainda não terminou ({ns['marcador'](etapa)} não existe)")


def carregar_conjuntos(ns, nomes=("treino", "val", "dev_shift")):
    from src.cache_v4 import Conjunto
    out = {n: Conjunto(ns["CACHE_DIR"], n) for n in nomes}
    for n, c in out.items():
        if len(c) == 0:
            raise RuntimeError(f"conjunto '{n}' sem nenhum shard concluído em {ns['CACHE_DIR']}")
        c.ok = (c.meta.erro_leitura.fillna(0).values == 0)
        if (~c.ok).any():
            print(f"  aviso: {n}: {(~c.ok).sum()} áudio(s) com erro de leitura ficam de fora", flush=True)
    return out


def num_workers(a, ns):
    if a.num_workers is not None:
        return a.num_workers
    return int(os.environ.get("V4_NUM_WORKERS", ns["NUM_WORKERS"]))


# ============================================================ medir
def etapa_medir(a, ns, t0):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.arquivos import salvar_json
    from src.cache_v4 import plano_treino
    from src.extracao_v4 import Baldes, LotesFixos, montar_lotes
    from src.preproc import ClipesV4, juntar_clipes
    from src.ssl import carregar_ssl

    hp = dict(ns["HP"])
    if a.balde_s:
        hp["BALDE_S"] = a.balde_s
    df, _ = manifest_e_sha()
    idx = plano_treino(df[df.split == "train"], max(2, a.n - a.n % 2), 1, hp["SEED"] + 1)[0]
    g = df.loc[idx].sort_values("duration_s", kind="stable")
    nw = num_workers(a, ns)
    t = time.time()
    model = carregar_ssl(hp["MODELO_SSL"], "cuda", hp["DTYPE"])
    t_carga = time.time() - t
    ds = ClipesV4(g.path.values, np.arange(len(g)), cortar_silencio=True, top_db=hp["TOP_DB"],
                  min_s=hp["MIN_S"], max_s=hp["MAX_S"])
    lotes = montar_lotes(g.duration_s.values, hp["MAX_AMOSTRAS_LOTE"], hp["MAX_CLIPES_LOTE"])
    dl = DataLoader(ds, batch_sampler=LotesFixos(lotes), num_workers=nw, collate_fn=juntar_clipes,
                    persistent_workers=False, prefetch_factor=4 if nw else None)
    from src.memoria import processos
    torch.cuda.reset_peak_memory_stats()
    t, n, t_primeiro, t_gpu, nan = time.time(), 0, None, 0.0, 0
    por_processo = []
    baldes = Baldes(model, hp, "cuda")          # o mesmo caminho da extração (lotes da GPU com formato fixo)
    for k, (clipes, est, pos) in enumerate(dl):
        if k == len(lotes) // 2:          # memória reservada de cada processo no meio da passada
            por_processo = [{"papel": p, "pid": pid, "commit_gb": round(c / 1024, 2)} for p, pid, _, c in processos()]
        if t_primeiro is None:
            t_primeiro = time.time() - t
        tg = time.time()
        res = baldes.adicionar(clipes, pos) + (baldes.esvaziar() if k == len(lotes) - 1 else [])
        torch.cuda.synchronize()
        t_gpu += time.time() - tg
        nan += sum(int((~np.isfinite(r[1])).any(axis=(1, 2)).sum()) for r in res)
        n += sum(len(r[0]) for r in res)
    total = time.time() - t
    r = {"n": n, "num_workers": nw, "segundos_total": round(total, 1), "segundos_ate_1o_lote": round(t_primeiro, 1),
         "clipes_por_s": round(n / total, 2), "clipes_por_s_sem_partida": round(n / max(total - t_primeiro, 1e-9), 2),
         "segundos_gpu": round(t_gpu, 1), "clipes_por_s_so_gpu": round(n / max(t_gpu, 1e-9), 2),
         "carga_modelo_s": round(t_carga, 1), "pico_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
         "clipes_com_nan": nan, "duracao_media_manifest_s": round(float(g.duration_s.mean()), 2),
         "medido_em": datetime.now().isoformat(timespec="seconds"),
         "pico_memoria_gb_processo_e_workers": round(AMOSTRADOR.pico_proc / 1024, 2),
         "pico_memoria_gb_sistema": round(AMOSTRADOR.pico_sis, 2), "memoria_por_processo_no_meio": por_processo,
         "fontes": g.source_dataset.value_counts().to_dict(), "lotes_gpu": baldes.lotes_gpu,
         "linhas_vazias": baldes.linhas_vazias, "refeitos_bf16": baldes.refeitos_bf16,
         "balde_s": hp.get("BALDE_S", 1)}
    salvar_json(r, ns["arq_result"]("vazao" if not a.balde_s else f"vazao_balde{a.balde_s}", "json"))
    print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
    return 0


# ============================================================ extrair
def etapa_extrair(a, ns, t0):
    from src.cache_v4 import Orcamento, carregar_ou_criar_plano, limpar_temporarios, shard_pronto
    from src.extracao_v4 import extrair
    from src.ssl import carregar_ssl

    hp = ns["HP"]
    if ns["marcador"]("extracao").exists():
        print("extração: já concluída (marcador presente); nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    df, sha = manifest_e_sha()
    plano = carregar_ou_criar_plano(df, hp, ns["CACHE_DIR"], ns["CFG_EXTRACAO"], sha)
    n_tmp = limpar_temporarios(ns["CACHE_DIR"])
    if n_tmp:
        print(f"extração: {n_tmp} shard(s) incompleto(s) de uma execução interrompida foram descartados", flush=True)
    resumo_plano = plano.groupby("conjunto", sort=False).agg(clipes=("idx_manifest", "size"), shards=("shard", "nunique"))
    print("plano:\n" + resumo_plano.to_string(), flush=True)
    orc = Orcamento(ns["ARQ_ESTADO"], hp["ORCAMENTO_H"])
    print(f"orçamento: {orc.base / 3600:.2f} h já gastas de {hp['ORCAMENTO_H']} h (tentativa "
          f"{orc.estado['tentativas_extracao']})", flush=True)
    nw = num_workers(a, ns)
    r = {"parou_por": "orçamento já esgotado antes de começar", "shards_feitos": 0}
    try:
        if orc.resta() > 0:
            model = carregar_ssl(hp["MODELO_SSL"], "cuda", hp["DTYPE"])
            r = extrair(df, plano, model, hp, ns["CACHE_DIR"], orc, ns["CFG_EXTRACAO"],
                        log=lambda s: print(s, flush=True), num_workers=nw)
    finally:
        orc.fechar()
    feitos = {}
    for conj, g in plano.groupby("conjunto", sort=False):
        shards = g.shard.unique()
        prontos = [int(s) for s in shards if shard_pronto(ns["CACHE_DIR"], conj, s)]
        feitos[conj] = {"shards_prontos": len(prontos), "shards_planejados": int(len(shards)),
                        "clipes_prontos": int(g[g.shard.isin(prontos)].shape[0]), "clipes_planejados": int(len(g))}
    print("feito até agora:\n" + json.dumps(feitos, ensure_ascii=False, indent=1), flush=True)
    completo = all(v["shards_prontos"] == v["shards_planejados"] for v in feitos.values())
    esgotou = orc.resta() <= 0 or r["parou_por"].startswith("orçamento")
    if not (completo or esgotou):
        print(f"extração incompleta e ainda há orçamento ({r['parou_por']}); rode de novo.\nSTATUS=INCOMPLETO", flush=True)
        return 2
    dev_val_ok = all(feitos[c]["shards_prontos"] == feitos[c]["shards_planejados"] for c in ("dev_shift", "val"))
    info = gravar_marcador(ns, "extracao", t0, {
        "plano_completo": completo, "parou_por": r["parou_por"], "dev_e_val_completos": dev_val_ok,
        "horas_extracao_acumuladas": round(orc.gasto() / 3600, 3), "orcamento_h": hp["ORCAMENTO_H"],
        "conjuntos": feitos, "ultima_passada": r, "num_workers": nw, "extracao": ns["CFG_EXTRACAO"]})
    print(f"extração CONCLUÍDA ({'plano completo' if completo else 'orçamento esgotado'}); "
          f"{info['horas_extracao_acumuladas']:.2f} h acumuladas.\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ camadas
def etapa_camadas(a, ns, t0):
    import numpy as np
    import pandas as pd

    from src.arquivos import salvar_csv, salvar_json, salvar_parquet
    from src.cabecas import N_SAIDAS, ajustar_rl, avaliar, media_desvio_treino
    from src.cache_v4 import Conjunto

    hp = ns["HP"]
    if ns["marcador"]("camadas").exists():
        print("camadas: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    exigir(ns, "extracao")
    ext = ler_json(ns["marcador"]("extracao"))
    if not ext["dev_e_val_completos"]:
        raise RuntimeError("dev_shift ou validação incompletos: não dá para avaliar as cabeças")
    C = carregar_conjuntos(ns)
    tr, va, dv = C["treino"], C["val"], C["dev_shift"]
    print(f"treino {len(tr)} ({tr.meta.label.mean():.1%} fake, {len(tr.shards)} shards) | val {len(va)} | dev {len(dv)}", flush=True)
    parc = ns["RESULTS_DIR"] / f"{prefixo(ns)}_parcial{ns['SUFIXO']}"
    parc.mkdir(parents=True, exist_ok=True)
    arq_pad = parc / "padronizacao.npz"
    if arq_pad.exists() and int(np.load(arq_pad)["n"]) == len(tr):
        z = np.load(arq_pad)
        mu, sd = z["mu"], z["sd"]
    else:
        mu, sd = media_desvio_treino(tr)
        np.savez(arq_pad, mu=mu, sd=sd, n=len(tr))
    ytr = tr.meta.label.values[tr.ok].astype(int)
    mv, md = va.meta[va.ok].reset_index(drop=True), dv.meta[dv.ok].reset_index(drop=True)

    def X(conj, l):
        return (conj.camada(l)[conj.ok] - mu[l]) / sd[l]

    linhas, zv_all, zd_all = [], {}, {}
    for l in range(N_SAIDAS):
        arq = parc / f"camada_{l:02d}.json"
        arq_z = parc / f"camada_{l:02d}_notas.npz"
        if arq.exists() and arq_z.exists():
            r = ler_json(arq)
            if r["n_treino"] == int(tr.ok.sum()):
                z = np.load(arq_z)
                zv_all[l], zd_all[l] = z["val"], z["dev"]
                linhas.append(r)
                print(f"camada {l:2d}: já feita (C={r['C']:g}, val EER média/fonte {100 * r['val EER média por fonte']:.2f}%)", flush=True)
                continue
        t = time.time()
        print(f"camada {l:2d}:", flush=True)
        Xtr, Xv, Xd = X(tr, l), X(va, l), X(dv, l)
        rl, c_best, grade = ajustar_rl(Xtr, ytr, Xv, mv, hp["GRADE_C"], hp["RL_MAX_ITER"],
                                       log=lambda s: print(s, flush=True))
        zv, zd = rl.decision_function(Xv), rl.decision_function(Xd)
        r = {"camada": l, "C": c_best, "n_treino": int(tr.ok.sum()), **avaliar(zv, mv, zd, md),
             "iterações": int(grade.loc[grade.C == c_best, "iterações"].iloc[0]),
             "convergiu": bool(grade.loc[grade.C == c_best, "convergiu"].iloc[0]),
             "grade": grade.to_dict("records"), "segundos": round(time.time() - t, 1)}
        np.savez(arq_z, val=zv, dev=zd)
        salvar_json(r, arq)
        zv_all[l], zd_all[l] = zv, zd
        linhas.append(r)
        print(f"  -> C={c_best:g} | val EER média/fonte {100 * r['val EER média por fonte']:.2f}% | "
              f"dev AUC {r['dev AUC']:.3f} EER {100 * r['dev EER']:.1f}% acc. bal. {r['dev acurácia balanceada']:.3f} | "
              f"{r['segundos']:.0f} s", flush=True)
    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "grade"} for r in linhas]).sort_values("camada")
    salvar_csv(tab, ns["arq_result"]("camadas", "csv"))
    cols_meta = ["idx_manifest", "path", "source_dataset", "label", "generator", "speaker_id", "duration_s"]
    for nome, meta, zz in [("val", mv, zv_all), ("dev_shift", md, zd_all)]:
        notas = meta[cols_meta].copy()
        for l in range(N_SAIDAS):
            notas[f"z_L{l:02d}"] = zz[l]
        salvar_parquet(notas, ns["arq_result"](f"notas_camadas_{nome}", "parquet"))

    # ---- concatenação das melhores camadas (pela validação)
    ordem = tab.sort_values("val EER média por fonte").camada.tolist()
    concat = []
    for k in hp["TOP_K_CONCAT"]:
        cams = sorted(ordem[:k])
        arq = parc / f"concat_{k}.json"
        if arq.exists() and ler_json(arq)["camadas"] == cams and ler_json(arq)["n_treino"] == int(tr.ok.sum()):
            concat.append(ler_json(arq))
            print(f"concatenação top-{k}: já feita", flush=True)
            continue
        t = time.time()
        print(f"concatenação top-{k} (camadas {cams}):", flush=True)
        Xtr = np.hstack([X(tr, l) for l in cams])
        Xv = np.hstack([X(va, l) for l in cams])
        Xd = np.hstack([X(dv, l) for l in cams])
        rl, c_best, grade = ajustar_rl(Xtr, ytr, Xv, mv, hp["GRADE_C"], hp["RL_MAX_ITER"],
                                       log=lambda s: print(s, flush=True))
        del Xtr
        zv, zd = rl.decision_function(Xv), rl.decision_function(Xd)
        r = {"cabeça": f"concat_top{k}", "camadas": cams, "C": c_best, "n_treino": int(tr.ok.sum()),
             **avaliar(zv, mv, zd, md), "grade": grade.to_dict("records"), "segundos": round(time.time() - t, 1)}
        salvar_parquet(mv[cols_meta].assign(z=zv), ns["arq_result"](f"notas_concat_top{k}_val", "parquet"))
        salvar_parquet(md[cols_meta].assign(z=zd), ns["arq_result"](f"notas_concat_top{k}_dev_shift", "parquet"))
        salvar_json(r, arq)
        concat.append(r)
        print(f"  -> val EER média/fonte {100 * r['val EER média por fonte']:.2f}% | dev AUC {r['dev AUC']:.3f}", flush=True)

    # ---- ablação do corte de silêncio (só se os conjuntos *_semcorte foram extraídos)
    ablacao = None
    try:
        sc = {n: Conjunto(ns["CACHE_DIR"], n) for n in ("treino_semcorte", "val_semcorte", "dev_shift_semcorte")}
        plano = pd.read_parquet(ns["CACHE_DIR"] / "plano.parquet")
        completo = all(len(sc[n]) == int((plano.conjunto == n).sum()) and len(sc[n]) > 0 for n in sc)
    except Exception as e:  # noqa: BLE001
        completo, sc = False, None
        print(f"ablação do corte: não disponível ({e})", flush=True)
    if completo:
        l = ordem[0]
        c_l = float(tab.loc[tab.camada == l, "C"].iloc[0])
        print(f"ablação do corte de silêncio (camada {l}, C={c_l:g}, mesmos áudios com e sem corte):", flush=True)
        ablacao = {"camada": l, "C": c_l}
        from sklearn.linear_model import LogisticRegression
        for rotulo, (ntr, nva, ndv) in {"com corte": ("treino", "val", "dev_shift"),
                                         "sem corte": ("treino_semcorte", "val_semcorte", "dev_shift_semcorte")}.items():
            fontes = {"treino": tr, "val": va, "dev_shift": dv}
            ids = {n: sc[n + "_semcorte"].meta.idx_manifest.values for n in ("treino", "val", "dev_shift")}
            mats, metas = {}, {}
            for base, nome in (("treino", ntr), ("val", nva), ("dev_shift", ndv)):
                if nome.endswith("semcorte"):
                    cj = sc[nome]
                    Xm, mt = cj.camada(l), cj.meta.reset_index(drop=True)
                else:
                    cj = fontes[base]
                    pos = pd.Series(np.arange(len(cj.meta)), index=cj.meta.idx_manifest.values)
                    sel = pos.loc[ids[base]].values
                    Xm, mt = cj.camada(l)[sel], cj.meta.iloc[sel].reset_index(drop=True)
                mats[base], metas[base] = Xm, mt
            m_, s_ = mats["treino"].mean(0), mats["treino"].std(0) + 1e-6
            rl = LogisticRegression(C=c_l, max_iter=hp["RL_MAX_ITER"], tol=1e-4).fit(
                (mats["treino"] - m_) / s_, metas["treino"].label.values.astype(int))
            zv = rl.decision_function((mats["val"] - m_) / s_)
            zd = rl.decision_function((mats["dev_shift"] - m_) / s_)
            ablacao[rotulo] = avaliar(zv, metas["val"], zd, metas["dev_shift"])
            ablacao[rotulo]["n_treino"] = len(metas["treino"])
            print(f"  {rotulo}: val EER média/fonte {100 * ablacao[rotulo]['val EER média por fonte']:.2f}% | "
                  f"dev AUC {ablacao[rotulo]['dev AUC']:.3f} EER {100 * ablacao[rotulo]['dev EER']:.1f}%", flush=True)
        ablacao["exemplos_silencio"] = {
            "silencio_inicio_mediana_s": float(sc["dev_shift_semcorte"].meta.silencio_inicio_s.median()),
            "silencio_fim_mediana_s": float(sc["dev_shift_semcorte"].meta.silencio_fim_s.median())}
        salvar_json(ablacao, ns["arq_result"]("ablacao_corte", "json"))
    else:
        print("ablação do corte: conjuntos *_semcorte não extraídos (orçamento); fica pendente.", flush=True)

    melhor = tab.sort_values("val EER média por fonte").iloc[0]
    gravar_marcador(ns, "camadas", t0, {
        "n_treino": int(tr.ok.sum()), "shards_treino": tr.shards, "melhor_camada": int(melhor.camada),
        "melhor_camada_C": float(melhor.C), "melhor_camada_val_eer_media_por_fonte": float(melhor["val EER média por fonte"]),
        "melhor_camada_dev": {k: float(melhor[k]) for k in ("dev AUC", "dev EER", "dev acurácia balanceada")},
        "ordem_camadas_val": ordem, "concat": [{k: v for k, v in r.items() if k != "grade"} for r in concat],
        "ablacao_corte_feita": ablacao is not None})
    print(f"camadas CONCLUÍDA | melhor camada pela validação: {int(melhor.camada)}\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ somas
def etapa_somas(a, ns, t0):
    import numpy as np
    import pandas as pd
    import torch

    from src.arquivos import salvar_csv, salvar_json, salvar_parquet, salvar_torch
    from src.cabecas import avaliar, prever_soma, treinar_soma

    hp = ns["HP"]
    if ns["marcador"]("somas").exists():
        print("somas: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    exigir(ns, "camadas")
    C = carregar_conjuntos(ns)
    tr, va, dv = C["treino"], C["val"], C["dev_shift"]
    z = np.load(ns["RESULTS_DIR"] / f"{prefixo(ns)}_parcial{ns['SUFIXO']}" / "padronizacao.npz")
    mu, sd = z["mu"], z["sd"]
    assert int(z["n"]) == len(tr)
    device = torch.device("cuda")
    mv, md = va.meta[va.ok].reset_index(drop=True), dv.meta[dv.ok].reset_index(drop=True)
    cols_meta = ["idx_manifest", "path", "source_dataset", "label", "generator", "speaker_id", "duration_s"]
    if not tr.ok.all():
        raise RuntimeError("treino com erro de leitura: a soma ponderada ainda não trata linhas excluídas")
    res, notas = [], {}
    for nome, oculta in (("soma_linear", None), ("soma_mlp", hp["SOMA_OCULTA"])):
        for seed in hp["SEEDS_CABECA"]:
            exp = f"{nome}_s{seed}"
            arq = ns["arq_result"](f"{exp}", "json")
            arq_n = {c: ns["arq_result"](f"notas_{exp}_{c}", "parquet") for c in ("val", "dev_shift")}
            if arq.exists() and all(p.exists() for p in arq_n.values()) and ler_json(arq)["n_treino"] == len(tr):
                r = ler_json(arq)
                notas[exp] = (pd.read_parquet(arq_n["val"]).z.values, pd.read_parquet(arq_n["dev_shift"]).z.values)
                res.append(r)
                print(f"{exp}: já feito", flush=True)
                continue
            print(f"{exp} ({'linear' if oculta is None else f'MLP {oculta}'}):", flush=True)
            t = time.time()
            model, hist, ep, pesos = treinar_soma(tr, va, mu, sd, hp, seed, oculta, device,
                                                  log=lambda s: print(s, flush=True))
            mu_t, sd_t = torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device)
            zv = prever_soma(model, va, mu_t, sd_t, device)[va.ok]
            zd = prever_soma(model, dv, mu_t, sd_t, device)[dv.ok]
            r = {"cabeça": nome, "seed": seed, "n_treino": len(tr), "melhor_época": ep, "épocas": len(hist),
                 **avaliar(zv, mv, zd, md), "pesos_camadas": [float(p) for p in pesos],
                 "segundos": round(time.time() - t, 1)}
            salvar_torch({"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "oculta": oculta,
                          "hp": hp, "seed": seed}, ns["MODELS_DIR"] / f"{prefixo(ns)}_{exp}{ns['SUFIXO']}.pt")
            salvar_csv(hist, ns["arq_result"](f"{exp}_historico", "csv"))
            salvar_parquet(mv[cols_meta].assign(z=zv), arq_n["val"])
            salvar_parquet(md[cols_meta].assign(z=zd), arq_n["dev_shift"])
            salvar_json(r, arq)
            notas[exp] = (zv, zd)
            res.append(r)
            print(f"  -> melhor época {ep} | val EER média/fonte {100 * r['val EER média por fonte']:.2f}% | "
                  f"dev AUC {r['dev AUC']:.3f} EER {100 * r['dev EER']:.1f}% acc. bal. {r['dev acurácia balanceada']:.3f} | "
                  f"camada de maior peso: {int(np.argmax(pesos))} | {r['segundos']:.0f} s", flush=True)
        # ensemble: média dos logits das 3 seeds (só reportado; não entra na escolha)
        zs = [notas[f"{nome}_s{s}"] for s in hp["SEEDS_CABECA"]]
        zv, zd = np.mean([z[0] for z in zs], 0), np.mean([z[1] for z in zs], 0)
        r = {"cabeça": f"{nome}_ensemble", "seed": "média das 3", "n_treino": len(tr), **avaliar(zv, mv, zd, md)}
        salvar_parquet(mv[cols_meta].assign(z=zv), ns["arq_result"](f"notas_{nome}_ensemble_val", "parquet"))
        salvar_parquet(md[cols_meta].assign(z=zd), ns["arq_result"](f"notas_{nome}_ensemble_dev_shift", "parquet"))
        res.append(r)
    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "pesos_camadas"} for r in res])
    salvar_csv(tab, ns["arq_result"]("somas", "csv"))
    gravar_marcador(ns, "somas", t0, {"n_treino": len(tr), "resultados": tab.to_dict("records")})
    print("somas CONCLUÍDA\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ atalhos
def etapa_atalhos(a, ns, t0):
    import warnings

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.arquivos import salvar_csv, salvar_json
    from src.cabecas import avaliar, classificador_fonte, matriz_estat
    from src.preproc import COLS_ATALHO

    if ns["marcador"]("atalhos").exists():
        print("atalhos: já concluída; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    exigir(ns, "camadas")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    C = carregar_conjuntos(ns)
    tr, va, dv = C["treino"], C["val"], C["dev_shift"]
    mt, mv, md = (c.meta[c.ok].reset_index(drop=True) for c in (tr, va, dv))
    out = {"colunas": COLS_ATALHO}

    # (a) regressão logística só com as estatísticas baratas
    esc = StandardScaler().fit(matriz_estat(mt, COLS_ATALHO))
    rl = LogisticRegression(C=1.0, max_iter=2000).fit(esc.transform(matriz_estat(mt, COLS_ATALHO)), mt.label.values)
    zv = rl.decision_function(esc.transform(matriz_estat(mv, COLS_ATALHO)))
    zd = rl.decision_function(esc.transform(matriz_estat(md, COLS_ATALHO)))
    out["a_rotulo_por_estatisticas"] = {**avaliar(zv, mv, zd, md),
                                        "coeficientes": dict(zip(COLS_ATALHO, rl.coef_[0].round(4).tolist()))}
    por_col = []
    for c in COLS_ATALHO:       # cada estatística sozinha (AUC na validação e no dev_shift)
        e1 = StandardScaler().fit(matriz_estat(mt, [c]))
        r1 = LogisticRegression(max_iter=1000).fit(e1.transform(matriz_estat(mt, [c])), mt.label.values)
        a1 = avaliar(r1.decision_function(e1.transform(matriz_estat(mv, [c]))), mv,
                     r1.decision_function(e1.transform(matriz_estat(md, [c]))), md)
        por_col.append({"estatística": c, "val AUC": a1["val AUC"], "val EER média por fonte": a1["val EER média por fonte"],
                        "dev AUC": a1["dev AUC"], "dev acurácia balanceada": a1["dev acurácia balanceada"]})
    out["a_por_estatistica"] = por_col
    print("(a) rótulo só pelas estatísticas: val AUC {:.3f} | val EER média/fonte {:.1f}% | dev AUC {:.3f}".format(
        out["a_rotulo_por_estatisticas"]["val AUC"], 100 * out["a_rotulo_por_estatisticas"]["val EER média por fonte"],
        out["a_rotulo_por_estatisticas"]["dev AUC"]), flush=True)
    print(pd.DataFrame(por_col).round(3).to_string(index=False), flush=True)

    # (b) prever a FONTE só com áudios reais (estatísticas; e a melhor camada do modelo, como comparação)
    rt, rv = mt[mt.label == 0], mv[mv.label == 0]
    b_est, conf_est = classificador_fonte(matriz_estat(rt, COLS_ATALHO), rt.source_dataset.values,
                                          matriz_estat(rv, COLS_ATALHO), rv.source_dataset.values)
    cam = ler_json(ns["marcador"]("camadas"))["melhor_camada"]
    Xt = tr.camada(cam)[tr.ok][mt.label.values == 0]
    Xv = va.camada(cam)[va.ok][mv.label.values == 0]
    b_ssl, conf_ssl = classificador_fonte(Xt, rt.source_dataset.values, Xv, rv.source_dataset.values, C=0.01)
    out["b_fonte_por_estatisticas"], out["b_fonte_por_camada"] = b_est, {**b_ssl, "camada": cam}
    out["b_confusao_estatisticas"] = conf_est.to_dict()
    out["b_confusao_camada"] = conf_ssl.to_dict()
    print(f"(b) fonte dos reais: estatísticas acc. bal. {b_est['acurácia balanceada']:.3f} | camada {cam} acc. bal. "
          f"{b_ssl['acurácia balanceada']:.3f} | acaso {b_est['acaso (1/nº de fontes da validação)']:.3f}", flush=True)

    # (c) nota média dos reais (e dos fakes) por fonte, com a melhor camada (regressão logística)
    nv = pd.read_parquet(ns["arq_result"]("notas_camadas_val", "parquet"))
    nd = pd.read_parquet(ns["arq_result"]("notas_camadas_dev_shift", "parquet"))
    col = f"z_L{cam:02d}"
    lim = eer_e_limiar_np(nv.label.values, nv[col].values)
    linhas = []
    for nome, n_ in (("validação", nv), ("dev_shift", nd)):
        for (fonte, lab), g in n_.groupby(["source_dataset", "label"]):
            p = 1 / (1 + np.exp(-g[col].values))
            linhas.append({"conjunto": nome, "fonte": fonte, "classe": "real" if lab == 0 else "fake", "n": len(g),
                           "prob. média de fake": float(p.mean()), "prob. mediana de fake": float(np.median(p)),
                           "% acima do limiar": float((g[col].values >= lim).mean())})
    out["c_nota_por_fonte"] = {"camada": cam, "limiar_logit": lim, "linhas": linhas}
    print(pd.DataFrame(linhas).round(3).to_string(index=False), flush=True)

    # (d) controle de sanidade: a mesma regressão da melhor camada com os rótulos do TREINO embaralhados.
    # Tem que dar AUC ~0,5 na validação e no dev_shift; se não der, há vazamento no código.
    z = np.load(ns["RESULTS_DIR"] / f"{prefixo(ns)}_parcial{ns['SUFIXO']}" / "padronizacao.npz")
    c_cam = float(ler_json(ns["marcador"]("camadas"))["melhor_camada_C"])
    Xtr = (tr.camada(cam)[tr.ok] - z["mu"][cam]) / z["sd"][cam]
    y_emb = np.random.default_rng(ns["SEED"]).permutation(mt.label.values)
    rl_e = LogisticRegression(C=c_cam, max_iter=ns["HP"]["RL_MAX_ITER"], tol=1e-4).fit(Xtr, y_emb)
    a_e = avaliar(rl_e.decision_function((va.camada(cam)[va.ok] - z["mu"][cam]) / z["sd"][cam]), mv,
                  rl_e.decision_function((dv.camada(cam)[dv.ok] - z["mu"][cam]) / z["sd"][cam]), md)
    out["d_controle_rotulos_embaralhados"] = {"camada": cam, "C": c_cam, "val AUC": a_e["val AUC"],
                                              "val EER média por fonte": a_e["val EER média por fonte"],
                                              "dev AUC": a_e["dev AUC"], "dev EER": a_e["dev EER"]}
    print(f"(d) controle com rótulos embaralhados (camada {cam}): val AUC {a_e['val AUC']:.3f} | dev AUC {a_e['dev AUC']:.3f}",
          flush=True)
    salvar_csv(pd.DataFrame(linhas), ns["arq_result"]("nota_por_fonte", "csv"))
    salvar_json(out, ns["arq_result"]("atalhos", "json"))
    gravar_marcador(ns, "atalhos", t0, {"a_val_auc": out["a_rotulo_por_estatisticas"]["val AUC"],
                                        "a_dev_auc": out["a_rotulo_por_estatisticas"]["dev AUC"]})
    print("atalhos CONCLUÍDA\nSTATUS=CONCLUIDO", flush=True)
    return 0


def eer_e_limiar_np(y, z):
    from src.metrics import eer_e_limiar
    return eer_e_limiar(y, z)[1]


# ============================================================ resumo
def etapa_resumo(a, ns, t0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.arquivos import gravar_atomico, salvar_csv, salvar_json

    if ns["marcador"]("resumo").exists():
        print("resumo: já concluído; nada a fazer.\nSTATUS=JA_CONCLUIDO", flush=True)
        return 0
    for e in ("extracao", "camadas", "somas", "atalhos"):
        exigir(ns, e)
    cam = pd.read_csv(ns["arq_result"]("camadas", "csv"))
    somas = pd.read_csv(ns["arq_result"]("somas", "csv"))
    mc = ler_json(ns["marcador"]("camadas"))
    l = mc["melhor_camada"]
    linhas = [{"cabeça": f"rl_camada_{l}", "tipo": "regressão logística, melhor camada", "seed": "-",
               **cam[cam.camada == l].iloc[0].to_dict()}]
    for r in mc["concat"]:
        linhas.append({"tipo": f"regressão logística, concatenação ({r['camadas']})", "seed": "-", **r})
    for _, r in somas.iterrows():
        linhas.append({"tipo": "soma ponderada + " + ("MLP" if "mlp" in r["cabeça"] else "linear")
                       + (" (ensemble, só reportado)" if "ensemble" in r["cabeça"] else ""), **r.to_dict()})
    at = ler_json(ns["arq_result"]("atalhos", "json"))
    linhas.append({"cabeça": "atalho_estatisticas", "tipo": "regressão logística só com estatísticas (Parte F)",
                   "seed": "-", **{k: v for k, v in at["a_rotulo_por_estatisticas"].items() if k != "coeficientes"}})
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, ns["arq_result"]("cabecas", "csv"))

    # escolha pela validação (pré-registro a): cabeças com seeds pela média das 3 seeds; ensembles e atalho fora
    cand = []
    cand.append({"cabeça": f"rl_camada_{l}", "val EER média por fonte": float(cam[cam.camada == l]["val EER média por fonte"].iloc[0])})
    for r in mc["concat"]:
        cand.append({"cabeça": r["cabeça"], "val EER média por fonte": r["val EER média por fonte"]})
    for nome in ("soma_linear", "soma_mlp"):
        s = somas[somas["cabeça"] == nome]
        cand.append({"cabeça": nome, "val EER média por fonte": float(s["val EER média por fonte"].mean())})
    cand = pd.DataFrame(cand).sort_values("val EER média por fonte")
    final = cand.iloc[0]["cabeça"]
    if final.startswith("soma"):
        s = somas[somas["cabeça"] == final]
        aucs = s["dev AUC"].tolist()
        dev_final = {k: float(s[k].mean()) for k in ("dev AUC", "dev EER", "dev acurácia balanceada")}
    else:
        rr = tab[tab["cabeça"] == final].iloc[0]
        aucs = [float(rr["dev AUC"])]
        dev_final = {k: float(rr[k]) for k in ("dev AUC", "dev EER", "dev acurácia balanceada")}
    if all(x > 0.6 for x in aucs):
        veredito = "melhora clara (AUC do dev_shift > 0,6)"
    elif all(x > 0.5 for x in aucs):
        veredito = "melhora fraca (AUC do dev_shift entre 0,5 e 0,6)"
    else:
        veredito = ("não resolveu (AUC do dev_shift <= 0,5 em pelo menos uma seed); pelo pré-registro (e), "
                    "se continuar abaixo de 0,5 isso indica problema de DADOS, não de modelo")
    somas_regra = {}
    for nome in ("soma_linear", "soma_mlp"):
        s = somas[somas["cabeça"] == nome]["dev AUC"]
        somas_regra[nome] = {"aucs": s.round(4).tolist(), "todas_acima_0_5": bool((s > 0.5).all())}

    # régua do ruído da v3 e comparação
    v3 = pd.read_csv("results/v3_acuracias.csv")
    v3d = v3[v3.conjunto == "dev_shift"]
    regua = {"dev AUC min": float(v3d.AUC.min()), "dev AUC max": float(v3d.AUC.max()),
             "dev EER min": float(v3d.EER.min()), "dev EER max": float(v3d.EER.max()),
             "dev acurácia balanceada min": float(v3d["acurácia balanceada"].min()),
             "dev acurácia balanceada max": float(v3d["acurácia balanceada"].max())}

    # gráfico: camada x EER média por fonte (validação) e x AUC (dev_shift)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), dpi=80)
    ax[0].plot(cam.camada, 100 * cam["val EER média por fonte"], "o-", color="#6250d6")
    ax[0].axvline(l, color="gray", ls=":", lw=1)
    ax[0].set(xlabel="saída do modelo (0 = entrada do transformer)", ylabel="EER média por fonte (%)",
              title="validação (escolhe a camada)")
    ax[1].plot(cam.camada, cam["dev AUC"], "o-", color="#1baf7a")
    ax[1].axhspan(regua["dev AUC min"], regua["dev AUC max"], color="#eb6834", alpha=0.15, label="v3 (régua do ruído)")
    ax[1].axhline(0.5, color="black", lw=0.8, ls="--", label="acaso")
    ax[1].axvline(l, color="gray", ls=":", lw=1)
    ax[1].set(xlabel="saída do modelo", ylabel="AUC", title="dev_shift (só reportado)", ylim=(0, 1))
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    png = ns["arq_result"]("camadas", "png")
    gravar_atomico(png, lambda t: fig.savefig(t, format="png"))
    plt.close(fig)

    ext = ler_json(ns["marcador"]("extracao"))
    resumo = {"cabeça_final_pela_validação": final, "candidatas": cand.to_dict("records"),
              "dev_cabeça_final": dev_final, "dev_AUC_por_seed_cabeça_final": aucs, "veredito_pré_registro_c": veredito,
              "regra_3_seeds_somas": somas_regra, "régua_v3": regua, "melhor_camada": l,
              "atalho_val_AUC": at["a_rotulo_por_estatisticas"]["val AUC"],
              "atalho_dev_AUC": at["a_rotulo_por_estatisticas"]["dev AUC"],
              "atalho_há_indício": bool(at["a_rotulo_por_estatisticas"]["val AUC"] > 0.6),
              "n_treino": mc["n_treino"], "extracao": {k: ext[k] for k in ("horas_extracao_acumuladas", "parou_por",
                                                                           "plano_completo", "conjuntos")}}
    salvar_json(resumo, ns["arq_result"]("resumo", "json"))
    gravar_marcador(ns, "resumo", t0, {"cabeça_final": final, "veredito": veredito})
    print(json.dumps(resumo, ensure_ascii=False, indent=1, default=str)[:4000], flush=True)
    print("resumo CONCLUÍDO\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ============================================================ status (leve) e concluir
def etapa_status(a, ns, t0):
    """Reescreve logs/v4_resumo.txt com o que já está no disco (não carrega modelo nem cache)."""
    from src.arquivos import gravar_atomico
    suf = ns["SUFIXO"]
    L = [f"Resumo {prefixo(ns)}{' (QUICK)' if ns['QUICK'] else ''} (atualizado em {datetime.now():%Y-%m-%d %H:%M:%S})",
         "Teste final NÃO avaliado (AVALIAR_TESTE_FINAL = False). Escolhas só pela validação; dev_shift só reportado.", ""]
    est = ler_json(ns["ARQ_ESTADO"], {})
    L.append(f"extração: {est.get('horas_extracao', 0):.2f} h gastas de {ns['HP']['ORCAMENTO_H']} h | último shard: "
             f"{est.get('ultimo_shard', '-')} | tentativas: {est.get('tentativas_extracao', 0)}")
    vz = ler_json(ns["arq_result"]("vazao", "json"))
    if vz:
        L.append(f"vazão medida antes (QUICK/medir): {vz['clipes_por_s']} clipes/s com {vz['num_workers']} workers")
    L.append("")
    L.append(f"{'etapa':<10} {'status':<12} {'min':>7} {'pico GB':>8}  observação")
    for e in ns["ETAPAS"]:
        m = ler_json(ns["marcador"](e))
        if m:
            obs = ""
            if e == "extracao":
                obs = (f"{m['parou_por']}; " + ", ".join(f"{k} {v['clipes_prontos']}/{v['clipes_planejados']}"
                                                          for k, v in m["conjuntos"].items()))
            elif e == "camadas":
                obs = (f"melhor camada {m['melhor_camada']} (C={m['melhor_camada_C']:g}): val EER média/fonte "
                       f"{100 * m['melhor_camada_val_eer_media_por_fonte']:.2f}% | dev AUC {m['melhor_camada_dev']['dev AUC']:.3f} "
                       f"EER {100 * m['melhor_camada_dev']['dev EER']:.1f}% acc.bal. {m['melhor_camada_dev']['dev acurácia balanceada']:.3f}"
                       f" | n treino {m['n_treino']}")
            elif e == "atalhos":
                obs = f"rótulo só por estatísticas: val AUC {m['a_val_auc']:.3f}, dev AUC {m['a_dev_auc']:.3f}"
            elif e == "resumo":
                obs = f"cabeça final (validação): {m['cabeça_final']} | {m['veredito']}"
            L.append(f"{e:<10} {'concluída':<12} {m['minutos_neste_processo']:>7.1f} "
                     f"{m['pico_memoria_gb_processo_e_workers']:>8.2f}  {obs}")
        else:
            L.append(f"{e:<10} {'pendente':<12} {'-':>7} {'-':>8}")
    ms = ler_json(ns["marcador"]("somas"))
    if ms:
        L += ["", f"{'cabeça':<22} {'seed':>11} {'val EER m/f':>11} {'dev AUC':>8} {'dev EER':>8} {'dev acc.bal':>11}"]
        for r in ms["resultados"]:
            L.append(f"{r['cabeça']:<22} {str(r['seed']):>11} {100 * r['val EER média por fonte']:>10.2f}% "
                     f"{r['dev AUC']:>8.3f} {100 * r['dev EER']:>7.1f}% {r['dev acurácia balanceada']:>11.3f}")
    mc = ler_json(ns["marcador"]("camadas"))
    if mc:
        for r in mc["concat"]:
            L.append(f"{r['cabeça']:<22} {'-':>11} {100 * r['val EER média por fonte']:>10.2f}% "
                     f"{r['dev AUC']:>8.3f} {100 * r['dev EER']:>7.1f}% {r['dev acurácia balanceada']:>11.3f}")
    rs = ler_json(ns["arq_result"]("resumo", "json"))
    if rs:
        L += ["", f"régua do ruído v3 (6 execuções, CNN): dev AUC {rs['régua_v3']['dev AUC min']:.3f} a "
                  f"{rs['régua_v3']['dev AUC max']:.3f} | acc. bal. {rs['régua_v3']['dev acurácia balanceada min']:.3f} a "
                  f"{rs['régua_v3']['dev acurácia balanceada max']:.3f}",
              f"cabeça final pela validação: {rs['cabeça_final_pela_validação']} | dev AUC por seed {rs['dev_AUC_por_seed_cabeça_final']}",
              f"veredito (pré-registro c): {rs['veredito_pré_registro_c']}"]
    L += ["", f"concluído: {'SIM' if ns['ARQ_CONCLUIDO'].exists() else 'não'} ({ns['ARQ_CONCLUIDO']})"]
    txt = "\n".join(L) + "\n"
    gravar_atomico(ns["LOGS_DIR"] / f"{prefixo(ns)}_resumo{suf}.txt", lambda t: Path(t).write_text(txt, encoding="utf-8-sig"))
    print(txt, flush=True)
    return 0


def etapa_concluir(a, ns, t0):
    from src.arquivos import salvar_json
    faltam = [e for e in ns["ETAPAS"] if not ns["marcador"](e).exists()]
    if faltam:
        print(f"ainda faltam: {faltam}\nSTATUS=INCOMPLETO", flush=True)
        return 3
    if not ns["ARQ_CONCLUIDO"].exists():
        salvar_json({"versao": prefixo(ns), "quick": ns["QUICK"], "concluido_em": datetime.now().isoformat(timespec="seconds"),
                     "etapas": {e: ler_json(ns["marcador"](e))["concluido_em"] for e in ns["ETAPAS"]},
                     "resumo": ler_json(ns["arq_result"]("resumo", "json"))}, ns["ARQ_CONCLUIDO"])
    print(f"{prefixo(ns)} CONCLUÍDA ({ns['ARQ_CONCLUIDO']})\nSTATUS=CONCLUIDO", flush=True)
    return 0


ETAPAS_FN = {"medir": etapa_medir, "extrair": etapa_extrair, "camadas": etapa_camadas, "somas": etapa_somas,
             "atalhos": etapa_atalhos, "resumo": etapa_resumo, "status": etapa_status, "concluir": etapa_concluir}


def main():
    global AMOSTRADOR
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=list(ETAPAS_FN))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--n", type=int, default=500, help="medir: nº de clipes")
    ap.add_argument("--balde-s", type=int, default=None, help="medir: largura dos baldes (comparação; padrão: a da configuração)")
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
        with contextlib.redirect_stdout(io.StringIO()) if leve else contextlib.nullcontext():
            ns = executar_config(src_cfg, a.quick, a.notebook)
        ns["_SRC_CONFIG"] = src_cfg
        assert ns["AVALIAR_TESTE_FINAL"] is False, "este script não avalia o teste final"
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
