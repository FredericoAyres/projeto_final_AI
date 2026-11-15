#!/usr/bin/env python3
"""v8: fine-tuning parcial do XLS-R a partir da T3 (protocolo: docs/PROTOCOLO_V8.md).

Mesma divisão da v7 (data/processed/splits_v7.parquet), mesmo treino da T3, mesma validação e o mesmo
teste mestre. Muda só uma coisa: as camadas 5, 6 e 7 do XLS-R voltam a aprender (src/finetune.py).

Etapas (um processo pesado por vez; cada uma retomável):
    python scripts/v8.py checar                       # passo zero = T3? e vazão (não decide nada)
    python scripts/v8.py treinar --receita R1 --seed 42
    python scripts/v8.py escolher                     # SÓ validação -> results/v8_escolha.json
    python scripts/v8.py testar --receita R1 --seed 42  # notas do teste mestre e controles (época escolhida)
    python scripts/v8.py avaliar                      # métricas do teste mestre: UMA vez (marcador)
    python scripts/v8.py fila                         # tudo acima na ordem, cada etapa num processo
    --quick: subamostras pequenas, saídas com sufixo _quick (só para testar o código; números não são lidos)
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import teste_internet as ti  # noqa: E402  (Vigia, sem_suspensao, config_v4)
import v7  # noqa: E402  (auc_val_por_fonte, chave_val, cortes_de, bal, medir, ic, REGRAS, CONTROLES)

R, LOGS, DP, MODELOS = Path("results"), Path("logs"), Path("data/processed"), Path("models")
STATUS = LOGS / "v8_status.txt"
LIMITE_GB = 10.0
SEED = 42
T3_NPZ = R / "v7_t3_coeficientes.npz"
SPLITS = DP / "splits_v7.parquet"
RAIZ_INTERNET = ROOT / "data/raw/teste_internet"

# ---------------- receitas (protocolo, seção 3) ----------------
RECEITAS = {
    "R1": {"treinaveis": (5, 6, 7), "inicio": "T3", "seeds": (42, 123),
           "descr": "camadas 5-7 + cabeça; cabeça começa na T3"},
    "R2": {"treinaveis": (7,), "inicio": "T3", "seeds": (42,), "descr": "só a camada 7 + cabeça; cabeça começa na T3"},
    "R3": {"treinaveis": (5, 6, 7), "inicio": "aleatorio", "seeds": (42,),
           "descr": "camadas 5-7 + cabeça; cabeça aleatória"},
}
RODADAS = [(rc, s) for rc, c in RECEITAS.items() for s in c["seeds"]]
DESCR = {"R0": "T3 da v7 (sem fine-tuning)", **{rc: c["descr"] for rc, c in RECEITAS.items()}}

# ---------------- treino (protocolo, seção 4) ----------------
EPOCAS = 2
CLIPES_EPOCA = 20_000
POR_LOTE = 16
LR_CAMADAS, LR_CABECA = 1e-5, 1e-4
WD = 0.01
AQUECIMENTO = 0.10
MAX_NORMA = 1.0
POR_LOTE_AVAL = 32
NUM_WORKERS = 2
N_BOOT = 2000

QUICK = False
ARGS = None


def sfx():
    return "_quick" if QUICK else ""


def agora():
    return datetime.now().isoformat(timespec="seconds")


def status(etapa, msg):
    linha = f"[{agora()}] v8{' QUICK' if QUICK else ''} | {etapa} | {msg} | pico memória " \
            f"{ti.VIGIA.pico if ti.VIGIA else float('nan'):.2f} GB (limite {LIMITE_GB})"
    print(linha, flush=True)
    alvo = STATUS if not QUICK else LOGS / "v8_status_quick.txt"
    alvo.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            with open(alvo, "a", encoding="utf-8") as f:
                f.write(linha + "\n")
            break
        except OSError:
            time.sleep(1)


def salvar_json(obj, p):
    v7.salvar_json(obj, p)


def nome(rc, seed):
    return f"{rc}_s{seed}"


def pasta_notas():
    p = DP / f"v8_notas{sfx()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def arq_ckpt(rc, seed, ep):
    MODELOS.mkdir(exist_ok=True)
    return MODELOS / f"v8_{nome(rc, seed)}_ep{ep}{sfx()}.pt"


def arq_val(rc, seed, ep):
    return pasta_notas() / f"val_{nome(rc, seed)}_ep{ep}.parquet"


def arq_teste(rc, seed):
    return pasta_notas() / f"teste_{nome(rc, seed)}.parquet"


def hp():
    ns = ti.config_v4()
    return {k: ns[k] for k in ("MODELO_SSL", "TOP_DB", "MIN_S", "MAX_S")}


def splits():
    sp = pd.read_parquet(SPLITS)
    assert len(sp) == 249_187, len(sp)
    sp["path"] = [str(RAIZ_INTERNET / c) if m == "internet" else str(ROOT / c)
                  for c, m in zip(sp.caminho, sp.manifesto)]
    return sp


PREP = DP / "v8_audio"
_INDICE = None


def indice_preparado():
    """Índice dos áudios preparados (linha -> arq, ini, n), ou None se a etapa 'preparar' não terminou."""
    global _INDICE
    if _INDICE is None and (PREP / "concluido.json").exists():
        _INDICE = pd.concat([pd.read_parquet(f) for f in sorted(PREP.glob("shard_*.parquet"))],
                            ignore_index=True).rename(columns={"chave": "linha"})
    return _INDICE


def fonte(df):
    """O que o carregador lê: os áudios preparados (se existirem) ou os arquivos originais."""
    ind = indice_preparado()
    if ind is None:
        return df.path.values
    f = df[["linha"]].merge(ind, on="linha", how="left")
    assert f.arq.notna().all(), "clipe sem áudio preparado"
    return f.reset_index(drop=True)


def dur(df):
    """Duração usada para montar os lotes: a do clipe preparado (exata) ou a do manifesto (teto)."""
    f = fonte(df)
    return f.n.values / 16000 if hasattr(f, "columns") else df.duracao_s.values


def treino_t3(sp):
    tr = sp[(sp.split == "treino") & ((sp.bloco != "radar_eval") | sp.pareado_treino)].sort_values("linha")
    assert len(tr) == 81_288, len(tr)
    return tr.reset_index(drop=True)


def amostra(df, chaves, n):
    """Até n clipes por grupo (sorteio fixo pela SEED)."""
    idx = [i for _, g in df.groupby(chaves) for i in g.sample(min(len(g), n), random_state=SEED).index]
    return df.loc[sorted(idx)]


def validacao(sp):
    v = sp[sp.split == "validacao"].sort_values("linha")
    if QUICK:
        v = amostra(v.assign(_fv=v7.chave_val(v)), ["_fv", "label"], 120).drop(columns="_fv")
    return v.sort_values("linha").reset_index(drop=True)


def teste(sp):
    t = sp[sp.split.isin(["teste_mestre", "controle"])].sort_values("linha")
    if QUICK:
        t = amostra(t, ["bloco", "fonte", "label"], 60)
    return t.sort_values("linha").reset_index(drop=True)


def notas_r0(linhas):
    """Notas da T3 (R0) a partir dos vetores já extraídos (data/processed/v7_vetores), pesos do colega."""
    X, _ = v7.carregar_vetores()
    t3 = np.load(T3_NPZ)
    return v7.pontuar(X, np.asarray(linhas), t3["mu"].astype(np.float32), t3["sd"].astype(np.float32),
                      t3["coef"].astype(np.float32).ravel(), float(t3["intercepto"]))


def construir(rc, seed, dispositivo="cuda"):
    import torch

    from src.finetune import DetectorFT
    from src.ssl import carregar_ssl
    torch.manual_seed(seed)
    t3 = np.load(T3_NPZ)
    assert [int(c) for c in t3["camadas"]] == [5, 6, 7] and abs(float(t3["C"]) - 0.01) < 1e-12
    cfg = RECEITAS[rc] if rc in RECEITAS else {"treinaveis": (5, 6, 7), "inicio": "T3"}
    ssl = carregar_ssl(hp()["MODELO_SSL"], device="cpu", dtype="float32")
    coef, b = (t3["coef"].ravel(), float(t3["intercepto"])) if cfg["inicio"] == "T3" else (None, None)
    return DetectorFT(ssl, t3["mu"], t3["sd"], coef, b, cfg["treinaveis"], seed).to(dispositivo)


def plano_epocas(tr, seed):
    """Para cada época: 20.000 clipes (10.000 reais e 10.000 falsos, sem reposição dentro da época),
    em lotes de 16 do mesmo balde de duração. Determinístico pela seed."""
    from src.finetune import lotes_treino
    n = 640 if QUICK else CLIPES_EPOCA
    y = tr.label.values.astype(int)
    planos = []
    for ep in range(1, EPOCAS + 1):
        rng = np.random.default_rng(seed * 1000 + ep)
        sel = np.concatenate([rng.choice(np.flatnonzero(y == c), n // 2, replace=False) for c in (0, 1)])
        lotes = lotes_treino(dur(tr)[sel], POR_LOTE, rng)
        planos.append([[int(sel[i]) for i in lote] for lote in lotes])
    return planos


# ======================================================================== preparar
def etapa_preparar(a):
    """Prepara UMA vez os áudios de treino (T3), validação, teste mestre e controles (execução, não método)."""
    from src.finetune import preparar_audios
    if QUICK:
        status("preparar", "QUICK: não prepara nada (usa o preparado completo, se existir, ou os arquivos originais)")
        return 0
    if (PREP / "concluido.json").exists():
        status("preparar", "já pronto; nada a fazer")
        return 0
    sp = splits()
    d = pd.concat([treino_t3(sp), sp[sp.split.isin(["validacao", "teste_mestre", "controle"])]]).sort_values("linha")
    assert d.linha.is_unique and len(d) == 81_288 + 58_997 + 93_368, len(d)
    nw = a.num_workers if a.num_workers != NUM_WORKERS else 6
    status("preparar", f"início: {len(d)} clipes, {nw} workers -> {PREP}")
    preparar_audios(d.path.values, d.linha.values, PREP, hp(), num_workers=nw, log=print)
    ind = pd.concat([pd.read_parquet(f) for f in sorted(PREP.glob("shard_*.parquet"))], ignore_index=True)
    assert len(ind) == len(d) and ind.chave.is_unique
    gb = sum(f.stat().st_size for f in PREP.glob("shard_*.npy")) / 2 ** 30
    salvar_json({"quando": agora(), "n": int(len(ind)), "ilegiveis": int(ind.erro_leitura.sum()), "gb": round(gb, 1),
                 "dtype": "float16", "pipeline": hp()}, PREP / "concluido.json")
    status("preparar", f"CONCLUÍDO: {len(ind)} clipes ({int(ind.erro_leitura.sum())} ilegíveis), {gb:.1f} GB")
    return 0


# ======================================================================== checar
def etapa_checar(a):
    """(1) O modelo no passo zero (cabeça da T3, nada treinado) dá as mesmas notas que a T3 sobre os vetores
    do cache? (2) Vazão de treino e de avaliação nesta máquina. Não decide nada da v8."""
    import torch

    from src.finetune import agenda, pontuar, treinar_epoca
    sp = splits()
    v = amostra(sp[sp.split == "validacao"], ["bloco", "label"], 64).sort_values("linha").reset_index(drop=True)
    model = construir("R1", SEED)
    h = hp()
    t0 = time.time()
    z, erro = pontuar(model, fonte(v), dur(v), h, a.num_workers, POR_LOTE_AVAL, log=print, rotulo="checar")
    t_aval = time.time() - t0
    z0 = notas_r0(v.linha.values)
    d = np.abs(z - z0)
    from sklearn.metrics import roc_auc_score
    res = {"quando": agora(), "n": int(len(v)), "ilegiveis": int(erro.sum()),
           "dif_logit_max": float(d.max()), "dif_logit_mediana": float(np.median(d)),
           "correlacao": float(np.corrcoef(z, z0)[0, 1]),
           "AUC passo zero": float(roc_auc_score(v.label, z)), "AUC T3 (cache)": float(roc_auc_score(v.label, z0)),
           "aval_clipes_por_s (inclui carregar e aquecer)": round(len(v) / t_aval, 1)}
    status("checar", f"passo zero x T3: dif. máx {res['dif_logit_max']:.3f}, mediana {res['dif_logit_mediana']:.4f}, "
                     f"correlação {res['correlacao']:.5f}")
    # vazão de treino: 60 passos (os pesos são descartados)
    tr = treino_t3(sp)
    rng = np.random.default_rng(0)
    from src.finetune import lotes_treino
    sel = rng.choice(len(tr), 60 * POR_LOTE * 2, replace=False)
    lotes = [[int(sel[i]) for i in l_] for l_ in lotes_treino(dur(tr)[sel], POR_LOTE, rng)][:60]
    camadas, cabeca = model.parametros()
    otim = torch.optim.AdamW([{"params": camadas, "lr": LR_CAMADAS}, {"params": cabeca, "lr": LR_CABECA}],
                             weight_decay=WD)
    torch.cuda.reset_peak_memory_stats()
    est = treinar_epoca(model, otim, agenda(otim, len(lotes), AQUECIMENTO), fonte(tr), tr.label.values,
                        dur(tr), lotes, h, a.num_workers, max_norma=MAX_NORMA, log=print, rotulo="checar")
    res["treino_60_passos"] = est
    res["gpu_pico_gb_treino"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    # estimativa de tempo da v8 inteira
    n_val, n_teste = int((sp.split == "validacao").sum()), int(sp.split.isin(["teste_mestre", "controle"]).sum())
    vt, va = est["clipes_por_s"], res["aval_clipes_por_s (inclui carregar e aquecer)"]
    por_rodada = EPOCAS * CLIPES_EPOCA / vt + EPOCAS * n_val / va
    res["estimativa_h"] = {"por_rodada_treino_e_validacao": round(por_rodada / 3600, 2),
                           "teste_por_rodada": round(n_teste / va / 3600, 2),
                           "total_4_rodadas": round(len(RODADAS) * (por_rodada + n_teste / va) / 3600, 1)}
    salvar_json(res, R / f"v8_checagem{sfx()}.json")
    status("checar", f"CONCLUÍDO: treino {vt} clipes/s, avaliação {va} clipes/s, GPU {res['gpu_pico_gb_treino']} GB; "
                     f"estimativa total {res['estimativa_h']['total_4_rodadas']} h")
    return 0


# ======================================================================== treinar
def etapa_treinar(a):
    import torch

    from src.finetune import agenda, pontuar, treinar_epoca
    rc, seed = a.receita, a.seed
    assert rc in RECEITAS and seed in RECEITAS[rc]["seeds"], (rc, seed)
    rot = nome(rc, seed)
    if all(arq_ckpt(rc, seed, ep).exists() and arq_val(rc, seed, ep).exists() for ep in range(1, EPOCAS + 1)):
        status(f"treinar {rot}", "já pronto; nada a fazer")
        return 0
    sp = splits()
    tr, v = treino_t3(sp), validacao(sp)
    planos = plano_epocas(tr, seed)
    total = sum(len(p) for p in planos)
    h = hp()
    model = construir(rc, seed)
    camadas, cabeca = model.parametros()
    otim = torch.optim.AdamW([{"params": camadas, "lr": LR_CAMADAS}, {"params": cabeca, "lr": LR_CABECA}],
                             weight_decay=WD)
    sched = agenda(otim, total, AQUECIMENTO)
    n_trein = sum(p.numel() for p in camadas + cabeca)
    status(f"treinar {rot}", f"início: {len(tr)} clipes no treino; {EPOCAS} épocas x {len(planos[0])} passos "
                             f"(lote {POR_LOTE}); {n_trein / 1e6:.1f} M parâmetros treináveis")
    hist = []
    for ep in range(1, EPOCAS + 1):
        ck = arq_ckpt(rc, seed, ep)
        if ck.exists():
            e = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(e["treinavel"], strict=False)
            otim.load_state_dict(e["otimizador"])
            sched.load_state_dict(e["agenda"])
            hist.append(e["estatisticas"])
            status(f"treinar {rot}", f"época {ep}: checkpoint já existe ({ck}); retomando dele")
        else:
            torch.cuda.reset_peak_memory_stats()
            est = treinar_epoca(model, otim, sched, fonte(tr), tr.label.values, dur(tr),
                                planos[ep - 1], h, a.num_workers, max_norma=MAX_NORMA, log=print, rotulo=f"{rot} ep{ep}")
            est.update({"epoca": ep, "gpu_pico_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                        "lr_camadas_fim": otim.param_groups[0]["lr"], "lr_cabeca_fim": otim.param_groups[1]["lr"]})
            torch.save({"treinavel": model.estado_treinavel(), "otimizador": otim.state_dict(),
                        "agenda": sched.state_dict(), "estatisticas": est, "receita": rc, "seed": seed}, ck)
            hist.append(est)
            status(f"treinar {rot}", f"época {ep}: perda média {est['perda_media']:.4f}; {est['clipes_por_s']} clipes/s; "
                                     f"{est['segundos'] / 60:.1f} min; GPU {est['gpu_pico_gb']} GB")
        av = arq_val(rc, seed, ep)
        if not av.exists():
            t0 = time.time()
            z, erro = pontuar(model, fonte(v), dur(v), h, a.num_workers, POR_LOTE_AVAL,
                              log=print, rotulo=f"{rot} ep{ep} val")
            assert not erro.any(), f"{int(erro.sum())} clipes ilegíveis na validação"
            assert np.isfinite(z).all(), "nota não finita na validação"
            pd.DataFrame({"linha": v.linha.values, "z": z}).to_parquet(av, index=False)
            m = v7.auc_val_por_fonte(v.label.values.astype(int), z, v7.chave_val(v))
            status(f"treinar {rot}", f"época {ep}: validação em {(time.time() - t0) / 60:.1f} min; "
                                     f"AUC média por fonte {np.mean(list(m.values())):.4f}")
    salvar_json({"quando": agora(), "receita": rc, "seed": seed, "historico": hist}, R / f"v8_treino_{rot}{sfx()}.json")
    status(f"treinar {rot}", "CONCLUÍDO")
    return 0


# ======================================================================== escolher (SÓ validação)
def etapa_escolher(a):
    from src.arquivos import salvar_csv
    if (R / f"v8_escolha{sfx()}.json").exists():      # a escolha é gravada uma vez, antes do teste; não regravar
        status("escolher", "já pronto; nada a fazer")
        return 0
    sp = splits()
    v = validacao(sp)
    y, fv = v.label.values.astype(int), v7.chave_val(v)
    duas = [f for f in sorted(set(fv)) if (y[fv == f] == 0).sum() >= v7.MIN_POR_CLASSE and (y[fv == f] == 1).sum() >= v7.MIN_POR_CLASSE]
    zs = {"R0": notas_r0(v.linha.values)}
    linhas, epoca = [], {}
    a0 = v7.auc_val_por_fonte(y, zs["R0"], fv)
    linhas.append({"candidato": "R0", "receita": "R0", "seed": None, "época": None, **{f"AUC {k}": x for k, x in a0.items()},
                   "AUC média por fonte": float(np.mean(list(a0.values()))), "época escolhida": True})
    for rc, seed in RODADAS:
        melhor = None
        for ep in range(1, EPOCAS + 1):
            d = pd.read_parquet(arq_val(rc, seed, ep))
            z = v[["linha"]].merge(d, on="linha", how="left").z.values
            assert np.isfinite(z).all(), (rc, seed, ep)
            av = v7.auc_val_por_fonte(y, z, fv)
            media = float(np.mean(list(av.values())))
            linhas.append({"candidato": nome(rc, seed), "receita": rc, "seed": seed, "época": ep,
                           **{f"AUC {k}": x for k, x in av.items()}, "AUC média por fonte": media, "época escolhida": False})
            if melhor is None or media > melhor[0]:          # empate: a época mais cedo
                melhor = (media, ep, z, len(linhas) - 1)
        linhas[melhor[3]]["época escolhida"] = True
        epoca[nome(rc, seed)] = melhor[1]
        zs[nome(rc, seed)] = melhor[2]
    tv = pd.DataFrame(linhas)
    esc_ = tv[tv["época escolhida"]]
    por_receita = esc_.groupby("receita")["AUC média por fonte"].mean().to_dict()     # R1: média das 2 seeds
    venc = max(por_receita, key=por_receita.get)
    cortes = {}
    linhas_c = []
    for cand, z in zs.items():
        ct = v7.cortes_de(y, z, fv)
        loso = {}
        for r in v7.REGRAS:
            accs = {f: v7.bal(y[fv == f], z[fv == f], v7.cortes_de(y[fv != f], z[fv != f], fv[fv != f])[r]) for f in duas}
            loso[r] = float(np.mean(list(accs.values())))
            linhas_c.append({"candidato": cand, "regra": r, "corte (validação inteira)": ct[r],
                             "acurácia balanceada média (deixando uma fonte de fora)": loso[r],
                             **{f"bal. fora: {f}": x for f, x in accs.items()}})
        cortes[cand] = {"regra": max(v7.REGRAS, key=lambda r: loso[r]), "cortes": ct, "loso": loso}
    salvar_csv(tv, R / f"v8_validacao{sfx()}.csv")
    salvar_csv(pd.DataFrame(linhas_c), R / f"v8_cortes_validacao{sfx()}.csv")
    salvar_json({"quando": agora(), "fontes_validacao_com_duas_classes": duas, "epoca_escolhida": epoca,
                 "AUC_media_por_receita (R1 = média das seeds)": por_receita, "receita_vencedora_pela_validacao": venc,
                 "criterio": "maior média da AUC por fonte na validação (6 fontes; >= 20 clipes por classe); "
                             "época: a melhor das 2 por rodada; receita com várias seeds: média das seeds",
                 "por_candidato": cortes}, R / f"v8_escolha{sfx()}.json")
    status("escolher", f"CONCLUÍDO: épocas {epoca}; AUC por receita {por_receita}; vencedora pela validação {venc}")
    return 0


# ======================================================================== testar
def etapa_testar(a):
    import torch

    from src.finetune import pontuar
    rc, seed = a.receita, a.seed
    rot = nome(rc, seed)
    esc = json.loads((R / f"v8_escolha{sfx()}.json").read_text(encoding="utf-8"))   # só depois da escolha
    if arq_teste(rc, seed).exists():
        status(f"testar {rot}", "já pronto; nada a fazer")
        return 0
    ep = esc["epoca_escolhida"][rot]
    t = teste(splits())
    model = construir(rc, seed)
    e = torch.load(arq_ckpt(rc, seed, ep), map_location="cpu", weights_only=False)
    model.load_state_dict(e["treinavel"], strict=False)
    t0 = time.time()
    z, erro = pontuar(model, fonte(t), dur(t), hp(), a.num_workers, POR_LOTE_AVAL,
                      log=print, rotulo=f"{rot} teste")
    assert np.isfinite(z).all()
    pd.DataFrame({"linha": t.linha.values, "z": z, "erro_leitura": erro}).to_parquet(arq_teste(rc, seed), index=False)
    status(f"testar {rot}", f"CONCLUÍDO: época {ep}; {len(t)} clipes em {(time.time() - t0) / 60:.1f} min; "
                            f"ilegíveis {int(erro.sum())}")
    return 0


# ======================================================================== avaliar (UMA vez)
def etapa_avaliar(a):
    from src.arquivos import salvar_csv
    marcador = R / f"v8_avaliar_concluido{sfx()}.json"
    if marcador.exists() and not a.forcar:
        status("avaliar", f"já avaliado ({marcador}); o teste mestre é avaliado UMA vez. Nada feito.")
        return 0
    esc = json.loads((R / f"v8_escolha{sfx()}.json").read_text(encoding="utf-8"))
    notas = teste(splits())
    notas["z_R0"] = notas_r0(notas.linha.values)
    cands = ["R0"] + [nome(rc, s) for rc, s in RODADAS]
    for c in cands[1:]:
        d = pd.read_parquet(pasta_notas() / f"teste_{c}.parquet")
        notas = notas.merge(d[["linha", "z"]].rename(columns={"z": f"z_{c}"}), on="linha", how="left")
        assert notas[f"z_{c}"].notna().all(), c
    n_boot = 50 if QUICK else N_BOOT
    tm, ct = notas[notas.split == "teste_mestre"], notas[notas.split == "controle"]
    rad_en, ml = tm[tm.bloco == "radar_eval"], tm[tm.bloco == "ml_itw"]
    recortes = {"ml_itw": (ml, "locutor", "principal"),
                "in_the_wild": (tm[tm.bloco == "in_the_wild"], "locutor", "principal"),
                "radar_en": (rad_en, "clipe", "principal"),
                "radar_en_pareado": (rad_en[rad_en.pareado_teste], "clipe", "inglês"),
                "ml_itw_ingles": (ml[ml.idioma == "english"], "locutor", "inglês"),
                "ml_itw_portugues": (ml[ml.idioma == "portuguese"], "locutor", "à parte (indício)")}
    for b in v7.CONTROLES:
        recortes[b] = (ct[ct.fonte == b], "clipe", "controle")
    t0 = time.time()
    linhas, AUCS = [], {}
    for c in cands:
        e = esc["por_candidato"][c]
        cortes = {r: e["cortes"][r] for r in v7.REGRAS}
        rc = c.split("_")[0]
        for nm, (g, un, papel) in recortes.items():
            L, aucs = v7.medir(g, c, cortes, un, n_boot)
            linhas.append({"candidato": c, "receita": rc, "descrição": DESCR[rc], "recorte": nm, "papel": papel,
                           "época": esc["epoca_escolhida"].get(c), "regra escolhida (validação)": e["regra"], **L})
            AUCS[(c, nm)] = aucs
        status("avaliar", f"{c}: {len(recortes)} recortes ({(time.time() - t0) / 60:.1f} min)")
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, R / f"v8_teste_mestre{sfx()}.csv")
    princ = []
    P = list(v7.PRINCIPAIS)
    M0 = np.vstack([AUCS[("R0", s)] for s in P])
    for c in cands:
        M = np.vstack([AUCS[(c, s)] for s in P])
        pt = tab[(tab.candidato == c) & tab.recorte.isin(P)].set_index("recorte").loc[P, "AUC"]
        p0 = tab[(tab.candidato == "R0") & tab.recorte.isin(P)].set_index("recorte").loc[P, "AUC"]
        dm, dp = M.mean(0) - M0.mean(0), M.min(0) - M0.min(0)
        princ.append({"candidato": c, "descrição": DESCR[c.split("_")[0]], "média AUC (3 fontes)": float(pt.mean()),
                      "média IC2.5": v7.ic(M.mean(0))[0], "média IC97.5": v7.ic(M.mean(0))[1],
                      "pior fonte": pt.idxmin(), "AUC pior fonte": float(pt.min()),
                      **{f"AUC {s}": float(pt[s]) for s in P},
                      "dif. média vs R0 (T3)": float(pt.mean() - p0.mean()),
                      "dif. média vs R0 IC2.5": v7.ic(dm)[0], "dif. média vs R0 IC97.5": v7.ic(dm)[1],
                      "dif. pior vs R0 IC2.5": v7.ic(dp)[0], "dif. pior vs R0 IC97.5": v7.ic(dp)[1],
                      "receita vencedora pela validação": c.split("_")[0] == esc["receita_vencedora_pela_validacao"]})
    tp = pd.DataFrame(princ)
    # R1 com duas seeds: a média das duas (sem juntar as notas)
    r1 = tp[tp.candidato.str.startswith("R1_")]
    if len(r1) > 1:
        num = r1.select_dtypes("number").mean()
        tp = pd.concat([tp, pd.DataFrame([{"candidato": "R1 (média das seeds)", "descrição": DESCR["R1"], **num.to_dict(),
                                           "receita vencedora pela validação": esc["receita_vencedora_pela_validacao"] == "R1"}])],
                       ignore_index=True)
    salvar_csv(tp, R / f"v8_principal{sfx()}.csv")
    salvar_csv(tab[tab.papel == "controle"], R / f"v8_controles{sfx()}.csv")
    salvar_csv(tab[tab.recorte.isin(["radar_en", "radar_en_pareado", "ml_itw_ingles", "in_the_wild",
                                     "llamapartialspoof", "asvspoof2019_la", "asvspoof2021_df"])], R / f"v8_bloco_ingles{sfx()}.csv")
    salvar_json({"quando": agora(), "n_boot": n_boot, "seed": SEED, "principal": json.loads(tp.to_json(orient="records", force_ascii=False)),
                 "escolha_validacao": esc["receita_vencedora_pela_validacao"], "epoca_escolhida": esc["epoca_escolhida"],
                 "minutos": round((time.time() - t0) / 60, 1)}, R / f"v8_resumo{sfx()}.json")
    salvar_json({"quando": agora()}, marcador)
    status("avaliar", f"CONCLUÍDO em {(time.time() - t0) / 60:.1f} min")
    return 0


# ======================================================================== fila
def etapa_fila(a):
    """Roda as etapas na ordem do protocolo, cada uma num processo novo; para no primeiro erro."""
    passos = [["preparar"]] + [["treinar", "--receita", rc, "--seed", str(s)] for rc, s in RODADAS] + [["escolher"]] + \
             [["testar", "--receita", rc, "--seed", str(s)] for rc, s in RODADAS] + [["avaliar"]]
    for p in passos:
        log = LOGS / ("v8_" + "_".join(x.lstrip("-") for x in p) + sfx() + ".log")
        cmd = [sys.executable, str(Path(__file__)), *p, "--num-workers", str(a.num_workers), "--log", str(log)]
        if QUICK:
            cmd.append("--quick")
        status("fila", f"rodando: {' '.join(p)} (log {log})")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            status("fila", f"PAROU: '{' '.join(p)}' terminou com código {r.returncode} (ver {log}). "
                           f"Rodar 'fila' de novo retoma do ponto onde parou.")
            return r.returncode
    status("fila", "CONCLUÍDA")
    return 0


def main():
    global QUICK, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["preparar", "checar", "treinar", "escolher", "testar", "avaliar", "fila"])
    ap.add_argument("--receita", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--forcar", action="store_true", help="refaz 'avaliar' mesmo com marcador (só em erro de código)")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    QUICK, ARGS = a.quick, a
    v7.QUICK = a.quick
    if a.log:
        from src.execucao import Tee
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        f = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.__stdout__, f), Tee(sys.__stderr__, f)
    print(f"\n######## {agora()} | {' '.join(sys.argv)} | pid {os.getpid()}", flush=True)
    if a.etapa != "fila":
        ti.VIGIA = ti.Vigia(LIMITE_GB, lambda msg: status(a.etapa, f"PAROU: {msg}"))
        ti.VIGIA.start()
    t0 = time.time()
    ti.sem_suspensao(True)
    try:
        return {"preparar": etapa_preparar, "checar": etapa_checar, "treinar": etapa_treinar, "escolher": etapa_escolher, "testar": etapa_testar,
                "avaliar": etapa_avaliar, "fila": etapa_fila}[a.etapa](a)
    except BaseException:
        traceback.print_exc()
        status(a.etapa, "FALHOU (ver log)")
        return 1
    finally:
        ti.sem_suspensao(False)
        if ti.VIGIA:
            ti.VIGIA.parar = True
        print(f"PICO_MEMORIA_GB={ti.VIGIA.pico if ti.VIGIA else 0:.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
