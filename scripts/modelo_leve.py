#!/usr/bin/env python3
"""Modelo leve (Fase 1): corte do XLS-R depois da camada 7 (V1), fp16 (V2) e int8 (V3) contra o modelo
inteiro (V0). Protocolo congelado em docs/MODELO_LEVE_PROTOCOLO.md. NÃO treina nada, NÃO reajusta a
regressão T3, NÃO toca no teste mestre.

Etapas (cada uma retomável; status em logs/modelo_leve_status.txt):
    python scripts/modelo_leve.py amostra            # 2.000 clipes da validação + recortes da latência
    python scripts/modelo_leve.py equiv              # vetores e notas de V0/V1/V2 (placa de vídeo) + regra da V1
    python scripts/modelo_leve.py v3notas            # vetores e notas da V3 (int8, processador)
    python scripts/modelo_leve.py lat A cpu4|cpu1    # (a) leitura e preparo
    python scripts/modelo_leve.py lat V0|V1|V2|V3 gpu|cpu4|cpu1
    python scripts/modelo_leve.py tamanho            # parâmetros, MB, salva V1/V2/V3 em models/
    python scripts/modelo_leve.py resumo             # tabelas finais, json e figuras
"""
import atexit
import ctypes
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

R, DP, LOGS = Path("results"), Path("data/processed"), Path("logs")
PASTA = DP / "modelo_leve_amostra"
PARC = R / "modelo_leve_parcial"
SPLITS = DP / "splits_v7.parquet"
SHA_SPLITS = "24d3aea93654711a80ea1482799c5de3ec180ec5e336c719935e98b7f7a54c63"
STATUS = LOGS / "modelo_leve_status.txt"
BLOQUEIO = LOGS / "modelo_leve_BLOQUEIO.txt"
CAMADAS = (5, 6, 7)
SEED, N_AMOSTRA, SR = 42, 2000, 16000
DURS = (3, 5, 8)
N_AQUEC, N_CLIPES, N_REP = 5, 5, 6
TOL_NOTA, TOL_AUC4 = 1e-3, 5e-5
LIM_MEDIA, LIM_FONTE = 0.003, 0.01
ENG = {"V0": ("completo", None), "V1": ("cortado", None), "V2": ("cortado", "fp16"), "V3": ("cortado", "int8")}


# --------------------------------------------------------------------------- utilidades
def agora():
    return datetime.now().isoformat(timespec="seconds")


def status(etapa, msg):
    LOGS.mkdir(exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"{agora()} [{etapa}] {msg}\n")
    print(f"[{etapa}] {msg}", flush=True)


def impedir_suspensao():
    """Impede a suspensão do Windows neste processo e restaura ao sair (o powercfg está bloqueado)."""
    try:
        k = ctypes.windll.kernel32
        k.SetThreadExecutionState(0x80000001)                 # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        atexit.register(lambda: k.SetThreadExecutionState(0x80000000))
    except Exception as e:  # noqa: BLE001
        status("sistema", f"não consegui impedir a suspensão: {e}")


def bloquear(msg):
    BLOQUEIO.parent.mkdir(exist_ok=True)
    BLOQUEIO.write_text(f"{agora()}\n{msg}\n", encoding="utf-8")
    status("BLOQUEIO", msg)
    sys.exit(4)


def salvar_json(obj, p):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def pico_ram_mb():
    import psutil
    mi = psutil.Process().memory_info()
    return float(getattr(mi, "peak_wset", mi.rss)) / 2**20


def chave_av(sp):
    """Fonte de avaliação (igual a scripts/v7.py: chave_val): estúdio por banco; os outros por bloco."""
    return np.where(sp.bloco == "estudio", sp.fonte, sp.bloco)


def resolver(p):
    """radar_eval e ml_itw têm caminho relativo a data/raw/teste_internet (os outros, a data/raw/... já completo)."""
    p = str(p)
    if os.path.exists(p):
        return p
    q = os.path.join("data", "raw", "teste_internet", p)
    return q if os.path.exists(q) else p


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- modelos
def sistema_t3():
    z = np.load(R / "v7_t3_coeficientes.npz")
    return (z["mu"].astype(np.float32), z["sd"].astype(np.float32), z["coef"].astype(np.float32),
            float(z["intercepto"]))


def montar(variante, device):
    """Devolve (modelo, dtype_torch). V0 completo; V1 cortado (fp32); V2 cortado em fp16; V3 cortado + int8."""
    from src.ssl import carregar_ssl
    _, extra = ENG[variante]
    dtype = {"V0": "float16" if device == "cuda" else "float32", "V1": "float32", "V2": "float16",
             "V3": "float32"}[variante]
    if variante == "V0":
        return carregar_ssl(device=device, dtype=dtype), dtype
    m = carregar_ssl(device="cpu", dtype="float32")
    m.encoder.layers = m.encoder.layers[:max(CAMADAS)]
    m.encoder.layer_norm = torch.nn.Identity()        # só a última saída do modelo inteiro o recebia (ver protocolo)
    if variante == "V3":
        torch.backends.quantized.engine = "onednn" if "onednn" in torch.backends.quantized.supported_engines else \
            torch.backends.quantized.engine
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        return m.eval(), "float32"
    if variante == "V2":
        m = m.half()
    return m.to(device).eval(), dtype


def normalizar(x):
    x = x.float()
    return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)


@torch.inference_mode()
def vetores(model, clipes, device, dtype, bs=16):
    """(n, 3072) float32: média no tempo, só nos quadros válidos, das saídas 5, 6, 7 (como src/ssl.py)."""
    dt = {"float16": torch.float16, "float32": torch.float32}[dtype]
    ordem = np.argsort([c.shape[0] for c in clipes], kind="stable")
    X = np.zeros((len(clipes), 3072), dtype=np.float32)
    for i in range(0, len(clipes), bs):
        idx = ordem[i:i + bs]
        cl = [torch.from_numpy(clipes[j]) for j in idx]
        n = torch.tensor([c.numel() for c in cl])
        L = int(n.max())
        lote = torch.zeros(len(cl), L)
        mask = torch.zeros(len(cl), L, dtype=torch.long)
        for k, c in enumerate(cl):
            lote[k, :c.numel()] = normalizar(c)
            mask[k, :c.numel()] = 1
        out = model(lote.to(device, dt), attention_mask=mask.to(device), output_hidden_states=True)
        nq = model._get_feat_extract_output_lengths(n).to(device)
        T = out.hidden_states[0].shape[1]
        valido = (torch.arange(T, device=device)[None, :] < nq[:, None]).float()[:, :, None]
        cont = valido.sum(dim=1)
        f = torch.cat([(out.hidden_states[k].float() * valido).sum(dim=1) / cont for k in CAMADAS], dim=1)
        X[idx] = f.cpu().numpy()
        del out, f, valido
        if device == "cuda":
            torch.cuda.empty_cache()          # formas variadas fragmentam o alocador (OOM visto no Windows)
    return X


def nota(X, mu, sd, coef, b):
    return (((X - mu) / sd) @ coef + b).astype(np.float64)


def carregar_amostra():
    meta = pd.read_parquet(PASTA / "amostra.parquet")
    flat = np.load(PASTA / "clipes.npy", mmap_mode="r")
    off = np.load(PASTA / "offsets.npy")
    clipes = [np.asarray(flat[off[i]:off[i + 1]]) for i in range(len(meta))]
    return meta, clipes


def auc_fontes(y, z, fontes):
    from sklearn.metrics import roc_auc_score
    return {f: float(roc_auc_score(y[fontes == f], z[fontes == f])) for f in sorted(set(fontes))}


# --------------------------------------------------------------------------- etapa: amostra
def etapa_amostra():
    from src.preproc import ler_mono_16k, preparar_clipe
    torch.set_num_threads(4)
    if sha256(SPLITS) != SHA_SPLITS:
        bloquear("SHA256 do splits_v7.parquet não bate com o do protocolo da v7")
    PASTA.mkdir(parents=True, exist_ok=True)
    sp = pd.read_parquet(SPLITS)
    va = sp[sp.split == "validacao"].copy()
    va["fonte_av"] = chave_av(va)
    cnt = va.groupby(["fonte_av", "label"]).size().unstack(fill_value=0)
    fontes = sorted(cnt[(cnt.get(0, 0) >= 20) & (cnt.get(1, 0) >= 20)].index)
    status("amostra", f"fontes com as duas classes: {fontes}")
    rng = np.random.default_rng(SEED)
    cotas = {f: N_AMOSTRA // len(fontes) + (1 if i < N_AMOSTRA % len(fontes) else 0) for i, f in enumerate(fontes)}
    partes = []
    for f in fontes:
        g = va[va.fonte_av == f]
        q = cotas[f]
        n0, n1 = (g.label == 0).sum(), (g.label == 1).sum()
        q0 = min(q // 2, n0)
        q1 = min(q - q0, n1)
        q0 = min(q - q1, n0)
        for lab, k in ((0, q0), (1, q1)):
            gg = g[g.label == lab]
            partes.append(gg.iloc[np.sort(rng.choice(len(gg), k, replace=False))])
    am = pd.concat(partes).reset_index(drop=True)
    assert len(am) == N_AMOSTRA and (am.split == "validacao").all(), len(am)
    clipes, erros, durs = [], [], []
    for i, p in enumerate(am.caminho):
        try:
            x, sr = ler_mono_16k(resolver(p))
            y, est = preparar_clipe(x, sr)
            clipes.append(y.numpy().astype(np.float32))
            durs.append(est["dur_entrada_s"])
        except Exception as e:  # noqa: BLE001
            erros.append((p, str(e)))
            clipes.append(None)
            durs.append(np.nan)
        if (i + 1) % 250 == 0:
            status("amostra", f"{i + 1}/{len(am)} preparados")
    ok = np.array([c is not None for c in clipes])
    am = am[ok].reset_index(drop=True)
    am["dur_entrada_s"] = np.array(durs)[ok]
    clipes = [c for c in clipes if c is not None]
    off = np.concatenate([[0], np.cumsum([c.shape[0] for c in clipes])]).astype(np.int64)
    np.save(PASTA / "clipes.npy", np.concatenate(clipes))
    np.save(PASTA / "offsets.npy", off)
    am[["linha", "caminho", "fonte_av", "bloco", "fonte", "label", "split", "duracao_s", "dur_entrada_s"]].to_parquet(
        PASTA / "amostra.parquet", index=False)
    # clipes da latência: reais da validação, duração original entre D e D+2 s (senão >= D), recorte central exato
    reais = va[(va.label == 0)]
    usados, lat = set(), []
    for D in DURS:
        pool = reais[(reais.duracao_s >= D) & (reais.duracao_s <= D + 2)]
        if len(pool) < N_CLIPES:
            pool = reais[reais.duracao_s >= D]
        pool = pool[~pool.caminho.isin(usados)]
        esc = pool.iloc[np.sort(rng.choice(len(pool), N_CLIPES, replace=False))]
        for p, d0 in zip(esc.caminho, esc.duracao_s):
            x, _ = ler_mono_16k(resolver(p))
            n = D * SR
            assert x.numel() >= n, (p, x.numel())
            a = (x.numel() - n) // 2
            lat.append({"duracao": D, "caminho": p, "dur_original_s": float(x.numel() / SR),
                        "recorte": x[a:a + n].numpy().astype(np.float32)})
            usados.add(p)
    np.savez(PASTA / "lat.npz", **{f"c{i}": l["recorte"] for i, l in enumerate(lat)})
    salvar_json([{k: v for k, v in l.items() if k != "recorte"} for l in lat], PASTA / "lat.json")
    salvar_json({"quando": agora(), "n": int(len(am)), "fontes": fontes, "cotas": cotas, "erros_leitura": erros,
                 "por_fonte_classe": am.groupby(["fonte_av", "label"]).size().rename("n").reset_index().to_dict("records"),
                 "sha256_splits": SHA_SPLITS}, PASTA / "amostra.json")
    status("amostra", f"CONCLUÍDO: {len(am)} clipes ({len(erros)} erros de leitura); {len(lat)} recortes de latência")


# --------------------------------------------------------------------------- etapa: equiv
def etapa_equiv():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta, clipes = carregar_amostra()
    mu, sd, coef, b = sistema_t3()
    notas = meta[["linha", "fonte_av", "label"]].copy()
    cfg = [("V0_fp32", "V0", "float32"), ("V1_fp32", "V1", "float32"), ("V0_fp16", "V0", "float16"),
           ("V1_fp16", "V2", "float16")]
    from src.ssl import carregar_ssl
    for nome, var, dt in cfg:
        arq = PASTA / f"X_{nome}.npy"
        if arq.exists():
            continue
        t0 = time.time()
        if var == "V0":
            m = carregar_ssl(device="cuda", dtype=dt)
        else:
            m, _ = montar(var if dt == "float16" else "V1", "cuda")
        X = vetores(m, clipes, "cuda", dt, bs=4 if dt == "float32" else 8)
        np.save(arq, X)
        del m
        torch.cuda.empty_cache()
        status("equiv", f"{nome}: {len(clipes)} clipes em {time.time() - t0:.0f} s")
    # conferência 50 clipes: V1 fp32 no processador (4 threads) contra a placa de vídeo
    arq = PASTA / "X_V1_fp32_cpu50.npy"
    if not arq.exists():
        torch.set_num_threads(4)
        m, _ = montar("V1", "cpu")
        np.save(arq, vetores(m, clipes[:50], "cpu", "float32", bs=8))
        status("equiv", "V1 fp32 no processador: 50 clipes")
    equivalencia(meta, notas, mu, sd, coef, b)


def comparar(nome, ref, var, y, fontes, X_ref, X_var, mu, sd, coef, b):
    z0, z1 = nota(X_ref, mu, sd, coef, b), nota(X_var, mu, sd, coef, b)
    a0, a1 = auc_fontes(y, z0, fontes), auc_fontes(y, z1, fontes)
    lin = {"comparacao": nome, "referencia": ref, "variante": var, "n": len(y),
           "dif_nota_max": float(np.abs(z0 - z1).max()), "dif_nota_mediana": float(np.median(np.abs(z0 - z1))),
           "auc_media_ref": float(np.mean(list(a0.values()))), "auc_media_var": float(np.mean(list(a1.values())))}
    lin["queda_auc_media"] = lin["auc_media_ref"] - lin["auc_media_var"]
    lin["maior_queda_auc_fonte"] = max(a0[f] - a1[f] for f in a0)
    for i, k in enumerate(CAMADAS):
        d = np.abs(X_ref[:, i * 1024:(i + 1) * 1024] - X_var[:, i * 1024:(i + 1) * 1024])
        lin[f"dif_saida{k}_max"], lin[f"dif_saida{k}_mediana"] = float(d.max()), float(np.median(d))
    porf = [{"comparacao": nome, "fonte": f, "auc_ref": a0[f], "auc_var": a1[f], "queda": a0[f] - a1[f]} for f in a0]
    return lin, porf, z0, z1


def equivalencia(meta, notas, mu, sd, coef, b):
    y, fo = meta.label.values, meta.fonte_av.values
    X = {k: np.load(PASTA / f"X_{k}.npy") for k in ("V0_fp32", "V1_fp32", "V0_fp16", "V1_fp16")}
    lins, porfs = [], []
    for nome, ref, var, kr, kv in [("V1 fp32 x V0 fp32", "V0 fp32", "V1 fp32", "V0_fp32", "V1_fp32"),
                                   ("V2 fp16 x V0 fp16", "V0 fp16", "V2 fp16", "V0_fp16", "V1_fp16"),
                                   ("V2 fp16 x V0 fp32 (informativo)", "V0 fp32", "V2 fp16", "V0_fp32", "V1_fp16"),
                                   ("V0 fp16 x V0 fp32 (informativo)", "V0 fp32", "V0 fp16", "V0_fp32", "V0_fp16")]:
        lin, pf, z0, z1 = comparar(nome, ref, var, y, fo, X[kr], X[kv], mu, sd, coef, b)
        lins.append(lin)
        porfs += pf
        notas[f"z_{kr}"], notas[f"z_{kv}"] = z0, z1
    cpu = np.load(PASTA / "X_V1_fp32_cpu50.npy")
    d50 = float(np.abs(nota(cpu, mu, sd, coef, b) - nota(X["V1_fp32"][:50], mu, sd, coef, b)).max())
    pd.DataFrame(lins).to_csv(PARC / "equiv_resumo.csv", index=False) if PARC.exists() else None
    PARC.mkdir(exist_ok=True)
    pd.DataFrame(lins).to_csv(PARC / "equiv_resumo.csv", index=False)
    pd.DataFrame(porfs).to_csv(PARC / "equiv_por_fonte.csv", index=False)
    notas.to_parquet(PASTA / "notas.parquet", index=False)
    salvar_json({"dif_nota_max_V1fp32_placa_x_processador_50clipes": d50}, PARC / "equiv_cpu50.json")
    v1 = lins[0]
    ok = v1["dif_nota_max"] <= TOL_NOTA and abs(v1["auc_media_ref"] - v1["auc_media_var"]) <= TOL_AUC4 and \
        all(abs(p["auc_ref"] - p["auc_var"]) <= TOL_AUC4 for p in porfs if p["comparacao"] == lins[0]["comparacao"])
    status("equiv", f"V1 fp32 x V0 fp32: dif máx nota {v1['dif_nota_max']:.2e}; queda AUC média {v1['queda_auc_media']:.2e}"
                    f"; V1 equivalente = {ok}")
    if not ok:
        bloquear(f"V1 não é equivalente à V0: diferença máxima de nota {v1['dif_nota_max']:.3e} (limite {TOL_NOTA}); "
                 f"AUC média V0 {v1['auc_media_ref']:.5f} x V1 {v1['auc_media_var']:.5f}")


# --------------------------------------------------------------------------- etapa: v3notas
def etapa_v3notas():
    torch.set_num_threads(4)
    meta, clipes = carregar_amostra()
    mu, sd, coef, b = sistema_t3()
    arq = PASTA / "X_V3.npy"
    if not arq.exists():
        try:
            m, _ = montar("V3", "cpu")
        except Exception as e:  # noqa: BLE001
            salvar_json({"funcionou": False, "erro": repr(e)}, PARC / "v3_erro.json")
            status("v3notas", f"QUANTIZAÇÃO NÃO FUNCIONOU: {e!r}")
            return
        t0 = time.time()
        # em pedaços, para o status andar e ser retomável
        partes = []
        for i in range(0, len(clipes), 200):
            arq_p = PASTA / f"X_V3_{i:05d}.npy"
            if arq_p.exists():
                partes.append(np.load(arq_p))
                continue
            Xp = vetores(m, clipes[i:i + 200], "cpu", "float32", bs=8)
            np.save(arq_p, Xp)
            partes.append(Xp)
            status("v3notas", f"{min(i + 200, len(clipes))}/{len(clipes)} ({time.time() - t0:.0f} s)")
        np.save(arq, np.concatenate(partes))
    y, fo = meta.label.values, meta.fonte_av.values
    X0, X3 = np.load(PASTA / "X_V0_fp32.npy"), np.load(arq)
    lin, pf, z0, z3 = comparar("V3 int8 x V0 fp32", "V0 fp32", "V3 int8", y, fo, X0, X3, mu, sd, coef, b)
    pd.DataFrame([lin]).to_csv(PARC / "equiv_v3_resumo.csv", index=False)
    pd.DataFrame(pf).to_csv(PARC / "equiv_v3_por_fonte.csv", index=False)
    nt = pd.read_parquet(PASTA / "notas.parquet")
    nt["z_V3_int8"] = z3
    nt.to_parquet(PASTA / "notas.parquet", index=False)
    salvar_json({"funcionou": True}, PARC / "v3_erro.json")
    status("v3notas", f"CONCLUÍDO: dif máx nota {lin['dif_nota_max']:.3f}; queda AUC média {lin['queda_auc_media']:.4f}")


# --------------------------------------------------------------------------- etapa: lat
def estat(v):
    v = np.asarray(v) * 1000
    return {"n": int(len(v)), "mediana_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)),
            "media_ms": float(v.mean())}


def etapa_lat(variante, ap):
    from src.preproc import ler_mono_16k, preparar_clipe
    threads = {"gpu": 4, "cpu4": 4, "cpu1": 1}[ap if variante != "A" else ap]
    torch.set_num_threads(threads)
    PARC.mkdir(exist_ok=True)
    saida = PARC / f"lat_{variante}_{ap}.json"
    if saida.exists():
        status("lat", f"{saida.name} já existe (pulando)")
        return
    lat = json.loads((PASTA / "lat.json").read_text(encoding="utf-8"))
    rec = np.load(PASTA / "lat.npz")
    por_dur = {D: [i for i, l in enumerate(lat) if l["duracao"] == D] for D in DURS}
    res = {"variante": variante, "aparelho": ap, "threads": threads, "quando": agora(), "duracoes": {}}
    if variante == "A":                                           # (a) leitura do arquivo e preparo
        for D in DURS:
            idx = por_dur[D]
            for _ in range(N_AQUEC):
                x, sr = ler_mono_16k(resolver(lat[idx[0]]["caminho"]))
                preparar_clipe(x, sr)
            t = []
            for r in range(N_REP):
                for i in idx:
                    t0 = time.perf_counter()
                    x, sr = ler_mono_16k(resolver(lat[i]["caminho"]))
                    preparar_clipe(x, sr)
                    t.append(time.perf_counter() - t0)
            res["duracoes"][D] = {"a": t, **{"estat_a": estat(t)}}
        salvar_json(res, saida)
        status("lat", f"A {ap}: " + ", ".join(f"{D}s {res['duracoes'][D]['estat_a']['mediana_ms']:.0f} ms" for D in DURS))
        return
    dev = "cuda" if ap == "gpu" else "cpu"
    mu, sd, coef, b = sistema_t3()
    model, dtype = montar(variante, dev)
    dt = {"float16": torch.float16, "float32": torch.float32}[dtype]
    mu_t, sd_t, coef_t = (torch.from_numpy(a).to(dev) for a in (mu, sd, coef))
    chamadas = {"n": 0}
    for l in model.encoder.layers:
        l.register_forward_hook(lambda *a: chamadas.__setitem__("n", chamadas["n"] + 1))

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def uma(x):
        sync()
        t0 = time.perf_counter()
        xt = normalizar(torch.from_numpy(x))[None].to(dev, dt)
        out = model(xt, output_hidden_states=True)
        hs = [out.hidden_states[k] for k in CAMADAS]
        sync()
        t1 = time.perf_counter()
        f = torch.cat([h.float().mean(dim=1) for h in hs], dim=1)         # lote 1, sem enchimento: todos válidos
        z = (((f - mu_t) / sd_t) @ coef_t).item() + b                    # .item() sincroniza
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, z

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for D in DURS:
        idx = por_dur[D]
        xs = [rec[f"c{i}"] for i in idx]
        for _ in range(N_AQUEC):
            uma(xs[0])
        bs, cs, zs = [], [], []
        for r in range(N_REP):
            for k in range(len(xs)):
                tb, tc, z = uma(xs[(k + r) % len(xs)])
                bs.append(tb), cs.append(tc), zs.append(z)
        res["duracoes"][D] = {"b": bs, "c": cs, "estat_b": estat(bs), "estat_c": estat(cs),
                              "estat_bc": estat(np.array(bs) + np.array(cs)), "nota": zs}
        status("lat", f"{variante} {ap} {D}s: modelo {res['duracoes'][D]['estat_b']['mediana_ms']:.0f} ms, "
                      f"regressão {res['duracoes'][D]['estat_c']['mediana_ms']:.2f} ms")
    # camadas calculadas por chamada (a contagem inclui o aquecimento): chamadas / nº de execuções
    n_exec = len(DURS) * (N_AQUEC + N_REP * N_CLIPES)
    res["camadas_executadas_por_audio"] = chamadas["n"] / n_exec
    res["pico_ram_processo_mb"] = pico_ram_mb()
    if dev == "cuda":
        res["pico_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
        res["gpu_modelo_mb"] = None
    salvar_json(res, saida)


# --------------------------------------------------------------------------- etapa: tamanho
def parametros(m):
    comp = {"extrator_convolucoes": m.feature_extractor, "projecao_caracteristicas": m.feature_projection,
            "convolucao_posicional": m.encoder.pos_conv_embed, "camadas_transformer": m.encoder.layers,
            "layer_norm_final": m.encoder.layer_norm}
    out = {k: sum(p.numel() for p in v.parameters()) for k, v in comp.items()}
    total = sum(p.numel() for p in m.parameters())
    out["outros"] = total - sum(out.values())
    out["total"] = total
    return out


def etapa_tamanho():
    from safetensors.torch import save_file
    from src.ssl import carregar_ssl
    Path("models").mkdir(exist_ok=True)
    PARC.mkdir(exist_ok=True)
    npz = R / "v7_t3_coeficientes.npz"
    reg_mb = npz.stat().st_size / 2**20
    snap = next((Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-xls-r-300m/snapshots").glob("*"))
    arq0 = max((p for p in snap.iterdir() if p.suffix in (".bin", ".safetensors")), key=lambda p: p.stat().st_size)
    linhas = []
    m0 = carregar_ssl(device="cpu", dtype="float32")
    p0 = parametros(m0)
    linhas.append({"variante": "V0", "descricao": "modelo completo (24 camadas)", **p0,
                   "arquivo": str(arq0).replace(str(Path.home()), "~"), "mb_modelo": arq0.stat().st_size / 2**20,
                   "mb_regressao": reg_mb})
    m1, _ = montar("V1", "cpu")
    p1 = parametros(m1)
    sd1 = {k: v.contiguous() for k, v in m1.state_dict().items()}
    a1 = Path("models/modelo_leve_V1_fp32.safetensors")
    save_file(sd1, str(a1))
    linhas.append({"variante": "V1", "descricao": "cortado depois da 7ª camada (fp32)", **p1, "arquivo": a1.as_posix(),
                   "mb_modelo": a1.stat().st_size / 2**20, "mb_regressao": reg_mb})
    a2 = Path("models/modelo_leve_V2_fp16.safetensors")
    save_file({k: v.half().contiguous() for k, v in sd1.items()}, str(a2))
    linhas.append({"variante": "V2", "descricao": "V1 em fp16", **p1, "arquivo": a2.as_posix(),
                   "mb_modelo": a2.stat().st_size / 2**20, "mb_regressao": reg_mb})
    try:
        m3, _ = montar("V3", "cpu")
        a3 = Path("models/modelo_leve_V3_int8.pt")
        torch.save(m3.state_dict(), a3)
        linhas.append({"variante": "V3", "descricao": "V1 com int8 dinâmico nas camadas lineares", **p1,
                       "arquivo": a3.as_posix(), "mb_modelo": a3.stat().st_size / 2**20, "mb_regressao": reg_mb})
    except Exception as e:  # noqa: BLE001
        status("tamanho", f"V3 não pôde ser salva: {e!r}")
    df = pd.DataFrame(linhas)
    df["parametros_regressao"] = 3073
    df.to_csv(PARC / "tamanho.csv", index=False)
    status("tamanho", "\n" + df[["variante", "total", "mb_modelo", "mb_regressao"]].to_string(index=False))


# --------------------------------------------------------------------------- etapa: resumo
def etapa_resumo():
    sys.path.insert(0, str(ROOT / "scripts"))
    import modelo_leve_resumo as rs
    rs.main()


def main():
    impedir_suspensao()
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    PARC.mkdir(exist_ok=True)
    e = a[0]
    status(e, "início " + " ".join(a[1:]))
    if e == "amostra":
        etapa_amostra()
    elif e == "equiv":
        etapa_equiv()
    elif e == "v3notas":
        etapa_v3notas()
    elif e == "lat":
        etapa_lat(a[1], a[2])
    elif e == "tamanho":
        etapa_tamanho()
    elif e == "resumo":
        etapa_resumo()
    else:
        print(__doc__)
        return 2
    status(e, "fim " + " ".join(a[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
