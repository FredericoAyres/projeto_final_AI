#!/usr/bin/env python3
"""Teste externo "WhatsApp" do sistema (detector T3/F0 + modulo de contexto de golpe). NAO treina nem ajusta nada.

Etapas (protocolo: docs/PROTOCOLO_TESTE_WHATSAPP.md, escrito ANTES das notas):
    manifesto   monta data/processed/manifest_whatsapp.parquet a partir do zip extraido (sem nomes de pessoas)
    recodificar condicao B: clones recodificados para Opus mono ~16 kbps .ogg (PyAV), em data/
    detector    notas T3 (cortes M, A', F5) e F0 (cortes A e B) nas condicoes A e B -> data/processed/tw_notas_detector.parquet
    contexto    Whisper + gpt-oss-120b (Groq) nos 100 WAV, uma chamada por clipe, a cada 12 s -> data/processed/tw_contexto.jsonl
    resumo      junta tudo, calcula metricas -> results/teste_whatsapp_*.csv/json (sem texto, sem nome)
    marcar      grava results/teste_whatsapp_concluido.json (impede rodar de novo)

Transcricoes e audios ficam so em data/ (fora do git). Em results/ entram apenas pessoa_id, notas e categorias.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R, DP = ROOT / "results", ROOT / "data" / "processed"
EXTR = ROOT / "data" / "teste_whatsapp" / "extraido"
MANIFESTO = DP / "manifest_whatsapp.parquet"
NOTAS_DET = DP / "tw_notas_detector.parquet"
CTX_JSONL = DP / "tw_contexto.jsonl"
OPUS_DIR = ROOT / "data" / "teste_whatsapp" / "opus16k"
MARCADOR = R / "teste_whatsapp_concluido.json"
PROTOCOLO = ROOT / "docs" / "PROTOCOLO_TESTE_WHATSAPP.md"
GRUPOS = ("real_sem_golpe", "real_com_golpe", "ia_pedido_dinheiro", "ia_neutro")
INTERVALO_S = 12


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lf(p):
    """SHA256 com fim de linha LF (o do commit/congelamento; no disco do Windows o arquivo esta em CRLF)."""
    return hashlib.sha256(Path(p).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def exigir_nao_concluido():
    if MARCADOR.exists():
        sys.exit(f"{MARCADOR.name} existe: o teste ja foi avaliado uma vez e nao roda de novo.")
    if not PROTOCOLO.exists():
        sys.exit("docs/PROTOCOLO_TESTE_WHATSAPP.md precisa existir ANTES de qualquer nota/chamada.")


# --------------------------------------------------------------------------------------------- manifesto
def etapa_manifesto(a):
    import soundfile as sf
    m = pd.read_csv(EXTR / "coleta" / "manifesto_geral.csv", encoding="utf-8-sig")
    assert len(m) == 100 and m.pessoa_id.nunique() == 23
    cont = m.groupby(["rotulo", "tema"]).size().to_dict()
    esperado = {("real", "com_golpe"): 27, ("real", "sem_golpe"): 17, ("sintetico", "pedidos_de_dinheiro"): 46,
                ("sintetico", "cotidiano"): 10}
    if cont != esperado:
        sys.exit(f"contagens fora do esperado: {cont}")
    faltam = [x for x in m.arquivo if not (EXTR / x).exists()]
    if faltam:
        sys.exit(f"{len(faltam)} caminhos do manifesto nao existem")
    linhas = []
    for i, r in m.iterrows():
        info = sf.info(str(EXTR / r.arquivo))
        real = r.rotulo == "real"
        if real:
            grupo = "real_com_golpe" if r.tema == "com_golpe" else "real_sem_golpe"
        else:
            grupo = "ia_pedido_dinheiro" if r.tema == "pedidos_de_dinheiro" else "ia_neutro"
        linhas.append({"id": f"tw{i + 1:03d}", "pessoa_id": r.pessoa_id, "caminho": str(Path("data/teste_whatsapp/extraido") / r.arquivo).replace(os.sep, "/"),
                       "autenticidade": "real" if real else "ia",
                       "conteudo": "golpe" if grupo in ("real_com_golpe", "ia_pedido_dinheiro") else "normal",
                       "grupo": grupo, "duracao_s": round(info.duration, 3), "taxa_hz": int(info.samplerate),
                       "canais": int(info.channels)})
    df = pd.DataFrame(linhas)                              # NAO copia nome_local, texto nem referencia
    assert not any("nome" in c for c in df.columns)
    df.to_parquet(MANIFESTO, index=False)
    print(df.groupby("grupo").agg(n=("id", "size"), dur_med=("duracao_s", "median"), taxa=("taxa_hz", lambda s: sorted(set(s))),
                                   canais=("canais", lambda s: sorted(set(s)))))
    print("autenticidade x conteudo:\n", pd.crosstab(df.autenticidade, df.conteudo))
    pp = df.groupby("pessoa_id").autenticidade.nunique()
    print("pessoas com real E ia:", int((pp == 2).sum()), "de", len(pp))
    print("sha256 manifesto:", sha256(MANIFESTO))


# --------------------------------------------------------------------------------------------- condicao B
def etapa_recodificar(a):
    import av
    from fractions import Fraction
    df = pd.read_parquet(MANIFESTO)
    OPUS_DIR.mkdir(parents=True, exist_ok=True)
    for _, r in df[df.autenticidade == "ia"].iterrows():
        dst = OPUS_DIR / f"{r.id}.ogg"
        with av.open(str(ROOT / r.caminho)) as src, av.open(str(dst), "w", format="ogg") as out:
            s_in = src.streams.audio[0]
            s_out = out.add_stream("libopus", rate=16000)
            s_out.bit_rate = 16000
            s_out.layout = "mono"
            rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for fr in src.decode(s_in):
                for f2 in rs.resample(fr):
                    for pk in s_out.encode(f2):
                        out.mux(pk)
            for f2 in rs.resample(None):
                for pk in s_out.encode(f2):
                    out.mux(pk)
            for pk in s_out.encode(None):
                out.mux(pk)
    n = len(list(OPUS_DIR.glob("*.ogg")))
    print("recodificados:", n)


# --------------------------------------------------------------------------------------------- detector
def etapa_detector(a):
    exigir_nao_concluido()
    import inferencia_referencia as ir
    from src.preproc import ler_mono_16k, preparar_clipe
    df = pd.read_parquet(MANIFESTO)
    model, dtype = ir.carregar("v1", "cuda" if _cuda() else "cpu")
    ap = "cuda" if _cuda() else "cpu"
    mu, sd, coef, b = ir.pesos_t3()
    f0 = np.load(R / "v4final_f0_coeficientes.npz")
    pad = np.load(R / "v4_parcial" / "padronizacao.npz")
    assert [int(c) for c in f0["camadas"]] == [5, 6, 7]
    f0_coef = f0["coef"].astype(np.float64)

    def f0_nota(v):                                          # mesma conta de scripts/teste_internet.py: aplicar_f0
        x = np.concatenate([(v[i * 1024:(i + 1) * 1024] - pad["mu"][l]) / pad["sd"][l] for i, l in enumerate((5, 6, 7))])
        return float(x @ f0_coef + float(f0["intercepto"]))

    linhas = []
    for _, r in df.iterrows():
        for cond in ("A", "B"):
            if cond == "B":
                if r.autenticidade == "real":
                    continue                                # B: so os clones sao recodificados
                caminho = OPUS_DIR / f"{r.id}.ogg"
            else:
                caminho = ROOT / r.caminho
            x, sr = ler_mono_16k(str(caminho))
            clipe, est = preparar_clipe(x, sr)
            v = ir.vetor_de_clipe(model, clipe, ap, dtype)
            linhas.append({"id": r.id, "condicao": cond, "z_T3": float(ir.nota(v, mu, sd, coef, b)), "z_F0": f0_nota(v),
                           "dur_usada_s": est["dur_entrada_s"], "corte_silencio": bool(est["corte_aplicado"])})
        print(r.id, flush=True)
    out = pd.DataFrame(linhas)
    out.to_parquet(NOTAS_DET, index=False)
    print(out.groupby("condicao").size())


def _cuda():
    import torch
    return torch.cuda.is_available()


# --------------------------------------------------------------------------------------------- contexto
def etapa_contexto(a):
    exigir_nao_concluido()
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY nao esta no ambiente: parar e avisar.")
    from src import contexto_golpe as cg
    cong = json.loads((R / "contexto_golpe_congelado.json").read_text(encoding="utf-8"))
    assert sha256_lf(ROOT / "src" / "contexto_golpe.py") == cong["sha256_src_contexto_golpe"], "src/contexto_golpe.py mudou"
    assert sha256_lf(ROOT / "docs" / "avaliacao_contexto" / "prompt_r1.txt") == cong["sha256_prompt_sistema"], "prompt mudou"
    assert cg.modelos()["llm"] == cong["parametros"]["modelo_llm"], "GROQ_LLM_MODEL fora do congelado"
    df = pd.read_parquet(MANIFESTO)
    feitos = set()
    if CTX_JSONL.exists():
        feitos = {json.loads(l)["id"] for l in CTX_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()}
    for _, r in df.iterrows():
        if r.id in feitos:
            continue                                         # nao repete chamada ja feita (retomada apos queda)
        t0 = time.perf_counter()
        res = cg.analisar_audio(str(ROOT / r.caminho), idioma="pt")      # config congelada do modulo
        with open(CTX_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": r.id, "res": res}, ensure_ascii=False) + "\n")
        print(r.id, res["status"], res["risco_conteudo"], res.get("idioma"), flush=True)
        if res["status"] == "erro" and "429" in str(res.get("erro")):
            sys.exit("429 persistente: parar e relatar (nao descartar clipes).")
        dt = time.perf_counter() - t0
        if dt < INTERVALO_S:
            time.sleep(INTERVALO_S - dt)


# --------------------------------------------------------------------------------------------- resumo
CORTES_T3 = None
N_BOOT, SEED_BOOT = 2000, 42


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def met(y, pred, score=None):
    """y, pred: 1 = positivo. Devolve acc, bacc, tpr, fpr, prec, auc (nan quando indefinido)."""
    from sklearn.metrics import roc_auc_score
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    P, N = (y == 1).sum(), (y == 0).sum()
    tp, tn = ((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    tpr = tp / P if P else np.nan
    tnr = tn / N if N else np.nan
    out = {"acc": (tp + tn) / len(y), "bacc": np.nanmean([tpr, tnr]), "tpr": tpr, "fpr": 1 - tnr if N else np.nan,
           "prec": tp / (tp + fp) if (tp + fp) else np.nan}
    out["auc"] = float(roc_auc_score(y, score)) if (score is not None and P and N) else np.nan
    return out


class BootPessoa:
    """Reamostra as pessoas (com reposicao) e usa todos os clipes de cada uma. 2.000 repeticoes, seed 42."""

    def __init__(self, pessoas):
        self.pessoas = np.asarray(pessoas)
        self.ids = sorted(set(self.pessoas))
        self.linhas = {p: np.where(self.pessoas == p)[0] for p in self.ids}
        rng = np.random.default_rng(SEED_BOOT)
        self.idx = []
        for _ in range(N_BOOT):
            esc = rng.integers(0, len(self.ids), len(self.ids))
            self.idx.append(np.concatenate([self.linhas[self.ids[i]] for i in esc]))

    def ic(self, f):
        """f(idx) -> dict de numeros. Devolve {chave: (lo, hi)} (percentis 2,5 e 97,5, ignorando nan)."""
        vals = [f(i) for i in self.idx]
        return {k: tuple(np.nanpercentile([v[k] for v in vals], [2.5, 97.5])) for k in vals[0]}


def _linha_met(y, pred, score, bp, rot):
    base = met(y, pred, score)
    ic = bp.ic(lambda i: met(y[i], pred[i], score[i] if score is not None else None))
    r = dict(rot)
    for k, v in base.items():
        r[k] = round(float(v), 4) if v == v else np.nan
        if k in ic:
            r[k + "_lo"], r[k + "_hi"] = (round(float(ic[k][0]), 4), round(float(ic[k][1]), 4))
    r["n"], r["n_pos"] = int(len(y)), int(np.sum(y))
    r["acc_trivial"] = round(float(max(np.mean(y), 1 - np.mean(y))), 4)
    r["acc_wilson_lo"], r["acc_wilson_hi"] = [round(float(x), 4) for x in wilson(int(round(base["acc"] * len(y))), len(y))]
    return r


def _atalhos(d):
    """Metadados do audio ORIGINAL: duracao, RMS (dBFS), fracao de silencio, taxa original, banda (95% da energia)."""
    import soundfile as sf
    from scipy.signal import welch
    linhas = []
    for _, r in d.iterrows():
        x, sr = sf.read(str(ROOT / r.caminho), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        q, h = int(0.025 * sr), int(0.010 * sr)
        e = np.array([np.sum(x[i:i + q] ** 2) for i in range(0, max(len(x) - q, 1), h)]) + 1e-20
        sil = float(np.mean(10 * np.log10(e / e.max()) < -35))
        f, pxx = welch(x, fs=sr, nperseg=2048)
        c = np.cumsum(pxx) / pxx.sum()
        banda = float(f[np.searchsorted(c, 0.95)])
        linhas.append({"duracao_s": float(len(x) / sr), "rms_db": 20 * np.log10(rms), "frac_silencio": sil,
                       "taxa_hz": float(sr), "banda95_hz": banda})
    return pd.DataFrame(linhas)


def _lopo_auc(X, y, pessoas):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for p in sorted(set(pessoas)):
        te = pessoas == p
        sc = StandardScaler().fit(X[~te])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[~te]), y[~te])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return oof


def etapa_resumo(a):
    from difflib import SequenceMatcher
    m = pd.read_parquet(MANIFESTO)
    nd = pd.read_parquet(NOTAS_DET)
    t3 = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))["cortes_da_validacao_logit"]
    f0r = json.loads((R / "v4final_resumo.json").read_text(encoding="utf-8"))["F0"]
    cortes = {("T3", "M"): t3["M"], ("T3", "A'"): t3["A'"], ("T3", "F5"): t3["F5"],
              ("F0", "A"): f0r["limiar_A_logit"], ("F0", "B"): f0r["limiar_B_logit"]}
    d = m.merge(nd[nd.condicao == "A"][["id", "z_T3", "z_F0", "dur_usada_s", "corte_silencio"]], on="id")
    d = d.merge(nd[nd.condicao == "B"][["id", "z_T3", "z_F0"]].rename(columns={"z_T3": "z_T3_B", "z_F0": "z_F0_B"}), on="id", how="left")
    d = d.rename(columns={"z_T3": "z_T3_A", "z_F0": "z_F0_A"})
    d["z_T3_B"] = d.z_T3_B.fillna(d.z_T3_A)      # B: reais ficam como estao
    d["z_F0_B"] = d.z_F0_B.fillna(d.z_F0_A)
    y = (d.autenticidade == "ia").astype(int).values
    bp = BootPessoa(d.pessoa_id.values)

    # ---- detector
    det, grp = [], []
    for (modelo, regra), corte in cortes.items():
        for cond in ("A", "B"):
            s = d[f"z_{modelo}_{cond}"].values
            pred = (s >= corte).astype(int)
            det.append(_linha_met(y, pred, s, bp, {"modelo": modelo, "corte_regra": regra, "corte_logit": round(corte, 5), "condicao": cond}))
            for g in GRUPOS:
                k = d.grupo.values == g
                n, kk = int(k.sum()), int(pred[k].sum())
                lo, hi = wilson(kk, n)
                grp.append({"modelo": modelo, "corte_regra": regra, "condicao": cond, "grupo": g, "n": n, "acusados_como_ia": kk,
                            "taxa": round(kk / n, 4), "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)})
    pd.DataFrame(det).to_csv(R / "teste_whatsapp_detector.csv", index=False)
    pd.DataFrame(grp).to_csv(R / "teste_whatsapp_grupos.csv", index=False)

    # ---- pareado por pessoa
    par = []
    for modelo in ("T3", "F0"):
        for cond in ("A", "B"):
            col = f"z_{modelo}_{cond}"
            tab = d.groupby(["pessoa_id", "autenticidade"])[col].mean().unstack()
            dif = (tab["ia"] - tab["real"]).values
            rng = np.random.default_rng(SEED_BOOT)
            bs = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(N_BOOT)]
            par.append({"modelo": modelo, "condicao": cond, "pessoas": len(dif), "ia_maior_que_real": int((dif > 0).sum()),
                        "dif_media": round(float(dif.mean()), 3), "dif_lo": round(float(np.percentile(bs, 2.5)), 3),
                        "dif_hi": round(float(np.percentile(bs, 97.5)), 3), "dif_min": round(float(dif.min()), 3)})
    pd.DataFrame(par).to_csv(R / "teste_whatsapp_pareado.csv", index=False)

    # ---- atalho por metadados
    meta = _atalhos(d)
    pes = d.pessoa_id.values
    at = []
    for nome, cols in (("todos os metadados", list(meta.columns)), ("sem a taxa original", [c for c in meta.columns if c != "taxa_hz"]),
                       ("so duracao", ["duracao_s"]), ("so banda95", ["banda95_hz"])):
        oof = _lopo_auc(meta[cols].values, y, pes)
        base = met(y, (oof >= 0).astype(int), oof)
        ic = bp.ic(lambda i: {"auc": met(y[i], (oof[i] >= 0).astype(int), oof[i])["auc"]})
        at.append({"metadados": nome, "AUC_loo": round(base["auc"], 4), "auc_lo": round(ic["auc"][0], 4), "auc_hi": round(ic["auc"][1], 4),
                   "acc": round(base["acc"], 4), "bacc": round(base["bacc"], 4)})
    for modelo in ("T3", "F0"):
        s = d[f"z_{modelo}_A"].values
        ic = bp.ic(lambda i: {"auc": met(y[i], (s[i] >= 0).astype(int), s[i])["auc"]})
        at.append({"metadados": f"DETECTOR {modelo} (condicao A)", "AUC_loo": round(met(y, (s >= 0).astype(int), s)["auc"], 4),
                   "auc_lo": round(ic["auc"][0], 4), "auc_hi": round(ic["auc"][1], 4), "acc": np.nan, "bacc": np.nan})
    pd.DataFrame(at).to_csv(R / "teste_whatsapp_atalho.csv", index=False)
    pd.concat([d[["id", "grupo"]], meta], axis=1).drop(columns=[]).to_csv(R / "teste_whatsapp_metadados.csv", index=False)

    # ---- contexto
    reg = {}
    for l in CTX_JSONL.read_text(encoding="utf-8").splitlines():
        if l.strip():
            j = json.loads(l)
            reg[j["id"]] = j["res"]
    assert set(reg) == set(d.id), "faltam clipes no contexto"
    c = []
    for i in d.id:
        r = reg[i]
        c.append({"ctx_status": r["status"], "ctx_risco": r["risco_conteudo"], "idioma": r.get("idioma"),
                  "n_palavras": len((r.get("transcricao") or "").split()), "sinais": ";".join(sorted({s["tipo"] for s in r.get("sinais", [])})),
                  "manipulacao": bool(r.get("manipulacao_detectada")), "transc_suspeita": bool(r.get("transcricao_suspeita")),
                  "lat_transc_ms": r["latencia_ms"].get("transcricao"), "lat_class_ms": r["latencia_ms"].get("classificacao"),
                  "lat_total_ms": r["latencia_ms"].get("total"), "erro": bool(r.get("erro"))})
    d = pd.concat([d, pd.DataFrame(c)], axis=1)
    d["alerta_medio"] = d.ctx_risco.isin(["medio", "alto"]).astype(int)
    d["alerta_alto"] = (d.ctx_risco == "alto").astype(int)
    yc = (d.conteudo == "golpe").astype(int).values
    ctx = []
    for nome, col in (("medio+ (principal)", "alerta_medio"), ("so alto", "alerta_alto")):
        pr = d[col].values
        r0 = _linha_met(yc, pr, None, bp, {"alerta": nome, "recorte": "todos"})
        ctx.append(r0)
        for rec, mask in (("so reais", d.autenticidade == "real"), ("so IA", d.autenticidade == "ia")):
            k = mask.values
            ctx.append({**met(yc[k], pr[k]), "alerta": nome, "recorte": rec, "n": int(k.sum()), "n_pos": int(yc[k].sum())})
    ctx = [{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and v == v else v) for k, v in r.items()} for r in ctx]
    pd.DataFrame(ctx).to_csv(R / "teste_whatsapp_contexto.csv", index=False)
    cg = []
    for g in GRUPOS:
        k = d.grupo == g
        for nome, col in (("medio+", "alerta_medio"), ("alto", "alerta_alto")):
            kk, n = int(d.loc[k, col].sum()), int(k.sum())
            lo, hi = wilson(kk, n)
            cg.append({"grupo": g, "alerta": nome, "n": n, "com_alerta": kk, "taxa": round(kk / n, 4), "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)})
        cg[-2].update({f"risco_{r}": int((d.loc[k, 'ctx_risco'] == r).sum()) for r in ("nenhum", "baixo", "medio", "alto", "indeterminado")})
    pd.DataFrame(cg).to_csv(R / "teste_whatsapp_contexto_grupos.csv", index=False)

    # ---- idioma e sobreposicao de texto (so contagens; nenhum texto vai para results/)
    idi = d.idioma.fillna("(nenhum)").str.lower().value_counts().to_dict()
    textos = []
    for f in ("textos_dev.jsonl", "textos_teste.jsonl"):
        for l in (ROOT / "docs" / "avaliacao_contexto" / f).read_text(encoding="utf-8").splitlines():
            if l.strip():
                textos.append(json.loads(l)["texto"].lower())
    textos.append((ROOT / "docs" / "avaliacao_contexto" / "prompt_r1.txt").read_text(encoding="utf-8").lower())
    quase, maxr, maxj = 0, 0.0, 0.0
    for i in d.id:
        t = (reg[i].get("transcricao") or "").lower()
        if not t:
            continue
        ws = set(t.split())
        rm, jm = 0.0, 0.0
        for x in textos:
            jm = max(jm, len(ws & set(x.split())) / max(len(ws | set(x.split())), 1))
            sm = SequenceMatcher(None, t, x, autojunk=False)
            if sm.real_quick_ratio() > rm and sm.quick_ratio() > rm:
                rm = max(rm, sm.ratio())
        maxr, maxj = max(maxr, rm), max(maxj, jm)
        quase += int(rm >= 0.8 or jm >= 0.6)

    # ---- combinado (regra congelada: voz IA se T3_A >= M; alerta se risco >= medio)
    d["voz_ia"] = (d.z_T3_A >= cortes[("T3", "M")]).astype(int)
    d["nivel"] = np.where(d.voz_ia + d.alerta_medio == 2, "alto", np.where(d.voz_ia + d.alerta_medio == 1, "atencao", "baixo"))
    comb = pd.crosstab(d.grupo, d.nivel).reindex(index=list(GRUPOS), columns=["baixo", "atencao", "alto"], fill_value=0)
    comb["n"] = comb.sum(1)
    comb.reset_index().to_csv(R / "teste_whatsapp_combinado.csv", index=False)

    # ---- notas por clipe (com pessoa_id, sem nome, sem texto)
    cols = ["id", "pessoa_id", "grupo", "autenticidade", "conteudo", "duracao_s", "taxa_hz", "z_T3_A", "z_T3_B", "z_F0_A", "z_F0_B",
            "voz_ia", "ctx_status", "ctx_risco", "idioma", "n_palavras", "sinais", "manipulacao", "transc_suspeita",
            "lat_transc_ms", "lat_class_ms", "lat_total_ms", "alerta_medio", "alerta_alto", "nivel"]
    d[cols].round(5).to_csv(R / "teste_whatsapp_notas.csv", index=False)

    resumo = {"quando": time.strftime("%Y-%m-%d %H:%M:%S"), "cortes": {f"{k[0]}_{k[1]}": v for k, v in cortes.items()},
              "idiomas_detectados": idi, "nao_ingles": int(sum(v for k, v in idi.items() if k not in ("english", "en"))),
              "status": d.ctx_status.value_counts().to_dict(), "risco": d.ctx_risco.value_counts().to_dict(),
              "sobreposicao_texto": {"quase_identicas": int(quase), "max_razao_difflib": round(maxr, 3), "max_jaccard": round(maxj, 3),
                                     "criterio": "razao >= 0,8 ou Jaccard >= 0,6"},
              "latencia_total_ms": {"mediana": float(d.lat_total_ms.median()), "p90": float(d.lat_total_ms.quantile(.9)), "max": float(d.lat_total_ms.max())},
              "palavras": {"mediana": float(d.n_palavras.median()), "min": int(d.n_palavras.min()), "max": int(d.n_palavras.max())},
              "manipulacao_detectada": int(d.manipulacao.sum()), "transcricao_suspeita": int(d.transc_suspeita.sum()),
              "sha256_manifesto": sha256(MANIFESTO)}
    (R / "teste_whatsapp_resumo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(pd.DataFrame(det)[["modelo", "corte_regra", "condicao", "acc", "acc_lo", "acc_hi", "bacc", "tpr", "fpr", "auc", "auc_lo", "auc_hi"]].to_string(index=False))
    print(pd.DataFrame(ctx).to_string(index=False))
    print(pd.DataFrame(par).to_string(index=False))
    print(pd.DataFrame(at).to_string(index=False))
    print(comb)
    print(json.dumps(resumo, indent=1, ensure_ascii=False))


def etapa_marcar(a):
    exigir_nao_concluido()
    MARCADOR.write_text(json.dumps({"concluido_em": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256_manifesto": sha256(MANIFESTO),
                                    "aviso": "o teste WhatsApp foi avaliado uma vez; nao rodar de novo"}, indent=1), encoding="utf-8")
    print("marcador gravado")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["manifesto", "recodificar", "detector", "contexto", "resumo", "marcar"])
    a = ap.parse_args()
    for f in (sys.stdout, sys.stderr):
        f.reconfigure(errors="replace")
    os.chdir(ROOT)
    globals()["etapa_" + a.etapa](a)
