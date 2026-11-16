#!/usr/bin/env python3
"""Reavaliacao do modulo de contexto com o idioma corrigido (protocolo: docs/PROTOCOLO_CONTEXTO_WHISPER.md).

Etapas (nesta ordem):
    dev_whisper [--modelo M]   Whisper so no DEV, modos auto e en (pt reaproveita 15/11) -> data/processed/ctxw_dev_*.jsonl
    escolher                   WER/idioma por modo, aplica a REGRA FIXA do protocolo, grava o congelamento
    reservado                  pipeline completo no RESERVADO, UMA vez (marcador impede repetir)
    dev_pipeline               pipeline completo no DEV com o modo congelado ("usado no diagnostico")
    resumo                     metricas agregadas -> results/diagnostico_whisper_idioma_*.csv

Transcricoes ficam so em data/. NUNCA le nem imprime transcricao do RESERVADO. Nunca usa traducao do Whisper.
Nao altera src/contexto_golpe.py, o prompt, o LLM nem os cortes.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R, DP = ROOT / "results", ROOT / "data" / "processed"
DIV = R / "diagnostico_whisper_divisao.csv"
MAN = DP / "manifest_whatsapp.parquet"
PROMPT = ROOT / "docs" / "avaliacao_contexto" / "prompt_r1.txt"
SRC = ROOT / "src" / "contexto_golpe.py"
PROTOCOLO = ROOT / "docs" / "PROTOCOLO_CONTEXTO_WHISPER.md"
CONGELADO = R / "diagnostico_whisper_congelado.json"
MARCADOR = R / "diagnostico_whisper_reservado_concluido.json"
GERAL = ROOT / "data/teste_whatsapp/extraido/coleta/manifesto_geral.csv"
OLD_CTX = DP / "tw_contexto.jsonl"
OLD_NOTAS = R / "teste_whatsapp_notas.csv"
INTERVALO_S = 12          # pipeline completo (limite de tokens por minuto do LLM)
INTERVALO_ASR_S = 4       # so Whisper
TURBO, V3 = "whisper-large-v3-turbo", "whisper-large-v3"
SHA_ESPERADOS = {"divisao": "77d1b1b8cb54113a8eb4936c75fd366b3c01d4d15dd9a6775c5c2e189ea18a9e",
                 "prompt_lf": "3f848dbf65e294475e86ee963ab09103f7b83853bec979f14fbad872bf6de503",
                 "src_lf": "d0327e1b3c177aba0b25d307199ea6c4900c7ce3d17e09254382780e15d81dfb"}


def sha(p, lf=False):
    b = Path(p).read_bytes()
    if lf:
        b = b.replace(bytes([13, 10]), bytes([10]))
    return hashlib.sha256(b).hexdigest()


def conferir_arquivos():
    assert sha(DIV) == SHA_ESPERADOS["divisao"], "divisao mudou"
    assert sha(PROMPT, True) == SHA_ESPERADOS["prompt_lf"], "prompt mudou"
    assert sha(SRC, True) == SHA_ESPERADOS["src_lf"], "src/contexto_golpe.py mudou"
    assert PROTOCOLO.exists(), "protocolo precisa existir antes de qualquer chamada"


def exigir_chave():
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY nao esta no ambiente: parar e avisar.")


# --------------------------------------------------------------------------------------------- dados
def tabela():
    """Manifesto com a divisao DEV/RESERVADO e o texto verdadeiro dos clones (so em memoria)."""
    m = pd.read_parquet(MAN).merge(pd.read_csv(DIV).rename(columns={"grupo": "divisao"}), on="pessoa_id")
    g = pd.read_csv(GERAL, encoding="utf-8-sig", usecols=["arquivo", "texto"])
    g["k"] = g.arquivo.astype(str).map(lambda a: Path(a).stem)
    m["k"] = m.caminho.map(lambda a: Path(a).stem)
    return m.merge(g[["k", "texto"]], on="k", how="left")


def _lin(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()] if Path(p).exists() else []


def _p(t):
    import re
    return re.sub(r"[^\w\s]", " ", str(t).lower()).split()


def wer(ref, hip):
    r, h = _p(ref), _p(hip)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


# --------------------------------------------------------------------------------------------- DEV: so Whisper
def _asr_um(cg, caminho, idioma):
    prep = cg.preparar_audio(str(ROOT / caminho))
    if prep["sem_fala"]:
        return {"texto": "", "idioma": None, "suspeita": True, "sem_fala": True}
    tr = cg.transcrever(prep["bytes"], prep["nome"], prep["mime"], idioma)
    return {"texto": tr["texto"], "idioma": tr["idioma"], "suspeita": tr["suspeita"], "sem_fala": False}


def etapa_dev_whisper(a):
    conferir_arquivos()
    exigir_chave()
    if a.modelo:
        os.environ["GROQ_ASR_MODEL"] = a.modelo
    from src import contexto_golpe as cg
    modelo = cg.modelos()["asr"]
    t = tabela()
    dev = t[t.divisao == "DEV"]
    assert len(dev) == 40 and dev.pessoa_id.nunique() == 8
    modos = [a.modo] if a.modo else ["auto", "en"]
    for modo in modos:
        arq = DP / f"ctxw_dev_{modelo}_{modo}.jsonl"
        feitos = {j["id"] for j in _lin(arq)}
        for r in dev.itertuples():
            if r.id in feitos:
                continue
            t0 = time.perf_counter()
            res = _asr_um(cg, r.caminho, modo)
            with open(arq, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": r.id, **res}, ensure_ascii=False) + "\n")
            print(modelo, modo, r.id, res["idioma"], flush=True)
            time.sleep(max(0, INTERVALO_ASR_S - (time.perf_counter() - t0)))


# --------------------------------------------------------------------------------------------- escolher
def _carregar_dev(modelo, modo):
    return {j["id"]: j for j in _lin(DP / f"ctxw_dev_{modelo}_{modo}.jsonl")}


def _carregar_dev_pt(ids):
    """15/11: Whisper forcado a pt (so DEV; filtra ANTES de usar qualquer campo)."""
    out = {}
    for l in OLD_CTX.read_text(encoding="utf-8").splitlines():
        j = json.loads(l)
        if j["id"] in ids:
            out[j["id"]] = {"texto": j["res"]["transcricao"], "idioma": j["res"]["idioma"]}
    return out


def _norm_txt(s):
    return " ".join(_p(s))


def etapa_escolher(a):
    conferir_arquivos()
    if CONGELADO.exists():
        sys.exit("congelamento ja existe: nao reescolher")
    t = tabela()
    dev = t[t.divisao == "DEV"].copy()
    ids = set(dev.id)
    clones = dev[dev.autenticidade == "ia"]
    modos = {"auto": _carregar_dev(TURBO, "auto"), "en": _carregar_dev(TURBO, "en"), "pt": _carregar_dev_pt(ids)}
    for k, v in modos.items():
        assert set(v) == ids, f"modo {k}: faltam clipes"
    linhas, det = [], []
    for modo, d in modos.items():
        w = np.array([wer(r.texto, d[r.id]["texto"]) for r in clones.itertuples()])
        idi = pd.Series([str(d[i]["idioma"]).lower() for i in dev.id])
        linhas.append({"modelo": TURBO, "modo": modo, "n_clones": len(w), "wer_medio": round(w.mean(), 4), "wer_mediano": round(float(np.median(w)), 4),
                       "wer_min": round(w.min(), 4), "wer_max": round(w.max(), 4),
                       "english_dev": int((idi == "english").sum()), "n_dev": len(idi), "frac_english": round(float((idi == "english").mean()), 4),
                       "english_clones": int((idi[dev.autenticidade.values == "ia"] == "english").sum()),
                       "english_reais": int((idi[dev.autenticidade.values == "real"] == "english").sum())})
        det.append(pd.DataFrame({"id": dev.id.values, "grupo": dev.grupo.values, "modo": modo, "idioma": idi.values,
                                 "wer": [wer(r.texto, d[r.id]["texto"]) if r.autenticidade == "ia" else np.nan for r in dev.itertuples()]}))
    tab = pd.DataFrame(linhas)
    # concordancia auto x en
    ig = sum(_norm_txt(modos["auto"][i]["texto"]) == _norm_txt(modos["en"][i]["texto"]) for i in ids)
    wae = np.mean([wer(modos["en"][i]["texto"], modos["auto"][i]["texto"]) for i in ids])
    conc = {"textos_identicos_auto_en": int(ig), "n": len(ids), "wer_medio_auto_vs_en": round(float(wae), 4)}
    pd.concat(det).to_csv(R / "diagnostico_whisper_idioma_dev_clipes.csv", index=False)

    def regra(tabela_modos):
        au, en = tabela_modos.set_index("modo").loc["auto"], tabela_modos.set_index("modo").loc["en"]
        a_ok = au.frac_english >= 0.90
        b_ok = au.wer_medio <= en.wer_medio + 0.05
        return ("auto" if (a_ok and b_ok) else "en"), bool(a_ok), bool(b_ok)
    modo, a_ok, b_ok = regra(tab)
    modelo = TURBO
    variante = None
    med = float(tab.set_index("modo").loc[modo].wer_mediano)
    if med > 0.15:
        v3 = _carregar_dev(V3, modo)
        if len(v3) == 40 and set(v3) == ids:
            w3 = np.mean([wer(r.texto, v3[r.id]["texto"]) for r in clones.itertuples()])
            w3med = float(np.median([wer(r.texto, v3[r.id]["texto"]) for r in clones.itertuples()]))
            wt = float(tab.set_index("modo").loc[modo].wer_medio)
            variante = {"modelo": V3, "modo": modo, "wer_medio": round(float(w3), 4), "wer_mediano": round(w3med, 4), "wer_medio_turbo": wt}
            if w3 < wt - 0.01:
                modelo = V3
            tab = pd.concat([tab, pd.DataFrame([{"modelo": V3, "modo": modo, "n_clones": len(clones), "wer_medio": round(float(w3), 4), "wer_mediano": round(w3med, 4)}])])
        else:
            tab.to_csv(R / "diagnostico_whisper_idioma_dev_wer.csv", index=False)
            sys.exit(f"WER mediano {med:.3f} > 0,15 no modo {modo}: rode `dev_whisper --modelo {V3} --modo {modo}` (UMA variante) e depois `escolher` de novo")
    tab.to_csv(R / "diagnostico_whisper_idioma_dev_wer.csv", index=False)
    cong = {"congelado_em": time.strftime("%Y-%m-%d %H:%M:%S (UTC-0300)"), "modo": modo, "modelo_asr": modelo, "idioma_enviado": None if modo == "auto" else modo,
            "regra": {"a_english_>=_90pct": a_ok, "b_wer_auto_<=_wer_en_+0.05": b_ok, "wer_mediano_do_modo": med, "variante_v3": variante},
            "somente_ingles": modo == "en", "concordancia_auto_en": conc,
            "sha256_divisao": sha(DIV), "sha256_prompt_r1_lf": sha(PROMPT, True), "sha256_src_contexto_golpe_lf": sha(SRC, True),
            "sha256_manifesto": sha(MAN), "intervalo_s": INTERVALO_S}
    CONGELADO.write_text(json.dumps(cong, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False))
    print(conc)
    print(json.dumps({k: cong[k] for k in ("modo", "modelo_asr", "regra", "somente_ingles")}, indent=1))


# --------------------------------------------------------------------------------------------- pipeline completo
def _pipeline(ids_df, arq, tag):
    exigir_chave()
    cong = json.loads(CONGELADO.read_text(encoding="utf-8"))
    conferir_arquivos()
    for k, v in (("sha256_divisao", sha(DIV)), ("sha256_prompt_r1_lf", sha(PROMPT, True)), ("sha256_src_contexto_golpe_lf", sha(SRC, True))):
        assert cong[k] == v, f"{k} mudou desde o congelamento"
    os.environ["GROQ_ASR_MODEL"] = cong["modelo_asr"]
    from src import contexto_golpe as cg
    cgel = json.loads((R / "contexto_golpe_congelado.json").read_text(encoding="utf-8"))
    assert cg.modelos()["llm"] == cgel["parametros"]["modelo_llm"]
    assert cg.modelos()["asr"] == cong["modelo_asr"]
    feitos = {j["id"] for j in _lin(arq)}
    for r in ids_df.itertuples():
        if r.id in feitos:
            continue
        t0 = time.perf_counter()
        res = cg.analisar_audio(str(ROOT / r.caminho), idioma=cong["modo"])
        with open(arq, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": r.id, "res": res}, ensure_ascii=False) + "\n")
        print(tag, r.id, res["status"], flush=True)           # so o status; nunca texto
        if res["status"] == "erro" and "429" in str(res.get("erro")):
            sys.exit("429 persistente: parar e relatar (nao descartar clipes; rodar de novo retoma).")
        time.sleep(max(0, INTERVALO_S - (time.perf_counter() - t0)))


def etapa_reservado(a):
    if MARCADOR.exists():
        sys.exit("RESERVADO ja foi avaliado uma vez; nao roda de novo.")
    if not CONGELADO.exists():
        sys.exit("congele a escolha (etapa escolher) antes de tocar no RESERVADO.")
    t = tabela()
    res = t[t.divisao == "RESERVADO"]
    assert len(res) == 60 and res.pessoa_id.nunique() == 15
    arq = DP / "ctxw_reservado.jsonl"
    _pipeline(res, arq, "RES")
    assert {j["id"] for j in _lin(arq)} == set(res.id)
    MARCADOR.write_text(json.dumps({"concluido_em": time.strftime("%Y-%m-%d %H:%M:%S (UTC-0300)"), "n_clipes": 60,
                                    "sha256_congelado": sha(CONGELADO),
                                    "aviso": "o RESERVADO foi avaliado uma vez; nao rodar de novo"}, indent=1), encoding="utf-8")
    print("RESERVADO concluido; marcador gravado")


def etapa_dev_pipeline(a):
    if not MARCADOR.exists():
        sys.exit("so depois do RESERVADO (marcador ausente).")
    t = tabela()
    _pipeline(t[t.divisao == "DEV"], DP / "ctxw_dev_pipeline.jsonl", "DEV")


# --------------------------------------------------------------------------------------------- resumo
def etapa_resumo(a):
    import teste_whatsapp as tw
    if not MARCADOR.exists():
        sys.exit("RESERVADO ainda nao concluido.")
    cong = json.loads(CONGELADO.read_text(encoding="utf-8"))
    t = tabela()
    t["yc"] = (t.conteudo == "golpe").astype(int)
    velho = pd.read_csv(OLD_NOTAS, usecols=["id", "ctx_risco", "ctx_status"]).rename(columns={"ctx_risco": "risco_old", "ctx_status": "status_old"})
    t = t.merge(velho, on="id")
    out_m, out_g, out_idi = [], [], []
    for conj, arq in (("RESERVADO (principal)", DP / "ctxw_reservado.jsonl"), ("DEV (usado no diagnostico)", DP / "ctxw_dev_pipeline.jsonl")):
        d = t[t.divisao == ("RESERVADO" if conj.startswith("RES") else "DEV")].copy()
        reg = {j["id"]: j["res"] for j in _lin(arq)}
        assert set(reg) == set(d.id), f"faltam clipes em {conj}"
        d["risco_new"] = d.id.map(lambda i: reg[i]["risco_conteudo"])
        d["status_new"] = d.id.map(lambda i: reg[i]["status"])
        d["idioma_new"] = d.id.map(lambda i: str(reg[i].get("idioma")).lower())
        bp = tw.BootPessoa(d.pessoa_id.values)
        y = d.yc.values
        for versao, col in (("15/11 (pt forcado)", "risco_old"), ("corrigido", "risco_new")):
            pr = d[col].isin(["medio", "alto"]).astype(int).values
            base = {"conjunto": conj, "versao": versao}
            out_m.append(tw._linha_met(y, pr, None, bp, {**base, "recorte": "todos"}))
            for rec, k in (("so reais", (d.autenticidade == "real").values), ("so IA", (d.autenticidade == "ia").values)):
                out_m.append({**tw.met(y[k], pr[k]), **base, "recorte": rec, "n": int(k.sum()), "n_pos": int(y[k].sum())})
            for g in tw.GRUPOS:
                k = (d.grupo == g).values
                kk, n = int(pr[k].sum()), int(k.sum())
                lo, hi = tw.wilson(kk, n)
                out_g.append({**base, "grupo": g, "n": n, "com_alerta": kk, "taxa": round(kk / n, 4), "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                              **{f"risco_{r}": int((d.loc[k, col] == r).sum()) for r in ("nenhum", "baixo", "medio", "alto", "indeterminado")}})
        sv = d.status_new.value_counts().to_dict()
        out_idi.append({"conjunto": conj, "n": len(d), "english": int((d.idioma_new == "english").sum()),
                        "outros": {k: int(v) for k, v in d.idioma_new[d.idioma_new != "english"].value_counts().items()},
                        "audio_sem_fala": int(sv.get("audio_sem_fala", 0)), "erro": int(sv.get("erro", 0)),
                        "indeterminado": int((d.risco_new == "indeterminado").sum())})
    pd.DataFrame(out_m).round(4).to_csv(R / "diagnostico_whisper_idioma_metricas.csv", index=False)
    pd.DataFrame(out_g).to_csv(R / "diagnostico_whisper_idioma_grupos.csv", index=False)
    pd.DataFrame(out_idi).to_csv(R / "diagnostico_whisper_idioma_idioma.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(pd.DataFrame(out_m).query("recorte == 'todos'")[["conjunto", "versao", "acc", "acc_lo", "acc_hi", "acc_trivial", "tpr", "tpr_lo", "tpr_hi", "fpr", "fpr_lo", "fpr_hi", "prec", "bacc"]].to_string(index=False))
    print(pd.DataFrame(out_g).iloc[:, :8].to_string(index=False))
    print(pd.DataFrame(out_idi).to_string(index=False))
    print(json.dumps({k: cong[k] for k in ("modo", "modelo_asr")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["dev_whisper", "escolher", "reservado", "dev_pipeline", "resumo"])
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--modo", default=None, choices=["auto", "en"])
    a = ap.parse_args()
    for f in (sys.stdout, sys.stderr):
        f.reconfigure(errors="replace")
    os.chdir(ROOT)
    globals()["etapa_" + a.etapa](a)
