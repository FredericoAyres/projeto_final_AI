#!/usr/bin/env python3
"""Gera o manifesto dos bancos de TESTE da internet (RADAR eval + ML-ITW), uma linha por audio.

Uso (na raiz do repo, com o env ativo):
    python scripts/build_manifest_teste_internet.py --raiz data/raw/teste_internet
    python scripts/build_manifest_teste_internet.py --raiz D:\\dados\\teste_internet \\
        --out data/processed/manifest_teste_internet.parquet

O que faz:
  1. RADAR eval: le label_RADAR2026-eval.txt e lang_RADAR2026-eval.txt; os audios ficam em
     <raiz>/radar_eval/RADAR2026-eval/flac/<id>.flac.
  2. ML-ITW: extrai os audios dos .parquet para <raiz>/ml_itw/audio/ (bytes copiados SEM recodificar,
     para preservar codec/extensao originais; pula o que ja existe) e le speaker/language/platform/label.
  3. Le o cabecalho de cada audio (soundfile.info, em paralelo); arquivo ilegivel ou ausente NAO e
     omitido: fica com readable=False e o erro na coluna error.
  4. Compara as contagens com o esperado e IMPRIME as diferencas (nada e corrigido em silencio).

Nao altera data/processed/manifest.parquet. Caminhos do manifesto sao RELATIVOS a --raiz, com '/'.

Colunas: as do manifesto atual (path, label int8 0=real/1=fake, source_dataset, generator, fake_type,
speaker_id, language, original_split, role, split, duration_s, sample_rate Int32, eval_subset) mais
fonte, label_nome (real|fake), plataforma, canais, codec (formato/subtipo do soundfile), extensao,
tamanho_bytes, readable, error. sr_original = sample_rate (do arquivo, sem reamostrar).
role="test_internet", split="test", eval_subset="test_internet" (nunca entra em treino nem dev_shift).
"""
import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "processed" / "manifest_teste_internet.parquet"

RADAR_SUB = "radar_eval"
RADAR_FLAC = "radar_eval/RADAR2026-eval/flac"
MLITW_SUB = "ml_itw"
LANG_RADAR = {"EN": "en", "ENSG": "en-sg", "ZH": "zh", "ZHTW": "zh-tw", "JA": "ja", "VI": "vi"}

# esperado (enunciado da tarefa). O RADAR publicado tem 1 audio a menos (EVAL068818, corrompido,
# excluido pelos autores: ver README do Zenodo), entao as contagens abaixo sao as do ARTIGO.
ESPERADO_RADAR = {"total": 102726, "real": 50000, "fake": 52726,
                  "por_idioma": {"en": (15000, 17923), "en-sg": (3500, 6711), "zh": (11500, 11234),
                                 "zh-tw": (3500, 1678), "ja": (9500, 8742), "vi": (7000, 6438)}}
ESPERADO_MLITW = {"total": 24529, "real": 18168, "fake": 6361, "locutores": 180, "idiomas": 14, "horas": 28.39}

COLS = ["path", "label", "label_nome", "fonte", "source_dataset", "generator", "fake_type", "speaker_id",
        "language", "plataforma", "original_split", "role", "split", "eval_subset", "duration_s", "sample_rate",
        "canais", "codec", "extensao", "tamanho_bytes", "readable", "error"]


# --------------------------------------------------------------------------- RADAR
def radar(raiz):
    lab_f, lang_f = raiz / RADAR_SUB / "label_RADAR2026-eval.txt", raiz / RADAR_SUB / "lang_RADAR2026-eval.txt"
    labs = {}
    for ln in lab_f.read_text(encoding="utf-8").splitlines():
        c = ln.split()
        if c:                          # <id> <duracao> <bonafide|spoof> <ini-fim-rotulo> ...
            labs[c[0]] = c[2]
    langs = {}
    for ln in lang_f.read_text(encoding="utf-8").splitlines():
        c = ln.split()
        if c:
            langs[c[0]] = c[1]
    if set(labs) != set(langs):
        print(f"AVISO RADAR: ids so no label: {len(set(labs) - set(langs))}; so no lang: {len(set(langs) - set(labs))}")
    bad = {v for v in labs.values() if v not in ("bonafide", "spoof")}
    if bad:
        print(f"AVISO RADAR: rotulos inesperados: {bad}")
    rows = []
    for uid in sorted(labs):
        spoof = labs[uid] == "spoof"
        code = langs.get(uid, "")
        rows.append(dict(path=f"{RADAR_FLAC}/{uid}.flac", label=int(spoof), fonte="radar_eval",
                         generator=None, fake_type="full" if spoof else "none", speaker_id=None,
                         language=LANG_RADAR.get(code, code or None), plataforma=None,
                         original_split="eval"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ML-ITW
def mlitw(raiz):
    import pyarrow.parquet as pq
    base = raiz / MLITW_SUB
    shards = sorted((base / "data").glob("eval-*.parquet"))
    if not shards:
        raise SystemExit(f"nenhum parquet em {base / 'data'}")
    (base / "audio").mkdir(exist_ok=True)
    rows, k = [], 0
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for batch in tqdm(pf.iter_batches(batch_size=64), total=(pf.metadata.num_rows + 63) // 64,
                          desc=f"extrair {sh.name}", mininterval=10):
            d = batch.to_pydict()
            for a, spk, lang, plat, lab in zip(d["file"], d["speaker"], d["language"], d["platform"], d["label"]):
                orig = (a.get("path") or "")
                ext = Path(orig).suffix.lower() or ".bin"
                name = f"{k:06d}{ext}"           # nome unico; o nome original fica so como referencia
                dest = base / "audio" / name
                data = a["bytes"]
                if not dest.exists() or dest.stat().st_size != len(data):
                    dest.write_bytes(data)
                rows.append(dict(path=f"{MLITW_SUB}/audio/{name}", label_txt=lab, fonte="ml_itw",
                                 speaker_id=f"mlitw:{spk}" if spk not in (None, "") else None,
                                 language=lang, plataforma=plat, original_split=sh.stem, nome_original=orig))
                k += 1
    df = pd.DataFrame(rows)
    print("ML-ITW rotulos brutos:", df["label_txt"].value_counts(dropna=False).to_dict())
    low = df["label_txt"].astype(str).str.lower()
    fake = low.isin(["spoof", "fake", "deepfake", "1", "ai", "synthetic"])
    real = low.isin(["bonafide", "bona-fide", "real", "0", "genuine"])
    if (~(fake | real)).any():
        raise SystemExit(f"ML-ITW: rotulos nao reconhecidos: {sorted(set(df.loc[~(fake | real), 'label_txt']))}")
    df["label"] = fake.astype(int)
    df["generator"] = None
    df["fake_type"] = np.where(fake, "full", "none")
    return df.drop(columns=["label_txt"])


# --------------------------------------------------------------------------- cabecalhos
def info(args):
    raiz, rel = args
    p = raiz / rel
    try:
        size = p.stat().st_size
    except OSError as e:
        return rel, False, f"{type(e).__name__}: arquivo ausente", np.nan, np.nan, np.nan, None, None, np.nan
    try:
        i = sf.info(str(p))
        return rel, True, None, i.frames / i.samplerate if i.samplerate else np.nan, i.samplerate, i.channels, \
            i.format, i.subtype, size
    except Exception as e:
        return rel, False, f"{type(e).__name__}: {e}"[:200], np.nan, np.nan, np.nan, None, None, size


def headers(raiz, paths, workers):
    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(info, [(raiz, p) for p in paths], chunksize=256), total=len(paths),
                        desc="soundfile.info", mininterval=5))
    return pd.DataFrame(res, columns=["path", "readable", "error", "duration_s", "sample_rate", "canais",
                                      "fmt", "subtype", "tamanho_bytes"])


# --------------------------------------------------------------------------- conferencia
def conferir(df):
    print("\n=== CONFERENCIA COM O ESPERADO (diferencas sao apenas relatadas) ===")
    r = df[df["fonte"] == "radar_eval"]
    E = ESPERADO_RADAR
    nr, nf = int((r.label == 0).sum()), int((r.label == 1).sum())
    print(f"RADAR eval: total {len(r)} (esperado {E['total']}, dif {len(r) - E['total']:+d}); "
          f"reais {nr} (esp. {E['real']}, dif {nr - E['real']:+d}); falsos {nf} (esp. {E['fake']}, dif {nf - E['fake']:+d})")
    for lg, (er, ef) in E["por_idioma"].items():
        s = r[r.language == lg]
        a, b = int((s.label == 0).sum()), int((s.label == 1).sum())
        print(f"  {lg:6s} real {a:6d} (esp. {er}, dif {a - er:+d}) | falso {b:6d} (esp. {ef}, dif {b - ef:+d})"
              + ("" if (a, b) == (er, ef) else "   <-- DIFERE"))
    extra = set(r.language.dropna()) - set(E["por_idioma"])
    if extra:
        print(f"  idiomas inesperados no RADAR: {extra}")
    m = df[df["fonte"] == "ml_itw"]
    E = ESPERADO_MLITW
    nr, nf = int((m.label == 0).sum()), int((m.label == 1).sum())
    horas = m["duration_s"].sum() / 3600
    nl = m["speaker_id"].nunique()
    ni = m["language"].nunique()
    print(f"ML-ITW: total {len(m)} (esp. {E['total']}, dif {len(m) - E['total']:+d}); reais {nr} (esp. {E['real']}, "
          f"dif {nr - E['real']:+d}); falsos {nf} (esp. {E['fake']}, dif {nf - E['fake']:+d})")
    print(f"  locutores {nl} (esp. {E['locutores']}); idiomas {ni} (esp. {E['idiomas']}); "
          f"horas {horas:.2f} (esp. {E['horas']})")
    print(f"Ilegiveis/ausentes: RADAR {int((~r.readable).sum())}, ML-ITW {int((~m.readable).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, required=True, help="pasta passada como --dest no download")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    raiz = args.raiz.resolve()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for s in (RADAR_SUB, MLITW_SUB):
        if not (raiz / s / ".done").exists():
            print(f"AVISO: {raiz / s} sem marcador .done (download incompleto?)")

    df = pd.concat([radar(raiz), mlitw(raiz)], ignore_index=True)
    assert not df["path"].duplicated().any(), "path duplicado"
    inf = headers(raiz, df["path"].tolist(), args.workers)
    df = df.merge(inf, on="path", how="left")
    df["codec"] = np.where(df["fmt"].notna(), df["fmt"].astype(str) + "/" + df["subtype"].astype(str), None)
    df["extensao"] = df["path"].map(lambda p: Path(p).suffix.lower().lstrip("."))
    df["label_nome"] = np.where(df["label"] == 1, "fake", "real")
    df["source_dataset"] = df["fonte"]
    df["role"], df["split"], df["eval_subset"] = "test_internet", "test", "test_internet"
    df["label"] = df["label"].astype("int8")
    df["sample_rate"] = df["sample_rate"].astype("Int32")
    df["canais"] = df["canais"].astype("Int8")
    df["tamanho_bytes"] = df["tamanho_bytes"].astype("Int64")
    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_parquet(out, index=False)
    print(f"\n{out}: {len(df)} linhas, {out.stat().st_size / 1e6:.1f} MB")
    conferir(df)
    if (~df["readable"]).any():
        print("\nArquivos ilegiveis/ausentes (primeiros 10):")
        print(df.loc[~df["readable"], ["path", "error"]].head(10).to_string())


if __name__ == "__main__":
    main()
