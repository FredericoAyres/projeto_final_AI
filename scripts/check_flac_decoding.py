#!/usr/bin/env python3
"""Decodifica TODOS os FLAC do manifest e conta quantos o soundfile não consegue ler.

A auditoria original (scripts/build_manifest.py) só leu os cabeçalhos com soundfile.info. No treino
do baseline apareceu o erro "unknown error in flac decoder" ao decodificar parte dos FLAC do
ASVspoof 2021 DF. Este script mede isso exatamente:
  1. tenta ler o arquivo inteiro com soundfile;
  2. se falhar, tenta com miniaudio (o plano B usado em src/dataset.py);
  3. em 300 arquivos que os dois leem, confere se as amostras são iguais.

Uso (na raiz do repo, com o env ativo):
    python scripts/check_flac_decoding.py
Saída: tabela no terminal e data/processed/flac_decode_check.parquet (só as falhas).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import miniaudio
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/processed/manifest.parquet"
OUT = ROOT / "data/processed/flac_decode_check.parquet"


def check(path):
    p = str(ROOT / path)
    try:
        sf.read(p, dtype="float32")
        return path, True, None, None
    except Exception as e:  # noqa: BLE001 - queremos registrar qualquer falha
        err = f"{type(e).__name__}: {e}"[:200]
    try:
        miniaudio.flac_read_file_f32(p)
        return path, False, err, True
    except Exception:  # noqa: BLE001
        return path, False, err, False


def same_samples(path):
    p = str(ROOT / path)
    a, _ = sf.read(p, dtype="float32", always_2d=True)
    d = miniaudio.flac_read_file_f32(p)
    b = np.asarray(d.samples, dtype=np.float32).reshape(-1, d.nchannels)
    return a.shape == b.shape and np.allclose(a, b, atol=1e-6)


def main():
    m = pd.read_parquet(MANIFEST, columns=["path", "source_dataset", "split"])
    m = m[m["path"].str.lower().str.endswith(".flac")].reset_index(drop=True)
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(check, m["path"], chunksize=512))
    r = pd.DataFrame(res, columns=["path", "soundfile_ok", "erro_soundfile", "miniaudio_ok"])
    m = m.merge(r, on="path")
    t = m.groupby("source_dataset").agg(
        flac=("path", "size"),
        falha_soundfile=("soundfile_ok", lambda s: int((~s).sum())),
        lidos_miniaudio=("miniaudio_ok", lambda s: int(s.fillna(False).astype(bool).sum())),
    )
    t["% falha"] = (100 * t["falha_soundfile"] / t["flac"]).round(2)
    t["ilegiveis_nos_dois"] = t["falha_soundfile"] - t["lidos_miniaudio"]
    print(t.to_string())
    falhas = m[~m["soundfile_ok"]]
    print("\nerros do soundfile:", falhas["erro_soundfile"].value_counts().head(5).to_dict())
    falhas.to_parquet(OUT, index=False)

    ok = m[m["soundfile_ok"] & m["source_dataset"].eq("asvspoof2021_df")].sample(300, random_state=42)
    with ProcessPoolExecutor(12) as ex:
        iguais = list(ex.map(same_samples, ok["path"]))
    print(f"\nasvspoof2021_df: amostras iguais (soundfile x miniaudio) em {sum(iguais)} de {len(iguais)} "
          f"arquivos que os dois leem")
    print(f"falhas salvas em {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
