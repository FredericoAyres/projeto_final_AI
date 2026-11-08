#!/usr/bin/env python3
"""Confere se todos os audios do manifest existem em disco (so existencia, nao le os audios).

Uso (na raiz do repo, com o env ativo):
    python scripts/check_data.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/processed/manifest.parquet"

# o que rodar quando faltar audio de cada banco
FIX = {"brspeech_df": "python scripts/extract_brspeech.py  (depois do download do brspeech_df)"}


def main():
    if not MANIFEST.exists():
        sys.exit(f"FALTA {MANIFEST.relative_to(ROOT).as_posix()} (vem no repositorio: confira o git pull)")
    m = pd.read_parquet(MANIFEST, columns=["path", "source_dataset"])
    m["existe"] = [(ROOT / p).is_file() for p in m["path"]]
    t = m.groupby("source_dataset")["existe"].agg(existem="sum", faltam=lambda s: int((~s).sum()))
    print(t.to_string())
    faltando = t.index[t["faltam"] > 0].tolist()
    if not faltando:
        print(f"\nOK: os {len(m)} audios do manifest existem em disco.")
        return
    print(f"\nFALTAM {int(t['faltam'].sum())} audios. Rode, na raiz do repo:")
    baixar = [d for d in faltando if d not in FIX]
    if baixar:
        print(f"  python scripts/download_all.py --only {','.join(baixar)} --max-gb 100")
    for d in faltando:
        if d in FIX:
            print(f"  {FIX[d]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
