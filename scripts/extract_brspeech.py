#!/usr/bin/env python3
"""Extrai os audios do BRSpeech-DF (embutidos nos .parquet) para data/processed/brspeech_df/test/.

Uso (na raiz do repo, com o env ativo):
    python scripts/extract_brspeech.py

Cada audio vira <nome original da coluna audio.path>.flac. Arquivos ja extraidos sao pulados.
Nao altera data/raw/ nem o manifest.
"""
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/test_ptbr/brspeech_df/data"
DST = ROOT / "data/processed/brspeech_df/test"


def extract_brspeech():
    """Extrai o que faltar e devolve [(path relativo a raiz, label, shard)] de todos os audios."""
    shards = sorted(SRC.glob("test-*.parquet"))
    if not shards:
        raise SystemExit(f"nenhum parquet em {SRC.relative_to(ROOT)}; rode antes o download do brspeech_df")
    DST.mkdir(parents=True, exist_ok=True)
    out, written = [], 0
    for f in shards:
        t = pq.read_table(f, columns=["audio.path", "label"])      # so nomes e rotulos (rapido)
        names, labels = t.column("path").to_pylist(), t.column("label").to_pylist()
        if not all((DST / n).exists() for n in names):              # le os bytes so se faltar algum arquivo
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                audio = pf.read_row_group(i, columns=["audio"]).column("audio").combine_chunks()
                for name, blob in zip(audio.field("path").to_pylist(), audio.field("bytes")):
                    if not (DST / name).exists():
                        (DST / name).write_bytes(blob.as_py())
                        written += 1
        out += [((DST / n).relative_to(ROOT).as_posix(), lab, f.name) for n, lab in zip(names, labels)]
    print(f"brspeech_df: {len(out)} audios em {DST.relative_to(ROOT).as_posix()} ({written} extraidos agora)")
    return out


if __name__ == "__main__":
    extract_brspeech()
