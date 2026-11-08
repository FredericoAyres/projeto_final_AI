#!/usr/bin/env python3
"""Imprime (em markdown) as tabelas de auditoria/verificacao usadas em docs/audit_report.md.

Uso: python scripts/audit_checks.py > saida.md   (depois de scripts/build_manifest.py)
"""
import os
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
DIRS = {  # source_dataset -> pasta(s) com os audios
    "asvspoof2019_la": ["data/raw/train_val_en/asvspoof2019_la"],
    "dfadd": ["data/raw/train_val_en/dfadd"],
    "shiftyspeech": ["data/raw/train_val_en/shiftyspeech"],
    "codecfake_plus": ["data/raw/train_val_en/codecfake_plus"],
    "librispeech": ["data/raw/real_en/librispeech"],
    "in_the_wild": ["data/raw/test_generalization/in_the_wild"],
    "llamapartialspoof": ["data/raw/test_generalization/llamapartialspoof"],
    "asvspoof2021_df": ["data/raw/test_generalization/asvspoof2021_df"],
    "brspeech_df": ["data/processed/brspeech_df"],
    "coraa": ["data/raw/test_ptbr/coraa"],
}


def md(df, index=True):
    return df.to_markdown(index=index) + "\n"


def main():
    m = pd.read_parquet(ROOT / "data/processed/manifest.parquet")
    info = pd.read_parquet(ROOT / "data/processed/audio_info.parquet")
    mi = m.merge(info[["path", "readable", "error", "channels", "format", "subtype"]], on="path", how="left")
    order = list(DIRS)

    print("## A. Arquivos de audio em disco x manifest\n")
    rows, missing = [], {}
    in_manifest = set(m["path"])
    for ds, dirs in DIRS.items():
        ext, on_disk = Counter(), set()
        for d in dirs:
            for r, _, fs in os.walk(ROOT / d):
                for f in fs:
                    e = os.path.splitext(f)[1].lower()
                    if e in AUDIO_EXT:
                        ext[e] += 1
                        on_disk.add((Path(r) / f).relative_to(ROOT).as_posix())
        miss = sorted(on_disk - in_manifest)
        missing[ds] = miss
        rows.append(dict(source_dataset=ds, **{f"n{e}": n for e, n in sorted(ext.items())},
                         manifest=int((m["source_dataset"] == ds).sum()), em_disco_fora_do_manifest=len(miss),
                         no_manifest_sem_arquivo=int((~mi.loc[mi.source_dataset == ds, "path"].isin(on_disk)).sum())))
    print(md(pd.DataFrame(rows).fillna(0).set_index("source_dataset").astype(int)))
    for ds, miss in missing.items():
        if miss:
            print(f"- {ds}: {len(miss)} audios em disco sem rotulo, ex.: {miss[:3]}")
    print()

    print("## B. Cabecalhos (soundfile.info)\n")
    g = mi.groupby("source_dataset")
    q = g["duration_s"].describe(percentiles=[.05, .5, .95])
    t = pd.DataFrame({"n": g.size(), "ilegiveis": g["readable"].apply(lambda s: int((~s).sum())),
                      "min_s": q["min"], "p5_s": q["5%"], "mediana_s": q["50%"], "media_s": q["mean"],
                      "p95_s": q["95%"], "max_s": q["max"], "total_h": g["duration_s"].sum() / 3600}).loc[order]
    print(md(t.round(2)))
    print("Duracao < 1 s / > 20 s por banco:\n")
    print(md(pd.DataFrame({"<1s": g["duration_s"].apply(lambda s: int((s < 1).sum())),
                           ">20s": g["duration_s"].apply(lambda s: int((s > 20).sum()))}).loc[order]))
    print("Taxa de amostragem (Hz) por banco:\n")
    print(md(mi.pivot_table(index="source_dataset", columns="sample_rate", values="path", aggfunc="count",
                            fill_value=0).loc[order]))
    print("Canais e formato por banco:\n")
    print(md(mi.groupby(["source_dataset", "channels", "format", "subtype"], dropna=False).size()
             .rename("n").reset_index(), index=False))
    bad = mi[~mi["readable"]]
    if len(bad):
        print("Ilegiveis:\n")
        print(md(bad[["path", "error"]].head(50), index=False))

    print("## C. Contagem source_dataset x label x split\n")
    c = m.assign(split=m["split"].fillna("excluded")).pivot_table(
        index="source_dataset", columns=["split", "label"], values="path", aggfunc="count", fill_value=0)
    print(md(c.loc[[d for d in order if d in c.index]]))
    print("Por role/split:\n")
    print(md(m.groupby(["role", m["split"].fillna("-"), "label"]).size().unstack(fill_value=0)))

    print("## D. Proporcao real/fake\n")
    tv = m[m.role == "train_val"]
    rows = [dict(conjunto=f"train_val/{s}", n=len(x), real=int((x.label == 0).sum()), fake=int((x.label == 1).sum()))
            for s, x in [("train+val", tv), ("train", tv[tv.split == "train"]), ("val", tv[tv.split == "val"])]]
    for ds, x in m[m.role.isin(["test_generalization", "test_ptbr"])].groupby("source_dataset", sort=False):
        sub = [(ds, x)] + ([(f"{ds} (so partial)", x[x.fake_type != "full"])] if ds == "llamapartialspoof" else [])
        for name, y in sub:
            rows.append(dict(conjunto=f"test/{name}", n=len(y), real=int((y.label == 0).sum()),
                             fake=int((y.label == 1).sum())))
    d = pd.DataFrame(rows)
    d["%fake"] = (100 * d.fake / d.n).round(1)
    print(md(d, index=False))
    print("train/val por banco do pool (audios e % em val):\n")
    p = tv.pivot_table(index="source_dataset", columns="split", values="path", aggfunc="count", fill_value=0)
    p["%val"] = (100 * p["val"] / (p["train"] + p["val"])).round(1)
    p["locutores_train"] = tv[tv.split == "train"].groupby("source_dataset")["speaker_id"].nunique()
    p["locutores_val"] = tv[tv.split == "val"].groupby("source_dataset")["speaker_id"].nunique()
    print(md(p))

    print("## E. Sobreposicao de locutores\n")
    tr = set(m.loc[m.split == "train", "speaker_id"].dropna())
    va = set(m.loc[m.split == "val", "speaker_id"].dropna())
    print(f"- locutores em train e val ao mesmo tempo: {len(tr & va)}\n")
    pool = tr | va
    rows = []
    for ds, x in m[m.role.isin(["test_generalization", "test_ptbr"])].groupby("source_dataset", sort=False):
        s = set(x["speaker_id"].dropna())
        inter = s & pool
        by_src = Counter(m.loc[(m.role == "train_val") & m.speaker_id.isin(inter), "source_dataset"])
        rows.append(dict(banco_teste=ds, locutores=len(s), em_comum_com_train_val=len(inter),
                         em_train=len(s & tr), em_val=len(s & va),
                         audios_teste_afetados=int(x.speaker_id.isin(inter).sum()),
                         bancos_train_val=", ".join(f"{k}" for k in by_src), exemplos=", ".join(sorted(inter)[:4])))
    print(md(pd.DataFrame(rows), index=False))
    ex = m[m.role == "excluded"]
    print(f"- excluded x teste: locutores do librispeech excluido que aparecem no llamapartialspoof: "
          f"{len(set(ex.speaker_id) & set(m.loc[m.source_dataset == 'llamapartialspoof', 'speaker_id']))}\n")

    print("## F. Duplicados entre bancos (nome do arquivo + duracao arredondada a 1 ms)\n")
    k = m.assign(name=m.path.str.rsplit("/", n=1).str[-1].str.lower(), dur=m.duration_s.round(3))
    k = k.dropna(subset=["dur"])
    grp = k.groupby(["name", "dur"])["source_dataset"].agg(lambda s: tuple(sorted(set(s))))
    multi = grp[grp.map(len) > 1]
    print(f"- chaves (nome, duracao) presentes em mais de um banco: {len(multi)}\n")
    if len(multi):
        print(md(multi.map(" + ".join).value_counts().rename("n_chaves").to_frame()))
        print("Exemplos:", ", ".join(f"{a} ({b}s)" for a, b in multi.index[:5]), "\n")
    k2 = k.assign(stem=k.name.str.replace(r"\.[a-z0-9]+$", "", regex=True))
    g2 = k2.groupby(["stem", "dur"])["source_dataset"].agg(lambda s: tuple(sorted(set(s))))
    m2 = g2[g2.map(len) > 1]
    print(f"- ignorando a extensao: {len(m2)} chaves\n")
    if len(m2):
        print(md(m2.map(" + ".join).value_counts().rename("n_chaves").to_frame()))
    within = k.groupby(["source_dataset", "name", "dur"]).size()
    w = within[within > 1].reset_index()
    print("- chaves (nome+duracao) repetidas DENTRO do mesmo banco:", len(w), dict(Counter(w.source_dataset)))
    if len(w):
        ex = k.merge(w[["source_dataset", "name", "dur"]].head(2), on=["source_dataset", "name", "dur"])
        print("  ex.:", ", ".join(ex.path.head(4)))
    print("\n## G. Geradores por banco\n")
    gen = m[m.label == 1].groupby("source_dataset")["generator"].agg(lambda s: f"{s.nunique()}: " + ", ".join(sorted(s.unique())))
    print(md(gen.to_frame()))
    print("Locutores por banco:", dict(m.groupby("source_dataset")["speaker_id"].nunique()))
    print("\nTaxa de amostragem x label (brspeech_df):\n")
    print(md(m[m.source_dataset == "brspeech_df"].groupby(["label", "sample_rate"]).size().rename("n").to_frame()))


if __name__ == "__main__":
    main()
