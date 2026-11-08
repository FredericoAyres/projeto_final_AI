#!/usr/bin/env python3
"""Gera data/processed/manifest.parquet (uma linha por audio) a partir de data/raw/.

Uso (na raiz do repo, com o env ativo):
    python scripts/build_manifest.py

O que faz:
  1. Chama scripts/extract_brspeech.py (extrai os audios do BRSpeech-DF dos .parquet para
     data/processed/brspeech_df/test/, pulando o que ja existe).
  2. Le os arquivos de rotulo de cada banco (protocolos, labels .txt, metadata .csv) e manda para
     role=excluded toda linha train_val cujo locutor esteja no LlamaPartialSpoof (vazamento).
  3. Le o cabecalho de todos os audios (soundfile.info, em paralelo). O resultado fica em
     cache em data/processed/audio_info.parquet; rodar de novo reaproveita o cache.
  4. Divide o pool train_val em train/val (80/20) por GRUPO DE LOCUTOR, global entre bancos,
     seed 42.

Nada em data/raw/ e modificado. Detalhes e duvidas em docs/audit_report.md.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from extract_brspeech import extract_brspeech   # scripts/extract_brspeech.py

ROOT = Path(__file__).resolve().parent.parent
RAW = "data/raw"
PROC = ROOT / "data" / "processed"
INFO_CACHE = PROC / "audio_info.parquet"
OUT = PROC / "manifest.parquet"
SEED = 42
VAL_FRAC = 0.20
COLS = ["path", "label", "source_dataset", "generator", "fake_type", "speaker_id", "language",
        "original_split", "role", "split", "duration_s", "sample_rate"]


def rows(records, **const):
    df = pd.DataFrame(records)
    for k, v in const.items():
        df[k] = v
    return df


def read_ws(path):
    """Arquivo texto separado por espacos -> lista de listas."""
    with open(ROOT / path, encoding="utf-8") as f:
        return [ln.split() for ln in f if ln.strip()]


# --------------------------------------------------------------------------- bancos
def asvspoof2019_la():
    base = f"{RAW}/train_val_en/asvspoof2019_la/LA"
    out, listed = [], set()
    for part, proto in [("train", "train.trn"), ("dev", "dev.trl"), ("eval", "eval.trl")]:
        # SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY   (README.LA.txt)
        for spk, utt, _, sysid, key in read_ws(f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{proto}.txt"):
            listed.add(utt)
            out.append(dict(path=f"{base}/ASVspoof2019_LA_{part}/flac/{utt}.flac", label=int(key == "spoof"),
                            generator=sysid if key == "spoof" else None, fake_type="full" if key == "spoof" else "none",
                            speaker_id=f"asvspoof:{spk}", original_split=part,
                            role="train_val" if part in ("train", "dev") else "test_generalization"))
    # audios de cadastro do ASV (LA_D_A*, LA_E_A*): bonafide, fora do protocolo CM -> excluded
    for part in ("dev", "eval"):
        for g in ("female", "male"):
            for spk, files in read_ws(f"{base}/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.{part}.{g}.trn.txt"):
                for utt in files.split(","):
                    if utt not in listed:
                        listed.add(utt)
                        out.append(dict(path=f"{base}/ASVspoof2019_LA_{part}/flac/{utt}.flac", label=0, generator=None,
                                        fake_type="none", speaker_id=f"asvspoof:{spk}", original_split=f"{part}_asv_enroll",
                                        role="excluded"))
    return rows(out, source_dataset="asvspoof2019_la", language="en")


def dfadd():
    base = f"{RAW}/train_val_en/dfadd"
    bona = f"{base}/DATASET_VCTK_BONAFIDE/home/isjwdu/Work/Dataset/DFADD/DATASET_VCTK_BONAFIDE"
    out, seen = [], set()
    for tts in ("GradTTS", "MatchaTTS", "NaturalSpeech2", "PflowTTS", "StyleTTS2"):
        d = f"{base}/DATASET_{tts}/DATASET_{tts}"
        for part in ("train", "valid", "test"):
            # mesmo formato do ASVspoof: SPEAKER UTT - - KEY ; os bonafide se repetem em cada subconjunto
            for spk, utt, _, _, key in read_ws(f"{d}/{part}.txt"):
                if key == "spoof":
                    p, gen = f"{d}/{part}/{utt}.flac", tts
                else:
                    p, gen = f"{bona}/{part}/{utt}.wav", None
                if p in seen:
                    continue
                seen.add(p)
                out.append(dict(path=p, label=int(key == "spoof"), generator=gen,
                                fake_type="full" if key == "spoof" else "none",
                                speaker_id=f"vctk:{spk}", original_split=part))
    return rows(out, source_dataset="dfadd", language="en", role="train_val")


def shiftyspeech():
    base = f"{RAW}/train_val_en/shiftyspeech"
    out = []
    # sem arquivo de rotulo: o rotulo vem da pasta (real_data_* = real; <vocoder>_* = vocoded)
    for folder, gen in [("real_data_audiobook_flac", None), ("hifigan_audiobook_flac", "hifigan"),
                        ("vocos_audiobook_flac", "vocos"), ("wavegrad_audiobook_flac", "wavegrad")]:
        d = f"{base}/{folder}/audiobook_flac"
        for f in sorted(os.listdir(ROOT / d)):
            # nomes no padrao LibriSpeech: <locutor>-<capitulo>-<utt>[_gen].flac
            out.append(dict(path=f"{d}/{f}", label=int(gen is not None), generator=gen,
                            fake_type="full" if gen else "none", speaker_id=f"librispeech:{f.split('-')[0]}"))
    return rows(out, source_dataset="shiftyspeech", language="en", original_split=None, role="train_val")


def codecfake_plus():
    base = f"{RAW}/train_val_en/codecfake_plus"
    out = []
    # <modelo/demo> <id> <quantizacao> <aux> <dominio> <bonafide|Spoof>
    for model, utt, _, _, _, key in read_ws(f"{base}/CoSG_labels.txt"):
        spoof = key.lower() == "spoof"
        out.append(dict(path=f"{base}/CoSG/{utt}.wav", label=int(spoof), generator=model if spoof else None,
                        fake_type="full" if spoof else "none", speaker_id=None, cosg_model=model))
    return rows(out, source_dataset="codecfake_plus", language="en", original_split="CoSG", role="train_val")


def librispeech():
    base = f"{RAW}/real_en/librispeech/LibriSpeech"
    out = []
    for subset in ("train-clean-100", "dev-clean", "test-clean"):
        for r, _, fs in os.walk(ROOT / base / subset):
            for f in fs:
                if f.endswith(".flac"):
                    rel = Path(r).relative_to(ROOT).as_posix()
                    out.append(dict(path=f"{rel}/{f}", speaker_id=f"librispeech:{f.split('-')[0]}", original_split=subset,
                                    role="train_val" if subset == "train-clean-100" else "excluded"))
    return rows(out, source_dataset="librispeech", language="en", label=0, generator=None, fake_type="none")


def in_the_wild():
    base = f"{RAW}/test_generalization/in_the_wild/release_in_the_wild"
    m = pd.read_csv(ROOT / base / "meta.csv")          # file,speaker,label (spoof | bona-fide)
    spoof = m["label"].eq("spoof")
    return pd.DataFrame(dict(path=base + "/" + m["file"], label=spoof.astype(int),
                             generator=np.where(spoof, "unknown", None), fake_type=np.where(spoof, "full", "none"),
                             speaker_id="itw:" + m["speaker"], source_dataset="in_the_wild", language="en",
                             original_split=None, role="test_generalization"))


def llamapartialspoof():
    base = f"{RAW}/test_generalization/llamapartialspoof"
    out = []
    for part in ("R01TTS.0.a", "R01TTS.0.b"):
        # <id> <duracao> <rotulo do audio> <ini-fim-rotulo> ...
        for line in read_ws(f"{base}/label_{part}.txt"):
            utt, key = line[0], line[2]
            prefix, spk = utt.split("_")[:2]
            if prefix in ("dev-clean", "test-clean"):        # real: LibriTTS dev-clean/test-clean
                gen, by_name, osplit = None, "none", prefix
            else:                                             # dev01-<tts>-full | dev01-<tts>-partial-<cf|cp|oa>
                _, gen, kind = prefix.split("-", 2)
                by_name, osplit = ("full" if kind == "full" else "partial"), None
            # fake_type vem dos segmentos: so falsos = full; mistura de reais e falsos = partial
            seg = {s.rsplit("-", 1)[1] for s in line[3:]}
            ftype = "none" if key != "spoof" else ("partial" if "bonafide" in seg else "full")
            out.append(dict(path=f"{base}/{part}/{utt}.wav", label=int(key == "spoof"), generator=gen,
                            fake_type=ftype, fake_type_by_name=by_name, speaker_id=f"librispeech:{spk}",
                            original_split=osplit))
    return rows(out, source_dataset="llamapartialspoof", language="en", role="test_generalization")


def asvspoof2021_df():
    base = f"{RAW}/test_generalization/asvspoof2021_df"
    flac = f"{base}/ASVspoof2021_DF_eval/flac"
    present = {f[:-5] for f in os.listdir(ROOT / flac) if f.endswith(".flac")}   # so 1 das 4 partes foi baixada
    out = []
    # SPK FILE CODEC SOURCE ATTACK KEY TRIM SUBSET VOCODER ...   (keys/DF/CM/trial_metadata.txt)
    for c in read_ws(f"{base}/keys/DF/CM/trial_metadata.txt"):
        if c[1] in present:
            spoof = c[5] == "spoof"
            out.append(dict(path=f"{flac}/{c[1]}.flac", label=int(spoof), generator=c[4] if spoof else None,
                            fake_type="full" if spoof else "none", speaker_id=f"{c[3]}:{c[0]}", original_split=c[7]))
    return rows(out, source_dataset="asvspoof2021_df", language="en", role="test_generalization")


def brspeech_df():
    out = []
    # label: ClassLabel {0: bonafide, 1: spoof}; nome: [<tts>_]<locutor>_<livro>_<segmento>[-<n>].flac
    # o audio mistura pt-BR e pt-PT (card do banco), sem indicacao por arquivo -> language="pt"
    for path, lab, _ in extract_brspeech():          # scripts/extract_brspeech.py (pula o ja extraido)
        parts = Path(path).name.split("_")
        gen = parts[0] if lab == 1 else None
        spk = parts[1] if lab == 1 else parts[0]
        out.append(dict(path=path, label=lab, generator=gen, fake_type="full" if lab else "none",
                        speaker_id=f"brspeech:{spk}"))
    return rows(out, source_dataset="brspeech_df", language="pt", original_split="test", role="test_ptbr")


def coraa():
    base = f"{RAW}/test_ptbr/coraa"
    m = pd.read_csv(ROOT / base / "metadata_test_final.csv")
    # 326 audios com variety=pt_pt (TEDx, portugues europeu) ficam fora do teste pt-BR
    return pd.DataFrame(dict(path=base + "/" + m["file_path"], label=0, generator=None, fake_type="none",
                             speaker_id=None, source_dataset="coraa", language="pt", original_split="test",
                             role=np.where(m["variety"].eq("pt_br"), "test_ptbr", "excluded")))


LOADERS = [asvspoof2019_la, dfadd, shiftyspeech, codecfake_plus, librispeech,
           in_the_wild, llamapartialspoof, asvspoof2021_df, brspeech_df, coraa]


# --------------------------------------------------------------------------- cabecalhos
def info(path):
    try:
        i = sf.info(str(ROOT / path))
        return path, True, None, i.frames / i.samplerate if i.samplerate else np.nan, i.samplerate, i.channels, i.format, i.subtype
    except Exception as e:
        return path, False, f"{type(e).__name__}: {e}"[:200], np.nan, np.nan, np.nan, None, None


def read_headers(paths):
    cache = pd.read_parquet(INFO_CACHE) if INFO_CACHE.exists() else pd.DataFrame(columns=["path"])
    todo = sorted(set(paths) - set(cache["path"]))
    if todo:
        with ThreadPoolExecutor(32) as ex:
            res = list(tqdm(ex.map(info, todo, chunksize=256), total=len(todo), desc="soundfile.info", mininterval=5))
        new = pd.DataFrame(res, columns=["path", "readable", "error", "duration_s", "sample_rate", "channels",
                                         "format", "subtype"])
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        cache.to_parquet(INFO_CACHE, index=False)
    return cache


# --------------------------------------------------------------------------- split
def assign_split(df):
    """train/val por grupo de locutor (global entre bancos). Sem locutor: grupo = demo/modelo do CoSG.

    ASVspoof 2019 LA e DFADD vem do VCTK, mas o ASVspoof anonimiza os locutores (LA_xxxx) e nao ha
    mapeamento para os ids do VCTK (pXXX). Para reduzir o vazamento, todo o ASVspoof 2019 train+dev
    vira UM grupo fixo em train (ver docs/audit_report.md).
    """
    tv = df["role"].eq("train_val")
    group = df["speaker_id"].fillna("cosg:" + df["cosg_model"])
    group = group.mask(df["source_dataset"].eq("asvspoof2019_la"), "asvspoof2019_trainval")
    # cada grupo pertence a um unico banco do pool (os namespaces de locutor nao se cruzam no train_val)
    home = df[tv].groupby(group[tv])["source_dataset"].agg(lambda s: s.mode()[0])
    sizes = group[tv].value_counts()
    order = np.random.default_rng(SEED).permutation(sorted(sizes.index))
    val_groups = set()
    for ds in sorted(set(home) - {"asvspoof2019_la"}):
        # por banco: percorre os grupos em ordem aleatoria e enche a validacao ate <= 20% dos audios do banco
        target, acc = VAL_FRAC * sizes[home.index[home == ds]].sum(), 0
        for g in order:
            if home[g] == ds and acc + sizes[g] <= target:
                val_groups.add(g)
                acc += sizes[g]
    df["split"] = "test"
    df.loc[tv, "split"] = np.where(group[tv].isin(val_groups), "val", "train")
    df.loc[df["role"].eq("excluded"), "split"] = None
    return df


def exclude_test_speakers(df):
    """Vazamento: locutores do LlamaPartialSpoof nao podem estar no pool train_val (de nenhum banco)."""
    lps = set(df.loc[df["source_dataset"].eq("llamapartialspoof"), "speaker_id"])
    hit = df["role"].eq("train_val") & df["speaker_id"].isin(lps)
    df.loc[hit, "role"] = "excluded"
    print(f"locutores do llamapartialspoof: {len(lps)}; linhas train_val -> excluded: {hit.sum()}")
    print(df[hit].groupby(["source_dataset", "label"]).agg(linhas=("path", "size"), locutores=("speaker_id", "nunique")))
    tc100 = set(df.loc[df["original_split"].eq("train-clean-100"), "speaker_id"])
    print(f"locutores do librispeech train-clean-100 no conjunto do llamapartialspoof: {len(tc100 & lps)}")
    return df


def main():
    df = pd.concat([f() for f in tqdm(LOADERS, desc="rotulos")], ignore_index=True)
    assert not df["path"].duplicated().any(), "path duplicado no manifest"
    lp = df[df["source_dataset"].eq("llamapartialspoof")]
    print("llamapartialspoof fake_type (nome do arquivo -> segmentos do label):")
    print(lp.groupby(["fake_type_by_name", "fake_type"]).size())
    print(f"divergentes: {(lp['fake_type_by_name'] != lp['fake_type']).sum()}")
    df = exclude_test_speakers(df)
    inf = read_headers(df["path"].tolist())
    df = df.merge(inf[["path", "readable", "duration_s", "sample_rate"]], on="path", how="left")
    df = assign_split(df)
    df["label"] = df["label"].astype("int8")
    df["sample_rate"] = df["sample_rate"].astype("Int32")
    df[COLS].to_parquet(OUT, index=False)
    print(f"{OUT.relative_to(ROOT)}: {len(df)} linhas, {OUT.stat().st_size / 1e6:.1f} MB; "
          f"ilegiveis: {(~df['readable']).sum()}")
    print(df.groupby(["role", "split"], dropna=False).size())


if __name__ == "__main__":
    main()
