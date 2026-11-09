"""Preparar os dados e treinar UM experimento (usado pelo notebook e por scripts/treinar_experimento.py).

`hp` é o dicionário de hiperparâmetros montado na célula de configuração do notebook (seção 2).

Memória: os DataLoaders NÃO são persistentes. No Windows, cada worker custa ~1 GB de memória
reservada (commit), principalmente as DLLs de CUDA que o `import torch` carrega. Com workers
persistentes no treino E na validação, ficavam 2 × NUM_WORKERS processos vivos o tempo todo. Agora,
no máximo NUM_WORKERS existem ao mesmo tempo (os do treino fecham no fim da época, antes da validação).
Isso não muda o que o modelo vê: as mesmas épocas, o mesmo sorteio de áudios e a mesma regra de
recorte e de augmentation (ver docs/HISTORICO_VERSOES.md, seção "Mudanças de execução").
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.augment import Augmentador
from src.dataset import AudioDataset
from src.model import CNNPequena
from src.sampling import IndicesDaEpoca, SorteadorPorEpoca, calcular_cotas, montar_grupos
from src.train import treinar


def ler_manifest(caminho="data/processed/manifest.parquet"):
    """O manifest com `split` dos excluídos preenchido como "excluded" e `label` como int."""
    df = pd.read_parquet(caminho)
    assert df.loc[df.split.isna(), "role"].eq("excluded").all()
    df["split"] = df["split"].fillna("excluded")
    df["label"] = df["label"].astype(int)
    return df


def preparar_treino(df, hp):
    """Pool de treino, grupos e cotas da amostragem balanceada, e a validação (inteira ou amostrada)."""
    pool_tr = df[df.split == "train"].reset_index(drop=True)
    grupos = montar_grupos(pool_tr)
    cotas = calcular_cotas(grupos, hp["PER_CLASS_PER_EPOCH"])
    val_df = df[df.split == "val"]
    if hp["VAL_MAX"]:
        val_df = val_df.sample(hp["VAL_MAX"], random_state=hp["SEED"])
    return pool_tr, grupos, cotas, val_df.reset_index(drop=True)


def loader_validacao(val_df, hp):
    """Recorte central de 4 s; workers só existem enquanto a validação roda (não persistente)."""
    ds = AudioDataset(val_df.path, val_df.label, val_df.sample_rate, val_df.duration_s,
                      crop_s=hp["CROP_S"], target_sr=hp["SAMPLE_RATE"], mode="center")
    return DataLoader(ds, batch_size=hp["BATCH_SIZE"], shuffle=False, num_workers=hp["NUM_WORKERS"],
                      pin_memory=True, persistent_workers=False)


def treinar_experimento(exp, cfg, hp, pool_tr, grupos, cotas, val_df, val_loader, logmel, device,
                        ckpt, hist_csv, versao, log=print):
    """Treina o experimento `exp` do zero e grava o checkpoint e o histórico. Devolve o histórico.

    Tudo é recriado com a mesma seed: Dataset (com a augmentation de `cfg`), CNN, Adam e o sorteador
    de épocas, então os experimentos só diferem na augmentation.
    """
    aug = Augmentador(**hp["AUG_BASE"], **cfg["aug"]) if cfg["aug"] else None
    log(f"\n=== {exp}: {cfg['descricao']}\n    augmentation: {aug}")
    random.seed(hp["SEED"])
    np.random.seed(hp["SEED"])
    torch.manual_seed(hp["SEED"])
    train_ds = AudioDataset(pool_tr.path, pool_tr.label, pool_tr.sample_rate, pool_tr.duration_s,
                            crop_s=hp["CROP_S"], target_sr=hp["SAMPLE_RATE"], mode="random", augment=aug)
    sampler = IndicesDaEpoca()
    train_loader = DataLoader(train_ds, batch_size=hp["BATCH_SIZE"], sampler=sampler,
                              num_workers=hp["NUM_WORKERS"], pin_memory=True, persistent_workers=False,
                              drop_last=True, generator=torch.Generator().manual_seed(hp["SEED"]))
    n_frames = int(hp["CROP_S"] * hp["SAMPLE_RATE"]) // hp["HOP_LENGTH"] + 1
    torch.manual_seed(hp["SEED"])
    model = CNNPequena(hp["N_MELS"], n_frames, hp["DROPOUT"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["LR"])
    config = {"experimento": exp, "versao": versao, "aug": cfg["aug"], "N_FRAMES": n_frames,
              **{k: v for k, v in hp.items()}}
    hist, minutos = treinar(model, logmel, criterion, optimizer, train_loader, sampler,
                            SorteadorPorEpoca(grupos, hp["SEED"]), cotas, val_loader,
                            val_df.source_dataset.values, device, hp["MAX_EPOCHS"], hp["PATIENCE"],
                            ckpt, config, log=log)
    hist.to_csv(hist_csv, index=False)   # gravado só no fim: sem histórico = treino incompleto
    log(f"{exp}: {minutos:.1f} min de treino")
    del train_loader, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hist
