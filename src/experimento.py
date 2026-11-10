"""Preparar os dados, treinar UM experimento e calcular as notas (usado pelo notebook e por
scripts/treinar_experimento.py).

`hp` é o dicionário de hiperparâmetros montado na célula de configuração do notebook (seção 2).
`cfg` é a entrada do experimento em EXPERIMENTOS. Chaves opcionais de `cfg` (v3), com os padrões
que reproduzem a v2:
- "short_audio_mode": "repeat" (padrão) ou "zero" (ver src/dataset.py: completar_curto);
- "seed": seed do TREINO (amostragem das épocas, recorte aleatório e inicialização); padrão hp["SEED"].
A seed dos DADOS (amostra da validação no QUICK, do dev_shift e do teste) é sempre hp["SEED"].

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

from src.arquivos import salvar_csv
from src.augment import Augmentador
from src.dataset import AudioDataset, AudioInteiroDataset, checar_modo_curto, juntar_lista
from src.model import CNNPequena
from src.sampling import IndicesDaEpoca, SorteadorPorEpoca, calcular_cotas, montar_grupos
from src.train import prever_janelas, treinar

COLS_NOTAS = ["path", "source_dataset", "label", "generator", "fake_type", "speaker_id", "duration_s"]


def ler_manifest(caminho="data/processed/manifest.parquet"):
    """O manifest com `split` dos excluídos preenchido como "excluded" e `label` como int."""
    df = pd.read_parquet(caminho)
    assert df.loc[df.split.isna(), "role"].eq("excluded").all()
    df["split"] = df["split"].fillna("excluded")
    df["label"] = df["label"].astype(int)
    return df


def modo_curto(cfg):
    return checar_modo_curto(cfg.get("short_audio_mode", "repeat"))


def seed_treino(cfg, hp):
    return int(cfg.get("seed", hp["SEED"]))


def preparar_treino(df, hp):
    """Pool de treino, grupos e cotas da amostragem balanceada, e a validação (inteira ou amostrada)."""
    pool_tr = df[df.split == "train"].reset_index(drop=True)
    grupos = montar_grupos(pool_tr)
    cotas = calcular_cotas(grupos, hp["PER_CLASS_PER_EPOCH"])
    val_df = df[df.split == "val"]
    if hp["VAL_MAX"]:
        val_df = val_df.sample(hp["VAL_MAX"], random_state=hp["SEED"])
    return pool_tr, grupos, cotas, val_df.reset_index(drop=True)


def amostra_por_classe(g, maximo, seed):
    """Até `maximo` áudios por classe de `g` (todos se `maximo` for None), sorteados com `seed`."""
    if not maximo:
        return g.reset_index(drop=True)
    return pd.concat([h.sample(min(len(h), maximo), random_state=seed)
                      for _, h in g.groupby("label")]).reset_index(drop=True)


def amostra_dev_shift(df, hp):
    """O dev_shift (30% dos locutores do In-the-Wild), inteiro ou até DEV_SHIFT_MAX_PER_CLASS por classe."""
    return amostra_por_classe(df[df.eval_subset == "dev_shift"], hp.get("DEV_SHIFT_MAX_PER_CLASS"), hp["SEED"])


def amostra_teste(df, hp, testes):
    """Amostra do TESTE FINAL (até TEST_MAX_PER_CLASS por classe e banco; In-the-Wild só test_final).

    Só pode ser chamada na etapa de avaliação final, depois de todas as decisões (AVALIAR_TESTE_FINAL).
    """
    teste_full = df[(df.split == "test") & (df.eval_subset == "test_final")]
    partes = []
    for s in testes:
        for lab in (0, 1):
            g = teste_full[(teste_full.source_dataset == s) & (teste_full.label == lab)]
            if len(g):
                partes.append(g.sample(min(len(g), hp["TEST_MAX_PER_CLASS"]), random_state=hp["SEED"]))
    return pd.concat(partes).reset_index(drop=True), teste_full


def loader_validacao(val_df, hp, short_audio_mode="repeat"):
    """Recorte central de 4 s; workers só existem enquanto a validação roda (não persistente)."""
    ds = AudioDataset(val_df.path, val_df.label, val_df.sample_rate, val_df.duration_s,
                      crop_s=hp["CROP_S"], target_sr=hp["SAMPLE_RATE"], mode="center",
                      short_audio_mode=short_audio_mode)
    return DataLoader(ds, batch_size=hp["BATCH_SIZE"], shuffle=False, num_workers=hp["NUM_WORKERS"],
                      pin_memory=True, persistent_workers=False)


def n_quadros(hp):
    return int(hp["CROP_S"] * hp["SAMPLE_RATE"]) // hp["HOP_LENGTH"] + 1


def treinar_experimento(exp, cfg, hp, pool_tr, grupos, cotas, val_df, val_loader, logmel, device,
                        ckpt, hist_csv, versao, log=print, estado_path=None, config_id=None, extra_config=None):
    """Treina o experimento `exp` do zero (ou continua de `estado_path`) e grava o checkpoint e o
    histórico. Devolve o histórico.

    Tudo é recriado com a seed de treino do experimento: Dataset (com a augmentation e a política de
    áudio curto de `cfg`), CNN, Adam e o sorteador de épocas. Fontes de aleatoriedade separadas:
    - amostragem das épocas: SorteadorPorEpoca(grupos, seed), gerador numpy próprio;
    - recorte aleatório: se hp["RECORTE_DETERMINISTICO"], gerador próprio por (seed, época, áudio);
      senão (v1/v2), o gerador do torch de cada worker;
    - inicialização da CNN: torch.manual_seed(seed) logo antes de criá-la;
    - dropout: gerador global do torch (CPU/GPU), também a partir da seed.
    `val_loader=None` cria a validação com a mesma política de áudio curto do experimento.
    """
    modo, seed = modo_curto(cfg), seed_treino(cfg, hp)
    aug = Augmentador(**hp["AUG_BASE"], **cfg["aug"]) if cfg["aug"] else None
    log(f"\n=== {exp}: {cfg['descricao']}\n    augmentation: {aug} | áudio curto: {modo} | seed de treino: {seed}")
    if val_loader is None:
        val_loader = loader_validacao(val_df, hp, modo)
    assert val_loader.dataset.short_audio_mode == modo, "validação com outra política de áudio curto"
    assert not val_loader.persistent_workers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_ds = AudioDataset(pool_tr.path, pool_tr.label, pool_tr.sample_rate, pool_tr.duration_s,
                            crop_s=hp["CROP_S"], target_sr=hp["SAMPLE_RATE"], mode="random", augment=aug,
                            short_audio_mode=modo,
                            seed_recorte=seed if hp.get("RECORTE_DETERMINISTICO") else None)
    sampler = IndicesDaEpoca()
    train_loader = DataLoader(train_ds, batch_size=hp["BATCH_SIZE"], sampler=sampler,
                              num_workers=hp["NUM_WORKERS"], pin_memory=True, persistent_workers=False,
                              drop_last=True, generator=torch.Generator().manual_seed(seed))
    n_frames = n_quadros(hp)
    torch.manual_seed(seed)
    model = CNNPequena(hp["N_MELS"], n_frames, hp["DROPOUT"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["LR"])
    config = {"experimento": exp, "versao": versao, "aug": cfg["aug"], "N_FRAMES": n_frames,
              **{k: v for k, v in hp.items()}, "SHORT_AUDIO_MODE": modo, "SEED_TREINO": seed,
              "config_id": config_id, **(extra_config or {})}
    hist, minutos = treinar(model, logmel, criterion, optimizer, train_loader, sampler,
                            SorteadorPorEpoca(grupos, seed), cotas, val_loader,
                            val_df.source_dataset.values, device, hp["MAX_EPOCHS"], hp["PATIENCE"],
                            ckpt, config, log=log, estado_path=estado_path, config_id=config_id)
    salvar_csv(hist, hist_csv)   # gravado só no fim (atômico): sem histórico = treino incompleto
    log(f"{exp}: {minutos:.1f} min de treino (soma das épocas)")
    del train_loader, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hist


def carregar_modelo(ckpt_path, device):
    """CNN do checkpoint, o próprio checkpoint e a política de áudio curto com que ela foi treinada.

    Checkpoints anteriores à v3 não têm SHORT_AUDIO_MODE: foram treinados com "repeat".
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    m = CNNPequena(c["N_MELS"], c["N_FRAMES"], c["DROPOUT"]).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck, checar_modo_curto(c.get("SHORT_AUDIO_MODE", "repeat"))


def calcular_notas(model, logmel, amostra, hp, short_audio_mode, device, num_workers=None):
    """Logit de cada áudio de `amostra` por janelas (média dos logits) e pelo recorte central.

    O áudio inteiro é lido e, se tiver menos que a janela, completado com a MESMA política do treino
    (`short_audio_mode`). Devolve as colunas de COLS_NOTAS + z_janelas, z_central, n_janelas.
    """
    ds = AudioInteiroDataset(amostra.path, amostra.label, target_sr=hp["SAMPLE_RATE"], min_s=hp["JANELA_S"],
                             short_audio_mode=short_audio_mode)
    dl = DataLoader(ds, batch_size=hp["AUDIOS_POR_LOTE_JANELAS"], shuffle=False,
                    num_workers=hp["NUM_WORKERS"] if num_workers is None else num_workers,
                    collate_fn=juntar_lista)
    n, passo = int(hp["JANELA_S"] * hp["SAMPLE_RATE"]), int(hp["PASSO_S"] * hp["SAMPLE_RATE"])
    y, z_jan, z_cen, n_jan = prever_janelas(model, logmel, dl, device, n, passo)
    assert (y == amostra.label.values).all()
    out = amostra[COLS_NOTAS].reset_index(drop=True).assign(z_janelas=z_jan, z_central=z_cen, n_janelas=n_jan)
    if not (np.isfinite(out.z_janelas).all() and np.isfinite(out.z_central).all()):
        raise FloatingPointError("nota NaN ou infinita")
    return out
