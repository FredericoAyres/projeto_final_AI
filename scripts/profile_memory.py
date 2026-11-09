#!/usr/bin/env python3
"""Mede a memória do treino (processo principal + cada worker do DataLoader), época a época.

Repete o pipeline do notebook (mesmo src/, mesmos DataLoaders), mas com épocas pequenas.
No Windows o que importa é o COMMIT (memória privada reservada, RAM + pagefile): quando o commit
do sistema chega ao limite, os processos começam a falhar, mesmo com RAM física livre.

Os imports pesados ficam DENTRO de main(): no Windows cada worker reimporta este arquivo, e um
import no topo (ex.: sklearn, que traz o scipy e uma 2ª cópia do OpenBLAS) inflaria cada worker,
o que não acontece no notebook.

Uso (na raiz do repo, com o env ativo):
    python scripts/profile_memory.py --workers 8 --epocas 3 --por-classe 3000 --val-max 3000 --aug
Opções:
    --treino-persistente  workers do treino sempre abertos (como no notebook v2 antes da correção)
    --val-persistente   workers da validação sempre abertos (como no notebook v2 antes da correção)
    --val-workers N     workers da validação (padrão: igual a --workers)
    --sem-limitar       NÃO aplica src.dataset.limitar_threads_dos_workers() (como antes da correção)
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.memoria import GB, MB, Amostrador, commit_sistema, processos  # noqa: E402  (só psutil e ctypes)


def foto(momento):
    ps = processos()
    sis, lim = commit_sistema()
    w = [c for papel, _, _, c in ps if papel == "worker"]
    media = sum(w) / len(w) if w else 0.0
    print(f"{momento:<28} principal {ps[0][3]:6.0f} MB | {len(w):2d} workers x {media:5.0f} MB "
          f"(máx {max(w) if w else 0:5.0f}) | treino {sum(c for *_, c in ps) / 1024:5.2f} GB | "
          f"sistema {sis / GB:5.1f} / {lim / GB:.1f} GB", flush=True)
    return media


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-workers", type=int, default=None)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--por-classe", type=int, default=3000)
    ap.add_argument("--val-max", type=int, default=3000)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--val-persistente", action="store_true")
    ap.add_argument("--sem-limitar", action="store_true")
    ap.add_argument("--treino-persistente", action="store_true",
                    help="workers do treino sempre abertos (como no notebook v2 antes da correção)")
    a = ap.parse_args()
    a.val_workers = a.workers if a.val_workers is None else a.val_workers
    print(vars(a))
    os.chdir(ROOT)

    amostrador = Amostrador()
    amostrador.start()
    foto("início (só psutil)")
    import pickle

    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.augment import Augmentador
    from src.dataset import AudioDataset, limitar_threads_dos_workers
    from src.metrics import eer_media_por_fonte
    from src.model import CNNPequena, LogMelPadronizado
    from src.sampling import IndicesDaEpoca, SorteadorPorEpoca, calcular_cotas, montar_grupos
    from src.train import prever
    foto("imports do notebook")
    if not a.sem_limitar:
        limitar_threads_dos_workers()

    df = pd.read_parquet("data/processed/manifest.parquet")
    print(f"manifest na memória (pandas): {df.memory_usage(deep=True).sum() / MB:.0f} MB")
    pool = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].sample(a.val_max, random_state=42).reset_index(drop=True)
    foto("manifest carregado")

    dev = torch.device("cuda")
    grupos = montar_grupos(pool)
    cotas = calcular_cotas(grupos, a.por_classe)
    aug = Augmentador(p_rms=0.5, p_ganho=0.5, p_ruido=0.3, p_banda=0.3, p_mulaw=0.3) if a.aug else None
    tr = AudioDataset(pool.path, pool.label, pool.sample_rate, pool.duration_s, mode="random", augment=aug)
    va = AudioDataset(val.path, val.label, val.sample_rate, val.duration_s, mode="center")
    print(f"Dataset de treino serializado (o que cada worker recebe): {len(pickle.dumps(tr)) / MB:.1f} MB")
    sampler = IndicesDaEpoca()
    tl = DataLoader(tr, batch_size=64, sampler=sampler, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.treino_persistente, drop_last=True,
                    generator=torch.Generator().manual_seed(42))
    vl = DataLoader(va, batch_size=64, num_workers=a.val_workers, pin_memory=True,
                    persistent_workers=a.val_persistente)
    logmel = LogMelPadronizado(16000, 512, 400, 160, 64).to(dev)
    torch.manual_seed(42)
    model = CNNPequena(64, 401, 0.3).to(dev)
    crit, opt = nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)
    sort = SorteadorPorEpoca(grupos, 42)
    foto("modelo na GPU")

    medias = []
    for ep in range(1, a.epocas + 1):
        sampler.indices = sort.epoca(cotas)
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = crit(model(logmel(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i == 5:
                foto(f"época {ep}: começo do treino")
        medias.append(foto(f"época {ep}: fim do treino"))
        yv, zv, _ = prever(model, logmel, vl, dev, crit)
        eer, _ = eer_media_por_fonte(yv, zv, val.source_dataset.values)
        foto(f"época {ep}: fim da validação")
        print(f"   época {ep}: {time.time() - t0:.0f} s, EER média por fonte {100 * eer:.1f}%")

    del tl, vl
    time.sleep(2)
    foto("loaders fechados")
    amostrador.parar = True
    print(f"\ncrescimento do commit médio por worker de treino: {medias[0]:.0f} -> {medias[-1]:.0f} MB "
          f"({(medias[-1] - medias[0]) / max(len(medias) - 1, 1):+.0f} MB por época)")
    print(f"PICO do commit do treino (principal + workers): {amostrador.pico_proc / 1024:.2f} GB")
    print(f"PICO do commit do sistema: {amostrador.pico_sis:.1f} GB (limite agora {commit_sistema()[1] / GB:.1f} GB)")
    print(f"memória da GPU (pico): {torch.cuda.max_memory_allocated() / GB:.2f} GB")


if __name__ == "__main__":
    main()
