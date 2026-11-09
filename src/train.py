"""Loop de treino (Aula 6) com early stopping (Aula 7) e as funções de predição.

Diferença para o baseline v1: o checkpoint é escolhido pela MÉDIA das EER por fonte na validação
(só fontes com real e fake), e não pela EER da validação inteira, que é ~79% DFADD.
"""
import time

import numpy as np
import pandas as pd
import torch

from src.metrics import eer_e_limiar, eer_media_por_fonte


@torch.no_grad()
def prever(model, logmel, loader, device, criterion=None):
    """Logit de cada áudio do loader (na ordem), rótulos e perda média (se `criterion`)."""
    model.eval()
    ys, zs, perda, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        z = model(logmel(x))
        if criterion is not None:
            perda += criterion(z, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu())
        zs.append(z.float().cpu())
    return torch.cat(ys).numpy().astype(int), torch.cat(zs).numpy(), (perda / n if criterion else np.nan)


def cortar_janelas(x, n, passo):
    """Lista de janelas de `n` amostras com passo `passo` (views, sem copiar o áudio).

    A última janela termina no fim do áudio, para cobrir tudo. Áudio com até `n` amostras: 1 janela.
    """
    L = x.numel()
    if L <= n:
        return [x[:n]]
    inicios = list(range(0, L - n + 1, passo))
    if inicios[-1] != L - n:
        inicios.append(L - n)
    return [x[s:s + n] for s in inicios]


@torch.no_grad()
def prever_janelas(model, logmel, loader, device, n, passo, max_lote=512):
    """Para cada áudio inteiro: média dos logits das janelas, logit do recorte central e nº de janelas.

    O recorte central é tirado do mesmo áudio já reamostrado (o trecho de `n` amostras do meio,
    mesmo critério do AudioDataset em mode="center"). As janelas de vários áudios vão juntas para
    a GPU em lotes de até `max_lote`.
    """
    model.eval()
    ys, z_media, z_central, n_jan = [], [], [], []
    for ondas, y in loader:
        linhas, conta = [], []
        for x in ondas:
            w = cortar_janelas(x, n, passo)
            linhas.extend(w)
            conta.append(len(w))
        for x in ondas:                               # recortes centrais, depois de todas as janelas
            ini = (x.numel() - n) // 2 if x.numel() > n else 0
            linhas.append(x[ini:ini + n])
        zs = torch.cat([model(logmel(torch.stack(linhas[i:i + max_lote]).to(device, non_blocking=True)))
                        .float().cpu() for i in range(0, len(linhas), max_lote)])
        z_jan, z_cen = zs[: sum(conta)], zs[sum(conta):]
        z_media.extend(t.mean().item() for t in torch.split(z_jan, conta))
        z_central.append(z_cen)
        n_jan.extend(conta)
        ys.append(y)
    return (torch.cat(ys).numpy().astype(int), np.asarray(z_media),
            torch.cat(z_central).numpy(), np.asarray(n_jan))


def treinar(model, logmel, criterion, optimizer, train_loader, sampler, sorteador, cotas,
            val_loader, val_fontes, device, max_epocas, paciencia, ckpt_path, config, log=print):
    """Treina e salva em `ckpt_path` o checkpoint com a MENOR EER média por fonte na validação.

    Para (early stopping) depois de `paciencia` épocas sem melhorar essa métrica.
    Devolve o histórico por época (DataFrame) e o tempo total em minutos.
    """
    n_pool = len(train_loader.dataset)
    vistos = np.zeros(n_pool, dtype=bool)
    historico, melhor, sem_melhora = [], float("inf"), 0
    t_inicio = time.time()
    for epoca in range(1, max_epocas + 1):
        t0 = time.time()
        sampler.indices = sorteador.epoca(cotas)
        vistos[sampler.indices] = True
        model.train()
        soma, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            perda = criterion(model(logmel(x)), y)
            optimizer.zero_grad()
            perda.backward()
            optimizer.step()
            soma += perda.item() * len(y)
            n += len(y)

        y_val, z_val, perda_val = prever(model, logmel, val_loader, device, criterion)
        eer_media, por_fonte = eer_media_por_fonte(y_val, z_val, val_fontes)
        eer_global, _ = eer_e_limiar(y_val, z_val)
        reg = {"época": epoca, "perda treino": soma / n, "perda val": perda_val,
               "EER média por fonte (val)": eer_media, "EER global (val)": eer_global,
               **{f"EER val {s}": v for s, v in por_fonte.items()},
               "% do pool já visto": 100 * vistos.mean(), "segundos": time.time() - t0}
        historico.append(reg)
        marca = ""
        if eer_media < melhor:
            melhor, sem_melhora, marca = eer_media, 0, "  <- melhor, checkpoint salvo"
            torch.save({"model_state": model.state_dict(), "epoca": epoca,
                        "eer_media_por_fonte_val": eer_media, "eer_global_val": eer_global,
                        "config": config}, ckpt_path)
        else:
            sem_melhora += 1
        log(f"época {epoca:2d} | perda treino {reg['perda treino']:.4f} | perda val {perda_val:.4f} | "
            f"EER média/fonte {100 * eer_media:5.2f}% | EER global {100 * eer_global:5.2f}% | "
            f"{reg['segundos']:.0f} s{marca}")
        if sem_melhora >= paciencia:
            log(f"early stopping: {paciencia} épocas sem melhorar a EER média por fonte")
            break
    return pd.DataFrame(historico), (time.time() - t_inicio) / 60
