"""Loop de treino (Aula 6) com early stopping (Aula 7) e as funções de predição.

Diferença para o baseline v1: o checkpoint é escolhido pela MÉDIA das EER por fonte na validação
(só fontes com real e fake), e não pela EER da validação inteira, que é ~79% DFADD.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.arquivos import salvar_torch
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


def _copia_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def treinar(model, logmel, criterion, optimizer, train_loader, sampler, sorteador, cotas,
            val_loader, val_fontes, device, max_epocas, paciencia, ckpt_path, config, log=print,
            estado_path=None, config_id=None):
    """Treina e salva em `ckpt_path` o checkpoint com a MENOR EER média por fonte na validação.

    Para (early stopping) depois de `paciencia` épocas sem melhorar essa métrica.
    Devolve o histórico por época (DataFrame) e o tempo total em minutos (soma das épocas).

    Execução (v3; não muda o método):
    - O checkpoint é gravado de forma atômica (arquivo temporário + renomear).
    - Antes de cada época, `train_loader.dataset.epoca` recebe o número da época (o recorte aleatório
      com `seed_recorte` depende dele; os workers não são persistentes, então recebem o valor novo).
    - Com `estado_path`, grava no fim de cada época tudo o que é preciso para continuar: modelo,
      Adam, melhor modelo até agora, sorteador de épocas, geradores aleatórios, histórico e contadores.
      Se o arquivo já existe ao chamar, o treino CONTINUA da época seguinte à última concluída (só se
      `config_id` for o mesmo; senão, para com erro, sem apagar nada). A retomada segue o mesmo
      método, mas não é garantidamente idêntica bit a bit a uma execução sem interrupção.
    """
    n_pool = len(train_loader.dataset)
    vistos = np.zeros(n_pool, dtype=bool)
    historico, melhor, sem_melhora = [], float("inf"), 0
    melhor_estado, retomadas, primeira, terminou = None, [], 1, False

    if estado_path is not None and Path(estado_path).exists():
        est = torch.load(estado_path, map_location="cpu", weights_only=False)
        if est["config_id"] != config_id:
            raise RuntimeError(f"{estado_path} é de outra configuração ({est['config_id']} ≠ {config_id}). "
                               "Nada foi apagado: confira e remova o arquivo à mão se quiser recomeçar.")
        model.load_state_dict(est["model_state"])
        optimizer.load_state_dict(est["optimizer_state"])
        sorteador.restaurar(est["sorteador"])
        torch.set_rng_state(est["rng_torch"])
        if est.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(est["rng_cuda"])
        if est.get("rng_loader") is not None and train_loader.generator is not None:
            train_loader.generator.set_state(est["rng_loader"])
        vistos, historico = est["vistos"], est["historico"]
        melhor, sem_melhora, melhor_estado = est["melhor"], est["sem_melhora"], est["melhor_estado"]
        retomadas, terminou = est["retomadas"], est["terminou"]
        primeira = est["epoca"] + 1
        if not terminou:
            retomadas = retomadas + [primeira]
            log(f"retomando de {estado_path}: {est['epoca']} época(s) já concluída(s); continua na época {primeira}")
        # o checkpoint em disco pode ser de uma época que não chegou a entrar no estado: regrava o melhor
        if melhor_estado is not None:
            salvar_torch(melhor_estado, ckpt_path)

    for epoca in range(primeira, max_epocas + 1):
        if terminou:
            break
        t0 = time.time()
        sampler.indices = sorteador.epoca(cotas)
        vistos[sampler.indices] = True
        if hasattr(train_loader.dataset, "epoca"):
            train_loader.dataset.epoca = epoca
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
            melhor_estado = {"model_state": _copia_cpu(model.state_dict()), "epoca": epoca,
                             "eer_media_por_fonte_val": eer_media, "eer_global_val": eer_global,
                             "config": config}
            salvar_torch(melhor_estado, ckpt_path)
        else:
            sem_melhora += 1
        log(f"época {epoca:2d} | perda treino {reg['perda treino']:.4f} | perda val {perda_val:.4f} | "
            f"EER média/fonte {100 * eer_media:5.2f}% | EER global {100 * eer_global:5.2f}% | "
            f"{reg['segundos']:.0f} s{marca}")
        if sem_melhora >= paciencia:
            log(f"early stopping: {paciencia} épocas sem melhorar a EER média por fonte")
            terminou = True
        if epoca == max_epocas:
            terminou = True
        if estado_path is not None:
            salvar_torch({"config_id": config_id, "epoca": epoca, "terminou": terminou,
                          "model_state": _copia_cpu(model.state_dict()),
                          "optimizer_state": optimizer.state_dict(), "sorteador": sorteador.estado(),
                          "rng_torch": torch.get_rng_state(),
                          "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                          "rng_loader": train_loader.generator.get_state() if train_loader.generator is not None else None,
                          "vistos": vistos, "historico": historico, "melhor": melhor,
                          "sem_melhora": sem_melhora, "melhor_estado": melhor_estado,
                          "retomadas": retomadas}, estado_path)
    hist = pd.DataFrame(historico)
    hist.attrs["retomadas"] = retomadas
    return hist, float(hist["segundos"].sum()) / 60 if len(hist) else 0.0
