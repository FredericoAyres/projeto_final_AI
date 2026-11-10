"""Medidas simples para a análise exploratória e o diagnóstico por duração."""
import numpy as np
import pandas as pd
import torch

from src.metrics import metricas

# Faixas de duração ORIGINAL (a do arquivo, antes de repetir ou completar até 4 s), pré-registradas na v3.
FAIXAS_DURACAO = ["< 2 s", "2 a < 4 s", "≥ 4 s"]


def faixa_dinamica_db(ondas, quadro=400, passo=160, alto=0.95, baixo=0.10):
    """Diferença (dB) entre a energia dos quadros mais fortes e a dos mais fracos de cada áudio.

    `ondas`: (lote, amostras). Quadros de 25 ms com passo de 10 ms (a 16 kHz). Devolve (lote,).
    Gravação limpa: as pausas são quase silêncio, então a diferença é grande (40 dB ou mais).
    Gravação com ruído de fundo: as pausas têm ruído, então a diferença é pequena (20 dB ou menos).
    É uma estimativa grosseira da relação sinal-ruído, sem precisar saber onde está a fala.
    """
    q = ondas.unfold(-1, quadro, passo)
    e = 10 * torch.log10(q.pow(2).mean(dim=-1) + 1e-10)
    return torch.quantile(e, alto, dim=-1) - torch.quantile(e, baixo, dim=-1)


def faixa_duracao(duracao_s):
    """Faixa de duração original: "< 2 s", "2 a < 4 s" ou "≥ 4 s" (categórica, nessa ordem)."""
    return pd.cut(np.asarray(duracao_s, dtype=float), [-np.inf, 2.0, 4.0, np.inf], right=False,
                  labels=FAIXAS_DURACAO)


def metricas_por_duracao(notas, coluna, limiar):
    """Métricas de `notas` (DataFrame com label, duration_s e a nota em `coluna`) por faixa de duração.

    Sempre: tamanho do grupo, nº de reais e de fakes, % de fake. AUC e EER só quando a faixa tem as
    duas classes (senão ficam vazias). Falso alarme e recall de fake usam o `limiar` dado (fixado na
    validação), então dependem dele.
    """
    d = notas.assign(faixa=faixa_duracao(notas.duration_s))
    linhas = []
    for faixa in FAIXAS_DURACAO:
        g = d[d.faixa == faixa]
        if len(g) == 0:
            linhas.append({"faixa": faixa, "n": 0})
            continue
        m = metricas(g.label.values, g[coluna].values, limiar)
        linhas.append({"faixa": faixa, "n": len(g), "n real": m["n real"], "n fake": m["n fake"],
                       "% fake": 100 * m["n fake"] / len(g), "AUC": m["AUC"], "EER": m["EER"],
                       "falso alarme": m["falso alarme"], "recall fake": m["recall fake"]})
    return pd.DataFrame(linhas)
