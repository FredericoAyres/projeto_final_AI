"""Medidas simples para a análise exploratória."""
import torch


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
