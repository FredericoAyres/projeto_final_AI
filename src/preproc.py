"""Pré-processamento da v4: uma passada por áudio que serve a tudo (extração e diagnóstico de atalhos).

Para cada áudio: mono, 16 kHz (reamostra sempre que a taxa do arquivo for outra), remoção do silêncio
nas bordas, recorte central de no máximo 8 s, SEM completar (nem repetição nem zeros). Junto, calcula
estatísticas baratas da gravação (duração, silêncio inicial e final, RMS, pico, largura de banda),
usadas na Parte F (diagnóstico de atalhos).

Este módulo NÃO importa o transformers: os workers do DataLoader (processos novos no Windows) importam
só o que o Dataset precisa (cada import de biblioteca custa memória reservada por worker).

⚠️ Fora da trilha: o corte de silêncio por limiar de energia (como `librosa.effects.trim`) e a largura de
banda pelo "rolloff" do espectro.
"""
import math

import numpy as np
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from src.dataset import AudioDataset

SR = 16_000
QUADRO, PASSO = 400, 160          # 25 ms / 10 ms a 16 kHz (os mesmos de src/analysis.py)
MIN_AMOSTRAS_MODELO = 400         # o 1º filtro do XLS-R tem 400 amostras (25 ms); abaixo disso não há quadro

# colunas numéricas devolvidas por preparar_clipe (na ordem do vetor de estatísticas)
COLS_ESTAT = ["dur_original_s", "dur_apos_corte_s", "dur_entrada_s", "silencio_inicio_s", "silencio_fim_s",
              "corte_aplicado", "corte_desfeito", "rms_db", "pico_db", "banda_hz", "sr_arquivo", "erro_leitura"]
# as que entram no classificador de atalhos (Parte F); as outras são de controle
COLS_ATALHO = ["dur_original_s", "silencio_inicio_s", "silencio_fim_s", "rms_db", "pico_db", "banda_hz"]


def ler_mono_16k(path):
    """Áudio inteiro, mono, float32, em 16 kHz (reamostrado com o filtro anti-aliasing do torchaudio).

    A leitura usa o soundfile e, se o decodificador FLAC dele falhar, o miniaudio (src/dataset.py).
    """
    x, sr = AudioDataset._read(path, 0, -1)
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    if sr != SR:
        x = AF.resample(x, sr, SR)
    return x, int(sr)


def energia_quadros_db(x, quadro=QUADRO, passo=PASSO):
    """Energia (dB) de cada quadro de `quadro` amostras com passo `passo` (sem centralizar)."""
    if x.numel() < quadro:
        q = torch.nn.functional.pad(x, (0, quadro - x.numel())).unsqueeze(0)
    else:
        q = x.unfold(0, quadro, passo)
    return 10 * torch.log10(q.double().pow(2).mean(dim=-1) + 1e-20)


def bordas_sem_silencio(x, top_db=35.0, quadro=QUADRO, passo=PASSO):
    """(início, fim) em amostras do trecho entre o 1º e o último quadro com energia acima de
    (energia do quadro mais forte − `top_db`). Mesma ideia de `librosa.effects.trim` (referência = máximo).

    Áudio todo em silêncio digital (energia ~0): devolve o áudio inteiro.
    """
    n = x.numel()
    e = energia_quadros_db(x, quadro, passo)
    if n == 0 or float(e.max()) < -150:
        return 0, n
    ativos = torch.nonzero(e > e.max() - top_db).flatten()
    ini = int(ativos[0]) * passo
    fim = min(n, int(ativos[-1]) * passo + quadro)
    return ini, fim


def banda_rolloff_hz(x, frac=0.99, n_fft=512):
    """Frequência (Hz) abaixo da qual está `frac` da energia do espectro médio do áudio (16 kHz).

    Medida simples de largura de banda: áudio de telefone (8 kHz de taxa original) fica perto de 4 kHz;
    áudio de banda cheia fica perto de 7 a 8 kHz.
    """
    if x.numel() < n_fft:
        x = torch.nn.functional.pad(x, (0, n_fft - x.numel()))
    esp = torch.stft(x, n_fft, hop_length=n_fft // 2, window=torch.hann_window(n_fft), return_complex=True)
    p = esp.abs().pow(2).mean(dim=-1).double()            # (n_fft/2 + 1,)
    acum = torch.cumsum(p, 0) / max(float(p.sum()), 1e-30)
    k = int(torch.searchsorted(acum, torch.tensor([frac], dtype=acum.dtype)).item())
    return float(min(k, p.numel() - 1) * SR / n_fft)


def preparar_clipe(x, sr_arquivo, cortar_silencio=True, top_db=35.0, min_s=0.5, max_s=8.0):
    """Do áudio inteiro em 16 kHz (`x`) ao clipe que entra no modelo, mais as estatísticas.

    1. corte do silêncio nas bordas (se `cortar_silencio`); se sobrar menos que `min_s`, usa o áudio sem
       corte e marca `corte_desfeito = 1`;
    2. recorte central de no máximo `max_s` (sem completar: áudio mais curto fica com o próprio tamanho;
       só abaixo de 400 amostras, 25 ms, completa com zeros, porque o modelo não tem quadro antes disso).
    Estatísticas da GRAVAÇÃO (RMS, pico, banda) são do áudio original inteiro em 16 kHz, antes do corte.
    Silêncio inicial e final são medidos mesmo quando o corte é desligado ou desfeito.
    """
    n = x.numel()
    ini, fim = bordas_sem_silencio(x, top_db)
    desfeito = (fim - ini) < min_s * SR
    aplicado = cortar_silencio and not desfeito
    y = x[ini:fim] if aplicado else x
    n_apos = y.numel()
    n_max = int(round(max_s * SR))
    if n_apos > n_max:
        a = (n_apos - n_max) // 2
        y = y[a:a + n_max]
    if y.numel() < MIN_AMOSTRAS_MODELO:
        y = torch.nn.functional.pad(y, (0, MIN_AMOSTRAS_MODELO - y.numel()))
    rms = float(x.double().pow(2).mean().sqrt()) if n else 0.0
    pico = float(x.abs().max()) if n else 0.0
    est = {"dur_original_s": n / SR, "dur_apos_corte_s": n_apos / SR, "dur_entrada_s": y.numel() / SR,
           "silencio_inicio_s": ini / SR, "silencio_fim_s": (n - fim) / SR,
           "corte_aplicado": float(aplicado), "corte_desfeito": float(cortar_silencio and desfeito),
           "rms_db": 20 * math.log10(rms + 1e-10), "pico_db": 20 * math.log10(pico + 1e-10),
           "banda_hz": banda_rolloff_hz(x), "sr_arquivo": float(sr_arquivo), "erro_leitura": 0.0}
    return y.contiguous(), est


class ClipesV4(Dataset):
    """Devolve (clipe 1D em 16 kHz, vetor de estatísticas na ordem de COLS_ESTAT, posição) de cada áudio.

    `posicoes` diz em que linha do shard cada áudio vai ser gravado (a ordem de leitura pode ser outra:
    os áudios são lidos ordenados por duração, para os lotes terem tamanhos parecidos e pouco padding).
    `cortar_silencio`: um bool para todos ou um por áudio.
    Se um arquivo não puder ser lido, devolve 25 ms de zeros e marca `erro_leitura = 1` (não derruba a
    extração inteira; o áudio fica identificado no meta do shard e é excluído na análise).
    """

    def __init__(self, paths, posicoes, cortar_silencio=True, top_db=35.0, min_s=0.5, max_s=8.0):
        self.paths = list(paths)
        self.posicoes = np.asarray(posicoes, dtype=np.int64)
        self.cortar = np.broadcast_to(np.asarray(cortar_silencio, dtype=bool), (len(self.paths),)).copy()
        self.kw = dict(top_db=top_db, min_s=min_s, max_s=max_s)
        self._threads_set = False

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if not self._threads_set:
            torch.set_num_threads(1)
            self._threads_set = True
        try:
            x, sr = ler_mono_16k(self.paths[i])
            y, est = preparar_clipe(x, sr, cortar_silencio=bool(self.cortar[i]), **self.kw)
        except Exception:                      # noqa: BLE001 (registrado em erro_leitura)
            y = torch.zeros(MIN_AMOSTRAS_MODELO)
            est = {c: np.nan for c in COLS_ESTAT}
            est["erro_leitura"] = 1.0
        return y, np.array([est[c] for c in COLS_ESTAT], dtype=np.float64), int(self.posicoes[i])


def juntar_clipes(lote):
    """collate_fn: (lista de clipes, estatísticas (lote, n_estat), posições (lote,))."""
    clipes, est, pos = zip(*lote)
    return list(clipes), np.stack(est), np.asarray(pos, dtype=np.int64)
