"""Entrada (log-mel padronizado) e a CNN pequena do baseline (igual à do notebook v1)."""
import torch
import torch.nn as nn
import torchaudio


class LogMelPadronizado(nn.Module):
    """(lote, amostras) -> (lote, 1, n_mels, quadros): log-mel com média 0 e desvio 1 por exemplo.

    A padronização por exemplo remove o volume global: um ganho na onda vira uma constante
    somada ao log-mel, que a subtração da média tira.
    """

    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                        win_length=win_length, hop_length=hop_length,
                                                        n_mels=n_mels)

    def forward(self, ondas):
        m = torch.log(self.mel(ondas) + 1e-6)
        m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
        return m.unsqueeze(1)


class CNNPequena(nn.Module):
    """4 blocos Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2 (16, 32, 64, 128 filtros) -> Dropout -> Linear."""

    def __init__(self, n_mels, n_frames, dropout):
        super().__init__()
        camadas, c_in = [], 1
        for c_out in (16, 32, 64, 128):
            camadas += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out),
                        nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*camadas)
        n_flat = 128 * (n_mels // 16) * (n_frames // 16)
        self.saida = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n_flat, 1))

    def forward(self, x):
        return self.saida(self.features(x)).squeeze(1)    # logit (sem sigmoide)
