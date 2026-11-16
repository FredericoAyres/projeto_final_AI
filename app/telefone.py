#!/usr/bin/env python3
"""Filtro "de telefone" para os exemplos do app: simula o canal G.711 (telefone fixo).

Os mesmos passos de golpe_dataset/voz/03_processar.py (canal "g711"), refeitos em Python porque não há ffmpeg aqui:
  passa-alta 300 Hz + passa-baixa 3.400 Hz (biquads do torchaudio, como os filtros do ffmpeg)
  -> reamostragem para 8 kHz -> μ-law de 8 bits (codifica e decodifica).
⚠️ Fora da trilha (processamento de sinal).

    python app/telefone.py entrada.wav saida.wav
"""
import sys

import soundfile as sf
import torch
import torchaudio.functional as AF

SR_TELEFONE = 8000


def aplicar(entrada, saida):
    x, sr = sf.read(str(entrada), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))                       # mono
    x = AF.highpass_biquad(x, sr, 300.0)
    x = AF.lowpass_biquad(x, sr, 3400.0)
    x = AF.resample(x, sr, SR_TELEFONE).clamp(-1, 1)
    x = AF.mu_law_decoding(AF.mu_law_encoding(x, 256), 256)
    sf.write(str(saida), x.numpy(), SR_TELEFONE, subtype="PCM_16")
    return sr


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(f"{sys.argv[1]}: {aplicar(sys.argv[1], sys.argv[2])} Hz -> {SR_TELEFONE} Hz em {sys.argv[2]}")
