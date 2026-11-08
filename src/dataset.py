"""Dataset de áudio usado pelo notebook.

Fica num arquivo separado só por causa do Windows: o DataLoader com
num_workers > 0 cria processos novos que precisam importar a classe, e uma
classe definida dentro do notebook não pode ser importada por eles (erro de
pickling). Por isso, só a classe Dataset mora aqui.
"""
import math

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Lê um trecho de `crop_s` segundos de cada áudio e devolve mono em `target_sr`.

    mode="random": trecho em posição aleatória (treino).
    mode="center": trecho central (validação e teste, sempre o mesmo).
    Áudios mais curtos que o recorte são repetidos até completar o tamanho.
    """

    def __init__(self, paths, labels, sample_rates, durations,
                 crop_s=4.0, target_sr=16000, mode="random"):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.sample_rates = np.asarray(sample_rates, dtype=np.int64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.crop_s = crop_s
        self.target_sr = target_sr
        self.mode = mode
        self.n_out = int(round(crop_s * target_sr))
        self._threads_set = False

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _read(path, start, frames):
        """Lê com soundfile; se o decodificador FLAC dele falhar, usa o miniaudio.

        Cerca de 40% dos FLAC do ASVspoof 2021 DF dão "unknown error in flac
        decoder" no libsndfile 1.2.2, mas são lidos pelo miniaudio, que
        devolve exatamente as mesmas amostras nos arquivos que os dois leem.
        """
        try:
            x, sr = sf.read(path, start=start, frames=frames,
                            dtype="float32", always_2d=True)
            return x.mean(axis=1), sr
        except sf.LibsndfileError:
            import miniaudio
            d = miniaudio.flac_read_file_f32(path)
            x = np.asarray(d.samples, dtype=np.float32)
            if d.nchannels > 1:
                x = x.reshape(-1, d.nchannels).mean(axis=1)
            stop = len(x) if frames < 0 else start + frames
            return x[start:stop], d.sample_rate

    def __getitem__(self, i):
        sr = int(self.sample_rates[i])
        n_total = int(self.durations[i] * sr)
        n_crop = int(math.ceil(self.crop_s * sr))

        if not self._threads_set:
            # Cada processo de leitura usa 1 thread; senão, 8 workers x 22 threads
            # disputam a CPU e a leitura fica mais lenta.
            torch.set_num_threads(1)
            self._threads_set = True

        # Lê só o trecho necessário (evita carregar áudios longos inteiros).
        start, frames = 0, -1
        if n_total > n_crop:
            if self.mode == "random":
                start = int(torch.randint(0, n_total - n_crop + 1, (1,)))
            else:
                start = (n_total - n_crop) // 2
            frames = n_crop
        x, file_sr = self._read(self.paths[i], start, frames)

        x = torch.from_numpy(np.ascontiguousarray(x))  # mono
        if file_sr != self.target_sr:                  # reamostra SEMPRE que precisar
            x = AF.resample(x, file_sr, self.target_sr)  # filtro anti-aliasing embutido

        if x.numel() == 0:
            x = torch.zeros(1)
        if x.numel() < self.n_out:                     # áudio curto: repete
            x = x.repeat(math.ceil(self.n_out / x.numel()))
        x = x[: self.n_out].contiguous()
        return x, torch.tensor(self.labels[i])
