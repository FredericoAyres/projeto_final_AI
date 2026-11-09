"""Datasets de áudio usados pelos notebooks.

Ficam num arquivo separado por causa do Windows: o DataLoader com
num_workers > 0 cria processos novos que precisam importar a classe, e uma
classe definida dentro do notebook não pode ser importada por eles (erro de
pickling).

- AudioDataset: um recorte de `crop_s` segundos (treino e validação). Aceita uma
  augmentation opcional (`augment`), aplicada só quando é passada (no treino).
- AudioInteiroDataset: o áudio inteiro em 16 kHz (avaliação por janelas deslizantes).

O notebook v1 (notebooks/versions/v1_baseline.ipynb) usa AudioDataset sem `augment`,
e o comportamento dele não mudou.
"""
import math
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset


VARS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def limitar_threads_dos_workers():
    """Faz cada worker do DataLoader usar 1 thread de álgebra linear. Chame ANTES de criar os DataLoaders.

    No Windows, cada worker é um processo novo que importa numpy e torch. Ao carregar, o OpenBLAS
    do numpy reserva um buffer por thread do processador (22 neste PC): cerca de 675 MB de memória
    reservada (commit) por worker, sem usar. Os workers só leem e cortam áudio (e já rodam com
    torch.set_num_threads(1)), então 1 thread basta e o resultado é o mesmo.

    O worker herda as variáveis de ambiente do processo principal no momento em que é criado. Isso
    não muda nada no processo principal, que já carregou o numpy.
    """
    for v in VARS_THREADS:
        os.environ[v] = "1"


class AudioDataset(Dataset):
    """Lê um trecho de `crop_s` segundos de cada áudio e devolve mono em `target_sr`.

    mode="random": trecho em posição aleatória (treino).
    mode="center": trecho central (validação e teste, sempre o mesmo).
    Áudios mais curtos que o recorte são repetidos até completar o tamanho.
    augment: função onda -> onda aplicada no fim (ex.: src.augment.Augmentador); None = nada.
    """

    def __init__(self, paths, labels, sample_rates, durations,
                 crop_s=4.0, target_sr=16000, mode="random", augment=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.sample_rates = np.asarray(sample_rates, dtype=np.int64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.crop_s = crop_s
        self.target_sr = target_sr
        self.mode = mode
        self.n_out = int(round(crop_s * target_sr))
        self.augment = augment
        self._threads_set = False

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _read(path, start, frames):
        """Lê com soundfile; se o decodificador FLAC dele falhar, usa o miniaudio.

        64.901 dos 152.955 FLAC do ASVspoof 2021 DF (42,4%) dão erro no
        decodificador do libsndfile 1.2.2, mas são lidos pelo miniaudio, que
        devolve exatamente as mesmas amostras nos arquivos que os dois leem
        (scripts/check_flac_decoding.py, docs/audit_report.md seção 3.1).
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
        if self.augment is not None:
            x = self.augment(x)
        return x, torch.tensor(self.labels[i])


class AudioInteiroDataset(Dataset):
    """O áudio inteiro, mono, em `target_sr`. Áudio mais curto que `min_s` é repetido até `min_s`.

    Usado na avaliação por janelas deslizantes: a GPU corta as janelas (src.train.prever_janelas).
    Os áudios têm tamanhos diferentes, então o DataLoader precisa de `collate_fn=juntar_lista`.
    """

    def __init__(self, paths, labels, target_sr=16000, min_s=4.0):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.target_sr = target_sr
        self.n_min = int(round(min_s * target_sr))
        self._threads_set = False

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if not self._threads_set:
            torch.set_num_threads(1)
            self._threads_set = True
        x, sr = AudioDataset._read(self.paths[i], 0, -1)
        x = torch.from_numpy(np.ascontiguousarray(x))
        if sr != self.target_sr:
            x = AF.resample(x, sr, self.target_sr)
        if x.numel() == 0:
            x = torch.zeros(1)
        if x.numel() < self.n_min:                     # mesmo critério do AudioDataset: repete
            x = x.repeat(math.ceil(self.n_min / x.numel()))[: self.n_min]
        return x.contiguous(), torch.tensor(self.labels[i])


def juntar_lista(lote):
    """collate_fn para áudios de tamanhos diferentes: devolve (lista de ondas, rótulos)."""
    ondas, rotulos = zip(*lote)
    return list(ondas), torch.stack(rotulos)
