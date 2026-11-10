"""Datasets de áudio usados pelos notebooks.

Ficam num arquivo separado por causa do Windows: o DataLoader com
num_workers > 0 cria processos novos que precisam importar a classe, e uma
classe definida dentro do notebook não pode ser importada por eles (erro de
pickling).

- AudioDataset: um recorte de `crop_s` segundos (treino e validação). Aceita uma
  augmentation opcional (`augment`), aplicada só quando é passada (no treino).
- AudioInteiroDataset: o áudio inteiro em 16 kHz (avaliação por janelas deslizantes).

Áudio mais curto que 4 s (v3): `short_audio_mode` escolhe como completar.
- "repeat" (padrão, o das versões v1 e v2): repete o áudio até completar.
- "zero": mantém a fala original e completa à direita com zeros (silêncio digital).
Áudios com 4 s ou mais não mudam em nenhum dos dois modos. Os padrões (short_audio_mode="repeat",
seed_recorte=None) reproduzem exatamente o comportamento das versões anteriores.
"""
import math
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset


VARS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
MODOS_CURTO = ("repeat", "zero")


def checar_modo_curto(modo):
    if modo not in MODOS_CURTO:
        raise ValueError(f"short_audio_mode deve ser um de {MODOS_CURTO}, recebi {modo!r}")
    return modo


def completar_curto(x, n, modo="repeat"):
    """Completa a onda 1D `x` até `n` amostras se ela for mais curta; se não, devolve `x` como está.

    "repeat": repete o áudio inteiro quantas vezes for preciso e corta em `n` (comportamento v1/v2).
    "zero": copia a fala original no começo e preenche o resto, à direita, com zeros.
    Áudio vazio vira `n` zeros nos dois modos (como antes).

    Atenção (zero): os zeros viram uma região de energia mínima no log-mel (log(0 + 1e-6)) e entram
    na padronização por exemplo (média e desvio do log-mel inteiro). A largura dessa região conta a
    duração do áudio, então o zero-padding também pode virar uma pista; não é "neutro".
    """
    checar_modo_curto(modo)
    if x.numel() >= n:
        return x
    if x.numel() == 0:
        return torch.zeros(n, dtype=x.dtype)
    if modo == "repeat":
        return x.repeat(math.ceil(n / x.numel()))[:n]
    return torch.nn.functional.pad(x, (0, n - x.numel()))


def limitar_threads_dos_workers():
    """Faz cada worker do DataLoader usar 1 thread de álgebra linear. Chame ANTES de criar os DataLoaders.

    No Windows, cada worker é um processo novo que importa numpy e torch. Ao carregar, o OpenBLAS
    do numpy reserva um buffer por thread do processador (22 neste PC): cerca de 675 MB de memória
    reservada (commit) por worker, sem usar. Os workers só leem e cortam áudio (e já rodavam com
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
    short_audio_mode: como completar áudios mais curtos que o recorte ("repeat" ou "zero"; ver
        completar_curto).
    augment: função onda -> onda aplicada no fim (ex.: src.augment.Augmentador); None = nada.
    seed_recorte: None = a posição do recorte aleatório vem do gerador do torch de cada worker (v1/v2;
        depende de quantos workers há e de qual worker lê qual lote). Um inteiro = a posição é sorteada
        por um gerador próprio, função só de (seed_recorte, época, índice do áudio): não depende dos
        workers nem de nenhum outro sorteio. Assim, na mesma seed, duas condições recebem os mesmos
        trechos. A época atual é passada em `self.epoca` antes de cada época (src.train.treinar).
    """

    def __init__(self, paths, labels, sample_rates, durations,
                 crop_s=4.0, target_sr=16000, mode="random", augment=None,
                 short_audio_mode="repeat", seed_recorte=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.sample_rates = np.asarray(sample_rates, dtype=np.int64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.crop_s = crop_s
        self.target_sr = target_sr
        self.mode = mode
        self.n_out = int(round(crop_s * target_sr))
        self.augment = augment
        self.short_audio_mode = checar_modo_curto(short_audio_mode)
        self.seed_recorte = seed_recorte
        self.epoca = 0
        self._threads_set = False

    def __len__(self):
        return len(self.paths)

    def _inicio_aleatorio(self, i, maximo):
        """Início do recorte aleatório, entre 0 e `maximo` (inclusive)."""
        if self.seed_recorte is None:
            return int(torch.randint(0, maximo + 1, (1,)))
        rng = np.random.default_rng([int(self.seed_recorte), int(self.epoca), int(i)])
        return int(rng.integers(0, maximo + 1))

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
                start = self._inicio_aleatorio(i, n_total - n_crop)
            else:
                start = (n_total - n_crop) // 2
            frames = n_crop
        x, file_sr = self._read(self.paths[i], start, frames)

        x = torch.from_numpy(np.ascontiguousarray(x))  # mono
        if file_sr != self.target_sr:                  # reamostra SEMPRE que precisar
            x = AF.resample(x, file_sr, self.target_sr)  # filtro anti-aliasing embutido

        x = completar_curto(x, self.n_out, self.short_audio_mode)   # áudio curto: repete ou zeros
        x = x[: self.n_out].contiguous()
        if self.augment is not None:
            x = self.augment(x)
        return x, torch.tensor(self.labels[i])


class AudioInteiroDataset(Dataset):
    """O áudio inteiro, mono, em `target_sr`. Áudio mais curto que `min_s` é completado até `min_s`
    (repetido ou com zeros à direita, conforme `short_audio_mode`, como no AudioDataset).

    Usado na avaliação por janelas deslizantes: a GPU corta as janelas (src.train.prever_janelas).
    Os áudios têm tamanhos diferentes, então o DataLoader precisa de `collate_fn=juntar_lista`.
    """

    def __init__(self, paths, labels, target_sr=16000, min_s=4.0, short_audio_mode="repeat"):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.target_sr = target_sr
        self.n_min = int(round(min_s * target_sr))
        self.short_audio_mode = checar_modo_curto(short_audio_mode)
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
        x = completar_curto(x, self.n_min, self.short_audio_mode)  # mesmo critério do AudioDataset
        return x.contiguous(), torch.tensor(self.labels[i])


def juntar_lista(lote):
    """collate_fn para áudios de tamanhos diferentes: devolve (lista de ondas, rótulos)."""
    ondas, rotulos = zip(*lote)
    return list(ondas), torch.stack(rotulos)
