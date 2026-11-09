"""Augmentation de áudio (só no treino), sem arquivos externos.

Cada operação recebe uma onda mono em 16 kHz (tensor 1D, valores em [-1, 1]) e devolve outra do
mesmo tamanho. O `Augmentador` sorteia quais operações aplicar em cada exemplo, com probabilidades
FIXAS que NÃO dependem do rótulo: reais e falsos passam pelas mesmas transformações com a mesma
chance. Se só os falsos (ou só os reais) fossem degradados, a própria degradação viraria o atalho.

Usa só o gerador aleatório do torch. O DataLoader dá uma seed diferente para cada processo de
leitura a partir do `generator` dele, então o sorteio é reproduzível.
"""
import math

import torch
import torchaudio.functional as AF

EPS = 1e-8


def rms_db(x):
    """Volume RMS em dB (relativo a 1,0 = fundo de escala)."""
    return 20 * math.log10(float(x.pow(2).mean().sqrt()) + EPS)


def evitar_clip(x, pico=0.99):
    """Se o pico passar de `pico`, reduz o volume inteiro (não corta a onda)."""
    m = float(x.abs().max())
    return x * (pico / m) if m > pico else x


def normalizar_rms(x, alvo_db=-26.0):
    """Leva o volume RMS para `alvo_db` (áudio mudo fica como está)."""
    atual = float(x.pow(2).mean().sqrt())
    if atual < 1e-6:
        return x
    return evitar_clip(x * (10 ** (alvo_db / 20) / atual))


def aplicar_ganho(x, ganho_db):
    """Multiplica a onda por um ganho em dB."""
    return evitar_clip(x * 10 ** (ganho_db / 20))


def ruido_colorido(n, expoente):
    """Ruído com espectro de potência ~ 1/f^expoente: 0 = branco, 1 = rosa, 2 = marrom."""
    branco = torch.randn(n)
    if expoente == 0:
        return branco
    espec = torch.fft.rfft(branco)
    f = torch.arange(espec.numel(), dtype=torch.float32)
    f[0] = 1.0                                   # evita dividir por zero na frequência 0
    r = torch.fft.irfft(espec / f.pow(expoente / 2), n=n)
    return r / (r.std() + EPS)


def adicionar_ruido(x, snr_db, expoente=0):
    """Soma ruído com a relação sinal-ruído `snr_db` (em dB) em relação ao próprio áudio."""
    p_sinal = float(x.pow(2).mean())
    if p_sinal < 1e-12:
        return x
    r = ruido_colorido(x.numel(), expoente)
    r = r * math.sqrt(p_sinal / (10 ** (snr_db / 10)) / float(r.pow(2).mean() + EPS))
    return evitar_clip(x + r)


def limitar_banda(x, sr=16000, sr_baixo=8000):
    """Reamostra para `sr_baixo` e volta: corta tudo acima de sr_baixo/2 (4 kHz, como telefone)."""
    return AF.resample(AF.resample(x, sr, sr_baixo), sr_baixo, sr)[: x.numel()]


def mu_law(x, canais=256):
    """Ida e volta pelo codec mu-law de 8 bits (o da telefonia digital): adiciona ruído de quantização."""
    x = x.clamp(-1, 1)
    return AF.mu_law_decoding(AF.mu_law_encoding(x, canais), canais)


class Augmentador:
    """Aplica, em ordem, cada operação com a sua probabilidade (independentes entre si).

    Ordem: normalização RMS -> ganho -> ruído -> limitação de banda -> mu-law
    (o volume vem antes, depois a "sala" com ruído, depois o "canal" de telefone).
    Probabilidade 0 desliga a operação. As mesmas probabilidades valem para real e fake.
    """

    def __init__(self, sr=16000, p_rms=0.0, rms_alvo_db=-26.0, p_ganho=0.0, ganho_db=(-12.0, 6.0),
                 p_ruido=0.0, snr_db=(5.0, 30.0), expoentes_ruido=(0, 1, 2), p_banda=0.0,
                 sr_banda=8000, p_mulaw=0.0, mulaw_canais=256):
        self.sr = sr
        self.p_rms, self.rms_alvo_db = p_rms, rms_alvo_db
        self.p_ganho, self.ganho_db = p_ganho, ganho_db
        self.p_ruido, self.snr_db, self.expoentes_ruido = p_ruido, snr_db, tuple(expoentes_ruido)
        self.p_banda, self.sr_banda = p_banda, sr_banda
        self.p_mulaw, self.mulaw_canais = p_mulaw, mulaw_canais

    @staticmethod
    def _sorteia(p):
        return p > 0 and float(torch.rand(())) < p

    @staticmethod
    def _uniforme(a, b):
        return a + (b - a) * float(torch.rand(()))

    def __call__(self, x):
        if self._sorteia(self.p_rms):
            x = normalizar_rms(x, self.rms_alvo_db)
        if self._sorteia(self.p_ganho):
            x = aplicar_ganho(x, self._uniforme(*self.ganho_db))
        if self._sorteia(self.p_ruido):
            e = self.expoentes_ruido[int(torch.randint(len(self.expoentes_ruido), ()))]
            x = adicionar_ruido(x, self._uniforme(*self.snr_db), e)
        if self._sorteia(self.p_banda):
            x = limitar_banda(x, self.sr, self.sr_banda)
        if self._sorteia(self.p_mulaw):
            x = mu_law(x, self.mulaw_canais)
        return x.contiguous()

    def __repr__(self):
        ativos = {k: v for k, v in vars(self).items() if k.startswith("p_") and v > 0}
        return f"Augmentador({ativos})" if ativos else "Augmentador(nenhuma operação)"
