"""Amostragem balanceada por época (a mesma regra do baseline v1, seção 6.1 do notebook).

1. Cada época tem `por_classe` reais e `por_classe` falsos (50/50).
2. Dentro de cada classe, as fontes recebem fatias iguais (teto por fonte); a sobra de uma fonte
   pequena vai para as outras.
3. Dentro de cada fonte, a fatia é dividida igualmente entre os geradores (teto por gerador).
4. Cada grupo (classe, fonte, gerador) tem uma fila embaralhada; cada época consome os próximos
   itens dela, e a fila só é reembaralhada quando acaba (o modelo passa pela maior parte do pool).
"""
import numpy as np
from torch.utils.data import Sampler


def montar_grupos(pool):
    """{(label, fonte, gerador): índices} do DataFrame `pool` (índice 0..n-1)."""
    gen = pool["generator"].fillna("(real)")
    return {k: np.asarray(v) for k, v in pool.assign(gen_key=gen)
            .groupby(["label", "source_dataset", "gen_key"]).indices.items()}


def dividir_igual(total, tamanhos):
    """Divide `total` igualmente entre os grupos, sem passar do tamanho de cada um; a sobra vai para os outros."""
    cotas = {k: 0 for k in tamanhos}
    ativos = [k for k, n in tamanhos.items() if n > 0]
    resta = total
    while resta > 0 and ativos:
        fatia = max(resta // len(ativos), 1)
        prox = []
        for k in ativos:
            dar = min(fatia, tamanhos[k] - cotas[k], resta)
            cotas[k] += dar
            resta -= dar
            if cotas[k] < tamanhos[k]:
                prox.append(k)
            if resta == 0:
                break
        ativos = prox
    return cotas


def calcular_cotas(grupos, por_classe):
    """Quantos áudios de cada grupo entram por época (teto por fonte e depois por gerador)."""
    cotas = {}
    for lab in (0, 1):
        fontes = {}
        for (l, s, _), v in grupos.items():
            if l == lab:
                fontes[s] = fontes.get(s, 0) + len(v)
        for src, cota_src in dividir_igual(por_classe, fontes).items():
            gens = {g: len(v) for (l, s, g), v in grupos.items() if l == lab and s == src}
            for g, c in dividir_igual(cota_src, gens).items():
                cotas[(lab, src, g)] = c
    return cotas


class SorteadorPorEpoca:
    """Cada grupo tem uma fila embaralhada; cada época consome os próximos itens dela."""

    def __init__(self, grupos, seed):
        self.rng = np.random.default_rng(seed)
        self.filas = {k: self.rng.permutation(v) for k, v in grupos.items()}
        self.pos = {k: 0 for k in grupos}

    def tirar(self, k, n):
        saida = []
        while n > 0:
            fila, p = self.filas[k], self.pos[k]
            pega = min(n, len(fila) - p)
            saida.append(fila[p:p + pega])
            p, n = p + pega, n - pega
            if p == len(fila):                      # fila acabou: reembaralha e recomeça
                self.filas[k], p = self.rng.permutation(fila), 0
            self.pos[k] = p
        return np.concatenate(saida)

    def epoca(self, cotas):
        idx = np.concatenate([self.tirar(k, c) for k, c in cotas.items() if c > 0])
        return self.rng.permutation(idx)            # embaralha a época (Aula 6)

    def estado(self):
        """Tudo o que é preciso para continuar o sorteio de onde parou (retomar um treino interrompido)."""
        return {"rng": self.rng.bit_generator.state, "filas": {k: v.copy() for k, v in self.filas.items()},
                "pos": dict(self.pos)}

    def restaurar(self, estado):
        assert set(estado["filas"]) == set(self.filas), "os grupos do treino mudaram desde o estado salvo"
        self.rng.bit_generator.state = estado["rng"]
        self.filas = {k: np.asarray(v) for k, v in estado["filas"].items()}
        self.pos = dict(estado["pos"])


class IndicesDaEpoca(Sampler):
    """Entrega ao DataLoader os índices sorteados para a época atual."""

    def __init__(self):
        self.indices = np.array([], dtype=int)

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)
