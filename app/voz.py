"""Ramo da VOZ do app: o áudio parece gerado por IA?

Usa o sistema principal do projeto (T3 da v7) pelo gabarito do Frederico, scripts/inferencia_referencia.py
(guia: docs/GUIA_PARA_O_SITE.md). Nenhuma conta é reescrita aqui: só chamamos as funções de lá.

Para trocar o detector de voz no futuro (outra versão, modelo leve etc.), basta manter a saída de
DetectorVoz.analisar: um dicionário com as chaves "nota", "corte", "parece_ia" e "chance_ia".
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import inferencia_referencia as ir  # noqa: E402


class DetectorVoz:
    def __init__(self, variante="v1", aparelho=None):
        """variante v1 (padrão, roda no processador, nota idêntica à T3) ou v2 (fp16, placa de vídeo).
        A v3 (int8) muda a nota e o corte da T3 não vale nela, então não é aceita aqui."""
        if variante not in ("v1", "v2"):
            raise ValueError("use v1 ou v2: na v3 (int8) o corte da T3 não foi validado")
        self.aparelho = ir.escolher_aparelho(variante, aparelho)
        self.modelo, self.dtype = ir.carregar(variante, self.aparelho)
        self.pesos = ir.pesos_t3()
        self.corte = ir.corte_m()          # regra M da validação da T3 (results/v7_t3_modelo.json)

    def analisar(self, caminho):
        z, _, est = ir.nota_de_arquivo(caminho, self.modelo, self.aparelho, self.dtype, self.pesos)
        return {
            "modelo": "T3 (v7, sistema principal)",
            "nota": z,                                  # logit; maior = mais parecido com IA
            "corte": self.corte,
            "parece_ia": bool(z >= self.corte),      # regra congelada: IA se T3 >= M
            # nota passada pela sigmoide (a probabilidade da regressão logística). A calibração dessa
            # porcentagem não foi conferida: no app ela serve só para desenhar a barra.
            "chance_ia": 1.0 / (1.0 + math.exp(-z)),
            "duracao_s": est["dur_original_s"],
        }
