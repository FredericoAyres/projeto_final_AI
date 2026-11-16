"""Ramo do CONTEXTO do app: o que a pessoa fala é típico de golpe?

Dois modos, com a MESMA saída (o contrato de docs/CONTEXTO_GOLPE.md, seção 4: status, transcricao,
risco_conteudo, sinais [{tipo, evidencia}], resumo...):

  "groq"     o módulo dos colegas, src/contexto_golpe.py: Whisper (transcrição, idioma "auto", como congelado em
             results/diagnostico_whisper_congelado.json) + LLM (classificação), pela API do Groq.
             Precisa de GROQ_API_KEY no ambiente e ENVIA O ÁUDIO para um serviço externo: só use com o
             consentimento de quem gravou (docs/CONTEXTO_GOLPE.md, seções 5a e 7).
  "offline"  reserva para a apresentação sem internet/sem chave: usa a transcrição digitada em app/exemplos.csv e
             uma lista de palavras-chave com regras escolhidas à mão (NÃO medidas). Não substitui o módulo dos
             colegas; a tela avisa quando este modo foi usado.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODOS = ("offline", "groq")
RISCOS = ("nenhum", "baixo", "medio", "alto")       # + "indeterminado" quando algo falha

# ---------------------------------------------------------------------------------------- modo offline
# tipo de sinal (os mesmos nomes do módulo dos colegas) -> expressões em português e em inglês
SINAIS_OFFLINE = {
    "pedido_dinheiro_ou_pix": [r"\bpix\b", r"transfer\w*", r"dep[óo]sit\w*", r"\bpag(ar|amento|ue)\b",
                               r"send me (the |some )?money", r"\bpayment\b", r"help me cover", r"\blend me\b"],
    "senha_ou_codigo": [r"c[óo]digo", r"senha", r"\btoken\b", r"\bcode\b", r"password"],
    "urgencia": [r"urgente", r"agora mesmo", r"imediatamente", r"hoje ainda", r"\burgent\w*", r"right now",
                 r"due today", r"next few minutes", r"as soon as"],
    "falsa_identidade": [r"central de seguran[çc]a", r"\bdo banco\b", r"\bgerente\b", r"from the bank"],
    "ameaca_ou_medo": [r"bloquead\w*", r"compra suspeita", r"fraude", r"acidente", r"blocked", r"declined",
                       r"stopped working", r"trouble making"],
    "troca_de_numero_ou_contato": [r"n[úu]mero novo", r"troquei de n[úu]mero", r"different number", r"new number",
                                   r"changed my (phone )?number"],
    "pedido_de_segredo": [r"n[ãa]o conta pra ningu[ée]m", r"n[ãa]o comenta", r"don'?t tell"],
}


def _offline(texto):
    sinais = []
    for tipo, padroes in SINAIS_OFFLINE.items():
        for p in padroes:
            m = re.search(p, texto, flags=re.IGNORECASE)
            if m:
                sinais.append({"tipo": tipo, "evidencia": m.group(0)})
                break                                   # uma evidência por tipo basta
    n = len(sinais)
    risco = RISCOS[min(n, 3)]                           # 0 tipos: nenhum; 1: baixo; 2: médio; 3+: alto (à mão)
    return {"status": "ok", "transcricao": texto, "idioma": None, "risco_conteudo": risco, "sinais": sinais,
            "resumo": "", "aviso": "Isto é um alerta, não um veredito.", "erro": None,
            "observacoes": ["modo offline: transcrição digitada e palavras-chave; não é o módulo dos colegas"]}


# ---------------------------------------------------------------------------------------- entrada única
def analisar(caminho, modo="offline", transcricao_digitada=None):
    """Devolve o JSON do contrato + "modo". Em qualquer falha, risco "indeterminado" (nunca "nenhum")."""
    if modo not in MODOS:
        raise ValueError(f"modo deve ser um de {MODOS}")
    if modo == "groq":
        if not os.environ.get("GROQ_API_KEY", "").strip():
            raise RuntimeError("modo groq pedido, mas GROQ_API_KEY não está no ambiente (docs/CONTEXTO_GOLPE.md, seção 2)")
        from src.contexto_golpe import analisar_audio
        r = analisar_audio(str(caminho), idioma="auto")
    else:
        if not transcricao_digitada:
            raise ValueError(f"modo offline precisa da transcrição digitada em app/exemplos.csv para {caminho}")
        r = _offline(transcricao_digitada)
    return {**r, "modo": modo}
