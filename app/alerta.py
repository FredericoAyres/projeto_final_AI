"""Junta os dois ramos (voz e contexto) num alerta só.

Regra = a congelada pelos colegas (docs/RESULTADO_COMBINADO_CORRIGIDO.md; proposta em docs/CONTEXTO_GOLPE.md, seção 6):
  voz = IA          se a nota da T3 >= corte M da validação
  conteúdo suspeito se risco_conteudo é "medio" ou "alto"
  alto = os dois; atenção = um; baixo = nenhum.

                          conteúdo não suspeito      conteúdo suspeito
  voz sem sinais de IA    baixo                      atenção
  voz com sinais de IA    atenção                    ALTO

Conteúdo "indeterminado" (falha, sem fala) NUNCA conta como "não suspeito" (seção 6 do mesmo documento):
  voz com sinais de IA  -> atenção (o LLM nunca rebaixa a voz)
  voz sem sinais de IA  -> "não avaliado" (não dá para dizer "baixo" sem olhar o conteúdo)
"""
SUSPEITO = ("medio", "alto")

TEXTOS = {
    ("ia", "suspeito"): ("alto", "Alerta de golpe",
                         "A voz tem sinais de ser gerada por IA e o pedido é típico de golpe. Não faça pagamentos nem "
                         "passe códigos. Desligue e ligue você mesmo para o número oficial."),
    ("real", "suspeito"): ("atencao", "Conteúdo suspeito",
                           "A voz não tem sinais de IA, mas o pedido é típico de golpe (golpistas também usam a própria "
                           "voz). Confirme por outro canal antes de fazer qualquer coisa."),
    ("ia", "normal"): ("atencao", "Voz possivelmente gerada por IA",
                       "O conteúdo não tem sinais de golpe, mas a voz tem sinais de ser sintética. Desconfie se vier "
                       "um pedido de dinheiro ou de dados depois."),
    ("real", "normal"): ("baixo", "Chance de golpe baixa",
                         "A voz não tem sinais de IA e o conteúdo não tem sinais típicos de golpe."),
    ("ia", "indeterminado"): ("atencao", "Voz possivelmente gerada por IA",
                              "A voz tem sinais de ser sintética e o conteúdo não pôde ser avaliado. Desconfie de "
                              "qualquer pedido de dinheiro ou de dados."),
    ("real", "indeterminado"): ("nao_avaliado", "Conteúdo não avaliado",
                                "A voz não tem sinais de IA, mas não foi possível avaliar o que foi dito. Isso não "
                                "quer dizer que não há risco."),
}


def classe_conteudo(risco):
    if risco in SUSPEITO:
        return "suspeito"
    if risco in ("nenhum", "baixo"):
        return "normal"
    return "indeterminado"


def combinar(voz, contexto):
    chave = ("ia" if voz["parece_ia"] else "real", classe_conteudo(contexto["risco_conteudo"]))
    nivel, titulo, mensagem = TEXTOS[chave]
    return {"nivel": nivel, "titulo": titulo, "mensagem": mensagem, "conteudo": chave[1]}
