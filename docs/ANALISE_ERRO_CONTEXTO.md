# Análise de erro do contexto de golpe (16/11/2026)

**Descritivo, pós-hoc; não usado para ajustar nada; contexto = parte mais fraca; sem mais dados para testar.** Usa só o risco já gravado (Whisper `auto`, prompt r1 congelado; 100 clipes, DEV e RESERVADO juntos) e o texto do roteiro dos clones. Nenhum áudio real foi transcrito aqui. Os 10 textos de pedido se repetem entre as pessoas, então um ajuste de prompt feito com isto não seria validável. O manifesto não tem `frase_id`: os ids F01–F16 foram criados aqui (um por texto distinto, 16 textos nos 56 clones; ordem = recall).

## Clones por frase (alerta = risco médio ou alto)

Tabela completa: `results/analise_erro_contexto_por_frase.csv` (primeiros 70 caracteres do roteiro). Resumo:

| Frases | Tema | Clipes | Alerta médio+ | Só alto |
|---|---|---|---|---|
| F01–F06: banco/cartão em viagem, "número novo", pagamento urgente, "não consigo transferir, pague você", conserto do carro | golpe | 26 | 26 (100%) | 10 |
| F13–F16: dinheiro para mercado, US$ 20 de um jantar, conta de luz, empréstimo de 50 até sexta | golpe | 20 | 0 (0%) | 0 |
| F07–F12: cotidiano (série, massa, viagem, fone, jogo, tempo) | cotidiano | 10 | 0 (correto) | 0 |

Cada frase é tudo ou nada (5/5 ou 0/5, 4/4 ou 0/4): o módulo julga o texto, e o mesmo texto recebe o mesmo risco em pessoas diferentes.

## Reais, risco por grupo (sem texto)

| Grupo | n | nenhum | baixo | médio | alto |
|---|---|---|---|---|---|
| real com golpe | 27 | 24 | 2 | 1 | 0 |
| real sem golpe | 17 | 2 | 11 | 3 | 1 |

## Conclusão

1. Os clones perdidos (20 de 46 de pedido de dinheiro) vêm de 4 roteiros, todos de **pedido leve de dinheiro entre conhecidos** (mercado, jantar, conta de luz, empréstimo curto), sem urgência, sem identidade falsa, sem código ou link: pelo prompt, "baixo" é o esperado, e parte deles talvez nem seja golpe.
2. Os 6 roteiros com sinais claros (número novo, urgência, "pague por mim", cartão em viagem) foram todos pegos (26 de 26): **não houve roteiro claramente de golpe perdido** neste banco.
3. Nos reais com golpe, 24 de 27 ficaram em "nenhum": o rótulo vem da coleta (gravações encenadas), então não dá para dizer o que o módulo perdeu sem ler as falas, o que não foi feito. O recall baixo reflete, em boa parte, o rótulo "golpe" do banco, não só falha do módulo.
