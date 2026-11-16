# Combinado voz × conteúdo com o idioma corrigido (16/11/2026)

Recálculo **descritivo** dos 100 clipes juntos, sem nova chamada à API e sem rodar modelos: notas do T3 de `results/teste_whatsapp_notas.csv` (corte M = 0,2326) e o risco do contexto corrigido (Whisper `auto`) gravado em 16/11. **Não é um teste novo e não traz acurácia do combinado**, só taxas por célula. Tabela completa (por conjunto): `results/diagnostico_whisper_combinado.csv`. Contexto: `docs/RESULTADO_CONTEXTO_WHISPER.md`, `docs/ERRATA_TESTE_WHATSAPP.md`.

Regra congelada: voz = IA se T3 ≥ M; conteúdo = alerta se risco ≥ médio. Alto = os dois; atenção = um; baixo = nenhum.

## (a) Contexto por grupo, 100 clipes (alerta médio+ / só alto)

| Grupo | n | Corrigido: médio+ | Corrigido: só alto | 15/11 (pt forçado): médio+ | 15/11: só alto |
|---|---|---|---|---|---|
| real sem golpe | 17 | 4 | 1 | 3 | 0 |
| real com golpe | 27 | 1 | 0 | 2 | 0 |
| IA pede dinheiro | 46 | 26 | 10 | 27 | 7 |
| IA neutra | 10 | 0 | 0 | 0 | 0 |

Origem: DEV (8 pessoas, usado no diagnóstico) = 8 / 8 / 16 / 8 clipes nos grupos acima; RESERVADO (15 pessoas) = 9 / 19 / 30 / 2. A divisão por conjunto está no CSV.

## (b) Grupo × nível combinado (corrigido | 15/11)

| Grupo | n | baixo | atenção | alto |
|---|---|---|---|---|
| real sem golpe | 17 | 11 \| 12 | 6 \| 5 | 0 \| 0 |
| real com golpe | 27 | 26 \| 25 | 1 \| 2 | 0 \| 0 |
| IA pede dinheiro | 46 | 0 \| 0 | 20 \| 19 | 26 \| 27 |
| IA neutra | 10 | 0 \| 0 | 10 \| 10 | 0 \| 0 |

## O que mudou

Quase nada: 1 a 3 clipes por célula, dentro do ruído de um julgamento com poucos clipes. Continua valendo que nenhuma IA fica em "baixo" (a voz sozinha já as marca) e que nenhum real chega a "alto". O alerta de contexto segue deixando passar boa parte dos golpes (26 de 46 nos clones de pedido de dinheiro; 1 de 27 nos reais com golpe).

## Ressalvas

Um só gerador; 23 pessoas; os 10 textos de pedido de dinheiro se repetem entre as pessoas; o rótulo "com golpe" dos reais vem da coleta e pode não ter pedido de dinheiro; clones de pedido de dinheiro nem sempre são golpe; o áudio é só inglês (o português do modo `auto` não foi validado por nós); 40 dos 100 clipes (DEV) foram usados no diagnóstico; os agregados de 15/11 já tinham sido vistos.
