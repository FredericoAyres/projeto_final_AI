# Resultado: módulo de contexto com o idioma corrigido (teste WhatsApp, 16/11/2026)

Protocolo (escrito antes de qualquer chamada): `docs/PROTOCOLO_CONTEXTO_WHISPER.md`. Errata do teste de 15/11: `docs/ERRATA_TESTE_WHATSAPP.md`. Código: `scripts/diagnostico_whisper_idioma.py` (figura: `scripts/diagnostico_whisper_idioma_figura.py`). Nada foi treinado; prompt, LLM, cortes e `src/contexto_golpe.py` ficaram iguais (SHA256 conferidos antes de cada fase).

## Resumo

- **A causa do erro era real, mas não era a causa do recall baixo.** Com o idioma corrigido, o Whisper transcreve os clones praticamente sem erro (WER médio 0,012) e o resultado do contexto **continua ruim neste banco**: no RESERVADO, acurácia 45% (trivial 82%), recall 37%, falso alarme 18%.
- O módulo **detecta inglês sozinho** (modo `auto`, 95% no DEV), então foi escolhido o `auto`. O português do modo `auto` **não foi validado por nós** (o banco só tem inglês).

## 1. DEV, só Whisper (8 pessoas, 40 clipes; 24 clones, 16 reais)

| Modo | WER médio (24 clones) | WER mediano | "english" nos 40 clipes (clones / reais) |
|---|---|---|---|
| **auto** (sem `language`) | **0,012** | 0,000 | **38 / 40 = 95%** (24/24 / 14/16) |
| en | 0,012 | 0,000 | 40 / 40 (forçado) |
| pt (15/11, forçado) | 0,464 | 0,054 | 0 / 40 (forçado) |

- Concordância auto × en: 38 de 40 textos idênticos; WER médio de um contra o outro 0,0475 (a diferença vem dos 2 clipes reais em que o `auto` detectou "portuguese").
- **Regra de escolha (fixa no protocolo):** (a) `english` em ≥ 90% → 95% ✔; (b) WER do auto ≤ WER do en + 0,05 → 0,012 ≤ 0,062 ✔. **Escolhido: `auto`, `whisper-large-v3-turbo`.** O WER mediano (0,000) ficou abaixo de 0,15, então a variante `whisper-large-v3` **não foi testada**.
- Congelado em `results/diagnostico_whisper_congelado.json` (2026-11-16 10:08, UTC-0300), antes de tocar no RESERVADO.
- Tabelas: `results/diagnostico_whisper_idioma_dev_wer.csv`, `results/diagnostico_whisper_idioma_dev_clipes.csv`.

## 2. RESERVADO (15 pessoas, 60 clipes; principal), pipeline completo, uma vez

Alerta = risco médio ou alto. IC95 por bootstrap de pessoas (2.000, seed 42). Mesmos clipes nas duas linhas.

| Versão | Acurácia (trivial 81,7%) | Acc. balanceada | Recall | Falso alarme | Precisão |
|---|---|---|---|---|---|
| 15/11 (Whisper forçado a pt) | 51,7% (37,0–67,2) | 66,9% | 42,9% (29,6–56,8) | 9,1% (0–33,3) | 95,5% |
| **Corrigido (auto)** | **45,0% (32,8–56,9)** | 59,3% | **36,7% (25,0–48,8)** | **18,2% (0–33,3)** | 90,0% |

Por grupo (clipes com alerta; Wilson 95%):

| Grupo | n | 15/11 | Corrigido |
|---|---|---|---|
| real sem golpe | 9 | 1 (11%) | 2 (22%; 6–55) |
| real com golpe | 19 | 2 (11%) | 1 (5%; 1–25) |
| IA pede dinheiro | 30 | 19 (63%) | 17 (57%; 39–73) |
| IA neutra | 2 | 0 | 0 (0%; 0–66) |

Não houve `audio_sem_fala`, erro nem `indeterminado`. Idioma detectado no RESERVADO: 58 "english", 2 "portuguese". As diferenças entre as duas versões são de poucos clipes e os ICs se sobrepõem: **o idioma corrigido não melhorou nem piorou de forma demonstrável**.

## 3. DEV, pipeline completo (usado no diagnóstico; não é teste)

| Versão | Acurácia (trivial 60%) | Recall | Falso alarme | Precisão |
|---|---|---|---|---|
| 15/11 (pt forçado) | 55,0% (38,5–72,7) | 33,3% (13,8–60,0) | 12,5% (0–25) | 80,0% |
| Corrigido (auto) | 57,5% (41,2–74,4) | 37,5% (20,7–61,9) | 12,5% (0–25) | 81,8% |

Idioma detectado: 38 "english", 2 "portuguese". Todas as tabelas: `results/diagnostico_whisper_idioma_metricas.csv`, `_grupos.csv`, `_idioma.csv`. Figura: `results/slides/ctxw_01_acuracia_antes_depois.png`.

## 4. Leitura

- Em ambos os conjuntos o alerta cobre cerca de 37% dos golpes, acima do acaso mas **bem abaixo da meta de recall 0,85** da avaliação com textos inventados (87,7%). O recall baixo **não vem da transcrição** (no DEV a dos clones é quase perfeita); a hipótese é o julgamento do conteúdo (o prompt r1 é calibrado para evitar falsos alarmes, e pedidos de dinheiro simples ou gravações encenadas podem cair em risco baixo), mas **isso não foi verificado**. No RESERVADO não olhamos nenhuma transcrição, então não investigamos casos.
- A acurácia fica abaixo da trivial nos dois conjuntos (no RESERVADO há 49 positivos em 60). Para este módulo, o par recall × falso alarme é a leitura correta, não a acurácia.
- Nenhuma decisão de prompt, modelo ou corte foi tomada com base nestes números.

## 5. Desvios e limitações

- **Desvios do protocolo:** nenhum. A variante `whisper-large-v3` não foi rodada porque a condição (WER mediano > 0,15) não ocorreu.
- **O português do modo `auto` não foi validado.** O banco é todo em inglês. Em 2 de 16 reais do DEV (e 2 de 60 no RESERVADO) o `auto` detectou "portuguese" mesmo com fala em inglês: com `auto`, clipes curtos ou com sotaque podem ser transcritos no idioma errado (limite de 10% do protocolo; observado 4–5%).
- 23 pessoas (15 no RESERVADO); ICs largos; apenas 2 clipes IA neutros no RESERVADO.
- Os 10 textos de pedido de dinheiro se repetem entre as pessoas: o RESERVADO **não é independente em texto**.
- O rótulo "com golpe" dos reais vem da coleta e pode não conter pedido de dinheiro; clones de pedido de dinheiro nem sempre são golpe.
- Os agregados dos 100 clipes (idioma forçado) já tinham sido vistos em 15/11: o RESERVADO não é virgem em agregado, embora nenhuma escolha dependa dele.
- Depende de serviço externo (Groq) e do `gpt-oss-120b` na data do teste.

## 6. O que a chamada do site deve passar como idioma

`analisar_audio(..., idioma="auto")` (não enviar `language`). **Não** usar o padrão `"pt"` da função (nem o `--idioma pt` da CLI): forçaria português em áudio de outro idioma e foi o erro deste teste. A validação cobre **somente inglês**; o site deve avisar que, em português, o módulo não foi validado por nós. Nunca usar a tradução do Whisper.
