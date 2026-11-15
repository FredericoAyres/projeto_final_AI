# Modelo leve (Fase 1): protocolo

**Congelado em 2026-11-15, 14:25 (hora local), ANTES de qualquer medida de latência, tamanho ou nota das variantes.** Branch `modelo-leve`. Nada é treinado, baixado ou reajustado. Nenhum arquivo existente é alterado. O teste mestre, os controles e os testes antigos não são tocados.

Só foi feito antes deste protocolo (leitura de código e uma checagem numérica de 2 s de áudio aleatório, sem clipes reais, só para entender o modelo):
- Nesta versão do `transformers` (5.22.2), `hidden_states[k]` (k = 1 a 24) é a **saída crua da camada k**, sem layer norm; o layer norm final do encoder só entra em `last_hidden_state`. As saídas 5, 6 e 7 do modelo cortado depois da camada 7 (layer norm final = identidade) saíram **idênticas** (diferença máxima 0,0) às do modelo inteiro.
- O modelo inteiro executa as 24 camadas (24 chamadas de camada contadas por gancho), mesmo quando só as saídas 5 a 7 são usadas.

## 1. O que é medido

Sistema = XLS-R 300M congelado + saídas 5, 6, 7 (média no tempo só nos quadros válidos, 3.072 números) + padronização (`mu`, `sd`) + regressão logística T3 (3.072 coeficientes + intercepto), pesos de `results/v7_t3_coeficientes.npz`. Nada é reajustado.

| variante | o que é | precisão / aparelho |
|:--|:--|:--|
| **V0** | modelo completo (24 camadas), referência | fp16 na placa de vídeo; fp32 no processador |
| **V1** | corte depois da 7ª camada do transformer (mantém extrator de convoluções, projeção e convolução posicional; layer norm final = identidade) | fp32 |
| **V2** | V1 em fp16 | placa de vídeo |
| **V3** | V1 com quantização dinâmica int8 das camadas lineares (`torch.ao.quantization.quantize_dynamic`) | processador |

Se a quantização não funcionar com este modelo, registra-se o erro e o resultado fica "não funcionou" (nenhum pacote é instalado).

As camadas 1 a 4 não podem ser retiradas (a 5 depende delas).

## 2. Amostra de equivalência (2.000 clipes da validação)

- Fonte: `data/processed/splits_v7.parquet`, `split == "validacao"` (SHA256 conferido contra o do protocolo da v7). Nenhum clipe do teste mestre nem dos controles.
- "Fonte" = a mesma chave da validação da v7 (`scripts/v7.py: chave_val`): banco para o estúdio, bloco para os demais. Só entram fontes com as duas classes e ≥ 20 clipes por classe na amostra (a LibriSpeech, só com reais, sai). Restam 6: codecfake_plus, dfadd, shiftyspeech, in_the_wild, ml_itw, radar_eval.
- Estratificação: 2.000 / 6 = 333 clipes por fonte (+1 para as 2 primeiras fontes em ordem alfabética, para somar 2.000); dentro da fonte, metade real e metade falso; se uma classe tiver menos do que a metade, entram todos os dela e o resto vem da outra. Sorteio com `seed = 42` (`numpy.random.default_rng`).
- Pré-processamento: o da v7/v4 (`src/preproc.py`: `ler_mono_16k` + `preparar_clipe` com corte de silêncio, no máximo 8 s do meio, 16 kHz, sem completar). Clipe com erro de leitura sai da amostra e é registrado (não é reposto).
- Os clipes preparados ficam em `data/processed/modelo_leve_amostra/` (fora do git).

## 3. Como as notas são calculadas

- Mesma conta da extração da T3 (`src/ssl.py: representar`): cada clipe é normalizado sozinho (média 0, variância 1) antes do enchimento, lote com `attention_mask`, média só nos quadros válidos; padronização `(x − mu)/sd` e `x @ coef + intercepto` em float32. Lotes ordenados por duração (até 16 clipes). Os vetores não são arredondados para fp16 antes da regressão (a T3 original foi pontuada a partir de vetores armazenados em fp16; por isso a V0 aqui é a referência, e não as notas antigas).
- **Referências da comparação:** V0 fp32 (referência da V1 e da V3) e V0 fp16 (referência da V2). As notas de V0 fp32 e V1 fp32 da amostra são calculadas na **placa de vídeo em fp32 com TF32 desligado**, para não gastar horas; a V3 é calculada no processador (4 threads). Uma conferência de 50 clipes mostra a diferença entre V1 fp32 na placa de vídeo e no processador. A V2 também é comparada à V0 fp32 (informativo).
- AUC por fonte com `sklearn.metrics.roc_auc_score`; "AUC média por fonte" = média simples das 6.

## 4. Regra de aceitação (escrita antes)

Uma variante é **ACEITA** se, na amostra, contra a sua referência:
1. a AUC média por fonte cair **no máximo 0,003** em relação à referência; **e**
2. **nenhuma fonte** cair mais de **0,01**.

A **V1 deve ser EQUIVALENTE**: diferença máxima de nota (logit) **≤ 1e-3** em fp32 contra a V0 em fp32, e AUC (média e por fonte) igual até a 4ª casa. **Se a V1 falhar, para tudo** e escreve `logs/modelo_leve_BLOQUEIO.txt`.

Quedas são medidas como (AUC da referência − AUC da variante); aumento não reprova.

## 5. Latência, tamanho, memória

- Lote de 1. Durações 3, 5 e 8 s. Para cada duração, **5 clipes reais (`label == 0`) diferentes da validação** com duração original ≥ alvo (sorteio com `seed = 42` entre os que têm duração original até alvo + 2 s; se não houver 5, entre os ≥ alvo), todos distintos entre as durações. O modelo recebe o **recorte central EXATO** de 3, 5 ou 8 s (em 16 kHz, tirado do áudio inteiro reamostrado, antes do corte de silêncio; assim a duração do que entra no modelo é exata).
- Por (variante, aparelho, duração): **5 repetições de aquecimento descartadas** e **30 medidas válidas** (os 5 clipes × 6 repetições, em ordem rotativa). Relata mediana e percentil 90.
- Três partes, cada uma cronometrada com `time.perf_counter`: **(a)** leitura do arquivo e preparo (`ler_mono_16k` + `preparar_clipe`: 16 kHz, corte de silêncio, recorte; medida no clipe de origem, **uma vez por aparelho/threads e compartilhada entre as variantes**, pois não depende do modelo; é trabalho de processador mesmo no caso da placa de vídeo, medido com 4 threads); **(b)** normalização do recorte, cópia para o aparelho, modelo até obter as saídas 5, 6 e 7; **(c)** média no tempo, padronização e regressão, até a nota chegar à memória principal. Na placa de vídeo, `torch.cuda.synchronize()` antes e depois de (b) e (c).
- O V0 é medido "como vem": pede `output_hidden_states=True` e usa as saídas 5, 6, 7 (calcula as 24 camadas). Também se relata (contando chamadas de camada por gancho) se o V0 já para na 7 ou não.
- Aparelhos: placa de vídeo do notebook (V0 fp16, V1 fp32, V2 fp16); processador do notebook com **4 threads** e com **1 thread** (V0 fp32, V1 fp32, V3 int8).
- **Tamanho em disco:** arquivo do modelo de cada variante (V0: o arquivo original do cache do Hugging Face; V1 a V3: salvos em `models/`, fora do git) + arquivo dos pesos da T3. **Parâmetros:** total e por componente (extrator de convoluções, projeção de características, convolução posicional, camadas do transformer, layer norm final), da regressão: 3.073.
- **Memória de pico:** placa de vídeo, `torch.cuda.max_memory_allocated` (modelo + um áudio de 8 s); processador, pico do conjunto de trabalho do processo (`PeakWorkingSetSize`, inclui a carga do modelo). Cada variante/aparelho roda em um processo novo.
- Aceleração = mediana da V0 / mediana da variante, no mesmo aparelho e duração.

## 6. Regras de execução

- Um processo pesado por vez; memória até 10 GB; o Windows é impedido de suspender (`SetThreadExecutionState` em cada processo) e restaurado ao final.
- Retomável por etapa, com status em `logs/modelo_leve_status.txt`.
- Os tempos valem para esta máquina e variam em outros aparelhos. O resultado no processador é só uma **referência** para celular, não uma medida em celular.
- Nada muda depois de ver os números. Estimativas, se houver, são rotuladas como estimativa.
- Todo valor reportado vem de medição; tudo está em `results/modelo_leve_*.csv/json`.
