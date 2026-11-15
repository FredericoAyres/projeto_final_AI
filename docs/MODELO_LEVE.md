# Modelo leve (Fase 1): o sistema principal cabe num modelo bem menor?

**Data da medição:** 15/11/2026. **Protocolo (congelado antes de medir):** `docs/MODELO_LEVE_PROTOCOLO.md`. Código: `scripts/modelo_leve.py` e `scripts/modelo_leve_resumo.py`. Tabelas: `results/modelo_leve_*.csv` e `results/modelo_leve_resumo.json`. Figuras: `results/slides/ml_01_tamanho_e_tempo.png` e `ml_02_equivalencia.png`.

Nada foi treinado nem reajustado: os pesos da regressão são os mesmos da T3 (`results/v7_t3_coeficientes.npz`).

## 1. O que é compressão de modelo

É fazer o mesmo trabalho com um modelo menor ou mais barato de calcular, sem refazer o treino. Três jeitos aparecem aqui:

- **Cortar o que não é usado.** O sistema só usa as saídas das camadas 5, 6 e 7 do XLS-R. As camadas 8 a 24 não afetam a nota, então podem sair. As camadas 1 a 4 ficam, porque a 5 depende delas.
- **Reduzir a precisão dos números (fp16).** Cada peso ocupa 2 bytes em vez de 4.
- **Quantizar (int8).** As contas das camadas lineares são feitas com inteiros de 8 bits (1 byte por peso).

## 2. Resultado em uma tabela

Variantes: **V0** modelo completo (referência); **V1** corte depois da 7ª camada (fp32); **V2** V1 em fp16 (placa de vídeo); **V3** V1 com int8 nas camadas lineares (processador).

| | V0 completo | V1 cortado | V2 cortado, fp16 | V3 cortado, int8 |
|:--|--:|--:|--:|--:|
| parâmetros do modelo | 315.438.720 | 101.300.864 | 101.300.864 | 101.300.864 |
| + regressão | 3.073 | 3.073 | 3.073 | 3.073 |
| tamanho em disco (MB, modelo + regressão) | 1.211 | 386 | 193 | 133 |
| camadas do transformer executadas por áudio | 24 | 7 | 7 | 7 |
| pico de memória na placa de vídeo (MB) | 712 (fp16) | 545 | 303 | – |
| pico de memória RAM do processo, processador 4 threads (MB) | 2.175 | 1.334 | – | 1.936 |
| **placa de vídeo**, ms por áudio de 3 / 5 / 8 s (mediana) | 22 / 18 / 28 (fp16) | 14 / 21 / 32 | **9 / 10 / 15** | – |
| **processador, 4 threads**, ms por áudio de 3 / 5 / 8 s | 683 / 1.256 / 1.936 | 338 / 529 / 848 | – | **193 / 350 / 545** |
| **processador, 1 thread**, ms por áudio de 3 / 5 / 8 s | 1.253 / 1.975 / 3.249 | 630 / 951 / 1.513 | – | **400 / 645 / 1.132** |
| aceita pela regra? | (referência) | **sim** | **sim** | **sim** (com ressalva, ver 4) |

Parâmetros por componente (V0 → V1): extrator de convoluções 4.210.176 (igual); projeção 526.336 (igual); convolução posicional 8.389.760 (igual); **camadas do transformer 302.309.376 → 88.173.568**; layer norm final 2.048 → 0 (vira identidade). Mais 1.024 parâmetros de outros itens, iguais.

Os tempos são **medianas** de 30 medidas válidas por duração (percentil 90 em `results/modelo_leve_latencia.csv`; no processador de 4 threads o P90 do V1 é 366 / 594 / 925 ms e o do V3 é 218 / 379 / 622 ms). O tempo total inclui a leitura e o preparo do arquivo (1 a 3 ms; os arquivos já estavam no cache do disco) e a média, a padronização e a regressão (menos de 1 ms). **O que domina é o modelo.**

## 3. Quanto tempo leva para ouvir um áudio e responder "real ou IA"

Neste notebook, com um áudio de 5 s (lote de 1):

- **Placa de vídeo, V2 (cortado, fp16): cerca de 10 ms** (3 s: 9 ms; 8 s: 15 ms).
- **Processador, V3 (cortado, int8), 4 threads: cerca de 0,35 s** (3 s: 0,19 s; 8 s: 0,54 s). Com 1 thread: cerca de 0,65 s (8 s: 1,13 s).
- **Processador, V1 (cortado, fp32), 4 threads: cerca de 0,53 s.**
- Modelo completo no processador (V0): cerca de 1,3 s com 4 threads e 2,0 s com 1 thread.

O corte reduz o tempo no processador de **2,0 a 2,4 vezes** (V1 contra V0); com int8, de **2,9 a 3,6 vezes** (V3 contra V0).

Duas leituras importantes:

- O V0, como ele vem do Hugging Face, **calcula as 24 camadas mesmo que só as saídas até a 7ª sejam usadas** (medido por contagem de chamadas das camadas: 24 contra 7). Por isso o corte não só reduz o tamanho: **acelera**, no processador.
- **Na placa de vídeo, o V1 (fp32) não é mais rápido que o V0 (fp16)**, mas os dois não são comparáveis: precisões diferentes. A comparação justa é V2 contra V0 (os dois em fp16): **1,8 a 2,5 vezes mais rápido**. As latências na placa de vídeo são de poucos milissegundos, então variam mais entre execuções (o V0 de 5 s saiu mais rápido que o de 3 s, por exemplo).

## 4. A regra de aceitação e o que passou

Regra escrita **antes** de medir: uma variante é aceita se a AUC média por fonte cair no máximo 0,003 contra a referência e nenhuma fonte cair mais de 0,01. A V1 ainda precisava ser **equivalente**: diferença máxima de nota ≤ 0,001 logit e AUC igual até a 4ª casa. Amostra: 2.000 clipes da **validação** (6 fontes, 333 a 334 clipes por fonte, metade real e metade falso quando possível), sem nenhum clipe do teste mestre nem dos controles.

| comparação | maior diferença de nota (logit) | mediana da diferença | queda da AUC média | maior queda numa fonte | resultado |
|:--|--:|--:|--:|--:|:--|
| V1 fp32 × V0 fp32 | 0 | 0 | 0 | 0 | **aceita e equivalente** (as notas são idênticas) |
| V2 fp16 × V0 fp16 | 0 | 0 | 0 | 0 | **aceita** |
| V3 int8 × V0 fp32 | 4,53 | 1,07 | 0,0025 | 0,0055 (codecfake_plus) | **aceita** pela regra (limites 0,003 e 0,01) |

AUC média por fonte na amostra: **0,9720** (V0, V1 e V2) e **0,9695** (V3). Por fonte, V3 contra V0: codecfake_plus 0,9386 → 0,9331; dfadd 1,0000 → 1,0000; in_the_wild 0,9891 → 0,9907; ml_itw 0,9541 → 0,9490; radar_eval 0,9615 → 0,9589; shiftyspeech 0,9890 → 0,9854.

Informativo: V2 e V0 em fp16 contra o V0 em fp32 diferem até 0,077 logit (mediana 0,007), sem mudar a AUC (queda máxima por fonte 0,00007). O corte também reproduz as saídas 5, 6 e 7 exatamente (diferença 0,0). O V1 em fp32 no processador contra o da placa de vídeo diferiu até 2,2e-5 logit em 50 clipes.

**Ressalva sobre a V3 (importante):** a regra olha só para a ordem dos clipes (AUC). O int8 **muda o valor da nota** (até 4,5 logit, mediana 1,1). Como o corte de decisão da T3 (por exemplo, 0,23 logit) foi escolhido para as notas originais, **ele não pode ser usado direto na V3**: seria preciso fixar um novo corte para a V3 na validação (o que não foi feito aqui; ver 6). A V1 e a V2 não têm esse problema (diferença zero na nota).

## 5. Arquivos dos modelos (fora do git, em `models/`)

| variante | arquivo | MB |
|:--|:--|--:|
| V0 | arquivo original do Hugging Face (cache local, `pytorch_model.bin`) | 1.210,9 |
| V1 | `models/modelo_leve_V1_fp32.safetensors` | 386,4 |
| V2 | `models/modelo_leve_V2_fp16.safetensors` | 193,2 |
| V3 | `models/modelo_leve_V3_int8.pt` | 133,0 |

A regressão (3.072 coeficientes, média e desvio) tem 0,04 MB em `results/v7_t3_coeficientes.npz`. A quantização dinâmica funcionou com este modelo (motor `onednn`, nenhum pacote foi instalado).

## 6. Limitações

- **Equipamento:** "placa de vídeo do notebook" e "processador do notebook". **Os tempos variam em outros aparelhos.** O resultado no processador é só uma **referência** para celular, **não uma medida em celular** (celulares têm outros processadores, outro número de núcleos, outra memória e outros kernels de int8).
- Lote de 1 áudio, com 5 repetições de aquecimento descartadas; arquivos já no cache do disco (a leitura de arquivos frios seria mais lenta). A latência da placa de vídeo inclui o tempo de Python e é de poucos milissegundos.
- A **memória de pico** em RAM é o pico do processo inteiro, **incluindo a carga do modelo**; a V2 e a V3 são carregadas a partir do modelo em fp32 e convertidas na hora, então o pico delas não reflete um aplicativo que carregue o arquivo já convertido (V2: 1.505 MB; V3: 1.936 MB, contra 1.334 MB da V1).
- A equivalência foi medida em **2.000 clipes da validação** (inglês e as outras línguas dessas fontes), não no teste mestre; a V3 foi medida no processador e a referência V0 fp32 na placa de vídeo com TF32 desligado (a diferença placa/processador é ~2e-5 logit).
- As notas aqui vêm dos vetores sem arredondar para fp16; a T3 original foi pontuada a partir de vetores guardados em fp16 (por isso a referência é a V0 recalculada, não as notas antigas).
- Não foi medido: celular real, lotes maiores, outras durações além de 3, 5 e 8 s, nem o efeito no teste mestre e nos controles (para isso seria preciso reavaliar; fica para decisão do grupo).
- **Decisão em aberto, listada para o grupo:** se a V3 (int8) for adotada, escolher um corte de decisão novo pela validação; a V1 e a V2 podem manter o corte da T3.
