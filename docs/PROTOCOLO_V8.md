# Protocolo da v8: fine-tuning parcial do XLS-R a partir da T3

**Estado: CARIMBADO em 2026-11-14 14:42, antes do primeiro treino de verdade.** Commit base `0e21522`. O código da v8 ainda não estava commitado, então vale o SHA256 (16 primeiros dígitos) dos arquivos que rodam:

- `scripts/v8.py` `d01940f3ce85b6d8`
- `src/finetune.py` `f8fa02b9a87f8c7f`
- `scripts/teste_internet.py` `4868fb3b44a3947d` (inclui a correção da vigia, que agora encerra também os workers)
- `scripts/v7.py` `b1e20207760d8cd9`

Antes do carimbo só rodaram: a reprodução da T3 nesta máquina, a checagem do passo zero e da memória, a preparação dos áudios e o teste rápido (`--quick`). Do teste rápido só foram lidos tempo, memória e "passou ou não". O pico foi de 8,8 GB, e o treino andou a cerca de 20 clipes/s.

- **Base:** a T3 da v7 (`docs/PROTOCOLO_V7.md`; pesos em `results/v7_t3_coeficientes.npz`, SHA256 `e991737396c420d5438e7fdc4d258bfeaafcdb5b8fc5dc17e5ed575473bf3301`).
- **Divisão:** a mesma `data/processed/splits_v7.parquet`, gerada de novo nesta máquina por `python scripts/v7.py splits`. As contagens e as regras são idênticas às do colega (`results/v7_splits_contagens.csv` e `v7_splits_mlitw_regras.csv` sem nenhuma diferença). O SHA256 local é `eb6fb62f…3a38`, e não `24d3aea9…4c63`, só porque o arquivo guarda o hash dos vetores, que aqui diferem no ruído do float16 (ver seção 7).
- Depois do carimbo, nada abaixo muda com base em resultado. Mudança por erro de execução vira adendo datado, com o motivo, e os números anteriores continuam reportados.

## 0. Por que a v8 existe

O retorno da apresentação pediu para **aplicar fine-tuning**. A v8 entrega isso de qualquer forma: **o resultado é reportado mesmo que a T3 continue melhor.** Superar a melhor versão anterior é bônus, não condição.

A pergunta da v8: **ajustar as camadas do XLS-R que a T3 usa (5, 6 e 7) melhora a validação sem esquecer os bancos antigos?**

## 1. O que fica igual à v7 (comparação justa)

- **Treino:** o da T3, ou seja, estúdio (40.000) + ML-ITW (10.656) + RADAR zh, zh-tw e ja pareado (30.632) = **81.288 clipes**.
- **Validação e teste mestre:** os mesmos da v7. A validação tem 58.997 clipes em 6 fontes com as duas classes. O teste mestre é ML-ITW (9.125), In-the-Wild (10.000) e RADAR en (32.923). Os controles de estúdio somam 41.320.
- **Pipeline de áudio da v4:** 16 kHz, corte de silêncio nas bordas (`TOP_DB` 35), recorte central de no máximo 8 s, sem completar (`src/preproc.py`).
- **Critérios da v7:**
  - média da AUC por fonte na validação (cada fonte pesa igual; só fontes com ≥ 20 clipes por classe);
  - regras de corte A', M e F5, com a escolha por validação cruzada deixando uma fonte de fora;
  - métricas do teste mestre e IC por bootstrap (2.000 repetições, seed 42; por locutor no ML-ITW e no In-the-Wild, por clipe no RADAR).

## 2. O modelo (`src/finetune.py`)

- **O mesmo caminho da T3 vira uma rede só:** XLS-R → média no tempo das saídas 5, 6 e 7 (3.072 números) → padronização da T3 (média e desvio fixos) → camada linear 3.072 → 1, que é a regressão logística escrita como camada.
- **XLS-R cortado depois da camada 7.** As camadas 8 a 24 não entram na T3, então o corte só economiza tempo e memória. O layer norm final do encoder vira identidade, para a saída 7 ficar igual à do modelo inteiro.
- **Sempre em modo `eval()`:** sem dropout, sem LayerDrop e sem o mascaramento do pré-treino. No passo zero, a saída é a da T3, e a única fonte de mudança é o gradiente.
- **Precisão:**
  - a parte congelada roda em float16 puro, como na extração dos vetores da T3;
  - as camadas treináveis guardam os pesos em float32 e fazem as contas em float16.
  - **Conferido:** no passo zero, a diferença para as notas da T3 (calculadas do cache) é de 0,003 de logit na mediana e 0,012 no máximo, em 48 clipes de validação. Em bfloat16, que estava no plano, eram 0,08 na mediana e 0,68 no máximo, por isso o plano mudou para float16 (seção 7).

## 3. Receitas

| receita | o que treina | cabeça começa | seeds | para quê |
|:--|:--|:--|:--|:--|
| **R0** | nada (a T3 da v7) | — | — | referência |
| **R1** | camadas 5, 6 e 7 + cabeça | pesos da T3 | 42 e 123 | a receita principal |
| **R2** | só a camada 7 + cabeça | pesos da T3 | 42 | quanto descongelar |
| **R3** | camadas 5, 6 e 7 + cabeça | aleatória | 42 | o começo pela T3 (LP-FT) ajuda? |

As notas da R0 vêm dos vetores já extraídos (`data/processed/v7_vetores`) com os pesos do colega.

## 4. Treino (igual em R1, R2 e R3)

- **Volume:** 2 épocas. Cada época sorteia **20.000 clipes** do treino da T3, 10.000 reais e 10.000 falsos, sem reposição dentro da época (seed × 1000 + época). O peso igual por classe é o mesmo efeito do `class_weight="balanced"` da T3.
- **Lotes:** 16 clipes do mesmo "balde" de duração (até 2, 4, 6 ou 8 s), para o lote ter formato fixo. A sobra de cada balde é descartada e a ordem dos lotes é sorteada. São 1.248 passos por época na seed 42 e 1.249 na seed 123 (contados com as durações dos áudios preparados).
- **Perda:** entropia cruzada binária no logit (maior = mais falso).
- **Otimizador:** AdamW, weight decay 0,01.
- **Taxa de aprendizado:** 1e-5 nas camadas do XLS-R e 1e-4 na cabeça, com aquecimento linear nos primeiros 10% dos passos e queda linear até zero.
- **Corte da norma do gradiente:** 1,0.
- **Candidatos:** o fim da época 1 e o fim da época 2. Os dois são avaliados na validação inteira, e vence o de maior média da AUC por fonte (empate: a época mais cedo).

**Por que cada escolha fora da trilha faz sentido aqui** (⚠️ Fora da trilha):

- **Fine-tuning de transformer.** A Aula 10 usa o modelo pré-treinado congelado (transfer learning). Aqui, as camadas que a T3 já usa voltam a aprender, devagar, para se especializar em detectar voz falsa. Só 3 das 24 camadas treinam (37,8 M parâmetros na R1 e na R3; 12,6 M na R2), o que cabe na GPU de 6 GB (RTX 4050 Laptop) e reduz o risco de "esquecer" o pré-treino.
- **Cabeça começando na T3 (LP-FT: primeiro a cabeça linear, depois o ajuste fino).** Com a cabeça aleatória, os primeiros passos mandam gradientes sem sentido para as camadas e estragam o que elas sabem. Começando pela T3, que já funciona, o ajuste é fino desde o primeiro passo. A R3 mede esse efeito.
- **AdamW.** É o otimizador padrão para ajustar transformers, o mesmo do artigo do wav2vec 2.0. Ele escolhe o tamanho do passo de cada peso sozinho, e o weight decay puxa os pesos de leve para não se afastarem demais.
- **Taxa menor nas camadas (1e-5) que na cabeça (1e-4).** As camadas já estão boas e só podem mudar pouco. A cabeça pode mudar mais.
- **Aquecimento e queda linear.** A taxa começa perto de zero, o que evita um tranco que apagaria o pré-treino, e termina em zero, o que deixa o fim do treino estável.
- **Corte da norma do gradiente.** Se um lote gerar um gradiente enorme, ele é encolhido para norma 1. Isso evita que um único lote estranho estrague o modelo.
- **Precisão mista float16.** As contas em 16 bits gastam metade da memória e são mais rápidas. Os pesos que aprendem ficam em 32 bits, e o "GradScaler" multiplica a perda para os gradientes pequenos não virarem zero.

## 5. Escolha (SÓ validação) e teste (uma vez)

- **Escolha da época de cada rodada:** seção 4.
- **Receita vencedora:** a de maior média da AUC por fonte na validação, entre R0, R1, R2 e R3. Na R1, a nota da receita é a média das duas seeds. Sem margem de empate. O resultado vai para `results/v8_escolha.json`, gravado **antes** de qualquer nota do teste.
- **Corte:** para cada candidato, as regras A', M e F5 são calculadas na validação, e a escolhida é a de maior acurácia balanceada deixando uma fonte de fora (como na v7).
- **Teste mestre:** cada rodada (R1 s42, R1 s123, R2 s42, R3 s42) é avaliada **uma vez**, na época escolhida pela validação. **Todas são reportadas.** A etapa `avaliar` grava um marcador e não roda de novo.
- **O que é reportado:**
  - média e pior fonte de [ML-ITW, In-the-Wild 38, RADAR en], com IC95;
  - diferença para a R0 com IC pareado (os mesmos sorteios do bootstrap);
  - AUC, EER, TPR/FPR e acurácia balanceada em cada regra;
  - recortes em inglês e o português do ML-ITW à parte (indício);
  - controles de estúdio, que medem esquecimento.
- **R1 com duas seeds:** cada seed aparece separada, mais uma linha com a média das duas. As notas não são juntadas.
- **O teste mestre NÃO é virgem:** a T3 já foi avaliada nele (v7), e quem desenhou a v8 conhece esses números (média 0,963). Nada da v8 é escolhido olhando o teste. Só clones próprios gravados depois seriam um teste intocado.

## 6. Execução (`scripts/v8.py`)

- **Etapas:** `preparar` → `treinar` (4 rodadas) → `escolher` → `testar` (4 rodadas) → `avaliar`, nessa ordem, cada uma num processo (`python scripts/v8.py fila`).
- **Retomada:** cada etapa retoma de onde parou. O treino retoma por época.
- **Vigia de memória:** encerra acima de 10 GB (processo + workers). O Windows fica impedido de suspender durante a execução.
- **Onde ficam as coisas** (tudo fora do git, exceto `results/v8_*`):
  - status: `logs/v8_status.txt`;
  - checkpoints: `models/v8_*.pt`;
  - notas por clipe: `data/processed/v8_notas/`;
  - áudios preparados: `data/processed/v8_audio/`.
- **Áudios preparados (execução, não método):** ler e decodificar os arquivos custava mais que a GPU (medido: cerca de 20 clipes/s, contra cerca de 45). Por isso, cada clipe é preparado uma vez pelo mesmo `src/preproc.py` e guardado em float16. Conferido em 120 clipes: a diferença entre ler o preparado e o original é de no máximo 0,04 de logit.
- **Lotes de avaliação:** no máximo 2,05 M amostras por lote (32 clipes de 2 ou 4 s, 21 de 6 s, 16 de 8 s). Com os 4,1 M da extração, a GPU passou de 5,9 GB, o driver começou a usar a RAM como extensão e o processo chegou a 13 GB.
- **Teste rápido:** `--quick` usa subamostras pequenas e grava as saídas com sufixo `_quick`. Os números não são lidos.

## Adendo 1 (2026-11-14, 23:30): interrupção na etapa `testar`

- **O que aconteceu:** às 23:22, durante o teste da R1 s42 (86.087 de 93.368 clipes), a fila inteira encerrou sem erro registrado. Caiu inclusive a janela do PowerShell que a comandava. A causa mais provável é o fechamento dessa janela.
- **Nada foi lido:** as notas do teste só são gravadas no fim, então nenhum número do teste existia.
- **O que muda:**
  - a fila é relançada e retoma do teste da R1 s42; os treinos e a escolha já estavam prontos;
  - `escolher` passa a pular quando `results/v8_escolha.json` já existe, para não regravar a escolha (feita às 19:30:16, antes de qualquer teste);
  - o novo SHA256 de `scripts/v8.py` é `8745c173c2052dba`.
- Nada do método muda.

## 7. Desvios em relação ao plano aprovado (antes de qualquer treino)

1. **bfloat16 → float16.** A conferência do passo zero mostrou que o bfloat16 não reproduz a T3 (diferença de até 0,68 de logit), e o float16 reproduz (0,012). Motivo: o cache da T3 foi extraído em float16.
2. **Lote de avaliação de 4,1 M → 2,05 M amostras**, por memória (seção 6).
3. **Áudios preparados uma vez**, por velocidade (seção 6). Não muda o método.
4. **Reprodução da T3 nesta máquina, antes da v8:**
   - as notas da R0 usam os pesos do colega sobre os vetores desta máquina, que foram reextraídos na v5 e batem com os da v4 dentro do ruído do float16;
   - as etapas `treinar`, `escolher` e `avaliar` da v7 não foram refeitas aqui. A conferência dela pede um arquivo que só existe na máquina do colega (`results/v4_teste_notas.parquet`), e refazer o `avaliar` seria olhar o teste de novo;
   - nesta máquina, as AUCs de validação da grade de C batem com as do colega na 3ª ou 4ª casa, o C escolhido é o mesmo em todas as receitas e os pesos da T3 refeitos têm correlação de 0,99995 com os dele.
