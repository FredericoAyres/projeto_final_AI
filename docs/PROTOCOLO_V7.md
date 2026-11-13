# Protocolo da v7: teste mestre fixo e 5 receitas (regressão logística sobre vetores já extraídos)

**Congelado em 2026-11-13, 19:36 (UTC−3), ANTES de qualquer treino da v7.** Branch `v7-treino-com-internet`, commit `66bd77d` (+ arquivos novos da v7, não commitados). Até aqui só foram feitos: juntar os vetores já extraídos (`scripts/v7.py vetores`) e montar a divisão com as salvaguardas (`scripts/v7.py splits`). Nenhum modelo foi treinado e nenhuma nota nova foi calculada.

- **Divisão congelada:** `data/processed/splits_v7.parquet`, **SHA256 `24d3aea93654711a80ea1482799c5de3ec180ec5e336c719935e98b7f7a54c63`** (249.187 linhas; fora do git; gerado de novo, igual, por `python scripts/v7.py splits`). Contagens: `results/v7_splits_contagens.csv`, `results/v7_splits_mlitw_regras.csv`, `results/v7_splits.json`.
- Depois deste carimbo, nada abaixo muda com base em resultado. Mudança por erro de execução vira adendo datado, com o motivo, e os números anteriores continuam reportados.

## 0. Este protocolo substitui o anterior para RADAR eval e ML-ITW

a) **Substitui** `docs/PROTOCOLO_TESTE_INTERNET.md` para esses dois bancos. Antes eles eram "só teste" e o F0 foi avaliado neles inteiros (AUC ~0,72 no RADAR e ~0,69 no ML-ITW; `docs/RESULTADO_TESTE_INTERNET.md`). Agora **partes deles entram no treino e na validação**, e fica um **teste mestre fixo**, igual para todas as receitas.

b) **Os números da v7 não são comparáveis** aos da avaliação inteira (outro recorte: aqui o teste é só o `en` do RADAR e parte dos locutores do ML-ITW; ver seção 2) **nem aos do desafio RADAR**. O site do desafio RADAR 2026 pede que os participantes **não treinem com os dados de avaliação**; aqui treinamos com parte deles (zh, zh-tw, ja), então estes resultados **não valem como resultado do desafio** e não devem ser apresentados assim. O dado é **CC BY 4.0** (`LICENSE.txt` do Zenodo 22110815), o que permite o uso com atribuição. A licença do ML-ITW continua não declarada.

c) **O teste mestre NÃO é virgem.** O F0 já foi avaliado nesses mesmos clipes, em agregado (RADAR e ML-ITW inteiros, 13/11). O In-the-Wild (38 locutores do `test_final`) foi visto da v4 à v6. Nada da v7 é escolhido olhando esses números, mas quem desenhou a v7 já conhecia o resultado agregado do F0 (por exemplo, que o RADAR `en` deu 0,724 e o `vi` 0,796).

d) **Só os clones próprios, quando existirem (`data/raw/clones_proprios`), seriam um teste intocado.** Hoje a pasta não existe; a etapa 3 não roda.

## 1. Modelo (igual em todas as receitas)

- Front-end: `facebook/wav2vec2-xls-r-300m` congelado, saídas **5, 6 e 7**, **média no tempo**, concatenadas (3.072 números), lidas dos caches já extraídos (`v4_cache`, `v4_cache_teste`, `teste_internet_cache`; nada é extraído de novo). Pipeline de áudio da v4 sem mudança.
- Cabeça: **regressão logística** do scikit-learn, `solver="lbfgs"`, `max_iter=1000`, `tol` padrão (1e-4).
- **Padronização ajustada em cada treino** (média e desvio do conjunto de treino da receita, sem pesos).
- Nota = logit (maior = mais falso). Nunca é invertida.

## 2. Divisão (o teste mestre fixo)

Uma linha por clipe em `splits_v7.parquet` (caminho, fonte, bloco, locutor, pessoa, idioma, label, plataforma ou gerador, split, regra, pareado_treino, pareado_teste, duração, hash do vetor).

| bloco | treino | validação | teste mestre | controle |
|:--|:--|:--|:--|:--|
| Estúdio (v4) | os 40.000 da v4 (20.000 / 20.000; mesmo plano, seed 42) | os 19.999 da v4, **menos 13** (ver abaixo) = 19.986 | — | LlamaPartialSpoof, BRSpeech-DF, CORAA, ASVspoof 2019 eval e 2021 DF (os 41.320 clipes do teste da v4): medem esquecimento; fora da métrica principal |
| In-the-Wild | nunca | os 16 locutores do dev_shift (10.614) | os 38 locutores do `test_final`: os **10.000 clipes da amostra da v4** (5.000 / 5.000; já extraídos) | — |
| ML-ITW | locutores sorteados (regra 4) | idem + Obama e Trump (regra 3) | regras 1, 2, 3 (Scarlett) e 4 | — |
| RADAR eval | zh, zh-tw, ja (**durações pareadas**) | en-sg, vi (natural) | **en** (natural e pareado) | — |

**Contagens (real / falso):**

| bloco | treino | validação | teste mestre |
|:--|:--|:--|:--|
| Estúdio | 20.000 / 20.000 | 5.991 / 13.995 | — |
| In-the-Wild | — | 8.633 / 1.981 (16 locutores) | 5.000 / 5.000 (38 locutores) |
| ML-ITW | 7.802 / 2.854 (68 / 68 locutores) | 3.754 / 994 (19 / 19 locutores) | 6.612 / 2.513 (56 / 73 locutores) |
| RADAR | 24.500 / 21.653; **pareado: 15.316 / 15.316** | 10.500 / 13.149 | 15.000 / 17.923; pareado 14.291 / 14.291 |
| Controle | — | — | 21.320 / 20.000 (fora da métrica principal) |

**ML-ITW (locutor atômico: todo o áudio de um locutor fica do mesmo lado), nesta ordem:**
1. Locutor com algum áudio em português → inteiro no teste mestre (8 locutores, 1.178 clipes).
2. Locutor de uma classe só → teste mestre (55; os 57 do banco menos `fausto`, que já caiu na regra 1, e `ScarlettJohansson`, que cai na 3).
3. Os 3 que coincidem por nome com o In-the-Wild (mesma pessoa) → o lado da pessoa no In-the-Wild: `obama` e `Trump` → **validação** (dev_shift); `ScarlettJohansson` → **teste mestre** (ela está no `test_final`). A regra 3 vale antes da 2 (Trump e Obama têm as duas classes; Scarlett só falsos, e cai no teste mestre de qualquer jeito).
4. Os demais locutores com as duas classes (113): sorteio **por locutor**, seed 42, 60% treino / 15% validação / 25% teste mestre, **estratificado pelo idioma principal do locutor** (chinês, inglês, outro). Decisão minha: a tarefa pedia "estratificado por locutor" sem dizer a variável; escolhi o idioma para os três lados terem chinês e inglês. Resultado: chinês 28/7/11, inglês 17/4/8, outro 23/6/10 locutores.
- Validação do ML-ITW: 19 locutores por classe (≥ 10, sem aviso).

**RADAR (sem locutor):** treino = zh, zh-tw, ja; validação = en-sg, vi; teste mestre = en. O **dev do RADAR nunca é usado** (não foi baixado; conferido: todos os clipes têm `original_split = eval`). **Pareamento de duração no treino:** dentro de cada idioma, faixas de 0,5 s (`duration_s` do manifesto), mesmo nº de reais e falsos por faixa (subamostra sem reposição, seed 42). No teste, a versão **natural** (todos os 32.923 clipes do en) e a **pareada** (o mesmo procedimento no en, seed 42: 14.291 / 14.291).

**Salvaguardas (assert; se falhar, `logs/v7_BLOQUEIO.txt` e a execução para):**
- Caminhos em comum entre treino, validação e teste mestre: **0**.
- Locutores (pessoas) em comum entre os lados, no ML-ITW e no In-the-Wild, com os 3 coincidentes mapeados para a mesma pessoa: **0**.
- Duplicatas exatas (SHA-1 dos bytes do vetor float16) entre lados: **0, depois de uma exclusão.** Na 1ª execução, a salvaguarda **parou** a divisão: 12 grupos de **reais do CodecFake+ (CoSG)** têm vetor idêntico entre o treino e a validação **de estúdio da v4** (o mesmo áudio de referência repetido em pastas de geradores diferentes, por exemplo `CLAMTTS_56.wav` no treino e `NS2_10.wav` na validação). É herança do plano da v4, não da v7. Como o treino é fixo (os 40.000 da v4), **os 13 clipes da validação foram excluídos da avaliação** (`split = "excluido"`). Desvio registrado: a regra pedia "mover o par para o mesmo lado"; mover para o treino mudaria os 40.000. Duplicatas exatas dentro do mesmo lado (118 clipes do ML-ITW, 4 do BRSpeech-DF) ficam como estão.
- Quase-duplicatas (cosseno ≥ 0,9995) entre treino/validação e teste mestre, no ML-ITW e no In-the-Wild: **nenhuma**. O cosseno é medido nos vetores padronizados com a média e o desvio do treino da v4 (os do F0), para tirar o deslocamento comum. Foram 26.018 × 19.125 pares, e o máximo entre lados foi 0,965. Regra registrada para o caso de achar algum par: o locutor inteiro do clipe de treino/validação iria para o teste mestre.
- Nenhum clipe do dev do RADAR: **conferido**.

## 3. Receitas

| receita | treino | pesos |
|:--|:--|:--|
| **T0** | o F0 **sem retreinar**: pesos de `results/v4final_f0_coeficientes.npz` e padronização de `results/v4_parcial/padronizacao.npz` (C = 0,1). Antes de usar, ele precisa reproduzir as notas gravadas do F0 (diferença máxima ≤ 1e-4 de logit) | — |
| **T1** | estúdio (40.000) + ML-ITW treino (10.656) | `class_weight="balanced"` |
| **T2** | estúdio + RADAR treino pareado (30.632) | `class_weight="balanced"` |
| **T3** | estúdio + ML-ITW + RADAR | `class_weight="balanced"` |
| **T4** | T3 com **peso igual por fonte** | cada **bloco** (estúdio, ML-ITW, RADAR) soma o mesmo peso total (N/3), e dentro dele reais e falsos somam o mesmo (N/6 cada); média 1. Sem `class_weight`. Decisão minha: "fonte" = bloco, porque parte das fontes do estúdio só tem reais (LibriSpeech) e não daria para balancear real/falso dentro delas |

- **C:** grade {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}. Em cada receita (T1 a T4), vence o C de **maior média da AUC por fonte na validação** (cada fonte pesa igual; só fontes com ≥ 20 clipes por classe). Fontes da validação: as do estúdio por banco (`dfadd`, `shiftyspeech`, `codecfake_plus`; o `librispeech` só tem reais e fica fora da AUC), `in_the_wild` (dev_shift), `ml_itw` e `radar_eval` (en-sg e vi **juntos, como uma fonte**, já que são do mesmo banco; por idioma vai só no diagnóstico). São 6 fontes. Empate exato: o primeiro da grade (o menor C).
- **Receita vencedora:** a de **maior média da AUC por fonte na validação** (o mesmo critério do C), entre T0 e T4. Sem margem de empate. O teste mostra todas.

## 4. Regras de corte (escolhidas SÓ na validação)

Calculadas nas notas da validação de cada receita:
- **(A')** ponto de EER da validação agrupada (todos os clipes da validação, inclusive o LibriSpeech só real);
- **(M)** mediana dos pontos de EER de cada fonte com as duas classes;
- **(F5)** em cada fonte com ≥ 20 reais, o corte que deixa no máximo 5% dos reais acusados (percentil 95 das notas dos reais); a mediana entre as fontes.

**Escolha:** validação cruzada deixando uma fonte de fora. Para cada uma das 6 fontes com as duas classes, o corte é calculado pela regra com as outras fontes da validação e aplicado nela. Vence a regra de maior acurácia balanceada média, sem margem de empate. O corte final é o da regra vencedora calculado na validação inteira. No teste, **as três regras são reportadas** (TPR e FPR), e a escolhida vem marcada.

## 5. Métricas no teste mestre

- **Principal:** média da AUC por fonte em **[ML-ITW teste, In-the-Wild 38, RADAR en (natural)]**, e a **pior fonte**. O "ML-ITW teste" é **todo** o teste mestre do ML-ITW, inclusive os locutores com português. O **português do ML-ITW** também vai **à parte**, como indício (8 locutores).
- **IC95:** bootstrap com 2.000 repetições, seed 42, estratificado por classe. **Por locutor** no ML-ITW e no In-the-Wild; **por clipe** no RADAR e nos controles. O do RADAR é marcado como **otimista**, porque o arquivo não tem locutor. Os mesmos sorteios valem para todas as receitas (pareado). O IC da média e da pior fonte sai da média e do mínimo das 3 fontes em cada repetição. A diferença para o T0 também é reportada, com IC pareado.
- Por recorte: AUC, EER; TPR (falsos detectados), FPR (reais acusados) e acurácia balanceada em cada regra de corte; nota média e mediana de reais e falsos. Recortes com < 20 clipes numa classe só dão contagem. **Menos de 30 locutores por classe = indício, não estimativa.**
- **Nunca** se juntam reais de uma fonte com falsos de outra.
- **Domínio visto × não visto** (por receita e fonte): T0 não viu nenhum. T1 viu o ML-ITW (outros locutores). Em T2, T3 e T4, o RADAR en é **"idioma novo, mas fonte, gerador e canal já vistos"**. T3 e T4 viram os dois. O **In-the-Wild nunca é visto** por nenhuma receita.
- **Bloco inglês (foco principal), tabela e figura à parte, por receita:** AUC (IC95) e TPR/FPR em cada regra para RADAR en (natural e pareado), ML-ITW **só os clipes em inglês** do teste mestre (marcado se < 30 locutores por classe), In-the-Wild (38) e, como controle de áudio limpo, LlamaPartialSpoof, ASVspoof 2019 eval e ASVspoof 2021 DF. Os dois ASVspoof são **secundários** (podem ter locutores em comum com o treino via VCTK) e não entram na métrica principal. Também: **AUC do RADAR en por faixa de duração** (< 2 s, 2 a 4, 4 a 6, ≥ 6 s).
- **Controles (esquecimento):** AUC dos 5 testes antigos de estúdio (CORAA: só o FPR) em cada receita.

## 6. Regras de uso do teste

- Cada receita é avaliada no teste mestre **uma vez** (a etapa `avaliar` grava um marcador e não roda de novo). Não se criam receitas novas nem se muda nada depois de ver o teste; o que vier depois vai numa seção **"pós-hoc"**.
- **A receita vencedora sai da validação** (`results/v7_escolha.json`, gravado antes do teste). O teste mostra todas.
- O teste rápido (`--quick`: subamostras pequenas, saídas `_quick`) roda antes, só para testar o código; os números dele não são lidos.

## 7. Diagnóstico (etapa 2; não decide nada)

Por receita: AUC por idioma no RADAR (em todos os lados; os idiomas de treino marcados como "dentro da amostra" em T2 a T4) e por plataforma no ML-ITW (teste mestre). TPR e FPR em cada regra. Nota média de reais e falsos por fonte (deslocamento de calibração). **Auditoria de atalho, só com metadados** (duração, fração de silêncio nas bordas, RMS, pico, tamanho do arquivo): regressão logística treinada no lado de treino de cada fonte nova e medida na validação e no teste mestre, mais validação cruzada em 5 partes dentro do teste mestre (por locutor no ML-ITW). Marca "sinal de atalho" acima de 0,7, como antes.

## 8. Execução

`scripts/v7.py` (etapas `vetores`, `splits`, `treinar`, `escolher`, `avaliar`, `figuras`) e `scripts/v7_figuras.py`. Um processo pesado por vez. Vigia de memória: encerra acima de 10 GB (processo + filhos). Status em `logs/v7_status.txt`. A suspensão do Windows fica impedida por `SetThreadExecutionState` em cada processo. O `powercfg` foi bloqueado pela permissão desta sessão, então nenhuma configuração do sistema foi mudada e não há nada a restaurar. Notas por clipe em `data/processed/v7_notas.parquet` (fora do git).
