# Histórico de versões

Registro vivo de tudo o que já foi tentado no detector. Serve para qualquer integrante (e o Claude Code dele) saber o que já foi feito **antes** de propor algo novo.

## a) Como usar

1. **Leia este arquivo inteiro antes de propor ou rodar uma nova versão.** Confira a tabela-resumo (d), os registros detalhados (e) e as ideias ainda não testadas (f). Não repita um experimento que já está aqui sem dizer o que muda.
2. **Ao terminar uma versão:** acrescente uma linha na tabela-resumo e um bloco no registro detalhado (modelo em e), arquive o notebook em `notebooks/versions/vN_nome.ipynb` (com as saídas) e atualize as ideias (f).
3. Reporte **todas** as versões e experimentos, inclusive os que pioraram.

## b) Formato do notebook (o padrão a seguir)

- **Seções numeradas:** 1 Introdução · 2 Configuração · 3 Mapa do banco · 4 Visualização · 5 Análise exploratória · 6 Treinamento · 7 Avaliação · 8 Conclusões. No topo, antes da seção 1, uma seção curta "Histórico" com a tabela-resumo lida de `results/`.
- **Seção 2:** todos os hiperparâmetros numa **única célula**, com a flag `QUICK` (teste rápido de ponta a ponta) e uma tabela com a justificativa de cada hiperparâmetro **em uma frase**. Essa célula tem a **tag `config`** e define `HP`, `EXPERIMENTOS`, `caminho_ckpt` e `caminho_hist`: o `scripts/treinar_experimento.py` executa exatamente essa célula para treinar cada experimento num processo separado.
- **Uma célula markdown antes de cada bloco de código**, dizendo **o que faz** e **por quê**.
- **"⚠️ Fora da trilha"** em tudo o que vier de fora do que a trilha ensinou (https://trilhas-insperai.vercel.app/trainees), com uma explicação curta.
- Texto em **português, direto**.
- **Código reutilizável em `src/`**, importado pelo notebook (sem duplicar entre versões).
- **No máximo 6 reprodutores de áudio** embutidos; figuras com **dpi moderado** (~70 a 90).
- Notebook final, com as saídas, com **menos de 10 MB**.
- Seções 3 a 5 só com o que é novo na versão; o resto é um resumo com link para `docs/dataset_overview.md` e para o notebook da versão em que a análise foi feita.

## c) Regras do jogo (anti-vazamento)

- **Bancos de teste nunca entram no treino nem na escolha de hiperparâmetros.**
- **Decisões entre versões usam validação + dev_shift.** O dev_shift são 30% dos locutores do In-the-Wild (coluna `eval_subset` do manifest, seed 42), separados do teste por locutor. Pode ser olhado quantas vezes for preciso.
- **O teste final (`eval_subset == "test_final"`) é avaliado uma única vez por versão**, e **todas** as versões são reportadas, sem escolher a melhor pelo teste.
- **Uma mudança por experimento**, mantendo o resto igual à versão de referência.
- **O split train/val é por locutor** (nenhum `speaker_id` em train e val ao mesmo tempo).
- **Nunca inverter a nota do modelo com base no teste** (por exemplo, "o In-the-Wild deu AUC 0,2, então vamos usar 1 − nota").

## d) Tabela-resumo das versões

AUC e EER são as métricas principais (não dependem de limiar). "val (média/fonte)" é a média das EER por fonte da validação (só fontes com real e fake). A v1 foi avaliada no recorte central de 4 s e no In-the-Wild inteiro; da v2 em diante, o teste usa janelas deslizantes e só o `test_final` do In-the-Wild.

| id | data | hipótese | mudança | EER val (global / média por fonte) | dev_shift AUC / EER | In-the-Wild AUC / EER | LlamaPartialSpoof AUC / EER | BRSpeech-DF AUC / EER | CORAA falso alarme | conclusão em uma frase | notebook arquivado |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| v1 | 2026-11-08 | uma CNN pequena em log-mel generaliza para geradores e gravações novos | baseline (sem augmentation) | 1,3% / 13,6% | — | 0,211 / 72,2% | 0,563 / 45,4% | 0,766 / 30,6% | 41,6% | ótimo em casa, péssimo fora: parece ter aprendido "gravação limpa = real" | `notebooks/versions/v1_baseline.ipynb` |
| v2 · v1r | 2026-11-08 | escolher o checkpoint pela EER média por fonte dá uma referência justa | só a regra do checkpoint | 1,4% / 12,6% | 0,325 / 63,7% | 0,262 / 68,2% | 0,625 / 40,7% | 0,776 / 27,9% | 34,9% | parecido com a v1; In-the-Wild continua invertido | `notebooks/versions/v2_augmentation.ipynb` |
| v2 · v2a | 2026-11-08 | volume não é pista (controle) | + RMS e ganho aleatório | 1,1% / 13,0% | 0,386 / 58,1% | 0,354 / 59,3% | 0,595 / 43,3% | 0,801 / 27,5% | 24,3% | validação igual; dev_shift +0,06, que tomamos como o tamanho do acaso do treino | `notebooks/versions/v2_augmentation.ipynb` |
| v2 · v2b | 2026-11-08 | ensinar ruído e canal ruim tira o atalho "limpo = real" | + ruído, banda 8 kHz, mu-law | 2,0% / 14,3% | 0,389 / 57,0% | 0,336 / 61,3% | 0,646 / 39,1% | 0,779 / 28,9% | 29,8% | robusto às degradações que viu, mas sem ganho no dev_shift além do controle; hipótese não confirmada | `notebooks/versions/v2_augmentation.ipynb` |
| v3 · repeat (3 seeds) | 2026-11-09 | (controle pareado da v3) | áudio < 4 s repetido até 4 s, como no v1r; seeds 42, 123 e 2026 | 2,1% / 13,3% (média das 3 seeds) | 0,282 (0,228 a 0,378) / 66,4% | não avaliado | não avaliado | não avaliado | não avaliado | referência da v3; o acaso entre seeds já é de 0,15 de AUC no dev_shift | `notebooks/versions/v3_padding.ipynb` |
| v3 · zero (3 seeds) | 2026-11-09 | completar com zeros, em vez de repetir, melhora a generalização | áudio < 4 s: fala original + zeros à direita até 4 s; seeds 42, 123 e 2026 | 1,9% / 12,4% (média das 3 seeds) | 0,266 (0,146 a 0,377) / 67,8% | não avaliado | não avaliado | não avaliado | não avaliado | sem efeito demonstrado no dev_shift (Δ EER pareado −3,1 / +20,1 / −12,8 pontos); AUC abaixo de 0,5 nas seis execuções | `notebooks/versions/v3_padding.ipynb` |

Na v3, cada linha é a média das três seeds (entre parênteses, o mínimo e o máximo da AUC do dev_shift); o teste final da v3 ainda não foi avaliado (etapa separada). Na v2, dev_shift e teste são por janelas; "EER val global" é no recorte central. Recall e falso alarme (CORAA) dependem do limiar, que mudou muito entre os experimentos da v2 (0,60 / 0,10 / 0,76).

## e) Registro detalhado

### v1 · baseline

- **Data:** 2026-11-08 (commit `2bc5ca4`).
- **Hipótese:** uma CNN pequena sobre log-mel, treinada com amostragem balanceada por fonte e gerador, já separa real de fake em bancos e geradores que não viu.
- **Mudança única:** é a primeira versão (referência). Sem augmentation, de propósito.
- **Hiperparâmetros:** 16 kHz mono; recorte de 4 s (aleatório no treino, central na validação/teste; áudio curto repetido); log-mel 64 faixas (`n_fft` 512, janela 400, passo 160) padronizado por exemplo; CNN de 4 blocos Conv3×3–BatchNorm–ReLU–MaxPool (16/32/64/128 filtros), Dropout 0,3, Linear(1), 110.433 parâmetros; `BCEWithLogitsLoss`; Adam, lr 0,001; batch 64; 25.000 reais + 25.000 falsos por época, com teto por fonte e por gerador; até 30 épocas, early stopping com paciência 5 pela **EER global da validação**; seed 42; limiar = EER da validação (0,039).
- **Resultados por banco** (`results/baseline_results.csv`; teste com até 5.000 áudios por classe e banco, recorte central):

  | banco | tipo | AUC | EER | recall fake | falso alarme |
  |:--|:--|--:|--:|--:|--:|
  | validação | — | 0,998 | 1,3% | 98,7% | 1,3% |
  | in_the_wild (inteiro) | principal | 0,211 | 72,2% | 76,6% | 91,2% |
  | llamapartialspoof | principal | 0,563 | 45,4% | 82,9% | 72,7% |
  | brspeech_df | principal (pt) | 0,766 | 30,6% | 47,7% | 7,4% |
  | asvspoof2019_la (eval) | secundário | 0,971 | 6,9% | 66,6% | 0,1% |
  | asvspoof2021_df | secundário | 0,685 | 35,3% | 59,2% | 33,9% |
  | coraa | falso alarme pt-BR | — | — | — | 41,6% |

  Validação por fonte: DFADD EER 0,26%, ShiftySpeech 7,8%, CodecFake+ 32,7% (média por fonte 13,6%).
- **Conclusão honesta:** **falhou nos bancos que mais se parecem com um golpe real.** Ótimo na validação (que é 79% DFADD) e péssimo fora: no In-the-Wild ordena ao contrário (AUC 0,211, acusa 91% dos reais), no CORAA acusa 41,6% das vozes reais, no ASVspoof 2021 DF a compressão de codec derruba a AUC de 0,97 para 0,69. Diagnóstico: o modelo parece ter aprendido **"gravação limpa = real"**, porque todo real do treino é limpo (VCTK em estúdio, LibriSpeech de audiolivro). O limiar de 0,039, fixado numa validação quase toda DFADD, agravou o falso alarme. Também ficou claro que a EER global da validação esconde o desempenho fora do DFADD.
- **Arquivos:** notebook `notebooks/versions/v1_baseline.ipynb`; resultados `results/baseline_results.csv`; checkpoint `models/baseline_cnn.pt` (fora do git).
- **Tempo de treino:** 23,9 min (21 épocas; melhor época 16), RTX 4060 8 GB.

### v2 · augmentation (três experimentos: v1r, v2a, v2b)

- **Data:** 2026-11-08 (treino de cada experimento num processo separado com `scripts/treinar_experimento.py`; notebook executado duas vezes com `REUSAR = True`).
- **Hipótese:** a v1 aprendeu "gravação limpa = real". Se ensinarmos com augmentation que ruído e canal ruim não dizem nada sobre real ou fake, o falso alarme no dev_shift e no CORAA cai e a AUC do In-the-Wild sobe.
- **Mudanças (uma por experimento):**
  - **v1r:** baseline v1 retreinado; só muda a escolha do checkpoint, que passa a ser pela **EER média por fonte** da validação (antes: EER global, 79% DFADD).
  - **v2a:** v1r + normalização RMS (−26 dB) e ganho aleatório (−12 a +6 dB), cada um com chance 0,5. É um controle: a padronização do log-mel já remove o volume.
  - **v2b:** v2a + ruído branco/rosa/marrom (SNR de 5 a 30 dB), banda de telefone (8 kHz) e mu-law de 8 bits, cada um com chance 0,3. MP3/Opus ficou de fora (sem `ffmpeg`).
  - **Mudanças de avaliação (iguais para os três):** dev_shift (30% dos locutores do In-the-Wild) para decidir entre versões; teste por janelas de 4 s com passo de 2 s (média dos logits) e também pelo recorte central; In-the-Wild só `test_final`.
- **Hiperparâmetros:** os da v1, com `NUM_WORKERS` 4 (memória; ver h). Augmentation só no treino, com a mesma chance para reais e fakes. Limiar = EER da validação de cada experimento: 0,603 / 0,098 / 0,756 (janelas).
- **Resultados** (`results/experiments.csv`; janelas; cada célula: AUC · EER · recall de fake · falso alarme):

  | conjunto | tipo | v1r | v2a | v2b |
  |:--|:--|:--|:--|:--|
  | validação (todas) | decisão | 0,999 · 1,0% · 99,0% · 1,0% | 0,999 · 0,9% · 99,1% · 0,9% | 0,998 · 1,5% · 98,5% · 1,5% |
  | dev_shift (In-the-Wild) | decisão | 0,325 · 63,7% · 62,4% · 77,7% | 0,386 · 58,1% · 61,5% · 71,5% | 0,389 · 57,0% · 50,9% · 67,2% |
  | in_the_wild (test_final) | principal | 0,262 · 68,2% · 70,4% · 86,3% | 0,354 · 59,3% · 62,8% · 78,9% | 0,336 · 61,3% · 44,8% · 72,7% |
  | llamapartialspoof (janelas) | principal | 0,625 · 40,7% · 73,2% · 53,6% | 0,595 · 43,3% · 63,2% · 50,0% | 0,646 · 39,1% · 56,6% · 34,4% |
  | llamapartialspoof (central) | principal | 0,606 · 42,0% · 69,4% · 52,2% | 0,578 · 44,8% · 61,9% · 51,1% | 0,636 · 39,7% · 52,9% · 32,3% |
  | brspeech_df | principal (pt) | 0,776 · 27,9% · 73,7% · 30,3% | 0,801 · 27,5% · 53,9% · 4,8% | 0,779 · 28,9% · 76,4% · 34,7% |
  | asvspoof2019_la (eval) | secundário | 0,936 · 12,3% · 70,1% · 0,4% | 0,936 · 11,7% · 57,4% · 0,0% | 0,986 · 5,6% · 78,9% · 0,4% |
  | asvspoof2021_df | secundário | 0,716 · 33,8% · 66,0% · 33,7% | 0,701 · 34,8% · 62,2% · 33,8% | 0,750 · 31,9% · 82,2% · 37,2% |
  | coraa | falso alarme pt-BR | 34,9% | 24,3% | 29,8% |

  Validação por fonte (EER, recorte central): CodecFake+ 29,0 / 31,0 / 30,1%; DFADD 0,2 / 0,2 / 0,7%; ShiftySpeech 8,3 / 7,4 / 11,8%; média por fonte 12,5 / 12,9 / 14,2% (no histórico de treino: 12,6 / 13,0 / 14,3%).

  Teste de estresse na validação (AUC com o áudio degradado, v1r / v2a / v2b): limpo 0,965 / 0,965 / 0,961; ruído branco 10 dB 0,487 / 0,499 / 0,776; banda de telefone 0,794 / 0,838 / 0,939; mu-law 0,887 / 0,890 / 0,949.
- **O que se esperava × o que aconteceu:**
  - **v1r:** parecido com a v1, como esperado.
  - **v2a:** era para não mudar nada. A validação não mudou, mas o dev_shift subiu 0,065 e o In-the-Wild 0,09. A augmentation usa o mesmo sorteador do recorte aleatório, então o v2a vê outros trechos de 4 s desde o primeiro lote: na prática, é outra execução com outra sorte. Tomamos isso como **o tamanho do acaso do treino**.
  - **v2b:** robusto às degradações que viu (estresse), e no teste melhor no ASVspoof 2019/2021 e no LlamaPartialSpoof (~+0,05 de AUC sobre o v2a; só reportado). Mas **sem ganho no dev_shift nem no In-the-Wild além do v2a**; pior no ShiftySpeech (validação), no BRSpeech-DF e no CORAA.
- **A hipótese "limpo = real" não se confirmou:**
  1. Os reais do treino não são mais limpos que os do In-the-Wild pela faixa dinâmica; no DFADD, os fakes é que são mais limpos.
  2. Sujar um áudio faz o v1r chamar **mais** coisas de real (ruído branco: recall de fake de 87% para 1% a 3%).
  3. No dev_shift, o v1r acusa os reais mais ruidosos tanto quanto os mais limpos (75,5% × 81,0%).
  4. O v2b não melhorou o dev_shift além do controle.
- **Pistas para a AUC abaixo de 0,5** (só dev_shift e validação):
  - A inversão acontece **dentro de cada locutor**. No v1r: Churchill 0,02, Martin Luther King 0,04, Mandela 0,14, Obama 0,26. Os piores são locutores históricos, com reais em **gravação antiga**.
  - A inversão é **pior nos áudios curtos** (AUC 0,25 até 2 s, 0,44 acima de 4,75 s). 68% dos reais do In-the-Wild têm menos de 4 s e são repetidos até 4 s.
- **O que está dentro do ruído:**
  - Há uma execução por experimento, e a EER média por fonte oscila 1 a 1,6 ponto entre épocas vizinhas (2 a 2,6 no CodecFake+). v1r × v2a na validação é ruído; o v2b ~1,5 ponto pior está no limite.
  - Diferenças de até ~0,07 a 0,09 de AUC no dev_shift e no In-the-Wild **não estão demonstradas**.
  - Os intervalos bootstrap do notebook medem só a amostra de teste, não o acaso do treino.
  - Recall e falso alarme não são comparáveis entre experimentos porque os limiares são muito diferentes.
- **Escolha entre versões:** por validação + dev_shift, **nenhum experimento é claramente melhor**. Os ganhos do v2b no ASVspoof e no LlamaPartialSpoof são de teste e não entram na escolha. O teste final foi avaliado uma única vez por experimento, e todos estão reportados.
- **Conclusão em uma frase:** a augmentation de ruído e canal deixou o modelo robusto às degradações que ele viu, mas não consertou as vozes de internet (dev_shift ainda com AUC abaixo de 0,5, sem ganho além do controle), e a hipótese "limpo = real" não se confirmou.
- **Arquivos:**
  - Notebook: `notebooks/versions/v2_augmentation.ipynb`.
  - Resultados: `results/experiments.csv` (no git); os históricos por época (`results/v*_historico.csv`) e as notas por áudio (`results/v*_notas_*.parquet`) ficam fora do git.
  - Checkpoints: `models/v1r_cnn.pt`, `models/v2a_cnn.pt` e `models/v2b_cnn.pt` (fora do git).
- **Tempos de treino** (RTX 4060 8 GB, 4 workers):

  | experimento | tempo | épocas | melhor época | por época |
  |:--|--:|--:|--:|--:|
  | v1r | 23,2 min | 18 | 13 | ~77 s |
  | v2a | 33,5 min | 25 | 20 | ~80 s |
  | v2b | 34,2 min | 22 | 17 | ~93 s |

  Nenhum chegou ao teto de 30 épocas: todos pararam pelo early stopping.

### v3 - pré-registro (padding de áudio curto)

- **Escrito em:** 2026-11-09 15:01 (−03:00), branch `v3-padding`, a partir do commit `bba9b9c`. **Antes de qualquer resultado da v3** (nem os testes QUICK tinham rodado). Nada abaixo pode ser mudado depois de ver resultados; se algo mudar, a mudança vai num bloco datado separado, com o motivo.

**Etapa 0: a premissa foi medida?** Sim, em parte. Medi de novo com as notas salvas da v2 (`results/v*_notas_val.parquet` e `results/v*_notas_dev_shift.parquet`; nada de teste), por duração **original** do arquivo, avaliação por janelas, limiar = EER da validação de cada experimento:

| conjunto | faixa | n | real / fake | AUC v1r · v2a · v2b | EER v1r · v2a · v2b |
|:--|:--|--:|:--|:--|:--|
| validação | < 2 s | 886 | 270 / 616 | 0,999 · 0,998 · 0,991 | 1,1% · 1,2% · 5,2% |
| validação | 2 a < 4 s | 25.300 | 7.189 / 18.111 | 0,999 · 0,999 · 0,998 | 0,9% · 0,8% · 1,5% |
| validação | ≥ 4 s | 25.003 | 8.271 / 16.732 | 0,998 · 0,999 · 0,998 | 1,0% · 1,0% · 1,5% |
| dev_shift | < 2 s | 2.420 | 2.067 / 353 | 0,241 · 0,272 · 0,315 | 72,9% · 68,6% · 63,7% |
| dev_shift | 2 a < 4 s | 4.531 | 3.772 / 759 | 0,300 · 0,363 · 0,390 | 66,6% · 60,8% · 57,3% |
| dev_shift | ≥ 4 s | 3.663 | 2.794 / 869 | 0,425 · 0,507 · 0,443 | 56,1% · 48,6% · 53,5% |

- **No dev_shift, o desempenho é pior nos curtos** nos três experimentos da v2 (é o que o fechamento da v2 já registrava em 7.8: "AUC 0,25 até 2 s, 0,44 acima de 4,75 s"). O padrão continua **dentro do locutor**: nos 13 locutores do dev_shift com as três faixas e as duas classes, a AUC média do v1r é 0,22 (< 2 s), 0,33 (2 a 4 s) e 0,40 (≥ 4 s). Os grupos por locutor são pequenos.
- **Na validação, não há diferença** entre faixas (AUC 0,99 em todas; o v2b é um pouco pior abaixo de 2 s, com só 886 áudios).
- **O que isso NÃO mostra:** que a **repetição** é a causa. É uma associação: nos curtos do dev_shift há mais reais (85%), outros locutores e outras condições de gravação. A v2 não testou nenhuma alternativa à repetição. Por isso a v3 é um **teste do efeito do padding**, e não a correção de um problema comprovado.
- **Não encontrei** nos arquivos salvos (histórico, notebooks v1/v2, docs) o achado "CORAA com mais de 6 s com 60% a 70% de falso alarme". Não o recalculei, porque o CORAA é teste.
- **"Limpo = real" não é um diagnóstico confirmado** (ver v2) e não é premissa da v3.

**Hipótese (a testar):** repetir áudios curtos até completar 4 s pode introduzir padrões artificiais que prejudicam a generalização; completar com silêncio (zeros), mantendo o resto do pipeline, pode reduzir esse problema. A hipótese alternativa, igualmente plausível, é que **os zeros criem uma pista de duração**: zeros viram uma região de energia mínima no log-mel, que entra na padronização por exemplo; e no treino a duração difere entre as classes dentro das fontes (DFADD: 75% dos reais e 54% dos fakes entre 2 e 4 s; ASVspoof 2019: 4,6% dos reais e 14,4% dos fakes abaixo de 2 s). Não assumimos que o zero-padding é melhor. (A v1 descartou o zero-padding por esse motivo: `docs/dataset_overview.md`.)

**Desenho:**
- Referência = v1r da v2: CNNPequena, log-mel 64 faixas, 16 kHz mono, entrada de 4 s, **sem augmentation**, mesma amostragem balanceada, mesmos splits, mesmos hiperparâmetros, checkpoint pela EER média por fonte da validação, avaliação por janelas de 4 s com passo de 2 s e média dos logits.
- **v3_repeat:** áudio < 4 s é repetido até 4 s (o tratamento atual). **v3_zero:** mantém a fala original e completa **à direita com zeros** até 4 s. Áudios com 4 s ou mais: comportamento atual nos dois. Sem ruído no lugar dos zeros, sem variar a posição do padding, sem máscaras. A padronização do log-mel é a mesma nos dois.
- A política vale em tudo: treino, validação durante o treino, avaliação central e por janelas.
- **Seeds 42, 123 e 2026** nas duas condições: 6 treinos independentes, na ordem repeat/42, zero/42, repeat/123, zero/123, repeat/2026, zero/2026. A seed controla a amostragem das épocas, o recorte aleatório e a inicialização, em fontes de aleatoriedade separadas; na mesma seed, as duas condições veem os mesmos áudios e os mesmos trechos. O split train/val e as amostras de avaliação **não** dependem da seed do treino (seed de dados 42).
- Os resultados antigos (v1r, v2) são contexto; a comparação é **só** entre os controles pareados da v3, com o mesmo código.

**Critérios (fixados agora):**
1. **Seleção do checkpoint:** EER média por fonte na validação (inalterada).
2. **Comparação principal:** diferença **pareada por seed** da EER global do dev_shift (janelas): Δ = EER(zero) − EER(repeat), para cada seed; reportar os três Δ, a média e o desvio padrão.
3. **Complementares:** AUC do dev_shift (pareada da mesma forma) e EER por fonte da validação (para achar regressões); avaliação central reportada ao lado.
4. **Quando chamar de efeito:** só se os três Δ tiverem o **mesmo sinal** E |média dos Δ| for **maior que a amplitude (máx − mín) da EER do dev_shift entre as três seeds do v3_repeat**. Caso contrário: "diferença dentro do acaso entre seeds, não demonstrada". Uma média que sobe sozinha não é melhora comprovada. Regressão na validação: mesma regra, fonte por fonte.
5. **Diagnóstico por duração original** (< 2 s, 2 a < 4 s, ≥ 4 s), na validação e no dev_shift: tamanho dos grupos e proporção real/fake sempre; AUC e EER só quando a faixa tem as duas classes. Verificar se uma eventual melhora se concentra nos curtos e se há perda nos ≥ 4 s (onde o padding não muda a entrada, então diferenças ali medem o acaso ou efeitos indiretos do treino).
6. **Limiares:** fixados na validação, **pelo mesmo procedimento nos dois braços** (EER da validação inteira, por janelas, em cada execução). Recall e falso alarme usam esses limiares específicos de cada execução, e isso é dito junto do número. **Nunca** ajustar limiar no teste final.
7. **Dados:** o dev_shift serve para desenvolvimento, nunca para treino. Test_final, LlamaPartialSpoof, BRSpeech-DF, CORAA e demais testes **não** escolhem configuração nem seed.
8. **Teste final:** flag `AVALIAR_TESTE_FINAL = False` no notebook. Execução normal e QUICK não consultam o teste. A avaliação final é uma etapa separada, depois dos seis treinos e da análise de validação/dev_shift, e reporta **todas** as condições e seeds.
9. **Ressalva sobre o In-the-Wild:** o In-the-Wild inteiro foi consultado na v1, antes de existirem dev_shift e test_final. A separação melhora o procedimento daqui em diante, mas o test_final do In-the-Wild **não** é uma avaliação historicamente nunca vista.

### v3 · estado da execução (atualizado em 2026-11-09 15:30)

*(Superado pelo fechamento abaixo; mantido como registro de como estava antes dos treinos.)*

- **Resultados completos:** nenhum. Os seis treinos oficiais estão **pendentes** (rodam fora da sessão do Claude Code, com `scripts/rodar_v3.ps1`). A tabela-resumo (d) só ganha a linha da v3 quando a versão fechar.
- **Teste final:** não avaliado (`AVALIAR_TESTE_FINAL = False`).
- **Testes QUICK (2026-11-09; 2 épocas de 300 áudios, validação de 300 e dev_shift de 100 áudios; os números não valem como resultado e não devem ser interpretados):**

  | execução | épocas | EER média/fonte (val) | dev_shift EER / AUC | pico de memória | observação |
  |:--|--:|--:|:--|--:|:--|
  | `v3_repeat_s42` (QUICK) | 2 | 32,0% | 38,0% / 0,667 | 7,4 GB | ~0,8 min |
  | `v3_zero_s42` (QUICK) | 2 | 25,7% | 54,0% / 0,456 | 7,2 GB | interrompido de propósito após a época 1 (processo e workers mortos) e retomado na época 2 |

- **O que os testes conferiram:** entrada mono com 64.000 amostras nas duas condições; em 150 áudios de treino/validação (16 e 24 kHz), `repeat` é idêntico bit a bit ao código da v2 (`bba9b9c`), `zero` preserva a fala e completa com zeros, e áudios com 4 s ou mais são idênticos nas duas condições; o recorte aleatório é o mesmo nas duas condições na mesma seed; checkpoints carregam com a política certa (`SHORT_AUDIO_MODE` no checkpoint); notas sem NaN/infinito; arquivos QUICK com sufixo `_quick`, separados dos oficiais; após a interrupção, o mesmo comando retomou da época 2, e uma terceira chamada não fez nada (marcador presente); sem marcador e sem as notas do dev_shift, o script pulou treino e validação e recalculou o dev_shift com notas idênticas às anteriores; o notebook v3 rodou inteiro em QUICK sem erro e sem consultar o teste; `scripts/rodar_v3.ps1` foi testado numa cópia apontando para os marcadores QUICK (pulou as concluídas, registrou falhas e seguiu, escreveu o resumo).
- **Não verificado:** determinismo bit a bit na GPU entre duas execuções completas; o caminho de compatibilidade `--exp ... --notebook notebooks/versions/v2_augmentation.ipynb` do script (mantido, mas não executado nesta sessão).

### v3 · fechamento (2026-11-09)

- **Data:** treinos em 2026-11-09, de 16:53 a 20:28 (−03:00), um processo por execução, com `scripts/rodar_v3.ps1`; análise escrita no mesmo dia. Branch `v3-padding`, commit `bba9b9c` com mudanças não commitadas (as mesmas nos seis marcadores). **Teste final não avaliado** (`AVALIAR_TESTE_FINAL = False`).
- **Hipótese e mudança:** as do pré-registro acima (repetir × completar com zeros, 3 seeds, comparação pareada). Os critérios foram aplicados **sem mudança**.
- **Resultados por execução** (validação inteira, 51.189 áudios; dev_shift inteiro, 10.614 áudios; janelas de 4 s com passo de 2 s; arquivos `results/v3_*_concluido.json` e `results/v3_*_notas_*.parquet`):

  | execução | épocas (melhor) | val: EER média por fonte (checkpoint) | val: EER global (janelas) | dev_shift EER | dev_shift AUC | min de treino | pico de memória |
  |:--|:--|--:|--:|--:|--:|--:|--:|
  | `v3_repeat_s42` | 14 (9) | 14,1% | 1,2% | 69,6% | 0,241 | 35,9 | 13,15 GB |
  | `v3_zero_s42` | 12 (7) | 13,7% | 2,2% | 66,5% | 0,274 | 28,2 | 13,14 GB |
  | `v3_repeat_s123` | 17 (12) | 13,2% | 1,7% | 59,2% | 0,378 | 41,2 | 13,16 GB |
  | `v3_zero_s123` | 24 (19) | 11,2% | 1,0% | 79,4% | 0,146 | 33,3 | 13,12 GB |
  | `v3_repeat_s2026` | 14 (9) | 12,6% | 1,2% | 70,5% | 0,228 | 19,0 | 13,13 GB |
  | `v3_zero_s2026` | 16 (11) | 12,3% | 1,1% | 57,7% | 0,377 | 45,2 | 13,11 GB |

  Média ± desvio por condição (`repeat` × `zero`): EER média por fonte 13,3 ± 0,7% × 12,4 ± 1,2%; EER global da validação (janelas) 1,4 ± 0,3% × 1,4 ± 0,6%; dev_shift EER 66,4 ± 6,3% × 67,8 ± 10,9%; dev_shift AUC 0,282 ± 0,083 × 0,266 ± 0,116. Validação por fonte (EER, janelas, média das seeds): CodecFake+ 29,5% × 26,3%; DFADD 0,20% × 0,25%; ShiftySpeech 7,4% × 8,1%. Limiares (probabilidade, EER da validação por janelas): 0,64 / 0,79 / 0,63 no `repeat` e 0,92 / 0,13 / 0,96 no `zero` (seeds 42 / 123 / 2026), por isso recall e falso alarme não são comparados.

- **Comparação pareada (critérios 2 a 4 do pré-registro):**

  | critério | Δ = zero − repeat (seeds 42 / 123 / 2026) | média (desvio) | régua: amplitude do `repeat` | resultado pela regra |
  |:--|:--|--:|--:|:--|
  | **principal:** EER do dev_shift (janelas) | −3,1 / +20,1 / −12,8 pontos | +1,4 (16,9) pontos | 11,3 pontos | **dentro do acaso** (sinais trocados) |
  | AUC do dev_shift (janelas) | +0,033 / −0,232 / +0,149 | −0,017 (0,195) | 0,149 | **dentro do acaso** (sinais trocados) |
  | EER do dev_shift (central) | −2,3 / +20,1 / −11,7 pontos | +2,0 pontos | 11,8 pontos | dentro do acaso |
  | val: EER média por fonte (checkpoint) | −0,4 / −2,0 / −0,4 pontos | −0,9 pontos | 1,5 pontos | dentro do acaso (mesmo sinal, mas abaixo da régua) |
  | val: CodecFake+ (EER, janelas) | −2,9 / −3,9 / −2,5 pontos | −3,1 pontos | 2,3 pontos | **passa a regra: `zero` melhor** |
  | val: DFADD (EER, janelas) | +0,2 / −0,2 / +0,1 pontos | +0,05 pontos | 0,3 pontos | dentro do acaso |
  | val: ShiftySpeech (EER, janelas) | +4,3 / −1,3 / −0,9 pontos | +0,7 pontos | 0,7 pontos | dentro do acaso |

  **A diferença no dev_shift está dentro do ruído; nenhuma regressão na validação.** O CodecFake+ da validação passa a regra a favor de `zero`, mas é um critério complementar, tem só 352 áudios (164 reais, 188 fakes) e a melhora aparece também nos ≥ 4 s (EER 23,2 → 17,4%, 18,7 → 14,9%, 18,3 → 15,4%), onde a entrada é igual nas duas condições: se for real, é um efeito indireto do treino, não da entrada.

- **Diagnóstico por duração original** (critério 5; janelas; média das 3 seeds, `repeat` × `zero`; todas as faixas têm as duas classes):

  | conjunto | faixa | n | real / fake | % fake | AUC | EER | Δ AUC por seed (42 / 123 / 2026) |
  |:--|:--|--:|:--|--:|:--|:--|:--|
  | validação | < 2 s | 886 | 270 / 616 | 69,5% | 0,997 × 0,992 | 2,8% × 4,7% | −0,009 / −0,001 / −0,003 |
  | validação | 2 a < 4 s | 25.300 | 7.189 / 18.111 | 71,6% | 0,999 × 0,999 | 1,1% × 1,1% | −0,001 / 0,000 / 0,000 |
  | validação | ≥ 4 s | 25.003 | 8.271 / 16.732 | 66,9% | 0,997 × 0,998 | 1,6% × 1,4% | −0,001 / +0,002 / 0,000 |
  | dev_shift | < 2 s | 2.420 | 2.067 / 353 | 14,6% | 0,215 × 0,371 | 72,9% × 61,5% | +0,151 / −0,128 / +0,444 |
  | dev_shift | 2 a < 4 s | 4.531 | 3.772 / 759 | 16,8% | 0,265 × 0,262 | 68,2% × 68,2% | +0,046 / −0,214 / +0,159 |
  | dev_shift | ≥ 4 s | 3.663 | 2.794 / 869 | 23,7% | 0,355 × 0,216 | 60,7% × 71,3% | −0,104 / −0,334 / +0,018 |

  Na média, `zero` é melhor nos curtos do dev_shift e **pior nos ≥ 4 s** (onde a entrada não muda), mas os Δ por seed trocam de sinal nas três faixas. Na validação, `zero` é um pouco pior abaixo de 2 s nas três seeds (EER +3,3 / +0,3 / +2,2 pontos; 886 áudios) e igual no resto. É uma pista, não um efeito.

- **Observações registradas:**
  - **As seis AUCs do dev_shift ficaram abaixo de 0,5** (0,146 a 0,378): nas duas condições e em todas as seeds, o modelo continua ordenando o In-the-Wild ao contrário. Nada foi invertido.
  - **A época 13 do `v3_zero_s2026` levou 1.331 s** (as vizinhas, ~80 s; a época 15 levou 241 s). Foi lentidão do computador, sem interrupção nem retomada; **sem efeito nos resultados**, porque o tempo não entra em nenhuma conta do treino.
  - As três primeiras execuções (`repeat_s42`, `zero_s42`, `repeat_s123`) tiveram épocas de ~150 a 165 s; as três últimas, ~80 s. Causa não medida.
  - **Pico de memória de 13,1 a 13,2 GB** por execução (processo + workers) e de 34,5 a 36,7 GB no sistema, acima dos 7,3 a 7,8 GB medidos na v2 (seção h). Causa não investigada; nenhum treino caiu.
  - Nenhuma execução foi retomada (`treino_retomado_nas_epocas` vazio nos seis marcadores). Todas pararam pelo early stopping (melhor época + 5).
- **A régua do ruído (referência para as próximas versões):** com **a mesma receita e só a seed mudando**, a AUC do dev_shift (janelas) foi de **0,228 a 0,378 no `repeat`** (amplitude 0,149) e de **0,146 a 0,377 no `zero`** (amplitude 0,232); as seis juntas, de 0,146 a 0,378. A EER do dev_shift foi de 59,2% a 70,5% (`repeat`, 11,3 pontos) e de 57,7% a 79,4% (`zero`, 21,7 pontos). Na validação, a régua é bem menor: 1,5 pontos na EER média por fonte do checkpoint, 2,3 no CodecFake+, 0,7 no ShiftySpeech e 0,3 no DFADD (amplitude do `repeat`). Consequências:
  - Uma diferença de AUC no dev_shift menor que **~0,15** entre duas execuções únicas não está demonstrada. A estimativa de acaso da v2 (~0,06, do v2a) estava subestimada; as diferenças entre v1r, v2a e v2b (0,325 / 0,386 / 0,389) cabem nessa régua.
  - Comparar versões daqui em diante pede várias seeds por condição, com a mesma regra pareada.
- **Conclusão em uma frase:** trocar a repetição por zeros não mudou o dev_shift de forma demonstrável (Δ EER pareado −3,1, +20,1 e −12,8 pontos, sinais trocados), as seis execuções continuam com AUC abaixo de 0,5, e o que a v3 mais mostrou foi o tamanho do acaso entre seeds (AUC do dev_shift de 0,146 a 0,378 com a mesma receita).
- **Escolha:** pela regra, nenhuma condição é melhor. Qual delas é a referência da v4 fica para o grupo decidir (sugestão: `repeat`, o padrão até aqui).
- **Arquivos:**
  - Notebook: `notebooks/versions/v3_padding.ipynb` (com as saídas; o `notebook.ipynb` foi executado duas vezes com `AVALIAR_TESTE_FINAL = False`).
  - Resultados: `results/experiments.csv` (72 linhas da v3: validação inteira, cada fonte e dev_shift, por janelas e central; as linhas da v2 não mudaram). Fora do git: `results/v3_*_concluido.json`, `results/v3_*_historico.csv`, `results/v3_*_notas_val.parquet`, `results/v3_*_notas_dev_shift.parquet` e os logs `logs/v3_*.log` e `logs/v3_resumo.txt`.
  - Checkpoints: `models/v3_<condição>_s<seed>_cnn.pt` (fora do git).
- **Tempos de treino** (RTX 4060 8 GB, 4 workers; minutos = soma das épocas; o processo inteiro inclui as notas de validação e dev_shift, ~1,3 min):

  | execução | treino | processo | épocas | melhor época | por época (mediana) | época mais lenta |
  |:--|--:|--:|--:|--:|--:|--:|
  | `v3_repeat_s42` | 35,9 min | 38,9 min | 14 | 9 | 155 s | 164 s |
  | `v3_zero_s42` | 28,2 min | 30,7 min | 12 | 7 | 153 s | 166 s |
  | `v3_repeat_s123` | 41,2 min | 43,1 min | 17 | 12 | 165 s | 174 s |
  | `v3_zero_s123` | 33,3 min | 34,9 min | 24 | 19 | 82 s | 108 s |
  | `v3_repeat_s2026` | 19,0 min | 20,5 min | 14 | 9 | 82 s | 90 s |
  | `v3_zero_s2026` | 45,2 min | 46,6 min | 16 | 11 | 81 s | 1.331 s |

  Total: 3,4 h de treino (3,6 h de processo). A execução do notebook (só leitura de notas, sem treino e sem teste) levou ~17 s, com pico de ~2,5 GB.

## f) Ideias ainda não testadas

**Da v3 (para a v4):**
- **Modelo pré-treinado congelado (XLS-R 300M) com cabeça linear.** ⚠️ Fora da trilha: a trilha faz transfer learning (Aula 10) com um modelo de imagem; aqui seria um modelo de fala auto-supervisionado, usado só como extrator de características.
- **Remoção de silêncio nas bordas** do áudio antes do recorte. ⚠️ Fora da trilha: corte por limiar de energia.
- **Real espontâneo e variado, com fakes re-sintetizados do mesmo áudio** (o fake e o real de cada par têm o mesmo conteúdo, locutor e gravação de origem).
- **Áudio sem padding (cada clipe no seu comprimento).** ⚠️ Fora da trilha: entrada de tamanho variável, com média sobre o tempo antes da camada final.

**Da v2 (para a v3):**
- *(Feito na v3: 3 seeds por condição. A régua do ruído está no fechamento da v3.)* **Repetir cada experimento com 2 ou 3 seeds** (~30 min cada) antes de comparar versões. É a única forma de separar efeito de acaso; na v2, o controle mexeu no dev_shift tanto quanto a mudança testada.
- **Augmentation de gravação antiga / sala:** reverberação sintética, banda estreita com chiado de fita, filtro de rádio AM. É o tipo de real que o modelo mais acusa (locutores históricos do dev_shift).
- *(Testado na v3: completar com zeros, sem efeito demonstrado. "Avaliar só a parte que existe" continua não testado.)* **Tratar áudio curto sem repetição** (completar com silêncio, ou avaliar só a parte que existe), já que a inversão é pior nos áudios curtos e 68% dos reais do In-the-Wild têm menos de 4 s. Uma mudança só, decidida no dev_shift.
- **Agregar as janelas pelo máximo** em vez da média, para os fakes parciais. Hoje não há fakes parciais na validação para decidir isso sem o teste.
- **Treinar por mais épocas com mais paciência** (o v2b ainda tinha perda de treino 0,17 ao parar) ou **reduzir o learning rate ao longo do treino** (por exemplo, dividir por 10 quando a validação parar de melhorar). ⚠️ Fora da trilha: a trilha usa Adam com taxa fixa e early stopping. Atenção: só "mais épocas" não basta, porque o early stopping parou todos antes do teto de 30.
- **Calibrar ou fixar o limiar de outra forma:** os limiares da v2 foram de 0,10 a 0,76, o que torna recall e falso alarme instáveis entre versões (por exemplo, fixar no dev_shift, ou pelo custo de cada erro).

**Já listadas:**
- Augmentation de **compressão MP3/Opus** (precisa de `ffmpeg`, que não está instalado; alternativa sem ffmpeg: o `soundfile` 0.14 / libsndfile 1.2.2 deste env grava MP3 e OGG).
- **Reverberação** (sala) como augmentation, sem arquivo externo (resposta ao impulso sintética).
- **Mais reais "sujos" no treino** (fala espontânea, telefone, internet) de bancos que não estão nos testes.
- **Mais geradores modernos no treino** (TTS e clonagem recentes), de bancos que não estão nos testes.
- **Modelo maior ou pré-treinado em fala** (transfer learning, Aula 10, com um modelo de áudio no lugar do ImageNet).
- **Teste pt-BR mais amplo** (ParlaSpoof-BR, MLAAD pt, FakeBrAccent).

## g) Como reproduzir

Na raiz do repositório, no PowerShell:

```powershell
env\Scripts\activate
python scripts/check_data.py            # confere se todos os áudios do manifest estão em disco (deve dizer OK)
python scripts/build_manifest.py        # só se as regras de divisão mudarem (o manifest já vem no git)
python scripts/check_flac_decoding.py   # opcional: confere a decodificação de todos os FLAC
code notebook.ipynb                     # abre a versão atual no VS Code; escolha o kernel "Python 3 (env)"
```

**v3 (seis execuções, retomável):** feche o Claude Code (e, se der, o VS Code e o OneDrive) e rode, na raiz:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rodar_v3.ps1                 # as seis, pulando as concluídas
python scripts/treinar_experimento.py --condicao zero --seed 123               # ou uma só (com o env ativo)
python scripts/treinar_experimento.py --condicao repeat --seed 42 --quick      # teste rápido (arquivos _quick)
```

Logs em `logs\v3_<condição>_s<seed>.log`, resumo em `logs\v3_resumo.txt`, erros em `logs\v3_erros.txt`. Depois, abra o notebook e execute tudo (`AVALIAR_TESTE_FINAL` continua `False` até a análise de validação/dev_shift estar escrita).

O env não tem `jupyterlab` instalado; o notebook é aberto no VS Code (extensão Jupyter) com o kernel do `env`.

**Rodada completa de uma versão (v2 em diante), pelo terminal:** um experimento por processo, e depois o notebook, que só lê o que foi treinado. Feche antes o Claude Code e, se der, o VS Code e o OneDrive (juntos eles reservam ~6 GB de memória; ver h).

```powershell
env\Scripts\activate
$env:OPENBLAS_NUM_THREADS = "1"                                   # ~0,7 GB a menos por processo (ver h)
python scripts/treinar_experimento.py --exp v1r                   # um processo por experimento
python scripts/treinar_experimento.py --exp v2a
python scripts/treinar_experimento.py --exp v2b
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb
```

A 1ª execução do notebook calcula as notas (o teste final, uma vez por experimento) e grava `results/experiments.csv`; a 2ª lê tudo do disco e só serve para a tabela "Histórico", no topo, mostrar a versão nova. Para testar o caminho todo em minutos, use `--quick` no script e `QUICK = True` no notebook (arquivos com sufixo `_quick`, que não valem como resultado). Com `REUSAR = True`, checkpoints, históricos e notas já calculados são lidos de `models/` e `results/` em vez de recalculados; para refazer um experimento, apague o `models/<exp>_cnn.pt` e o `results/<exp>_historico.csv` dele.

## h) Mudanças de execução (não alteram o método)

Registro de mudanças em **como** o código roda (memória, processos, leitura), separadas das mudanças de experimento.

### 2026-11-08 · memória no Windows (v2)

- **O que aconteceu:** a primeira execução completa da v2 foi derrubada por falta de memória no meio do primeiro treino (v1r). Nenhum resultado foi aproveitado: o checkpoint parcial foi apagado.
- **Causa medida** (`scripts/profile_memory.py`, 3 a 6 épocas pequenas, com psutil):
  - No Windows, o limite é a **memória reservada (commit) = RAM + pagefile ≈ 34 GB**, e não os 31,5 GB de RAM. Sem nenhum Python aberto, o resto do sistema (VS Code, OneDrive, Edge, sessões do Claude Code…) já reservava **~26 GB**.
  - Cada processo de leitura (worker) do DataLoader reservava **~1,7 GB**: **~0,7 GB do OpenBLAS do numpy** (um buffer por thread, com as 22 threads do PC, reservado ao importar o numpy) e **~0,9 GB das DLLs de CUDA** que o `import torch` carrega. O processo principal reservava ~5,2 GB.
  - No notebook havia 8 workers no treino **e** 8 workers **persistentes** na validação, vivos ao mesmo tempo: ~16 × 1,7 + 5,2 ≈ **32 GB**.
  - **A memória não crescia** de época para época (média por worker de −5 a −7 MB por época; principal estável). O Dataset que cada worker recebe tem só 29 MB (caminhos e rótulos em lista/array, sem o DataFrame); não há cache de áudio nem tensores acumulados entre épocas.
- **Correções:**
  1. `src/dataset.py: limitar_threads_dos_workers()`, chamada na configuração do notebook: os workers usam 1 thread de álgebra linear (`OPENBLAS/OMP/MKL_NUM_THREADS=1`). Os workers só leem e cortam áudio e já rodavam com `torch.set_num_threads(1)`. **Lotes conferidos idênticos bit a bit** com e sem a correção (com a augmentation do v2b).
  2. DataLoaders de treino e de validação **não persistentes** (`src/experimento.py`): no máximo `NUM_WORKERS` workers vivos ao mesmo tempo. Validação não tem sorteio, então não muda nada.
  3. **`NUM_WORKERS` de 8 para 4.** Com 6, o pico subia para 9,9 GB e a época ficava um pouco mais lenta (abrir cada worker custa caro no Windows).
  4. `OPENBLAS_NUM_THREADS=1` também no processo principal (no terminal, antes de rodar; `scripts/treinar_experimento.py` já faz isso sozinho): principal de 5,2 para 3,7 GB. O principal não usa álgebra linear na CPU para nenhum resultado (o treino roda na GPU).
  5. **Um experimento por processo** (`scripts/treinar_experimento.py`): o script executa a célula de configuração do notebook (tag `config`) e o mesmo `src/experimento.py` que o notebook usa, sem copiar hiperparâmetros. O notebook, com `REUSAR = True`, só lê os checkpoints.
- **Memória medida depois:** treino de um experimento **~7,3 a 7,8 GB** (4 workers × ~1,0 GB + principal ~3,7 GB), igual com e sem augmentation; notebook (avaliação) **~8,8 GB** no teste rápido.
- **Isso altera os resultados?** As correções 1, 2 (validação), 4 e 5 **não alteram nenhum número**. A correção 3 (e o treino não persistente) **não altera o método**: mesmas épocas, mesmos áudios sorteados por época, mesma regra de recorte e de augmentation. Mas muda **quais** sorteios aleatórios cada áudio recebe (qual trecho de 4 s, qual ruído), porque cada worker tem a sua sequência aleatória. Conferimos: com 4 e com 6 workers, os lotes saem diferentes. Por isso uma rodada com 4 workers não é idêntica bit a bit a uma com 8. É uma variação do mesmo tipo que a do acaso do treino na GPU. Os três experimentos da v2 usam a mesma configuração, então a comparação entre eles é justa.

### 2026-11-09 · execução retomável e recorte por seed (v3)

- **Recorte aleatório com fonte própria** (`src/dataset.py`, `seed_recorte`; ligado por `RECORTE_DETERMINISTICO = True` na v3): a posição do recorte de 4 s no treino passa a ser sorteada por um gerador que depende só de (seed, época, índice do áudio), e não mais do gerador do torch de cada worker. **Muda quais trechos são sorteados** em relação à v2 (o método é o mesmo: posição uniforme), por isso o `v3_repeat` não é idêntico ao v1r e os controles são retreinados. Com isso, na mesma seed, `repeat` e `zero` recebem os mesmos trechos, e o resultado não depende do número de workers. Com `seed_recorte=None` (padrão), o comportamento da v2 é mantido.
- **Seed de treino separada da seed dos dados:** `cfg["seed"]` controla amostragem das épocas, recorte e inicialização; a amostra de validação/dev_shift/teste continua na seed 42.
- **Treino retomável** (`src/train.py`, `models/<exp>_estado.pt`): no fim de cada época grava modelo, Adam, melhor modelo, sorteador e geradores aleatórios; rodar de novo continua da época seguinte. A retomada segue o mesmo método, mas não é garantidamente idêntica bit a bit a uma execução sem interrupção (fica registrada no marcador, em `treino_retomado_nas_epocas`).
- **Gravação atômica** (`src/arquivos.py`): checkpoint, estado, histórico, notas e marcador são gravados num temporário e renomeados.
- **Marcador de conclusão** (`results/<exp>_concluido.json`), gravado por último: só com ele a execução conta. Registra configuração (`config_id`), hash do manifest, commit, arquivos com mudança não commitada e o sha256 de cada arquivo de código, `NUM_WORKERS`, tempos, pico de memória e o resumo de validação/dev_shift.
- **O script de treino calcula as notas da validação e do dev_shift** (nunca do teste); o notebook só lê.
- `NUM_WORKERS` continua 4 (sobrescrevível com `--num-workers`); fica fora do `config_id`, porque não muda o método.
