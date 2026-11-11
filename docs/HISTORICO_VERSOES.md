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

| v4 · S2 `concat_top3` (final) | 2026-11-10 | representações de um modelo de fala pré-treinado e congelado generalizam melhor para vozes de internet que a CNN | XLS-R 300M congelado (camadas 5, 6, 7, média no tempo) + regressão logística; corte de silêncio; até 8 s sem completar | 1,0% / 5,9% | 0,983 / 5,9% | 0,979 / 6,8% | 0,925 / 14,7% | 0,842 / 24,9% | 16,7% (lim. A) · 2,5% (lim. B) | resolveu a inversão das vozes de internet; o ganho sobre o S1 não está demonstrado; fraco em pt-BR | `notebooks/versions/v4_ssl.ipynb` |
| v4 · S1 (só camada 24) | 2026-11-10 | (referência: "só transfer learning") | a última camada do XLS-R + regressão logística (C pela validação) | 2,6% / 8,6% | 0,984 / 5,8% | 0,986 / 5,4% | 0,936 / 14,0% | 0,775 / 30,7% | 35,6% · 11,6% | o modelo congelado sozinho já explica quase todo o ganho | `notebooks/versions/v4_ssl.ipynb` |
| v4 · S3 (S2 + CNN) | 2026-11-10 | juntar a CNN da v3 ajuda | regressão de 2 entradas (S2 e S0) treinada na validação | 0,7% (no próprio treino) / — | 0,943 / 10,9% | 0,888 / 18,8% | 0,900 / 17,8% | 0,840 / 25,3% | 31,8% · 8,7% | não demonstrado pela regra, mas piorou o In-the-Wild (−0,09 de AUC): não usar | `notebooks/versions/v4_ssl.ipynb` |
| v4 · S0 = `v3_zero_s123` | 2026-11-10 | (referência: a CNN da v3 com menor EER média por fonte na validação) | CNN da v3, janelas de 4 s | 2,1% / 11,2% | 0,146 / 79,4% | 0,114 / 82,2% | 0,556 / 46,3% | 0,735 / 29,3% | 65,3% · 41,3% | no teste, igual ao dev_shift: ordena o In-the-Wild ao contrário | `notebooks/versions/v4_ssl.ipynb` |
| v5 · X2 = XLS-R refeito (escolhido) | 2026-11-10 | (referência pareada da v5: o S2 da v4 refeito noutra placa) | o mesmo `concat_top3` da v4 (camadas 5, 6, 7; C = 0,1) | 1,1% / 5,9% | 0,983 / 6,0% | 0,978 / 6,8% | 0,924 / 14,8% | 0,842 / 25,0% | 16,9% (lim. A) · 2,5% (lim. B) | reproduziu a v4 (AUC do teste a até 0,0008 do S2); segue como o modelo do projeto | `notebooks/versions/v5_wavlm.ipynb` |
| v5 · W2 = WavLM Large | 2026-11-10 | um modelo pré-treinado também a "limpar" fala com ruído (WavLM, Microsoft, ~94 mil h de inglês) generaliza melhor para vozes de internet que o XLS-R (Meta) | só o modelo congelado; o resto igual à v4 (cabeça final pela validação: `concat_top3`, camadas 4, 6, 8; C = 0,1) | 1,1% / 6,8% | 0,970 / 7,5% | 0,950 / 11,4% | 0,914 / 14,5% | 0,768 / 32,3% | 34,5% · 8,5% | pior que o XLS-R em AUC nos 5 bancos com duas classes (demonstrado no teste); na decisão (dev_shift), empate pela regra; muito pior em pt-BR | `notebooks/versions/v5_wavlm.ipynb` |
| v5 · W1 = WavLM, só camada 24 | 2026-11-10 | (referência: "só transfer learning" com o WavLM) | a última camada do WavLM + regressão logística (C = 1, pela validação) | 6,5% / 15,1% | 0,956 / 10,0% | 0,946 / 10,3% | 0,899 / 17,9% | 0,557 / 45,3% | 87,3% · 73,6% | a última camada do WavLM quase não serve para pt-BR (BRSpeech-DF perto do acaso) | `notebooks/versions/v5_wavlm.ipynb` |
| v6 · F2 = sistema final do funil | 2026-11-10 | escolher front-end, cabeça e ajuste pela AUC da **pior** fonte (val + dev_shift) acha um sistema melhor que a v4 | funil E1–E5: front-end só Meta (WavLM e fusão empataram), cabeça RL (MLP e CNN empataram), camada 11 sozinha, C = 1, pesos iguais por fonte, limiar A | 2,3% / 7,3% (pior fonte 0,910) | 0,972 / 6,1% | 0,972 / 8,3% | 0,937 / 13,6% | 0,744 / 32,7% | 40,4% (lim. A) · 17,3% (lim. B) | pior que a v4 no teste (média da acurácia nos 3 principais 0,829 × 0,851), sobretudo em pt-BR: a pior fonte foi sempre o CodecFake+ (352 clipes) e ela não protege o português | `notebooks/versions/v6_funil.ipynb` |
| v6 · F1 = só Meta + RL básica | 2026-11-10 | (referência: o front-end vencedor com a cabeça mais simples) | XLS-R, camada 11 sozinha, C = 1 | 2,2% / 7,6% (pior fonte 0,904) | 0,972 / 6,1% | 0,972 / 8,2% | 0,940 / 13,2% | 0,741 / 33,3% | 41,2% · 17,4% | igual ao F2 no teste (o ajuste fino da E5 não fez diferença fora da validação) | `notebooks/versions/v6_funil.ipynb` |
| v6 · F0 = v4 refeita | 2026-11-10 | (referência: a v4 recalculada nesta rodada) | XLS-R `concat_top3` (5, 6, 7), C = 0,1 | 1,0% / 5,9% (pior fonte 0,903) | 0,983 / 5,9% | 0,979 / 6,8% | 0,925 / 14,7% | 0,842 / 24,9% | 16,7% · 2,5% | reproduziu a v4 exatamente (validação, dev_shift e teste); continua o modelo do projeto | `notebooks/versions/v6_funil.ipynb` |

Na v5, os clipes de validação, dev_shift e teste são exatamente os da v4 (mesma seed), e os dois modelos rodaram na mesma máquina (RTX 4050 6 GB); a escolha entre eles foi feita na validação + dev_shift, antes do teste. Nas linhas da v4, "EER val" é na validação da v4 (19.999 clipes, subamostra estratificada); o teste usa até 5.000 clipes por classe e banco, avaliado uma vez; o falso alarme do CORAA vem com os dois limiares pré-registrados (A = EER da validação; B = EER do dev_shift). As outras cinco execuções da v3 não foram avaliadas no teste.

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

### v4 - pré-registro (modelo auto-supervisionado congelado + cabeças leves)

- **Escrito em:** 2026-11-09 21:52 (−03:00), branch `v4-ssl`, a partir do commit `cbfb649` (v3 fechada). **Antes de qualquer resultado da v4** (nem os testes QUICK tinham rodado; só o modelo tinha sido baixado e carregado). Nada abaixo pode ser mudado depois de ver resultados; se algo mudar, a mudança vai num bloco datado separado, com o motivo.
- **Hipótese:** a CNN pequena sobre log-mel aprendeu o estilo de gravação do real do treino (fala lida e limpa: audiolivro, VCTK). Representações de um modelo de fala auto-supervisionado pré-treinado em muita fala variada (XLS-R 300M, congelado), com cabeças leves por cima, podem separar real de fake por pistas que generalizam melhor para vozes de internet (dev_shift). ⚠️ Fora da trilha: o modelo auto-supervisionado em si; usar um modelo pré-treinado congelado como extrator é transfer learning (Aula 10).
- **Mudanças agrupadas nesta versão** (regra nova do CLAUDE.md: agrupar o que dá para fazer junto e medir cada parte com ablações baratas dentro da versão): front-end XLS-R 300M congelado no lugar de log-mel + CNN; remoção de silêncio nas bordas; recorte central de até 8 s sem completar (nem repetição nem zeros); escolha de camada; cabeças leves; diagnóstico de atalhos; métricas de acurácia.
- **Critérios (fixados agora):**
  - **(a) Escolha da camada e da cabeça final:** pela **EER média por fonte na validação** (só fontes com real e fake). O dev_shift é reportado para **todas** as camadas e cabeças, mas **não escolhe** camada, cabeça, C nem seed (são só 16 locutores). O C da regressão logística de cada camada também é escolhido pela EER média por fonte da validação. Nas cabeças com 3 seeds, a comparação usa a média das 3 seeds na validação.
  - **(b) Comparação principal entre versões:** AUC e EER do dev_shift e acurácia balanceada do dev_shift (limiar da validação).
  - **(c) O que conta como melhora real:** AUC do dev_shift **acima de 0,5 nas três seeds** da cabeça final; entre 0,5 e 0,6 é **melhora fraca**; acima de 0,6 é **melhora clara**; igual ou abaixo de 0,5, **não resolveu**. Sempre comparado com a régua do ruído da v3 (AUC do dev_shift de 0,146 a 0,378 nas seis execuções; amplitude de ~0,15 a 0,23 entre seeds). Se a cabeça final for uma regressão logística (solver determinístico, convexo), o resultado não depende de seed: vale o valor único, e a regra das três seeds é conferida também nas cabeças de soma ponderada.
  - **(d) Limiar:** fixado na validação (ponto de EER da validação inteira), **nunca** no dev_shift nem no teste. Acurácias também reportadas com limiar 0,5 (logit 0) e ao lado da acurácia trivial ("sempre a classe majoritária").
  - **(e) Se o AUC do dev_shift continuar abaixo de 0,5 mesmo com o modelo pré-treinado**, isso indica **problema de dados** (o que o treino chama de real e de fake), e não de modelo, e deve ser dito assim na conclusão.
  - **(f) Atalhos:** se a regressão logística só com estatísticas baratas (duração, silêncio inicial e final, RMS, pico, largura de banda) passar bem do acaso (AUC da validação acima de ~0,6), há atalho nos dados, e isso entra na conclusão.
  - **(g) Dados:** o dev_shift só avalia; o teste final não é avaliado nesta execução (`AVALIAR_TESTE_FINAL = False`). In-the-Wild test_final, LlamaPartialSpoof, BRSpeech-DF, CORAA, ASVspoof 2019 eval e 2021 DF **não** escolhem nada.

### v3 · acurácias retroativas (Parte A da v4, 2026-11-09)

Calculadas por `scripts/acuracias_v3.py` **só com as notas já salvas** da v3 (`results/v3_*_notas_val.parquet` e `results/v3_*_notas_dev_shift.parquet`; nada retreinado, nenhum teste lido), por janelas, como no fechamento. Arquivo: `results/v3_acuracias.csv`. AUC e EER conferem com o fechamento da v3.

| execução | val: acurácia / balanceada (limiar val) | dev_shift: acurácia / balanceada (limiar val) | dev_shift: acurácia / balanceada (0,5) | dev_shift: acurácia trivial |
|:--|:--|:--|:--|--:|
| `v3_repeat_s42` | 0,988 / 0,988 | 0,216 / 0,434 | 0,207 / 0,439 | 0,813 |
| `v3_zero_s42` | 0,978 / 0,978 | 0,212 / 0,400 | 0,196 / 0,479 | 0,813 |
| `v3_repeat_s123` | 0,983 / 0,983 | 0,312 / 0,503 | 0,273 / 0,515 | 0,813 |
| `v3_zero_s123` | 0,990 / 0,990 | 0,164 / 0,365 | 0,155 / 0,275 | 0,813 |
| `v3_repeat_s2026` | 0,988 / 0,988 | 0,191 / 0,432 | 0,186 / 0,442 | 0,813 |
| `v3_zero_s2026` | 0,989 / 0,989 | 0,343 / 0,444 | 0,222 / 0,471 | 0,813 |

A acurácia trivial da validação é 0,693 (69% fake). No dev_shift, **a acurácia (0,16 a 0,34) fica muito abaixo da trivial (0,81)** e a balanceada (0,37 a 0,50) não passa do acaso: o mesmo diagnóstico da AUC abaixo de 0,5, numa escala mais fácil de explicar. **Régua do ruído em acurácia balanceada (dev_shift, limiar da validação): 0,365 a 0,503.**

### v4 · estado da execução (atualizado em 2026-11-09 22:40)

*(Superado pelo fechamento da v4, mais abaixo; mantido como registro de como estava antes da execução.)*

- **Resultados completos:** nenhum ainda. A execução oficial roda **fora desta sessão**, pelo vigia (`scripts/vigia_v4.ps1`, lançado às 22:38 oculto, PID 30428) → `scripts/rodar_v4.ps1` → `scripts/v4.py <etapa>`. Conferir: `Get-Content logs\v4_status.txt` e `Get-Content logs\v4_resumo.txt`. A tabela-resumo (d) só ganha a linha da v4 quando a versão fechar.
- **Teste final:** não avaliado (`AVALIAR_TESTE_FINAL = False`; `scripts/v4.py` para com erro se a flag for True).
- **Dependências (Parte C):** `pip install transformers safetensors` no env → `transformers 5.22.2`, `safetensors 0.8.0`, `tokenizers 0.23.3` (+ pydantic, typer, rich, regex como dependências). O `pip install --dry-run` confirmou antes que **torch, torchaudio e huggingface_hub não seriam tocados** (continuam 2.14.1+cu130 / 2.11.0+cu130 / 2.4.0). Funciona com Python 3.13.14. Acrescentados ao `requirements.txt`. `accelerate` **não** foi instalado (só serviria para carregar os pesos direto na GPU).
- **Modelo:** `facebook/wav2vec2-xls-r-300m`, revisão `1a640f32ac3e39899438a2931f9924c02f080a54`, só `config.json`, `preprocessor_config.json`, `README.md` e `pytorch_model.bin` (1.269.737.156 bytes = 1,27 GB; o repositório não tem safetensors). Cache em `%USERPROFILE%\.cache\huggingface\hub`. Depois do download, tudo roda com `HF_HUB_OFFLINE=1` e `local_files_only=True`.
- **Decisões tomadas sem supervisão (com o motivo):**
  1. **fp16** (testado: sem NaN; valores até ~330, longe do limite do float16). Lote com NaN seria refeito em bf16 e registrado (`dtype` no meta do shard).
  2. **Lotes da GPU com formato fixo ("baldes" de 2, 4, 6 e 8 s).** Medido: com formatos de lote variados, o processo principal chegou a **13,2 GB de memória reservada** (pico total 19,2 GB, sistema 41,9 GB) e a vazão caiu para 19 clipes/s; com 1 formato fixo, 4,5 GB; com 25 formatos, 10,5 GB estáveis; com os 4 baldes, **7,5 GB e 44 clipes/s**. Causa provável: cada formato novo faz a GPU carregar mais variantes de kernels na memória da CPU (carregamento "preguiçoso" do CUDA, que já é o padrão; no modo ansioso seriam 12,6 GB logo de início). O enchimento é mascarado (`attention_mask`) e média/desvio usam só quadros válidos: **lote no balde × clipe isolado, diferença relativa máxima 0,42% em fp16 (0,12% em fp32)** em 12 clipes reais. Execução, não método.
  3. **cuDNN continua ligado.** Desligá-lo economizava ~1,2 GB, mas mudava os números (até 20% numa camada, com ruído aleatório na entrada); não vale trocar a implementação das convoluções por isso.
  4. **Treino: 20 shards = 40.000 clipes (20.000 reais + 20.000 fakes).** Calculado com o plano: até ~20 shards, o teto por fonte vale para todas as fontes grandes (reais: ASVspoof 24%, DFADD 24%, LibriSpeech 24%, ShiftySpeech 24%, CodecFake+ 3,5%); depois disso, ShiftySpeech, ASVspoof e CodecFake+ se esgotam e todo real novo viria só do DFADD (VCTK) e do LibriSpeech (com 69 shards, 85% dos reais), ou seja, fala lida e limpa, o viés da hipótese. Não é limitação de orçamento.
  5. **Validação:** 20.000 clipes (CodecFake+ e ShiftySpeech inteiros; DFADD e LibriSpeech na mesma fração por fonte e rótulo). **dev_shift inteiro** (10.614). **Ablação do corte** (opcional): os mesmos 10.000 clipes de treino (5 primeiros shards), 5.000 da validação e o dev_shift inteiro, **sem** corte de silêncio.
  6. **Cabeças:** MLP de 256 neurônios, dropout 0,3, lote 256, paciência 5 (os da CNN); grade de C 0,001/0,01/0,1/1; concatenação das 3 **e** das 4 melhores camadas (a validação escolhe).
  7. **Controle de sanidade acrescentado** à Parte F: a regressão da melhor camada com os rótulos do treino embaralhados (tem que dar AUC ~0,5).
  8. Os `.ps1` são gravados em UTF-8 **com BOM** (o PowerShell 5.1 lê sem BOM como ANSI e estraga os acentos). O `Get-Counter` com nomes em inglês falha neste Windows (pt-BR); o vigia descobre os nomes pelo próprio Windows (`\Memória\Limite de confirmação`, `\Memória\Bytes confirmados`) e, se falhar, usa `Win32_OperatingSystem`.
  9. O `rodar_v4.ps1` encerra **workers órfãos** do DataLoader (processo `spawn_main` cujo pai morreu): no teste de interrupção à força, um worker ficou vivo segurando 0,9 GB.
  10. Um primeiro teste do vigia (que apagava os resultados QUICK e matava o processo num só script) foi **bloqueado pelo classificador de segurança** do Claude Code. Refeito sem apagar nada (sufixo de teste `V4_SUFIXO_EXTRA`, que só vale no QUICK) e com a interrupção em passos separados.
- **Vazão medida** (`results/v4_vazao.json`; 500 clipes reais de treino, balanceados por fonte, 4 workers, corte de silêncio, baldes de 2 s): **44,1 clipes/s** (52,5 sem a partida dos workers; a GPU sozinha faria 55,9), pico de memória **11,45 GB** (principal 7,5 GB + 4 workers de ~0,9 GB), GPU 3,8 GB, 0 NaN.
- **Tamanhos e estimativa (é estimativa):** 10.614 (dev_shift) + 20.000 (validação) + 40.000 (treino) + 25.614 (ablação sem corte) = **96.228 clipes ≈ 36 min de extração a 44 clipes/s** (orçamento: 4 h cumulativas, `results/v4_estado.json`); cabeças, atalhos e resumo: ~30 a 60 min. Total **~1,2 a 1,7 h**, mais até 60 min de espera por memória no início (a folga estava em 12 GB com esta sessão aberta; a regra pede 14 GB).
- **Testes QUICK (2026-11-09; ~300 clipes: 100 de dev_shift, 99 de validação, 100 de treino, +150 sem corte; os números não valem como resultado e não devem ser interpretados):**
  - Shapes `(n, 25, 1024)` float16, média e desvio; **nenhum NaN/infinito**; 0 erros de leitura; 0 cortes desfeitos.
  - **Lote com padding × clipe isolado:** diferença relativa máxima por camada de 0,6% (fp16) e 0,05% (fp32) com padding simples; 0,42% / 0,12% no balde de 32 × 8 s.
  - **Corte de silêncio:** em 6 áudios da validação, silêncio inicial de 0 a 0,79 s e final de 0,01 a 0,75 s (ex.: `p311_391.wav` de 2,56 s → 1,02 s); mediana no dev_shift QUICK: 0,26 s no início e 0,20 s no fim.
  - **Retomada:** a extração foi interrompida à força depois do 1º shard de treino; a nova tentativa pulou os 5 shards prontos e continuou do `treino/00001`. Os shards refeitos são **idênticos bit a bit** aos de uma execução sem interrupção (média, desvio e meta). O orçamento **acumulou** entre as tentativas (2 tentativas).
  - **Vigia:** trava de instância única (um 2º vigia saiu na hora), espera por memória (folga 12,3 < 14 GB; com esperas encurtadas no teste), falha detectada (código 10), nova tentativa e conclusão; `logs\v4_status_quick_t1.txt` e `logs\v4_FIM_quick_t1.txt` gravados.
  - **Pipeline inteiro pelo `rodar_v4.ps1 -Quick`** com o código final (sufixo `_quick_t2`): as cinco etapas + conclusão, código 0; uma 2ª chamada pulou tudo. Controle com rótulos embaralhados: AUC 0,53 (val) e 0,55 (dev_shift).
  - **Notebook em QUICK** (cópia no scratchpad, `V4_QUICK=1`): rodou inteiro sem erro, 1,5 MB, sem consultar o teste.
  - Picos de memória (processo + workers): extração 11,5 a 13,2 GB; cabeças 1,1 a 2,1 GB.
- **Pendente:** a execução completa (extração, cabeças, atalhos, resumo), a escrita das conclusões do notebook com os números reais e o arquivamento em `notebooks/versions/v4_ssl.ipynb`; o teste final da v4 (etapa separada).

### v4 - teste final: pré-registro

- **Escrito em:** 2026-11-10 07:25 (−03:00), branch `v4-ssl`, commit `cbfb649` com as mudanças não commitadas da v4. **Antes de qualquer resultado de teste:** nenhum áudio de teste foi extraído nem avaliado pela v4, e nenhuma nota de teste da v3 existe (o teste da v3 nunca foi avaliado). A execução da v4 (validação + dev_shift) já terminou em 2026-11-10 00:11 (`logs/v4_resumo.txt`); tudo o que está abaixo foi decidido com esses números e o manifest, e **nada pode ser mudado depois de ver o teste**. Se algo mudar por erro de execução, a mudança vai num bloco datado separado, com o motivo.
- **Regra central:** o teste final é avaliado **uma vez**. Nenhum sistema, camada, C, limiar ou coeficiente é reescolhido depois. **Todos** os sistemas, bancos e limiares abaixo são reportados. Se um processo cair no meio, rodar de novo repete exatamente as mesmas contas (todas determinísticas), sem nenhuma escolha nova.
- **Sistemas** (todos avaliados nos **mesmos** clipes de cada banco):
  - **S0 · CNN da v3:** o checkpoint da v3 com **menor EER média por fonte na validação**: `v3_zero_s123` (11,25%; os outros: 12,3% `zero_s2026`, 12,6% `repeat_s2026`, 13,2% `repeat_s123`, 13,7% `zero_s42`, 14,1% `repeat_s42`). Registro, sem mudar a regra: é também o de **pior AUC no dev_shift** das seis execuções (0,146). Avaliação como na v3: áudio inteiro (sem corte de silêncio), áudio < 4 s completado com zeros (a política com que foi treinado), janelas de 4 s com passo de 2 s, **média dos logits** (`src/experimento.py: calcular_notas`).
  - **S1 · "só transfer learning":** XLS-R 300M congelado, **somente a última camada (saída 24)**, média no tempo, padronizada com a média e o desvio do treino da v4, regressão logística (scikit-learn, lbfgs) treinada nos mesmos 40.000 clipes de treino da v4. C escolhido na grade 0,001 / 0,01 / 0,1 / 1 pela **EER média por fonte na validação** (o mesmo critério da v4). Sem escolha de camada. (A etapa `camadas` da v4 já tinha feito essa mesma conta para a saída 24: C = 1, val EER média por fonte 8,57%, dev_shift AUC 0,984; a grade é refeita e tem de dar o mesmo C.)
  - **S2 · v4 final:** a cabeça **`concat_top3`** (saídas 5, 6 e 7 concatenadas, C = 0,1), já escolhida pela validação na v4. **Não é reescolhida.** Os coeficientes não tinham sido gravados; a regressão é **reajustada com os mesmos dados e o mesmo C** (lbfgs é determinístico) e conferida contra as notas gravadas da validação e do dev_shift (`results/v4_notas_concat_top3_*.parquet`); a diferença máxima é reportada.
  - **S3 · v4 + CNN:** regressão logística (lbfgs, C = 1) de **duas entradas**, o logit do S2 e o logit do S0, cada um padronizado com a média e o desvio **da validação**, treinada **só nos 19.999 clipes da validação da v4**. Os coeficientes são reportados. Atenção: o limiar (A) do S3 é o ponto de EER na própria validação em que ele foi treinado.
- **Clipes de validação e dev_shift:** os da v4 (19.999 da validação, subamostra estratificada; dev_shift inteiro, 10.614). O S0 é recalculado nesses mesmos clipes (as notas antigas da v3 servem só para conferir).
- **Bancos e amostra** (`src/experimento.py: amostra_teste`: `split == "test"` e `eval_subset == "test_final"`, até 5.000 clipes por classe e banco, seed 42, a mesma regra das versões anteriores; se a classe tem menos, entram todos). Total: **51.320 clipes**.

  | banco | tipo | real | fake | no banco (test_final) |
  |:--|:--|--:|--:|--:|
  | in_the_wild | principal (só test_final, locutores fora do dev_shift) | 5.000 | 5.000 | 11.330 / 9.835 |
  | llamapartialspoof | principal | 5.000 | 5.000 | 10.573 / 130.043 |
  | brspeech_df | principal (pt-BR) | 1.320 | 5.000 | 1.320 / 6.587 |
  | coraa | só real (pt-BR): vale o falso alarme = 1 − acurácia | 5.000 | 0 | 12.350 / 0 |
  | asvspoof2019_la (eval) | secundário (pode compartilhar locutores com o treino via VCTK) | 5.000 | 5.000 | 7.355 / 63.882 |
  | asvspoof2021_df | secundário (idem) | 5.000 | 5.000 | 5.535 / 147.420 |

  Um clipe que não puder ser lido na extração sai de **todos** os sistemas naquele banco (o número é reportado).
- **Pré-processamento do teste (S1 a S3):** o mesmo da v4 (mono 16 kHz, corte de silêncio de 35 dB, no máximo 8 s, recorte central, sem completar, fp16, 25 saídas, média e desvio no tempo), extraído em shards com retomada, num cache separado (`data/processed/v4_cache_teste/`).
- **Limiares** (ambos declarados agora, cada sistema com os seus; **nenhum** é ajustado no teste): **(A)** ponto de EER da validação; **(B)** ponto de EER do dev_shift.
- **Métricas por sistema, banco e limiar:** acurácia, acurácia balanceada (média do recall de real e de fake), acurácia trivial (sempre a classe majoritária), AUC, EER, recall de fake, falso alarme de real. No CORAA: só falso alarme (= 1 − acurácia); AUC, EER e acurácia balanceada não existem.
- **Bootstrap:** 1.000 reamostragens por banco, com reposição **dentro de cada classe** (mantém o nº de reais e de fakes), seed 42, intervalo de 2,5% a 97,5%, para AUC e acurácia balanceada (nos dois limiares) e para o falso alarme do CORAA. **O intervalo mede só a amostra de teste; não mede a variação do treino** (seed, amostra de treino), que a v3 mostrou ser grande.
- **Resumo:** média simples da **acurácia** dos três bancos principais com duas classes (in_the_wild, llamapartialspoof, brspeech_df), por sistema e limiar (a média da acurácia balanceada vai ao lado); **separado**, o falso alarme do CORAA. Os secundários não entram na média.
- **Leitura pré-definida:** "transfer learning sozinho" = **S1**; "contribuição das nossas escolhas" = **S2 − S1**; "contribuição da CNN" = **S3 − S2**. Cada diferença é medida em **AUC** e em **acurácia balanceada (limiar A)**, separadamente, com bootstrap **pareado** (as mesmas reamostragens para os dois sistemas). É **relevante** só se tiver **o mesmo sinal nos três bancos principais** E, **em cada um deles**, o intervalo de 95% da diferença pareada **não incluir zero** (a diferença é maior que a incerteza da amostra). Caso contrário: **"não demonstrada"**. A acurácia balanceada no limiar B e os secundários são só reportados.
- **Contaminação:** qualquer mudança futura baseada nestes resultados **contamina** estes bancos como teste. A partir daqui, In-the-Wild test_final, LlamaPartialSpoof, BRSpeech-DF, CORAA, ASVspoof 2019 eval e 2021 DF deixam de ser "nunca vistos" para qualquer versão que for ajustada olhando estes números (o In-the-Wild, aliás, já tinha sido consultado inteiro na v1).

### v4 · fechamento (2026-11-10)

- **Data:** execução de validação + dev_shift em 2026-11-09 23:38 a 2026-11-10 00:11 (vigia `scripts/vigia_v4.ps1`, 1 tentativa); teste final em 2026-11-10 07:31 a 07:52 (vigia `scripts/vigia_v4_teste.ps1`, 1 tentativa), avaliado **uma vez**, pelo pré-registro acima, **sem nenhuma mudança**. Branch `v4-ssl`, commit `cbfb649` com mudanças não commitadas.
- **Hipótese e mudanças:** as do pré-registro da v4 (XLS-R 300M congelado + cabeças leves; corte de silêncio; até 8 s sem completar; escolha de camada; atalhos; acurácias).
- **Validação e dev_shift** (`results/v4_camadas.csv`, `v4_cabecas.csv`, `v4_resumo.json`, `v4_atalhos.json`, `v4_ablacao_corte.json`; treino de 40.000 clipes, validação de 19.999, dev_shift inteiro, 10.614):
  - **Camadas:** as do meio são as melhores (5 a 9: EER média por fonte de 6,0% a 6,6%; a melhor, a 7, com 6,0%; a 0, 22%; a 24, 8,6%). No dev_shift, todas a partir da 5 passam de 0,97 de AUC.
  - **Cabeça final pela validação:** `concat_top3` (camadas 5, 6, 7; C = 0,1; EER média por fonte 5,92%; candidatas de 5,92% a 6,73%). dev_shift: **AUC 0,983**, EER 5,9%, acurácia balanceada 0,897 (limiar da validação), acurácia 0,843 (trivial 0,813). Pelo critério (c): **melhora clara** (régua da v3: AUC 0,146 a 0,378; acurácia balanceada 0,365 a 0,503).
  - **Somas ponderadas** (3 seeds): maiores pesos nas camadas 5, 6 e 7; AUC do dev_shift de 0,982 a 0,986 (o acaso entre seeds é ~0,005, contra 0,15 a 0,23 na CNN da v3).
  - **Atalhos (f):** só com estatísticas baratas, AUC 0,668 na validação (**indício de atalho nos dados**, acima de 0,6), mas 0,523 no dev_shift. A camada 7 reconhece o banco de origem dos reais com acurácia balanceada 0,80 (estatísticas: 0,57; acaso 0,25). Controle com rótulos embaralhados: AUC 0,474 (val) e 0,486 (dev_shift).
  - **Ablação do corte de silêncio:** sem efeito (EER média por fonte 7,6% × 7,9%; AUC do dev_shift 0,992 × 0,990).
- **Teste final** (`results/v4_teste_final.csv`, `v4_teste_diferencas.csv`, `v4_teste_final_resumo.json`, `v4_teste_final.png`; 51.320 clipes, 0 erros de leitura; cada célula: AUC · EER · acurácia · acurácia balanceada no limiar A / no limiar B):

  | banco | S0 CNN v3 | S1 só camada 24 | S2 v4 final | S3 v4 + CNN | trivial |
  |:--|:--|:--|:--|:--|--:|
  | in_the_wild | 0,114 · 82,2% · 0,336/0,181 · 0,336/0,181 | 0,986 · 5,4% · 0,829/0,946 · 0,829/0,946 | 0,979 · 6,8% · 0,836/0,931 · 0,835/0,931 | 0,888 · 18,8% · 0,641/0,810 · 0,641/0,810 | 0,500 |
  | llamapartialspoof | 0,556 · 46,3% · 0,545/0,523 · 0,545/0,523 | 0,936 · 14,0% · 0,851/0,734 · 0,851/0,734 | 0,925 · 14,7% · 0,855/0,738 · 0,855/0,738 | 0,900 · 17,8% · 0,826/0,726 · 0,826/0,726 | 0,500 |
  | brspeech_df (pt-BR) | 0,735 · 29,3% · 0,671/0,447 · 0,704/0,609 | 0,775 · 30,7% · 0,849/0,762 · 0,671/0,716 | 0,842 · 24,9% · 0,863/0,839 · 0,679/0,771 | 0,840 · 25,3% · 0,867/0,766 · 0,718/0,754 | 0,791 |
  | coraa (só real): falso alarme | 65,3% / 41,3% | 35,6% / 11,6% | 16,7% / 2,5% | 31,8% / 8,7% | — |
  | asvspoof2019_la (secundário) | 0,967 · 8,0% · 0,900/0,795 · 0,901/0,795 | 0,997 · 2,3% · 0,974/0,902 · 0,974/0,902 | 0,999 · 1,4% · 0,981/0,901 · 0,981/0,901 | 1,000 · 0,3% · 0,988/0,889 · 0,988/0,889 | 0,500 |
  | asvspoof2021_df (secundário) | 0,688 · 36,0% · 0,665/0,608 · 0,665/0,608 | 0,996 · 2,8% · 0,956/0,964 · 0,956/0,964 | 0,997 · 2,6% · 0,942/0,964 · 0,942/0,964 | 0,909 · 17,9% · 0,829/0,830 · 0,829/0,830 | 0,500 |
  | **média dos 3 principais (acurácia)** | 0,517 / 0,384 | 0,843 / 0,814 | **0,851 / 0,836** | 0,778 / 0,767 | 0,597 |

  Intervalos bootstrap (95%, só a amostra de teste) estreitos: de ±0,001 a ±0,016 na AUC e ±0,005 a ±0,015 na acurácia balanceada (os maiores no BRSpeech-DF, com 1.320 reais). Limiares (logit; A / B): S0 −2,15 / 2,84; S1 −0,01 / 6,30; S2 −0,04 / 8,22; S3 0,74 / 5,40. Coeficientes do S3 (entradas padronizadas pela validação): intercepto 4,75; S2 5,36; S0 4,19.
- **Leitura pré-definida:**
  - **Transfer learning sozinho (S1):** In-the-Wild 0,986, LlamaPartialSpoof 0,936, BRSpeech-DF 0,775 de AUC (S0: 0,114 / 0,556 / 0,735).
  - **Nossas escolhas (S2 − S1): não demonstrada.** AUC −0,008 / −0,011 / +0,066 (sinais trocados); acurácia balanceada (A) +0,006 / +0,004 / +0,008, com intervalos que incluem zero. Só reportado: CORAA 35,6% → 16,7% de falso alarme (A).
  - **CNN (S3 − S2): não demonstrada pela regra** (BRSpeech-DF: AUC −0,002, intervalo inclui zero; acurácia balanceada A troca de sinal: −0,195 / −0,029 / +0,039). Em nenhum banco principal a AUC subiu; no In-the-Wild caiu 0,091 e no ASVspoof 2021 DF, 0,088.
- **Conferências:** o reajuste do S1 e do S2 reproduziu as notas gravadas da v4 com diferença 0; o S1 escolheu o mesmo C (1) da etapa de camadas. O S0 recalculado difere das notas gravadas da v3 em até 0,017 de logit na validação e 0,0005 no dev_shift (lotes diferentes na GPU); AUC do dev_shift igual (0,146).
- **Conta descritiva depois do teste (não mudou nada):** no LlamaPartialSpoof, os fakes **completos** escapam mais que os parciais (recall do S2 no limiar A: 0,70 × 0,77; AUC 0,873 × 0,944).
- **Conclusão em uma frase:** o XLS-R congelado com uma regressão logística resolveu a inversão das vozes de internet (AUC do In-the-Wild de 0,11 para 0,98, no dev_shift e no teste), as nossas escolhas por cima não mostraram ganho demonstrável, a CNN só atrapalhou, e o ponto fraco que sobra é o pt-BR e a escolha do limiar.
- **Limitações:** amostra de até 5.000 por classe e banco (BRSpeech-DF: 1.320 reais); bootstrap mede só a amostra de teste, não o treino; dev_shift pequeno (16 locutores), e é ele que define o limiar B; uma única amostra de treino e uma única regressão por sistema; ASVspoof pode compartilhar locutores com o treino (VCTK); os seis bancos de teste agora estão **consumidos**.
- **Arquivos:** notebook `notebooks/versions/v4_ssl.ipynb` (e o `notebook.ipynb` da raiz, igual, 1,7 MB); resultados no git: `results/v4_teste_final.csv`, `v4_teste_diferencas.csv`, `v4_teste_final_resumo.json`, `v4_teste_final.png`, `v4_teste_escolha_s0.json`, `v4_teste_final_concluido.json` (e os da v4 já listados no `.gitignore`); fora do git: `results/v4_teste_notas.parquet` (notas por clipe dos 4 sistemas), `results/v4_teste_notas_s0_*.parquet`, os marcadores de etapa, o cache `data/processed/v4_cache_teste/` (~5 GB) e os logs `logs/v4_teste_*`.
- **Tempos** (RTX 4060 8 GB, 4 workers):

  | etapa | tempo | pico de memória (processo + workers) |
  |:--|--:|--:|
  | v4: extração (96.227 clipes) | 24,9 min | 14,2 GB |
  | v4: camadas / somas / atalhos / resumo | 2,9 / 4,2 / 0,2 / 0,1 min | 2,8 / 5,9 / 1,6 / 1,1 GB |
  | teste: extração (51.320 clipes; ~63 clipes/s com a carga do modelo; shards de 52 a 86 clipes/s) | 13,5 min | 15,0 GB (sistema 37,4 GB) |
  | teste: CNN (S0) em validação, dev_shift e teste | 3,4 min | 12,1 GB |
  | teste: avaliação (S1, S2, S3, métricas, 1.000 bootstraps por banco) | 3,4 min | 2,2 GB |

  Teste final inteiro: 20,9 min (07:31:44 a 07:52:37). Execução do notebook (só leitura): ~1 min.

### v5 - pré-registro (WavLM Large × XLS-R, o mesmo pipeline da v4)

- **Escrito em:** 2026-11-10, branch `v5-wavlm`, a partir do commit `3887f87` (v4 fechada). **Antes de qualquer resultado da v5:** só o modelo foi baixado (`microsoft/wavlm-large`, revisão `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, `pytorch_model.bin` de 1.261.990.257 bytes) e o seu `config.json` conferido. Nada abaixo pode ser mudado depois de ver resultados; se algo mudar, vai num bloco datado separado, com o motivo.
- **Motivo:** o grupo quer comparar o modelo da Meta (XLS-R, v4) com um segundo modelo pronto e seguir melhorando o melhor dos dois. O Gemini Embedding 2 (Google) foi considerado e descartado por ora: só existe por API paga, o plano gratuito usa os dados enviados para melhorar os produtos do Google (problema com as licenças dos bancos) e ele é feito para captar significado, não detalhe acústico.
- **Hipótese:** o **WavLM Large** (Microsoft; ~94 mil horas de fala em inglês; pré-treinado também a "limpar" fala com ruído e sobreposição) dá representações que separam real de IA **pelo menos tão bem** quanto o XLS-R 300M, e melhor nas vozes de internet (dev_shift). ⚠️ Fora da trilha: o modelo auto-supervisionado, como na v4.
- **Uma única mudança:** o modelo congelado. Os dois têm 25 saídas de 1024 números a cada 20 ms, e a mesma normalização da entrada (`do_normalize: true`, `attention_mask`). Tudo o mais igual à v4: mesmos clipes (plano com seed 42: 40.000 de treino, 19.999 da validação, dev_shift inteiro com 10.614), corte de silêncio de 35 dB, até 8 s sem completar, fp16, média e desvio no tempo, regressão logística por camada (C pela validação), concatenação das 3 e 4 melhores camadas, somas ponderadas (linear e MLP) com 3 seeds, atalhos e resumo.
- **O XLS-R é refeito nesta máquina** (prefixo `v5_xlsr`), no mesmo código e no mesmo hardware que o WavLM (`v5_wavlm`). Motivos: as notas por clipe da v4 não estão no git (sem elas não há comparação pareada) e a v4 rodou noutra placa (RTX 4060 8 GB; aqui, RTX 4050 6 GB). **Conferência de reprodução:** a EER média por fonte da validação de cada camada do `v5_xlsr` tem de ficar a no máximo 0,5 ponto da do `results/v4_camadas.csv`, e a cabeça final pela validação tem de ser a mesma (`concat_top3`, camadas 5, 6, 7). Se não bater, isso é registrado e investigado **antes** de comparar os modelos.
- **Fora desta versão:** a ablação do corte de silêncio (sem efeito na v4; `SEMCORTE = None`, o que não muda o sorteio dos outros clipes).
- **Critérios (fixados agora):**
  - **(a) Dentro de cada modelo:** camada, C e cabeça final escolhidos pela **EER média por fonte na validação**, exatamente como na v4. O dev_shift não escolhe nada dentro do modelo.
  - **(b) Entre os modelos:** compara-se a **cabeça final** de cada um (e, só reportada, a "só última camada", o S1 da v4). O **WavLM é melhor** se, ao mesmo tempo: (1) tiver EER média por fonte menor na validação e (2) tiver AUC maior no dev_shift, com o intervalo de 95% da diferença pareada **não incluindo zero**. O **WavLM é pior** na situação inversa. Fora disso: **empate**, e o grupo segue com o **XLS-R** (já testado no teste final da v4).
  - **(c) Bootstrap pareado do dev_shift:** 1.000 reamostragens **por locutor** (sorteia os 16 locutores com reposição, com todos os clipes de cada um; seed 42), as mesmas para os dois modelos. É por locutor porque clipes do mesmo locutor não são independentes, e reamostrar clipes daria um intervalo estreito demais.
  - **(d) Também reportados (não decidem):** EER e acurácia balanceada do dev_shift (limiar da validação e 0,5), acurácia trivial, AUC por faixa de duração, EER por fonte e por gerador na validação, curva de camadas dos dois modelos e os atalhos (b) (quão bem a melhor camada reconhece o banco de origem dos reais).
  - **(e) Análise do "porquê"** (pedida pelo grupo para a apresentação): em quais camadas, fontes, geradores e durações cada modelo ganha. É descritiva: não muda a decisão de (b).
- **Teste final:** **não** é avaliado nesta execução (`AVALIAR_TESTE_FINAL = False`). Depois da decisão escrita aqui, é avaliado **uma vez**, numa etapa separada com pré-registro próprio, com **os dois modelos**, sem reescolher nada. Lembrete: os seis bancos de teste já foram consultados na v4, então o teste da v5 serve para reportar, não para escolher.
- **Execução:** `scripts\rodar_v4.ps1 -Modelo xlsr` e `-Modelo wavlm`, fora desta sessão, um modelo de cada vez. Se a placa de 6 GB não comportar o lote da v4, o tamanho do lote da GPU pode ser reduzido: é mudança de execução (registrada em h), não de método.

### v5 · resultado na validação + dev_shift (decisão, 2026-11-10)

- **Execução:** `rodar_v4.ps1 -Modelo xlsr` (2026-11-10 11:57 a 13:50; extração 1,70 h, 10 a 23 clipes/s, pico 14,2 GB) e `-Modelo wavlm` (13:50 a 15:35; ~12 a 15 clipes/s), código 0 nos dois, 1 tentativa, 0 erros de leitura, 0 NaN. Comparação: `python scripts/comparar_v5.py` (`src/comparacao.py`) → `results/v5_comparacao.json` e `results/v5_comparacao_{camadas,fontes,geradores,duracao,locutores}.csv`. **Teste final não lido.**
- **Conferência de reprodução (XLS-R refeito × v4): passou.** Maior diferença de EER média por fonte por camada: 0,37 ponto (camada 11; a maioria abaixo de 0,1). Cabeça final igual (`concat_top3`, camadas 5, 6, 7; 5,92% × 5,92%; dev_shift AUC 0,983 × 0,983). Em 4 camadas (4, 12, 21, 24) a grade escolheu outro C (valores de C quase empatados na v4; a pequena diferença de fp16 entre as placas inverteu a ordem).
- **Cabeça final de cada modelo (pela validação):** XLS-R `concat_top3` (camadas 5, 6, 7; C = 0,1); WavLM `concat_top3` (camadas **4, 6, 8**; C = 0,1). No WavLM, a melhor camada sozinha é a 4.

  | sistema | val EER média por fonte | dev AUC | dev EER | dev acc. bal. (limiar val / 0,5) | dev acurácia (trivial 0,813) |
  |:--|--:|--:|--:|:--|--:|
  | XLS-R, cabeça final | **5,92%** | **0,983** | **6,0%** | 0,895 / 0,895 | 0,841 |
  | WavLM, cabeça final | 6,77% | 0,970 | 7,5% | 0,883 / 0,892 | 0,823 |
  | XLS-R, só camada 24 (S1) | 8,62% | 0,984 | 6,3% | 0,869 / 0,870 | 0,795 |
  | WavLM, só camada 24 | 15,10% | 0,956 | 10,0% | 0,882 / 0,891 | 0,826 |

- **Bootstrap pareado por locutor no dev_shift** (16 locutores, 1.000 reamostragens, seed 42; WavLM − XLS-R): cabeça final, AUC **−0,013** (IC95 −0,061 a **+0,001**), EER +1,5 ponto (IC95 −1,0 a +9,4). Só camada 24: AUC −0,028 (IC95 −0,100 a +0,011).
- **Decisão pela regra (pré-registro b): EMPATE → segue o XLS-R.** O WavLM ficou **pior na validação** (6,77% × 5,92%) e **pior no dev_shift** (AUC 0,970 × 0,983), mas o intervalo da diferença no dev_shift encosta no zero (+0,001), então a piora não está demonstrada pela regra. Em nenhum critério principal o WavLM ficou à frente.
- **Por que (descritivo, pré-registro e; não muda a decisão):**
  - **Camadas:** no WavLM, as melhores camadas são **mais baixas** (3 e 4; EER 9,3% e 7,9%), e a qualidade cai rápido depois da 8 (camada 24: 15,1%, contra 8,6% no XLS-R). No XLS-R, as camadas 5 a 9 ficam todas perto de 6%. Nas camadas 0 a 4, o WavLM vai melhor no dev_shift que o XLS-R (por exemplo, camada 3: AUC 0,906 × 0,793). Leitura (hipótese, não testada): o pré-treino do WavLM a "limpar" ruído e outra voz faz as camadas de cima descartarem justamente detalhes acústicos que denunciam a voz sintética; os artefatos ficam só nas camadas de baixo.
  - **Fontes da validação:** empate no DFADD (EER 0,0% × 0,1%) e no ShiftySpeech (2,9% × 2,9%); o WavLM é pior no **CodecFake+** (fakes de codec neural; EER 17,4% × 14,8%). Por gerador: resultados mistos e com poucos clipes no CodecFake+ (5 a 32 fakes por gerador).
  - **Duração (dev_shift):** o XLS-R é melhor nas três faixas; a maior diferença é nos áudios curtos (< 2 s: AUC 0,931 × 0,915).
  - **Locutores (dev_shift):** boa parte da diferença vem de **Winston Churchill** (gravações antigas; AUC 0,980 no XLS-R × 0,707 no WavLM) e de Donald Trump (0,973 × 0,910). O WavLM é um pouco melhor em Barack Obama, Lyndon Johnson e Mr. Rogers. Gravação antiga, de novo, é o ponto fraco (como na CNN da v1 a v3).
  - **Atalhos (b):** a melhor camada do WavLM reconhece o banco de origem dos reais um pouco melhor (acurácia balanceada 0,82 × 0,80; estatísticas 0,57; acaso 0,25): ele guarda tanta informação de canal e gravação quanto o XLS-R.
  - **Limiar:** como na v4, o limiar da validação (estúdio) dá muito falso alarme em alguns locutores do dev_shift (por exemplo, Eisenhower ~98% nos dois modelos). O problema do limiar por domínio continua (ideia da seção f).
- **Hipótese da v5 (WavLM melhor em voz de internet): não confirmada.**
- *(O teste final, o notebook e o arquivamento foram feitos depois: ver "v5 · fechamento", mais abaixo.)*

### v5 - teste final: pré-registro

- **Escrito em:** 2026-11-10 (−03:00), branch `v5-wavlm`, **antes de qualquer resultado de teste da v5**: nenhum áudio de teste foi extraído pela v5. A decisão da v5 (empate pela regra → segue o XLS-R) já está escrita acima, com a validação e o dev_shift, e **o teste não a muda**. Nada abaixo pode ser mudado depois de ver o teste; erro de execução vai num bloco datado separado.
- **Regra central:** avaliado **uma vez**, só para reportar. Nenhuma camada, C, cabeça ou limiar é reescolhido. **Todos** os sistemas, bancos e limiares abaixo são reportados.
- **Sistemas** (todos nos **mesmos** clipes; cada um reajustado de forma determinística, lbfgs, com os dados de treino da v5 e as escolhas já feitas na validação; as notas da validação e do dev_shift são conferidas contra as gravadas e a diferença máxima é reportada):
  - **X2 · XLS-R, cabeça final:** `concat_top3` (camadas 5, 6, 7; C = 0,1) do `v5_xlsr`.
  - **W2 · WavLM, cabeça final:** `concat_top3` (camadas 4, 6, 8; C = 0,1) do `v5_wavlm`.
  - **X1 · XLS-R, só a camada 24** (C = 0,1, o escolhido pela validação no `v5_xlsr`) e **W1 · WavLM, só a camada 24** (C = 1, idem no `v5_wavlm`): só reportados.
- **Amostra:** a mesma regra e a mesma seed da v4 (`src/teste_v4.py: amostra_teste_idx`; até 5.000 por classe e banco, seed 42), ou seja, os **mesmos 51.320 clipes** do teste da v4 (In-the-Wild só `test_final`, LlamaPartialSpoof, BRSpeech-DF, CORAA só real; ASVspoof 2019 eval e 2021 DF como secundários). Conferência: os tamanhos por banco têm de bater com os de `results/v4_teste_final.csv`. O X2 é o mesmo sistema do S2 da v4 refeito noutra placa: a diferença de AUC por banco é reportada (conferência de reprodução, não decide nada).
- **Pré-processamento:** o mesmo da v5 (corte de silêncio de 35 dB, até 8 s, fp16), um cache de teste por modelo (`data/processed/v5_xlsr_cache_teste/`, `v5_wavlm_cache_teste/`, fora do git).
- **Limiares** (cada sistema com os seus): **(A)** ponto de EER da validação; **(B)** ponto de EER do dev_shift.
- **Métricas por sistema, banco e limiar:** as da v4 (acurácia, acurácia balanceada, acurácia trivial, AUC, EER, recall de fake, falso alarme de real; no CORAA, só o falso alarme). **Bootstrap:** 1.000 reamostragens por banco, com reposição dentro de cada classe, seed 42, as mesmas para todos os sistemas (pareado).
- **Leitura pré-definida (só reportada, não muda a decisão):** "WavLM − XLS-R" = **W2 − X2**, em AUC e em acurácia balanceada (limiar A). É **relevante** só se tiver o mesmo sinal nos três bancos principais (In-the-Wild, LlamaPartialSpoof, BRSpeech-DF) e, em cada um, o intervalo de 95% da diferença pareada não incluir zero; senão, **"não demonstrada"**. W1 − X1 e o limiar B só reportados.
- **Resumo:** média simples da acurácia (e da acurácia balanceada) dos três principais, por sistema e limiar; o falso alarme do CORAA à parte.
- **Contaminação:** estes bancos já tinham sido consultados na v4; os números do teste da v5 servem para reportar a comparação, não para escolher nada nas versões seguintes.

### v5 · fechamento (2026-11-10)

- **Data:** validação + dev_shift em 2026-11-10 11:57 a 15:35 (`rodar_v4.ps1 -Modelo xlsr` e `-Modelo wavlm`); comparação e decisão às ~15:45 (`scripts/comparar_v5.py`); teste final em 2026-11-10 16:06 a 17:52 (`scripts/v5_teste.py`), avaliado **uma vez**, pelo pré-registro acima, **sem nenhuma mudança**. Extração do teste: XLS-R 40,1 min, WavLM 62,2 min (pico 15,2 GB); avaliação 3,4 min; 0 erros de leitura nos 51.320 clipes.
- **Hipótese e mudança:** as do pré-registro da v5 (só o modelo congelado: WavLM Large no lugar do XLS-R 300M).
- **Decisão (validação + dev_shift; ver "v5 · resultado"):** empate pela regra → **segue o XLS-R**. O WavLM ficou pior na validação (EER média por fonte 6,77% × 5,92%) e no dev_shift (AUC 0,970 × 0,983, IC95 por locutor −0,061 a +0,001).
- **Teste final** (`results/v5_teste_final.csv`, `v5_teste_diferencas.csv`, `v5_teste_final_resumo.json`, `v5_teste_final.png`; cada célula: AUC · EER · acurácia balanceada no limiar A / no limiar B):

  | banco | X1 XLS-R, camada 24 | X2 XLS-R, final | W1 WavLM, camada 24 | W2 WavLM, final | trivial |
  |:--|:--|:--|:--|:--|--:|
  | in_the_wild | 0,988 · 5,2% · 0,829/0,949 | 0,978 · 6,8% · 0,834/0,930 | 0,946 · 10,3% · 0,826/0,895 | 0,950 · 11,4% · 0,839/0,848 | 0,500 |
  | llamapartialspoof | 0,934 · 14,0% · 0,855/0,736 | 0,924 · 14,8% · 0,854/0,735 | 0,899 · 17,9% · 0,830/0,792 | 0,914 · 14,5% · 0,856/0,708 | 0,500 |
  | brspeech_df (pt-BR) | 0,776 · 30,8% · 0,678/0,712 | 0,842 · 25,0% · 0,680/0,769 | 0,557 · 45,3% · 0,524/0,557 | 0,768 · 32,3% · 0,643/0,683 | 0,791 |
  | coraa (só real): falso alarme A / B | 36,1% / 11,5% | 16,9% / 2,5% | 87,3% / 73,6% | 34,5% / 8,5% | — |
  | asvspoof2019_la (secundário) | 0,997 · 2,0% · 0,977/0,906 | 0,999 · 1,4% · 0,981/0,898 | 0,986 · 5,6% · 0,944/0,911 | 0,996 · 2,4% · 0,970/0,817 | 0,500 |
  | asvspoof2021_df (secundário) | 0,996 · 2,6% · 0,952/0,968 | 0,997 · 2,6% · 0,942/0,964 | 0,986 · 5,7% · 0,895/0,945 | 0,990 · 4,8% · 0,904/0,937 | 0,500 |
  | **média dos 3 principais (acurácia, A / B)** | 0,846 / 0,813 | **0,850 / 0,833** | 0,813 / 0,806 | 0,846 / 0,758 | 0,597 |

- **Leitura pré-definida (W2 − X2):** em **AUC, "relevante: WavLM pior"**: −0,029 no In-the-Wild, −0,010 no LlamaPartialSpoof e −0,075 no BRSpeech-DF, com os intervalos de 95% sem o zero nos três (e também pior nos dois secundários). Em **acurácia balanceada (limiar A), "não demonstrada"**: +0,005 e +0,002 (intervalos com o zero) no In-the-Wild e no LlamaPartialSpoof, −0,037 no BRSpeech-DF. W1 − X1: AUC pior nos cinco bancos (BRSpeech-DF −0,219).
- **Conferência de reprodução no teste:** X2 × S2 da v4, diferença de AUC de −0,0008 a +0,0008 por banco. O reajuste dos quatro sistemas reproduziu as notas gravadas da validação e do dev_shift com diferença 0.
- **Por que o XLS-R ganhou (descritivo):**
  1. **Língua do pré-treino.** O XLS-R viu 128 línguas (inclusive português); o WavLM Large, só inglês. A maior diferença do teste é no pt-BR: BRSpeech-DF AUC 0,842 × 0,768 e falso alarme no CORAA 16,9% × 34,5% (limiar A). Só com a última camada, o WavLM fica perto do acaso no BRSpeech-DF (0,557) e acusa 87% dos reais do CORAA.
  2. **Onde ficam as pistas.** No WavLM, as melhores camadas são as de baixo (3 e 4), e as de cima pioram muito (camada 24: EER média por fonte 15,1% × 8,6%). Hipótese (não testada): o pré-treino a "limpar" ruído e fala sobreposta faz as camadas de cima descartarem justamente os detalhes acústicos que denunciam a voz sintética.
  3. **Gravações antigas e áudios curtos**, os pontos fracos do projeto, também pioram com o WavLM (dev_shift: Churchill AUC 0,707 × 0,980; < 2 s 0,915 × 0,931).
  4. **No inglês, a diferença é pequena** (acurácia balanceada no limiar A igual dentro do intervalo no In-the-Wild e no LlamaPartialSpoof); o que separa os dois é a ordenação (AUC) e, sobretudo, o pt-BR.
- **Conclusão em uma frase:** o WavLM Large não superou o XLS-R: empate pela regra no dev_shift e pior em AUC em todos os bancos do teste (de forma demonstrada), principalmente em português, então o projeto segue com o XLS-R `concat_top3`.
- **Limitações:** uma única amostra de treino e uma regressão por sistema; dev_shift com 16 locutores (por isso a decisão deu empate); bootstrap do teste mede só a amostra de teste; os seis bancos de teste estavam consumidos desde a v4 (o teste da v5 só reporta); a explicação das camadas é hipótese; só dois modelos comparados (o Gemini Embedding 2, do Google, ficou de fora por ser API paga).
- **Arquivos:** notebook `notebooks/versions/v5_wavlm.ipynb` (e o `notebook.ipynb` da raiz, igual); no git: `results/v5_{xlsr,wavlm}_{camadas,cabecas,somas,resumo,atalhos,concluido}.*`, `results/v5_comparacao*`, `results/v5_teste_{final,diferencas}.csv`, `v5_teste_final_resumo.json`, `v5_teste_final.png`, `v5_teste_final_concluido.json`; fora do git: notas por clipe (`results/v5_*notas*.parquet`, `v5_teste_notas.parquet`), os caches `data/processed/v5_{xlsr,wavlm}_cache{,_teste}/` e os logs.

### v6 - pré-registro (funil: front-end, fusão, cabeça, maximização)

- **Escrito em:** 2026-11-10 22:15 (−03:00), branch `v6-fusao`, a partir do commit `9d62c03` (v5 fechada), **antes de qualquer resultado da v6** (nem os testes QUICK rodaram). Só foi feito: o download dos pesos `microsoft/wavlm-large` (revisão `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, `pytorch_model.bin` de 1.261.990.257 bytes, o mesmo do colega) e a conferência do estado (manifest `f2309eb356b4…`; `data/processed/v4_cache` com 40.000 / 19.999 / 10.614 clipes de treino / validação / dev_shift, iguais clipe a clipe ao `plano.parquet`). Nada abaixo muda depois de ver resultados; se algo mudar por erro de execução, vai num bloco datado separado, com o motivo. As justificativas de cada candidato ("por que X e não Y") estão em `docs/DECISOES_V6.md`, escritas também antes dos resultados.
- **O que já sabemos (a rodada NÃO é cega):** os números de validação e dev_shift por camada dos dois modelos (v4 e v5) e os do teste (v4 e v5). Pela EER da validação, o CodecFake+ é a fonte mais difícil nos dois modelos (14,8% e 17,4% na cabeça final); é provável que ele seja a "pior fonte" abaixo. **Os seis bancos de teste já foram consultados duas vezes (v4 e v5):** nada desta rodada é escolhido olhando para eles.
- **Pergunta da v6:** com o mesmo pipeline e os mesmos clipes, (1) qual front-end usar: só Meta (XLS-R 300M), só Microsoft (WavLM Large) ou a fusão dos dois; (2) qual cabeça por cima: regressão logística (RL), MLP ou CNN; (3) quanto se ganha "apertando" a cabeça vencedora. ⚠️ Fora da trilha: os modelos auto-supervisionados (como na v4/v5), o bootstrap, o stacking e a CNN que anda ao longo das camadas.

**Conjuntos e métricas (valem para todas as etapas):**
- **Clipes:** os do plano do `v4_cache` (seed 42): treino 40.000 (20.000 reais + 20.000 fakes), validação 19.999, dev_shift 10.614; o WavLM é extraído **nos mesmos clipes** (o plano é copiado do `v4_cache`, sem os conjuntos `*_semcorte`) e a identidade (mesmo `idx_manifest` e mesmo caminho, na mesma ordem) é conferida e registrada. O XLS-R é lido do `v4_cache` (não é reextraído).
- **"Fontes de decisão" (G):** as três fontes da validação com as duas classes (`dfadd`, `shiftyspeech`, `codecfake_plus`) e o `dev_shift`. O LibriSpeech da validação só tem reais: fica fora das AUCs, mas entra nas regras de limiar (E5).
- **Métrica primária:** **AUC da pior fonte** = menor AUC entre as quatro fontes de G.
- **Secundárias:** média do AUC nas quatro fontes; acurácia balanceada de cada fonte e a média (limiar A); EER de cada fonte e a média nas quatro; e, para continuidade com a v4/v5, a EER média por fonte só da validação (três fontes).
- **Limiares:** (A) ponto de EER da validação inteira; (B) ponto de EER do dev_shift; (E5) o escolhido na etapa 5 entre A, C e D (definidos lá). Nenhum limiar é ajustado no teste.
- **Bootstrap pareado** (2.000 reamostragens, seed 42, os mesmos sorteios para todos os sistemas): estratificado por fonte de G; dentro de cada fonte, sorteia **locutores** com reposição (`speaker_id`: DFADD 22, ShiftySpeech 12, dev_shift 16) e, onde não há locutor (CodecFake+), **clipes**. Em cada reamostragem: AUC de cada fonte, a pior delas, para cada sistema; diferença = desafiante − atual. Reamostragem sem as duas classes numa fonte é descartada (o número é reportado).
- **Diferença relevante:** IC95 da diferença de AUC da pior fonte **exclui o zero** E a diferença (na amostra inteira) é **≥ 0,005** a favor do desafiante. Senão, **empate**, e fica a opção mais simples.
- **Regra do desafiante (sequencial):** começa na opção mais simples; cada opção mais complexa, na ordem, só toma o lugar se tiver diferença relevante sobre a que está no lugar. Ordens: E3 só Meta → só Microsoft → fusão; dentro da fusão (E2), média das notas → stacking → vetores colados; E4 RL → MLP → CNN.
- **Igualdade:** toda comparação usa os mesmos clipes, a mesma cabeça (quando é o front-end que se compara), os mesmos splits e as mesmas seeds.

**E1 · Meta × Microsoft (mesmo pipeline, mesma máquina, notas por clipe):**
- Vazão do WavLM medida com 500 clipes antes; orçamento de 3 h (cumulativo) para a extração de treino, validação e dev_shift; sem a ablação sem corte.
- Nos dois modelos: RL por camada (saídas 0 a 24, média no tempo, padronizada com média e desvio do treino; lbfgs, `max_iter` 1.000, `tol` 1e-4), C em {1e-3, 1e-2, 1e-1, 1}, **C de cada camada escolhido pela AUC da pior fonte**. As camadas são ordenadas pela AUC da pior fonte; `concat_top3` e `concat_top4` = as 3 e as 4 melhores, com C escolhido do mesmo jeito. **Melhor cabeça de cada modelo** = a de maior AUC da pior fonte entre as 25 camadas, `concat_top3` e `concat_top4` (desempate: maior AUC média).
- **Reprodução do XLS-R:** a mesma grade, escolhendo C pela regra da v4 (EER média por fonte da validação), tem de dar EER média por fonte por camada a no máximo **0,5 ponto** da `results/v4_camadas.csv`. Se não der, isso é registrado e investigado antes de comparar; o XLS-R comparado é o recalculado aqui.
- Entregas: AUC por fonte e pior fonte por camada e por cabeça, nos dois modelos (tabela e gráfico); bootstrap pareado da melhor cabeça do WavLM − melhor cabeça do XLS-R; a referência v4 (XLS-R, camadas 5, 6, 7, C = 0,1) calculada nas mesmas métricas.

**E2 · Fusão (sobre a melhor cabeça de cada modelo):**
- Antes: correlação (Pearson e Spearman) dos logits dos dois modelos, por classe, na validação e no dev_shift; tabela 2×2 de erros (os dois acertam, só a Meta acerta, só a Microsoft acerta, os dois erram), cada modelo no seu limiar A, na validação e no dev_shift.
- **(a) Média das notas:** logit de cada modelo padronizado com a média e o desvio **da validação**; média simples dos dois.
- **(b) Vetores colados:** as camadas das melhores cabeças dos dois modelos, lado a lado; cada dimensão padronizada com o treino e cada bloco (modelo) dividido pela raiz do seu nº de dimensões (cada modelo pesa o mesmo na entrada); RL com C em {1e-3, 1e-2, 1e-1, 1}, escolhido pela AUC da pior fonte.
- **(c) Stacking:** RL de duas entradas (os dois logits padronizados pela validação; C = 1) com **cross-fitting agrupado por fonte dentro da validação**: em cada uma das 4 fontes da validação (DFADD, ShiftySpeech, CodecFake+, LibriSpeech), a RL é treinada nas outras três e dá a nota dessa fonte (sempre fora da amostra). No dev_shift, a nota vem da RL treinada na validação inteira. Coeficientes reportados (validação inteira e por dobra).
- Vencedor da fusão: regra do desafiante (a → c → b).

**E3 · Front-end:** candidatos = melhor cabeça só Meta, melhor cabeça só Microsoft, vencedor da E2; regra do desafiante (Meta → Microsoft → fusão). O vencedor e as razões são registrados.

**E4 · Três cabeças básicas sobre o vencedor (uma de cada vez):**
- **RL básica:** a melhor camada única (AUC da pior fonte, com C = 1), padronizada, C = 1.
- **MLP básico:** a mesma entrada; 1 camada escondida de 256, ReLU, dropout 0,3; AdamW (lr 1e-3, weight decay 1e-3), lote 256, até 30 épocas, early stopping pela EER média por fonte **da validação** (3 fontes; paciência 5); seeds 42, 123 e 2026.
- **CNN básica sobre as 25 camadas** (⚠️ fora da trilha: a convolução anda ao longo das camadas do modelo, não no tempo): entrada 25 posições × 2.048 números (média e desvio de cada saída, padronizados com o treino); Conv1d(2048→64, k=3) + BatchNorm + ReLU, Conv1d(64→64, k=3) + ReLU, média global ao longo das posições, Linear(64→1); mesmo otimizador, lote, early stopping e seeds do MLP. Atenção (registrada antes): a CNN recebe também o desvio, que a RL e o MLP não recebem.
- Se o vencedor da E3 for a média das notas: cada cabeça é treinada nos dois modelos e as notas são juntadas pela média (padronizadas pela validação); stacking: idem, juntadas pelo stacking com cross-fitting; vetores colados: a entrada é a concatenação das melhores camadas dos dois (na CNN, os canais dos dois modelos lado a lado).
- Decisão: regra do desafiante RL → MLP → CNN; um MLP ou uma CNN só toma o lugar se a diferença for relevante **em cada uma das 3 seeds** (contra a mesma RL). Reportados: tempo, nº de parâmetros, métricas por fonte de cada seed e a referência v4.

**E5 · Maximizar a cabeça vencedora (uma modificação de cada vez, cumulativas, nesta ordem):**
- **Regra de manter:** a modificação fica se a AUC da pior fonte subir **≥ 0,003** sobre o estado atual **E** a AUC do dev_shift não cair mais que **0,01**. O efeito de cada passo (antes e depois) é reportado, inclusive o dos descartados.
- **Se for a RL:** (1) zona de camadas: a melhor entre as camadas únicas e os trios de vizinhas (l−1, l, l+1), com C = 1, pela AUC da pior fonte; (2) grade ampla de C: 8 valores de 1e-4 a 10 em escala log (`logspace(-4, 1, 8)`), pela AUC da pior fonte; (3) pesos iguais por fonte de treino (`sample_weight` = 1 / nº de clipes da fonte, reescalado para reais e fakes somarem o mesmo peso) no lugar de peso igual por clipe; (4) regra de limiar, ver abaixo.
- **Se for o MLP ou a CNN:** (1) weight decay 1e-2; (2) dropout 0,5; (3) até 60 épocas com paciência 10; (4) ensemble das 3 seeds (média dos logits); (5) regra de limiar. A regra de manter usa a média das 3 seeds (no ensemble, o próprio ensemble).
- **Se for fusão:** os passos da RL são feitos em cada modelo (cada um pela sua AUC da pior fonte) e a regra de manter é medida no sistema fundido.
- **Regra de limiar:** (A) EER global da validação; (C) mediana dos limiares de EER de cada fonte da validação com as duas classes; (D) em cada fonte da validação com reais, o limiar que deixa 5% de falso alarme nos reais (percentil 95 das notas dos reais); a mediana entre as fontes. Escolha por **validação cruzada deixando uma fonte de fora**: para cada fonte da validação com as duas classes (DFADD, ShiftySpeech, CodecFake+), o limiar é calculado com as outras fontes da validação e aplicado nela; vence a regra de maior acurácia balanceada média; se a diferença para a A for menor que 0,005, fica a A (a mais simples). O dev_shift só confirma (reportado, não escolhe). O limiar final é calculado na validação inteira pela regra escolhida.

**E6 · Teste (uma vez, só para mostrar os finalistas; roda num script separado, depois de E1 a E5 e dos finalistas gravados em `results/v6_finalistas.json`):**
- **Finalistas:** **F0** = v4 (XLS-R `concat_top3`, camadas 5, 6, 7, C = 0,1, limiar A), referência; **F1** = o front-end vencedor da E3 com a RL básica da E4; **F2** = o sistema final da E5, com o limiar escolhido.
- **Amostra:** os mesmos 51.320 clipes da v4 e da v5 (`src/teste_v4.py: amostra_teste_idx`, até 5.000 por classe e banco, seed 42). Se o vencedor envolver o WavLM (sozinho ou em fusão), extrai-se o teste do WavLM (orçamento de 2 h); senão, **não** se extrai e os números do WavLM no teste são os do colega (`results/v5_teste_*`), rotulados "do colega, sem notas por clipe".
- **Métricas por banco:** acurácia, acurácia balanceada, acurácia trivial, AUC, EER, recall de fake, falso alarme; média da acurácia dos três principais com duas classes (In-the-Wild, LlamaPartialSpoof, BRSpeech-DF); falso alarme do CORAA; ASVspoof 2019 e 2021 como secundários. Limiares A, B e o da E5. Bootstrap 1.000 por banco, dentro de cada classe, seed 42, pareado (**mede só a amostra de teste**, não o treino).
- **Leitura pré-definida (só reportada):** F1 − F0 e F2 − F0 em AUC e acurácia balanceada; "relevante" só com o mesmo sinal nos três principais e o IC95 sem o zero em cada um (a regra da v4). Nada é reescolhido depois do teste.
- **Contaminação:** os bancos já foram vistos duas vezes (v4 e v5); o teste da v6 é um terceiro olhar, que só serve para mostrar os finalistas.

**Execução:** `scripts/vigia_v6.ps1` → `scripts/rodar_v6.ps1` → `scripts/v6.py <etapa>` (E1 a E5 e finalistas; `AVALIAR_TESTE_FINAL = False`) e `scripts/v6_teste.py <etapa>` (E6); um processo pesado por vez, `NUM_WORKERS` 4 e depois 2, threads = 1, orçamento total da noite de 7 h.

**Detalhes fixados ao escrever o código (2026-11-10 22:35, ainda antes de qualquer resultado do funil; só a vazão do WavLM tinha sido medida):**
- Todas as etapas usam os **mesmos clipes**: treino, validação e dev_shift são sempre abertos nos dois modelos e um clipe com erro de leitura em qualquer um sai de todos, mesmo quando a etapa só usa um modelo.
- `sample_weight` por fonte (E5, passo 3): além de reais e fakes somarem o mesmo, os pesos são reescalados para **média 1** (assim o C continua na mesma escala do peso por clipe).
- Nos vetores colados, a E5 ajusta a zona de camadas de cada modelo pela RL dele sozinho, mas o C é **um só** (o da RL única dos vetores colados), escolhido pela pior fonte do sistema colado.
- Se a vencedora da E4 for MLP ou CNN e o ensemble não for mantido na E5, o F2 usa a **seed 42** (a primeira da lista).
- A extração do WavLM (treino, validação, dev_shift) foi lançada **diretamente** (`scripts/v6.py medir` e `extrair`, num processo oculto) antes do vigia, para ganhar tempo; o vigia só foi lançado depois que ela terminou (com o marcador gravado, ele pula a extração). Vazão medida (500 clipes, 4 workers): **35,4 clipes/s** (`results/v6_wavlm_vazao.json`), pico de GPU 3,76 GB, 0 NaN.

### v6 · fechamento (2026-11-10)

- **Data e execução:** extração do WavLM às 22:24 a 22:48 (`scripts/v6.py medir` e `extrair`, direto); funil e teste às 23:07 a 23:26 pelo vigia (`scripts/vigia_v6.ps1`, 1 tentativa, nenhuma falha). Branch `v6-fusao`, commit `9d62c03` com mudanças não commitadas. Tudo pelo pré-registro acima, **sem nenhuma mudança** de regra; o teste foi avaliado **uma vez**, depois dos finalistas gravados (`results/v6_finalistas.json`, 23:23:25).
- **Conferências:** os dois modelos têm os mesmos 40.000 / 19.999 / 10.614 clipes (mesmo `idx_manifest` e caminho, na mesma ordem; 0 erros de leitura). Reprodução do XLS-R × v4: **0,0 ponto** de diferença de EER média por fonte em todas as camadas; notas da cabeça da v4 iguais às gravadas (diferença 0). No teste, F0 reproduziu a v4 (In-the-Wild 0,979, CORAA 16,7% / 2,5%) e as notas dos três finalistas, recalculadas, bateram com as gravadas antes do teste (diferença 0).
- **Funil** (`results/v6_funil.csv`; AUC da pior fonte, val + dev_shift; a pior fonte foi o **CodecFake+** em todas as cabeças comparadas):

  | etapa | candidatos (AUC da pior fonte) | vencedor pela regra | por quê |
  |:--|:--|:--|:--|
  | E1 | Meta `concat_top4` (5, 7, 8, 11; C = 1) 0,907 · Microsoft `concat_top3` (4, 6, 7; C = 1) 0,891 · v4 0,903 | empate | Microsoft − Meta −0,017 (IC95 −0,048 a +0,001) |
  | E2 | média das notas 0,905 · stacking 0,905 · vetores colados 0,874 | média das notas | stacking +0,0004 (IC95 −0,0005 a +0,0013); colados pior |
  | E3 | só Meta 0,907 · só Microsoft 0,891 · fusão (média) 0,905 | **só Meta** | nenhum desafiante relevante (fusão −0,002, IC95 −0,012 a +0,008) |
  | E4 | RL 0,904 · MLP 0,912 / 0,906 / 0,902 · CNN 0,906 / 0,908 / 0,910 | **RL** | nenhuma seed com diferença relevante (todos os IC95 com o zero) |
  | E5 | zona (+0,000) · C ampla (+0,001) · pesos por fonte (+0,006) · limiar A/C/D | pesos por fonte; limiar **A** | só os pesos passaram a régua (+0,003); D não ganhou 0,005 na validação cruzada (0,917 × 0,916) |

  Antes de comparar as fusões: os dois modelos acertam juntos 97,7% da validação e erram juntos 0,5%; no dev_shift, só a Meta acerta 6,4% e só a Microsoft 3,2%; correlação dos logits de 0,51 a 0,77 por classe. Stacking: pesos 4,4 (Meta) e 4,0 (Microsoft). Os vetores colados tiveram dev_shift 0,996, mas CodecFake+ 0,874.
- **Teste (E6; `results/v6_teste_final.csv`, `v6_teste_diferencas.csv`, `v6_teste_resumo.json`; cada célula: AUC · acurácia · acurácia balanceada, no limiar principal, que foi A nos três):**

  | banco | F0 (v4) | F1 (só Meta + RL básica) | F2 (sistema final) | trivial |
  |:--|:--|:--|:--|--:|
  | in_the_wild | 0,979 · 0,836 · 0,835 | 0,972 · 0,807 · 0,807 | 0,972 · 0,812 · 0,812 | 0,500 |
  | llamapartialspoof | 0,925 · 0,855 · 0,855 | 0,940 · 0,863 · 0,863 | 0,937 · 0,863 · 0,863 | 0,500 |
  | brspeech_df (pt-BR) | 0,842 · 0,863 · 0,679 | 0,741 · 0,814 · 0,576 | 0,744 · 0,813 · 0,571 | 0,791 |
  | coraa (só real): falso alarme A / B | 16,7% / 2,5% | 41,2% / 17,4% | 40,4% / 17,3% | — |
  | asvspoof2019_la (secundário) | 0,999 · 0,981 · 0,981 | 0,997 · 0,975 · 0,975 | 0,997 · 0,975 · 0,975 | 0,500 |
  | asvspoof2021_df (secundário) | 0,997 · 0,942 · 0,942 | 0,996 · 0,954 · 0,954 | 0,996 · 0,950 · 0,950 | 0,500 |
  | **média dos 3 principais (acurácia)** | **0,851** | 0,828 | 0,829 | 0,597 |

  Leitura pré-definida: F1 − F0 e F2 − F0 **"não demonstradas"** pela regra (sinais trocados): piores no In-the-Wild (AUC −0,007) e no BRSpeech-DF (−0,100 / −0,097), melhores no LlamaPartialSpoof (+0,015 / +0,012), todos com IC95 sem o zero. O WavLM não foi extraído no teste (nenhum finalista o usa); os números dele no teste são os da v5 (do colega, sem notas por clipe).
- **Conclusão em uma frase:** nada superou o XLS-R com regressão logística pelas regras (nem o WavLM, nem a fusão, nem MLP ou CNN), e o sistema escolhido pela AUC da pior fonte ficou pior que a v4 no teste, sobretudo em pt-BR, porque a pior fonte foi sempre o CodecFake+ (352 clipes), que não representa o português; a v4 continua sendo o modelo do projeto.
- **Por que (descritivo):** escolher "a melhor camada única pela pior fonte, com C = 1" levou à camada 11 (dev_shift 0,972), em vez das camadas do meio (camada 7: 0,988), e tirou a regularização da v4 (C = 0,1). Ajudou no CodecFake+ (parecido com o treino) e atrapalhou o português (diferente do treino): o tema overfitting × regularização. A régua da E5 sem bootstrap aceitou +0,006 numa fonte de 352 clipes, e no teste o F2 ficou igual ao F1.
- **Limitações:** uma amostra de treino; dev_shift com 16 locutores; CodecFake+ pequeno e sem locutor; os bancos de teste estavam consumidos (terceiro olhar); o bootstrap do teste mede só a amostra de teste; a CNN recebeu média e desvio e a RL/MLP só a média; a projeção da CNN da v3 (opcional) não foi feita.
- **Arquivos:** notebook `notebooks/versions/v6_funil.ipynb` (igual ao `notebook.ipynb`); decisões `docs/DECISOES_V6.md`; no git: `results/v6_*.csv`, `.json`, `.png` e `results/v6_conclusao.md`; fora do git: notas por clipe (`results/v6_notas_*.parquet`, `v6_teste_notas.parquet`), `results/v6_parcial/`, o cache `data/processed/v6_wavlm_cache/` (6,8 GB), os checkpoints `models/v6_*.pt` e os logs.
- **Tempos** (RTX 4060 8 GB, 4 workers; pico = processo + workers):

  | etapa | tempo | pico |
  |:--|--:|--:|
  | vazão do WavLM (500 clipes) | 0,4 min | 11,6 GB |
  | extração do WavLM (70.613 clipes) | 23,3 min | 14,2 GB (sistema 33,6 GB) |
  | E1 (25 camadas × 4 C × 2 modelos + concatenações + bootstrap) | 7,8 min | 2,7 GB |
  | E2 / E3 / E4 / E5 / finalistas | 0,4 / 0,1 / 3,0 / 3,8 / 0,1 min | 7,2 / 1,8 / 9,3 / 2,3 / 1,2 GB |
  | E6 (teste: 3 finalistas, 1.000 bootstraps por banco) | 1,1 min | 2,6 GB |
  | relatório (funil, gráficos, PCA e t-SNE) / notebook | 1,0 / 0,1 min | 1,4 GB / — |

### Notebook final: volta ao sistema v4 (F0) (2026-11-11)

- **O que mudou:** o `notebook.ipynb` da raiz deixou de ser a v6 (funil) e voltou a ser o **sistema v4 (F0)** completo, como documento final: pipeline, a regressão logística explicada, 14 gráficos (G1 a G14), o teste da v4, o caminho v1 a v6 e as limitações. Cópia em `notebooks/versions/v4_final.ipynb`. O notebook da v6 continua arquivado, sem mudança, em `notebooks/versions/v6_funil.ipynb` (era idêntico ao da raiz antes da troca).
- **Motivo:** nada depois da v4 superou o F0 no teste (v5: WavLM; v6: fusão, MLP, CNN, ajuste fino), e o sistema declarado pela v6 (F2) ficou pior no teste. **Ressalva:** a decisão de voltar ao F0 foi tomada depois de ver o teste da v6; o F0 em si foi escolhido pela validação antes de qualquer teste (pré-registro da v4), e os números de teste do F0 são os da v4 original.
- **Nenhuma decisão nova de modelo:** camadas 5, 6, 7; C = 0,1; limiares A = −0,0395 e B = 8,2184 (logit), todos lidos de `results/v4_*` e conferidos. O F0 foi **reajustado** sobre o `v4_cache` (mesmos dados e C; lbfgs, determinístico) e reproduziu as notas gravadas da v4 (diferença máxima ~1e-5 de logit; AUC igual com 3 casas em todos os bancos). Nada foi extraído nem baixado.
- **Diagnósticos novos (só para os gráficos, não escolhem nada):** curva de aprendizado (2 mil a 40 mil clipes) e caminho do C (0,0001 a 10). Achado registrado como hipótese para o futuro: o dev_shift melhora com **mais** freio (C = 0,0001: AUC 0,994 × 0,983 do F0) e **não** melhora com mais clipes de treino (2 mil: 0,993; 40 mil: 0,983), enquanto a validação prefere C = 0,1. Foi visto olhando o dev_shift (16 locutores) e não muda o F0.
- **Arquivos:** `scripts/v4final_figuras.py` (figuras e tabelas, ~2 min, pico 5,5 GB) → `results/v4final_*.png`, `results/v4final_*.csv`, `results/v4final_resumo.json`, `results/v4final_f0_coeficientes.npz`; `scripts/nb_final.py` (monta e executa o notebook, só lê do disco; 3,4 MB, 0 erros); `docs/ROTEIRO_APRESENTACAO.md`.
- **Atenção (execução):** a célula `config` do notebook da raiz é agora a da v4 (os scripts da v4 voltam a lê-la). O `scripts/v6.py` e o `scripts/v6_notebook.py` exigem a célula da v6: para rodá-los de novo, use `--notebook notebooks/versions/v6_funil.ipynb` no `scripts/v6.py` e no `scripts/v6_teste.py`; o `scripts/v6_notebook.py` lê o `notebook.ipynb` fixo e pararia com erro (não foi alterado).

## f) Ideias ainda não testadas

**Da v6 (para a v7):**
- **Um conjunto de decisão em português** (reais e fakes pt-BR fora dos testes): a v6 mostrou que a validação atual (inglês, estúdio) não protege o pt-BR; sem isso, qualquer escolha fina pode piorar o português sem ninguém ver.
- **Uma fonte de codec maior** (por exemplo, o CoRS do CodecFake+, que não foi baixado): a pior fonte foi sempre o CodecFake+ de 352 clipes; decisões em milésimos de AUC nele são frágeis.
- **Pior fonte com régua de ruído:** usar a pior fonte só quando a diferença passar no bootstrap também na E5, ou usar a média das duas piores.
- **CNN sobre as camadas só com a média** (ablação do desvio), para separar o efeito da arquitetura do efeito da entrada.
- Seguem valendo: limiar por domínio, várias amostras de treino, reais pt-BR no treino.

**Da v5 (para a v6):**
- *(Testado na v6: juntar o WavLM com o XLS-R, por média das notas, stacking e vetores colados, não superou o XLS-R sozinho.)*
- **Modelos pré-treinados multilíngues** (o XLS-R ganhou sobretudo em pt-BR; o WavLM, só inglês, perdeu ali): se o projeto testar outro modelo, preferir um multilíngue (por exemplo, o XLS-R 1B ou o MMS, também da Meta).
- **Juntar as camadas de baixo do WavLM com as do meio do XLS-R** (no dev_shift, as camadas 0 a 4 do WavLM foram melhores que as do XLS-R): só com escolha pela validação + dev_shift e confirmação num banco novo.
- As ideias da v4 continuam valendo (reais em pt-BR no treino, limiar por domínio, várias amostras de treino).

**Da v4 (para a v5)** (atenção: os seis bancos de teste estão consumidos; qualquer decisão nova tem de sair da validação e do dev_shift, ou de um banco novo):
- **Reais em pt-BR no treino** (e fakes pt-BR de geradores que não estão nos testes): o pior resultado da v4 é o pt-BR (BRSpeech-DF AUC 0,842; CORAA com 16,7% de falso alarme no limiar A).
- **Limiar por domínio:** o ponto de EER da validação (estúdio) e o do dev_shift (internet) diferem muito (S2: logit −0,04 × 8,2). Testar uma validação com mais áudio "de verdade", ou calibrar o limiar num conjunto de desenvolvimento do domínio de uso.
- **Várias amostras de treino / seeds** para o S1 e o S2, para medir quanto do resultado é sorte do sorteio de treino (a v4 tem uma só).
- **Diferença S2 × S1 no pt-BR** (+0,066 de AUC no BRSpeech-DF): ver se ela aparece também num banco pt-BR novo, antes de chamar de efeito.
- *(Testado na v4: juntar a CNN da v3 por uma regressão na validação piorou o In-the-Wild; não repetir sem um conjunto de ajuste fora do domínio de estúdio.)*

**Da v3 (para a v4):**
- *(Feito na v4: XLS-R congelado + regressão logística, corte de silêncio e áudio sem padding. O modelo congelado explicou quase todo o ganho; o corte, nenhum.)*
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

**v4 (modelo congelado + cabeças; retomável, com vigia):** na raiz, no PowerShell:

```powershell
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File scripts\vigia_v4.ps1"   # noite inteira, sozinho
Get-Content logs\v4_status.txt          # STATUS, tentativa, etapa, hora, horas de extração
Get-Content logs\v4_resumo.txt          # resultados principais, atualizados a cada etapa
powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1           # ou: uma rodada sem o vigia
powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1 -Quick    # teste rápido (arquivos _quick)
python scripts/v4.py medir --n 500      # vazão (com o env ativo)
python scripts/v4.py camadas            # uma etapa só: extrair | camadas | somas | atalhos | resumo | status | concluir
python scripts/acuracias_v3.py          # Parte A: acurácias retroativas da v3
```

**Teste final da v4 (avaliado uma vez; já feito em 2026-11-10):** o mesmo padrão, com `scripts\vigia_v4_teste.ps1` → `scripts\rodar_v4_teste.ps1` → `scripts/v4_teste.py <extrair|cnn|avaliar|status|concluir>`. Status em `logs\v4_teste_status.txt`, resumo em `logs\v4_teste_resumo.txt`, marcador final `results\v4_teste_final_concluido.json`. Com o marcador presente, rodar de novo não faz nada. `--quick` (com `V4_SUFIXO_EXTRA=_t2`) usa bancos de mentira tirados da validação.

Cada etapa grava um marcador `results/v4_<etapa>_concluido.json`; rodar de novo pula o que terminou e continua o resto (shards, camadas e seeds prontos). O orçamento de extração é cumulativo (`results/v4_estado.json`). O cache fica em `data/processed/v4_cache/` (~10 GB, fora do git). Para refazer do zero, é preciso apagar à mão o cache e os `results/v4_*` (os scripts nunca apagam nada de outra configuração: param com erro).

**v5 (WavLM Large × XLS-R; o mesmo pipeline da v4, um modelo por vez):** na raiz, no PowerShell, com o Claude Code e o VS Code fechados:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1 -Modelo xlsr     # XLS-R refeito (prefixo v5_xlsr)
powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1 -Modelo wavlm    # WavLM Large (prefixo v5_wavlm)
powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1 -Quick -Modelo wavlm   # teste rápido (arquivos _quick)
Get-Content logs\v5_wavlm_resumo.txt    # resumo de cada modelo, atualizado a cada etapa
```

Sem `-Modelo`, o `rodar_v4.ps1` e o `scripts/v4.py` usam o prefixo `v4` (como a v4 foi rodada). Cache em `data/processed/v5_<modelo>_cache/` (fora do git). O `vigia_v4.ps1` não foi adaptado para a v5.

**v6 (funil; retomável, com vigia):** na raiz, no PowerShell:

```powershell
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File scriptsigia_v6.ps1"   # noite inteira, sozinho
Get-Content logs6_status.txt          # STATUS, tentativa, etapa, folga de memória
Get-Content logs6_resumo.txt          # funil e teste, atualizados a cada etapa (também logs6_PROGRESSO.md)
powershell -ExecutionPolicy Bypass -File scripts
odar_v6.ps1           # ou: uma rodada sem o vigia
powershell -ExecutionPolicy Bypass -File scripts
odar_v6.ps1 -Quick    # teste rápido (arquivos _quick; V6_SUFIXO_EXTRA separa testes)
python scripts/v6.py e1                 # uma etapa só: medir | extrair | e1 | e2 | e3 | e4 | e5 | finalistas | relatorio | status | concluir
python scripts/v6_teste.py avaliar      # E6 (só depois dos finalistas): extrair | avaliar
python scripts/v6_notebook.py           # monta e executa o notebook (só lê do disco) e copia para notebooks/versions/v6_funil.ipynb
```

O XLS-R é lido do `data/processed/v4_cache/` (não é reextraído); o WavLM fica em `data/processed/v6_wavlm_cache/` (6,8 GB, fora do git), extraído nos mesmos clipes. Cada etapa grava `results/v6_<etapa>_concluido.json`; com o marcador, rodar de novo pula a etapa (o relatório e o notebook são refeitos quando há marcador mais novo). A conclusão escrita do notebook fica em `results/v6_conclusao.md` (sem ele, a seção 8 diz "pendente").

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

### 2026-11-09 · memória e execução sem supervisão (v4)

- **Processo principal da extração:** ~7,5 GB de memória reservada com o modelo em fp16 na GPU (os kernels de GPU ficam na memória da CPU; já é o carregamento "preguiçoso" do CUDA). Cada worker de leitura: ~0,9 GB. Pico medido com 4 workers: 11,5 a 13,2 GB; com formatos de lote variados (antes dos baldes), 19,2 GB. Cabeças: 1,1 a 2,1 GB (sem workers).
- **Lotes da GPU com formato fixo** (baldes de 2, 4, 6 e 8 s, `BALDE_S = 2`): ver "v4 · estado da execução", decisão 2. Não muda o método (resultado de cada clipe igual dentro da tolerância do fp16).
- **Vigia** (`scripts/vigia_v4.ps1`): instância única (PID em `logs\v4_vigia.pid`), impede a suspensão do Windows, espera folga de memória ≥ 14 GB (até 60 min), tentativas 1 e 2 com 4 workers e da 3ª em diante com 2, espera de 10 min entre tentativas, até 8. O número de workers não entra na configuração do método (a leitura é determinística: sem sorteio na extração).
- **Workers órfãos:** depois de uma queda à força, um worker pode ficar vivo (medido no teste: 0,9 GB); o `rodar_v4.ps1` encerra workers cujo processo principal já morreu.

### 2026-11-10 · teste final da v4 (execução)

- **Scripts novos, no mesmo padrão da v4** (os da v4 não foram alterados): `src/teste_v4.py` (amostra, plano, escolha do S0, métricas, bootstrap, gráfico), `scripts/v4_teste.py` (etapas `extrair`, `cnn`, `avaliar`, `status`, `concluir`), `scripts/rodar_v4_teste.ps1` e `scripts/vigia_v4_teste.ps1` (cópia do `vigia_v4.ps1` com os nomes trocados: status em `logs/v4_teste_status.txt`, resumo em `logs/v4_teste_resumo.txt`, marcador final `results/v4_teste_final_concluido.json`). Cache em `data/processed/v4_cache_teste/` (fora do git).
- **Teste QUICK antes** (2026-11-10 07:27): bancos **de mentira tirados da validação** (dfadd, shiftyspeech, codecfake_plus e librispeech só real; 140 clipes fora da validação QUICK), cache QUICK da v4 (`_quick_t2`); **nenhum áudio de teste foi lido**. As três etapas e a conclusão rodaram sem erro; o reajuste do S1 e do S2 reproduziu as notas gravadas da v4 com diferença máxima 0; o vigia QUICK saiu com CONCLUIDO. Os números do QUICK não valem e não foram interpretados.
- **Folga de memória do vigia: 11 GB em vez de 14** (`-FolgaGB 11`). Motivo: na execução da v4, com esta sessão aberta, a folga ficou em ~12,1 GB e o vigia esperou os 60 min inteiros à toa; depois rodou com 4 workers sem queda (pico 14,2 GB do processo + workers; o sistema chegou a 36,5 GB porque o arquivo de paginação cresce). As outras regras (tentativas com 4 e depois 2 workers, instância única, sem suspensão) são as mesmas. Não muda o método.

### 2026-11-10 · v5 (execução, antes dos resultados)

- **Código da v4 reaproveitado, sem cópia:** o `scripts/v4.py` e o `scripts/rodar_v4.ps1` passaram a usar o prefixo da configuração (`PREFIXO`; `-Modelo xlsr|wavlm` no `.ps1`, que define `V5_MODELO`). Sem prefixo, tudo continua com `v4` (os arquivos da v4 não mudam de nome). O `src/ssl.py` carrega o modelo com `AutoModel` (escolhe `Wav2Vec2Model` para o XLS-R, a mesma classe da v4, e `WavLMModel` para o WavLM) e confere que há 25 saídas de 1024.
- **Máquina:** RTX 4050 Laptop 6 GB, 31,6 GB de RAM, limite de memória reservada (commit) 35,4 GB.
- **Vazão do WavLM** (`scripts/v4.py medir --n 500`, 4 workers, baldes de 2 s; `results/v5_wavlm_vazao.json`): **18,5 clipes/s** (22,6 sem a partida dos workers; GPU sozinha 23,9), **pico de GPU 3,76 GB** (cabe nos 6 GB sem mudar o lote), pico de memória 11,4 GB (principal 7,4 GB + 4 workers de ~0,9 GB), 0 NaN. Estimativa (é estimativa): 70.613 clipes / ~20 clipes/s ≈ **1 h de extração por modelo**, mais ~15 min de cabeças.
- **fp16 nos dois modelos (como na v4):** em 16 clipes da validação, o erro relativo do fp16 contra o fp32 foi parecido nos dois (mediana 0,04% no WavLM e 0,06% no XLS-R; nas camadas 5 a 9, abaixo de 0,1% nos dois; o máximo, 3,3% e 2,3%, nas últimas camadas). O bf16 erra ~10 vezes mais nos dois (mediana 0,6 a 0,7%) e não foi usado.
- **Padding do lote:** lote no balde (32 × 8 s) × clipe isolado, em 12 clipes: WavLM 0,08% em fp32 (3,3% em fp16, o mesmo erro do fp16 acima); XLS-R 0,50% / 0,57%. O `attention_mask` funciona no WavLM. Aviso inofensivo do PyTorch no WavLM: "mismatched key_padding_mask and attn_mask is deprecated".
- **Plano de clipes:** idêntico entre os dois modelos (conferido no QUICK) e, no plano completo, com os mesmos tamanhos da v4 (dev_shift 10.614, validação 19.999, treino 40.000; seed 42, mesmo manifest). A identidade clipe a clipe com a v4 é conferida pela reprodução dos números do XLS-R (pré-registro da v5).
- **QUICK (2026-11-10 11:47 a 11:52, ~300 clipes; os números não valem e não foram interpretados):** `rodar_v4.ps1 -Quick -Modelo wavlm` e `-Modelo xlsr` rodaram as cinco etapas e a conclusão, código 0, com arquivos separados por prefixo (`results/v5_wavlm_*_quick*`, `results/v5_xlsr_*_quick*`).

### 2026-11-10 · teste final da v5 (execução, antes dos resultados)

- **Script novo** `scripts/v5_teste.py` (etapas `extrair --modelo xlsr|wavlm`, `avaliar`, `status`, `concluir`), que reaproveita `src/teste_v4.py` (amostra, plano, métricas, bootstrap) e `src/extracao_v4.py`. O `src/teste_v4.py: grafico` passou a aceitar outros sistemas (sem argumentos, desenha os da v4 como antes). Cache de teste por modelo (`data/processed/v5_<modelo>_cache_teste/`); resultados em `results/v5_teste_*`; a extração impede a suspensão do Windows.
- **Um clipe com erro de leitura em qualquer dos dois modelos sai dos dois** (comparação pareada).
- **QUICK (2026-11-10 15:52 a 15:56; bancos de mentira da validação, 140 clipes; nenhum áudio de teste lido; os números não valem e não foram interpretados):** as etapas e a conclusão rodaram. O reajuste reproduziu as notas gravadas da v5 com diferença 0. No QUICK, a cabeça final às vezes é uma camada só (`rl_camada_N`): o script trata esse caso como o "só camada 24" (soma ponderada como cabeça final para o teste e não está prevista: o script para).
- **Erro visto no QUICK e corrigido:** na extração do WavLM, a renomeação da pasta do shard (`shard_00000.tmp` → `shard_00000`) falhou com "Acesso negado" (o Windows segurava a pasta; antivírus ou indexador). A nova tentativa descartou o shard incompleto e refez sem erro. Para a execução longa, `src/cache_v4.py: gravar_shard` passou a tentar a renomeação até 10 vezes, com espera de 1 a 10 s. Execução, não método: o conteúdo do shard é o mesmo.

### 2026-11-10 · notebook da v5 (execução)

- O env não tinha `nbconvert`; para executar o `notebook.ipynb` sem abrir o VS Code, foi instalado só o `nbclient` (0.11.0; `pip install nbclient`, que não mexe em torch nem em transformers). Ele não entra no `requirements.txt`, porque o projeto não depende dele: o notebook também roda no VS Code com o kernel do `env`.
- O notebook da v5 só lê do disco (executado em ~10 s, 0 erros, 0,33 MB) e foi copiado com as saídas para `notebooks/versions/v5_wavlm.ipynb`. As figuras da seção 4 (histograma das notas) e o plano da seção 3 dependem de arquivos fora do git; sem eles, as células mostram "Pendente".

### 2026-11-10 · v6 (execução)

- **Código novo, no padrão da v4** (os scripts da v4 e da v5 não foram alterados): `src/v6_metricas.py` (pior fonte, bootstrap pareado por locutor com pesos de reamostragem fixos, regra do desafiante, regras de limiar A/C/D e validação cruzada por fonte), `src/v6_cabecas.py` (dados dos dois caches com os mesmos clipes, RL, MLP, CNN ao longo das camadas, fusões), `src/v6_relatorio.py` (funil, gráficos, PCA/t-SNE, status), `scripts/v6.py`, `scripts/v6_teste.py`, `scripts/v6_notebook.py`, `scripts/rodar_v6.ps1` e `scripts/vigia_v6.ps1` (cópia do `vigia_v4.ps1`: folga mínima de 11 GB, espera de até 10 min, 4 e depois 2 workers, até 8 tentativas, teto de 7 h, `logs/v6_BLOQUEIO.txt` se não concluir).
- **`src/ssl.py: carregar_ssl`** ganhou o parâmetro opcional `revisao`: o WavLM foi baixado por revisão fixa (`c1423ed…`) e o cache do Hugging Face ficou sem o ponteiro `refs/main`, então o modo offline não achava o modelo. Sem `revisao`, o comportamento é o de antes.
- **Por que não o `rodar_v4.ps1 -Modelo wavlm`:** ele grava em `results/v5_wavlm_*` e sobrescreveria arquivos do colega que estão no git. A extração usa as mesmas funções (`src/extracao_v4.py`, `src/ssl.py`), com o plano copiado do `v4_cache`.
- **Memória:** com esta sessão aberta, a extração deixou a folga em ~1 GB (sistema 33,6 GB de 34,3 GB); por isso os testes QUICK só rodaram depois que ela terminou. As etapas do funil ficaram entre 1,2 e 9,3 GB.
- **QUICK (2026-11-10 22:23 a 23:07; ~300 clipes; os números não valem e não foram interpretados):** `rodar_v6.ps1 -Quick` de ponta a ponta (sufixo `_quick_t1`; um erro de código achado e corrigido, a segunda chamada retomou sem refazer a extração); o vigia QUICK do zero (`_quick_t2`, código 0 em 2,5 min); e um teste de interrupção (`_quick_t3`): a extração foi morta à força depois do 1º shard da validação, o `rodar_v6` registrou a falha e pulou as etapas dependentes, o vigia esperou 2 min e a 2ª tentativa pulou os 3 shards prontos e terminou tudo; os shards da execução interrompida são **idênticos bit a bit** aos da execução sem interrupção.
- **nbclient** (já instalado na v5) executa o notebook; ele só lê do disco (0 erros, 1,0 MB).

