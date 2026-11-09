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

Na v2, dev_shift e teste são por janelas; "EER val global" é no recorte central. Recall e falso alarme (CORAA) dependem do limiar, que mudou muito entre os experimentos da v2 (0,60 / 0,10 / 0,76).

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

## f) Ideias ainda não testadas

**Da v2 (para a v3):**
- **Repetir cada experimento com 2 ou 3 seeds** (~30 min cada) antes de comparar versões. É a única forma de separar efeito de acaso; na v2, o controle mexeu no dev_shift tanto quanto a mudança testada.
- **Augmentation de gravação antiga / sala:** reverberação sintética, banda estreita com chiado de fita, filtro de rádio AM. É o tipo de real que o modelo mais acusa (locutores históricos do dev_shift).
- **Tratar áudio curto sem repetição** (completar com silêncio, ou avaliar só a parte que existe), já que a inversão é pior nos áudios curtos e 68% dos reais do In-the-Wild têm menos de 4 s. Uma mudança só, decidida no dev_shift.
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
