# Dados do apêndice, parte 2 (bancos, métodos e resultados v1 a v6)

**Gerado em 16/11/2026, só por leitura do repositório.** Nada foi treinado, nenhuma API foi chamada e nenhum arquivo existente foi alterado. Complementa `docs/DADOS_APENDICE.md` (que cobre a v7, a v8, o modelo leve e os testes externos).

Cada número traz o arquivo de onde veio. Onde o dado não existe no repositório, está escrito **NÃO ENCONTRADO**.

Alguns números são **contas simples sobre valores já gravados**, sem estimativa, e vêm marcados com "(calculado)". O principal caso é a acurácia balanceada = (recall de IA + 1 − falso alarme) / 2.

Convenções:
- AUC e EER são as métricas principais.
- "Acc. bal." = acurácia balanceada.
- Nota = logit (maior = mais parece IA).
- Nomes de pessoas e modelos de máquina foram omitidos de propósito.

---

## 1. Bancos

Fontes gerais:
- `docs/datasets.yaml` (URL, licença, papel);
- `docs/dataset_overview.md` (origem, rótulo, geradores);
- `docs/audit_report.md` (estrutura e locutor de cada banco);
- `docs/dados_teste_internet.md` (RADAR e ML-ITW);
- os README/LICENSE que vieram junto dos próprios bancos em `data/raw/` (só lidos).

"Publicado por" traz só a organização ou o repositório citado no repositório. Os nomes de autores dos artigos não foram copiados.

| banco | publicado por / onde | tipo de áudio | geradores citados | licença | regras especiais e ressalvas |
|:--|:--|:--|:--|:--|:--|
| **ASVspoof 2019 LA** | desafio ASVspoof; Edinburgh DataShare (`datasets.yaml`) | Real: leitura em estúdio. Os locutores `LA_*` são do VCTK anonimizado (`audit_report.md` §2 e nota *). IA: TTS e conversão de voz | A01–A19. O `README.LA.txt` local classifica cada um: TTS (A01–A04, A07–A12, A16), VC (A05, A06, A17–A19) e TTS_VC (A13–A15) | ODC-By (`datasets.yaml`). O README local não traz licença | train+dev inteiros no treino. O eval é teste **secundário**: pode repetir locutores do treino via VCTK/DFADD (`audit_report.md`). 838 áudios de cadastro do ASV foram excluídos |
| **DFADD** | Hugging Face (URL em `datasets.yaml`). Instituição: **NÃO ENCONTRADO** | Real = VCTK inteiro (44.455 `.wav`, leitura em estúdio, 110 locutores). IA = TTS | GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2 (`dataset_overview.md`) | **NÃO ENCONTRADO** (`null` no yaml; "não documentado" no overview) | É 64% do treino da v1. Por isso o projeto usa teto por fonte e por gerador (`dataset_overview.md`, Decisões) |
| **ShiftySpeech** | Hugging Face (URL em `datasets.yaml`). Instituição: **NÃO ENCONTRADO** | **Audiolivro.** Real = LibriSpeech dev/test clean/other. IA = **ressíntese por vocoder** do mesmo áudio real (`audit_report.md`). Só a parte `audiobook` foi baixada | hifigan, vocos, wavegrad | **NÃO ENCONTRADO** | Não há arquivo de rótulo: o rótulo vem do nome da pasta (decisão do projeto). Foram excluídos 21.048 áudios dos 79 locutores do LlamaPartialSpoof (vazamento) |
| **CodecFake+** (só CoSG) | Hugging Face (organização CodecFake, URL em `datasets.yaml`) | Amostras de **páginas de demonstração** de 17 TTS baseados em codec neural, reais e IA. Sem locutor (`audit_report.md`) | MASKGCT, CLAMTTS, VALLE, NS3, SIMPLESPEECH1/2, GPST, NS2, SINGLECODEC, UNIAUDIO, ELLAV, SPEECHX, TACOLM, VIOLA, TI1G1R, RALLE, USLM (`dataset_overview.md`) | **NÃO ENCONTRADO** | O CoRS não foi baixado (só os rótulos). São 352 clipes na validação, a "pior fonte" da v6 |
| **LibriSpeech** (train-clean-100) | OpenSLR 12 (`datasets.yaml`) | **Audiolivro** (leitura, LibriVox), só real (`dataset_overview.md`) | — | CC BY 4.0 (`datasets.yaml`) | dev-clean e test-clean excluídos: são a mesma origem dos reais do LlamaPartialSpoof |
| **In-the-Wild** | Hugging Face (URL em `datasets.yaml`). Instituição: **NÃO ENCONTRADO** | Áudio de internet de 54 celebridades. Real e IA do mesmo locutor (`audit_report.md`). O `ROTEIRO_APRESENTACAO.md` chama de "entrevistas" | **NÃO ENCONTRADO** (`generator="unknown"` no manifest). O `attribution.txt` local só agradece a um criador de deepfakes | **NÃO ENCONTRADO** | Nunca no treino. Dividido em dev_shift (30% dos locutores, seed 42) e test_final. Já tinha sido consultado inteiro na v1 (`HISTORICO_VERSOES.md`, v3, critério 9) |
| **LlamaPartialSpoof** (v1.0.b) | GitHub (URL em `datasets.yaml`). Artigo arXiv 2409.14743 (`README.md` local) | Texto gerado por um LLM e lido por TTS, "do ponto de vista do atacante" (simulação de desinformação). IA completa (33.461) e **parcial** (96.582: crossfade, cut/paste, overlap/add). Real = LibriTTS dev/test-clean (10.573; `audit_report.md`) | cosyvoice, elevenlab, gptsovits, ljjets, xttsv2, yourtts | **CC BY 4.0** (`LICENSE.txt` e `README.md` locais; `null` no yaml) | Só avaliação. O dev do RADAR é derivado dele (`README.txt` do RADAR) |
| **BRSpeech-DF** (split test) | Hugging Face, organização AKCIT-Deepfake (`datasets.yaml`) | **Audiolivro: sim.** Os reais vêm do LibriVox segundo o card citado em `audit_report.md` §2, e o nome dos arquivos segue o padrão locutor_livro_segmento do MLS/LibriVox. **O card não está no repositório**: a afirmação vem só do `audit_report.md`. Mistura pt-BR e pt-PT, sem indicação por arquivo | toucantts, xtts, f5tts, yourtts, fish-speech | Apache-2.0 (`datasets.yaml`) | **Atalho de taxa:** reais em 48 kHz, IA em 24 kHz. Tudo é reamostrado para 16 kHz (`audit_report.md`). São 10 locutores, deduzidos do nome do arquivo |
| **CORAA** v1.1 (test) | Hugging Face (URL em `datasets.yaml`). Instituição: **NÃO ENCONTRADO** | Corpus de reconhecimento de fala, só real. Subpastas `CORAL`, `NURC_RE`, `Ted_part1/3`, `alip`, `sp` (`audit_report.md`). Tipo de fala de cada subconjunto (espontânea, entrevista, palestra): **NÃO ENCONTRADO** no repositório | — | **CC BY-NC-ND 4.0** (`datasets.yaml`) | 326 `pt_pt` (TEDx) excluídos. Serve só para medir falso alarme |
| **ASVspoof 2021 DF** (parte 1 de 4) | Zenodo 4835108 (`datasets.yaml`) | Real e IA **comprimidos por codec** (coluna 3 de `trial_metadata.txt`). Corpora de origem: asvspoof, vcc2018 e vcc2020 (`audit_report.md`) | A07–A19, HUB-*, SPO-*, Task1/2-teamNN (110 rótulos; `dataset_overview.md`) | **DbCL** (`LICENSE.DF.txt` local; `null` no yaml) | Teste secundário: os 67 locutores `asvspoof:*` são os do 2019 eval. 42,4% dos FLAC só abrem com o `miniaudio` |
| **RADAR 2026** (só eval) | Desafio APSIPA Grand Challenge (RADAR Challenge 2026); Zenodo 22110815; artigo arXiv 2605.09568 (`README.txt` local) | Fala real e IA passada por **transformações de mídia**: codec, reamostragem, ruído, música, sala, ajuste de pico, fades, corte de silêncio, enchimento com zeros, compressão de faixa dinâmica e limitação de banda; parte delas não divulgada. 6 idiomas (`README.txt`) | **NÃO ENCONTRADO** no arquivo (sem coluna de gerador). O repositório chama de "vozes comerciais de plataforma" (`RESULTADO_TESTE_INTERNET.md`), mas **a origem dessa afirmação não foi encontrada** | **CC BY 4.0** (`LICENSE.txt` local) | Ver o quadro abaixo |
| **ML-ITW** | Hugging Face, repositório `nccm2p2/ML-ITW`. Autoria e artigo **não verificados** (`dados_teste_internet.md`) | Vídeos de **redes sociais** (youtube, bilibili, x, facebook, rednote, tiktok, douyin). 14 idiomas, 180 locutores. Rótulo vindo do contexto da postagem (`HISTORICO_VERSOES.md`) | **NÃO ENCONTRADO** | **NÃO ENCONTRADO** ("não declarada") | 3 pessoas em comum com o In-the-Wild, por nome. Português: 8 locutores (indício). Na v7, foi dividido por locutor entre treino, validação e teste mestre |

**Regras do RADAR (o que está escrito e onde):**

- **README local** (`data/raw/teste_internet/radar_eval/README.txt`):
  - não há treino oficial;
  - LlamaPartialSpoof, LibriTTS dev/test e dados "derivados ou sobrepostos" ficaram fora do treino dos participantes, porque formam o dev;
  - os rótulos do eval foram publicados "para pesquisa pós-desafio e reprodutibilidade".
  - **O README não diz literalmente "não treinar com o eval".**
- **Onde aparece a regra "não treinar com o eval":** `docs/PROTOCOLO_V7.md` (seção 0 b) e `docs/RESULTADO_V7.md`, atribuída ao site do desafio. **O texto do site não está no repositório.**
- **O que a v7 fez:** treinou com o eval zh, zh-tw e ja. Por isso os números da v7 **não valem como resultado do desafio** (`PROTOCOLO_V7.md`).
- **O dev do RADAR não foi baixado** (`dados_teste_internet.md`).
- **Observação desta leitura (não conferida a fundo):** o ShiftySpeech do treino guarda os locutores LibriSpeech dev-other/test-other. Se o desafio contar isso como "derivado de LibriTTS dev/test", haveria mais um conflito com a política. Não muda a ressalva já escrita: os números não valem para o desafio.

---

## 2. CNN da v1 à v3

Fontes:
- `src/model.py` (CNN e log-mel);
- `src/augment.py`;
- `src/sampling.py`;
- `src/train.py` e `src/experimento.py` (perda, otimizador e early stopping);
- células de configuração e de justificativa de `notebooks/versions/v1_baseline.ipynb` (seções 2 e 6.3), `v2_augmentation.ipynb` e `v3_padding.ipynb`;
- `docs/HISTORICO_VERSOES.md`.

### 2.1 Entrada (igual da v1 à v3)

| item | valor | por quê (escrito no notebook v1, seção 2) |
|:--|:--|:--|
| taxa / canais | 16 kHz, mono (média dos canais) | a maioria dos bancos já está em 16 kHz; uma taxa única tira o atalho da taxa |
| recorte | 4 s: aleatório no treino, central na validação e no teste da v1 | a mediana de duração fica entre 3 e 5,5 s; entrada de tamanho fixo |
| áudio curto | **repetido** até 4 s (v1, v2, v3 `repeat`); v3 `zero` completa com zeros à direita | repetir não cria silêncio artificial. O zero foi descartado na v1 porque "a quantidade de silêncio contaria a duração" (`dataset_overview.md`) |
| log-mel | `n_mels` 64, `n_fft` 512, janela 400 (25 ms), passo 160 (10 ms); `log(mel + 1e-6)` | janela de 25 ms com passo de 10 ms é o padrão em fala; entrada pequena (64 × 401) |
| normalização | **por exemplo**: média 0 e desvio 1 no espectrograma inteiro (`src/model.py`: `LogMelPadronizado`) | tira o volume: um ganho vira uma constante, que a subtração da média remove |
| avaliação (v2 e v3) | janelas de 4 s com passo de 2 s, média dos logits; o recorte central também é reportado | a janela tem o tamanho do treino; com passo de meia janela, todo trecho cai inteiro em pelo menos uma janela |

### 2.2 Arquitetura (`CNNPequena`, igual da v1 à v3)

- **4 blocos**, cada um `Conv2d 3×3 (padding 1) → BatchNorm2d → ReLU → MaxPool 2×2`, com **16, 32, 64 e 128 canais**. A entrada tem 1 canal (64 × 401).
- Saída dos blocos: 128 × 4 × 25 → `Flatten` (12.800 números) → `Dropout 0,3` → `Linear(12.800 → 1)`. A saída é o logit, sem sigmoide.
- **Parâmetros: 110.433** (`HISTORICO_VERSOES.md`, v1). Conferido pela soma camada a camada (calculado): convoluções 160 + 4.640 + 18.496 + 73.856, BatchNorm 32 + 64 + 128 + 256, linear 12.801.
- **Por quê** (notebook v1, seção 6.3):
  - é o padrão "conv → pool" da Aula 10;
  - 4 blocos reduzem 64 × 401 para 4 × 25 e dão um campo receptivo grande;
  - tem ~110 mil parâmetros, "pequena de propósito, porque é o baseline";
  - o BatchNorm entrou a pedido do enunciado, para estabilizar o treino.
- A escolha específica do kernel 3×3 e de 16/32/64/128 canais não tem outra justificativa escrita além de "filtros crescendo" (**NÃO ENCONTRADO**).

### 2.3 Treino

| item | v1 | v2 (v1r, v2a, v2b) | v3 (6 execuções) |
|:--|:--|:--|:--|
| perda | `BCEWithLogitsLoss` | igual | igual |
| otimizador | Adam, lr 0,001, taxa fixa | igual | igual |
| lote | 64 | 64 | 64 |
| por época | 25.000 reais + 25.000 IA, sorteados de novo a cada época, com teto por fonte e por gerador (`src/sampling.py`) | igual | igual |
| épocas | até 30 | até 30 | até 30 |
| early stopping e checkpoint | paciência 5 pela **EER global da validação** (79% DFADD) | paciência 5 pela **EER média por fonte** da validação (`src/train.py`) | igual à v2 |
| dropout | 0,3 | 0,3 | 0,3 |
| seed | 42 | 42 | 42, 123, 2026 (seed dos dados fixa em 42) |
| workers | 8 | 4 (memória; não muda o método) | 4 |
| época escolhida (de quantas) | 16 (de 21) | 13 (18) / 20 (25) / 17 (22) | 9 (14), 7 (12), 12 (17), 19 (24), 9 (14), 11 (16) |
| limiar | EER da validação (prob. 0,039) | EER da validação (janelas): 0,603 / 0,098 / 0,756 | 0,64 / 0,79 / 0,63 (`repeat`) e 0,92 / 0,13 / 0,96 (`zero`), seeds 42 / 123 / 2026 |

Fontes: `HISTORICO_VERSOES.md` (v1, v2, v3 fechamento) e `results/experiments.csv` (coluna "época escolhida").

**Por quê de cada escolha** (notebook v1, seção 2):
- lote 64: faixa da Aula 6 (32 a 256);
- lr 0,001: valor recomendado na Aula 6 para o Adam;
- dropout 0,3: regularização moderada da Aula 7;
- paciência 5: early stopping da Aula 7;
- 25 mil por classe e época: ~1 min por época e cobertura do pool em 15 a 30 épocas.

**Por que a checagem do checkpoint mudou na v2:** a EER global da validação "esconde o desempenho fora do DFADD" (`HISTORICO_VERSOES.md`, v1, conclusão).

### 2.4 Augmentation da v2 (`src/augment.py`)

Só no treino, com **a mesma chance para real e IA**. Aplicada nesta ordem:

| operação | parâmetro | chance | experimento | por quê (notebook v2, seção 2) |
|:--|:--|:--|:--|:--|
| normalização RMS | alvo −26 dB | 0,5 | v2a, v2b | nível típico de fala (mediana dos bancos entre −19 e −27 dB) |
| ganho aleatório | −12 a +6 dB (uniforme) | 0,5 | v2a, v2b | acima de +6 dB o pico estouraria em muitos áudios |
| ruído | branco, rosa ou marrom (expoente 0/1/2, sorteado); SNR de 5 a 30 dB | 0,3 | v2b | de "ruído forte, ainda inteligível" a "quase limpo" |
| banda de telefone | reamostra para 8 kHz e volta (corta acima de 4 kHz) | 0,3 | v2b | taxa de telefone |
| mu-law | 8 bits (256 níveis) | 0,3 | v2b | codec da telefonia digital (G.711) |

- Em todas as operações, o pico é limitado a 0,99 por redução de volume (não há corte da onda).
- Com chance 0,3 em cada uma das três degradações, ~34% dos exemplos continuam limpos (0,7³).
- MP3/Opus ficou de fora por falta de `ffmpeg`.
- **Hipótese da v2:** a v1 aprendeu "gravação limpa = real". **Resultado:** não confirmada (`HISTORICO_VERSOES.md`).

### 2.5 Mudança da v3

- **O que mudou:** o áudio com menos de 4 s passou a ser completado **com zeros à direita** (`zero`) em vez de repetido (`repeat`).
- **Como foi testado:** 3 seeds por condição. O recorte aleatório depende só de (seed, época, áudio), então as duas condições veem os mesmos trechos.
- **Por quê** (pré-registro da v3): no dev_shift, a AUC era pior nos áudios curtos (0,24 abaixo de 2 s × 0,43 acima de 4 s no v1r), e a repetição podia criar padrões artificiais.
- **Hipótese alternativa registrada antes:** os zeros podem virar uma pista de duração.
- **Resultado:** diferença dentro do acaso entre seeds.

---

## 3. Resultados da v1 à v3 em todas as amostras medidas

### 3.1 v1 (`results/baseline_results.csv`; teste com recorte central de 4 s)

Amostra do teste: até 5.000 por classe e banco. **No In-the-Wild, a amostra foi tirada do banco inteiro** (31.779 áudios, antes de existir o dev_shift).

| conjunto | n real / IA | AUC | EER | recall IA | falso alarme | acc. bal. (calculado) |
|:--|:--|--:|--:|--:|--:|--:|
| validação (todas as fontes) | 15.730 / 35.459 | 0,998 | 1,3% | 98,7% | 1,3% | 0,987 |
| In-the-Wild (inteiro) | 5.000 / 5.000 | 0,211 | 72,2% | 76,6% | 91,2% | 0,427 |
| LlamaPartialSpoof | 5.000 / 5.000 | 0,563 | 45,4% | 82,9% | 72,7% | 0,551 |
| BRSpeech-DF | 1.320 / 5.000 | 0,766 | 30,6% | 47,7% | 7,4% | 0,7015 |
| ASVspoof 2019 LA eval | 5.000 / 5.000 | 0,971 | 6,9% | 66,6% | 0,1% | 0,833 |
| ASVspoof 2021 DF | 5.000 / 5.000 | 0,685 | 35,3% | 59,2% | 33,9% | 0,627 |
| CORAA (só real) | 5.000 / 0 | — | — | — | 41,6% | — |

- Validação por fonte (EER; `HISTORICO_VERSOES.md`, v1): DFADD 0,26%, ShiftySpeech 7,8%, CodecFake+ 32,7%; média por fonte 13,6%.
- AUC por fonte da validação na v1: **NÃO ENCONTRADO** (o CSV só tem a validação inteira).
- **dev_shift:** não existia na v1.

### 3.2 v2 (`results/experiments.csv`; janelas; teste = test_final do In-the-Wild e os outros bancos, uma vez por experimento)

Cada célula: AUC · acc. bal. no limiar da validação (calculado de recall e falso alarme gravados).

| conjunto | n real / IA | v1r | v2a | v2b |
|:--|:--|:--|:--|:--|
| validação (todas) | 15.730 / 35.459 | 0,999 · 0,990 | 0,999 · 0,991 | 0,998 · 0,985 |
| val CodecFake+ (AUC · EER) | 164 / 188 | 0,731 · 30,1% | 0,720 · 31,0% | 0,757 · 28,7% |
| val DFADD (AUC · EER) | 8.710 / 31.800 | 1,000 · 0,0% | 1,000 · 0,0% | 1,000 · 0,2% |
| val ShiftySpeech (AUC · EER) | 1.157 / 3.471 | 0,985 · 6,4% | 0,988 · 5,8% | 0,951 · 10,9% |
| dev_shift (In-the-Wild) | 8.633 / 1.981 | 0,325 · 0,423 | 0,386 · 0,450 | 0,389 · 0,419 |
| In-the-Wild test_final | 5.000 / 5.000 | 0,262 · 0,420 | 0,354 · 0,419 | 0,336 · 0,360 |
| LlamaPartialSpoof | 5.000 / 5.000 | 0,625 · 0,598 | 0,595 · 0,566 | 0,646 · 0,611 |
| BRSpeech-DF | 1.320 / 5.000 | 0,776 · 0,717 | 0,801 · 0,746 | 0,779 · 0,709 |
| ASVspoof 2019 LA eval | 5.000 / 5.000 | 0,936 · 0,849 | 0,936 · 0,787 | 0,986 · 0,893 |
| ASVspoof 2021 DF | 5.000 / 5.000 | 0,716 · 0,662 | 0,701 · 0,642 | 0,750 · 0,725 |
| CORAA: falso alarme | 5.000 / 0 | 34,9% | 24,3% | 29,8% |

- EER global da validação (janelas): 1,0% / 0,9% / 1,5%.
- EER média por fonte (checkpoint): 12,6% / 13,0% / 14,3% (`HISTORICO_VERSOES.md`).
- O `experiments.csv` também tem a avaliação central e o LlamaPartialSpoof separado em "só full" e "só partial".
- **Teste de estresse na validação** (AUC; `HISTORICO_VERSOES.md`, v2):

  | condição | v1r | v2a | v2b |
  |:--|--:|--:|--:|
  | limpo | 0,965 | 0,965 | 0,961 |
  | ruído branco 10 dB | 0,487 | 0,499 | 0,776 |
  | telefone | 0,794 | 0,838 | 0,939 |
  | mu-law | 0,887 | 0,890 | 0,949 |

### 3.3 v3 (`results/experiments.csv`, `results/v3_acuracias.csv`; janelas; validação inteira e dev_shift inteiro)

| execução | val AUC · EER global | val CodecFake+ AUC | val DFADD AUC | val ShiftySpeech AUC | EER média por fonte (checkpoint) | dev_shift AUC · EER | dev_shift acc. bal. (limiar val) |
|:--|:--|--:|--:|--:|--:|:--|--:|
| `v3_repeat_s42` | 0,998 · 1,2% | 0,704 | 1,000 | 0,981 | 14,1% | 0,241 · 69,6% | 0,434 |
| `v3_zero_s42` | 0,996 · 2,2% | 0,751 | 1,000 | 0,954 | 13,7% | 0,274 · 66,5% | 0,400 |
| `v3_repeat_s123` | 0,998 · 1,7% | 0,733 | 1,000 | 0,976 | 13,2% | 0,378 · 59,2% | 0,503 |
| `v3_zero_s123` | 0,999 · 1,0% | 0,730 | 1,000 | 0,987 | 11,2% | 0,146 · 79,4% | 0,365 |
| `v3_repeat_s2026` | 0,998 · 1,2% | 0,730 | 1,000 | 0,976 | 12,6% | 0,228 · 70,5% | 0,432 |
| `v3_zero_s2026` | 0,998 · 1,1% | 0,731 | 1,000 | 0,980 | 12,3% | 0,377 · 57,7% | 0,444 |

Acurácia trivial do dev_shift: 0,813 (`v3_acuracias.csv`).

**A v3 foi avaliada no teste final? Não, como versão.**
- `AVALIAR_TESTE_FINAL = False` no notebook.
- O pré-registro do teste da v4 diz que "o teste da v3 nunca foi avaliado" (`HISTORICO_VERSOES.md`).
- No teste da v3, LlamaPartialSpoof, BRSpeech-DF, CORAA e ASVspoof estão como "não avaliado" na tabela-resumo.
- **Exceção:** um checkpoint da v3, o `v3_zero_s123`, entrou como **S0** no teste da v4.
  - Foi escolhido pela menor EER média por fonte da validação (11,25%; `results/v4_teste_escolha_s0.json`).
  - Também é o de pior AUC no dev_shift (0,146).
  - Resultados do S0 no teste da v4 (`results/v4_teste_final.csv`, limiar A; mesma amostra de 51.320 clipes):

    | banco | AUC | acc. bal. |
    |:--|--:|--:|
    | In-the-Wild test_final | 0,114 | 0,336 |
    | LlamaPartialSpoof | 0,556 | 0,545 |
    | BRSpeech-DF | 0,735 | 0,704 |
    | ASVspoof 2019 | 0,967 | 0,901 |
    | ASVspoof 2021 | 0,688 | 0,665 |

    Falso alarme no CORAA: 65,3% no limiar A e 41,3% no limiar B.
  - As outras cinco execuções da v3 nunca foram ao teste.

---

## 4. XLS-R (v4 em diante)

| item | valor | fonte |
|:--|:--|:--|
| checkpoint | `facebook/wav2vec2-xls-r-300m`, revisão `1a640f32ac3e39899438a2931f9924c02f080a54`, `pytorch_model.bin` de 1.269.737.156 bytes (sem safetensors) | `HISTORICO_VERSOES.md`, v4, estado da execução |
| pré-treino | sem rótulos, ~436 mil h de fala em 128 línguas | `src/ssl.py` (docstring) |
| camadas | 24 camadas do transformer → **25 saídas** (índice 0 = entrada do transformer; 1 a 24 = saída de cada camada), cada uma com **1.024 números a cada 20 ms** | `src/ssl.py` (`N_SAIDAS, DIM = 25, 1024`, conferido por `assert`) |
| precisão | fp16; bf16 testado e não usado (erro ~10 vezes maior) | `HISTORICO_VERSOES.md`, v4 decisão 1 e v5 execução |
| leitura | mono, float32, reamostrado para 16 kHz com o filtro do torchaudio sempre que a taxa for outra | `src/preproc.py`: `ler_mono_16k` |
| corte de silêncio | **confirmado: 35 dB.** Quadros de 400 amostras com passo 160. Mantém do 1º ao último quadro com energia acima de (quadro mais forte − 35 dB). Se sobrar menos de 0,5 s, desfaz o corte | `src/preproc.py`: `bordas_sem_silencio`, `preparar_clipe` |
| recorte | **confirmado: recorte central de no máximo 8 s, sem completar** (nem repetição nem zeros). Só abaixo de 400 amostras (25 ms) completa com zeros, porque o modelo não tem quadro antes disso | `src/preproc.py`: `preparar_clipe` |
| normalização da onda | cada clipe com média 0 e variância 1 antes do lote (o `do_normalize` do extrator); `attention_mask` para o enchimento | `src/ssl.py`: `normalizar`, `representar` |
| camada → vetor | **média no tempo** só dos quadros válidos. O desvio também é guardado no cache, mas a regressão usa só a média (a CNN da v6 usou média e desvio) | `src/ssl.py`: `representar`; `PROTOCOLO_V7.md` §1; `DECISOES_V6.md` E4 |
| concatenação | médias das saídas **5, 6 e 7**, nessa ordem → **3.072 números** | `results/v7_t3_modelo.json` (`n_numeros: 3072`) |
| padronização | (x − média) / desvio, com **média e desvio do conjunto de treino** e sem pesos. v4: os 40.000 clipes de treino. T3: os 81.288 do treino da T3; desvio 0 vira 1 | `HISTORICO_VERSOES.md` (S1/S2 da v4); `PROTOCOLO_V7.md` §1; `v7_t3_modelo.json` (`padronizacao`) |
| nota | **nota = ((x − mu) / sd) · coef + intercepto**, em logit (maior = mais IA). A probabilidade é a sigmoide da nota, sem calibração. Pesos: `coef` (3.072), `intercepto`, `mu` (3.072), `sd` (3.072) | `results/v7_t3_modelo.json` (`como_usar`); `results/v7_t3_coeficientes.npz` |
| lote na GPU | baldes de duração fixa (2, 4, 6 e 8 s). Diferença máxima de 0,42% (fp16) entre lote e clipe isolado. É execução, não método | `HISTORICO_VERSOES.md`, v4 decisão 2 |

---

## 5. Regressão logística

| item | v4 (F0) | v6 (funil) | v7 (T3) |
|:--|:--|:--|:--|
| biblioteca | scikit-learn `LogisticRegression` | igual | igual |
| solver | padrão (lbfgs) | lbfgs | `lbfgs` (explícito) |
| penalidade | padrão (L2) | L2 | L2 |
| grade de C | 0,001 / 0,01 / 0,1 / 1 | E1: 1e-3 a 1; E4: C = 1 fixo ("básica" = padrão do scikit-learn); E5: `logspace(-4, 1, 8)` | 1e-4 / 1e-3 / 1e-2 / 0,1 / 1 / 10 |
| class_weight | não usado (treino já 50/50: 20.000 + 20.000) | não usado; E5 testou `sample_weight` por fonte | **`balanced`** (T0 a T3); T4 usa `sample_weight` por bloco |
| max_iter / tol | 1.000 / 1e-4 | 1.000 / 1e-4 | 1.000 / 1e-4 (142 iterações no C escolhido) |
| como o C foi escolhido | **menor EER média por fonte da validação**, por camada e por cabeça; `concat_top3` ficou com **C = 0,1** | **AUC da pior fonte** (validação + dev_shift) | **maior média da AUC por fonte na validação** (6 fontes); T3: **C = 0,01** |

Fontes: `src/cabecas.py` (`ajustar_rl`), `scripts/v4.py`, `scripts/v4final_figuras.py` (`RL_MAX_ITER = 1000`), `HISTORICO_VERSOES.md` (v4 decisão 6, v6 pré-registro), `scripts/v7.py` (linhas 51 e 491), `results/v7_t3_modelo.json`, `docs/PROTOCOLO_V7.md` §1 e §3.

**Motivos escritos para usar a regressão logística:**

1. **Transfer learning com cabeça leve** (pré-registro da v4): o modelo congelado é o extrator, como na Aula 10.
2. **É determinística e convexa**, então não depende de seed (`HISTORICO_VERSOES.md`, v4 critério c; `DECISOES_V6.md` E4).
3. **No empate, vence o mais simples**, "mais fácil de explicar e tende a generalizar melhor" (`DECISOES_V6.md`, Geral).
4. **MLP e CNN não superaram a RL** no funil da v6 (E4: nenhuma seed com diferença relevante; `results/v6_funil.csv`).
   - Também está escrito que não havia "motivo forte para esperar ganho sobre uma RL que já passa de 0,98 no dev_shift", e que outras cabeças aumentariam o risco de overfitting (`DECISOES_V6.md`).
5. **Na decisão do sistema principal** (`DECISAO_SISTEMA_PRINCIPAL.md`, motivos 4 e 5):
   - são 3.073 números no git;
   - treina em segundos sobre vetores já extraídos;
   - o ajuste fino precisou de horas.

**Achado registrado como hipótese, sem mudar nada** (`HISTORICO_VERSOES.md`, "Notebook final", 11/11): o dev_shift melhora com mais regularização (C = 0,0001: AUC 0,994 × 0,983), enquanto a validação prefere C = 0,1.

---

## 6. v1 × v4 (F0) por banco de teste

Fontes:
- **v1:** `results/baseline_results.csv`. Recorte central de 4 s; In-the-Wild amostrado do banco inteiro; limiar da validação (prob. 0,039). A acc. bal. é calculada.
- **F0 (S2 da v4):** `results/v4_teste_final.csv`. Áudio com corte de silêncio e até 8 s; In-the-Wild só test_final; limiar A (EER da validação, logit −0,04).

| banco | n real / IA | v1 AUC | F0 AUC (IC95) | v1 acc. bal. | F0 acc. bal. (IC95) |
|:--|:--|--:|:--|--:|:--|
| In-the-Wild | 5.000 / 5.000 | 0,211 | 0,979 (0,976–0,981) | 0,427 | 0,835 (0,829–0,842) |
| LlamaPartialSpoof | 5.000 / 5.000 | 0,563 | 0,925 (0,920–0,930) | 0,551 | 0,855 (0,849–0,862) |
| BRSpeech-DF | 1.320 / 5.000 | 0,766 | 0,842 (0,827–0,854) | 0,7015 | 0,679 (0,667–0,693) |
| ASVspoof 2019 LA eval | 5.000 / 5.000 | 0,971 | 0,999 (0,999–0,999) | 0,833 | 0,981 (0,979–0,984) |
| ASVspoof 2021 DF | 5.000 / 5.000 | 0,685 | 0,997 (0,996–0,997) | 0,627 | 0,942 (0,938–0,947) |
| CORAA: falso alarme | 5.000 / 0 | 41,6% | 16,7% (limiar A); 2,5% (limiar B) | — | — |

Cuidados ao comparar:

- **Amostra do In-the-Wild:** não é a mesma. A v1 sorteou do banco inteiro; o F0, só do test_final.
- **Outros bancos:** a regra de amostragem é a mesma (até 5.000 por classe e banco, seed 42; `HISTORICO_VERSOES.md`, v4 pré-registro do teste). Se os clipes são idênticos um a um: **NÃO VERIFICADO**.
- **BRSpeech-DF:** o F0 tem AUC maior, mas acc. bal. menor no limiar A, porque acusa muitos reais.
- **Intervalos:** os ICs medem só a amostra de teste, não o treino. IC da v1: **NÃO ENCONTRADO** (o CSV não tem).

---

## 7. WavLM Large (v5)

| item | valor | fonte |
|:--|:--|:--|
| checkpoint | `microsoft/wavlm-large`, revisão `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, `pytorch_model.bin` de 1.261.990.257 bytes | `HISTORICO_VERSOES.md`, v5 pré-registro |
| pré-treino | ~94 mil h de inglês; também treinado a "limpar" fala com ruído e sobreposição | `HISTORICO_VERSOES.md`, v5; `src/ssl.py` |
| camadas | 25 saídas de 1.024 a cada 20 ms (como o XLS-R) | `src/ssl.py` |
| como foi usado | **uma única mudança**: o modelo congelado. Mesmos clipes (40.000 / 19.999 / 10.614), corte de 35 dB, até 8 s, fp16, média no tempo, RL por camada com C pela validação, `concat_top3/4` e somas ponderadas | `HISTORICO_VERSOES.md`, v5 pré-registro |
| cabeça final | **`concat_top3` com camadas 4, 6 e 8, C = 0,1** (confirmado). A melhor camada sozinha foi a 4. W1 (só camada 24): C = 1 | `HISTORICO_VERSOES.md`, v5 resultado |
| decisão (validação + dev_shift) | **empate pela regra → segue o XLS-R.** EER média por fonte 6,77% × 5,92%; dev_shift AUC 0,970 × 0,983; diferença −0,013 (IC95 por locutor −0,061 a +0,001) | `HISTORICO_VERSOES.md`; `results/v5_comparacao.json` |

**Teste (uma vez, 51.320 clipes; `results/v5_teste_final.csv`, citado em `HISTORICO_VERSOES.md`).** Cada célula: AUC · acc. bal. no limiar A / no limiar B.

| banco | XLS-R final (X2) | WavLM final (W2) | XLS-R camada 24 (X1) | WavLM camada 24 (W1) |
|:--|:--|:--|:--|:--|
| In-the-Wild | 0,978 · 0,834/0,930 | 0,950 · 0,839/0,848 | 0,988 · 0,829/0,949 | 0,946 · 0,826/0,895 |
| LlamaPartialSpoof | 0,924 · 0,854/0,735 | 0,914 · 0,856/0,708 | 0,934 · 0,855/0,736 | 0,899 · 0,830/0,792 |
| BRSpeech-DF | 0,842 · 0,680/0,769 | 0,768 · 0,643/0,683 | 0,776 · 0,678/0,712 | 0,557 · 0,524/0,557 |
| CORAA: falso alarme A / B | 16,9% / 2,5% | 34,5% / 8,5% | 36,1% / 11,5% | 87,3% / 73,6% |
| ASVspoof 2019 | 0,999 · 0,981/0,898 | 0,996 · 0,970/0,817 | 0,997 · 0,977/0,906 | 0,986 · 0,944/0,911 |
| ASVspoof 2021 DF | 0,997 · 0,942/0,964 | 0,990 · 0,904/0,937 | 0,996 · 0,952/0,968 | 0,986 · 0,895/0,945 |

Leitura pré-definida (W2 − X2):
- **AUC:** "relevante: WavLM pior" nos 3 principais (−0,029 / −0,010 / −0,075, IC95 sem o zero).
- **Acc. bal. (limiar A):** "não demonstrada" (`HISTORICO_VERSOES.md`, v5 fechamento).

---

## 8. Funil da v6 (E1 a E6, F0 a F2)

Fontes: `HISTORICO_VERSOES.md` (v6 pré-registro e fechamento), `docs/DECISOES_V6.md`, `results/v6_funil.csv`, `results/v6_e4_cabecas.csv`, `results/v6_e5_passos.csv`, `src/v6_cabecas.py`.

**Métrica e regra (fixadas antes):**
- Métrica primária: **AUC da pior fonte** entre DFADD, ShiftySpeech, CodecFake+ (validação) e dev_shift.
- Bootstrap pareado: 2.000 reamostragens, por locutor; no CodecFake+, por clipe.
- **Regra do desafiante:** a opção mais complexa só toma o lugar se o IC95 da diferença excluir o zero **e** a diferença for ≥ 0,005. Senão, fica a mais simples.

| etapa | o que testou | candidatos (AUC da pior fonte) | vencedor |
|:--|:--|:--|:--|
| E1 | Meta (XLS-R) × Microsoft (WavLM), mesmo pipeline, C por camada pela pior fonte | Meta `concat_top4` (5, 7, 8, 11; C = 1) 0,907; Microsoft `concat_top3` (4, 6, 7; C = 1) 0,891; v4 0,903 | empate (Microsoft − Meta −0,017; IC95 −0,048 a +0,001) |
| E2 | fusão: média das notas → stacking (RL de 2 entradas, cross-fitting por fonte) → vetores colados | 0,905 / 0,905 / 0,874 | média das notas |
| E3 | front-end: só Meta → só Microsoft → fusão | 0,907 / 0,891 / 0,905 | **só Meta** |
| E4 | cabeça: RL → MLP → CNN (3 seeds cada) | RL 0,904; MLP 0,912 / 0,906 / 0,902; CNN 0,906 / 0,908 / 0,910 | **RL** (nenhuma seed com diferença relevante) |
| E5 | ajuste fino, uma modificação por vez | zona +0,000; C ampla +0,001; pesos por fonte **+0,006 (mantido)**; limiar A/C/D | pesos por fonte; **limiar A** (D não ganhou 0,005: 0,917 × 0,916) |
| E6 | teste, uma vez, dos finalistas | — | — |

**Cabeças da E4** (entrada: a melhor camada única, a 11; `results/v6_e4_cabecas.csv`):

| cabeça | estrutura | parâmetros | treino | AUC pior fonte (seeds 42 / 123 / 2026) | dev_shift AUC |
|:--|:--|--:|:--|:--|:--|
| RL básica | camada 11 padronizada, C = 1 | 1.025 | lbfgs | 0,904 | 0,972 |
| MLP | 1.024 → 256 (ReLU, dropout 0,3) → 1 | 262.657 | AdamW (lr 1e-3, weight decay 1e-3), lote 256, até 30 épocas, early stopping pela EER média por fonte da validação (paciência 5) | 0,912 / 0,906 / 0,902 | 0,985 / 0,989 / 0,981 |
| CNN ao longo das camadas | entrada 25 × 2.048 (média e desvio); Conv1d(2048→64, k=3) + BatchNorm + ReLU; Conv1d(64→64, k=3) + ReLU; média nas posições; Linear(64→1) | 405.825 | igual ao MLP | 0,906 / 0,908 / 0,910 | 0,979 / 0,983 / 0,979 |

A CNN recebeu média e desvio, enquanto a RL e o MLP receberam só a média. Isso foi registrado antes, e a ablação não foi feita (`DECISOES_V6.md`).

**E5, passo a passo** (`results/v6_e5_passos.csv`; regra de manter: +0,003 na pior fonte e o dev_shift sem cair mais que 0,01):

| passo | pior fonte antes → depois | mantido? |
|:--|:--|:--|
| 1. zona de camadas (fica a camada 11) | 0,9037 → 0,9037 | não |
| 2. grade ampla de C (C = 10) | 0,9037 → 0,9047 | não |
| 3. pesos iguais por fonte | 0,9037 → 0,9100 | **sim** |
| 4. regra de limiar | acc. bal. com uma fonte de fora: A 0,9156 · C 0,9155 · D 0,9171 | fica A |

**Finalistas e teste** (`results/v6_teste_final.csv`; AUC · acurácia · acc. bal., limiar A):

- **F0** = v4.
- **F1** = só Meta + RL básica (camada 11, C = 1).
- **F2** = F1 + pesos por fonte + limiar A.

| banco | F0 | F1 | F2 |
|:--|:--|:--|:--|
| In-the-Wild | 0,979 · 0,836 · 0,835 | 0,972 · 0,807 · 0,807 | 0,972 · 0,812 · 0,812 |
| LlamaPartialSpoof | 0,925 · 0,855 · 0,855 | 0,940 · 0,863 · 0,863 | 0,937 · 0,863 · 0,863 |
| BRSpeech-DF | 0,842 · 0,863 · 0,679 | 0,741 · 0,814 · 0,576 | 0,744 · 0,813 · 0,571 |
| CORAA: falso alarme A / B | 16,7% / 2,5% | 41,2% / 17,4% | 40,4% / 17,3% |
| ASVspoof 2019 | 0,999 · 0,981 · 0,981 | 0,997 · 0,975 · 0,975 | 0,997 · 0,975 · 0,975 |
| ASVspoof 2021 DF | 0,997 · 0,942 · 0,942 | 0,996 · 0,954 · 0,954 | 0,996 · 0,950 · 0,950 |

- F1 − F0 e F2 − F0: "não demonstradas" (sinais trocados).
- **Por quê, como está escrito:** a regra "melhor camada única pela pior fonte, C = 1" levou à camada 11 e tirou a regularização. Isso ajudou o CodecFake+ (352 clipes) e prejudicou o português.

---

## 9. Contexto de golpe: parâmetros das chamadas e regra de combinação

Fonte: `src/contexto_golpe.py`.

**Transcrição (Whisper)** (linhas 33–45 e 261–266):
- Modelo `whisper-large-v3-turbo`, pela API do Groq (`/audio/transcriptions`).
- `response_format = "verbose_json"`, **`temperature = "0"`**.
- `language` vai só quando o idioma não é `"auto"`. **O padrão da função `transcrever` e de `analisar_audio` é `idioma="pt"`** (linhas 261 e 450).
- Antes de enviar:
  - corte de silêncio com `TOP_DB = 35`;
  - no máximo 120 s (`MAX_SEG_PADRAO`);
  - com menos de 1 s de fala (`MIN_FALA_S`), nada é enviado.
- Filtro de segmentos depois da resposta:
  - descarta segmento com `no_speech_prob > 0,6` **e** `avg_logprob < −1,0`;
  - marca a transcrição como suspeita se tiver menos de 10 caracteres ou trouxer frases típicas de alucinação;
  - com menos de 4 palavras, o LLM não é chamado.

**Classificação (LLM)** (linhas 36–55 e 372–378):
- Modelo `openai/gpt-oss-120b`, pela API do Groq (`/chat/completions`).
- **`temperature = 0`**, **`seed = 42`**, `max_tokens = 1200`, `response_format = {"type": "json_object"}`, **`reasoning_effort = "low"`**.
- Prompt de sistema congelado = `docs/avaliacao_contexto/prompt_r1.txt`.
- Texto truncado em 6.000 caracteres.
- Se a resposta for inválida, tenta mais uma vez.
- Rede: timeout de 30 s; até 3 novas tentativas em 429/5xx, com espera de no máximo 30 s.

**Regra de combinação exata no código:**
- **Em `src/contexto_golpe.py`: NÃO ENCONTRADO** (o módulo não combina com a voz).
- A regra que rodou está em **`scripts/teste_whatsapp.py`, linhas 420–422**:
  - `voz_ia = (z_T3_A >= corte M da T3)`;
  - `alerta_medio = risco em {medio, alto}`;
  - nível = **alto** se os dois, **atenção** se só um, **baixo** se nenhum.
- A mesma tabela está proposta em `docs/CONTEXTO_GOLPE.md`, seção 6, onde a regra do sistema final segue **em aberto**.

---

## 10. Conferência no banco WhatsApp (100 clipes, Whisper `auto`)

Fontes:
- `results/diagnostico_whisper_combinado.csv`, tabela `contexto_por_grupo`, conjunto `todos` (DEV 40 + RESERVADO 60), coluna `alerta_medio_mais` = versão corrigida;
- `results/diagnostico_whisper_idioma_metricas.csv`;
- `docs/RESULTADO_CONTEXTO_WHISPER.md` (confirma que "corrigido" = modo `auto`).

| grupo | n | golpe? | alertou (médio ou mais) | certo |
|:--|--:|:--|--:|--:|
| IA pedindo dinheiro | 46 | sim | 26 | 26 |
| real com golpe | 27 | sim | 1 | 1 |
| real sem golpe | 17 | não | 4 (falso alarme) | 13 |
| IA neutra | 10 | não | 0 | 10 |
| **total** | **100** | **73 com golpe / 27 sem** | **31** | **50** |

**Suas hipóteses:**

| hipótese | resultado |
|:--|:--|
| rotulados "golpe": 73? | **73: CONFERE** (46 + 27). Bate com `n_pos` 24 (DEV) + 49 (RESERVADO) |
| 27 golpes alertados (26 clones + 1 real)? | **CONFERE** (recall 37,0%, calculado) |
| sem golpe: 27? | **27: CONFERE** (17 reais + 10 IA neutras) |
| 23 de 27 acertados? | **CONFERE** (falso alarme 4/27 = 14,8%, calculado). Os 4 falsos alarmes são todos de reais sem golpe |
| o contexto alertou 27 clipes no total? | **DIVERGE.** O total de alertas é **31**: 27 golpes + 4 falsos alarmes |

- **Na versão de 15/11** (`pt` forçado; coluna `alerta_medio_mais_1511`): 27 IA + 2 reais com golpe = 29 golpes alertados, e 3 falsos alarmes.
- **Só no RESERVADO** (60 clipes, a avaliação principal; `diagnostico_whisper_idioma_metricas.csv`): recall 36,7% (18 de 49), falso alarme 18,2% (2 de 11), acurácia 45,0% contra 81,7% da trivial.
- **Atenção:** o conjunto `todos` inclui o DEV, que serviu para escolher o modo `auto`.

---

## Resumo do que ficou NÃO ENCONTRADO

1. Instituição que publicou DFADD, ShiftySpeech, In-the-Wild e CORAA, e autoria e artigo do ML-ITW (só há o URL ou o repositório).
2. Licença de DFADD, ShiftySpeech, CodecFake+, In-the-Wild e ML-ITW.
3. Geradores do In-the-Wild, do RADAR eval e do ML-ITW. A origem da descrição "vozes comerciais de plataforma" do RADAR também não foi encontrada.
4. O texto literal da regra "não treinar com o eval" do RADAR: está só citada (atribuída ao site) em `PROTOCOLO_V7.md` e `RESULTADO_V7.md`, e não aparece no README local.
5. O card do BRSpeech-DF dentro do repositório: "reais do LibriVox (audiolivro)" vem só do `audit_report.md`.
6. Tipo de fala de cada subconjunto do CORAA.
7. Justificativa específica do kernel 3×3 e dos canais 16/32/64/128 da CNN (só há "padrão da Aula 10, filtros crescendo").
8. AUC por fonte da validação da v1, e intervalo de confiança dos resultados de teste da v1.
9. Identidade clipe a clipe entre as amostras de teste da v1/v2 e da v4 (mesma regra de sorteio, não verificado).
10. Regra de combinação voz × conteúdo dentro de `src/contexto_golpe.py`: só existe em `scripts/teste_whatsapp.py`.
