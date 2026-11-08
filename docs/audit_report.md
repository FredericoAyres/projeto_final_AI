# Relatório de auditoria dos dados e do manifest

Gerado em 2026-11-07 e atualizado em 2026-11-08 com as decisões da seção 5. Scripts:
- `scripts/extract_brspeech.py`: extrai os áudios do BRSpeech-DF.
- `scripts/build_manifest.py`: gera o manifest e chama o script de extração.
- `scripts/audit_checks.py`: gera as tabelas das seções 3 e 4 (`python scripts/audit_checks.py`).
- `scripts/check_data.py`: confere se os áudios do manifest existem em disco.

Nada em `data/raw/` foi modificado, movido ou apagado.

Dependências (versões em `requirements.txt`): `pandas`, `pyarrow`, `soundfile`, `numpy`, `tabulate`, `pyyaml`, `tqdm`, `requests` e `huggingface_hub`.

> ⚠ **Leia antes de usar os testes:**
> 1. **BRSpeech-DF tem taxa de amostragem diferente por classe.** Os 1.320 reais estão em **48 kHz** e os 6.587 fakes em **24 kHz**, embora o card diga que tudo foi reamostrado para 24 kHz. Um modelo pode separar as classes só pela taxa ou pela banda. O manifest **não** foi alterado: reamostre tudo para a mesma taxa no carregamento e desconfie de resultados bons demais nesse banco.
> 2. **asvspoof2019_la (eval) e asvspoof2021_df são testes SECUNDÁRIOS.** Eles podem ter locutores em comum com o treino via DFADD (VCTK compartilhado, sem mapeamento de ids). Servem para medir generalização a **geradores e ataques novos**, mas **não a locutores novos**. Para locutor novo, use in_the_wild, llamapartialspoof e os testes pt.

## 1. Bancos auditados

As pastas vazias (só com `.gitkeep`) foram ignoradas: `vctk`, `sonar`, `spoofceleb`, `parlaspoof_br`, `mlaad_pt`, `fakebraccent`.
Foram auditadas 10 pastas: `asvspoof2019_la`, `dfadd`, `shiftyspeech`, `codecfake_plus`, `librispeech`, `in_the_wild`, `llamapartialspoof`, `asvspoof2021_df`, `brspeech_df`, `coraa`.

## 2. Estrutura, rótulos, locutor e gerador de cada banco

### asvspoof2019_la (`data/raw/train_val_en/asvspoof2019_la/LA/`)
- Estrutura: `ASVspoof2019_LA_{train,dev,eval}/flac/*.flac`, `ASVspoof2019_LA_cm_protocols/`, `ASVspoof2019_LA_asv_protocols/`, `ASVspoof2019_LA_asv_scores/`, `README.LA.txt`.
- Áudio: 122.299 `.flac` (train 25.380, dev 24.986, eval 71.933).
- Rótulos: `ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt`, separados por espaço: `SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY` (README.LA.txt). `KEY` ∈ {bonafide, spoof}.
- Locutor: coluna 1 (`LA_0001`…`LA_0108`, ids **anonimizados**; train 20, dev 20 e eval 67, sem sobreposição entre as partições). Gerador: coluna 4 (`A01`–`A19`, `-` para bonafide).
- 838 arquivos (`LA_D_A*`, `LA_E_A*`) não aparecem nos protocolos CM. São os áudios bonafide de cadastro do ASV, listados em `ASVspoof2019_LA_asv_protocols/*.trn.txt` (`locutor lista,de,arquivos`). Entram no manifest como label 0, `role=excluded`, `original_split=dev_asv_enroll|eval_asv_enroll`.

### dfadd (`data/raw/train_val_en/dfadd/`)
- Estrutura: `DATASET_<TTS>/DATASET_<TTS>/{train,valid,test}/*.flac` e `{train,valid,test}.txt` para os 5 TTS (GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2). Os reais estão em `DATASET_VCTK_BONAFIDE/home/isjwdu/Work/Dataset/DFADD/DATASET_VCTK_BONAFIDE/{train,valid,test}/*.wav` (o zip trouxe esse caminho absoluto do autor).
- Áudio: 163.500 `.flac` (fake) e 44.455 `.wav` (real, ou seja, o VCTK inteiro).
- Rótulos: `{train,valid,test}.txt` no formato ASVspoof `SPK UTT - - KEY`. As linhas bonafide se repetem nos 5 subconjuntos e apontam para `DATASET_VCTK_BONAFIDE/<split>/<UTT>.wav`. As linhas spoof apontam para `DATASET_<TTS>/<split>/<UTT>.flac`. Todos os 207.955 arquivos têm rótulo.
- Locutor: coluna 1 (ids VCTK `p225`…`p376` e `s5`; 110 locutores). O split original do DFADD tem 106 locutores em train, 2 em valid e 2 em test. Gerador: nome da pasta `DATASET_<TTS>`.

### shiftyspeech (`data/raw/train_val_en/shiftyspeech/`)
- Estrutura: `{real_data,hifigan,vocos,wavegrad}_audiobook_flac/audiobook_flac/*.flac` (11.126 arquivos por pasta, 44.504 no total).
- Rótulos: **não há arquivo de rótulo**. O rótulo vem do nome da pasta: `real_data_*` é real e `<vocoder>_*` é fake (decisão nossa, conforme o nome dos arquivos baixados por `download_all.py`).
- Locutor: os nomes seguem o padrão LibriSpeech `<locutor>-<capítulo>-<utt>` (os do hifigan terminam em `_gen`). Portanto o locutor **é identificável**. Os 146 locutores e os 11.126 arquivos são exatamente os de LibriSpeech dev-clean + test-clean + dev-other + test-other (40 + 40 + 33 + 33 locutores). Os 5.323 arquivos reais de dev/test-clean são idênticos (mesmo nome e mesma duração) aos do nosso `librispeech` (seção 4.4). Gerador: hifigan, vocos ou wavegrad (resíntese por vocoder do mesmo áudio real).

### codecfake_plus (`data/raw/train_val_en/codecfake_plus/`)
- Estrutura: `CoSG/*.wav` (1.797), `CoSG_labels.txt`, `CoRS_labels.txt`. O áudio do CoRS **não foi baixado** (só o arquivo de rótulos), então o CoRS fica fora do manifest.
- Rótulos: `CoSG_labels.txt`, separado por espaço: `<modelo> <id> <quantização> <aux> <domínio> <bonafide|Spoof>`. São 866 bonafide e 931 Spoof, e cada `<id>.wav` tem uma linha.
- Locutor: **não há**. Os nomes são `<MODELO>_<n>` e vêm de amostras de páginas de demo de 17 TTS baseados em codec. Gerador: coluna 1 nas linhas Spoof. Nas linhas bonafide, a coluna 1 indica de qual demo vem o áudio real.

### librispeech (`data/raw/real_en/librispeech/LibriSpeech/`)
- Estrutura: `{train-clean-100,dev-clean,test-clean}/<locutor>/<capítulo>/<locutor>-<capítulo>-<utt>.flac` mais `*.trans.txt`, `SPEAKERS.TXT`, `CHAPTERS.TXT` etc.
- Áudio: 33.862 `.flac` (train-clean-100 28.539, dev-clean 2.703, test-clean 2.620). Todos reais.
- Locutor: pasta de 1º nível (251 em train-clean-100, 40 em dev-clean, 40 em test-clean).

### in_the_wild (`data/raw/test_generalization/in_the_wild/release_in_the_wild/`)
- Estrutura: 31.779 `*.wav` soltos, `meta.csv` e `attribution.txt`.
- Rótulos: `meta.csv` com as colunas `file,speaker,label`, onde `label` ∈ {`spoof`, `bona-fide`}.
- Locutor: coluna `speaker` (54 celebridades). Gerador: **não informado**, e o manifest usa `generator="unknown"` nos fakes.

### llamapartialspoof (`data/raw/test_generalization/llamapartialspoof/`)
- Estrutura: `R01TTS.0.a/*.wav` (76.228) e `R01TTS.0.b/*.wav` (64.388), `label_R01TTS.0.{a,b}.txt`, `metadata_crossfade.csv`, `README.md`/`README.txt`.
- Rótulos: `label_<parte>.txt` com `<id> <duração> <rótulo do áudio> <ini-fim-rótulo> …` (README). O id é o nome do wav.
- Prefixo do id: `dev-clean_…` / `test-clean_…` são reais (LibriTTS dev-clean e test-clean: 10.573). `dev01-<tts>-full_…` são fakes completos (33.461, todos os segmentos spoof). `dev01-<tts>-partial-{cf,cp,oa}_…` são parcialmente fakes (96.582, todos com segmentos bonafide e spoof misturados).
- Locutor: o campo depois do prefixo, `<prefixo>_<locutor>_<capítulo>_…` (id LibriTTS = id LibriSpeech). Gerador: `<tts>` no prefixo (cosyvoice, elevenlab, gptsovits, ljjets, xttsv2, yourtts).

### asvspoof2021_df (`data/raw/test_generalization/asvspoof2021_df/`)
- Estrutura: `ASVspoof2021_DF_eval/flac/*.flac` (152.955, só **1 das 4 partes**), `ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt` (lista sem rótulo, 611.829 ids), `keys/DF/CM/trial_metadata.txt` (rótulos) e os scores dos baselines.
- Rótulos: `trial_metadata.txt`, separado por espaço, com 13 colunas. As que usamos: 1 locutor, 2 arquivo, 3 codec, 4 corpus de origem (asvspoof/vcc2018/vcc2020), 5 ataque, 6 `bonafide|spoof`, 7 trim e 8 subconjunto (`progress|eval|hidden`). Ficaram no manifest só os 152.955 ids presentes em disco.
- Locutor: coluna 1, com o namespace do corpus da coluna 4 (`asvspoof:LA_0023`, `vcc2020:TEF2`…). Os 67 locutores `asvspoof:*` são exatamente os do ASVspoof 2019 eval. Gerador: coluna 5 (A07–A19, `HUB-*`, `SPO-*`, `Task1/2-teamNN`).

### brspeech_df (`data/raw/test_ptbr/brspeech_df/data/test-0000{0..6}-of-00007.parquet`)
- Formato: config `default`, split `test`, 7.907 linhas. Colunas: `audio: struct<bytes: binary, path: string>` e `label: int64`. O metadado HF do parquet define `label` como ClassLabel `{0: bonafide, 1: spoof}`. Os bytes são FLAC.
- **Como tratamos:** `scripts/extract_brspeech.py` (que o `build_manifest.py` também chama) extrai cada áudio para `data/processed/brspeech_df/test/<audio.path>` (7.907 `.flac`, 2,0 GB). A extração só acontece se o arquivo ainda não existir, e o nome original de `audio.path` é mantido (os nomes são únicos). O `path` do manifest aponta para esses arquivos.
- Locutor e gerador: **não há colunas**. Ambos vêm do nome do arquivo: `[<tts>_]<n1>_<n2>_<n3>[-<k>].flac`. O gerador é o prefixo (f5tts, fish-speech, toucantts, xtts, yourtts). Assumimos que `<n1>` é o locutor, seguindo o padrão `locutor_livro_segmento` do MLS/LibriVox (o card diz que o real vem do LibriVox). São 10 locutores. Card consultado: `https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF` (README.md).

### coraa (`data/raw/test_ptbr/coraa/`)
- Estrutura: `test/{CORAL,NURC_RE/*,Ted_part1,Ted_part3,alip,sp}/*.wav` (12.676) e `metadata_test_final.csv`.
- Rótulos: todos reais (é um corpus de ASR). O CSV traz `file_path,task,variety,dataset,accent,…,text`.
- Locutor: **não há** coluna de locutor (`speaker_id` nulo). `variety`: 12.350 `pt_br` e **326 `pt_pt`** (TEDx). Os `pt_pt` ficam em `role=excluded`.

## 3. Leitura de cabeçalhos (soundfile.info, 32 threads, todos os 756.350 áudios)

O cache fica em `data/processed/audio_info.parquet`. **Nenhum arquivo corrompido ou ilegível (0 de 756.350).** Todos os arquivos com rótulo existem em disco, e não há áudio em disco sem rótulo (o CoRS e as 3 partes não baixadas do DF não contam, porque não estão em disco).

| source_dataset    | .flac  | .wav   | ilegíveis | min_s | p5_s | mediana_s | média_s | p95_s | max_s  | total_h |
|:------------------|-------:|-------:|---:|-----:|-----:|------:|------:|------:|-------:|-------:|
| asvspoof2019_la   | 122299 | 0      | 0 | 0.47 | 1.36 | 2.99  | 3.27  | 6.14  | 16.55  | 111.15 |
| dfadd             | 163500 | 44455  | 0 | 1.22 | 2.22 | 3.66  | 3.83  | 5.99  | 16.56  | 221.50 |
| shiftyspeech      | 44504  | 0      | 0 | 1.06 | 2.29 | 5.49  | 6.88  | 16.27 | 35.16  | 85.05  |
| codecfake_plus    | 0      | 1797   | 0 | 0.85 | 2.78 | 5.57  | 6.05  | 10.30 | 28.58  | 3.02   |
| librispeech       | 33862  | 0      | 0 | 1.28 | 3.22 | 13.54 | 11.84 | 16.11 | 34.96  | 111.38 |
| in_the_wild       | 0      | 31779  | 0 | 0.44 | 1.19 | 3.27  | 4.29  | 10.80 | 24.99  | 37.85  |
| llamapartialspoof | 0      | 140616 | 0 | 0.28 | 1.44 | 5.14  | 6.24  | 14.97 | 56.54  | 243.74 |
| asvspoof2021_df   | 152955 | 0      | 0 | 0.35 | 1.32 | 2.76  | 2.99  | 5.06  | 29.22  | 126.93 |
| brspeech_df       | 7907 (extraídos) | 0 | 0 | 0.02 | 1.63 | 11.04 | 10.72 | 20.33 | 375.93 | 23.55 |
| coraa             | 0      | 12676  | 0 | 0.15 | 0.72 | 2.34  | 3.19  | 8.60  | 39.42  | 11.24  |

Áudios curtos ou longos: < 1 s: asvspoof2019_la 1.277, codecfake_plus 1, in_the_wild 865, llamapartialspoof 2.022, asvspoof2021_df 1.833, brspeech_df 100, coraa 1.425. > 20 s: shiftyspeech 958, codecfake_plus 2, librispeech 154, in_the_wild 197, llamapartialspoof 1.613, asvspoof2021_df 12, brspeech_df 437 (máx. 376 s), coraa 26.

Taxas de amostragem e formato:

| source_dataset | taxa (Hz) | formato |
|:--|:--|:--|
| asvspoof2019_la, asvspoof2021_df, librispeech | 16000 | FLAC PCM_16, mono |
| dfadd | 16000 | FLAC PCM_16 (fake) / WAV PCM_16 (real), mono |
| codecfake_plus, llamapartialspoof | 16000 | WAV PCM_16, mono |
| in_the_wild, coraa | 16000 | WAV FLOAT, mono |
| shiftyspeech | **24000** (real e fake) | FLAC PCM_16, mono |
| brspeech_df | **48000 nos 1.320 reais e 24000 nos 6.587 fakes** | FLAC PCM_16, mono |

Todos os áudios são mono. ⚠ No brspeech_df, a taxa de amostragem separa perfeitamente real de fake, embora o card diga que tudo foi reamostrado para 24 kHz (ver dúvidas).

## 4. Manifest

`data/processed/manifest.parquet`: **756.350 linhas, 9,7 MB**. Colunas: `path, label, source_dataset, generator, fake_type, speaker_id, language, original_split, role, split, duration_s, sample_rate`. `path` é relativo à raiz do repo, com `/` (nenhum caminho absoluto). O manifest é versionado no git (exceção no `.gitignore`).

Regras de `role`:
- **train_val**: asvspoof2019_la (train, dev), dfadd, shiftyspeech, codecfake_plus (CoSG) e librispeech train-clean-100, **menos** os locutores do LlamaPartialSpoof (abaixo).
- **test_generalization**: in_the_wild, llamapartialspoof, asvspoof2021_df e asvspoof2019_la eval. Os dois ASVspoof são testes secundários (ver o aviso no topo).
- **test_ptbr**: brspeech_df e coraa (só `pt_br`).
- **excluded** (27.535 linhas):

  | o que | linhas | motivo |
  |:--|--:|:--|
  | librispeech dev-clean + test-clean | 5.323 | mesma origem (áudio e locutores) dos reais do LlamaPartialSpoof (LibriTTS dev/test-clean) |
  | shiftyspeech, locutores do LlamaPartialSpoof | 21.048 (5.262 reais + 15.786 fakes, 79 locutores) | vazamento de locutor e de gravação para o teste LlamaPartialSpoof |
  | asvspoof2019_la, áudios de cadastro ASV | 838 | fora do protocolo CM |
  | coraa `pt_pt` | 326 | português europeu, fora do teste pt-BR |

Divisão train/val: por grupo de locutor, global entre bancos, com `speaker_id` com namespace (`vctk:p225`, `librispeech:1089`, `asvspoof:LA_0079`…), seed 42. A divisão foi refeita depois das exclusões. Em cada banco do pool, os grupos são percorridos em ordem aleatória e vão para val até chegar a ≤ 20% dos áudios daquele banco. Exceções: o ASVspoof 2019 train+dev inteiro vai para train, e no CoSG o grupo é o modelo/demo. **Nenhum `speaker_id` aparece em train e val ao mesmo tempo.**

### 4.1 Contagem por source_dataset × split × label

| source_dataset    | excluded/0 | excluded/1 | test/0 | test/1 | train/0 | train/1 | val/0 | val/1 |
|:------------------|---:|---:|---:|---:|---:|---:|---:|---:|
| asvspoof2019_la   | 838  | 0     | 7355  | 63882  | 5128  | 45096  | 0    | 0     |
| dfadd             | 0    | 0     | 0     | 0      | 35745 | 131700 | 8710 | 31800 |
| shiftyspeech      | 5262 | 15786 | 0     | 0      | 4707  | 14121  | 1157 | 3471  |
| codecfake_plus    | 0    | 0     | 0     | 0      | 702   | 743    | 164  | 188   |
| librispeech       | 5323 | 0     | 0     | 0      | 22840 | 0      | 5699 | 0     |
| in_the_wild       | 0    | 0     | 19963 | 11816  | 0     | 0      | 0    | 0     |
| llamapartialspoof | 0    | 0     | 10573 | 130043 | 0     | 0      | 0    | 0     |
| asvspoof2021_df   | 0    | 0     | 5535  | 147420 | 0     | 0      | 0    | 0     |
| brspeech_df       | 0    | 0     | 1320  | 6587   | 0     | 0      | 0    | 0     |
| coraa             | 326  | 0     | 12350 | 0      | 0     | 0      | 0    | 0     |

| split | áudios | real | fake |
|:--|--:|--:|--:|
| train | 260.782 | 69.122 | 191.660 |
| val | 51.189 | 15.730 | 35.459 |
| test (test_generalization) | 396.587 | 43.426 | 353.161 |
| test (test_ptbr) | 20.257 | 13.670 | 6.587 |
| excluded | 27.535 | 11.749 | 15.786 |

Fração em val e locutores por banco do pool:
- dfadd: 19,5% (88 locutores em train, 22 em val).
- librispeech: 20,0% (199 / 52).
- shiftyspeech: 19,7% (55 / 12; restam 67 locutores, os de dev-other, test-other e o 2094).
- codecfake_plus: 19,6% (grupos = demos).
- asvspoof2019_la: 0% (40 locutores, todos em train).

**No total do pool: train 260.782 (83,6%) e val 51.189 (16,4%).**

### 4.2 Proporção real/fake

| conjunto | n | real | fake | % fake |
|:--|--:|--:|--:|--:|
| train_val (train+val) | 311.971 | 84.852 | 227.119 | 72,8 |
| train | 260.782 | 69.122 | 191.660 | 73,5 |
| val | 51.189 | 15.730 | 35.459 | 69,3 |
| test / asvspoof2019_la (eval) — secundário | 71.237 | 7.355 | 63.882 | 89,7 |
| test / in_the_wild | 31.779 | 19.963 | 11.816 | 37,2 |
| test / llamapartialspoof | 140.616 | 10.573 | 130.043 | 92,5 |
| ↳ só real + partial | 107.155 | 10.573 | 96.582 | 90,1 |
| test / asvspoof2021_df (parte 1) — secundário | 152.955 | 5.535 | 147.420 | 96,4 |
| test / brspeech_df | 7.907 | 1.320 | 6.587 | 83,3 |
| test / coraa (pt_br) | 12.350 | 12.350 | 0 | 0,0 |
| test_ptbr (brspeech + coraa) | 20.257 | 13.670 | 6.587 | 32,5 |

### 4.3 Sobreposição de locutores entre train_val e os bancos de teste

Por `speaker_id`, **nenhum banco de teste tem locutor em comum com train_val**, incluindo o LlamaPartialSpoof (antes eram 79).

| banco de teste | locutores | em comum com train_val (por id) |
|:--|--:|--:|
| asvspoof2019_la (eval) | 67 | 0* |
| in_the_wild | 54 | 0 |
| llamapartialspoof | 79 | 0 |
| asvspoof2021_df | 93 | 0* |
| brspeech_df | 10 | 0 |
| coraa | sem id | – |

\* **Sobreposição oculta, não corrigível:** os locutores `asvspoof:LA_*` são do VCTK anonimizado, e o DFADD (train_val) contém os 110 locutores do VCTK. Por isso esses dois testes são **secundários**: valem para geradores/ataques novos, não para locutores novos.

O locutor `librispeech:2094` (LibriSpeech test-clean) **não está** no LlamaPartialSpoof. Por isso as 244 linhas dele no ShiftySpeech (61 reais + 183 fakes) continuam no pool (train). Nenhum locutor do LibriSpeech train-clean-100 está entre os 79 do LlamaPartialSpoof.

### 4.4 Duplicados entre bancos (nome do arquivo + duração arredondada a 1 ms)

- Há 5.323 pares `librispeech` × `shiftyspeech` (o LibriSpeech dev/test-clean é idêntico aos reais do ShiftySpeech). Agora 5.262 deles têm as duas cópias em `excluded`. Os 61 restantes são do locutor 2094: a cópia do ShiftySpeech está em train e a do LibriSpeech em excluded. Não há duplicado entre splits usados (train/val/test).
- Dentro do shiftyspeech, 697 chaves se repetem: o áudio real e a versão vocodada com o mesmo nome e a mesma duração. Ficam sempre no mesmo split.
- O LlamaPartialSpoof (LibriTTS) não casa por nome com o LibriSpeech, porque o LibriTTS re-segmenta as falas. O vazamento de conteúdo foi tratado por locutor (acima).

## 5. Decisões tomadas

1. **BRSpeech-DF**: extraído por `scripts/extract_brspeech.py` para `data/processed/brspeech_df/test/`, com o nome original de `audio.path` e pulando o que já existe. Só o split `test` do config `default` (7.907 = 1.320 + 6.587, igual ao card). `language="pt"`: o banco **mistura pt-BR e pt-PT** sem indicação por arquivo. A diferença de taxa de amostragem (48 kHz nos reais, 24 kHz nos fakes) **não** foi alterada no manifest (ver o aviso no topo).
2. **Namespaces de locutor**: `speaker_id` leva o prefixo do corpus de origem (`vctk:`, `asvspoof:`, `librispeech:`, `itw:`, `brspeech:`, `vcc2018:`, `vcc2020:`). LibriSpeech, LibriTTS (LlamaPartialSpoof) e ShiftySpeech compartilham `librispeech:`. O ASVspoof 2019 e o 2021 DF compartilham `asvspoof:`.
3. **Vazamento para o LlamaPartialSpoof**: montei o conjunto dos 79 `speaker_id` do LlamaPartialSpoof, e toda linha `train_val` de qualquer banco com locutor nesse conjunto virou `excluded`. Na prática, só o ShiftySpeech foi atingido: 21.048 linhas (5.262 reais + 15.786 fakes, 79 locutores). Nenhum locutor do train-clean-100 está no conjunto. A divisão train/val foi refeita em seguida.
4. **VCTK compartilhado (ASVspoof × DFADD)**: o ASVspoof anonimiza os locutores (`LA_xxxx`) e não há mapeamento para os `pXXX` do VCTK. Tentei casar pela duração e não funcionou (nível de acaso). Mantido como está. O ASVspoof 2019 train+dev inteiro vai para train, como um único grupo, para reduzir o vazamento train↔val. Ainda é possível que uns 8 dos 22 locutores do DFADD em val estejam entre os 40 anônimos do ASVspoof em train. **asvspoof2019_la (eval) e asvspoof2021_df são testes secundários** (só geradores/ataques novos).
5. **Meta de 20% por banco** (não global): como o ASVspoof 2019 fica todo em train, o total do pool fica em 83,6/16,4.
6. **CodecFake+ CoSG sem locutor**: o grupo é o modelo/demo (coluna 1 do `CoSG_labels.txt`), juntando o fake e o real da mesma demo. Os modelos do CoSG em val não aparecem em train.
7. **ShiftySpeech**: o locutor vem do nome do arquivo (padrão LibriSpeech). O rótulo vem do nome da pasta (não existe arquivo de rótulo).
8. **LlamaPartialSpoof `fake_type`** vem dos **segmentos** do arquivo de label: `partial` quando o áudio mistura segmentos `bonafide` e `spoof`, e `full` quando todos são `spoof`. Comparado com o nome do arquivo (`-full` / `-partial-*`), **0 linhas divergiram**: 33.461 full, 96.582 partial e 10.573 reais.
9. **Gerador**: código do ataque no ASVspoof (`A07`, `Task1-team20`, `HUB-N16`…) e nome do TTS/vocoder nos demais. No In-the-Wild, `generator="unknown"` (o banco não informa). Nos reais, nulo.
10. **`speaker_id` do `ljjets`** (LlamaPartialSpoof) é **apenas informativo**: é o locutor do texto de origem, mas a voz é a da LJSpeech (TTS de voz única).
11. **`original_split`**:
    - asvspoof2019_la: `train`, `dev` ou `eval`.
    - dfadd: `train`, `valid` ou `test`.
    - librispeech e reais do llamapartialspoof: subconjunto do LibriSpeech/LibriTTS (ex.: `train-clean-100`, `dev-clean`).
    - asvspoof2021_df: `progress`, `eval` ou `hidden`.
    - brspeech_df e coraa: `test`.
    - codecfake_plus: `CoSG`.
    - Nulo no shiftyspeech, no in_the_wild e nos fakes do llamapartialspoof.
12. **CORAA**: os 326 `pt_pt` (TEDx) ficam em `excluded`. `speaker_id` nulo (sem coluna de locutor).
13. Os áudios muito curtos ou muito longos (BRSpeech 0,02 s; 376 s) não foram removidos. A duração fica no manifest para filtrar depois.

## 6. Dúvidas e pendências

Resolvidas nesta revisão:
- ShiftySpeech × LlamaPartialSpoof: tratada (decisão 3).
- VCTK compartilhado: mantida e documentada (decisão 4).
- `fake_type`: conferido nos segmentos, sem divergência (decisão 8).
- CORAA pt_pt: excluído (decisão 12).
- In-the-Wild `"unknown"`: mantido (decisão 9).
- `ljjets`: informativo (decisão 10).

Em aberto:
1. **BRSpeech-DF 48 kHz × 24 kHz** (ver o aviso no topo): a reamostragem tem que acontecer no carregamento. Vale também conferir a banda útil dos reais (se foram gravados em banda mais larga, a reamostragem sozinha pode não bastar).
2. **BRSpeech-DF: locutor inferido do nome** (`<n1>` de `n1_n2_n3`). O formato não é documentado no card.
3. **Locutor 2094** (LibriSpeech test-clean) continua no pool via ShiftySpeech. Não está no LlamaPartialSpoof, então não vaza para nenhum teste. Fica registrado.
4. **Desbalanceamento**: o pool tem 72,8% de fakes (o ASVspoof 2019 tem ~9 fakes por real). Fica para o treino (pesos ou amostragem).
5. **Val sem ASVspoof 2019**: consequência da decisão 4. Os ataques A01–A06 só aparecem em train.
6. **Cobertura parcial**: o ASVspoof 2021 DF tem só a parte 1 de 4 (152.955 de 611.829) e o CodecFake+ só o CoSG (o CoRS não foi baixado).
