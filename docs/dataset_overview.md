# Visão geral do banco de dados

Gerado automaticamente pelo `notebook.ipynb` (seção 3) a partir de `data/processed/manifest.parquet`, `docs/datasets.yaml` e `docs/audit_report.md`. Não edite à mão: rode o notebook de novo.

## Funil

| split    |   áudios |   % do total |   real |   fake |
|:---------|---------:|-------------:|-------:|-------:|
| train    |   260782 |         34.5 |  69122 | 191660 |
| val      |    51189 |          6.8 |  15730 |  35459 |
| test     |   416844 |         55.1 |  57096 | 359748 |
| excluded |    27535 |          3.6 |  11749 |  15786 |
| total    |   756350 |        100   | 153697 | 602653 |

Mais da metade do banco (55%) está reservada como teste, de propósito: os bancos de teste são inteiros e nunca vistos no treino, para medir generalização.

## Glossário das colunas do manifest

| coluna         | significado                                                                                       |
|:---------------|:--------------------------------------------------------------------------------------------------|
| path           | caminho do arquivo de áudio, relativo à raiz do repositório                                       |
| label          | 0 = real (voz humana gravada), 1 = fake (gerada ou modificada por IA)                             |
| source_dataset | banco de origem (mesmo nome do `docs/datasets.yaml`)                                              |
| generator      | sistema que gerou o fake (TTS, vocoder, código do ataque); vazio nos reais                        |
| fake_type      | `none` (real), `full` (áudio todo falso) ou `partial` (só trechos falsos)                         |
| speaker_id     | locutor, com prefixo do corpus de origem (ex.: `vctk:p225`); vazio quando o banco não informa     |
| language       | `en` ou `pt`                                                                                      |
| original_split | partição no banco original (ex.: `train`, `dev`, `eval`)                                          |
| role           | papel no projeto: `train_val`, `test_generalization`, `test_ptbr` ou `excluded`                   |
| split          | uso neste projeto: `train`, `val` ou `test`; vazio nos excluídos (o notebook chama de `excluded`) |
| duration_s     | duração em segundos (lida do cabeçalho do arquivo)                                                |
| sample_rate    | taxa de amostragem original do arquivo, em Hz                                                     |

## Tabela por fonte

| banco             | ano             | origem                                                           | licença         | papel                                                                              | idioma                                                | contém      | geradores (áudios por gerador)                                                                                               | como o rótulo foi obtido                                                                                | contagens (split/label)                                                                                              |
|:------------------|:----------------|:-----------------------------------------------------------------|:----------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------|:------------|:-----------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| asvspoof2019_la   | não documentado | https://datashare.ed.ac.uk/handle/10283/3336                     | ODC-By          | treino/validação (train+dev); a parte eval é teste de generalização **secundário** | en                                                    | real e fake | A01: 7.516; A02: 7.516; A03: 7.516; A04: 7.516; A05: 7.516; A06: 7.516; A11: 4.914; A14: 4.914; … e mais 11 geradores        | protocolos CM (`ASVspoof2019.LA.cm.*.txt`), coluna KEY: bonafide → 0, spoof → 1                         | excluded/real: 838; test/real: 7.355; test/fake: 63.882; train/real: 5.128; train/fake: 45.096                       |
| dfadd             | não documentado | https://huggingface.co/datasets/isjwdu/DFADD                     | não documentado | treino/validação                                                                   | en                                                    | real e fake | PflowTTS: 33.000; GradTTS: 32.700; NaturalSpeech2: 32.700; StyleTTS2: 32.700; MatchaTTS: 32.400                              | arquivos `{train,valid,test}.txt` (formato ASVspoof), coluna KEY: bonafide → 0, spoof → 1               | train/real: 35.745; train/fake: 131.700; val/real: 8.710; val/fake: 31.800                                           |
| shiftyspeech      | não documentado | https://huggingface.co/datasets/ash56/ShiftySpeech               | não documentado | treino/validação                                                                   | en                                                    | real e fake | hifigan: 5.864; vocos: 5.864; wavegrad: 5.864                                                                                | não há arquivo de rótulo: nome da pasta, `real_data_*` → 0 e `<vocoder>_*` → 1 (decisão do projeto)     | excluded/real: 5.262; excluded/fake: 15.786; train/real: 4.707; train/fake: 14.121; val/real: 1.157; val/fake: 3.471 |
| codecfake_plus    | não documentado | https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset | não documentado | treino/validação (só a parte CoSG)                                                 | en                                                    | real e fake | MASKGCT: 576; CLAMTTS: 78; VALLE: 57; NS3: 32; SIMPLESPEECH1: 32; SIMPLESPEECH2: 31; GPST: 30; NS2: 23; … e mais 9 geradores | `CoSG_labels.txt`, última coluna: bonafide → 0, Spoof → 1                                               | train/real: 702; train/fake: 743; val/real: 164; val/fake: 188                                                       |
| librispeech       | não documentado | https://www.openslr.org/12                                       | CC BY 4.0       | treino/validação (só train-clean-100; dev-clean e test-clean excluídos)            | en                                                    | só real     | — (só real)                                                                                                                  | corpus de leitura de audiolivros, todo real → 0                                                         | excluded/real: 5.323; train/real: 22.840; val/real: 5.699                                                            |
| in_the_wild       | não documentado | https://huggingface.co/datasets/mueller91/In-The-Wild            | não documentado | teste de generalização (principal)                                                 | en                                                    | real e fake | unknown: 11.816                                                                                                              | `meta.csv`, coluna label: bona-fide → 0, spoof → 1                                                      | test/real: 19.963; test/fake: 11.816                                                                                 |
| llamapartialspoof | não documentado | https://github.com/hieuthi/LlamaPartialSpoof                     | não documentado | teste de generalização (principal)                                                 | en                                                    | real e fake | elevenlab: 21.738; gptsovits: 21.738; ljjets: 21.738; yourtts: 21.738; xttsv2: 21.729; cosyvoice: 21.362                     | `label_R01TTS.0.{a,b}.txt` e prefixo do id: reais (LibriTTS dev/test-clean) → 0; `full` e `partial` → 1 | test/real: 10.573; test/fake: 130.043                                                                                |
| asvspoof2021_df   | não documentado | https://zenodo.org/records/4835108                               | não documentado | teste de generalização **secundário** (só a parte 1 de 4)                          | en                                                    | real e fake | A11: 2.865; A10: 2.837; A16: 2.817; A12: 2.815; A13: 2.808; A07: 2.803; A08: 2.771; A17: 2.759; … e mais 102 geradores       | `keys/DF/CM/trial_metadata.txt`, coluna 6: bonafide → 0, spoof → 1                                      | test/real: 5.535; test/fake: 147.420                                                                                 |
| brspeech_df       | não documentado | https://huggingface.co/datasets/AKCIT-Deepfake/BRSpeech-DF       | Apache-2.0      | teste pt (principal)                                                               | pt (mistura pt-BR e pt-PT, sem indicação por arquivo) | real e fake | toucantts: 1.320; xtts: 1.320; f5tts: 1.316; yourtts: 1.316; fish-speech: 1.315                                              | coluna `label` do parquet (ClassLabel 0 = bonafide, 1 = spoof)                                          | test/real: 1.320; test/fake: 6.587                                                                                   |
| coraa             | não documentado | https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1          | CC BY-NC-ND 4.0 | teste pt-BR, só real: mede a taxa de falso alarme                                  | pt-BR (os 326 `pt_pt` foram excluídos)                | só real     | — (só real)                                                                                                                  | corpus de reconhecimento de fala, todo real → 0                                                         | excluded/real: 326; test/real: 12.350                                                                                |

## Contagens por banco, split e label

| source_dataset    |   train/real |   train/fake |   val/real |   val/fake |   test/real |   test/fake |   excluded/real |   excluded/fake |
|:------------------|-------------:|-------------:|-----------:|-----------:|------------:|------------:|----------------:|----------------:|
| asvspoof2019_la   |         5128 |        45096 |          0 |          0 |        7355 |       63882 |             838 |               0 |
| dfadd             |        35745 |       131700 |       8710 |      31800 |           0 |           0 |               0 |               0 |
| shiftyspeech      |         4707 |        14121 |       1157 |       3471 |           0 |           0 |            5262 |           15786 |
| codecfake_plus    |          702 |          743 |        164 |        188 |           0 |           0 |               0 |               0 |
| librispeech       |        22840 |            0 |       5699 |          0 |           0 |           0 |            5323 |               0 |
| in_the_wild       |            0 |            0 |          0 |          0 |       19963 |       11816 |               0 |               0 |
| llamapartialspoof |            0 |            0 |          0 |          0 |       10573 |      130043 |               0 |               0 |
| asvspoof2021_df   |            0 |            0 |          0 |          0 |        5535 |      147420 |               0 |               0 |
| brspeech_df       |            0 |            0 |          0 |          0 |        1320 |        6587 |               0 |               0 |
| coraa             |            0 |            0 |          0 |          0 |       12350 |           0 |             326 |               0 |
| total             |        69122 |       191660 |      15730 |      35459 |       57096 |      359748 |           11749 |           15786 |

## Exclusões

| o que                                        |   linhas | motivo                                                                                                                     |
|:---------------------------------------------|---------:|:---------------------------------------------------------------------------------------------------------------------------|
| librispeech dev-clean + test-clean           |     5323 | mesma origem (áudio e locutores) dos reais do LlamaPartialSpoof (LibriTTS dev/test-clean)                                  |
| shiftyspeech, locutores do LlamaPartialSpoof |    21048 | 5.262 reais + 15.786 fakes de 79 locutores que também estão no teste LlamaPartialSpoof: vazamento de locutor e de gravação |
| asvspoof2019_la, áudios de cadastro do ASV   |      838 | fora do protocolo de detecção (CM), sem uso aqui                                                                           |
| coraa `pt_pt`                                |      326 | português europeu, fora do teste pt-BR                                                                                     |

## Decisões

| O que decidimos                                                                                  | Por quê                                                                                                                                                                                | Alternativa descartada e por quê                                                                                                                                                                                                |
|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dividir treino/validação **por locutor**                                                         | o modelo não pode acertar por reconhecer a voz de quem já ouviu                                                                                                                        | Dividir **por arquivo**: descartada, porque o mesmo locutor cai em treino e validação e infla o resultado                                                                                                                       |
| Bancos de teste **inteiros e separados** do treino                                               | medem generalização para geradores, locutores e gravações nunca vistos                                                                                                                 | **Misturar tudo e dividir em 3 partes**: descartada, porque os mesmos geradores nos 3 conjuntos não medem generalização                                                                                                         |
| Não usar o **In-the-Wild** no treino                                                             | é o único teste com deepfakes reais tirados da internet                                                                                                                                | Treinar com ele: descartada, perderíamos a medida mais próxima de um golpe real                                                                                                                                                 |
| **ASVspoof 2019** (train+dev) inteiro em train                                                   | os locutores são anônimos e podem ser os mesmos do VCTK do DFADD; um bloco só evita vazamento treino↔validação dentro do próprio ASVspoof                                              | Dividir os 40 locutores do ASVspoof entre train e val: descartada, a validação herdaria ataques A01–A06 com locutores possivelmente repetidos                                                                                   |
| Excluir **LibriSpeech dev/test-clean**                                                           | são a mesma origem (áudio e locutores) dos reais do LlamaPartialSpoof                                                                                                                  | Usá-los como reais extras no treino: descartada, vazaria para o teste                                                                                                                                                           |
| Excluir os **locutores do LlamaPartialSpoof no ShiftySpeech**                                    | 21.048 áudios com os mesmos locutores e gravações do teste                                                                                                                             | Mantê-los: descartada, inflaria o resultado no LlamaPartialSpoof                                                                                                                                                                |
| Excluir o **CORAA pt-PT**                                                                        | o teste de falso alarme é de português brasileiro                                                                                                                                      | Manter os 326: descartada, misturaria outra variedade sem poder separar no resultado                                                                                                                                            |
| **Reamostrar tudo para 16 kHz** no carregamento                                                  | a maioria dos bancos já está em 16 kHz, e uma taxa única elimina o atalho do BRSpeech-DF (real 48 kHz, fake 24 kHz)                                                                    | Usar a taxa original de cada arquivo: descartada, a taxa (e a banda) viraria uma pista do rótulo                                                                                                                                |
| Recorte de **4 s**, aleatório no treino e central na validação/teste; áudio curto é **repetido** | entrada de tamanho fixo; o recorte aleatório mostra trechos diferentes a cada época; repetir não cria silêncio artificial                                                              | Completar com zeros: descartada, a quantidade de silêncio contaria a duração do áudio, que difere entre real e fake                                                                                                             |
| Log-mel **padronizado por exemplo** (média 0, desvio 1)                                          | tira o volume do jogo: um ganho no áudio vira uma constante no log-mel, que a padronização remove                                                                                      | Usar o log-mel cru: descartada, o volume difere entre bancos e entre real/fake e seria um atalho                                                                                                                                |
| Amostra **50/50 real/fake com teto por fonte e por gerador, sorteada de novo a cada época**      | o pool é 73% fake e o DFADD sozinho é 64% do treino; assim nenhuma fonte ou gerador (como os antigos do ASVspoof 2019) domina, e ao longo das épocas o modelo vê a maior parte do pool | **Todos os áudios com pesos de classe**: alternativa válida para o real/fake, mas não resolve a dominância do DFADD nem dos geradores (64% dos gradientes continuariam vindo de um banco) e deixa as épocas 5 vezes mais longas |
| Escolher o checkpoint pela **menor EER na validação**                                            | a EER não depende da proporção real/fake nem de um limiar fixo                                                                                                                         | Pela acurácia: descartada, a validação é 69% fake e a acurácia muda com essa proporção; pela perda: mede a calibração, não a separação                                                                                          |
| **Sem augmentation** no baseline                                                                 | primeiro medimos o modelo puro, para saber depois quanto a augmentation ajuda                                                                                                          | Já começar com augmentation de codec/ruído: fica para a próxima etapa, senão não há referência                                                                                                                                  |

## Geradores por banco (todos os splits usados)

| source_dataset    | generator      |   áudios |
|:------------------|:---------------|---------:|
| asvspoof2019_la   | A01            |     7516 |
| asvspoof2019_la   | A02            |     7516 |
| asvspoof2019_la   | A03            |     7516 |
| asvspoof2019_la   | A04            |     7516 |
| asvspoof2019_la   | A05            |     7516 |
| asvspoof2019_la   | A06            |     7516 |
| asvspoof2019_la   | A07            |     4914 |
| asvspoof2019_la   | A08            |     4914 |
| asvspoof2019_la   | A09            |     4914 |
| asvspoof2019_la   | A10            |     4914 |
| asvspoof2019_la   | A11            |     4914 |
| asvspoof2019_la   | A12            |     4914 |
| asvspoof2019_la   | A13            |     4914 |
| asvspoof2019_la   | A14            |     4914 |
| asvspoof2019_la   | A15            |     4914 |
| asvspoof2019_la   | A16            |     4914 |
| asvspoof2019_la   | A17            |     4914 |
| asvspoof2019_la   | A18            |     4914 |
| asvspoof2019_la   | A19            |     4914 |
| asvspoof2021_df   | A07            |     2803 |
| asvspoof2021_df   | A08            |     2771 |
| asvspoof2021_df   | A09            |     2687 |
| asvspoof2021_df   | A10            |     2837 |
| asvspoof2021_df   | A11            |     2865 |
| asvspoof2021_df   | A12            |     2815 |
| asvspoof2021_df   | A13            |     2808 |
| asvspoof2021_df   | A14            |     2743 |
| asvspoof2021_df   | A15            |     2717 |
| asvspoof2021_df   | A16            |     2817 |
| asvspoof2021_df   | A17            |     2759 |
| asvspoof2021_df   | A18            |     2662 |
| asvspoof2021_df   | A19            |     2650 |
| asvspoof2021_df   | HUB-B00        |      161 |
| asvspoof2021_df   | HUB-B01        |     1292 |
| asvspoof2021_df   | HUB-D01        |     1266 |
| asvspoof2021_df   | HUB-D02        |     1223 |
| asvspoof2021_df   | HUB-D03        |     1240 |
| asvspoof2021_df   | HUB-D04        |     1283 |
| asvspoof2021_df   | HUB-D05        |     1233 |
| asvspoof2021_df   | HUB-N03        |     1240 |
| asvspoof2021_df   | HUB-N04        |     1292 |
| asvspoof2021_df   | HUB-N05        |     1264 |
| asvspoof2021_df   | HUB-N06        |     1256 |
| asvspoof2021_df   | HUB-N07        |     1328 |
| asvspoof2021_df   | HUB-N08        |     1272 |
| asvspoof2021_df   | HUB-N09        |     1217 |
| asvspoof2021_df   | HUB-N10        |     1273 |
| asvspoof2021_df   | HUB-N11        |     1304 |
| asvspoof2021_df   | HUB-N12        |     1240 |
| asvspoof2021_df   | HUB-N13        |     1290 |
| asvspoof2021_df   | HUB-N14        |     1238 |
| asvspoof2021_df   | HUB-N15        |     1242 |
| asvspoof2021_df   | HUB-N16        |     1332 |
| asvspoof2021_df   | HUB-N17        |     1254 |
| asvspoof2021_df   | HUB-N18        |     1275 |
| asvspoof2021_df   | HUB-N19        |     1252 |
| asvspoof2021_df   | HUB-N20        |     1221 |
| asvspoof2021_df   | SPO-B00        |       96 |
| asvspoof2021_df   | SPO-B01        |     1220 |
| asvspoof2021_df   | SPO-N03        |     1250 |
| asvspoof2021_df   | SPO-N04        |     1228 |
| asvspoof2021_df   | SPO-N05        |     1263 |
| asvspoof2021_df   | SPO-N06        |     1254 |
| asvspoof2021_df   | SPO-N10        |     1279 |
| asvspoof2021_df   | SPO-N11        |     1299 |
| asvspoof2021_df   | SPO-N12        |     1257 |
| asvspoof2021_df   | SPO-N13        |     1258 |
| asvspoof2021_df   | SPO-N16        |     1302 |
| asvspoof2021_df   | SPO-N17        |     1292 |
| asvspoof2021_df   | SPO-N18        |     1310 |
| asvspoof2021_df   | Task1-team01   |      944 |
| asvspoof2021_df   | Task1-team02   |      944 |
| asvspoof2021_df   | Task1-team03   |      879 |
| asvspoof2021_df   | Task1-team04   |      910 |
| asvspoof2021_df   | Task1-team06   |      887 |
| asvspoof2021_df   | Task1-team07   |      885 |
| asvspoof2021_df   | Task1-team08   |      895 |
| asvspoof2021_df   | Task1-team09   |      893 |
| asvspoof2021_df   | Task1-team10   |      904 |
| asvspoof2021_df   | Task1-team11   |      930 |
| asvspoof2021_df   | Task1-team12   |      902 |
| asvspoof2021_df   | Task1-team13   |      858 |
| asvspoof2021_df   | Task1-team14   |      926 |
| asvspoof2021_df   | Task1-team16   |      902 |
| asvspoof2021_df   | Task1-team17   |      936 |
| asvspoof2021_df   | Task1-team18   |      877 |
| asvspoof2021_df   | Task1-team19   |      869 |
| asvspoof2021_df   | Task1-team20   |      943 |
| asvspoof2021_df   | Task1-team21   |      866 |
| asvspoof2021_df   | Task1-team22   |      886 |
| asvspoof2021_df   | Task1-team23   |      919 |
| asvspoof2021_df   | Task1-team24   |      847 |
| asvspoof2021_df   | Task1-team25   |      949 |
| asvspoof2021_df   | Task1-team26   |      855 |
| asvspoof2021_df   | Task1-team27   |      874 |
| asvspoof2021_df   | Task1-team28   |      943 |
| asvspoof2021_df   | Task1-team29   |      858 |
| asvspoof2021_df   | Task1-team30   |      902 |
| asvspoof2021_df   | Task1-team31   |      893 |
| asvspoof2021_df   | Task1-team32   |      908 |
| asvspoof2021_df   | Task1-team33   |      938 |
| asvspoof2021_df   | Task2-team02   |     1387 |
| asvspoof2021_df   | Task2-team03   |     1312 |
| asvspoof2021_df   | Task2-team05   |     1368 |
| asvspoof2021_df   | Task2-team06   |     1346 |
| asvspoof2021_df   | Task2-team07   |     1357 |
| asvspoof2021_df   | Task2-team08   |     1322 |
| asvspoof2021_df   | Task2-team09   |     1370 |
| asvspoof2021_df   | Task2-team10   |     1352 |
| asvspoof2021_df   | Task2-team11   |     1434 |
| asvspoof2021_df   | Task2-team12   |     1386 |
| asvspoof2021_df   | Task2-team13   |     1366 |
| asvspoof2021_df   | Task2-team15   |     1408 |
| asvspoof2021_df   | Task2-team16   |     1324 |
| asvspoof2021_df   | Task2-team18   |     1320 |
| asvspoof2021_df   | Task2-team19   |     1347 |
| asvspoof2021_df   | Task2-team20   |     1347 |
| asvspoof2021_df   | Task2-team22   |     1350 |
| asvspoof2021_df   | Task2-team23   |     1302 |
| asvspoof2021_df   | Task2-team24   |     1332 |
| asvspoof2021_df   | Task2-team25   |     1339 |
| asvspoof2021_df   | Task2-team26   |     1383 |
| asvspoof2021_df   | Task2-team27   |     1245 |
| asvspoof2021_df   | Task2-team28   |     1339 |
| asvspoof2021_df   | Task2-team29   |     1401 |
| asvspoof2021_df   | Task2-team30   |     1332 |
| asvspoof2021_df   | Task2-team31   |     1342 |
| asvspoof2021_df   | Task2-team32   |     1320 |
| asvspoof2021_df   | Task2-team33   |     1337 |
| brspeech_df       | f5tts          |     1316 |
| brspeech_df       | fish-speech    |     1315 |
| brspeech_df       | toucantts      |     1320 |
| brspeech_df       | xtts           |     1320 |
| brspeech_df       | yourtts        |     1316 |
| codecfake_plus    | CLAMTTS        |       78 |
| codecfake_plus    | ELLAV          |        8 |
| codecfake_plus    | GPST           |       30 |
| codecfake_plus    | MASKGCT        |      576 |
| codecfake_plus    | NS2            |       23 |
| codecfake_plus    | NS3            |       32 |
| codecfake_plus    | RALLE          |        5 |
| codecfake_plus    | SIMPLESPEECH1  |       32 |
| codecfake_plus    | SIMPLESPEECH2  |       31 |
| codecfake_plus    | SINGLECODEC    |       14 |
| codecfake_plus    | SPEECHX        |        8 |
| codecfake_plus    | TACOLM         |        8 |
| codecfake_plus    | TI1G1R         |        6 |
| codecfake_plus    | UNIAUDIO       |       11 |
| codecfake_plus    | USLM           |        5 |
| codecfake_plus    | VALLE          |       57 |
| codecfake_plus    | VIOLA          |        7 |
| dfadd             | GradTTS        |    32700 |
| dfadd             | MatchaTTS      |    32400 |
| dfadd             | NaturalSpeech2 |    32700 |
| dfadd             | PflowTTS       |    33000 |
| dfadd             | StyleTTS2      |    32700 |
| in_the_wild       | unknown        |    11816 |
| llamapartialspoof | cosyvoice      |    21362 |
| llamapartialspoof | elevenlab      |    21738 |
| llamapartialspoof | gptsovits      |    21738 |
| llamapartialspoof | ljjets         |    21738 |
| llamapartialspoof | xttsv2         |    21729 |
| llamapartialspoof | yourtts        |    21738 |
| shiftyspeech      | hifigan        |     5864 |
| shiftyspeech      | vocos          |     5864 |
| shiftyspeech      | wavegrad       |     5864 |
