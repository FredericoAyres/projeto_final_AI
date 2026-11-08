# projeto_final_AI

Um detector que classifica se uma voz em inglês foi gerada por IA, para alertar contra golpes de clonagem de voz e evitar que ouvintes e sistemas confiem em um áudio que finge ser uma pessoa real.

## Estrutura

```
data/
  raw/
    train_val_en/         asvspoof2019_la, dfadd, shiftyspeech, codecfake_plus
    real_en/              librispeech, vctk
    test_generalization/  in_the_wild, llamapartialspoof, asvspoof2021_df, sonar, spoofceleb
    test_ptbr/            brspeech_df, parlaspoof_br, mlaad_pt, fakebraccent, coraa
  processed/              manifests e splits (gerados depois)
scripts/                  scripts de download e preparo
notebooks/
docs/
  datasets.yaml           registro único dos bancos
```

Os dados não vão para o git. Veja [`data/README.md`](data/README.md).

## Como baixar os dados

Passo a passo para quem acabou de clonar o repositório no Windows (PowerShell). **Todos os comandos rodam na raiz do repo** (a pasta que contém este `README.md`).

### 1. Pré-requisitos

- Python **3.13** (a versão que usamos) e git.
- Cerca de **100 GB livres** em disco.
- Uma conexão estável (o download completo leva em torno de 3 horas).

### 2. Criar o ambiente

```powershell
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

No Mac/Linux, troque a segunda linha por `source env/bin/activate`. Ative o `env` sempre que abrir um terminal novo.
_Não testado num clone novo. O `requirements.txt` foi conferido com `pip install -r requirements.txt --dry-run` no nosso env._

### 3. (Opcional) Login no Hugging Face

```powershell
hf auth login
```

Escolha **"Log in with your browser"**. **Nunca** cole o token no chat nem em arquivos do repositório. O login só é necessário se algum download do Hugging Face for negado. _Não testado._

### 4. Pasta sincronizada? Saia dela

Se o repositório estiver dentro do OneDrive, do Google Drive ou de outra pasta sincronizada, pause a sincronização ou, melhor, clone o repo numa pasta fora dela (ex.: `C:\Projetos`). São centenas de milhares de arquivos.

### 5. Baixar os bancos

**(a) Só o necessário para treinar (~38 GB):**

```powershell
python scripts/download_all.py --only asvspoof2019_la,dfadd,shiftyspeech,codecfake_plus,librispeech --max-gb 100
```

**(b) Tudo, com os bancos de teste (~88 GB, cerca de 3 horas):**

```powershell
python scripts/download_all.py --only asvspoof2019_la,librispeech,in_the_wild,brspeech_df,dfadd,shiftyspeech,codecfake_plus,coraa,llamapartialspoof,asvspoof2021_df --max-gb 100
```

Se a conexão cair, **rode o mesmo comando de novo**: o download continua de onde parou, e os bancos já concluídos são pulados. Para ver o que ele faria sem baixar nada, use `--dry-run`. Ao final, o resumo fica em `logs\download_summary.txt`.
_Os dois comandos, exatamente como estão acima, não foram testados nesta revisão. O `download_all.py` já foi usado para baixar os dados atuais, e os tamanhos citados são os medidos em disco._

### 6. Extrair o BRSpeech-DF

O BRSpeech-DF vem em `.parquet` com o áudio embutido. Este passo grava os áudios em `data/processed/brspeech_df/test/` e pula os que já foram extraídos:

```powershell
python scripts/extract_brspeech.py
```

_Testado (com tudo já extraído: "0 extraidos agora")._

### 7. Conferir

```powershell
python scripts/check_data.py
```

Deve terminar com `OK`. Se faltar algo, o script mostra o comando a rodar. _Testado: OK nos 756.350 áudios._ O manifest `data/processed/manifest.parquet` **já vem no repositório e não precisa ser gerado de novo**. Ele só precisa ser regenerado (`python scripts/build_manifest.py`) se as regras de divisão mudarem.

### 8. Bancos baixados

| banco | papel | tamanho aprox. | licença |
|:--|:--|--:|:--|
| asvspoof2019_la | treino/validação (eval = teste de generalização secundário) | 7,6 GB | ODC-By |
| dfadd | treino/validação | 16 GB | ver página do dataset |
| shiftyspeech | treino/validação | 6,5 GB | ver página do dataset |
| codecfake_plus (só CoSG) | treino/validação | 0,4 GB | ver página do dataset |
| librispeech (train-clean-100, dev-clean, test-clean) | treino/validação (só train-clean-100) | 7,1 GB | CC BY 4.0 |
| in_the_wild | teste de generalização | 8,4 GB | ver página do dataset |
| llamapartialspoof | teste de generalização | 27 GB | ver página do dataset |
| asvspoof2021_df (parte 1 de 4) | teste de generalização (secundário) | 8,7 GB | ver página do dataset |
| brspeech_df (split test) | teste pt-BR | 2,0 GB (+2,0 GB extraído) | Apache-2.0 |
| coraa (split test) | teste pt-BR | 2,5 GB | CC BY-NC-ND 4.0 |

As licenças vêm de [`docs/datasets.yaml`](docs/datasets.yaml), e os tamanhos foram medidos em disco depois do download. Os links de cada banco também estão no `datasets.yaml`.

### 9. O que o script NÃO baixa, e por quê

| banco | motivo |
|:--|:--|
| spoofceleb | acesso restrito (*gated*): exige aceite e e-mail institucional |
| mlaad_pt | pulado por decisão do projeto (acesso restrito, licença não comercial) |
| fakebraccent | pulado por decisão do projeto (exige login no Kaggle) |
| sonar | só existe como pasta do Google Drive, sem link direto: baixar pelo navegador |
| vctk | URL a confirmar no Edinburgh DataShare (os reais do VCTK já vêm no dfadd) |
| parlaspoof_br | os arquivos misturam todos os subconjuntos, então não dá para baixar só o de avaliação |
| Common Voice, VoxCeleb | exigem cadastro e aceite de termos |

Duas partes ficaram de fora **de propósito**: o **CoRS** do codecfake_plus (um único `.tar.xz` em partes de ~25 GB que só extrai inteiro; baixamos só o CoSG e os rótulos) e **3 das 4 partes** do asvspoof2021_df (por espaço; a parte 1 já tem 152.955 áudios).

### 10. Avisos

- **Nunca** rode `git add .` sem conferir antes o `git status`.
- Os dados **não vão para o git** (ver `.gitignore`). Só o manifest é versionado.
- Alguns bancos têm licença **não comercial** ou **sem derivados** (ex.: CORAA, CC BY-NC-ND). Use-os só para o projeto e **não redistribua os arquivos**.

### 11. Limitações dos dados

Os detalhes estão em [`docs/audit_report.md`](docs/audit_report.md). Em resumo:
- **Vazamento de locutor tratado:** os locutores do LlamaPartialSpoof (que vêm do LibriSpeech dev/test-clean) foram tirados do treino, inclusive as cópias no ShiftySpeech.
- **VCTK compartilhado:** o asvspoof2019_la (eval) e o asvspoof2021_df podem ter locutores em comum com o treino (via DFADD). São testes **secundários**: valem para geradores e ataques novos, não para locutores novos.
- **BRSpeech-DF:** os reais estão em 48 kHz e os fakes em 24 kHz. Reamostre tudo no carregamento, senão o modelo pode separar as classes só pela taxa de amostragem.
