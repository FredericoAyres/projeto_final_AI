# data/

Os dados **não vão para o git** (ver `.gitignore`); só as pastas (via `.gitkeep`) e este README são versionados.

Cada pasta de `data/raw/` corresponde a um item de [`docs/datasets.yaml`](../docs/datasets.yaml), que é o registro único dos bancos (URL, licença, acesso, status e `local_path`).

## Manifest (gerado depois em `data/processed/`)

Um CSV com uma linha por áudio e estas colunas:

| coluna | descrição |
|---|---|
| `path` | caminho do arquivo de áudio |
| `label` | 0 = real, 1 = fake |
| `source_dataset` | nome do banco (igual ao `name` em `datasets.yaml`) |
| `generator` | sistema/vocoder que gerou o áudio (vazio se real) |
| `speaker_id` | identificador do locutor |
| `language` | `en` ou `pt` |
| `split` | `train`, `val` ou `test` |

## Regras

- Os splits de treino/validação são separados **por locutor ou por gerador, nunca por arquivo**.
- Os bancos de teste (`test_generalization` e `test_ptbr`) **nunca entram no treino**.
