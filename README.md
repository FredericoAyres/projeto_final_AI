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

_A ser preenchida conforme baixarmos cada banco._
