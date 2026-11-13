# Dados de teste da internet: RADAR eval e ML-ITW

Gerado em 2026-11-13. Download: `scripts/download_teste_internet.py`. Manifesto: `scripts/build_manifest_teste_internet.py` → `data/processed/manifest_teste_internet.parquet` (127.254 linhas; fora do git, é gerado de novo pelo script). Uso previsto: **só teste** (ver `docs/PROTOCOLO_TESTE_INTERNET.md`). Nenhum áudio foi avaliado e o `manifest.parquet` da v4 (SHA f2309eb3…) não foi tocado.

## Resumo

| | RADAR eval | ML-ITW |
|---|---|---|
| Áudios | 102.725 | 24.529 |
| Reais / falsos | 50.000 (48,7%) / 52.725 (51,3%) | 18.168 (74,1%) / 6.361 (25,9%) |
| Tamanho em disco | 18,35 GB (9,57 GB de áudio + 8,75 GB do .tar.gz mantido) | 3,27 GB (parquets) + 3,27 GB de áudios extraídos pelo manifesto |
| Locutores | **sem informação no arquivo** | 180 (143 com áudio real, 160 com falso) |
| Idiomas | 6 (en, en-sg, zh, zh-tw, ja, vi) | 14 |
| Duração total | 155,2 h | 28,39 h |

## RADAR Challenge 2026 (só o conjunto de avaliação)

- **Origem:** Zenodo 22110815 (DOI 10.5281/zenodo.22110815), publicado em 2026-08-27. Artigo: arXiv 2605.09568. Acesso aberto.
- **Licença:** CC BY 4.0 (`LICENSE.txt` na pasta).
- **Arquivos baixados (md5 conferido):** `RADAR2026-eval.tar.gz` (8.752.649.011 B), `label_RADAR2026-eval.txt`, `lang_RADAR2026-eval.txt`, `README.txt`, `LICENSE.txt`. O **dev (`RADAR2026-dev.tar.gz`, 5,1 GB) não foi baixado**: vem de LlamaPartialSpoof e LibriTTS, que se sobrepõem aos nossos dados.
- **Contagens e diferença do esperado:** 102.725 áudios, não 102.726. O README do Zenodo explica: o `RADAR2026-EVAL068818` está corrompido e foi excluído do arquivo de áudio, do de rótulos e do de idiomas. Conferi que o id não existe nos arquivos (o `lang` pula de 068817 para 068819). A diferença cai em **ja/falso: 8.741 contra 8.742 esperado** (e, por consequência, falsos 52.725 contra 52.726). Todas as outras células bateram.
- **Pegadinha de extração:** o tar traz 102.725 arquivos `._*` (metadados do macOS, não são áudio) ao lado dos FLAC. O manifesto parte dos rótulos, então os ignora. Nada foi apagado.

| idioma | real | falso | % falso |
|---|---:|---:|---:|
| en | 15.000 | 17.923 | 54,4 |
| en-sg | 3.500 | 6.711 | 65,7 |
| zh | 11.500 | 11.234 | 49,4 |
| zh-tw | 3.500 | 1.678 | 32,4 |
| ja | 9.500 | 8.741 | 47,9 |
| vi | 7.000 | 6.438 | 47,9 |

- **Duração (s):** real min 0,38 · mediana 4,28 · máx 66,5; falso min 0,06 · mediana 5,28 · máx 109,9; geral min 0,06 · mediana 4,72 · máx 109,9. **Menos de 2 s: 8,9% (real 8,2%, falso 9,6%).** Falsos são um pouco mais longos (mediana +1 s; tamanho médio do arquivo 99,9 kB contra 86,1 kB).
- **Formato:** todos os 102.725 arquivos são FLAC, PCM_16, 16 kHz, mono, nas duas classes e em todos os idiomas. **Não há diferença de formato entre classes no cabeçalho.** Ressalva: o README diz que o eval passou por transformações de mídia (codec, reamostragem, ruído, música, sala, fades, silêncio, compressão de faixa dinâmica, limitação de banda) antes de virar FLAC. Vestígios disso ficam no sinal, não no cabeçalho; a auditoria de atalho (protocolo) precisa olhar o sinal.
- **Locutores e gerador:** o protocolo só traz id, duração, rótulo e segmentos. Não há locutor nem gerador (`speaker_id`, `generator` nulos). O IC95 por clipe será otimista.
- **Domínio:** o eval é multilíngue e com transformações de mídia, parte delas ausente até do dev do desafio. É um teste de robustez, não do mesmo domínio do treino.

## ML-ITW

- **Origem:** Hugging Face, **`nccm2p2/ML-ITW`** (split `eval`, 4 parquets, 3.266.883.799 B; sha256 conferido). ⚠️ O repositório pedido no enunciado, `chlo1/ML-ITW`, **não existe** (404). Escolhi o `nccm2p2/ML-ITW` porque bate exatamente com o esperado: 24.529 áudios, 18.168 reais, 6.361 falsos, 180 locutores, 14 idiomas, 28,39 h, e tem as colunas `speaker`, `language`, `platform`, `label`. **Precisa da sua confirmação** de que é o banco certo (autoria do repositório e artigo de referência não foram verificados por mim).
- **Acesso:** aberto, sem login nem aceite de termos (`gated: False`).
- **Licença:** **não declarada** no card do repositório. Confirmar antes de redistribuir qualquer coisa.
- **Contagens:** bateram com o esperado em tudo (total, reais, falsos, locutores, idiomas, horas). 0 arquivos ilegíveis.
- **Duração (s):** real min 2,00 · mediana 3,59 · máx 50,7; falso min 2,01 · mediana 3,49 · máx 34,6. **Menos de 2 s: 0%** (o banco já veio filtrado em ≥ 2 s).
- **Formato:** todos WAV, PCM_16, 16 kHz, mono, nas duas classes. **Sem diferença de formato entre classes.** Ressalva: o banco já foi padronizado pelos autores; codec e taxa originais das plataformas não estão preservados.
- **Locutores:** 180 no total; 143 têm áudio real e 160 têm falso; 123 aparecem nas duas classes e 57 em uma só. Alguns "locutores" são nomes genéricos (ex.: `dontknow`, `olg`, `rm`), então a contagem pode não ser de pessoas distintas.

| idioma | real | falso | locutores |
|---|---:|---:|---:|
| chinese | 5.399 | 3.074 | 69 |
| english | 6.931 | 1.598 | 61 |
| japanese | 1.292 | 277 | 14 |
| portuguese | 911 | 108 | 8 |
| turkish | 412 | 176 | 15 |
| korean | 818 | 110 | 6 |
| german | 704 | 356 | 6 |
| spanish | 480 | 166 | 5 |
| russian | 329 | 44 | 5 |
| hindi | 292 | 358 | 3 |
| hungarian | 296 | 3 | 1 |
| french | 195 | 61 | 3 |
| ukrainian | 73 | 17 | 4 |
| hebrew | 36 | 13 | 1 |

| plataforma | real | falso | % falso | locutores |
|---|---:|---:|---:|---:|
| youtube | 5.948 | 531 | 8,2 | 31 |
| bilibili | 4.263 | 793 | 15,7 | 14 |
| x | 3.046 | 1.061 | 25,8 | 28 |
| facebook | 3.027 | 1.129 | 27,2 | 36 |
| rednote | 1.333 | 1.393 | 51,1 | 52 |
| tiktok | 369 | 305 | 45,3 | 16 |
| douyin | 182 | 1.149 | 86,3 | 15 |

A proporção real/falso varia muito por plataforma (douyin 86% falso, youtube 8%). Por isso as métricas são por plataforma e a acurácia sozinha engana. Idiomas com 1 a 6 locutores (hebrew, hungarian, french, ukrainian, hindi, russian, spanish, german, korean) e `portuguese` (8) têm menos de 30 locutores: indício, não estimativa. **Portuguese tem 911 reais e 108 falsos, mas só 8 locutores.**

### Sobreposição de locutores com o In-the-Wild (54 locutores, `manifest.parquet`)

Comparação por nome (sem acento, sem espaço, sem maiúscula). Só nomes: não confirmei que é a mesma pessoa nem que há áudio em comum.

| ML-ITW | In-the-Wild | tipo |
|---|---|---|
| `ScarlettJohansson` | `Scarlett Johansson` | nome completo igual |
| `Trump` | `Donald Trump` | só sobrenome |
| `obama` | `Barack Obama` | só sobrenome |

São 3 de 54 locutores do In-the-Wild. O treino do F0 não usa o In-the-Wild (ele é teste), então isso não é vazamento de treino. Mas o In-the-Wild já foi avaliado e o ML-ITW tem essas mesmas pessoas: os dois testes não são totalmente independentes quanto a locutores, e isso deve ser citado ao interpretar o ML-ITW. O áudio é de origem diferente (plataformas sociais contra o conjunto do In-the-Wild), mas não conferi duplicatas de gravação.

## Manifesto

Colunas do manifesto atual (`path`, `label` int8 0=real/1=falso, `source_dataset`, `generator`, `fake_type`, `speaker_id`, `language`, `original_split`, `role`, `split`, `duration_s`, `sample_rate` Int32, `eval_subset`) mais `fonte`, `label_nome` (real|fake), `plataforma`, `canais`, `codec` (formato/subtipo), `extensao`, `tamanho_bytes`, `readable`, `error`. Mapeamento dos nomes pedidos: `idioma`=`language`, `locutor`=`speaker_id`, `gerador`=`generator`, `duração`=`duration_s`, `sr_original`=`sample_rate` (taxa do arquivo, sem reamostrar). `role="test_internet"`, `split="test"`, `eval_subset="test_internet"` (nunca cai em treino nem em dev_shift). Caminhos relativos à `--raiz`.

Os áudios do ML-ITW são extraídos dos parquets para `<raiz>/ml_itw/audio/NNNNNN.wav` pelo script do manifesto (bytes copiados sem recodificar), o que ocupa mais 3,3 GB.
