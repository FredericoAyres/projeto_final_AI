# Resultado da v7: teste mestre fixo e 5 receitas

**Protocolo:** `docs/PROTOCOLO_V7.md`, congelado às 19:36 de 2026-11-13, antes de qualquer treino (`splits_v7.parquet`, SHA256 `24d3aea9…7a54c63`). **Execução:** treino 19:36 a 19:40; escolha pela validação gravada às 19:41:00 (`results/v7_escolha.json`); teste mestre avaliado **uma vez**, às 19:41 a 19:43. Tudo com a regressão logística sobre os vetores já extraídos (XLS-R 300M, camadas 5, 6, 7, média no tempo). Nada foi extraído nem baixado.

Código: `scripts/v7.py`, `scripts/v7_figuras.py`, `scripts/v7_poshoc.py`. Tabelas: `results/v7_*.csv` e `.json`. Figuras: `results/slides/v7_01` a `v7_06`.

> **Leia antes dos números.** (1) Estes números **não são comparáveis** aos da avaliação dos bancos inteiros (13/11) nem aos do desafio RADAR: treinamos com parte do eval do RADAR, o que o desafio pede para não fazer. (2) O teste mestre **não é virgem**: o F0 já tinha sido avaliado nesses clipes em agregado, e o In-the-Wild foi visto da v4 à v6. (3) No RADAR, o idioma do teste (en) é novo, mas **a fonte, os geradores e o canal são os mesmos do treino**. O que o RADAR en mede aqui é "aprender os falsos do RADAR", não "generalizar para geradores novos".

## 1. Escolha pela validação (antes do teste)

AUC por fonte na validação (C escolhido em cada receita pela média; `results/v7_validacao.csv`, `v7_grade_c.csv`):

| receita | C | dfadd | shiftyspeech | codecfake+ | In-the-Wild dev | ML-ITW val | RADAR val (en-sg + vi) | **média** | pior |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| T0 (F0) | 0,1 | 1,000 | 0,997 | 0,896 | 0,983 | 0,769 | 0,779 | 0,904 | 0,769 |
| T1 (+ ML-ITW) | 0,001 | 1,000 | 0,996 | 0,905 | 0,994 | 0,945 | 0,832 | 0,945 | 0,832 |
| T2 (+ RADAR) | 0,01 | 1,000 | 0,986 | 0,934 | 0,990 | 0,893 | 0,972 | 0,963 | 0,893 |
| **T3** (+ os dois) | 0,01 | 1,000 | 0,986 | 0,939 | 0,991 | 0,960 | 0,973 | **0,975** | 0,939 |
| T4 (T3, peso igual por fonte) | 0,01 | 1,000 | 0,981 | 0,937 | 0,989 | 0,961 | 0,968 | 0,973 | 0,937 |

- **Vencedora pela validação: T3** (0,975; a T4 fica a 0,002, dentro do que seria empate se houvesse margem, mas o protocolo não tinha margem).
- Todos os ajustes convergiram (36 a 608 iterações). Em T1 a T4, a validação preferiu **mais freio** (C = 0,001 a 0,01) que o F0 (0,1).
- **Regra de corte** (validação cruzada deixando uma fonte de fora; `v7_cortes_validacao.csv`): T0 e T1 → A'; T2, T3 e T4 → M. **As três regras empatam na prática:** a diferença de acurácia balanceada entre elas fica entre 0,0004 e 0,010, e a escolha é quase um sorteio. No teste, as três aparecem.

## 2. Teste mestre: métrica principal

Média da AUC por fonte em [ML-ITW teste, In-the-Wild 38, RADAR en] e a pior fonte (IC95 por bootstrap, 2.000 repetições, pareado entre receitas; `results/v7_principal.csv`):

| receita | ML-ITW teste (IC por locutor) | In-the-Wild 38 (IC por locutor) | RADAR en (IC por clipe, otimista) | **média (IC95)** | pior fonte (IC95) | média − T0 (IC95) |
|:--|:--|:--|:--|:--|:--|:--|
| T0 | 0,588 (0,487 a 0,686) | 0,979 (0,961 a 0,990) | 0,724 (0,718 a 0,730) | **0,764** (0,730 a 0,798) | 0,588 ML-ITW (0,487 a 0,686) | — |
| T1 | 0,923 (0,873 a 0,959) | **0,994** (0,985 a 0,998) | 0,810 (0,805 a 0,814) | 0,909 (0,892 a 0,921) | 0,810 RADAR (0,805 a 0,814) | +0,145 (0,115 a 0,174) |
| T2 | 0,882 (0,831 a 0,924) | 0,983 (0,969 a 0,992) | 0,955 (0,953 a 0,957) | 0,940 (0,922 a 0,955) | 0,882 ML-ITW (0,831 a 0,924) | +0,177 (0,144 a 0,210) |
| **T3** ★ | 0,939 (0,903 a 0,966) | 0,986 (0,973 a 0,994) | **0,964** (0,962 a 0,965) | **0,963** (0,950 a 0,973) | 0,939 ML-ITW (0,903 a 0,964) | **+0,200** (0,168 a 0,232) |
| T4 | **0,948** (0,915 a 0,971) | 0,985 (0,971 a 0,993) | 0,958 (0,956 a 0,960) | 0,963 (0,952 a 0,972) | 0,948 ML-ITW (0,915 a 0,959) | +0,200 (0,168 a 0,232) |

★ = escolhida pela validação. **Domínio visto × não visto:** no ML-ITW, T1, T3 e T4 viram o banco (outros locutores); no RADAR en, T2, T3 e T4 viram a fonte, os geradores e o canal (idioma novo); o In-the-Wild **nunca** foi visto por nenhuma.

**TPR / FPR (%) na regra escolhida pela validação** (todas as regras: `results/v7_teste_mestre.csv`; figura 4):

| receita (regra) | ML-ITW teste | In-the-Wild 38 | RADAR en |
|:--|:--|:--|:--|
| T0 (A') | 27,8 / 16,5 | 98,9 / 30,8 | 53,4 / 19,6 |
| T1 (A') | 73,9 / 6,2 | 99,2 / 12,2 | 72,8 / 24,3 |
| T2 (M) | 76,1 / 14,5 | 98,2 / 24,7 | 92,0 / 16,0 |
| T3 (M) | 79,2 / 5,8 | 97,8 / 15,6 | 92,2 / 14,0 |
| T4 (M) | 83,8 / 8,2 | 97,5 / 15,1 | 91,5 / 15,0 |

**À parte, como indício (7 / 8 locutores):** ML-ITW em português, AUC T0 0,692 (0,435 a 0,900) → T3 0,852 (0,717 a 0,953) e T4 0,861 (0,727 a 0,956). O IC é largo demais para concluir. Na regra escolhida, a T3 pega só 51,9% dos falsos em português (FPR 4,7%).

## 3. Bloco inglês (foco principal)

AUC (IC95) por receita (`results/v7_bloco_ingles.csv`; figura 6):

| recorte | T0 | T1 | T2 | T3 | T4 |
|:--|:--|:--|:--|:--|:--|
| RADAR en, natural (IC otimista) | 0,724 (0,718 a 0,730) | 0,810 | 0,955 | **0,964** (0,962 a 0,965) | 0,958 |
| RADAR en, pareado por duração | 0,721 | 0,807 | 0,955 | 0,964 | 0,958 |
| ML-ITW só inglês (**indício**: 18 / 26 locutores real / IA) | 0,579 (0,473 a 0,732) | 0,937 | 0,832 | 0,943 (0,899 a 0,974) | **0,959** (0,926 a 0,981) |
| In-the-Wild 38 (nunca visto) | 0,979 (0,961 a 0,990) | **0,994** (0,985 a 0,998) | 0,983 | 0,986 (0,973 a 0,994) | 0,985 |
| LlamaPartialSpoof (controle limpo) | 0,925 | **0,949** | 0,942 | 0,942 | 0,934 |
| ASVspoof 2019 eval (secundário*) | 0,999 | 0,999 | 0,998 | 0,997 | 0,995 |
| ASVspoof 2021 DF (secundário*) | 0,997 | 0,996 | 0,995 | 0,994 | 0,993 |

\* Pode haver locutores em comum com o treino (VCTK). Fica fora da métrica principal.

TPR / FPR (%) nas três regras, T0 → T3: RADAR en A' 53,4/19,6 → 93,0/15,7; M 58,0/24,1 → 92,2/14,0; F5 50,9/17,7 → 92,3/14,1. ML-ITW inglês: A' 17,7/9,5 → 73,0/4,5; M 21,7/11,2 → 70,8/3,8. In-the-Wild: A' 98,9/30,8 → 97,9/17,2; M 99,1/36,2 → 97,8/15,6. LlamaPartialSpoof: A' 74,2/3,7 → 83,8/8,5.

**RADAR en por faixa de duração** (`v7_radar_en_duracao.csv`; real / IA: < 2 s 1.414 / 2.410; 2 a 4 s 4.117 / 5.071; 4 a 6 s 4.491 / 4.375; ≥ 6 s 4.978 / 6.067):

| faixa | T0 | T1 | T2 | T3 | T4 |
|:--|--:|--:|--:|--:|--:|
| < 2 s | 0,709 | 0,849 | 0,912 | 0,945 | 0,939 |
| 2 a 4 s | 0,719 | 0,807 | 0,954 | 0,962 | 0,956 |
| 4 a 6 s | 0,715 | 0,785 | 0,964 | 0,967 | 0,962 |
| ≥ 6 s | 0,737 | 0,821 | 0,963 | 0,971 | 0,967 |

Pareado × natural é igual (diferença ≤ 0,003): **a duração não explica o resultado do RADAR**. Os clipes curtos (< 2 s) continuam os mais difíceis.

## 4. Controles (esquecimento)

`results/v7_controles.csv`; figura 3. AUC, T0 → T1 / T2 / T3 / T4:

| controle | T0 | T1 | T2 | T3 | T4 | TPR / FPR (%) na regra escolhida, T0 → T3 |
|:--|--:|--:|--:|--:|--:|:--|
| LlamaPartialSpoof | 0,925 | 0,949 | 0,942 | 0,942 | 0,934 | 74,2 / 3,7 → 82,9 / 7,5 |
| BRSpeech-DF (pt-BR) | 0,842 | 0,893 | 0,934 | **0,956** | **0,961** | 99,4 / 62,3 → 99,6 / **75,3** |
| CORAA (pt-BR, só reais): FPR | 15,9% | 15,0% | 13,3% | 11,1% | 13,4% | — |
| ASVspoof 2019 eval* | 0,999 | 0,999 | 0,998 | 0,997 | 0,995 | 96,3 / 0,1 → 95,1 / 0,8 |
| ASVspoof 2021 DF* | 0,997 | 0,996 | 0,995 | 0,994 | 0,993 | 99,2 / 10,3 → 98,8 / 13,7 |

**Não há esquecimento relevante em AUC:** a maior queda é −0,004 (ASVspoof 2019 e 2021, na T4). O pt-BR **sobe**: BRSpeech-DF de 0,842 para 0,956 (T3) e 0,961 (T4). Mas a calibração do BRSpeech-DF piora: os reais dele ficam acima do corte (mediana 2,3 na T3), e **75% dos reais são acusados** (62% no T0). No CORAA (outro banco real pt-BR), o falso alarme cai de 15,9% para 11,1%.

## 5. Diagnóstico (não decide nada)

- **Por idioma no RADAR** (`v7_diagnostico.csv`): en-sg (validação) 0,747 → 0,961 (T3); vi (validação) 0,796 → 0,979. Os idiomas de treino (zh, zh-tw, ja) vão a 0,98 a 0,99 em T2 a T4, mas estão **dentro da amostra de treino**.
- **Por plataforma no ML-ITW** (teste mestre; todas são **indício**, < 30 locutores por classe): T0 → T3: bilibili 0,79 → 0,98; douyin 0,72 → 0,99; facebook 0,68 → 0,92; rednote 0,47 → 0,95; tiktok 0,59 → 0,98; **x 0,33 → 0,82**; **youtube 0,45 → 0,84**. X e YouTube continuam os piores em todas as receitas.
- **Deslocamento da nota** (figura 5): no T0, os falsos do ML-ITW têm nota mediana −4,7 (abaixo de qualquer corte) e os do RADAR en +0,9. Na T3, eles sobem para +4,0 e +5,4, e os reais ficam em −4,5 e −3,0. A T3 também **comprime** a escala: os falsos do In-the-Wild vão de 16,4 para 8,6 e os do ASVspoof 2021 de 21,3 para 10,5.
- **Auditoria de atalho, só com metadados** (duração, fração de silêncio, RMS, pico, tamanho; `v7_atalho.csv`): treinado no lado de treino, dá AUC 0,569 no ML-ITW teste e 0,595 no RADAR en (0,591 pareado). Em validação cruzada dentro do teste mestre, 0,647 e 0,624. Nenhum passa de 0,7, e as receitas treinadas ficam em 0,94 a 0,96. **Os ganhos não são explicados por esses metadados.** No T0 do ML-ITW (0,588), porém, os metadados (0,569 a 0,647) chegam praticamente ao nível do modelo.

## 6. Leitura honesta

**O que sobe:**
- **Onde o domínio foi visto, a AUC sobe muito.** ML-ITW teste: 0,59 → 0,94 (T3), com **outros locutores** (o locutor é atômico, 0 em comum entre os lados). RADAR en: 0,72 → 0,96, com um **idioma que não estava no treino**. A média principal vai de 0,764 para 0,963 (+0,200; IC95 de +0,168 a +0,232).
- **O ganho no ML-ITW acontece dentro do mesmo locutor** (pós-hoc, seção 8). Não é efeito de separar locutores de uma classe só.
- **Juntar os dois bancos (T3/T4) é melhor que cada um sozinho** no conjunto: T1 é fraca no RADAR (0,81) e T2 é mais fraca no ML-ITW (0,88).
- **O In-the-Wild, nunca visto, não piora** (0,979 → 0,983 a 0,994). Os ICs se sobrepõem, então não chamo isso de ganho demonstrado.
- **Não há esquecimento** nos testes antigos de estúdio, e o BRSpeech-DF (pt-BR) sobe em AUC (0,84 → 0,96).

**O que não sobe, ou não dá para afirmar:**
- **Isto não é generalização para geradores novos.** No RADAR, os geradores e o canal do teste (en) são os mesmos do treino (zh, zh-tw, ja). No ML-ITW, o tipo de rótulo, as plataformas e o tipo de conteúdo são os mesmos. O único banco realmente fora do treino é o In-the-Wild, e nele a mudança é pequena.
- **T3 × T4:** iguais na média (0,963 × 0,963). O peso igual por fonte não mudou nada demonstrável.
- **Cortes:** a calibração muda entre bancos. Na T3, o FPR na regra escolhida vai de 5,8% (ML-ITW) a 15,6% (In-the-Wild) e **75% no BRSpeech-DF**. Um corte fixo não serve para todos os domínios, e as três regras dão praticamente o mesmo resultado.
- **Português:** só indício (7 a 8 locutores no ML-ITW), e o BRSpeech-DF tem os reais acusados demais.
- **X e YouTube** no ML-ITW seguem os piores (~0,82 a 0,84 na T3), com poucos locutores.

## 7. Limitações

- O teste mestre não é virgem (seção 0 do protocolo). O RADAR en tem IC **otimista** (por clipe, sem locutor). O ML-ITW só inglês e todas as plataformas têm < 30 locutores por classe (**indício**).
- Uma única divisão de locutores (seed 42) e uma única regressão por receita: a variação de sorteio do treino não está medida.
- In-the-Wild 38 = a amostra de 10.000 clipes da v4 (5.000 por classe), não os 21.165 clipes do `test_final`.
- O ML-ITW tem rótulos vindos do contexto da postagem, nomes genéricos de locutor e licença não declarada. O RADAR proíbe usar o eval no treino para resultados do desafio (estes números não valem como resultado do desafio).
- Nenhuma conclusão sobre golpes reais por ligação: nenhum destes bancos é ligação telefônica real.

## 8. Pós-hoc (depois de ver o teste; não muda receita, regra nem escolha)

`scripts/v7_poshoc.py` → `results/v7_poshoc.csv`. Motivo: o teste mestre do ML-ITW tem 55 locutores de uma classe só.

| receita | (a) só locutores com as duas classes: AUC (IC95 por locutor; 36 / 36 locutores, 4.117 / 937 clipes) | (b) AUC dentro de cada locutor (≥ 5 clipes por classe; 32 locutores): média / mediana |
|:--|:--|:--|
| T0 | 0,591 (0,460 a 0,734) | 0,734 / 0,818 |
| T1 | 0,919 (0,851 a 0,967) | 0,897 / 0,978 |
| T2 | 0,860 (0,777 a 0,922) | 0,866 / 0,918 |
| T3 | 0,920 (0,853 a 0,964) | 0,912 / 0,979 |
| T4 | 0,928 (0,868 a 0,970) | 0,917 / 0,986 |

O ganho se mantém com os dois recortes.

## 9. Desvios do protocolo e da tarefa, e problemas

1. **`powercfg` bloqueado** pela permissão desta sessão. No lugar, cada processo da v7 chama `SetThreadExecutionState`, que impede a suspensão enquanto ele roda. Nenhuma configuração do Windows foi mudada, então não há nada a restaurar. Entre um processo e outro, a proteção não existe (intervalos de segundos).
2. **13 clipes excluídos da validação de estúdio** (duplicatas exatas de reais do CodecFake+ que estavam no treino da v4; herança do plano da v4). A salvaguarda parou a 1ª execução da divisão, como previsto. Detalhe no protocolo, seção 2.
3. **Decisões minhas, registradas no protocolo antes do treino:** estratificação da regra 4 pelo idioma principal do locutor; "fonte" da T4 = bloco (estúdio, ML-ITW, RADAR); RADAR val como uma fonte só (en-sg + vi); "ML-ITW teste" na métrica principal inclui os locutores com português; regra 3 vale antes da regra 2; sem margem de empate na escolha do C, da regra e da receita.
4. **O teste rápido (`--quick`)** treinou mini-modelos (300 clipes por bloco e classe) e mediu 5% do teste mestre só para testar o código. Ao conferir o **layout** das figuras do teste rápido, vi números desses mini-modelos. Eles não são as receitas e nada foi escolhido por eles (todas as regras já estavam congeladas às 19:36).
5. **1ª tentativa do treino falhou antes de treinar qualquer coisa:** o monitor de status que eu abri (`tail -f`) segurava o `logs/v7_status.txt`, e a troca atômica do arquivo falhou no Windows. O status passou a ser gravado por acréscimo (append), e o treino rodou de novo do zero, com o mesmo resultado determinístico.
6. **Figuras refeitas depois do teste,** só no layout e no texto dos títulos (o título é a conclusão, como pedido). Nenhum número mudou.
7. **`docs/HISTORICO_VERSOES.md` não foi atualizado:** a tarefa proíbe alterar docs antigos, mas o CLAUDE.md pede registrar cada versão fechada lá. Fica pendente de decisão sua.
8. A extração não foi necessária: todos os vetores já existiam (40.000 / 19.999 / 10.614 / 51.320 / 127.254, camadas 5, 6, 7). A etapa 3 (clones próprios) não rodou, porque `data/raw/clones_proprios` não existe.

**Tempo e memória:** vetores 0,8 min (pico 4,7 GB); divisão e salvaguardas 0,3 min (2,5 GB); treino das 4 receitas × 6 C mais as notas de 249.187 clipes: 4,5 min (4,8 GB); escolha 0,1 min; teste mestre com 2.000 bootstraps 1,7 min (2,3 GB); figuras < 1 min. Total da execução: ~15 min, bem abaixo do orçamento de 3 h. Um processo pesado por vez; nenhum passou de 10 GB.
