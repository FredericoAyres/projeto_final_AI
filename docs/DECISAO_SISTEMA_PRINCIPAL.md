# Decisão: sistema principal = v7 (T3); v8 = experimento não adotado

**Data:** 15/11/2026. **Quem decidiu:** o grupo. **Natureza:** decisão **posterior** ao protocolo da v8 (`docs/PROTOCOLO_V8.md`). Ela não muda o protocolo, os resultados nem nenhum arquivo de `results/`.

Todos os números abaixo foram lidos de `results/` (o arquivo de cada um está indicado).

## 1. O que é o sistema principal

O sistema principal é a **v7, receita T3**. Ele faz quatro coisas com cada áudio:

1. **Prepara o áudio** como na v4: 16 kHz, corta o silêncio das bordas e usa no máximo 8 s do meio.
2. **Transforma o áudio em números** com o XLS-R 300M (modelo de fala pronto, da Meta), **congelado**: nenhum peso dele é treinado por nós. Tira a média no tempo das saídas das camadas 5, 6 e 7 e junta tudo em 3.072 números.
3. **Padroniza** esses números (subtrai a média e divide pelo desvio do treino da T3).
4. **Dá uma nota** com uma regressão logística (C = 0,01, classes balanceadas). Nota em logit: quanto maior, mais parece voz de IA. Acima de um corte, a voz é chamada de gerada por IA.

A regressão foi treinada com 81.288 áudios: estúdio (20.000 reais e 20.000 de IA, os da v4), internet real do ML-ITW (7.802 reais e 2.854 de IA, separados por locutor) e IA moderna em canal degradado do RADAR (15.316 e 15.316, em chinês, chinês de Taiwan e japonês, com durações pareadas). Fonte: `results/v7_t3_modelo.json`.

**Resultado no teste mestre** (`results/v7_principal.csv`; o mesmo valor aparece como R0 em `results/v8_principal.csv`): a média da AUC em internet real (ML-ITW), entrevistas (In-the-Wild) e RADAR en foi de **0,764 (v4, F0)** para **0,963 (T3)**, IC95 0,950 a 0,973. Por fonte: ML-ITW 0,939, In-the-Wild 0,986, RADAR en 0,964.

## 2. Os arquivos que definem o sistema principal

| arquivo | o que tem |
|:--|:--|
| `results/v7_t3_coeficientes.npz` | os pesos: `coef` (3.072), `intercepto` (1), `mu` e `sd` da padronização (3.072 cada), `C` e `camadas` |
| `results/v7_t3_modelo.json` | como usar (`nota = ((x - mu) / sd) @ coef + intercepto`), hiperparâmetros, composição do treino, cortes da validação e a conferência dos pesos |
| `docs/PROTOCOLO_V7.md` | o protocolo da v7, congelado antes do treino |
| `docs/RESULTADO_V7.md` | o resultado da v7 |
| `results/v7_*.csv` e `results/v7_*.json` | as tabelas da v7 (validação, teste mestre, controles, cortes) |
| `scripts/v7.py`, `scripts/v7_t3_exportar.py` | o código da v7 e a exportação dos pesos |
| `notebook.ipynb` (cópia em `notebooks/versions/principal_v7.ipynb`) | o documento do sistema principal, gerado por `scripts/nb_principal.py` |

Para usar, também é preciso o XLS-R 300M original (`facebook/wav2vec2-xls-r-300m`, Hugging Face) e o pipeline de áudio da v4 (`src/preproc.py`).

## 3. A decisão

**O sistema principal é a v7 (T3). A v8 (ajuste fino dos blocos 5 a 7 do XLS-R, receita R1) fica como experimento não adotado.**

### O que a regra da v8 escolheu (sem esconder)

Pela regra do protocolo da v8 (maior média da AUC por fonte na validação), a vencedora foi a **R1** (camadas 5, 6 e 7 + cabeça, cabeça começando na T3). A escolha foi gravada em `results/v8_escolha.json` às 19:30:16 de 14/11/2026, antes de qualquer teste:

| receita | média da AUC por fonte na validação |
|:--|--:|
| R0 (= T3) | 0,9750 |
| **R1** (média das sementes 42 e 123) | **0,9758** |
| R2 (só a camada 7) | 0,9756 |
| R3 (cabeça aleatória) | 0,9723 |

**A regra apontou uma variante do ajuste fino (R1). O grupo decidiu não adotá-la.** Essa é uma decisão posterior ao protocolo, tomada depois de ver o teste mestre da v8. O protocolo e os resultados ficam como estão, e a v8 continua reportada inteira (`docs/HISTORICO_VERSOES.md`, `results/v8_*`, anexo do notebook).

### Os motivos (com os números)

1. **O ganho é pequeno** (`results/v8_principal.csv`). No teste mestre, a média da AUC foi de 0,9631 (T3) para 0,9671 (R1, semente 42) e 0,9680 (semente 123). As diferenças pareadas são +0,0040 (IC95 +0,0025 a +0,0056) e +0,0049 (+0,0030 a +0,0071); na média das duas sementes, +0,0044. Na pior fonte (ML-ITW), o IC pareado começa abaixo de zero: −0,0004 (semente 42) e −0,0006 (semente 123).
2. **Só há 2 sementes da R1.** As duas diferem pouco (validação 0,9757 e 0,9760, em `results/v8_validacao.csv`; teste 0,9671 e 0,9680), mas duas sementes não medem a variação do treino, e não há tempo para mais. O ganho passa do ruído de sortear o teste, mas com 2 sementes não dá para afirmar que passa do ruído do treino.
3. **O corte da validação variou muito entre as sementes, e a calibração mudou** (`results/v8_escolha.json`). Corte escolhido (logit): T3 regra M = 0,234; R1 semente 42 regra A' = 0,713; R1 semente 123 regra A' = −0,024. As outras regras também variam (M: 0,721 × 0,605; F5: 0,827 × 0,114). A escala da nota mudou (`results/v8_teste_mestre.csv`): a nota mediana dos reais do ASVspoof 2019 foi de −5,8 (T3) para −9,4 e −9,3, e a do In-the-Wild de −2,8 para −4,0 e −4,5.
4. **Os pesos ajustados não estão no repositório** (`results/v8_r1_pesos.json`). São 2 arquivos de 72,1 MB (37.791.745 parâmetros cada), numa release do GitHub (`v8-pesos-r1`). A T3 são 3.073 números de regressão (3.072 coeficientes + o intercepto), mais a média e o desvio da padronização, num arquivo pequeno dentro do git.
5. **Custo e reprodução.** A regressão da T3 treina sobre vetores já extraídos, com solver determinístico (lbfgs); refeita, ela reproduz as notas gravadas com diferença máxima de 1,1e-5 de logit (`results/v7_t3_modelo.json`). O ajuste fino levou ~23 min de treino por rodada (`results/v8_treino_R1_s42.json`: 694 + 703 s; `R1_s123`: 705 + 705 s), mais ~25 min por validação inteira e ~30 min por teste; a fila inteira levou ~11 h (`docs/HISTORICO_VERSOES.md`). **Atenção:** o tempo de treino da regressão da T3 ("segundos") não está registrado em `results/`; ver as dúvidas no fim.
6. **O problema do corte em português não melhorou de forma confiável** (`results/v8_controles.csv`). % dos reais do BRSpeech-DF (português de audiolivro) acusados como IA, na regra escolhida de cada rodada: T3 75,2%; R1 semente 42 74,2%; R1 semente 123 81,5%. No CORAA (só reais, português), 11,1% (T3) para 9,4% e 9,6%. Uma semente melhora um pouco e a outra piora no BRSpeech-DF.

## 4. O que NÃO afirmamos

- **O teste mestre não é virgem.** O F0 já tinha sido avaliado nesses clipes, a T3 foi avaliada nele na v7, e quem desenhou a v8 conhecia esses números. Só clones próprios, gravados depois de fechar o sistema, seriam um teste intocado.
- **O RADAR en mede idioma novo, não gerador novo.** Os geradores e o canal do RADAR en são os mesmos do treino (chinês e japonês). O 0,964 diz que o sistema reconhece esses falsos em outro idioma, não que ele pega geradores que nunca viu.
- **O corte não se transfere entre ambientes.** Com a mesma regra, a T3 acusa de ~1% dos reais (ASVspoof 2019, 0,8%) a 75% (BRSpeech-DF), conforme o banco (`results/v7_teste_mestre.csv` e `results/v8_controles.csv`). No português de audiolivro, a maioria dos reais é acusada.
- **Nada sobre ligações ou WhatsApp reais.** Nenhum banco é de ligação telefônica ou de aplicativo de mensagem real: o RADAR simula o canal, e o ML-ITW são vídeos de redes sociais.
- O português de internet tem poucos locutores (ML-ITW em português: 7 reais / 8 de IA): é indício, não estimativa.

## 5. O que mudaria a decisão

- **Mais sementes da R1 (3 ou mais)** confirmando o ganho sobre a T3, com o IC pareado acima de zero, e
- **um corte estável** entre essas sementes, escolhido só pela validação.

Com isso, a R1 poderia virar o sistema principal, registrada em `docs/HISTORICO_VERSOES.md` antes de qualquer novo teste.
