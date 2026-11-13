# Protocolo de avaliação: RADAR eval e ML-ITW

**Congelado em 2026-11-13, 14:58 (UTC−3), ANTES de qualquer avaliação.** Nenhum áudio destes bancos foi passado por modelo, e nenhum embedding foi extraído. Dados e contagens: `docs/dados_teste_internet.md`. Depois deste carimbo, nada abaixo pode mudar com base em resultado; qualquer mudança vira um adendo datado, com o motivo, e os números anteriores continuam reportados.

## 1. Modelo congelado (F0 = v4)

- Front-end: `facebook/wav2vec2-xls-r-300m` congelado, **camadas 5, 6 e 7**, média no tempo, concatenadas (`concat_top3`).
- Cabeça: **regressão logística, C = 0,1**, treinada nos **40.000 áudios** da v4. Os pesos e o cache de treino são os da v4 (`results/v4_*`, `data/processed/v4_cache/`); **nada é treinado ou reajustado** com os dados novos.
- Nota: logit da regressão logística (maior = mais falso). A nota nunca é invertida com base no teste.
- **Cortes de decisão (logit), fixados na validação da v4:** A = **−0,0395** (ponto de EER) e B = **8,2184**. Não há corte novo.
- Entrada: áudio como está nos bancos (16 kHz mono), pelo mesmo pré-processamento da v4. Sem reamostragem ou normalização diferente da v4.

## 2. Uso dos dados

- **Só teste, avaliação única.** Nenhum hiperparâmetro, camada, corte ou modelo é escolhido, e nada é treinado, com estes dados. Eles não entram em validação nem em dev_shift (`eval_subset="test_internet"`).
- Cada banco é avaliado **uma única vez**; todos os resultados são reportados, bons ou ruins.
- O **dev do RADAR fica de fora**: vem de LlamaPartialSpoof e LibriTTS, que se sobrepõem aos nossos dados.
- Se algum áudio for ilegível na avaliação, ele é contado e reportado, não descartado em silêncio.

## 3. Métricas

Calculadas **por fonte** (RADAR eval; ML-ITW), e dentro de cada fonte **por idioma** (RADAR: en, en-sg, zh, zh-tw, ja, vi) e **por plataforma** (ML-ITW: youtube, bilibili, x, facebook, rednote, tiktok, douyin). **Nunca** se juntam reais de uma fonte com falsos de outra, nem os dois bancos em uma métrica só.

Para cada recorte:
- **AUC e EER** (principais; não dependem de corte nem da proporção real/falso).
- **Acurácia e acurácia balanceada** (média dos recalls de real e de falso) nos cortes A e B, sempre ao lado da acurácia trivial ("sempre a classe majoritária").
- **TPR (falso detectado) e FPR (real chamado de falso)** nos cortes A e B.
- Recortes sem as duas classes, ou com poucos áudios, não geram AUC/EER; aparecem só como contagem.

## 4. Incerteza

- **ML-ITW:** IC95 por **bootstrap por locutor** (reamostra locutores, 1.000 repetições, seed 42), pois áudios do mesmo locutor não são independentes.
- **RADAR:** o arquivo não tem locutor. IC95 por **bootstrap por clipe** (mesma seed e repetições), **marcado como OTIMISTA** (clipes do mesmo locutor ou da mesma frase-fonte contam como independentes). Nunca apresentar como se fosse por locutor.
- **Fontes ou recortes com menos de 30 locutores por classe são indício, não estimativa.** No ML-ITW, só chinese (69 locutores) e english (61) passam de 30 locutores no total, e a contagem por classe é menor; os demais idiomas, inclusive `portuguese` (8 locutores), são indício. Por plataforma, só rednote (52), facebook (36), youtube (31) passam de 30 no total (por classe, conferir na hora). No RADAR não dá para contar locutores, então todo o banco leva a ressalva do IC otimista.

## 5. Auditoria de atalho (planejada)

Para checar se o sinal vem de metadados e não da voz: um classificador simples (regressão logística / árvore rasa) **só com metadados**: duração, taxa de amostragem, canais, codec/extensão, tamanho do arquivo e nível do sinal (RMS/pico), prevendo real × falso dentro de cada fonte, com validação cruzada por locutor no ML-ITW e por clipe no RADAR. Se esse classificador chegar perto do AUC do F0, o resultado do F0 naquele banco é suspeito de atalho. Estado atual (só cabeçalhos, sem rodar o classificador): o formato é idêntico entre classes (FLAC 16 kHz mono no RADAR; WAV 16 kHz mono no ML-ITW), mas a **duração difere** (RADAR: falsos mais longos; ML-ITW: sem corte abaixo de 2 s) e a proporção real/falso varia muito por plataforma no ML-ITW.

## 6. Leitura pré-definida

- Reportar os números de todos os recortes, com IC95, sem escolher o que mostrar depois do resultado.
- Comparar com o desempenho da v4 nos testes anteriores (In-the-Wild AUC 0,979; ver `docs/HISTORICO_VERSOES.md`) apenas descritivamente.
- ML-ITW e In-the-Wild têm locutores em comum (Scarlett Johansson, Trump, Obama, por nome): citar na interpretação.
- Resultado ruim em idioma fora do treino (treino em inglês) é esperado e é resultado, não motivo para ajustar o modelo.
