# Resultado: F0 (v4) nos bancos da internet (RADAR eval e ML-ITW)

**Avaliado uma única vez em 2026-11-13 (15:15 a 18:40), depois do protocolo congelado às 14:58** (`docs/PROTOCOLO_TESTE_INTERNET.md`). Nada foi treinado, reajustado ou escolhido com estes dados: o modelo, as camadas, o C e os cortes A e B são os da v4. Dados: `docs/dados_teste_internet.md`.

Código: `scripts/teste_internet.py` (etapas `validar`, `extrair`, `notas`, `metricas`) e `scripts/teste_internet_figuras.py`. Tabelas: `results/teste_internet_por_fonte.csv`, `_por_idioma.csv`, `_por_plataforma.csv`, `_sensibilidade.csv`, `_auditoria_atalho.csv`, `results/teste_internet_resumo.json` (e `teste_internet_validacao_f0.json`, `teste_internet_vazao.json`). Figuras: `results/slides/ti_01_auc_recortes.png` a `ti_05_atalho.png`. Fora do git: o cache `data/processed/teste_internet_cache/` (~13 GB) e as notas por clipe `data/processed/teste_internet_notas.parquet`.

## 1. O que foi feito

1. **F0 conferido antes de usar.** Pesos de `results/v4final_f0_coeficientes.npz` (chaves `coef`, `intercepto`, `camadas`, `C`, `limiar_A`, `limiar_B`). O npz **não traz a padronização**; a média e o desvio vieram de `results/v4_parcial/padronizacao.npz` (o mesmo arquivo que `scripts/v4final_figuras.py` usa; treino da v4, n = 40.000). Aplicado aos caches da v4, o F0 do disco reproduziu as notas gravadas em `results/v4_teste_notas.parquet` em todos os 8 conjuntos (validação, dev_shift e os 6 testes): **diferença máxima de nota 1,8e-5** (limite 1e-4), **AUC igual com 3 casas** em todos, cortes recalculados A = −0,0395 e B = 8,2184.
2. **Extração** com o pipeline da v4 sem mudança de método (`src/preproc.py`, `src/ssl.py`, `src/extracao_v4.py`, `src/cache_v4.py`; hiperparâmetros lidos da célula `config` do `notebook.ipynb`): mono, 16 kHz, corte de silêncio de 35 dB (desfeito se sobrar < 0,5 s), até 8 s do meio, sem completar, fp16. 65 shards de até 2.000 clipes, com marcador por shard. **127.254 clipes, 0 ilegíveis.**
3. **Teste rápido antes:** 200 clipes (50 de cada fonte e classe), com uma queda forçada depois do 1º shard e a retomada. A retomada pulou o shard pronto, e os 4 shards ficaram **idênticos bit a bit** aos de uma execução sem interrupção. As etapas de notas e métricas rodaram nesses 200 clipes só para testar o código; os números não foram lidos. A vazão foi medida em 500 clipes: 30 clipes/s, com estimativa de ~71 min para tudo.
4. **Notas e métricas** pelo protocolo, **por fonte** (nunca juntando reais de uma fonte com falsos de outra). IC95 por bootstrap com 2.000 repetições e seed 42: no ML-ITW, **por locutor** (reamostra locutores com reposição dentro de cada classe); no RADAR, **por clipe**, marcado como **otimista** porque o arquivo não informa o locutor.

## 2. Resultado por fonte

Corte A = −0,0395 (ponto de EER da validação da v4, estúdio); corte B = 8,2184 (ponto de EER do dev_shift, internet). TPR = falsos detectados; FPR = reais acusados como IA.

| fonte | n real / IA | AUC (IC95) | EER | acurácia A / B (chute) | acurácia balanceada A | acurácia balanceada B | TPR / FPR em A | TPR / FPR em B |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **RADAR eval** (IC otimista, por clipe) | 50.000 / 52.725 | **0,721** (0,718 a 0,725) | 33,9% | 0,664 / 0,593 (0,513) | 0,667 (0,664 a 0,669) | 0,603 (0,601 a 0,605) | 56,9% / 23,6% | 24,7% / 4,2% |
| **ML-ITW** (IC por locutor; 143 / 160 locutores) | 18.168 / 6.361 | **0,689** (0,613 a 0,759) | 35,5% | 0,745 / 0,754 (0,741) | 0,634 (0,578 a 0,690) | 0,556 (0,514 a 0,606) | 40,4% / 13,5% | 14,5% / 3,2% |

IC de TPR e FPR (mesmo bootstrap): RADAR A 56,4 a 57,3% / 23,2 a 23,9%; B 24,3 a 25,1% / 4,0 a 4,3%. ML-ITW A 31,7 a 50,0% / 8,3 a 20,5%; B 7,3 a 23,0% / 0,6 a 7,8%.

**Corte de EER da própria fonte (só diagnóstico de calibração; não escolhe nada):** RADAR −2,33 (acurácia balanceada 0,661); ML-ITW −5,94 (0,645). Os dois ficam **abaixo** do corte A e muito abaixo do B. Nos dois bancos, os cortes da v4 estão "do lado conservador": acusam poucos reais e deixam passar muitos falsos.

No ML-ITW, a **acurácia** (0,745 em A, 0,754 em B) fica praticamente no chute de "sempre real" (0,741), porque 74% do banco é real: aqui a acurácia sozinha engana.

## 3. Por idioma e por plataforma

**RADAR (IC por clipe, otimista):**

| idioma | n real / IA | AUC (IC95) | acurácia balanceada A / B | TPR / FPR em A |
|:--|:--|:--|:--|:--|
| en | 15.000 / 17.923 | 0,724 (0,718 a 0,730) | 0,669 / 0,581 | 54,4% / 20,5% |
| en-sg | 3.500 / 6.711 | 0,747 (0,738 a 0,757) | 0,689 / 0,657 | 70,0% / 32,3% |
| zh | 11.500 / 11.234 | 0,695 (0,687 a 0,702) | 0,639 / 0,582 | 51,4% / 23,5% |
| zh-tw | 3.500 / 1.678 | 0,665 (0,649 a 0,681) | 0,606 / 0,557 | 47,0% / 25,7% |
| ja | 9.500 / 8.741 | 0,692 (0,684 a 0,699) | 0,640 / 0,581 | 51,2% / 23,1% |
| vi | 7.000 / 6.438 | 0,796 (0,788 a 0,803) | 0,723 / 0,689 | 69,8% / 25,3% |

**ML-ITW por idioma (IC por locutor).** Só chinês e inglês têm ≥ 30 locutores por classe; os outros são **indício, não estimativa**. Húngaro (3 falsos), ucraniano (17) e hebraico (13) têm menos de 20 clipes em uma classe: só contagem.

| idioma | n real / IA | locutores real / IA | AUC (IC95) | acurácia balanceada A / B | marca |
|:--|:--|:--|:--|:--|:--|
| chinês | 5.399 / 3.074 | 52 / 62 | 0,679 (0,518 a 0,816) | 0,638 / 0,536 | estimativa |
| inglês | 6.931 / 1.598 | 42 / 50 | 0,652 (0,525 a 0,771) | 0,595 / 0,524 | estimativa |
| japonês | 1.292 / 277 | 11 / 10 | 0,623 (0,240 a 0,819) | 0,635 / 0,546 | indício |
| **português** | 911 / 108 | 7 / 8 | **0,692 (0,435 a 0,900)** | 0,692 / 0,687 | indício |
| coreano | 818 / 110 | 6 / 5 | 0,652 (0,483 a 0,805) | 0,599 / 0,553 | indício |
| alemão | 704 / 356 | 5 / 5 | 0,756 (0,612 a 0,909) | 0,593 / 0,551 | indício |
| espanhol | 480 / 166 | 3 / 4 | 0,315 (0,137 a 0,760) | 0,398 / 0,481 | indício |
| turco | 412 / 176 | 10 / 14 | 0,588 (0,300 a 0,817) | 0,485 / 0,492 | indício |
| russo | 329 / 44 | 3 / 3 | 0,764 (0,274 a 1,000) | 0,625 / 0,533 | indício |
| híndi | 292 / 358 | 3 / 3 | 0,856 (0,296 a 0,971) | 0,764 / 0,788 | indício |
| francês | 195 / 61 | 2 / 3 | 0,855 (0,622 a 0,949) | 0,541 / 0,516 | indício |
| húngaro, ucraniano, hebraico | 296 / 3; 73 / 17; 36 / 13 | — | — | — | só contagem |

**ML-ITW por plataforma (IC por locutor).** Só a rednote tem ≥ 30 locutores por classe; as outras são indício.

| plataforma | n real / IA | locutores real / IA | AUC (IC95) | acurácia balanceada A / B | chute |
|:--|:--|:--|:--|:--|:--|
| youtube | 5.948 / 531 | 30 / 23 | 0,689 (0,535 a 0,794) | 0,625 / 0,552 | 0,918 |
| bilibili | 4.263 / 793 | 14 / 14 | 0,643 (0,344 a 0,908) | 0,643 / 0,600 | 0,843 |
| x | 3.046 / 1.061 | 24 / 24 | 0,706 (0,505 a 0,842) | 0,648 / 0,565 | 0,742 |
| facebook | 3.027 / 1.129 | 26 / 34 | 0,734 (0,578 a 0,851) | 0,659 / 0,632 | 0,728 |
| rednote | 1.333 / 1.393 | 34 / 47 | 0,650 (0,495 a 0,802) | 0,613 / 0,519 | 0,511 |
| tiktok | 369 / 305 | 12 / 16 | 0,615 (0,461 a 0,775) | 0,561 / 0,534 | 0,547 |
| douyin | 182 / 1.149 | 10 / 14 | 0,700 (0,519 a 0,863) | 0,649 / 0,519 | 0,863 |

## 4. Sensibilidade e atalhos (diagnóstico; não mudam o F0)

**(a) Duração no RADAR.** Pareando real e IA em faixas de 0,5 s (mesmo número por classe e faixa; 42.238 de cada; 20 sorteios, seed 42; medianas passam a 4,65 s e 4,67 s), a AUC fica em **0,720 (0,719 a 0,721 entre os sorteios)**, contra 0,721 no banco original. Por faixa de duração original: RADAR < 2 s 0,698 (0,687 a 0,709), 2 a 4 s 0,715, ≥ 4 s 0,728; ML-ITW 2 a 4 s 0,686, ≥ 4 s 0,693 (o ML-ITW não tem clipes < 2 s). **A diferença de duração entre as classes do RADAR não explica a AUC.** Os curtos são um pouco piores, como na v4.

**(b) Locutores no ML-ITW (IC por locutor):**

| recorte | n real / IA | locutores real / IA | AUC (IC95) | acurácia balanceada A / B |
|:--|:--|:--|:--|:--|
| inteiro | 18.168 / 6.361 | 143 / 160 | 0,689 (0,613 a 0,759) | 0,634 / 0,556 |
| só locutores com as duas classes | 15.673 / 4.785 | 123 / 123 | 0,714 (0,628 a 0,791) | 0,653 / 0,572 |
| sem os 3 possíveis sobrepostos com o In-the-Wild | 17.853 / 6.063 | 141 / 157 | 0,688 (0,611 a 0,759) | 0,634 / 0,555 |
| as duas coisas | 15.358 / 4.492 | 121 / 121 | 0,715 (0,622 a 0,794) | 0,654 / 0,571 |

Tirar os 3 nomes que também aparecem no In-the-Wild (dois deles estão no dev_shift, que calibrou o corte B) **não muda nada** (0,689 → 0,688). Ficar só com quem tem as duas classes sobe a AUC 0,025, dentro do intervalo.

**(c) Auditoria de atalho.** Uma regressão logística **só com metadados** (duração, taxa de amostragem original, canais, fração de silêncio nas bordas, RMS e pico), com validação cruzada em 5 partes (agrupada por locutor no ML-ITW, por clipe no RADAR):

| fonte | AUC metadados (pedido) | + tamanho do arquivo (protocolo) | só duração | AUC do F0 | marca (> 0,7) |
|:--|:--|:--|:--|:--|:--|
| RADAR | 0,647 | 0,665 | 0,588 | 0,721 | sem sinal forte de atalho |
| ML-ITW | 0,633 | 0,633 | 0,519 | 0,689 | sem sinal forte de atalho |

Taxa de amostragem e canais são **constantes** nos dois bancos (16 kHz, mono): não carregam informação. Uma árvore rasa (profundidade 3), secundária, deu 0,651 / 0,652 no RADAR e 0,613 no ML-ITW. **Nenhuma fonte passa de 0,7**, mas os metadados ficam **a só 0,06 a 0,07 da AUC do F0** nos dois bancos. Pela regra do protocolo ("se chegar perto do AUC do F0, suspeito de atalho"), isso pede cautela: o banco tem diferenças de nível e de silêncio entre as classes que um modelo trivial já explora. Mas a nota do F0 **não acompanha a duração** dentro de cada classe (Pearson: RADAR real −0,02, IA +0,03; ML-ITW real −0,07, IA −0,05; Spearman parecido). Não medi se o F0 usa o nível ou o silêncio (fica como ideia na seção 8).

**(d) Contexto: o mesmo F0 no teste da v4** (`results/v4_teste_final.csv`, sistema S2, sem recalcular):

| banco | AUC (IC95) | EER | acurácia balanceada A / B | TPR / FPR em A | TPR / FPR em B |
|:--|:--|:--|:--|:--|:--|
| In-the-Wild (test_final) | 0,979 (0,976 a 0,981) | 6,8% | 0,835 / 0,931 | 98,9% / 31,8% | 92,0% / 5,9% |
| LlamaPartialSpoof | 0,925 (0,920 a 0,930) | 14,7% | 0,855 / 0,738 | 75,0% / 4,0% | 47,9% / 0,2% |
| BRSpeech-DF (pt-BR) | 0,842 (0,827 a 0,854) | 24,9% | 0,679 / 0,771 | 99,5% / 63,6% | 88,8% / 34,5% |
| **RADAR eval** | **0,721** (0,718 a 0,725) | 33,9% | 0,667 / 0,603 | 56,9% / 23,6% | 24,7% / 4,2% |
| **ML-ITW** | **0,689** (0,613 a 0,759) | 35,5% | 0,634 / 0,556 | 40,4% / 13,5% | 14,5% / 3,2% |

## 5. Leitura em linguagem simples

**O que o resultado diz:**
- Em dois bancos que o projeto nunca tinha usado, o sistema **continua acima do acaso, mas muito pior que nos testes anteriores**: AUC ~0,72 no RADAR e ~0,69 no ML-ITW, contra 0,98 no In-the-Wild. Em palavras: se você sorteia um áudio real e um gerado por IA, o sistema dá nota maior ao falso em ~70% das vezes (no In-the-Wild eram 98%).
- **A queda é geral.** Aparece em todos os idiomas do RADAR (de 0,67 a 0,80, incluindo inglês com 0,72) e em todas as plataformas e idiomas do ML-ITW com amostra suficiente. Não é um problema só de idioma fora do treino: o inglês também caiu.
- **Os cortes fixados antes funcionam mal aqui.** O corte B, que no In-the-Wild pegava 92% dos falsos, aqui pega só 25% (RADAR) e 15% (ML-ITW): **de cada 4 falsos, 3 passam como reais**. O corte A pega mais (57% e 40%), mas acusa 24% e 14% dos reais. O corte que daria o EER em cada banco fica abaixo dos dois (−2,3 e −5,9): neste tipo de áudio, os falsos recebem notas mais baixas do que o sistema espera.
- **Duração não explica a queda** (RADAR pareado: 0,720 × 0,721). **A possível sobreposição de locutores com o In-the-Wild também não** (0,688 sem eles).
- **Só com metadados** (duração, silêncio, volume) já se chega a 0,63 a 0,67 de AUC, perto do sistema. Não passa da regra de 0,7, mas mostra que nesses bancos boa parte da diferença entre real e IA está em características de gravação, e que o sistema não está muito acima disso.

**O que o resultado NÃO diz:**
- **Não diz como o sistema se sai em golpes reais por ligação.** Nenhum dos dois bancos é ligação telefônica real: o RADAR simula o canal (codecs, ruído, música, compressão) e o ML-ITW são vídeos de redes sociais.
- **Não diz nada firme sobre português.** O português só existe no ML-ITW (911 reais, 108 falsos, 7 e 8 locutores; poucas horas): AUC 0,69 com intervalo de 0,44 a 0,90, ou seja, **indício**. O RADAR não tem português.
- **Não identifica a causa.** Pode ser o tipo de gerador (vozes comerciais de plataforma, mais novas que as do treino), as transformações de mídia do RADAR, o tipo de rótulo do ML-ITW, ou tudo isso junto. Este teste mede, não explica.
- **Não diz que outro modelo do projeto seria melhor aqui.** Só o F0 foi avaliado (pelo protocolo).
- O IC do RADAR é **otimista**: trata clipes do mesmo locutor ou da mesma frase como independentes. O IC real é mais largo.

## 6. Limitações

- **RADAR:** canal simulado (transformações de mídia aplicadas pelos autores); **sem português**; **sem locutor no arquivo** (IC por clipe, otimista); os falsos são **vozes comerciais de plataforma**, não clones de uma pessoa específica; o dev do desafio ficou de fora (sobrepõe-se aos nossos dados).
- **ML-ITW:** os rótulos de falso vêm do **contexto** da postagem (não de um gerador conhecido); **57 dos 180 locutores têm uma classe só** (por isso a sensibilidade b); alguns "locutores" são nomes genéricos; **possível sobreposição com o In-the-Wild** por nome (3 pessoas; sem efeito medido, seção 4b); a licença não está declarada; o repositório usado (`nccm2p2/ML-ITW`) bate em todas as contagens, mas a autoria não foi confirmada (`docs/dados_teste_internet.md`).
- **Português:** só no ML-ITW, poucas horas, 7 a 8 locutores por classe: indício.
- **Cortes:** fixados em outros dados (A no estúdio, B no dev_shift de 16 locutores); aqui só são medidos.
- **Nenhuma conclusão** sobre golpes reais de ligação.
- Estes dois bancos agora estão **consumidos**: qualquer mudança no sistema feita olhando estes números os contamina como teste.

## 7. Desvios do protocolo e decisões de execução

Todos decididos **antes** de olhar qualquer número de teste, exceto o 9, que corrige um erro de código e não muda nenhum outro número.

1. **Bootstrap com 2.000 repetições** (o protocolo dizia 1.000). Foi pedido assim na tarefa. Mais repetições só reduzem o ruído da própria simulação, e o que se estima continua o mesmo.
2. **No ML-ITW, o bootstrap por locutor é estratificado por classe:** reamostra, com reposição, os locutores que têm reais e, separado, os que têm falsos (um locutor com as duas classes entra nos dois sorteios). O protocolo dizia só "reamostra locutores".
3. **"Poucos áudios" = menos de 20 clipes em uma classe:** sem AUC, EER nem IC, só contagem. O protocolo não definia o número. Afeta húngaro, ucraniano e hebraico no ML-ITW.
4. **2 workers em vez de 4.** Com 4 workers, processo + workers chegaram a 10,9 GB e o vigia de memória encerrou o processo (limite de 10 GB, também pedido); com 3, o pico foi 9,91 GB, perto demais. Os dois pedidos não cabiam juntos; fiquei com o de memória. **Não muda nenhuma nota:** a leitura é determinística, e um shard feito com 3 workers e retomado com 2 saiu idêntico bit a bit à referência.
5. **Aquecimento com ruído sintético.** Antes da extração, o modelo roda uma vez com ruído no maior balde (32 × 8 s), e o resultado não é gravado. Medido com áudio sintético: sem aquecimento, o processo principal ia a 8,6 a 9,2 GB (o alocador da GPU cresce aos pedaços e, no Windows, isso conta na memória reservada); com aquecimento, ~7,2 GB. Execução, não método: o modelo está congelado.
6. **Orçamento de extração estendido de 3,0 h para 3,3 h,** com autorização do usuário. O relógio do orçamento correu durante a **espera do Windows** (pelo log de energia, o notebook ficou suspenso ~2,5 h no total: 15:34 a 15:57, 16:25 a 16:34, 16:36 a 18:28 e intervalos curtos). A extração parou com 63 de 65 shards; os 2 restantes (2.529 clipes) levaram 1 min. O script passou a impedir a suspensão (como os da v4 a v6). O tempo de máquina ativa ficou em ~0,7 h, bem abaixo das 2,5 h.
7. **Auditoria de atalho:** foram rodadas a lista pedida (duração, taxa, canais, fração de silêncio, RMS, pico) e a do protocolo (+ tamanho do arquivo; codec e extensão são constantes dentro de cada fonte), com uma regressão logística (principal) e uma árvore rasa (secundária), as duas previstas no protocolo. A "fração de silêncio" é o silêncio das bordas (início + fim) medido pelo mesmo corte de 35 dB da extração, dividido pela duração.
8. **Análises de sensibilidade (seção 4a e 4b) e o corte de EER de cada fonte** foram pedidos na tarefa depois do congelamento do protocolo, mas antes dos resultados. São diagnóstico: não mudam o F0 nem os cortes.
9. **Erro de código corrigido depois da 1ª rodada de métricas:** o filtro dos 3 locutores sobrepostos não achou nenhum, porque os ids do manifesto têm o prefixo `mlitw:`. O filtro foi corrigido, com uma checagem de que os 3 aparecem, e as métricas foram recalculadas por inteiro. Todas as contas são determinísticas, e só as linhas desse filtro mudaram.
10. O teste rápido calculou notas em 200 clipes destes bancos (para testar o código), sem ler os números. Esses clipes fazem parte da avaliação completa.

## 8. Tempo e execução

- Validação do F0: 0,4 min. Teste rápido (com interrupção) e vazão: ~10 min, incluindo as medidas de memória.
- **Extração:** 15:24 a 18:31, com ~2,5 h de suspensão do Windows no meio (seção 7, item 6). Nos 59 shards sem suspensão, de 53 a 60 clipes/s (RADAR e ML-ITW iguais); os 6 shards que pegaram uma suspensão mostram de 0,3 a 32 clipes/s. Tempo de máquina ativa estimado em ~0,7 h (127.254 clipes a ~56 clipes/s, mais as cargas do modelo). Pico de 9,4 GB (processo + 2 workers); GPU 3,8 GB; 0 lotes refeitos em bf16; 0 ilegíveis.
- Notas: 0,5 min (4,6 GB). Métricas com 2.000 bootstraps: 3 min (3,2 GB). Figuras: ~1 min.

Ideias que este resultado sugere (para depois; nenhuma foi testada): medir se a nota do F0 acompanha o nível (RMS) e o silêncio dentro de cada classe; um conjunto de decisão com vozes comerciais de plataforma e canal degradado, fora destes bancos; calibrar o corte por domínio, já que aqui os dois cortes ficaram do lado conservador.
