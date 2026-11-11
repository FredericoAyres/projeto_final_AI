# Decisões da v6 (funil)

Este arquivo tem duas partes. A **parte 1** foi escrita em 2026-11-10 22:15 (−03:00), **antes de qualquer resultado da v6**, junto do pré-registro (`docs/HISTORICO_VERSOES.md`, seção "v6 - pré-registro"): o que vamos testar e por que esses candidatos, e não outros. A **parte 2** é escrita depois e diz o que de fato aconteceu, com os números.

## Parte 1 · Por que testamos X e não Y (antes dos resultados)

### Geral

| decisão | escolhemos | e não | por quê |
|:--|:--|:--|:--|
| métrica primária | AUC da **pior** fonte (DFADD, ShiftySpeech, CodecFake+ da validação e dev_shift) | EER média por fonte (v4/v5) | Um detector é tão bom quanto o banco em que ele mais erra. A média esconde uma fonte ruim atrás de duas fáceis (DFADD tem EER perto de 0%). A AUC não depende de limiar nem da proporção real/fake. |
| incluir o dev_shift na pior fonte | sim | só a validação | O dev_shift (vozes de internet) é o domínio que mais importa e é o único "fora do estúdio" que podemos olhar à vontade. Custo: são só 16 locutores, por isso o bootstrap é por locutor. |
| régua de "diferença relevante" | IC95 do bootstrap pareado sem o zero **e** ≥ 0,005 de AUC | só o IC, ou só o valor | Só o IC deixaria passar diferenças minúsculas que não mudam nada na prática; só o valor ignoraria a incerteza (a v3 mostrou o quanto o acaso pesa). |
| bootstrap por locutor | sim (por clipe só no CodecFake+, que não tem locutor) | por clipe em tudo | Clipes do mesmo locutor não são independentes; reamostrar clipes dá um intervalo estreito demais (registrado na v5). |
| empate | vence o mais simples | vence o maior número | Com a amostra que temos, ganhar por pouco no ponto não é evidência; o mais simples é mais fácil de explicar e tende a generalizar melhor (Aula de overfitting e regularização). |
| regra do desafiante, em sequência | cada opção mais complexa só entra se superar a que está no lugar | comparar todas contra todas | Uma ordem fixada antes evita escolher a comparação que dá o resultado que queremos. |
| clipes | os do `v4_cache` (mesmo plano) para os dois modelos | um plano novo | A comparação fica pareada clipe a clipe, e a v4 continua reproduzível. |
| XLS-R | lido do `v4_cache` | reextrair | Os vetores já existem nesta máquina; reextrair custaria ~30 min sem mudar nada. A reprodução da v4 confere que o código ainda dá os mesmos números. |
| WavLM | extrair aqui (`scripts/v6.py extrair`, com `src/ssl.py` e `src/extracao_v4.py`) | rodar o `rodar_v4.ps1 -Modelo wavlm` como está | O `rodar_v4.ps1 -Modelo wavlm` grava em `results/v5_wavlm_*` e sobrescreveria arquivos do colega que estão no git (por exemplo, `v5_wavlm_vazao.json`). O código de extração usado é o mesmo; só os nomes dos arquivos mudam (`v6_wavlm_*`). |

### E1 · Meta × Microsoft

- **Por que refazer a comparação, se a v5 já fez?** As notas por clipe do WavLM só existem na máquina do colega; sem elas não há fusão (E2) nem bootstrap pareado com a nova métrica. E a v5 rodou noutra placa.
- **Por que só os dois e não um terceiro modelo** (XLS-R 1B, MMS, HuBERT)? Orçamento da noite (7 h) e a regra de não baixar nada além do WavLM. Fica como ideia (f).
- **Por que a grade de C da v4 (1e-3 a 1) na E1, e não a grade ampla?** Para a E1 ser a mesma conta da v4/v5 (é o que permite conferir a reprodução). A grade ampla é uma modificação testada na E5.
- **Por que `concat_top3` e `concat_top4`?** São as cabeças que a v4 e a v5 já usaram; mantê-las deixa a comparação com as versões anteriores direta.

### E2 · Fusão

- **Por que testar fusão?** Na v5, as camadas de baixo do WavLM foram melhores que as do XLS-R no dev_shift (por exemplo, camada 3: AUC 0,906 × 0,793) e os dois erram em lugares diferentes (Churchill × Obama). Se os erros forem pouco correlacionados, juntar ajuda; por isso medimos a correlação e a tabela 2×2 **antes** de comparar.
- **(a) média das notas:** sem nenhum parâmetro novo, é o jeito mais simples de juntar dois classificadores. Padronizamos pela validação para nenhum modelo dominar só por ter logits maiores.
- **(b) vetores colados:** deixa a RL achar a combinação no nível das características (mais flexível), ao custo de milhares de pesos. Cada bloco é dividido pela raiz do nº de dimensões para um modelo não pesar mais só por ter mais camadas na entrada.
- **(c) stacking:** aprende o peso de cada modelo com só 3 números. O risco conhecido (v4, S3): aprender os pesos na própria validação deixa a nota "dentro da amostra". Por isso o cross-fitting **por fonte**: a nota de cada fonte vem de uma RL que nunca viu essa fonte (é também um teste de generalização entre fontes).
- **Por que não fusão por máximo, votação, ou rede de atenção?** O máximo e a votação jogam fora a confiança de cada modelo; uma rede de atenção tem parâmetros demais para 20 mil clipes de validação e sai da trilha.

### E3 · Front-end

- **Ordem Meta → Microsoft → fusão:** o XLS-R é a referência do projeto (testado na v4 e na v5); o WavLM é uma troca do mesmo tamanho; a fusão custa os dois modelos (o dobro de tempo e de memória no uso real). Na dúvida, fica o que custa menos.

### E4 · Cabeças

- **RL básica com C = 1, e não com C escolhido?** "Básica" quer dizer sem ajuste: o ajuste do C é a modificação (2) da E5. C = 1 é o padrão do scikit-learn.
- **MLP de 256 com dropout 0,3:** é a cabeça MLP da v4 (soma ponderada) e o dropout da CNN da v1 a v3; nada de novo para ajustar.
- **CNN ao longo das camadas, e não no tempo:** os vetores em cache já foram resumidos (média e desvio no tempo), então não há eixo do tempo. Andar pelas 25 camadas deixa a rede usar o padrão "o que muda de uma camada para a seguinte". Reextrair os quadros no tempo (para uma CNN no tempo) custaria horas e dezenas de GB. ⚠️ Fora da trilha.
- **Por que a CNN recebe média e desvio (2.048) e a RL/MLP só a média (1.024)?** Foi o pedido da rodada (a CNN usa tudo o que está no cache). Fica registrado: se a CNN ganhar, parte do ganho pode ser do desvio, e não da arquitetura (ablação não feita; ideia em f).
- **3 seeds e a exigência de vencer nas três:** a v3 mostrou que, em rede neural, a seed sozinha move a AUC do dev_shift em até 0,23. A RL é determinística (lbfgs, convexa), então não precisa de seeds.
- **Por que não Transformer, LSTM, SVM ou gradient boosting?** Fora da trilha e sem motivo forte para esperar ganho sobre uma RL que já passa de 0,98 no dev_shift; aumentaria o risco de overfitting na validação.

### E5 · Maximização

- **Ordem (1) camadas, (2) C, (3) pesos, (4) limiar:** do que mais muda a entrada para o que menos muda; o limiar por último porque não mexe na AUC.
- **Régua de manter (+0,003 na pior fonte, dev_shift sem cair mais que 0,01):** menor que a régua da E3/E4 (0,005) porque é ajuste fino dentro do mesmo sistema; o dev_shift não pode piorar porque é o domínio que mais importa. Atenção (registrada antes): esta régua **não** usa bootstrap; passos pequenos podem ser ruído da validação e isso será dito.
- **Zona de 3 vizinhas, e não qualquer trio:** camadas vizinhas carregam informação parecida (v4: 5 a 9 quase iguais); juntar vizinhas suaviza a escolha sem multiplicar as combinações (23 trios em vez de 2.300).
- **Pesos iguais por fonte:** no treino, o DFADD e o ASVspoof têm muito mais clipes que o CodecFake+ (1.445 de 40.000); se a pior fonte for o CodecFake+, dar a ele o mesmo peso pode ajudar. Reescalamos para reais e fakes pesarem o mesmo (o treino é 50/50).
- **Limiar C e D:** o limiar A (EER global da validação) é dominado pelo DFADD (66% da validação). C (mediana dos limiares por fonte) e D (5% de falso alarme por fonte) são menos sensíveis a uma fonte grande. A escolha deixando uma fonte de fora mede justamente "esse limiar funciona numa fonte que ele não viu?". O limiar B (dev_shift) não entra na escolha: é reportado no teste como na v4/v5.

### E6 · Teste

- **F0, F1, F2:** F0 mostra de onde partimos (v4); F1 mostra o efeito do front-end (com a cabeça mais simples); F2 mostra o efeito do ajuste fino. Três sistemas, declarados antes, para não escolher pelo teste.
- **Não extrair o teste do WavLM se ele não estiver no vencedor:** economiza ~1 h e não muda nenhuma decisão; os números do WavLM no teste já existem (do colega).

## Parte 2 · O que de fato aconteceu (escrita em 2026-11-10, depois da execução)

Execução: extração do WavLM lançada direto às 22:24 (23 min, 70.613 clipes, 0 erros de leitura, 35 clipes/s medidos); vigia lançado às 23:07, terminou às 23:26 (1 tentativa, nenhuma falha). Números de `results/v6_*`.

| etapa | decidimos | alternativa descartada | o que aconteceu (números: AUC da pior fonte, val + dev_shift) |
|:--|:--|:--|:--|
| E1 | comparar a melhor cabeça de cada modelo | — | Meta `concat_top4` (camadas 5, 7, 8, 11; C = 1) **0,907**; Microsoft `concat_top3` (4, 6, 7; C = 1) **0,891**; v4 (5, 6, 7; C = 0,1) 0,903. Microsoft − Meta = −0,017 (IC95 −0,048 a +0,001): empate. Reprodução da v4: diferença de 0,0 ponto em todas as camadas. |
| E2 | fusão por **média das notas** | stacking (0,905; +0,0004 sobre a média, não relevante) e vetores colados (0,874; pior) | média das notas 0,905. Erros em comum: os dois acertam 97,7% da validação e 77,5% do dev_shift; só a Meta acerta 6,4% do dev_shift, só a Microsoft 3,2%. Correlação dos logits de 0,51 a 0,77. Pesos do stacking parecidos (Meta 4,4; Microsoft 4,0). |
| E3 | **só Meta (XLS-R)** | só Microsoft (−0,017) e fusão (−0,002, IC95 −0,012 a +0,008) | nenhuma opção mais complexa ganhou com diferença relevante. Por isso **não** extraímos o teste do WavLM (economia de ~25 min); os números do WavLM no teste são os do colega. |
| E4 | **RL** (camada 11 sozinha, C = 1: a melhor camada única pela pior fonte com C = 1) | MLP (0,902 a 0,912 nas 3 seeds) e CNN (0,906 a 0,910) | RL 0,904. Nenhuma seed de MLP ou CNN com diferença relevante sobre a RL (todos os IC95 com o zero). Custo: RL 1.025 parâmetros e 2 s; MLP 262.657 e ~2 s; CNN 405.825 e ~13 s. |
| E5 | manter só os **pesos iguais por fonte**; limiar **A** | zona de camadas (+0,000), grade ampla de C (+0,001); limiares C e D | pesos por fonte +0,006 (0,904 → 0,910; dev_shift igual, 0,972). Limiar: validação cruzada deixando uma fonte de fora, acurácia balanceada A 0,916, C 0,916, D 0,917 (D não ganhou 0,005) → A. |
| E6 | mostrar F0, F1, F2 uma vez | — | média da acurácia nos 3 principais: **F0 0,851**, F1 0,828, F2 0,829 (trivial 0,597); falso alarme do CORAA: **F0 16,7%**, F1 41,2%, F2 40,4%. F1/F2 − F0 em AUC: In-the-Wild −0,007, LlamaPartialSpoof +0,015/+0,012, BRSpeech-DF −0,100/−0,097 (todos com IC95 sem o zero, mas sinais trocados: "não demonstrada" pela regra). |

**O que aprendemos com as decisões:**
- **A métrica primária dependeu de uma fonte só.** O CodecFake+ (352 clipes, sem locutor) foi a pior fonte em todas as cabeças comparadas. As diferenças entre candidatos ficaram em milésimos nessa fonte, e a regra "pela pior fonte, com C = 1" levou a RL básica para a camada 11 (dev_shift 0,972), em vez das camadas do meio (camada 7: 0,988). Essa troca não aparece como problema na validação, mas piorou o português no teste (BRSpeech-DF −0,10 de AUC; CORAA de 17% para 41% de falso alarme). A decisão de usar "a pior fonte" foi boa na intenção (não esconder a fonte ruim) e ruim na prática, porque a pior fonte era pequena e de um domínio estreito (fakes de codec de páginas de demonstração).
- **"Básica com C = 1" tirou a regularização que a v4 tinha** (C = 0,1 na concatenação). Com menos regularização, a cabeça se ajusta mais ao estúdio do treino. É o tema de overfitting e regularização das aulas: menos regularização ajudou no CodecFake+ (que se parece com o treino) e atrapalhou no português (que não se parece).
- **A régua da E5 sem bootstrap** deixou passar os pesos por fonte (+0,006 numa fonte de 352 clipes); no teste, F2 ficou igual ao F1 (0,829 × 0,828), ou seja, o ganho foi da validação, não do mundo.
- **O que fica:** a v4 (F0) continua sendo o modelo do projeto. A v6 não muda o modelo; muda o que sabemos: o WavLM, a fusão e as cabeças não lineares não ajudam aqui, e escolher pela validação atual não protege o português.

**Desvios e execução (nenhum muda o método):**
- A extração do WavLM foi lançada diretamente (`scripts/v6.py medir` e `extrair`), antes do vigia, para ganhar tempo; o vigia pulou a extração pelo marcador.
- O WavLM foi baixado por revisão fixa e o cache do Hugging Face ficou sem o ponteiro `refs/main`; em vez de editar o cache, `src/ssl.py: carregar_ssl` ganhou o parâmetro opcional `revisao` (sem ele, o comportamento é o de antes).
- Dois erros de código foram corrigidos antes da execução oficial: a ordem dos argumentos no bootstrap (achado no QUICK) e a conversão de números do pandas para JSON (achada na revisão do código).
- A projeção da CNN da v3 (opcional) não foi feita.
