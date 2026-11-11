### Conclusão (escrita em 2026-11-10, depois de ver os números)

**Em uma frase:** pelas regras pré-registradas, nada superou o XLS-R da Meta com uma regressão logística (nem o WavLM, nem a fusão, nem MLP, nem CNN), e o "sistema final" da v6, escolhido pela AUC da pior fonte, ficou **pior que a v4 no teste**, principalmente em português; a v4 (`concat_top3`, camadas 5, 6, 7) continua sendo o melhor modelo do projeto.

**O que funcionou**
- **O pipeline é reproduzível:** o XLS-R recalculado deu exatamente os números da v4 (diferença 0,0 ponto de EER em todas as 25 camadas; notas da cabeça da v4 idênticas às gravadas), e o F0 reproduziu o teste da v4 (In-the-Wild AUC 0,979; CORAA 16,7% de falso alarme).
- **Os dois modelos nos mesmos clipes:** o WavLM foi extraído em 23 min nos 70.613 clipes do plano da v4, sem nenhum erro de leitura; a comparação é pareada clipe a clipe.
- **A execução sozinha:** vigia, retomada e marcadores funcionaram (a noite oficial terminou numa tentativa, em 0,3 h).

**O que não funcionou (e por quê, até onde dá para saber)**
1. **WavLM (E1):** pior que o XLS-R na AUC da pior fonte (0,891 × 0,907; diferença −0,017, IC95 −0,048 a +0,001): empate pela regra, fica o mais simples. O mesmo resultado da v5.
2. **Fusão (E2/E3):** os dois modelos acertam juntos 97,7% da validação e erram juntos 0,5%; no dev_shift, só a Meta acerta 6,4% e só a Microsoft 3,2%. Os erros não são independentes o bastante (correlação dos logits de 0,51 a 0,77 por classe), e a melhor fusão (média das notas, 0,905) não passou do XLS-R sozinho (0,907). Os vetores colados ficaram ótimos no dev_shift (AUC 0,996), mas piores no CodecFake+ (0,874): mais pesos, mais ajuste ao estúdio.
3. **MLP e CNN (E4):** um pouco acima da RL no ponto (pior fonte 0,902 a 0,912 contra 0,904), mas nenhuma seed com diferença relevante (todos os IC95 incluem o zero). Com o modelo congelado fazendo o trabalho pesado, uma cabeça não linear não acrescenta nada demonstrável (de novo, como as somas ponderadas da v4).
4. **Ajuste fino (E5):** a zona de camadas e a grade ampla de C não mudaram nada (+0,000 e +0,001); só os pesos iguais por fonte passaram a régua (+0,006 na pior fonte, sem bootstrap). A regra de limiar ficou a A (C e D não ganharam 0,005 na validação cruzada).
5. **O teste mostrou o limite da métrica nova:** F1 e F2 (XLS-R, **camada 11 sozinha**, C = 1) ficaram piores que a v4 no In-the-Wild (AUC 0,972 × 0,979), muito piores no BRSpeech-DF (0,74 × 0,84) e acusaram 40% a 41% dos reais do CORAA (v4: 17%); só no LlamaPartialSpoof foram melhores (0,94 × 0,925). Média da acurácia nos três principais: F0 0,851, F1 0,828, F2 0,829 (trivial 0,597).

**A lição principal:** a "AUC da pior fonte" virou, na prática, **a AUC do CodecFake+**: ele foi a pior fonte em todas as cabeças comparadas nas etapas (e em todas as camadas a partir da 4; só em algumas das camadas 0 a 3 a pior era o dev_shift), tem só 352 clipes (de 17 páginas de demonstração, sem locutor) e as diferenças entre candidatos ficaram em milésimos. Escolher pela pior fonte trocou as camadas do meio (5 a 7, mais regularizadas, melhores no dev_shift) pela camada 11 com C = 1, e essa troca piorou exatamente o que a validação não mede: **português**. Uma métrica pensada para não esconder a fonte ruim acabou dependendo de uma fonte pequena e pouco representativa. A régua da E5 (+0,003 sem bootstrap) também é fina demais para essa fonte.

**Ressalvas:** os bancos de teste já tinham sido vistos na v4 e na v5 (o teste aqui só mostra); o bootstrap do teste mede só a amostra de teste; o dev_shift tem 16 locutores; uma única amostra de treino.

**O que fica para o grupo decidir:** seguir com a v4 (F0) como modelo do projeto; e, para escolher melhor no futuro, ter um conjunto de decisão em **português** e uma fonte de codec maior (ideias na seção f do histórico).
