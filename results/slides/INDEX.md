# Figuras para os slides (sistema v4 = F0)

Geradas por `scripts/v4final_figuras.py` (13,33 × 7,5 pol., dpi 180). Corte A = calibrado no estúdio (nota −0,04); corte B = calibrado na internet (nota 8,22).

| figura | o que mostra | o que olhar |
|:--|:--|:--|
| `01_composicao_banco.png` | Quantos áudios reais e gerados por IA há em cada papel (treino, validação, dev_shift, testes) e de que banco vem cada um. | O treino e a validação são inglês de estúdio ou leitura; a internet (dev_shift) decide entre versões; os testes são internet, fakes parciais e português. |
| `02_log_mel.png` | Um áudio real e um gerado por IA (mesma frase, validação): forma de onda, espectrograma e log-mel de 64 faixas. | O log-mel é a 'imagem' que a CNN das versões 1 a 3 recebia; o sistema final (v4) lê a onda direto com o XLS-R. |
| `03_plano_regressao.png` | A nota da regressão (eixo x) de cada áudio no estúdio (validação) e na internet (dev_shift), com os cortes A e B. | No estúdio quase todo real fica à esquerda do corte A; na internet, uma cauda de reais passa dele, e o corte B (mais à direita) os recupera. |
| `04_distribuicao_nota.png` | Histograma da nota de reais e de gerados por IA em seis grupos: estúdio, internet (dev_shift e teste), fakes parciais, português e português só real. | No estúdio as duas montanhas quase não se tocam; na internet e no português os reais se deslocam para a direita e passam do corte A. |
| `05_camadas.png` | Erro e AUC de uma regressão por camada do XLS-R, no estúdio (validação) e na internet (dev_shift). | As camadas 0 a 3 quase não servem na internet; da 5 em diante tudo passa de 0,97; as do meio (5 a 7) têm o menor erro. |
| `06_vetores.png` | Os vetores das camadas 5, 6, 7 e 24, sem a regressão, achatados em 2D: só estúdio (real × IA), estúdio + internet (origem) e só internet (real × IA). | Nas camadas do meio as cores real/IA já se agrupam; na linha do meio, a separação mais forte é estúdio × internet; na 24 tudo se mistura mais. |
| `07_c_freio.png` | Erro no estúdio e na internet, e o tamanho dos pesos, para 11 valores do freio de regularização (C). | O estúdio tem o menor erro no C usado (0,1); a internet melhora com freio mais forte (C = 0,0001: erro 2,7% contra 5,9%); diagnóstico, não muda o sistema. |
| `08_curva_aprendizado.png` | Acerto da mesma regressão treinada com 2 mil a 40 mil áudios, no próprio treino, no estúdio e na internet. | O treino fica em 100%; o estúdio melhora pouco; a internet não melhora (AUC 0,993 com 2 mil, 0,983 com 40 mil). |
| `09_acuracia_grupos.png` | Acurácia no teste por grupo: sistema v4 nos cortes A e B, CNN antiga e o chute da classe comum; à direita, % de reais do CORAA aceitos. | Na internet a CNN fica abaixo do chute e o v4 vai a 84% (A) e 93% (B); no português (BRSpeech-DF) o v4 passa pouco do chute; no CORAA, o corte A aceita 83% dos reais. |
| `10_erros_por_grupo.png` | Os dois erros do sistema por grupo: reais acusados como IA e gerados por IA que passam, nos cortes A e B. | O corte A acusa muitos reais na internet e no português; o corte B quase não acusa reais, mas deixa passar metade dos fakes parciais. |
| `11_duracao.png` | AUC do sistema por faixa de duração original do áudio (< 2 s, 2 a 4 s, ≥ 4 s) nos três bancos principais, com o nº de áudios. | Em todos os bancos os curtos são os piores; no português abaixo de 2 s a ordenação fica perto de cara ou coroa (0,53). |
| `12_meta_microsoft.png` | Erro no estúdio e AUC na internet de uma regressão por camada, nos dois modelos prontos: XLS-R (Meta) e WavLM (Microsoft). | No WavLM as melhores camadas são as de baixo (3 e 4) e as de cima pioram muito; no XLS-R as do meio (5 a 9) ficam todas boas. |
| `13_secundarios.png` | Acurácia (cortes A e B) e AUC nos dois testes secundários (ASVspoof 2019 e 2021 DF), sistema v4 e CNN antiga. | Números altos, mas esses bancos podem ter locutores em comum com o treino: valem para geradores novos, não para pessoas novas. |
| `14_roc.png` | Curva ROC do sistema nos três bancos principais do teste, com os pontos dos cortes A e B. | Internet e fakes parciais ficam perto do canto superior esquerdo (bom); o português fica bem abaixo; os pontos A e B mudam de lugar em cada banco. |

Tabelas: `grupos_bancos.csv` (banco, grupo, o que é, real/IA, idioma, ano, papel, nº de áudios) e `hiperparametros_versoes.csv` (hiperparâmetros de v1, v2, v3, v4 e v6).
