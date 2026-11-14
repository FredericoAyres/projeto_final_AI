# Figuras para os slides (sistema atual: v7 = T3; sistema anterior: v4 = F0)

Todas em 13,33 × 7,5 pol., dpi 180. Em ordem de história. Geradas por: `scripts/v4final_figuras.py` (01 a 14), `scripts/teste_internet_figuras.py` (ti_*), `scripts/v7_figuras.py` (v7_*) e `scripts/v7_slides.py` (f_*; mesmo estilo: real = azul, IA = laranja, neutro = cinza; "visto no treino" em tom claro ou hachurado). Corte A = calibrado no estúdio; corte B = calibrado nas entrevistas da internet (v4); A', M e F5 = as três regras de corte da v7, calculadas só na validação.

| # | figura | o que mostra | o que olhar | parte da história |
|:--|:--|:--|:--|:--|
| 1 | `01_composicao_banco.png` | Quantos áudios reais e de IA há em cada papel (treino, validação, dev_shift, testes) e de que banco vem cada um. | O treino é inglês de estúdio ou leitura; a internet e o português só aparecem fora do treino. | O problema e os dados (v1 a v4) |
| 2 | `02_log_mel.png` | Um áudio real e um de IA (mesma frase): forma de onda, espectrograma e log-mel. | O log-mel era a "imagem" da CNN das versões 1 a 3; a v4 lê a onda com o XLS-R. | Como o áudio vira números |
| 3 | `05_camadas.png` | Erro e AUC de uma regressão por camada do XLS-R, no estúdio e na internet. | As camadas do meio (5 a 7) são as melhores; as de baixo quase não servem na internet. | O pipeline (o mesmo no F0 e na T3) |
| 4 | `03_plano_regressao.png` | A nota da regressão de cada áudio no estúdio e na internet, com os cortes A e B. | No estúdio quase todo real fica à esquerda do corte A; na internet, uma cauda de reais passa dele. | A nossa parte: a regressão logística |
| 5 | `04_distribuicao_nota.png` | Histograma da nota de reais e de IA em seis grupos. | Na internet e no português, os reais se deslocam para a direita. | A nota muda de escala entre ambientes (primeiro sinal) |
| 6 | `07_c_freio.png` | Erro no estúdio e na internet para 11 valores do freio (C). | A internet melhora com mais freio; diagnóstico, não muda o F0. | Por que regressão / overfitting |
| 7 | `08_curva_aprendizado.png` | A mesma regressão treinada com 2 mil a 40 mil áudios. | O treino fica em 100%; a internet não melhora com mais estúdio. | Overfitting: mais do mesmo não ajuda |
| 8 | `09_acuracia_grupos.png` | Acurácia no teste da v4 por grupo, F0 × CNN antiga × chute. | A CNN fica abaixo do chute na internet; o F0 vai a 84% (A) e 93% (B). | Resultado do sistema anterior (v4 = F0) |
| 9 | `10_erros_por_grupo.png` | Reais acusados e IA que passa, por grupo, nos cortes A e B. | Nenhum corte único é bom em todos os grupos. | Resultado do sistema anterior: os dois erros |
| 10 | `11_duracao.png` | AUC do F0 por faixa de duração nos três bancos principais da v4. | Os curtos são os piores; o português curto fica perto do acaso. | Ponto fraco do sistema anterior |
| 11 | `14_roc.png` | Curva ROC do F0 nos três bancos principais da v4, com os cortes A e B. | Inglês perto do canto; português bem abaixo. | Resultado do sistema anterior (reserva) |
| 12 | `12_meta_microsoft.png` | Erro e AUC por camada no XLS-R (Meta) e no WavLM (Microsoft). | O XLS-R é melhor no conjunto. | O caminho v5 e v6 (o que não funcionou) |
| 13 | `13_secundarios.png` | Acurácia e AUC nos testes secundários (ASVspoof). | Números altos, mas valem só para geradores novos. | Reserva (perguntas) |
| 14 | `f_00_queda_externa.png` | AUC do F0 por tipo de áudio: estúdio e entrevistas da internet × internet real × IA moderna em canal degradado, com IC95. | Cinza (já usados) entre 0,842 e 0,998; preto (bancos novos) em 0,72 e 0,69. | **A prova externa: o F0 cai** |
| 15 | `ti_01_auc_recortes.png` | AUC do F0 nos bancos novos por idioma e plataforma. | A queda aparece em todos os idiomas, inclusive o inglês. | A prova externa (detalhe) |
| 16 | `ti_04_roc.png` | ROC do F0 nos bancos novos, com os cortes A e B. | Com o corte B, 3 de cada 4 falsos passam. | A prova externa: os cortes da v4 não servem |
| 17 | `ti_02_histogramas.png` | Notas de reais e de IA nos bancos novos, com os cortes. | As duas montanhas se sobrepõem muito. | A prova externa (reserva) |
| 18 | `ti_03_duracao.png` | AUC por faixa de duração e com durações pareadas. | Todas as faixas perto de 0,7. | A prova externa: a duração não explica |
| 19 | `ti_05_atalho.png` | AUC do F0 × um classificador que só vê metadados. | Metadados abaixo de 0,7, mas perto do F0. | A prova externa: auditoria de atalho |
| 20 | `f_02_desenho_dos_dados.png` | Treino, validação e teste mestre por tipo de fonte, com clipes reais / de IA e locutores. | A faixa verde: nenhuma pessoa, arquivo ou áudio idêntico em dois lados. | **v7: o novo desenho dos dados** |
| 21 | `f_08_anti_vazamento.png` | Cada trava contra vazamento, o que verifica e o resultado. | A linha laranja: a trava que parou a 1ª execução (12 grupos de reais duplicados, herança da v4) e a correção. | **Como evitamos vazamento** |
| 22 | `f_01_antes_depois.png` | AUC no teste mestre por tipo de áudio, F0 × T3, com IC95, marcando visto × nunca visto. | Os saltos grandes estão nos tipos vistos; o nunca visto não piora. | **Resultados da v7: antes e depois** |
| 23 | `f_03_receitas.png` | As 5 receitas (T0 a T4) por fonte do teste mestre e dois controles. | T1 fraca no RADAR, T2 mais fraca na internet real; T3 e T4 altas nas duas. | Resultados da v7: por que a T3 |
| 24 | `f_04_sinal_de_generalizacao.png` | Setas F0 → T1 e F0 → T2 só nos tipos de áudio que a receita não viu. | T1 sobe o RADAR (+0,09); T2 sobe a internet real (+0,29). | Resultados da v7: o sinal de generalização mais limpo |
| 25 | `f_06_ingles.png` | Tabela colorida: AUC por receita no bloco inglês. | A internet real em inglês sobe de 0,58 para 0,94 a 0,96 (indício: 18 / 26 locutores). | Resultados da v7: o bloco inglês |
| 26 | `f_05_estudio_nao_piorou.png` | Controles de estúdio e português, F0 × T3, e % de reais do CORAA acusados. | Maior queda de AUC: −0,003; o português de audiolivro sobe. | Resultados da v7: sem esquecimento |
| 27 | `f_09_o_que_nao_afirmamos.png` | Quadro: o que o teste mostra × o que não mostra. | Cada item da direita acompanha qualquer número da v7. | **O que não dá para afirmar** |
| 28 | `f_07_cortes_e_calibracao.png` | Nota média de reais e de IA por fonte (T3), as três regras de corte, e TPR/FPR por regra no teste mestre; português destacado. | Com o mesmo corte, de 1% a 75% dos reais acusados; as três regras quase iguais. | **Cortes: decisão em aberto** |
| 29 | `v7_02_media_principal.png` | Média principal e pior fonte por receita, com IC95. | T3 e T4 empatam em 0,963. | Reserva (perguntas sobre a v7) |
| 30 | `v7_01_auc_por_fonte.png` | AUC por receita e fonte (versão da execução da v7). | Mesmo conteúdo da f_03, em outro formato. | Reserva |
| 31 | `v7_03_esquecimento_controles.png` | AUC dos controles por receita e FPR do CORAA. | Nenhuma queda relevante. | Reserva |
| 32 | `v7_04_regras_corte.png` | TPR e FPR das três regras por receita nas 3 fontes principais. | As regras quase não diferem. | Reserva |
| 33 | `v7_05_deslocamento_nota.png` | Faixa de notas de reais e IA por fonte, T0 × T3. | A T3 sobe os falsos da internet acima do corte e comprime a escala. | Reserva (calibração) |
| 34 | `v7_06_bloco_ingles.png` | AUC por receita no bloco inglês (barras). | Mesmo conteúdo da f_06. | Reserva |

Tabelas: `grupos_bancos.csv` (banco, grupo, o que é, real/IA, idioma, ano, papel, nº de áudios), `hiperparametros_versoes.csv` (hiperparâmetros de v1 a v6) e `resumo_v7_para_slides.csv` (uma linha por tipo de fonte: o que representa, nome técnico, AUC do F0 e da T3 no teste mestre, visto no treino, clipes, locutores, observação).
