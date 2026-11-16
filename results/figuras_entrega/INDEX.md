# Figuras da entrega (paper e slides)

Geradas por `scripts/figuras_entrega.py`, só a partir de `results/*.csv|json`. Cada figura tem duas saídas: `slides/` (1920×1080, fonte ≥ 16 pt) e `paper/` (6,5 pol, 300 dpi, fonte ≥ 8 pt). Miniaturas: `folha_resumo.png`. Cores: azul = detector, laranja = contexto, verde = real, vermelho = IA, cinza = congelado/não adotado/referência.

| figura | o que mostra | dados | equivalente já existente em results/slides | onde usar |
|:--|:--|:--|:--|:--|
| `fe_01_fluxo_sistema.png` | O caminho do áudio até o alerta: detector de voz, contexto, combinação. | (diagrama; sem dados) | nenhum equivalente (v8_0_esquema é só da v8) | ambos |
| `fe_02_linha_do_tempo.png` | Os marcos do projeto, uma frase por marco. | v7_principal.csv, v8_principal.csv, modelo_leve_variantes.csv, teste_whatsapp_detector.csv, contexto_golpe_avaliacao_resumo.json, teste_3_ia_notas.csv, diagnostico_whisper_idioma_metricas.csv, ti_* | nenhum | ambos |
| `fe_03_evolucao_versoes.png` | Evolução do detector em painéis (um por teste, não comparáveis). | v4final_09_acuracia_grupos.csv, v7_principal.csv, v8_principal.csv, teste_whatsapp_detector.csv | parcial: 09_acuracia_grupos, f_01_antes_depois | ambos |
| `fe_04_f0_vs_t3_por_fonte.png` | AUC por fonte no teste mestre, F0 x T3, com IC. | v7_teste_mestre.csv, v7_principal.csv | equivalente: v7_01_auc_por_fonte, f_01_antes_depois | ambos (já existe parecida) |
| `fe_05_receitas_t0_t4.png` | Média e pior fonte de T0 a T4, T3 destacada. | v7_principal.csv | equivalente: v7_02_media_principal, f_03_receitas | ambos (já existe parecida) |
| `fe_06_t3_vs_v8.png` | T3 x R1: ganho pequeno e corte instável; não adotado. | v8_principal.csv, v8_escolha.json | equivalente: v8_1_diferenca_vs_t3, v8_3_validacao_x_teste | ambos |
| `fe_07_modelo_leve.png` | Tamanho, tempo e AUC das variantes do modelo leve. | modelo_leve_variantes.csv, modelo_leve_latencia.csv, modelo_leve_equivalencia.csv | equivalente: ml_01_tamanho_e_tempo, ml_02_equivalencia | ambos (já existe parecida) |
| `fe_08_whatsapp_detector.png` | Acurácia, acurácia balanceada, TPR e FPR no WhatsApp, T3 x F0. | teste_whatsapp_detector.csv | parcial: tw_01 (traz o contexto errado) | ambos |
| `fe_09_whatsapp_notas.png` | Notas da T3, reais x clones, com o corte M. | teste_whatsapp_notas.csv | equivalente: tw_03_histograma_notas | ambos (já existe parecida) |
| `fe_10_canal_opus.png` | Condição A x B (clones em Opus): acurácia e AUC. | teste_whatsapp_detector.csv | nenhum | paper |
| `fe_11_atalho_metadados.png` | AUC do detector x classificador só de metadados. | teste_whatsapp_atalho.csv | parcial: ti_05_atalho (outro banco) | ambos |
| `fe_12_estresse_3_ia.png` | Onde caem os 3 áudios do estresse entre reais e clones. | teste_3_ia_notas.csv, teste_whatsapp_notas.csv, v7_t3_modelo.json | nenhum | ambos |
| `fe_13_portugues_corte.png` | Reais acusados com o mesmo corte em cada ambiente. | v7_teste_mestre.csv, v7_controles.csv | equivalente: f_07_cortes_e_calibracao, v7_04_regras_corte | ambos |
| `fe_14_contexto_textos_inventados.png` | Recall e falso alarme do contexto por tipo de texto. | contexto_golpe_avaliacao_resumo.json | equivalente: ctx_01_recall_e_falso_alarme | ambos (já existe parecida) |
| `fe_15_contexto_antes_depois_idioma.png` | Contexto com idioma errado x corrigido, com IC. | diagnostico_whisper_idioma_metricas.csv | equivalente: ctxw_01_acuracia_antes_depois | ambos (já existe parecida) |
| `fe_16_contexto_por_grupo.png` | Alerta do contexto por grupo no WhatsApp (corrigido). | diagnostico_whisper_combinado.csv | substitui tw_02 (contexto errado) | ambos |
| `fe_17_roteiros_claros_vs_leves.png` | Alerta por roteiro: claros 26/26, leves 0/20, dia a dia 0/10. | analise_erro_contexto_por_frase.csv | nenhum | ambos |
| `fe_18_combinado_grupo_nivel.png` | Grupo x nível combinado, com contagens. | diagnostico_whisper_combinado.csv | substitui tw_05 (versão de 15/11) | ambos |
| `fe_19_limitacoes.png` | O que a evidência sustenta e o que não sabemos. | vários (números lidos dos CSV) | equivalente: f_09_o_que_nao_afirmamos | ambos |
| `fe_20_resumo_final.png` | Quatro números-chave do sistema final e as ressalvas. | v7_principal.csv, teste_whatsapp_detector.csv, teste_3_ia_notas.csv, diagnostico_whisper_combinado.csv | nenhum | slide (e paper, como figura 1) |

**Já existem e não foram refeitas:** ml_01_tamanho_e_tempo, ml_02_equivalencia, v7_01 a v7_06, f_00 a f_09, ctx_01 a ctx_03, ctxw_01, tw_03_histograma_notas (não refeitas; equivalentes indicados na coluna acima).

## Figuras puladas

Nenhuma: todos os dados necessários existem em `results/`.

**Observação:** `tw_01`, `tw_02`, `tw_04` e `tw_05` (15/11) trazem o contexto com o idioma do Whisper forçado errado; os substitutos corrigidos são `fe_08`, `fe_16` e `fe_18` (ver `docs/ERRATA_TESTE_WHATSAPP.md`).
