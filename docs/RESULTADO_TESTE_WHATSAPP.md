# Resultado do teste externo "WhatsApp" (detector de voz + contexto de golpe)

**Data:** 15/11/2026. Protocolo congelado antes das notas: `docs/PROTOCOLO_TESTE_WHATSAPP.md`. Código: `scripts/teste_whatsapp.py`, `scripts/teste_whatsapp_figuras.py`. Tabelas: `results/teste_whatsapp_*.csv/json`. Figuras: `results/slides/tw_01..tw_05.png`. **Nada foi treinado, reajustado ou trocado; o teste rodou uma vez** (`results/teste_whatsapp_concluido.json`). Só `pessoa_id`; nenhuma transcrição, nome ou áudio neste arquivo.

**Dados:** 100 áudios, 23 pessoas (todas com real e IA): 17 `real_sem_golpe`, 27 `real_com_golpe`, 46 `ia_pedido_dinheiro`, 10 `ia_neutro`. Um só gerador (Chatterbox nano). IC95 = bootstrap por pessoa (2.000, seed 42), salvo onde indicado.

## 1. Achado que muda a leitura: o áudio é em português, não em inglês

O Whisper detectou **português nos 100 clipes** (`results/teste_whatsapp_resumo.json`), inclusive nos clones. A alegação da coleta ("tudo em inglês") **não se confirma**. O detector (T3) **nunca viu português no treino**; o módulo de contexto só foi avaliado em pt-BR com texto. Portanto este é um teste em português, não em inglês como o protocolo supunha.

## 2. As duas acurácias (principal)

| modelo | acurácia (IC95) | acurácia balanceada (IC95) | TPR / recall (IC95) | FPR / falso alarme (IC95) | acurácia trivial |
|:--|:--|:--|:--|:--|--:|
| **Detector T3, corte M** (IA = positivo) | **98%** (93,6–100) | 97,7% (92,7–100) | 100% (100–100) | 4,5% (0–14,6) | 56% |
| F0, corte A (comparação) | 87% (80,6–93,1) | 86,4% (79,5–92,8) | 91,1% (82,1–100) | 18,2% (6,8–31,9) | 56% |
| F0, corte B (comparação) | 58% (49,5–67,0) | 62,0% | 28,6% | 4,5% | 56% |
| **Contexto, alerta médio+** (golpe = positivo) | **53%** (42,2–64,4) | 64,3% (55,6–73,4) | 39,7% (28,6–52,2) | 11,1% (0–23,1) | 73% |
| Contexto, só "alto" | 34% (22,2–45,5) | 54,8% | 9,6% (3,9–16,2) | 0% | 73% |

Precisão do contexto médio+: 90,6% (76,5–100). **A acurácia do contexto (53%) fica abaixo da trivial (73%, "chamar tudo de golpe")**. T3 nos cortes A' e F5: idêntico ao M (mesmas 2 falsas acusações). Wilson da acurácia do T3-M: 93,0–99,5% (ignora o agrupamento por pessoa; é otimista). AUC do T3: 1,000 (a AUC do F0: 0,923, IC 0,850–0,983).

## 3. Detector: grupos, canal, pessoas

**Taxa de "IA" por grupo (T3, corte M, condição A):** `real_sem_golpe` 2/17 (11,8%), `real_com_golpe` 0/27, `ia_pedido_dinheiro` 46/46, `ia_neutro` 10/10. As **duas falsas acusações são da mesma pessoa** (p022). F0, corte A: 4/17, 4/27, 43/46, 8/10.

**Condição B (clones recodificados para Opus mono ~16 kbps, ~15,9 kbps medidos):** T3-M acurácia 97% (92,1–100), TPR 98,2% (94–100; 45/46 e 10/10), FPR 4,5% (os reais não mudam), AUC 0,998 (0,990–1,000). F0-A: acurácia 89%, TPR 94,6%. **O resultado não depende do formato 24 kHz × opus.** B não responde a outras diferenças (conteúdo, duração, silêncio).

**Por pessoa:** nota média IA menos real, T3: +13,2 logit (IC 11,8–14,4); **23 de 23 pessoas com IA > real** (menor diferença +6,0). F0: +10,3 (8,4–12,3), 23 de 23 (menor +0,07). Em B: 23 de 23 nos dois.

## 4. Atalho por metadados (cuidado com a leitura do 98%)

Classificador só com duração, RMS, fração de silêncio, taxa original e largura de banda, deixando uma pessoa de fora: **AUC 1,000 com a taxa original** (24 kHz × 48 kHz separa tudo) e **0,987 (0,960–1,000) sem ela**; só duração 0,64; só banda 0,54. **O detector (AUC 1,000 no T3) não é distinguível de um atalho de formato/nível**: com estes dados não dá para afirmar que o T3 "ouve a voz" e não o canal. A condição B (taxa original igualada, AUC 0,998) enfraquece o atalho da taxa, mas não elimina os de nível e silêncio. O T3 só é melhor que os metadados se o atalho for descontado, e o teste não faz isso.

## 5. Contexto de golpe

- **Status:** 100 `ok` (nenhum `indeterminado`, `audio_sem_fala` ou erro); 0 transcrições suspeitas; 0 manipulações detectadas; mediana de 34 palavras (12 a 72); latência total mediana 1,6 s (p90 2,0 s; máx. 3,1 s). Risco: nenhum 36, baixo 32, médio 25, alto 7.
- **Por grupo (alerta médio+):** `real_com_golpe` 2/27 (7,4%), `ia_pedido_dinheiro` 27/46 (58,7%; 7 "alto"), `real_sem_golpe` 3/17 (17,6%), `ia_neutro` 0/10.
- **Reais × IA:** recall em `real_com_golpe` **7,4%** × em clones que pedem dinheiro **58,7%**. **O módulo quase não pega os reais "com golpe"** (gravações de pesquisa, 24 de 27 em "nenhum"); pega mais da metade dos clones que pedem dinheiro. O falso alarme em reais normais (17,6%) é maior que o do teste com textos inventados (3,4%).
- **Em português vs. a avaliação anterior:** recall em textos inventados era 87,7%; aqui 39,7%. Não são comparáveis (texto inventado explícito × fala transcrita de gravações encenadas, com rótulo de "golpe" que pode não ser golpe de fato), mas **a meta de recall ≥ 0,85 não foi atingida neste banco**.
- **Sobreposição de texto:** **0** transcrições quase idênticas aos textos do dev/teste da avaliação anterior ou ao prompt (máx. razão difflib 0,45; máx. Jaccard 0,20). O teste é virgem para o contexto neste aspecto.

## 6. Resultado combinado (regra congelada: voz IA se T3 ≥ M; alerta se risco ≥ médio)

| grupo | baixo | atenção | alto |
|:--|--:|--:|--:|
| real sem golpe (17) | 12 | 5 | 0 |
| real com golpe (27) | 25 | 2 | 0 |
| IA pede dinheiro (46) | 0 | 19 | 27 |
| IA neutra (10) | 0 | 10 | 0 |

A hipótese do protocolo bateu em: IA neutra → atenção (10/10), IA pede dinheiro → alto (59%, o resto atenção), real sem golpe → baixo (71%). **Falhou em real com golpe:** previsto "atenção", ficou "baixo" em 93%, porque a voz é real e o conteúdo quase não gerou alerta. Nada disto é acurácia, só taxas por célula.

## 7. Desvios e ressalvas

1. **Idioma do Whisper:** a chamada usou `idioma="auto"` em vez do padrão `"pt"` (decidido no protocolo, antes das chamadas), para poder detectar o idioma. O módulo e o prompt ficaram iguais. Como o áudio é português, forçar `"pt"` teria dado resultado parecido; não foi testado (não repetimos chamadas).
2. **Fim de linha:** os SHA256 de `prompt_r1.txt` e `src/contexto_golpe.py` no disco diferem dos do congelamento só por CRLF; normalizados para LF, batem (protocolo, seção 0).
3. **Vetores em fp32** (caminho de referência V1), não fp16 como no treino/avaliação originais: diferença esperada de centésimos de logit.
4. Nas linhas "só reais" e "só IA" do contexto (`teste_whatsapp_contexto.csv`) não há IC (só valores pontuais); as linhas por grupo trazem Wilson. Alguns ICs de bootstrap chegam a 1,0 ou 0,0 por serem limitados pelos extremos com 23 pessoas.
5. Os IC de Wilson ignoram o agrupamento por pessoa.

## 8. Limitações

100 áudios, 23 pessoas, um gerador, ICs largos; textos dos reais e dos clones diferentes (o conteúdo pode denunciar a origem); reais com golpe são gravações de pesquisa, não fraude; clones com pedido de dinheiro nem sempre são golpe (rótulo "golpe" é por construção); atalho de metadados (seção 4); idioma português, sem português no treino do detector; nada testado em ligação ao vivo. O contexto depende de um serviço externo e do modelo `gpt-oss-120b` na data do teste.

## 9. Pendências

- **(a)** Conferir as transcrições (idioma e qualidade) antes de interpretar o recall do contexto: o baixo recall pode vir da transcrição e não do julgamento do conteúdo.
- **(b)** O Whisper rodou com `idioma="auto"` (e não `"pt"`, o padrão do módulo); o efeito disso no resultado não foi medido.
- **(c)** O contexto de golpe não deve ser apresentado como resultado final: é um protótipo avaliado só aqui e com textos inventados.
