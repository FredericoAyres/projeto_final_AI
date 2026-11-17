# Auditoria das afirmações da apresentação

**Feita em 16/11/2026, só por leitura do repositório** (docs/, results/, notebook.ipynb, README, scripts e histórico do git). Nada foi treinado, nenhuma API foi chamada, nenhum arquivo existente foi alterado. Este arquivo não traz nomes de pessoas nem modelos de máquina: onde eles aparecem no repositório, só o arquivo e a linha são indicados.

Legenda: **CONFERE** = o repositório diz o mesmo número; **CONFERE COM RESSALVA** = o número existe, mas a frase do slide pode enganar; **DIVERGE** = o repositório diz outra coisa; **NÃO ENCONTRADO** = não há o número no repositório.

Números de apoio detalhados: `docs/DADOS_APENDICE.md`.

---

## Parte 1. Afirmações dos slides

### 1. AUC média T0 0,764 → T3 0,963 no teste mestre; In-the-Wild 0,986

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| média T0 0,764 | CONFERE | 0,7635 (IC95 0,730–0,798) | `results/v7_principal.csv` |
| média T3 0,963 | CONFERE | 0,9631 (IC95 0,950–0,973); +0,200 (IC95 pareado 0,168–0,232) | `results/v7_principal.csv` |
| In-the-Wild 0,986 (T3) | CONFERE | 0,986 (IC95 0,973–0,994); T0 0,979 | `results/v7_principal.csv` |

Ressalva para a fala: o In-the-Wild nunca foi visto no treino e os ICs de T0 e T3 se sobrepõem: **não é ganho demonstrado** (`docs/RESULTADO_V7.md`, seção 6). O teste mestre não é virgem.

### 2. Corte M = 0,233; A' = 0,027; F5 = 0,211; as três regras empatam?

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| M = 0,233 | CONFERE | 0,23256 | `results/v7_t3_modelo.json`, `results/v7_escolha.json` |
| M = mediana dos pontos de EER por fonte | CONFERE | "mediana dos pontos de EER de cada fonte com as duas classes" | `docs/PROTOCOLO_V7.md`, seção 4 |
| escolhido só na validação | CONFERE | validação cruzada deixando uma fonte de fora; gravado 19:41:00, antes do teste | `docs/PROTOCOLO_V7.md`, `results/v7_escolha.json` |
| A' = 0,027 | CONFERE | 0,02678 | `results/v7_t3_modelo.json` |
| F5 = 0,211 | CONFERE | 0,21069 | `results/v7_t3_modelo.json` |
| as três empatam | CONFERE | acurácia balanceada (deixando uma fonte de fora) 0,9325 / 0,9331 / 0,9318 na T3; diferença entre regras 0,0004 a 0,010 entre receitas | `results/v7_escolha.json`, `docs/RESULTADO_V7.md` |

Pequena divergência entre documentos: `docs/DECISAO_SISTEMA_PRINCIPAL.md` (motivo 3) escreve "T3 regra M = 0,234"; esse valor é o da T3 **refeita na v8** (R0, 0,2335; `results/v8_escolha.json`). O corte congelado e usado nos testes externos é 0,2326.

### 3. v4 (Meta XLS-R): 0,21 → 0,98 no In-the-Wild; caiu para ~0,7 em ~127 mil áudios de internet; variação entre seeds 0,15 em v1–v3; o que a v3 faz de diferente

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| 0,21 → 0,98 no In-the-Wild | CONFERE COM RESSALVA | 0,211 é a **v1** (CNN, In-the-Wild **inteiro**, recorte central); 0,979 é a **v4 S2** no `test_final` (amostra 5.000/5.000). Amostras diferentes. A comparação da v4 nos mesmos clipes é **0,114 (CNN da v3, S0) → 0,979** (S1, só a camada 24: 0,986) | `docs/HISTORICO_VERSOES.md` (v1 e fechamento da v4), `results/v4_teste_final.csv` |
| caiu para ~0,7 | CONFERE | RADAR eval 0,721 (IC por clipe, otimista); ML-ITW 0,689 (IC95 por locutor 0,613–0,759) | `docs/RESULTADO_TESTE_INTERNET.md` |
| ~127 mil áudios | CONFERE | 102.725 (RADAR) + 24.529 (ML-ITW) = 127.254 | `docs/dados_teste_internet.md` |
| "de internet" | CONFERE COM RESSALVA | só o ML-ITW é internet (vídeos de redes sociais). O RADAR eval é **IA comercial com canal degradado simulado** (102.725 dos 127.254) | `docs/dados_teste_internet.md`, `docs/DECISAO_SISTEMA_PRINCIPAL.md` |
| variação entre seeds 0,15 em v1–v3 | DIVERGE (parcial) | a variação só foi **medida na v3** (3 seeds × 2 condições): amplitude 0,149 (`repeat`) e 0,232 (`zero`); as seis execuções de 0,146 a 0,378. Na v1 e na v2 houve 1 execução por experimento (a v2 estimou ~0,06, depois considerado subestimado). O roteiro diz "até 0,23" | `docs/HISTORICO_VERSOES.md` (fechamento da v3), `docs/ROTEIRO_APRESENTACAO.md` slide 5 |
| o que a v3 faz de diferente | (descrição) | completa áudios < 4 s **com zeros** em vez de **repetir**; 3 seeds por condição; comparação pareada pré-registrada. Resultado: sem efeito demonstrado (Δ EER do dev_shift −3,1 / +20,1 / −12,8 pontos); as seis AUCs abaixo de 0,5 | `docs/HISTORICO_VERSOES.md` (v3) |

### 4. Fine-tuning R1

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| +0,004 a +0,005 de AUC | CONFERE | +0,0040 (IC95 0,0025–0,0056) seed 42; +0,0049 (0,0030–0,0071) seed 123 | `results/v8_principal.csv` |
| cortes 0,71 vs −0,02 | CONFERE | A' = 0,713 (seed 42) e −0,024 (seed 123) | `results/v8_escolha.json` |
| 2 sementes | CONFERE | 42 e 123 | `results/v8_escolha.json` |
| não adotado | CONFERE | decisão do grupo de 15/11, posterior ao protocolo | `docs/DECISAO_SISTEMA_PRINCIPAL.md` |

Para a fala: a regra da v8 **escolheu a R1** pela validação (0,9758 × 0,9750); "não adotado" é decisão posterior, tomada depois de ver o teste.

### 5. Modelo leve

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| corte após a camada 7 | CONFERE | V1 = cortado depois da 7ª camada; 7 de 24 camadas executadas | `docs/MODELO_LEVE.md` |
| 386 MB fp32 | CONFERE | 386,4 (modelo) / 386,5 (com regressão) | `results/modelo_leve_variantes.csv` |
| 193 MB fp16 | CONFERE | 193,2 / 193,3 | idem |
| 133 MB int8 | CONFERE | 133,0 / 133,1 | idem |
| "6× menor" | NÃO ENCONTRADO | o repositório diz **3,1 vezes menor** (V1 386 MB × V0 1.211 MB; `docs/HISTORICO_VERSOES.md`) e "cerca de 3 vezes" (`notebook.ipynb`). "6×" só sai como conta: **V2 fp16 cortado (193 MB) contra o modelo completo fp32 (1.211 MB) = 6,3×**, misturando corte e fp16. O int8 dá 9,1× | `results/modelo_leve_variantes.csv` |
| fp16 mantém as mesmas decisões? | CONFERE COM RESSALVA | contra o completo **em fp16**: notas idênticas (diferença 0). Contra o completo **em fp32** (referência da T3): até 0,077 logit, AUC igual; **decisões no corte M: 1 de 2.000 muda** (0,05%) — **calculado nesta auditoria** (o repositório não mede decisões). O int8 muda 67 de 2.000 no corte M e precisa de corte próprio | `results/modelo_leve_equivalencia.csv`; `docs/DADOS_APENDICE.md`, seção H |

### 6. Português

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| 75% dos reais acusados no audiolivro (BRSpeech-DF) | CONFERE | 75,3% (T3, corte M); 75,2% na T3 refeita da v8 | `results/v7_controles.csv`, `results/v8_controles.csv` |
| 11% no CORAA | CONFERE | 11,1% (corte M); 12,2% no A' | `results/v7_controles.csv` |

Para a fala: a AUC no BRSpeech-DF é alta (0,956); o problema é o corte. Não generalizar para "em português 75%" (`docs/ROTEIRO_APRESENTACAO.md`, slide 18).

### 7. Contexto

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| prompt r1 congelado | CONFERE | congelado em 2026-11-15 17:45:57, SHA256 `3f848dbf…` | `docs/PROTOCOLO_CONTEXTO.md`, seção 7; `results/contexto_golpe_congelado.json` |
| "iteração 1 de 9 golpes disfarçados" | CONFERE | no dev, o prompt r0 pegou 1 de 9 golpes escondidos; o r1, 8 de 9 (uma revisão de prompt, de 3 permitidas) | `docs/AVALIACAO_CONTEXTO.md`, seção 3 |
| 40 textos de desenvolvimento | CONFERE | 40 | idem, seção 2 |
| 130 textos de teste | CONFERE | 130 (115 não ambíguos + 15 ambíguos) | idem |
| recall 88% (50/57) | CONFERE | 87,7% (50/57; IC95 77–94%); escondidos 80% (abaixo da meta de 85%) | `docs/AVALIACAO_CONTEXTO.md`, seção 4 |
| falso alarme 3% | CONFERE | 3,4% (2/58; IC95 1–12%) | idem |
| acurácia 92% (115 não ambíguos) | CONFERE | 92,2% (106/115; IC95 86–96%) | idem |
| idioma "auto" | CONFERE COM RESSALVA | `auto` foi escolhido na **reavaliação do WhatsApp** (16/11) e é a recomendação para o site. A avaliação com 130 textos **não passou pelo Whisper**. O padrão no código ainda é `"pt"`. Português no modo `auto` **não validado** | `docs/RESULTADO_CONTEXTO_WHISPER.md`, `docs/CONTEXTO_GOLPE.md` (linha 22) |

### 8. Banco WhatsApp

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| 23 pessoas | CONFERE | 23 (`pessoa_id`) | `docs/RESULTADO_TESTE_WHATSAPP.md` |
| 44 reais | CONFERE | 17 sem golpe + 27 "com golpe" | idem |
| 56 clones (46 golpe, 10 neutros) | CONFERE | 46 pedem dinheiro + 10 neutros | idem |
| detector 98% (IC 94–100) | CONFERE | 98% (IC95 93,6–100) | `results/teste_whatsapp_detector.csv` |
| 2 de 44 reais acusados | CONFERE | 2 (FPR 4,5%), as duas da mesma pessoa | idem; `results/teste_whatsapp_notas.csv` |
| 97% recodificado | CONFERE | 97% (IC95 92,1–100), clones em Opus ~16 kbps | `results/teste_whatsapp_detector.csv` |
| contexto 26/26 e 0/20 | CONFERE | roteiros claros 26/26; pedidos leves 0/20 | `docs/ANALISE_ERRO_CONTEXTO.md` |
| acurácia geral 50% | CONFERE COM RESSALVA | 50% nos 100 clipes juntos (descritivo; trivial 73%; 40 clipes usados no diagnóstico). O número principal é o do **RESERVADO: 45%** (trivial 81,7%). O de 15/11 (53%) é inválido (errata) | `notebook.ipynb` (parte G), `docs/RESULTADO_CONTEXTO_WHISPER.md` |
| nenhuma IA em "baixo" no combinado | CONFERE | 0 de 56 (corrigido e 15/11) | `docs/RESULTADO_COMBINADO_CORRIGIDO.md` |

Ressalva obrigatória: um classificador **só com metadados** chega a AUC 1,000 (0,987 sem a taxa original): o 98% não prova que o detector "ouve a voz" (`docs/RESULTADO_TESTE_WHATSAPP.md`, seção 4).

### 9. Teste de estresse

| afirmação | status | valor no repositório | arquivo |
|:--|:--|:--|:--|
| 1 de 3 áudios do Dessa detectados | CONFERE | só `ia_3` (nota 2,72) passa do corte M; `ia_1` −0,60 e `ia_2` −3,38 | `results/teste_3_ia_notas.csv`, `docs/TESTE_ESTRESSE_3_IA.md` |
| Dessa anunciado em 23/10/2026 | NÃO ENCONTRADO | o repositório só cita a página do produto, sem data | `docs/TESTE_ESTRESSE_3_IA.md` |

---

## Parte 2. Buscas adicionais

### (a) Números contraditórios entre documentos

| tema | onde | o que diverge | gravidade |
|:--|:--|:--|:--|
| corte M | `docs/DECISAO_SISTEMA_PRINCIPAL.md` (motivo 3) × `results/v7_t3_modelo.json` | 0,234 (T3 refeita na v8) × 0,233 (congelado) | baixa |
| reais acusados BRSpeech-DF | `docs/RESULTADO_V7.md` × `docs/DECISAO_SISTEMA_PRINCIPAL.md` | 75,3% (v7) × 75,2% (T3 refeita na v8); o roteiro já explica | baixa |
| tamanho do modelo leve | `docs/HISTORICO_VERSOES.md` linha 698 e `notebook.ipynb` × slide | "3,1 vezes" / "cerca de 3 vezes" × "6×" | **média** (dizer contra o quê) |
| variação entre seeds | `docs/ROTEIRO_APRESENTACAO.md` slide 5 × slide atual | "0,146 a 0,378 (até 0,23)" × "0,15 em v1–v3" | **média** |
| passo zero da v8 | `results/v8_checagem.json` × `docs/PROTOCOLO_V8.md` linha 40 × `src/finetune.py` linha 14 | diferença para a T3: máx. 0,957 / mediana 0,071 (512 clipes) × máx. 0,012 / mediana 0,003 (48 clipes) × mediana 0,005 | **média** (ver e) |
| roteiro desatualizado | `docs/ROTEIRO_APRESENTACAO.md` slide 21 e "o que ainda precisa ser feito" | diz que o modelo leve e o contexto de golpe "não existem"; ambos existem desde 15/11 (`docs/MODELO_LEVE.md`, `docs/AVALIACAO_CONTEXTO.md`) | **média** (se o roteiro for usado) |
| idioma do WhatsApp | `docs/RESULTADO_TESTE_WHATSAPP.md` linhas 7, 9, 56, 64, 69; `docs/PROTOCOLO_TESTE_WHATSAPP.md` 56, 82; `docs/HISTORICO_VERSOES.md` 974, 979, 980 | "português nos 100 clipes", "idioma=auto", acurácia 53%: **errados**, corrigidos pela errata, mas o texto antigo continua lá | média (já tem errata) |
| idioma padrão do módulo | `src/contexto_golpe.py` (padrão `"pt"`) × `docs/RESULTADO_CONTEXTO_WHISPER.md` (recomenda `auto`) | o próprio `docs/CONTEXTO_GOLPE.md` linha 22 avisa | baixa |

### (b) Frases que exageram

| frase | onde | problema |
|:--|:--|:--|
| "Treinamos em inglês e testamos também em português" | `docs/ROTEIRO_APRESENTACAO.md` slide 4; `CLAUDE.md` linha 3 ("treino em inglês"); `README.md` ("voz em inglês") | **Falso para o sistema principal.** O treino da T3 inclui RADAR em chinês, chinês de Taiwan e japonês (15.316 / 15.316) e ML-ITW com outros idiomas além do inglês (sorteio por grupo de idioma; `results/v7_splits_mlitw_regras.csv`). Verdadeiro até a v6 (F0). Correto dizer: "sem português no treino". |
| "melhor que a literatura" / "generaliza para qualquer gerador" | `scripts/nb_entrega.py` linha 473 e `notebook.ipynb` | Aparecem **só negadas** ("não afirmamos..."). OK. |
| números de um artigo (Interspeech 2024, XLS-R 2B: RL 6,5% × MLP 8,2%) | `notebook.ipynb` (seção sobre a regressão) | Marcados como "NÃO VERIFICADO POR NÓS". Não usar em slide sem ler a fonte. |
| "generaliza" | `notebook.ipynb` | Usado com ressalva ("não é a mesma coisa que generalizar para geradores novos"). OK. |

### (c) Nomes de pessoas, máquinas e segredos

- **Listas `PROIBIDOS` versionadas com nomes próprios:** `scripts/nb_final.py` linha 30, `scripts/nb_v7.py` linha 35 e `scripts/nb_principal.py` linha 40 contêm, em texto, nomes e sobrenomes de pessoas do grupo, o nome de usuário do autor e nomes de figuras públicas. No total, **12 ocorrências de nomes do grupo/usuário em 7 arquivos versionados**: os 3 scripts acima, `scripts/nb_v8.py`, `docs/HISTORICO_VERSOES.md` (linha 669, URL da release com o nome de usuário do GitHub), `results/v8_r1_pesos.json` e `notebooks/versions/v8_finetuning.ipynb`. O `scripts/nb_entrega.py` já faz certo: lê os nomes de um arquivo local fora do git.
- **Figuras públicas (locutores do In-the-Wild/ML-ITW):** 100 ocorrências em 16 arquivos (por exemplo `docs/HISTORICO_VERSOES.md`, `docs/PROTOCOLO_V7.md`, `docs/audit_report.md`, `results/v5_comparacao_locutores.csv`, `notebooks/versions/v5_wavlm.ipynb`). São metadados dos bancos públicos; decidir se ficam.
- **Participantes do WhatsApp:** nenhum nome vazou. Os 6 "acertos" da busca automática contra os nomes locais são coincidências com palavras comuns ou nomes inventados nos textos fictícios (conferido).
- **Modelos de placa de vídeo/máquina em arquivos versionados:** `CLAUDE.md` linha 7; `docs/HISTORICO_VERSOES.md` linhas 59, 86, 138, 264, 400, 592, 920, 951; `docs/PROTOCOLO_V8.md` linha 65; `src/finetune.py` linha 173; `results/v6_medir_concluido.json` linha 61; `results/v6_wavlm_vazao.json` linha 13; e as próprias listas `PROIBIDOS`. O texto do `notebook.ipynb` está limpo (os acertos da busca estavam dentro de imagens codificadas).
- **Chaves e segredos no histórico do git:** busca com `git log -p --all` (25 commits) por chaves do Groq, do Hugging Face, do tipo `sk-`, da AWS, tokens do GitHub, `GROQ_API_KEY=` com valor, `api_key` com valor e chaves privadas: **0 ocorrências**. Nenhum `.env` versionado. Limite: só padrões de texto; arquivos binários não foram inspecionados.

### (d) Figuras antigas do WhatsApp

- `results/slides/tw_01`, `tw_02`, `tw_04` e `tw_05` (contexto com o Whisper forçado a português) **continuam em `results/slides/` e listadas em `results/slides/INDEX.md` (linhas 72, 73, 75, 76) sem aviso**; a legenda do `tw_01` ainda diz "a do contexto fica abaixo da acurácia trivial".
- O `notebook.ipynb` **não** as usa (só `tw_03`, que não depende do contexto).
- `results/figuras_entrega/INDEX.md` avisa e indica os substitutos (`fe_08`, `fe_16`, `fe_18`). `docs/RESULTADO_TESTE_WHATSAPP.md` ainda as cita (coberto pela errata).
- **Risco:** quem montar slides a partir de `results/slides/INDEX.md` pega a figura errada.

### (e) Inconsistência em `results/v8_checagem.json`

- O arquivo (2026-11-14 11:48, antes do carimbo das 14:42, 512 clipes) registra diferença entre o passo zero e a T3 de **0,957 logit no máximo e 0,071 na mediana** (correlação 0,9997; AUC 0,9773 × 0,9774).
- O `docs/PROTOCOLO_V8.md` (linhas 40 e 114) diz **0,012 no máximo e 0,003 na mediana, em 48 clipes**, e que o bfloat16 dava 0,68 / 0,08. O `src/finetune.py` (linha 14) diz mediana 0,005.
- A mediana 0,071 do JSON é parecida com a do bfloat16 (0,08). **Hipótese (não verificada):** o JSON versionado é de uma rodada anterior à troca para float16 e não foi refeito. O repositório não guarda o tipo numérico usado nessa rodada, então não dá para confirmar.
- Outro número desatualizado no mesmo arquivo: estimativa de 13,7 h para a fila (a real foi ~11 h, `docs/HISTORICO_VERSOES.md`).
- Não afeta os resultados da v8 (a escolha e o teste usam outras etapas), mas o arquivo **não sustenta** a frase "o passo zero reproduz a T3".

### (f) README e licenças

- **Não há arquivo de licença do código** (nenhum `LICENSE` ou `COPYING` na raiz). Se o repositório for público, sem licença ninguém pode reutilizar o código formalmente.
- **README desatualizado:** descreve `results/` como só `baseline_results.csv` e `experiments.csv`; não cita o sistema principal (v7, T3), os pesos `results/v7_t3_coeficientes.npz`, o modelo leve, o contexto de golpe, o RADAR, o ML-ITW nem o teste WhatsApp; diz "voz em inglês" (ver b).
- **`docs/datasets.yaml` não registra o RADAR nem o ML-ITW** (só `docs/dados_teste_internet.md` fala deles).
- **Licenças dos dados:**
  - ML-ITW: licença **não declarada** (`docs/dados_teste_internet.md` linha 42) e usado no treino da T3.
  - RADAR eval: CC BY 4.0, mas o desafio pede para **não treinar com o eval**; nós treinamos (já declarado: os números não valem para o desafio).
  - CORAA: CC BY-NC-ND 4.0 (só uso no projeto, sem redistribuir).
  - DFADD, ShiftySpeech, CodecFake+, In-the-Wild, LlamaPartialSpoof e ASVspoof 2021: licença `null` no `docs/datasets.yaml`.
  - `data/processed/manifest.parquet` (versionado) traz caminhos e ids de locutor desses bancos.
- **Consentimento dos 23 participantes do WhatsApp** (voz clonada e áudio enviado ao Groq): **NÃO ENCONTRADO** registro no repositório. O `CLAUDE.md` proíbe enviar "áudio dos bancos do projeto" ao serviço externo. Vale confirmar se o banco WhatsApp tinha autorização para isso.

---

## Divergências graves (resumo)

1. **"Treinamos em inglês"** é falso para a T3 (treino com chinês, chinês de Taiwan, japonês e ML-ITW multilíngue). Trocar por "sem português no treino".
2. **"6× menor"** não está no repositório. O documentado é 3,1× (corte). 6,3× só comparando o fp16 cortado com o completo em fp32: dizer isso no slide.
3. **"0,15 de variação entre seeds em v1–v3"**: só foi medida na v3, com amplitude de 0,15 a 0,23 (0,146 a 0,378 nas seis execuções).
4. **"0,21 → 0,98"** compara a v1 no In-the-Wild inteiro com a v4 no `test_final`. Nos mesmos clipes: 0,114 → 0,979.
5. **"~127 mil áudios de internet"**: 102.725 deles (RADAR) são IA comercial com canal degradado simulado, não internet.
6. **Data do Dessa (23/10/2026)**: NÃO ENCONTRADO. Precisa de fonte externa.
7. **Figuras `tw_01/02/04/05`** ainda listadas em `results/slides/INDEX.md` sem aviso. Usar `fe_08`, `fe_16` e `fe_18`.
8. **Nomes de pessoas** (grupo e usuário) em 7 arquivos versionados e **modelos de placa** em pelo menos 13 lugares. Nenhum segredo encontrado no histórico do git.
9. **`results/v8_checagem.json`** contradiz o protocolo da v8 (0,957 × 0,012 de diferença máxima no passo zero).
10. **Sem licença do código, README desatualizado, ML-ITW sem licença e consentimento do WhatsApp não registrado.**

## Dúvidas para o grupo decidir (não decidi nada)

- Remover os nomes das listas `PROIBIDOS` versionadas (passar a ler de um arquivo local, como já faz o `nb_entrega.py`) e tirar o nome de usuário da URL da release?
- Manter ou anonimizar os nomes de figuras públicas (locutores dos bancos) nos documentos?
- Acrescentar um aviso em `results/slides/INDEX.md` para as figuras `tw_01/02/04/05`, ou removê-las?
- Refazer o `results/v8_checagem.json` ou anotar no protocolo que ele é de uma rodada anterior?
- Escolher uma licença para o código e atualizar o README?
- Confirmar o registro de consentimento dos participantes do WhatsApp.
