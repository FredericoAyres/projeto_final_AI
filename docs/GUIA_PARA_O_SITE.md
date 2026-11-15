# Guia para quem vai construir um site com o sistema principal (v7, T3)

**Data:** 15/11/2026. **Para quem:** quem vai colocar o detector num site (servidor ou navegador). **O que este guia NÃO é:** um resultado novo. Nada foi treinado nem reavaliado para escrevê-lo; todos os números vêm de `results/` e `docs/` (lista no fim).

Documentos relacionados: `docs/DECISAO_SISTEMA_PRINCIPAL.md` (o que é o sistema principal), `docs/PROTOCOLO_V7.md` e `docs/RESULTADO_V7.md` (a v7), `docs/MODELO_LEVE.md` (as variantes leves). Gabarito em código: `scripts/inferencia_referencia.py`. Notebook: `notebook.ipynb`, partes A, B e C.

> **Dois nomes parecidos.** A **variante leve V3 (int8)** do modelo leve **não é** a **v3 do projeto** (a versão do preenchimento de áudios curtos). Neste guia, "V3" sempre quer dizer "variante leve V3 (int8)".

## 1. O caminho do áudio até a resposta

Cada passo foi conferido no código (o arquivo e a função estão à direita).

```
arquivo de áudio
  │  ler e converter para MONO (média dos canais) e 16 kHz       src/preproc.py: ler_mono_16k
  │  (reamostra com o filtro anti-aliasing do torchaudio sempre que a taxa for outra)
  ▼
cortar o silêncio das bordas: quadros de 25 ms (passo 10 ms) com energia
35 dB abaixo do quadro mais forte são silêncio
  │  se sobrar menos de 0,5 s, usa o áudio sem corte              src/preproc.py: preparar_clipe
  ▼
recorte CENTRAL de no máximo 8 s, SEM completar (áudio curto fica com o próprio
tamanho; só abaixo de 400 amostras, 25 ms, completa com zeros)
  ▼
normalizar o clipe: média 0, variância 1                          src/ssl.py: normalizar
  ▼
XLS-R 300M (Meta), CONGELADO e cortado depois da 7ª camada         src/ssl.py: representar
  ▼                                                                (corte: scripts/modelo_leve.py: montar)
saídas das camadas 5, 6 e 7 (hidden_states[5], [6], [7]; 1.024 números a cada 20 ms)
  ▼
média no tempo de cada uma (só nos quadros válidos) → 3 × 1.024 números
  ▼
colar as três, NESSA ORDEM (5, 6, 7) → x com 3.072 números
  ▼
padronizar: (x − mu) / sd, com mu e sd do treino da T3            results/v7_t3_coeficientes.npz
  ▼
nota = (x padronizado) · coef + intercepto                        results/v7_t3_modelo.json ("como_usar")
  │  (logit; pode ser negativa; maior = mais parecido com IA)
  ▼
comparar com o corte → "parece real" ou "parece gerada por IA"
```

Detalhes que importam:
- **Camadas.** No `transformers` usado (5.22.2), `hidden_states[k]` (k = 1 a 24) é a saída **crua** da camada k; o layer norm final só entra na última saída. No modelo cortado, esse layer norm vira **identidade** (senão ele seria aplicado à saída 7 e mudaria a nota). Fonte: `docs/MODELO_LEVE_PROTOCOLO.md`.
- **Cada áudio é normalizado sozinho** antes de entrar no modelo (como o `Wav2Vec2FeatureExtractor` com `do_normalize`).
- **Lote de 1 áudio** no site: não há enchimento, e todos os quadros são válidos.
- O cálculo da padronização e da nota é feito em **float32**.

## 2. Arquivos necessários

| o quê | onde | no git? |
|:--|:--|:--|
| pesos da regressão: `coef` (3.072, float32), `intercepto` (1 número, 0,306), `mu` e `sd` (3.072 cada, float32), `C` (0,01), `camadas` ([5, 6, 7]) | `results/v7_t3_coeficientes.npz` (0,04 MB) | sim |
| como usar, hiperparâmetros, composição do treino e os três cortes da validação | `results/v7_t3_modelo.json` | sim |
| modelo da Meta completo (V0) | Hugging Face, identificador **`facebook/wav2vec2-xls-r-300m`** (`src/ssl.py: NOME_PADRAO`); arquivo `pytorch_model.bin`, 1.210,9 MB | não |
| V1: cortado, fp32 | `models/modelo_leve_V1_fp32.safetensors` (386,4 MB) | **não** |
| V2: cortado, fp16 | `models/modelo_leve_V2_fp16.safetensors` (193,2 MB) | **não** |
| variante leve V3 (int8) | `models/modelo_leve_V3_int8.pt` (133,0 MB) | **não** |

**Como regerar os modelos leves** (sem treinar nada; precisa do modelo da Meta no cache local do Hugging Face):
- `python scripts/inferencia_referencia.py --verificar` gera, a partir do cache, só o arquivo da variante pedida se ele faltar (conferido: o V1 gerado assim saiu idêntico, tensor por tensor, ao de `models/`);
- `python scripts/modelo_leve.py tamanho` gera os três (e também regrava `results/modelo_leve_parcial/tamanho.csv`).

Os arquivos V1 e V2 guardam só os pesos (`state_dict`). Para carregar: monte um `Wav2Vec2Model` com o `config.json` do modelo da Meta e `num_hidden_layers = 7`, troque `encoder.layer_norm` por identidade e carregue os pesos (`scripts/inferencia_referencia.py: esqueleto_cortado` e `carregar`). A variante leve V3 (int8) precisa da mesma montagem seguida de `torch.ao.quantization.quantize_dynamic` antes de carregar o arquivo.

## 3. Qual variante usar

Números de `results/modelo_leve_variantes.csv`, `modelo_leve_latencia.csv` e `modelo_leve_equivalencia.csv` (mediana de 30 medidas, lote de 1, áudio de 5 s, incluindo ler o arquivo, preparar, o modelo e a regressão):

| variante | tamanho (MB, modelo + regressão) | placa de vídeo do notebook | processador do notebook, 4 threads | processador do notebook, 1 thread | nota igual à do modelo completo? |
|:--|--:|--:|--:|--:|:--|
| V0 completo | 1.211 | 18 ms (fp16) | 1.256 ms | 1.975 ms | (referência) |
| **V1** cortado, fp32 | 386 | 21 ms | 529 ms | 951 ms | **sim** (diferença 0 contra V0 fp32) |
| **V2** cortado, fp16 | 193 | **10 ms** | – | – | **sim** (diferença 0 contra V0 fp16) |
| variante leve V3 (int8) | 133 | – | **350 ms** | 645 ms | **não** (até 4,53 logit; mediana 1,07) |

**Recomendação (a decisão é do grupo):**
- **V2** é o modelo leve recomendado quando houver placa de vídeo.
- **V1** é a referência e a opção para processador quando a nota precisa ser exatamente a da T3.
- **Variante leve V3 (int8): com ressalva.** A AUC cai pouco (0,0025 na média e 0,0055 na pior fonte, dentro da regra), mas **o valor da nota muda**, então **o corte da T3 não vale nela**. Seria preciso escolher um corte novo, só pela validação, e **isso não foi feito**. Ela também não foi medida no teste mestre. Medido agora pelo `--verificar`: a nota dela muda também com o **tamanho do lote** (seção 8).

Os tempos são de um notebook, não de um celular: o processador é só uma referência.

## 4. O corte

A nota é um número; para dizer "real" ou "IA", é preciso um **corte**. A v7 calculou **três regras, só na validação** (`results/v7_t3_modelo.json`, `results/v7_escolha.json`):

| regra | o que é | corte (logit) | acurácia balanceada média, deixando uma fonte de fora |
|:--|:--|--:|--:|
| A' | ponto de EER da validação inteira | 0,027 | 0,9325 |
| **M** | mediana dos pontos de EER de cada fonte | **0,233** | **0,9331** (a escolhida) |
| F5 | no máximo 5% de reais acusados em cada fonte (mediana entre fontes) | 0,211 | 0,9318 |

As três empatam na prática. **A regra final para o produto é DECISÃO EM ABERTO.** O script de referência usa a M por padrão (`--corte` muda o valor).

**Com o corte M**, no teste mestre e nos controles da T3 (`results/v7_teste_mestre.csv`; TPR = % das vozes de IA detectadas; FPR = % das vozes reais acusadas):

| tipo de áudio | papel | reais / IA | TPR | FPR |
|:--|:--|--:|--:|--:|
| internet real (ML-ITW) | principal | 6.612 / 2.513 | 79,2% | 5,8% |
| entrevistas e discursos da internet (In-the-Wild, 38 locutores; nunca visto no treino) | principal | 5.000 / 5.000 | 97,8% | 15,6% |
| IA moderna em canal degradado, inglês (RADAR en) | principal | 15.000 / 17.923 | 92,2% | 14,0% |
| RADAR en, durações pareadas | inglês | 14.291 / 14.291 | 92,2% | 14,0% |
| internet real, só inglês (indício) | inglês | 1.984 / 586 | 70,8% | 3,8% |
| internet real, só português (indício: 7 / 8 locutores) | à parte | 911 / 108 | 51,9% | 4,7% |
| falso parcial, estúdio (LlamaPartialSpoof) | controle | 5.000 / 5.000 | 82,9% | 7,5% |
| português de audiolivro (BRSpeech-DF) | controle | 1.320 / 5.000 | 99,6% | **75,3%** |
| português falado, só reais (CORAA) | controle | 5.000 / 0 | – | 11,1% |
| estúdio (ASVspoof 2019) | controle | 5.000 / 5.000 | 95,1% | 0,8% |
| estúdio (ASVspoof 2021 DF) | controle | 5.000 / 5.000 | 98,8% | 13,7% |

## 5. Contrato sugerido para o site

**Entrada**
- Um arquivo de áudio de **qualquer formato que o ffmpeg leia**, convertido para **mono 16 kHz** antes do modelo. O script de referência lê com o soundfile (WAV, FLAC, OGG, MP3); a conversão de outros formatos (m4a, aac, opus em webm...) pelo ffmpeg **não foi testada**.
- **O modelo só usa os 8 s centrais** (depois do corte de silêncio). Vale avisar isso ao usuário.
- **Ideia NÃO testada** para áudios longos: avaliar janelas de 8 s e combinar as notas (por exemplo, a média ou a maior). Nada disso foi medido; precisaria de validação antes de ir para o site.

**Saída**
- A **nota** (logit).
- Um **nível de alerta: "baixo / atenção / alto"**, com cortes **ainda a decidir** (só pela validação).
- Sempre a mensagem: **"Isto é um alerta, não um veredito."**

**O que o site NÃO deve prometer**
- Que detecta **qualquer** voz de IA (o RADAR en mede idioma novo, não gerador novo; `docs/DECISAO_SISTEMA_PRINCIPAL.md`, seção 4).
- Que funciona em **ligações, WhatsApp ou celular** (nada disso foi testado).
- Que funciona igual em **português** (ver os limites abaixo).
- Uma **"probabilidade de ser IA"**: a escala da nota muda entre ambientes.

## 6. Limites honestos

- **Nenhum áudio em português entrou no treino.** O estúdio é inglês; da internet real (ML-ITW), todo locutor com algum áudio em português foi para o teste (regra 1 do protocolo); o RADAR do treino é chinês, chinês de Taiwan e japonês.
- **No português de audiolivro (BRSpeech-DF), cerca de 75% dos reais são acusados** com o corte M; **no português falado (CORAA), cerca de 11%**.
- **O corte não se transfere entre ambientes:** com o mesmo corte M, o falso alarme vai de 0,8% (ASVspoof 2019) a 75,3% (BRSpeech-DF).
- **Nada foi testado em ligações, WhatsApp nem celular.**
- O teste mestre **não é virgem** (`docs/RESULTADO_V7.md`, seção 7).
- **Privacidade (RECOMENDAÇÃO, não resultado):** processar o áudio no próprio aparelho do usuário sempre que der; **não guardar o áudio sem consentimento**; se precisar guardar algo, guardar só a nota.

## 7. Checklist de integração

- [ ] O áudio chega ao modelo em **mono** e **16 kHz** (16.000 amostras por segundo), em float32 entre −1 e 1.
- [ ] O **corte de silêncio** usa 35 dB, quadros de 400 amostras e passo de 160, e é desfeito se sobrar menos de 0,5 s.
- [ ] O **recorte é central**, de no máximo 8 s (128.000 amostras), **sem completar** áudios curtos.
- [ ] O clipe é **normalizado** (média 0, variância 1) antes do modelo.
- [ ] O modelo é o XLS-R 300M **cortado depois da 7ª camada**, com o **layer norm final trocado por identidade**.
- [ ] As saídas lidas são `hidden_states[5]`, `[6]` e `[7]`, nessa ordem, com **média no tempo**.
- [ ] A **padronização** usa `mu` e `sd` de `results/v7_t3_coeficientes.npz` (não recalcular com os áudios do site).
- [ ] `nota = ((x − mu) / sd) · coef + intercepto`, em float32.
- [ ] O corte é o da variante certa (o da T3 **só** para V0, V1 e V2).
- [ ] O site mostra "Isto é um alerta, não um veredito."
- [ ] Para alguns arquivos de teste, a nota do site bate com a de `scripts/inferencia_referencia.py` (seção 8).

## 8. Erros comuns

| erro | o que acontece |
|:--|:--|
| **taxa de amostragem diferente de 16 kHz** (por exemplo, 44,1 ou 48 kHz direto no modelo) | o modelo "ouve" o áudio acelerado ou lento; a nota perde o sentido |
| **estéreo não convertido para mono** | o tensor tem a forma errada ou só um canal é usado; o projeto usa a **média dos canais** |
| **esquecer de padronizar** (usar x direto com `coef`) | a nota sai numa escala completamente diferente e o corte não vale |
| **usar a saída final do encoder** (`last_hidden_state`, camada 24 com layer norm) em vez das camadas 5, 6 e 7 | outros números; a regressão foi treinada nas camadas 5, 6 e 7 |
| **cortar o modelo sem trocar o layer norm final por identidade** | a saída 7 passa por um layer norm que no modelo inteiro ela não tinha; a nota muda |
| **esquecer o corte de silêncio** | o recorte de 8 s pega silêncio das bordas; a nota muda |
| **completar áudios curtos** (com zeros ou repetindo) | o projeto não completa (só abaixo de 25 ms); a nota muda |
| **usar o corte da T3 na variante leve V3 (int8)** | a nota da V3 muda até 4,53 logit; o corte não foi validado para ela |
| **comparar notas da variante leve V3 (int8) calculadas em lotes diferentes** | na V3 a nota também depende do lote (abaixo) |

## 9. Como testar

1. Rode a verificação do script de referência:
   ```
   env\Scripts\activate
   python scripts/inferencia_referencia.py --verificar        # 20 clipes, variante v1 (processador)
   python scripts/inferencia_referencia.py --verificar 50 --variante v2
   ```
   Ele relê os arquivos originais de 20 clipes sorteados (seed 42) da amostra da validação do modelo leve (`data/processed/modelo_leve_amostra/`), refaz todo o caminho e compara: (a) o clipe preparado com o guardado; (b) a nota com a do modelo completo em fp32 (critério da V1: ≤ 0,001 logit); (c) a nota com a nota da T3 gravada (`data/processed/v7_notas.parquet`, informativo).
2. Rode um arquivo seu: `python scripts/inferencia_referencia.py meu_audio.wav` e compare com a nota que o site dá para o **mesmo arquivo**. Com V1 ou V2, as notas devem bater até a 3ª casa (fp32) ou a 2ª casa (fp16).

**Resultado da verificação feita em 15/11/2026** (20 clipes, processador do notebook e placa de vídeo do notebook):

| variante | clipe preparado × guardado | nota × referência da amostra | nota × T3 gravada (`v7_notas.parquet`) | mesma decisão no corte M |
|:--|:--|:--|:--|:--|
| v1 (fp32, processador) | idêntico (0) | máx 2,6e-5, mediana 7,9e-6 (contra V0 fp32) → **passou** (≤ 1e-3) | máx 0,034, mediana 0,011 | 20 de 20 |
| v2 (fp16, placa de vídeo) | idêntico (0) | máx 0,0097, mediana 0,0032 (contra a V2 gravada, calculada em lotes de 8) | máx 0,015, mediana 0,005 | 20 de 20 |
| variante leve V3 (int8, processador) | idêntico (0) | máx 1,41, mediana 0,57 (contra a V3 gravada, calculada em lotes de 8) | máx 1,60, mediana 0,84 | 19 de 20 |

Por que a nota da T3 gravada difere em centésimos: ela foi calculada a partir de vetores extraídos em fp16 na placa de vídeo e **guardados em fp16** (`docs/MODELO_LEVE_PROTOCOLO.md`, seção 3). Por que a variante leve V3 (int8) difere da gravada: o int8 dinâmico calcula a escala de cada camada a partir do **lote inteiro**, então a nota muda com o lote (lote de 1 aqui; lotes de até 8 com enchimento na medição do modelo leve). Isso reforça a ressalva da seção 3: **a V3 precisaria de um corte próprio e de um modo de uso fixo (sempre lote de 1)**.

## 10. Pergunta em aberto para o grupo

**O site roda o modelo no servidor ou no navegador do usuário?**
- **No servidor:** o caminho deste guia funciona como está (Python, PyTorch, V1 ou V2). O áudio do usuário sai do aparelho dele, o que pesa na privacidade.
- **No navegador:** exigiria **converter o modelo para outro formato** (por exemplo, ONNX) e refazer o pré-processamento em outra linguagem. **Nada disso foi testado.** Seria preciso repetir a verificação da seção 9 com a versão convertida.

## Números conferidos (de onde vem cada um)

- Chaves e formatos do `results/v7_t3_coeficientes.npz`: lidos com numpy (coef 3.072 float32; intercepto 0,30601; mu e sd 3.072 float32; C 0,01; camadas [5, 6, 7]).
- Cortes A' 0,0268, M 0,2326, F5 0,2107; regra marcada M: `results/v7_t3_modelo.json` e `results/v7_escolha.json` (que também tem a acurácia balanceada deixando uma fonte de fora: 0,9325 / 0,9331 / 0,9318).
- TPR e FPR do corte M por tipo de áudio: `results/v7_teste_mestre.csv` (linhas da T3).
- Tamanhos, parâmetros, tempos e diferenças de nota das variantes: `results/modelo_leve_variantes.csv`, `modelo_leve_latencia.csv`, `modelo_leve_equivalencia.csv`, `modelo_leve_resumo.json`.
- Composição do treino (nenhum clipe em português): `data/processed/splits_v7.parquet` (contagem por bloco e idioma no lado de treino) e `docs/PROTOCOLO_V7.md`, seção 2.
- Resultado do `--verificar`: execução de `scripts/inferencia_referencia.py` em 15/11/2026 (seção 9).
