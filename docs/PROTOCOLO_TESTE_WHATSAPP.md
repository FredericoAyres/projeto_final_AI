# Protocolo do teste externo "WhatsApp" (detector de voz + módulo de contexto de golpe)

**Escrito em 2026-11-15, 21:17 (UTC−3), ANTES de qualquer nota do detector e de qualquer chamada ao Groq neste teste.** Branch `main`, base `ac0984a` (+ arquivos novos, não commitados). Até aqui só foram feitos: ler os documentos do projeto, extrair o zip, conferir contagens, montar o manifesto novo (só metadados: duração, taxa, canais) e instalar o PyAV (`pip install av`, 19.0.2) para a condição B.

**Nada é treinado, reajustado ou trocado.** Nenhum áudio deste banco escolhe corte, prompt ou modelo. O teste roda **uma vez** (`results/teste_whatsapp_concluido.json` impede repetir).

## 0. Carimbos (SHA256 calculados em 2026-11-15, 21:10:24, UTC−3)

| arquivo | SHA256 |
|:--|:--|
| `results/v7_t3_coeficientes.npz` | `e991737396c420d5438e7fdc4d258bfeaafcdb5b8fc5dc17e5ed575473bf3301` |
| `results/v4final_f0_coeficientes.npz` | `c4a61c005a87c16d5227480a5b27309e4560da18611d066d689ab1ef3a02798f` |
| `results/v4_parcial/padronizacao.npz` | `6ab24c1b9a7c6241e652fb0449ce352bd01d81903ace5701807c299a3cd8dc79` |
| `docs/avaliacao_contexto/prompt_r1.txt` (bytes no disco, CRLF) | `fb0dfe5d623ee46b3eac079ecec102edf6b245ce4120bb5273a47c5b6696a6d6` |
| `docs/avaliacao_contexto/prompt_r1.txt` (fim de linha LF = o do commit e o do congelamento) | `3f848dbf65e294475e86ee963ab09103f7b83853bec979f14fbad872bf6de503` |
| `src/contexto_golpe.py` (bytes no disco, CRLF) | `9902d9f810bdeff2dcc56b83f7fd94a39236cd0f956f3df802e47fa2ee04adb9` |
| `src/contexto_golpe.py` (LF = o do commit e o do congelamento) | `d0327e1b3c177aba0b25d307199ea6c4900c7ce3d17e09254382780e15d81dfb` |
| `results/contexto_golpe_congelado.json` | `39d0ae22d5a16d9fe0188fbe57066e5cf87156f14e1c94b94c937965e91d11af` |
| `data/processed/manifest_whatsapp.parquet` | `0fcac35d27fc0cadcbfeb8c99395b8a478dfc758a0d1e5cea9bf7c360f655b29` |
| `scripts/inferencia_referencia.py` | `8b755cc61e6d968735c978d30ae90918095dc7dace6d6b2f5428b4ef001ab23c` |

**Observação:** os SHA256 do prompt e do módulo no disco diferem dos de `contexto_golpe_congelado.json` **só pelo fim de linha** (Windows grava CRLF; o commit e o congelamento usam LF). Conferido: normalizando CRLF→LF, os dois batem exatamente com o congelado e com o `HEAD`. O script do teste confere o hash normalizado.

## 1. Dados

Zip `data/teste_whatsapp/coleta_para_equipe_organizada.zip`, extraído com `zipfile` (trocando `\` por `/`) em `data/teste_whatsapp/extraido/`. Conferido antes de qualquer nota: **44 reais** (27 com golpe, 17 sem golpe), **46 clones com pedido de dinheiro**, **10 clones neutros**, 100 linhas no manifesto, 23 `pessoa_id`, todos os caminhos existem, todos os 23 têm real e IA. Os OGG em `coleta/originais` são as mesmas 44 gravações dos WAV e **não são usados**. Os reais são 48 kHz mono (WhatsApp ogg/opus convertido para wav); as IA são 24 kHz mono (Chatterbox nano).

`data/processed/manifest_whatsapp.parquet`: `id, pessoa_id, caminho, autenticidade (real/ia), conteudo (golpe/normal), grupo, duracao_s, taxa_hz, canais`. **A coluna `nome_local` do manifesto da coleta não é copiada para lugar nenhum.**

- `conteudo = golpe`: real com golpe + clone com pedido de dinheiro (**27 + 46 = 73**). `normal`: real sem golpe + clone neutro (**17 + 10 = 27**).
- Grupos: `real_sem_golpe` (17), `real_com_golpe` (27), `ia_pedido_dinheiro` (46), `ia_neutro` (10).
- Detector: positivo = IA (56 IA, 44 reais; acurácia trivial 56%). Contexto: positivo = golpe (73; acurácia trivial 73%).
- A coleta diz que tudo é em inglês. **Isso não é aceito como fato:** o idioma é o que o Whisper detectar (seção 4).

## 2. Detector de voz

Mesmo pipeline da v7, **pelo caminho de referência** (`scripts/inferencia_referencia.py`, variante V1 = modelo cortado depois da camada 7, fp32; nenhuma conta reescrita): mono 16 kHz, corte de silêncio a 35 dB, recorte central de 8 s sem completar, normalização do clipe, XLS-R 300M congelado, média no tempo das saídas 5, 6 e 7, padronização, regressão.

- **T3.** Nota = `((x − mu)/sd)·coef + intercepto` (`results/v7_t3_coeficientes.npz`). Cortes congelados, de `results/v7_t3_modelo.json`: **M = 0,23256 (principal)**, A' = 0,02678 e F5 = 0,21069 (secundários). Decisão: IA se nota ≥ corte.
- **F0 (v4, só comparação).** Mesmos vetores x (camadas 5, 6, 7); padronização por camada de `results/v4_parcial/padronizacao.npz` (mu, sd [25×1024]); nota = `x·coef + intercepto` (`results/v4final_f0_coeficientes.npz`), como em `scripts/teste_internet.py: aplicar_f0`. Cortes lidos de `results/v4final_resumo.json` e conferidos com o npz: **A = −0,03954** (ponto de EER da validação) e **B = 8,21838** (ponto de EER do dev_shift). Não são ambíguos.
- Os vetores vêm do modelo em **fp32**; a T3/F0 original foi pontuada de vetores fp16. A diferença esperada é de centésimos de logit (`docs/GUIA_PARA_O_SITE.md`, seção 9).
- Nenhum corte, camada ou pesos são ajustados com estes áudios. Nada de inverter a nota.

### Métricas (por modelo e corte)
Acurácia, acurácia balanceada, **TPR** (IA detectada), **FPR** (real acusado), AUC. **IC95 por bootstrap por pessoa:** reamostrar as 23 `pessoa_id` com reposição, 2.000 repetições, `numpy.random.default_rng(42)`, tomando todos os clipes de cada pessoa sorteada; IC = percentis 2,5 e 97,5. Complemento: **IC de Wilson** nas proporções, **com o aviso de que ele ignora o agrupamento por pessoa** (os clipes da mesma pessoa não são independentes), logo é otimista. Taxas por grupo (os 4): fração com nota ≥ corte.

## 3. Condições

- **(A) como veio** (principal).
- **(B) controle de canal:** os **56 clones** são recodificados para **Opus mono ~16 kbps em `.ogg`** com o PyAV (libopus, 16 kHz, mono, 16.000 bit/s), parecido com um áudio de WhatsApp; os 44 reais **ficam como estão** (já vieram de opus). B mostra se o resultado depende do formato/canal. Se o PyAV falhar, B é marcada como **não feita** (nada é inventado). *(PyAV 19.0.2 instalou e importou.)*

## 4. Contexto (Groq)

`src/contexto_golpe.analisar_audio` nos **100 WAV**, config congelada (prompt r1, `openai/gpt-oss-120b`, temperature 0, seed 42, Whisper `whisper-large-v3-turbo`), **uma chamada por clipe, uma a cada 12 s**, sem repetir chamada (só as retentativas internas do módulo). Sem `GROQ_API_KEY` no ambiente, para-se e avisa-se; um 429 persistente para o teste e é relatado, **sem descartar clipes**.

- **Decisão sobre o idioma do Whisper (tomada agora, antes de ver qualquer saída):** a função tem `idioma="pt"` como valor padrão, que **forçaria português** em áudio que a coleta diz ser inglês, e mascararia o "idioma detectado". Será passado **`idioma="auto"`** (opção já prevista no módulo: o Whisper detecta). É um argumento da chamada, **não muda o módulo nem o prompt**. A avaliação congelada (`docs/AVALIACAO_CONTEXTO.md`) só testou o **texto** (pt-BR); o caminho de áudio nunca foi avaliado, então não há um "idioma congelado" do ASR a violar. O prompt r1 pede resumo em português, mas aceita texto em qualquer língua (as evidências são copiadas literalmente do texto); **o prompt não foi traduzido nem alterado**. Isso é um **desvio assumido** em relação a "tudo igual à avaliação" e vai para o relatório.
- **Alerta** = `risco_conteudo` em {`medio`, `alto`} (principal); só `alto` (secundário). `indeterminado`/`audio_sem_fala`/`erro` contam como **sem alerta** na principal e são **contados à parte**.
- Métricas: acurácia, recall (golpe com alerta), falso alarme (normal com alerta), precisão; ICs como na seção 2 (bootstrap por pessoa e Wilson). Por grupo; e **separando reais e IA** (recall em `real_com_golpe` × `ia_pedido_dinheiro`; falso alarme em `real_sem_golpe` × `ia_neutro`).
- Registrados por clipe: idioma detectado (e quantos **não** são `en`), nº de palavras, status, risco, tipos de sinais, `manipulacao_detectada`, `transcricao_suspeita`, latências. **As transcrições ficam só em `data/processed/tw_contexto.jsonl` (fora do git). Nada de texto vai para `results/`.**
- **Sobreposição de texto:** cada transcrição é comparada com todos os textos de `docs/avaliacao_contexto/textos_dev.jsonl` e `textos_teste.jsonl` e com `prompt_r1.txt`, por (i) razão do `difflib.SequenceMatcher` sobre o texto em minúsculas e (ii) Jaccard de palavras. **"Quase idêntica" = razão ≥ 0,8 ou Jaccard ≥ 0,6.** Reporta-se quantas e, se houver, a ressalva. Como os textos do projeto são pt-BR e a coleta diz ser inglês, espera-se zero, mas **mede-se**.

## 5. Atalhos (metadados)

Classificador **só com metadados** do áudio original (sem o modelo de voz): duração (s), nível RMS (dBFS), fração de silêncio (fração dos quadros de 25 ms, passo de 10 ms, com energia mais de 35 dB abaixo do quadro mais forte), taxa de amostragem original (Hz) e **largura de banda aproximada** (frequência abaixo da qual ficam 95% da energia espectral, Welch). Regressão logística padronizada (C = 1), **validação deixando uma pessoa de fora** (23 dobras); AUC das notas fora-da-dobra, com IC por bootstrap por pessoa. Duas versões: **com** e **sem** a taxa original (a taxa sozinha separa 48 kHz de 24 kHz). Compara-se a AUC com a do detector. Alvo: IA × real.

**Análise pareada por pessoa:** para cada uma das 23 pessoas, nota média dos clones menos nota média dos reais (T3 e F0); reporta-se a média das diferenças com IC por bootstrap por pessoa e **quantas das 23 têm IA > real**.

## 6. Resultado combinado (regra congelada agora)

Voz = **IA se nota T3 ≥ M** (0,23256). Conteúdo = **alerta se risco ≥ medio**. Nível: **alto** = os dois; **atenção** = um dos dois; **baixo** = nenhum. Tabela grupo × nível. Conteúdo `indeterminado` conta como sem alerta e é contado à parte. **Hipótese (não promessa):** `ia_pedido_dinheiro` → alto; `ia_neutro` e `real_com_golpe` → atenção; `real_sem_golpe` → baixo. **Isto não é uma "acurácia"**; só se reportam as taxas por célula.

## 7. Limites (escritos antes)

100 áudios, 23 pessoas (ICs largos; os clipes da mesma pessoa são correlacionados); **um único gerador** (Chatterbox nano); a coleta alega inglês, o que será checado; os textos dos reais e dos clones **são diferentes**, então o conteúdo pode denunciar a origem sem que o modelo de voz acerte nada; os reais com golpe são **gravações de pesquisa (encenação), não fraude real**; os clones com pedido de dinheiro **nem sempre são golpe**; **nada foi testado em ligação ao vivo**; os reais são 48 kHz (opus convertido) e os clones 24 kHz, então formato e largura de banda podem virar atalho (por isso a condição B e a seção 5); o teste **não é virgem para o contexto** se houver sobreposição de texto com os textos de desenvolvimento/teste (mede-se na seção 4); o corte M não se transfere entre ambientes (`docs/DECISAO_SISTEMA_PRINCIPAL.md`, seção 4).

## 8. Execução e saídas

`scripts/teste_whatsapp.py <etapa>` (`manifesto`, `recodificar`, `detector`, `contexto`, `resumo`, `marcar`), na ordem: detector (T3 e F0, A e B), depois contexto, depois combinação. Notas por clipe (com `pessoa_id`, sem nome) em `results/teste_whatsapp_notas.csv`. Figuras `results/slides/tw_01..tw_05.png` (`scripts/teste_whatsapp_figuras.py`) e relatório `docs/RESULTADO_TESTE_WHATSAPP.md`. Desvios do protocolo serão registrados como **adendo datado** no fim deste arquivo, com o motivo, e os números anteriores continuam reportados.

## Adendo (2026-11-15, depois da execução)

Sem mudança de método. Registros: (1) o protocolo supunha áudio em inglês; o Whisper detectou **português nos 100 clipes** (ver `docs/RESULTADO_TESTE_WHATSAPP.md`, seção 1). (2) Nas linhas "só reais" e "só IA" do contexto só há valores pontuais, sem IC. (3) O marcador `results/teste_whatsapp_concluido.json` foi gravado depois de `resumo`. (4) Os números do relatório são os de `results/teste_whatsapp_*.csv`, nenhum foi alterado depois de vistos.
