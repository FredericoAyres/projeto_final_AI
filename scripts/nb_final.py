#!/usr/bin/env python3
"""Monta e executa o notebook FINAL do sistema v4 (F0). O notebook só LÊ do disco: não treina nem recalcula nada.

- As figuras e tabelas pesadas vêm de `scripts/v4final_figuras.py` (results/v4final_*.png e *.csv).
- A célula com a tag "config" é a da v4 (copiada de notebooks/versions/v4_ssl.ipynb, sem mudança), mais um
  bloco curto do documento final. Assim, `scripts/v4.py` e `scripts/v4_teste.py` continuam lendo a configuração
  do sistema v4 a partir do notebook.ipynb da raiz.
- Executa com o nbclient (kernel do env), grava notebook.ipynb e copia para notebooks/versions/v4_final.ipynb.

    python scripts/nb_final.py [--sem-executar]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)

import nbformat as nbf  # noqa: E402

LIMITE_MB = 10.0


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s, tags=None):
    c = nbf.v4.new_code_cell(s.strip("\n"))
    if tags:
        c.metadata["tags"] = tags
    return c


def config_v4():
    nb = nbf.read("notebooks/versions/v4_ssl.ipynb", as_version=4)
    cel = [c for c in nb.cells if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(cel) == 1
    src = "".join(cel[0].source)
    assert 'VERSAO = "v4"' in src and "AVALIAR_TESTE_FINAL = False" in src
    return src + '''

# ---------------- documento final do sistema v4 (F0) ----------------
# Nada abaixo muda o método: só diz de onde o notebook lê. Camadas, C e limiares são LIDOS de results/
# (gravados pela v4 e conferidos por scripts/v4final_figuras.py), não digitados aqui.
DOCUMENTO = "v4 final (F0)"
F0_CABECA = "concat_top3"                       # a cabeça final escolhida pela validação na v4
ARQ_F0 = RESULTS_DIR / "v4final_resumo.json"    # valores confirmados do F0 (scripts/v4final_figuras.py)
print(f"{DOCUMENTO} | AVALIAR_TESTE_FINAL = {AVALIAR_TESTE_FINAL} | lê de {RESULTS_DIR}/")
'''


AJUDA = r'''
import json
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)
R = RESULTS_DIR


def existe(p):
    return Path(p).exists()


def ler(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pendente(o_que):
    display(Markdown(f"> **Não encontrei** {o_que} em disco. Rode `python scripts/v4final_figuras.py` "
                     "(precisa dos caches da v4, que ficam fora do git)."))


def figura(nome, largura=None):
    p = R / f"v4final_{nome}.png"
    if existe(p):
        display(Image(filename=str(p), width=largura))
    else:
        pendente(f"a figura `{p}`")


def tabela(nome, arred=3):
    p = R / nome
    if not existe(p):
        pendente(f"a tabela `{p}`")
        return None
    t = pd.read_csv(p)
    display(t.round(arred))


F0 = ler(ARQ_F0)
if F0:
    f = F0["F0"]
    display(Markdown(
        f"**F0 confirmado (lido de `{ARQ_F0}`):** camadas **{f['camadas']}** do XLS-R, média no tempo, "
        f"{f['n_pesos']} números por áudio; regressão logística com **C = {f['C']:g}**, solver {f['solver']}, "
        f"peso das classes: {f['class_weight']}; treinada em {f['n_treino']} clipes. Limiar **A** = "
        f"{f['limiar_A_logit']:.4f} (probabilidade {f['limiar_A_prob']:.3f}); limiar **B** = "
        f"{f['limiar_B_logit']:.4f} (probabilidade {f['limiar_B_prob']:.4f})."))
else:
    pendente(f"o arquivo `{ARQ_F0}`")
'''


def montar():
    C = []
    # ======================================================== 1
    C.append(md(r'''
# Detector de voz gerada por IA: o sistema v4 (F0), documento final

> **A frase do projeto:** um modelo de fala pronto e congelado (o XLS-R, da Meta) transforma cada áudio em números; uma **regressão logística** treinada por nós decide se a voz é real ou gerada por IA; e a avaliação foi feita com regras anti-vazamento e com todas as versões reportadas.

## 1. Introdução

**Objetivo.** Dizer se uma voz foi gerada por IA. O treino é em inglês (estúdio e audiolivro); o teste inclui vozes de internet (In-the-Wild), fakes parciais (LlamaPartialSpoof) e português (BRSpeech-DF e CORAA).

**Resultados em 5 linhas** (todos lidos de `results/`; detalhes nas seções 7 e 8):
1. A CNN das versões 1 a 3 ordenava as vozes de internet **ao contrário** (AUC do In-the-Wild 0,11 a 0,35). Com o XLS-R congelado + regressão logística (v4), a AUC do In-the-Wild foi para **0,979** e a do dev_shift para **0,983**.
2. No teste (avaliado uma vez, na v4), a média da acurácia nos 3 bancos principais foi **0,851** (limiar A) contra **0,597** do "chute da classe comum".
3. O modelo pronto fez a maior parte do trabalho: **só a última camada** (S1) já dava 0,843 de média; a diferença das nossas escolhas por cima **não foi demonstrada**.
4. O ponto fraco é o **português**: BRSpeech-DF com acurácia balanceada 0,679 (limiar A) e CORAA com 16,7% de reais acusados (limiar A).
5. Nada do que tentamos depois (WavLM da Microsoft na v5; fusão, MLP, CNN e ajuste fino na v6) superou o F0 no teste: o sistema declarado pelo funil da v6 (F2) ficou pior. A volta ao F0 foi uma escolha informada pelo teste (seção 10), embora o F0 em si tenha sido escolhido pela validação antes de qualquer teste; por isso o notebook voltou a ser o F0.

**Como ler.** Cada bloco de código tem antes um parágrafo dizendo o que faz e por quê. Depois de cada gráfico, duas frases: **o que olhar** e **o que concluir** (com os limites). Este notebook **só lê** arquivos já gravados: não treina, não extrai e não consulta o teste (`AVALIAR_TESTE_FINAL = False`); os números de teste são os gravados pela v4, que avaliou o teste uma única vez. As figuras foram geradas por `scripts/v4final_figuras.py` a partir dos caches da v4. Tudo o que não foi ensinado na trilha está marcado com "⚠️ Fora da trilha".
'''))
    # ======================================================== 2
    C.append(md(r'''
## 2. Configuração

**O que faz:** a célula abaixo (tag `config`) é **a configuração da v4, sem mudança**, mais um bloco curto que diz onde este documento lê os valores do F0. **Por quê:** o F0 é a v4; os scripts da v4 (`scripts/v4.py`, `scripts/v4_teste.py`) executam esta célula para saber os hiperparâmetros, então ela tem de continuar sendo a da v4. `AVALIAR_TESTE_FINAL = False`: nada aqui consulta o teste.
'''))
    C.append(code(config_v4(), tags=["config"]))
    C.append(md(r'''
**Os hiperparâmetros que importam para o F0, em uma frase cada:**

| hiperparâmetro | valor | por quê |
|:--|:--|:--|
| `MODELO_SSL` | `facebook/wav2vec2-xls-r-300m` | modelo pronto de fala, multilíngue (128 línguas), usado congelado (transfer learning, Aula 10) |
| `DTYPE` | float16 | metade da memória; a v5 mediu erro de ~0,06% contra float32 |
| `TOP_DB`, `MIN_S`, `MAX_S` | 35 dB, 0,5 s, 8 s | corta o silêncio das bordas (se sobrar menos de 0,5 s, não corta) e usa no máximo 8 s do meio, sem completar |
| `TREINO_MAX_SHARDS` × `TAM_SHARD` | 20 × 2.000 | 40.000 clipes de treino (metade real, metade falso), com teto por fonte e por gerador |
| `VAL_MAX` | 20.000 | subamostra da validação (19.999 clipes); CodecFake+ e ShiftySpeech entram inteiros |
| `GRADE_C` | 0,001 / 0,01 / 0,1 / 1 | o "freio" da regressão (C pequeno = freio forte); escolhido pela validação |
| `RL_MAX_ITER` | 1.000 | iterações máximas do lbfgs (a cabeça final convergiu em 56) |
| `TOP_K_CONCAT` | 3 e 4 | juntar as 3 ou as 4 melhores camadas (a validação escolheu 3) |
| `AVALIAR_TESTE_FINAL` | False | o teste só é lido dos arquivos gravados pela v4 |
'''))
    C.append(md(r'''
**O que faz:** carrega pequenas funções para mostrar figuras e tabelas de `results/` e imprime os valores do F0 **confirmados** pelo script das figuras (que reajustou a cabeça e conferiu contra as notas gravadas da v4). **Por quê:** os valores vêm do disco, não do texto.
'''))
    C.append(code(AJUDA))
    # ======================================================== 3
    C.append(md(r'''
## 3. Mapa do banco (curto)

Resumo de `docs/dataset_overview.md` e `docs/audit_report.md` (detalhes lá).

| papel | bancos | o que entra |
|:--|:--|:--|
| treino e validação (inglês) | ASVspoof 2019 LA (train+dev, só treino), DFADD, ShiftySpeech, CodecFake+ (CoSG), LibriSpeech train-clean-100 (só real) | 260.782 áudios de treino e 51.189 de validação, **divididos por locutor** |
| decisão entre versões | **dev_shift**: 16 locutores do In-the-Wild (30%, seed 42) | 10.614 áudios (81% real); pode ser olhado à vontade |
| teste principal | In-the-Wild (`test_final`, os outros 38 locutores), LlamaPartialSpoof, BRSpeech-DF (português) | avaliado uma vez por versão |
| teste de falso alarme | CORAA (português, só real) | mede quantos reais são acusados |
| teste secundário | ASVspoof 2019 eval, ASVspoof 2021 DF | podem ter locutores em comum com o treino (VCTK, via DFADD) |

**O que foi excluído e por quê:** LibriSpeech dev/test-clean (5.323; mesma origem dos reais do LlamaPartialSpoof); ShiftySpeech dos 79 locutores do LlamaPartialSpoof (21.048; vazamento de locutor); áudios de cadastro do ASVspoof 2019 (838; fora do protocolo); CORAA em português europeu (326). Tudo é reamostrado para 16 kHz (o BRSpeech-DF tinha reais a 48 kHz e fakes a 24 kHz: a taxa seria uma pista falsa).

**O que faz:** conta, no manifest, os áudios por banco, divisão e rótulo. **Por quê:** é a única fonte da verdade sobre os dados; a tabela confirma a divisão acima.
'''))
    C.append(code(r'''
df = pd.read_parquet("data/processed/manifest.parquet", columns=["source_dataset", "split", "label", "eval_subset"])
df["split"] = df["split"].fillna("excluído")
df.loc[df.eval_subset == "dev_shift", "split"] = "dev_shift"
t = df.groupby(["source_dataset", "split", "label"]).size().unstack(["split", "label"], fill_value=0)
t.columns = [f"{s}/{'fake' if l else 'real'}" for s, l in t.columns]
display(t[sorted(t.columns)])
'''))
    C.append(md(r'''
**Conclusão:** o treino é todo em inglês e quase todo de estúdio ou audiolivro; o dev_shift é o único conjunto "de internet" que podemos usar para decidir. Os clipes realmente usados pelo F0 são uma amostra: 40.000 de treino, 19.999 de validação e o dev_shift inteiro.
'''))
    # ======================================================== 4
    C.append(md(r'''
## 4. O pipeline do sistema v4, passo a passo

1. **Áudio → pré-processamento** (`src/preproc.py`): mono, 16 kHz; corte do silêncio nas bordas (quadros 35 dB abaixo do mais forte; se sobrar menos de 0,5 s, não corta); recorte central de no máximo 8 s; **sem completar** (nem repetição nem zeros: cada clipe fica com o seu tamanho). Cada clipe é normalizado (média 0, variância 1). ⚠️ Fora da trilha: o corte de silêncio por energia.
2. **XLS-R 300M da Meta, congelado** (`src/ssl.py`): nenhum peso do modelo é treinado por nós. **O que usamos exatamente:** a classe `Wav2Vec2Model` do pacote `transformers` (na v4 ela era chamada direto; hoje o `AutoModel` escolhe a mesma classe, conferido abaixo). Ela é só o **codificador**: a parte convolucional que lê a onda + 24 camadas transformer. O arquivo de pesos foi salvo como `Wav2Vec2ForPreTraining` (o modelo do pré-treino), mas a parte usada no pré-treino para **adivinhar os trechos escondidos** (o quantizador e as projeções da tarefa contrastiva) **não é carregada nem usada**: nós só lemos os vetores internos. ⚠️ Fora da trilha: o modelo auto-supervisionado e o pacote `transformers`.
3. **Vetores por camada:** o modelo devolve 25 saídas (a entrada do transformer e a saída de cada uma das 24 camadas), cada uma com um vetor de 1.024 números a cada 20 ms.
4. **Média e desvio no tempo:** cada saída vira um vetor de 1.024 números pela média (e outro pelo desvio) só nos quadros válidos. Os dois ficam no cache; **o F0 usa só a média** das saídas 5, 6 e 7 (3 × 1.024 = 3.072 números).
5. **Regressão logística** (scikit-learn): a única parte treinada com os nossos rótulos (seção 5).

**Por que o modelo pronto ajudou tanto (EXPLICAÇÃO PROVÁVEL, NÃO PROVADA):** os vetores do modelo pronto vêm de um treino enorme e variado, sem rótulos; os da nossa CNN vinham de um treino estreito e foram moldados só para separar real de falso naquela mistura. Não testamos isso diretamente (por exemplo, não treinamos uma regressão logística sobre os vetores internos da CNN); é a leitura mais provável, não um resultado.

**O que faz:** mostra qual classe o `AutoModel` escolhe para o XLS-R e o que diz o `config.json` em cache (lidos pelo script das figuras, sem carregar os pesos). **Por quê:** confirmar o que de fato é usado, em vez de supor.
'''))
    C.append(code(r'''
if F0:
    display(pd.Series(F0["modelo_hf"], name="valor").to_frame())
'''))
    C.append(md(r'''
**Conclusão:** a classe é `Wav2Vec2Model` (só o codificador), com 24 camadas de 1.024 números, como no código da v4.
'''))
    # ======================================================== 5
    C.append(md(r'''
## 5. A NOSSA PARTE: a regressão logística

**O que é, em linguagem simples.** Para cada áudio temos 3.072 números (x). A regressão multiplica cada número por um peso, soma tudo e soma um número fixo: **nota z = w·x + b**. Se a nota passa de um corte (o limiar), o áudio é chamado de **falso**. A nota vira probabilidade pela sigmoide, p = 1 / (1 + e^(−z)). Na figura, isso é uma **fronteira de decisão reta** (um plano, em 3.072 dimensões): de um lado "real", do outro "falso" (Aula de regressão logística e fronteira de decisão).

**O que ela aprende com os nossos rótulos:** só os 3.072 pesos w e o número b (3.073 parâmetros). O XLS-R não muda.

**O que entra:** a média no tempo das saídas **5, 6 e 7** do XLS-R, cada número **padronizado** com a média e o desvio do treino (assim nenhum número pesa mais só por ter escala maior), concatenadas.

**Hiperparâmetros:**
- **C = 0,1**: o inverso da força do "freio" de regularização L2 (Aula 7, overfitting e regularização). C pequeno = freio forte = pesos pequenos. ⚠️ Fora da trilha só o nome do parâmetro no scikit-learn; a ideia é a da aula.
- **Solver lbfgs** (o padrão do scikit-learn), até 1.000 iterações, tolerância 0,0001. ⚠️ Fora da trilha: o lbfgs (a trilha usa descida do gradiente; o problema é o mesmo, e é convexo, então o resultado não depende de seed).
- **Peso das classes: nenhum.** O treino já tem 20.000 reais e 20.000 falsos.

**Como a camada, o C e o limiar foram escolhidos (tudo na v4, antes do teste):**
1. Uma regressão por camada (25), cada uma com C na grade 0,001 / 0,01 / 0,1 / 1, escolhido pela **EER média por fonte na validação** (média da EER do DFADD, do ShiftySpeech e do CodecFake+; o LibriSpeech só tem reais).
2. As camadas foram ordenadas por essa EER: 7, 5, 6, 8, 9… As 3 melhores (5, 6, 7) foram concatenadas e o C foi escolhido de novo pela mesma régua: **C = 0,1** (tabela abaixo).
3. Entre as candidatas (melhor camada sozinha, 3 camadas, 4 camadas, somas ponderadas), venceu a de menor EER média por fonte: `concat_top3`, 5,92%.
4. **Limiar A** = o ponto de EER (erro igual nos reais e nos falsos) da **validação inteira**: logit −0,04 (probabilidade 0,49). **Limiar B** = o ponto de EER do **dev_shift**: logit 8,22 (probabilidade 0,9997). Os dois foram declarados antes do teste. O B é muito mais alto porque, nas vozes de internet, os reais recebem notas mais altas que os reais de estúdio (seção 7, G2).

**O que a nota significa:** quanto maior, mais "cara de falso". A probabilidade só é bem calibrada no mundo de estúdio (G8): nas vozes de internet, uma nota alta não quer dizer "certamente falso".

**O que faz:** mostra a grade de C da cabeça F0 e a lista de cabeças candidatas da v4. **Por quê:** é a escolha feita pela validação, com os números exatos.
'''))
    C.append(code(r'''
g = ler(R / "v4_parcial" / "concat_3.json")
if g:
    display(Markdown(f"**Grade de C da concatenação das camadas {g['camadas']}** (escolhido: C = {g['C']:g}):"))
    display(pd.DataFrame(g["grade"]).round(4))
else:
    pendente("`results/v4_parcial/concat_3.json` (fica fora do git)")
rs = ler(R / "v4_resumo.json")
display(Markdown(f"**Candidatas da v4 (EER média por fonte na validação):** vencedora `{rs['cabeça_final_pela_validação']}`"))
display(pd.DataFrame(rs["candidatas"]).round(4))
mc = ler(R / "v4_camadas_concluido.json")
display(Markdown(f"Ordem das camadas pela validação: {mc['ordem_camadas_val'][:6]}…; melhor camada sozinha: "
                 f"{mc['melhor_camada']} (C = {mc['melhor_camada_C']:g})."))
'''))
    C.append(md(r'''
**Conclusão:** C = 0,1 teve a menor EER média por fonte (5,92%) na grade; com C = 1 (freio fraco) a EER subiu para 6,82%, e com C = 0,001 (freio forte) para 7,02%. As diferenças entre as candidatas são pequenas (5,92% a 6,73%).
'''))
    # ======================================================== 6
    C.append(md(r'''
## 6. Por que a regressão logística foi a melhor opção

**(a) Nos nossos testes, as cabeças mais complexas não ganharam.**
- **v4** (regra: menor EER média por fonte na validação): regressão em 1 camada 5,99%, em 3 camadas 5,92%, em 4 camadas 5,98%, soma ponderada + linear 6,63% a 6,85% (3 seeds), soma ponderada + MLP 6,28% a 6,57%. A vencedora foi uma regressão logística. As diferenças entre as regressões (0,06 ponto) são muito menores que a régua de ruído que a v3 mediu na validação (1,5 ponto na EER média por fonte, entre seeds da CNN); **não fizemos teste formal de ruído na v4**, então "empate" aqui é leitura, não prova.
- **v6, E4** (regra do desafiante: uma cabeça mais complexa só entra com diferença relevante, IC95 sem o zero e ≥ 0,005 na AUC da pior fonte, nas 3 seeds): RL 0,904; MLP 0,902 a 0,912; CNN 0,906 a 0,910. **Nenhuma seed de MLP ou CNN teve diferença relevante** → fica a RL, a mais simples.
- Pelas duas regras (menor erro na v4; desafiante na v6), a cabeça que fica é uma regressão logística. Atenção: a RL da E4 era a camada 11 com C = 1 (não o F0); o F0 entra na tabela como referência.

**(b) Menos parâmetros, menos risco de overfitting (Aula 7) e treino em segundos:** RL 3.073 parâmetros (F0) contra 262.657 (MLP) e 405.825 (CNN) da E4; o F0 treina em ~2 s na CPU.

**(c) Da literatura (NÃO VERIFICADO POR NÓS; informado pela pesquisa de apoio do projeto, que não reproduzimos):** em Pascu et al. (Interspeech 2024), sobre o XLS-R 2B congelado, a regressão logística teve EER médio de 6,5%, contra 8,2% do MLP e 11,3% da autoatenção. Não lemos o artigo nesta sessão nem reproduzimos esses números.

**(d) Controle com rótulos embaralhados:** a mesma regressão (camada 7), treinada com os rótulos do treino embaralhados, deu AUC 0,474 na validação e 0,486 no dev_shift (o acaso é 0,5). Ou seja: a regressão não "inventa" separação; quando os rótulos não têm relação com o áudio, ela não acerta nada.

**(e)** Curva de aprendizado e caminho do C: gráficos G6 e G7 (seção 7).

**O que NÃO foi provado:** que a RL seja melhor que MLP ou CNN (só que não perdeu de forma demonstrável); que as 3 camadas sejam melhores que a última sozinha no teste (S2 − S1 "não demonstrada", seção 8); que o C = 0,1 seja o melhor fora do estúdio (G7 sugere o contrário, ver abaixo).

**O que faz:** mostra as tabelas das cabeças da v4 e da E4 da v6, lidas do disco. **Por quê:** os números da comparação, sem resumo.
'''))
    C.append(code(r'''
display(Markdown("**v4: cabeças sobre o XLS-R (escolha pela EER média por fonte na validação; dev_shift só reportado)**"))
tabela("v4final_cabecas_v4.csv", 4)
display(Markdown("**v6, E4: RL × MLP × CNN (AUC da pior fonte = menor AUC entre DFADD, ShiftySpeech, CodecFake+ e dev_shift)**"))
tabela("v4final_cabecas_v6_e4.csv", 4)
at = ler(R / "v4_atalhos.json")
display(Markdown("**Controle com rótulos embaralhados (v4):**"))
display(pd.Series(at["d_controle_rotulos_embaralhados"]).round(3).to_frame("valor"))
'''))
    C.append(md(r'''
**Conclusão:** em duas rodadas e com duas regras diferentes, a regressão logística ficou; MLP e CNN tiveram números parecidos com 100 vezes mais parâmetros, e o controle embaralhado deu acaso, como deveria.
'''))
    # ======================================================== 7
    C.append(md(r'''
## 7. Gráficos da regressão logística sobre os vetores

Todas as figuras vêm de `scripts/v4final_figuras.py`, que **reajustou o F0** (mesmos dados, camadas e C; lbfgs é determinístico) e conferiu contra as notas gravadas na v4 antes de desenhar. As regressões extras (G6, G7) são **só diagnóstico**: não mudaram camada, C nem limiar. Os bancos de teste aparecem só para **mostrar**: a cabeça e os limiares foram fixados antes do teste.

**O que faz:** mostra a conferência da reprodução (AUC com 3 casas e diferença máxima de nota, por conjunto). **Por quê:** se o reajuste não reproduzisse a v4, os gráficos seriam de outro modelo.
'''))
    C.append(code(r'''
tabela("v4final_reproducao.csv", 6)
'''))
    C.append(md(r'''
**Conclusão:** a AUC bate nas 3 casas em todos os bancos e a maior diferença de nota é de ~0,00001 de logit (as notas gravadas estão em float32): os gráficos abaixo são do F0 de verdade.
'''))

    figs = [
        ("g1_plano_decisao", "G1 · Plano de decisão exato (validação e dev_shift)",
         "Para cada áudio, o eixo x é a própria nota da regressão (w·x + b) e o eixo y é a direção em que os vetores mais variam **depois de tirar a direção de w** (PCA do que sobra, ajustado em 20.000 clipes do treino). Como y não tem nenhuma componente de w, a fronteira de decisão é **exatamente** a reta vertical nota = limiar (tracejada: A; pontilhada: B). Até 6.000 pontos por painel. ⚠️ Fora da trilha: a PCA.",
         "**O que olhar:** de que lado da reta tracejada cada cor cai; no DFADD e no LibriSpeech os reais ficam todos à esquerda, mas no dev_shift há uma \"cauda\" de reais (azul) passando da reta A, e o CodecFake+ (352 clipes) tem falsos dos dois lados. **O que concluir:** a regressão separa muito bem o mundo do treino, e o erro que sobra nas vozes de internet é de reais com nota alta, o que explica por que o limiar B (mais à direita) reduz o falso alarme ali; o eixo y explica só 11% da variação fora de w, então é uma fatia do espaço, não o espaço inteiro."),
        ("g1_plano_decisao_teste", "G1-teste · O mesmo plano nos bancos de teste (SÓ VISUALIZAÇÃO, NÃO DECIDE NADA)",
         "O mesmo eixo x (nota) e o mesmo eixo y (calculado no treino) para In-the-Wild, LlamaPartialSpoof, BRSpeech-DF e CORAA, a partir do cache de teste. Nada aqui foi usado para escolher qualquer coisa.",
         "**O que olhar:** no BRSpeech-DF uma boa parte dos reais (azul) fica à direita da reta A, e no CORAA (só reais) também há reais à direita; no LlamaPartialSpoof, falsos (laranja) à esquerda. **O que concluir:** os erros do teste são do mesmo tipo do dev_shift (reais com nota alta, sobretudo em português) mais falsos parciais que passam; a figura não diz por quê, e não deve ser usada para mexer no limiar."),
        ("g2_histogramas", "G2 · Distribuição da nota, real × falso, por banco",
         "Histograma (densidade por classe) da nota do F0 na validação, no dev_shift e nos bancos de teste, com os limiares A e B.",
         "**O que olhar:** a distância entre as duas montanhas e onde ficam os limiares; na validação elas quase não se tocam, no In-the-Wild e no dev_shift os reais se deslocam para a direita, e no BRSpeech-DF os reais têm duas montanhas (uma delas no meio dos falsos). **O que concluir:** o limiar A (estúdio) fica cedo demais para as vozes de internet e para o português, o que produz falso alarme; o limiar B corrige parte disso mas deixa passar mais falsos (LlamaPartialSpoof); por que parte dos reais do BRSpeech-DF tem nota alta **não foi investigado** (o banco mistura pt-BR e pt-PT, é uma hipótese, não um achado)."),
        ("g3_projecao", "G3 · Os vetores crus, sem a regressão (PCA e t-SNE)",
         "3.000 clipes da validação + 3.000 do dev_shift, na melhor camada sozinha (7) e na última (24), **sem padronizar e sem a regressão**. Cada projeção aparece duas vezes: cor por real × falso e cor por origem (validação × dev_shift). Diferente do G1, aqui ninguém escolheu a direção a favor dos rótulos. ⚠️ Fora da trilha: PCA e t-SNE.",
         "**O que olhar:** na camada 7, as cores real/falso já formam grupos separados, mas a maior separação visível é **validação × dev_shift** (o tipo de gravação); na camada 24 tudo se mistura mais. **O que concluir:** os vetores do modelo pronto carregam muita informação de \"de onde veio a gravação\", e a regressão do G1 é o que acha a direção real × falso no meio disso; o t-SNE distorce distâncias, então só os agrupamentos (não as posições) têm leitura."),
        ("g4_camadas", "G4 · AUC e EER por camada",
         "Uma regressão logística por camada (C de cada uma pela validação), lida de `results/v4_camadas.csv`. Faixas verdes: as camadas do F0; tracejada: a 24 (S1).",
         "**O que olhar:** as camadas 0 a 3 quase não servem no dev_shift (AUC 0,54 a 0,79), e da 5 em diante tudo passa de 0,97; as camadas 5 a 7 têm a menor EER na validação e no dev_shift. **O que concluir:** as pistas úteis estão nas camadas do meio do XLS-R, e a escolha pela validação caiu na região certa também para o dev_shift; mas a última camada (24) já é quase tão boa no dev_shift, o que bate com o S1 no teste."),
        ("g5_pesos_camadas", "G5 · Quanto cada camada pesa dentro do F0",
         "Soma dos valores absolutos dos 1.024 pesos de cada camada na regressão final (as entradas são padronizadas, então os pesos são comparáveis), e a distribuição dos pesos dentro de cada camada.",
         "**O que olhar:** as três barras são parecidas (36%, 31% e 33%) e as distribuições dos pesos se sobrepõem. **O que concluir:** a regressão usa as três camadas quase por igual, sem uma dominar; como as camadas vizinhas são muito parecidas (correlacionadas), o peso de cada uma não diz o quanto ela é \"necessária\" (tirar uma provavelmente mudaria pouco, mas isso não foi testado aqui)."),
        ("g6_curva_aprendizado", "G6 · Curva de aprendizado (diagnóstico)",
         "O F0 (mesmas camadas, C = 0,1) treinado em 2 mil, 5 mil, 10 mil, 20 mil e 40 mil clipes (metade real, metade falso, sorteio com seed 42, um sorteio por tamanho). Acurácia balanceada no limiar A de cada modelo e AUC, no próprio treino, na validação e no dev_shift.",
         "**O que olhar:** o treino fica em 1,00 em todos os tamanhos; a validação sobe devagar (AUC 0,996 → 0,998); o dev_shift **não sobe**: a AUC cai de 0,993 (2 mil) para 0,983 (40 mil). **O que concluir:** mais clipes do mesmo mundo (estúdio) ajudam a validação, mas não as vozes de internet; a regressão acerta 100% do próprio treino, que é o sinal de que ela consegue decorar o treino (overfitting, Aula 7); é um sorteio por tamanho, sem repetição, então a ordem exata dos pontos pode ser acaso."),
        ("g7_caminho_c", "G7 · Caminho do C (diagnóstico)",
         "A mesma regressão com 11 valores de C (0,0001 a 10, escala log). O F0 usa C = 0,1, escolhido na grade da v4 (linhas tracejadas) pela EER média por fonte da validação.",
         "**O que olhar:** a validação tem o mínimo perto de C = 0,1, mas o dev_shift melhora sem parar quando o freio aumenta (C = 0,0001: EER 2,7% e AUC 0,994, contra 5,9% e 0,983 no F0), enquanto os pesos encolhem (painel da direita). **O que concluir:** um freio mais forte (mais regularização) parece generalizar melhor para fora do estúdio, o mesmo tema da v6 (C = 1 piorou o português); isto é diagnóstico, feito depois de olhar o dev_shift, com 16 locutores: **não muda o F0**, e só valeria como hipótese para uma versão nova, confirmada num banco nunca visto."),
        ("g8_calibracao", "G8 · Calibração (probabilidade prevista × frequência real)",
         "Os áudios agrupados por faixa de probabilidade prevista (0 a 0,1, 0,1 a 0,2…); para cada faixa, a fração que é falsa de verdade. Os números são os clipes de cada faixa. ⚠️ Fora da trilha: calibração e Brier.",
         "**O que olhar:** na validação quase todos os clipes estão nas pontas (0 a 0,1 e 0,9 a 1) e a curva fica acima da diagonal no meio (poucos clipes); no dev_shift, entre 0,1 e 0,9 quase nada é falso, e na faixa 0,9 a 1 só 61% é falso. **O que concluir:** a nota ordena bem, mas a probabilidade não deve ser lida como \"chance de ser IA\" fora do estúdio: no dev_shift, \"99% falso\" está errado 4 em cada 10 vezes; a proporção de falsos também muda entre os conjuntos (70% na validação, 19% no dev_shift), e a calibração depende dela."),
        ("g9_roc", "G9 · Curva ROC por banco de teste (só mostra)",
         "A curva ROC do F0 em cada banco de teste com duas classes, com os pontos de operação dos limiares A (●) e B (■).",
         "**O que olhar:** a curva do BRSpeech-DF fica bem abaixo das outras, e os pontos A e B ficam em lugares diferentes de cada curva. **O que concluir:** a ordenação é boa em inglês (AUC 0,925 a 0,999) e fraca em português (0,842); a troca entre falso alarme e falsos que passam depende do limiar, e nenhum dos dois é bom em todos os bancos."),
        ("g9_confusao", "G9 · Matrizes de confusão no teste, limiares A e B (só mostra)",
         "Contagens e porcentagem por linha (linha = verdade). CORAA só tem reais.",
         "**O que olhar:** no limiar A, 32% dos reais do In-the-Wild e 64% dos reais do BRSpeech-DF são acusados; no limiar B, 52% dos falsos do LlamaPartialSpoof passam. **O que concluir:** os dois limiares trocam um erro pelo outro; o pior caso é o BRSpeech-DF, ruim nos dois."),
        ("g10_acuracia_bancos", "G10 · Acurácia por banco: F0 × CNN antiga (S0) × chute da classe comum",
         "Lido de `results/v4_teste_final.csv` (teste da v4, avaliado uma vez). S0 é a CNN da v3 com menor EER média por fonte na validação. No CORAA (só reais), acurácia = 1 − falso alarme, e o \"chute\" (sempre real) acerta 100%.",
         "**O que olhar:** no In-the-Wild a CNN fica abaixo do chute (0,34 contra 0,50) e o F0 vai a 0,84 (A) / 0,93 (B); no BRSpeech-DF, com o limiar A, o F0 (0,86) mal passa do chute (0,79), porque 79% do banco é falso. **O que concluir:** o F0 é muito melhor que a CNN em todos os bancos; a acurácia sozinha engana em banco desbalanceado (BRSpeech-DF, CORAA), por isso a acurácia balanceada e a AUC vêm junto (seção 8)."),
        ("g11_erros", "G11 · Falso alarme e falso que passa, por fonte e por banco",
         "Os dois erros nos limiares A e B: % de reais acusados e % de falsos que passam como reais, nas fontes da validação, no dev_shift e nos bancos de teste (só mostra).",
         "**O que olhar:** no limiar A, o falso alarme é baixo no estúdio (0 a 12%) e alto no dev_shift (19%), In-the-Wild (32%), BRSpeech-DF (64%) e CORAA (17%); no limiar B ele cai, mas os falsos que passam sobem (CodecFake+ 63%, LlamaPartialSpoof 52%). **O que concluir:** o limiar escolhido no estúdio não serve para o mundo real, e o escolhido nas vozes de internet deixa passar fakes de outros tipos; um limiar por domínio é a ideia que ficou para depois (não testada)."),
        ("g12_duracao", "G12 · AUC por faixa de duração original",
         "AUC do F0 por duração original do arquivo (< 2 s, 2 a 4 s, ≥ 4 s) no dev_shift e nos bancos de teste, com o número de clipes.",
         "**O que olhar:** em todos os bancos os áudios curtos são os piores; no BRSpeech-DF, abaixo de 2 s a AUC é 0,533 (quase acaso, 500 clipes) e de 2 a 4 s é 0,702. **O que concluir:** áudio curto é um ponto fraco geral, e o português curto é onde o sistema praticamente não funciona; a duração se mistura com outras diferenças (locutor, gravador), então é associação, não causa."),
        ("g13_geradores", "G13 · Recall por gerador na validação",
         "% dos falsos pegos por gerador (barra: limiar A; traço: limiar B). Cor = fonte. No CodecFake+, cada gerador tem de 5 a 32 falsos.",
         "**O que olhar:** DFADD e ShiftySpeech ficam perto de 100% no limiar A; no CodecFake+ alguns geradores escapam muito (SIMPLESPEECH1 12%, GPST 53%, SIMPLESPEECH2 68%). **O que concluir:** os fakes de codec neural são o tipo mais difícil dentro da validação; com tão poucos clipes por gerador, os números são instáveis e servem de pista, não de ranking."),
        ("g14_meta_microsoft", "G14 · AUC/EER por camada, Meta × Microsoft",
         "Os dois modelos prontos, camada por camada: **v5, feita pelo colega David Chung** (EER na validação e AUC no dev_shift; `results/v5_comparacao_camadas.csv`) e v6 (AUC da pior fonte, nesta máquina, nos mesmos clipes; `results/v6_e1_camadas.csv`).",
         "**O que olhar:** no WavLM (Microsoft) as melhores camadas são as de baixo (3 e 4) e a qualidade cai depois da 8; no XLS-R (Meta) as camadas 5 a 9 ficam todas boas e as de cima pioram pouco. **O que concluir:** os dois modelos guardam as pistas em lugares diferentes e o XLS-R foi melhor no conjunto (sobretudo em português, no teste da v5); a explicação (o pré-treino do WavLM a \"limpar\" ruído descartaria os artefatos nas camadas de cima) é hipótese, não testada."),
    ]
    for nome, titulo, antes, depois in figs:
        C.append(md(f"### {titulo}\n\n**O que faz:** {antes}"))
        C.append(code(f'figura("{nome}")'))
        C.append(md(depois))
    C.append(md(r'''
**O que faz:** mostra as tabelas por trás de G6, G7, G11, G12 e G13. **Por quê:** os números exatos dos gráficos.
'''))
    C.append(code(r'''
for nome in ("g6_curva_aprendizado", "g7_caminho_c", "g11_erros", "g12_duracao", "g13_geradores"):
    display(Markdown(f"**`results/v4final_{nome}.csv`**"))
    tabela(f"v4final_{nome}.csv", 4)
'''))
    # ======================================================== 8
    C.append(md(r'''
## 8. Resultados no teste (F0)

O teste foi avaliado **uma única vez**, na v4 (2026-11-10), pelo pré-registro, com até 5.000 clipes por classe e banco (seed 42); nada foi reescolhido depois. Os números abaixo são os gravados (`results/v4_teste_final.csv`) e o reajuste os reproduz (seção 7).

**O que faz:** tabela por banco com acurácia, acurácia balanceada, acurácia trivial, AUC, EER, recall de fake e falso alarme, nos limiares A e B, para o F0 (e, para comparar, S1 = só a camada 24 e S0 = a CNN da v3). **Por quê:** é o resultado final do sistema.
'''))
    C.append(code(r'''
t8 = pd.read_csv(R / "v4final_teste_por_banco.csv")
cols = ["banco", "limiar", "acurácia", "acurácia balanceada", "acurácia trivial", "AUC", "EER", "recall fake",
        "falso alarme real", "acurácia balanceada IC95", "AUC IC95", "n real", "n fake"]
for s in ("F0 (v4 final)", "S1 (só camada 24)", "S0 (CNN da v3)"):
    display(Markdown(f"**{s}**"))
    display(t8[t8.sistema == s][cols].round(3))
'''))
    C.append(md(r'''
**O que faz:** a média da acurácia dos 3 bancos principais (In-the-Wild, LlamaPartialSpoof, BRSpeech-DF), o falso alarme do CORAA à parte, e a leitura pré-definida das diferenças pareadas (S2 − S1 = "nossas escolhas"; S3 − S2 = "juntar a CNN"). **Por quê:** é o resumo combinado antes do teste.
'''))
    C.append(code(r'''
tr = ler(R / "v4_teste_final_resumo.json")
res = pd.DataFrame(tr["resumo"]).replace({"sistema": {"S2": "S2 = F0"}})
display(res.round(3))
display(Markdown("**Leitura pré-definida** (relevante só com o mesmo sinal nos 3 principais e IC95 sem o zero em cada um):"))
display(pd.Series(tr["regra_leitura"]).to_frame("resultado"))
dif = pd.read_csv(R / "v4_teste_diferencas.csv")
display(dif.round(4))
'''))
    C.append(md(r'''
**Conclusão do teste:**
- **F0:** AUC 0,979 (In-the-Wild), 0,925 (LlamaPartialSpoof), 0,842 (BRSpeech-DF); média da acurácia nos 3 principais **0,851** (limiar A) e 0,836 (B), contra **0,597** do chute da classe comum. CORAA: 16,7% (A) e 2,5% (B) dos reais acusados.
- **Secundários** (ASVspoof 2019 e 2021 DF, AUC 0,999 e 0,997): números altos, mas esses bancos podem ter locutores em comum com o treino (VCTK, via DFADD), então valem para geradores novos, não para locutores novos.
- **Nossas escolhas × só a última camada (S2 − S1): não demonstrada.** AUC −0,008 (In-the-Wild), −0,011 (LlamaPartialSpoof), +0,066 (BRSpeech-DF): sinais trocados. A única diferença grande a favor do F0 é no português (e no CORAA, 35,6% → 16,7% de falso alarme no A, só reportado).
- **CNN (S0) e juntar a CNN (S3):** a CNN ordena o In-the-Wild ao contrário (AUC 0,114); juntar a CNN ao F0 piorou o In-the-Wild em 0,091 de AUC.
- **Bootstrap:** os intervalos (±0,001 a ±0,016) medem **só a amostra de teste**; não medem o acaso do treino (uma única amostra de treino e uma regressão), que a v3 mostrou ser grande para a CNN.
'''))
    # ======================================================== 9
    C.append(md(r'''
## 9. O caminho até aqui (v1 a v6)

| versão | o que foi tentado | resultado | conclusão |
|:--|:--|:--|:--|
| v1 | CNN pequena em log-mel, sem augmentation | validação EER 1,3%; In-the-Wild AUC 0,211 | ótima em casa, invertida nas vozes de internet |
| v2 | regra do checkpoint (v1r), volume (v2a), ruído e canal (v2b); criação do dev_shift | dev_shift AUC 0,325 / 0,386 / 0,389 | robusta ao ruído que viu, sem ganho além do acaso; "limpo = real" não se confirmou |
| v3 | repetir × completar com zeros, 3 seeds | dev_shift AUC 0,146 a 0,378 nas seis execuções | sem efeito; a lição foi o **tamanho do acaso** |
| **v4** | **XLS-R congelado + regressão logística** (F0) | dev_shift 0,983; In-the-Wild 0,979; média de acurácia 0,851 | resolveu a inversão; fraco em português |
| v5 | WavLM Large (Microsoft) no lugar do XLS-R (**David Chung**) | empate no dev_shift; pior no teste em AUC nos 3 principais | segue o XLS-R |
| v6 | funil: Meta × Microsoft, fusão, RL × MLP × CNN, ajuste fino (pela AUC da pior fonte) | F2: média de acurácia 0,829 | pior que o F0 no teste; o notebook voltou ao F0 |

**A régua do ruído (v3):** com a mesma receita e só a seed mudando, a AUC do dev_shift da CNN variou de 0,146 a 0,378 (amplitude ~0,15 a 0,23); a acurácia balanceada, de 0,365 a 0,503. Diferenças menores que isso entre execuções únicas não são efeito. Na v4, as somas ponderadas com 3 seeds variaram só ~0,005 de AUC no dev_shift: com o modelo pronto, o acaso ficou muito menor.

**Meta × Microsoft:** na v5 (do colega David Chung), o WavLM ficou pior na validação (EER 6,77% × 5,92%) e no dev_shift (AUC 0,970 × 0,983, diferença não demonstrada pela regra), e pior no teste em AUC nos 3 principais (BRSpeech-DF 0,768 × 0,842). A v6 refez a comparação nesta máquina, nos mesmos clipes, e chegou ao mesmo lugar (G14).

**O funil da v6:** cada degrau comparou candidatos pela AUC da pior fonte (validação + dev_shift), com bootstrap por locutor, e só deixava entrar algo mais complexo com diferença relevante.

**O que faz:** mostra `results/v6_funil.csv` e os finalistas F0, F1 e F2 lado a lado no teste, com a acurácia trivial. **Por quê:** é a evidência de por que o notebook voltou ao F0.
'''))
    C.append(code(r'''
fun = pd.read_csv(R / "v6_funil.csv")
display(fun[["etapa", "vencedor", "por quê", "AUC pior fonte (cada candidato)"]])
f012 = pd.read_csv(R / "v4final_f0_f1_f2.csv")
display(Markdown("**F0, F1 e F2 no teste (limiar A; F1 = camada 11, C = 1; F2 = F1 + pesos iguais por fonte):**"))
p = f012[f012.limiar == "A"].pivot_table(index="banco", columns="sistema", values=["acurácia", "acurácia balanceada", "AUC", "falso alarme real"], sort=False)
p[("acurácia trivial", "")] = f012[(f012.limiar == "A") & (f012.sistema == "F0")].set_index("banco")["acurácia trivial"]
display(p.round(3))
'''))
    C.append(md(r'''
**A lição honesta da v6:** o sistema declarado pelo funil (F2) ficou **pior no teste que o F0** (média da acurácia nos 3 principais 0,829 × 0,851; BRSpeech-DF AUC 0,744 × 0,842; CORAA 40,4% × 16,7% de falso alarme no limiar A). A validação e o dev_shift (16 locutores) **não distinguiram** os dois: na AUC da pior fonte, F0 0,903 e F2 0,910. Causas prováveis, todas **HIPÓTESES**: (1) F1 e F2 usam **uma** camada (a 11), o F0 usa **três** (5, 6, 7); (2) F1 e F2 usam C = 1 (freio fraco), o F0 usa C = 0,1 (o G7 sugere que mais freio generaliza melhor); (3) a "pior fonte" foi sempre o CodecFake+ (352 clipes, inglês, codec), que não representa o português.
'''))
    # ======================================================== 10
    C.append(md(r'''
## 10. Limitações

- **Os bancos de teste foram vistos três vezes** (v4, v5 e v6). Os números do F0 são os da v4 original (o primeiro olhar), mas os bancos já não são "nunca vistos".
- **dev_shift pequeno:** 16 locutores; é ele que define o limiar B e que decidiu entre versões.
- **Uma única amostra de treino** (40.000 clipes) e uma regressão: o bootstrap do teste não mede o acaso do treino.
- **O português é o ponto fraco:** BRSpeech-DF com acurácia balanceada ~0,68 no limiar A (0,77 no B); áudios de menos de 2 s quase no acaso (G12); CORAA com 16,7% de reais acusados no limiar A.
- **Os dois limiares trocam um erro pelo outro:** A acusa reais de internet e de português; B deixa passar falsos parciais e de codec (G11).
- **A decisão de voltar ao F0 veio depois de ver o teste da v6.** O F0 em si foi escolhido pela validação **antes de qualquer teste** (pré-registro da v4), por isso os números do F0 são os da v4 original; mas preferir o F0 ao F2 agora é uma escolha informada pelo teste. **Recomendação:** antes de apresentar um número final como definitivo, avaliar o F0 **uma vez** num banco nunca visto (por exemplo, um banco do grupo que nunca tenha decidido nada), com o limiar declarado antes.
- **Explicações não provadas:** por que o modelo pronto funciona (seção 4), por que o WavLM perdeu e por que o F2 piorou são hipóteses.
'''))
    # ======================================================== 11
    C.append(md(r'''
## 11. Próximos passos (não executados)

1. **Melhorar a regressão logística** (um colega do grupo vai tentar): por exemplo, testar um freio mais forte (G7), sempre escolhendo pela validação + dev_shift e confirmando num banco novo.
2. **Dados em português e fala espontânea no treino** (reais pt-BR e fakes pt-BR de geradores fora dos testes), e um conjunto de decisão em português, para que as escolhas deixem de piorar o português sem ninguém ver.
3. **O contexto de golpes:** áudio curto, de telefone ou de aplicativo de mensagem, onde o sistema é mais fraco (G12); definir o limiar pelo custo de cada erro nesse uso.
4. **O modelo final unificado:** um único pipeline (áudio → XLS-R → regressão → nota e decisão) empacotado para o grupo, com o limiar e o domínio de uso escritos junto.
'''))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-executar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (env)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = montar()
    erros = 0
    if not a.sem_executar:
        from nbclient import NotebookClient
        cli = NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=True,
                             resources={"metadata": {"path": str(ROOT)}})
        cli.execute()
        for c in nb.cells:
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    erros += 1
                    print("ERRO:", o.get("ename"), o.get("evalue"), "\n", "".join(c.source)[:300])
    nbf.write(nb, "notebook.ipynb")
    mb = Path("notebook.ipynb").stat().st_size / 2 ** 20
    shutil.copyfile("notebook.ipynb", "notebooks/versions/v4_final.ipynb")
    print(f"notebook.ipynb: {mb:.2f} MB, {len(nb.cells)} células, {erros} erro(s), {time.time() - t0:.0f} s")
    return 1 if (erros or mb > LIMITE_MB) else 0


if __name__ == "__main__":
    sys.exit(main())
