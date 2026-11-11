#!/usr/bin/env python3
"""Monta e executa o notebook da v6 a partir dos resultados em disco (o notebook só LÊ; não treina nada).

- Preserva a célula com a tag "config" do notebook.ipynb atual (a fonte dos hiperparâmetros).
- Executa com o nbclient (kernel do env), grava notebook.ipynb e uma cópia em notebooks/versions/v6_funil.ipynb.
- Se results/v6_conclusao.md existir (texto escrito depois de ver os números), entra na seção 8; senão,
  a seção 8 mostra "Conclusão: pendente".
- QUICK (--quick): executa com V6_QUICK=1 e grava só em results/v6_notebook_quick*.ipynb (não mexe no
  notebook.ipynb nem em notebooks/versions).

    python scripts/v6_notebook.py [--quick] [--sem-executar] [--log arquivo]
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import nbformat as nbf  # noqa: E402

from src.execucao import Tee  # noqa: E402

LIMITE_MB = 10.0


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s, tags=None):
    c = nbf.v4.new_code_cell(s.strip("\n"))
    if tags:
        c.metadata["tags"] = tags
    return c


AJUDA = r'''
from pathlib import Path
import json
import warnings

import pandas as pd
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)


def existe(p):
    return Path(p).exists()


def ler(p, padrao=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao


def pendente(o_que):
    display(Markdown(f"> **Pendente:** {o_que} ainda não existe em disco (a etapa não terminou ou os arquivos "
                     "ficam fora do git). Rode `scripts\\vigia_v6.ps1` e execute o notebook de novo."))


def figura(p, largura=None):
    if existe(p):
        display(Image(filename=str(p), width=largura))
    else:
        pendente(f"a figura `{p}`")


def tabela(p, colunas=None, arred=4, filtro=None):
    if not existe(p):
        pendente(f"a tabela `{p}`")
        return None
    t = pd.read_csv(p)
    if filtro is not None:
        t = filtro(t)
    if colunas:
        t = t[[c for c in colunas if c in t.columns]]
    display(t.round(arred))
    return t


def f4(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


def pct(x):
    try:
        return f"{100 * float(x):.1f}%"
    except (TypeError, ValueError):
        return "—"


R = RESULTS_DIR
print("arquivos lidos de", R.resolve(), "| sufixo:", repr(SUFIXO))
'''


def montar(config_src):
    C = []
    C.append(md(r'''
# Detector de voz sintética: v6 (funil: front-end, fusão, cabeça, maximização)

> **O que é:** um detector que diz se uma voz foi gerada por IA. Treinado em inglês, testado também em português.
> **O que a v6 pergunta:** com o mesmo pipeline da v4 e da v5 (um modelo de fala pré-treinado e **congelado**, com uma cabeça leve por cima), (1) qual "ouvido" usar: o da **Meta** (XLS-R), o da **Microsoft** (WavLM) ou **os dois juntos**; (2) qual cabeça: **regressão logística, MLP ou CNN**; (3) quanto se ganha **apertando** a cabeça vencedora.
> **Como decidimos:** em seis etapas, como um funil; cada etapa só usa **validação + dev_shift**; os bancos de teste só aparecem na etapa 6, uma vez, para os finalistas declarados antes.

Este notebook **só lê** resultados já calculados (`scripts/vigia_v6.ps1` → `scripts/rodar_v6.ps1` → `scripts/v6.py` e `scripts/v6_teste.py`). Ele não treina nem consulta o teste (`AVALIAR_TESTE_FINAL = False`): os números da seção 7 vêm do arquivo gravado pela etapa 6.
'''))
    C.append(md(r'''
## Histórico das versões (resumo)

| versão | o que mudou | resultado em uma frase |
|:--|:--|:--|
| v1 a v3 | CNN pequena sobre log-mel (augmentation, padding) | ótima na validação e **invertida** nas vozes de internet (AUC do dev_shift 0,15 a 0,39) |
| v4 | XLS-R 300M (Meta) congelado + regressão logística | resolveu a inversão (dev_shift AUC 0,983; In-the-Wild 0,979); fraco em pt-BR |
| v5 | WavLM Large (Microsoft) no lugar do XLS-R | empate no dev_shift, pior no teste, sobretudo em pt-BR; o projeto seguiu com o XLS-R |
| **v6** | **funil: os dois modelos, a fusão, três cabeças e o ajuste fino**, com a métrica nova "AUC da pior fonte" | ver a tabela do funil (seção 6) e as conclusões (seção 8) |

Registro completo: `docs/HISTORICO_VERSOES.md`; decisões da v6: `docs/DECISOES_V6.md`.
'''))
    C.append(md(r'''
## 1. Introdução

### 1.1 O ponto de partida
Na v4, um modelo de fala pré-treinado (o **XLS-R**, da Meta) foi usado **congelado**, só como "extrator de características": cada áudio vira, em cada uma das 25 saídas do modelo, um vetor de 1.024 números (a média no tempo). Em cima disso, uma **regressão logística** decide real × fake. Isso é **transfer learning** (Aula 10): aproveitar o que um modelo aprendeu com muitos dados (aqui, ~436 mil horas de fala em 128 línguas) num problema novo com poucos dados rotulados. ⚠️ Fora da trilha: o modelo auto-supervisionado em si (a trilha faz transfer learning com um modelo de imagem).

Na v5, o colega trocou o XLS-R pelo **WavLM** (Microsoft, só inglês, pré-treinado também a "limpar" fala com ruído). Deu empate na decisão e pior no teste. Mas os dois modelos erraram em lugares diferentes, o que levanta a pergunta: **e se usarmos os dois?**

### 1.2 O funil (seis etapas, nesta ordem)
1. **E1 · Meta × Microsoft:** o mesmo pipeline nos dois modelos, nesta máquina, com notas por clipe (para poder comparar e juntar).
2. **E2 · Fusão:** três jeitos de juntar os dois (média das notas, stacking, vetores colados).
3. **E3 · Front-end:** só Meta, só Microsoft ou a fusão.
4. **E4 · Cabeça:** regressão logística, MLP ou CNN sobre o vencedor.
5. **E5 · Maximização:** uma modificação de cada vez na cabeça vencedora, e a regra do limiar.
6. **E6 · Teste:** os finalistas F0 (v4), F1 (front-end vencedor + cabeça básica) e F2 (sistema final), **uma vez**.

### 1.3 A métrica nova: AUC da **pior** fonte
Até a v5, escolhíamos pela **EER média por fonte**. A média esconde uma fonte ruim atrás de fontes fáceis (no DFADD a EER é quase 0%). Na v6, a métrica primária é a **AUC da pior fonte** entre as quatro "fontes de decisão": DFADD, ShiftySpeech e CodecFake+ (da validação) e o **dev_shift** (vozes de internet, 16 locutores do In-the-Wild). A AUC (área sob a curva ROC) mede se o modelo **ordena** bem real e fake, sem depender de limiar nem da proporção de cada classe: 0,5 é chute, 1,0 é perfeito.

### 1.4 As regras (pré-registradas antes de qualquer resultado)
- **Diferença relevante:** no bootstrap pareado (2.000 reamostragens, sorteando **locutores**), o intervalo de 95% da diferença de AUC da pior fonte **não inclui o zero**, e a diferença é de pelo menos **0,005**. ⚠️ Fora da trilha: o bootstrap.
- **Empate → vence o mais simples** (só Meta < só Microsoft < fusão; RL < MLP < CNN). É a ideia de **regularização** da aula de overfitting aplicada à escolha do modelo: na dúvida, menos parâmetros.
- **O teste não escolhe nada.** Os seis bancos de teste já tinham sido vistos na v4 e na v5; a E6 só mostra os finalistas.
'''))
    C.append(md(r'''
## 2. Configuração

**O que faz:** define **todos os hiperparâmetros numa célula só** (tag `config`). Os scripts (`scripts/v6.py`, `scripts/v6_teste.py`) executam exatamente esta célula, então não há números copiados em dois lugares. `QUICK = True` roda tudo com poucos clipes (teste de ponta a ponta); `AVALIAR_TESTE_FINAL = False` garante que nada aqui consulta o teste.
'''))
    C.append(code(config_src, tags=["config"]))
    C.append(md(r'''
**Cada hiperparâmetro em uma frase:**

| hiperparâmetro | valor | por quê |
|:--|:--|:--|
| `MODELOS` | XLS-R 300M (Meta) e WavLM Large (Microsoft) | os dois modelos já testados (v4 e v5); nenhum terceiro, porque a regra da rodada só permitia baixar o WavLM |
| `REVISAO_WAVLM` | `c1423ed…` | a mesma versão dos pesos que o colega usou na v5 |
| `DTYPE` | float16 | metade da memória e ~2× mais rápido; a v5 mediu erro de 0,04 a 0,06% contra o float32 |
| `TOP_DB`, `MIN_S`, `MAX_S` | 35 dB, 0,5 s, 8 s | o pré-processamento da v4 (corte de silêncio nas bordas; até 8 s do meio do áudio) |
| `ORCAMENTO_EXTRACAO_H` | 3 h | teto de tempo da extração do WavLM (medido: ~35 clipes/s, ~35 min) |
| `FONTES_VAL` + dev_shift | DFADD, ShiftySpeech, CodecFake+, dev_shift | as fontes com real e fake onde dá para medir AUC sem tocar no teste |
| `N_BOOT`, `SEED_BOOT` | 2.000, 42 | intervalos estáveis; os mesmos sorteios para todos os sistemas (comparação pareada) |
| `DIF_MIN` | 0,005 | diferença de AUC abaixo disso não muda nada na prática |
| `GRADE_C` | 0,001 a 1 | a grade da v4 (assim dá para conferir a reprodução); C é o inverso da força da regularização L2 |
| `TOP_K` | 3 e 4 | as concatenações de camadas que a v4 e a v5 já usaram |
| `V4_REF` | XLS-R, camadas 5, 6, 7, C = 0,1 | a cabeça final da v4 (F0, a referência) |
| `STACK_C` | 1 | o C do stacking da v4 (S3); com 2 entradas quase não há o que regularizar |
| `C_BASICO` | 1 | "básica" = sem ajuste; o padrão do scikit-learn |
| `NN` | 256 neurônios, dropout 0,3, AdamW lr 0,001, wd 0,001, lote 256, até 30 épocas, paciência 5 | a MLP da v4; nada novo para ajustar; o early stopping evita overfitting |
| `CNN_CANAIS`, `CNN_K` | 64, 3 | uma CNN pequena (2 convoluções) para não ter muito mais parâmetros que o MLP |
| `SEEDS` | 42, 123, 2026 | a v3 mostrou que a seed sozinha muda muito uma rede; a RL não precisa (é determinística) |
| `E5_GANHO_MIN`, `E5_QUEDA_DEV_MAX` | +0,003 e −0,01 | ajuste fino só fica se ajudar a pior fonte sem piorar as vozes de internet |
| `GRADE_C_AMPLA` | 8 valores de 0,0001 a 10 | a modificação (2) da E5: procurar a regularização certa numa faixa maior |
| `E5_FPR_D` | 5% | a regra de limiar D aceita 5% de falso alarme em cada fonte |
| `TESTE_*` | 5.000 por classe e banco, 1.000 bootstraps | os mesmos 51.320 clipes e a mesma regra da v4 e da v5 |
'''))
    C.append(md("**O que faz:** funções pequenas para ler tabelas, figuras e arquivos JSON do disco (nada é recalculado)."))
    C.append(code(AJUDA))
    C.append(md(r'''
## 3. Mapa do banco (resumo)

O banco **não mudou** desde a v2 (mesmo manifest, SHA-256 `f2309eb3…`). O mapa completo está em `docs/dataset_overview.md` e no notebook da v1. **O que é novo na v6:** o WavLM foi extraído **nos mesmos clipes** do cache da v4 (o plano foi copiado), e o código confere, clipe a clipe, que os dois modelos têm o mesmo `idx_manifest` e o mesmo caminho, na mesma ordem.
'''))
    C.append(code(r'''
e1 = ler(R / f"v6_e1_resumo{SUFIXO}.json")
if e1:
    display(Markdown("**Clipes conferidos nos dois modelos** (mesmos índices e caminhos, na mesma ordem):"))
    display(pd.DataFrame(e1["identidade_clipes"]).T)
    display(Markdown("**Fontes de decisão e unidade do bootstrap** (locutor quando há `speaker_id`; senão, clipe):"))
    display(pd.DataFrame(e1["bootstrap"]).T)
    nv = R / f"v6_notas_e1_val{SUFIXO}.parquet"
    if existe(nv):
        v = pd.read_parquet(nv)
        d = pd.read_parquet(R / f"v6_notas_e1_dev_shift{SUFIXO}.parquet")
        t = pd.concat([v.assign(conjunto="validação"), d.assign(conjunto="dev_shift")])
        display(t.groupby(["conjunto", "source_dataset", "label"]).size().unstack("label").rename(columns={0: "real", 1: "fake"}))
else:
    pendente("o resumo da E1")
'''))
    C.append(md(r'''
**Conclusão:** treino de 40.000 clipes (metade real, metade fake), validação de 19.999 e dev_shift inteiro (10.614), os mesmos para os dois modelos e para todas as etapas. O CodecFake+ da validação é pequeno (352 clipes, sem locutor), por isso o bootstrap nele sorteia clipes.
'''))
    C.append(md(r'''
## 4. Visualização: como os vetores se organizam

**O que faz:** projeta em 2D os vetores de 1.024 números da **melhor camada única** (E4) e, para contraste, da **última camada** (24), numa amostra de até 6.000 clipes da validação e do dev_shift (o mesmo número por fonte, quando dá). À esquerda de cada par, a cor é real × fake; à direita, a fonte. **Por quê:** ver se real e fake ficam separados e se o modelo agrupa os áudios mais por **banco de origem** (gravação) do que por real/fake, um sinal de atalho. ⚠️ Fora da trilha: PCA (a projeção linear que guarda mais variância) e t-SNE (uma projeção não linear que preserva vizinhos; as distâncias grandes no t-SNE não querem dizer nada).
'''))
    C.append(code(r'''
figura(R / f"v6_projecao{SUFIXO}.png")
rel = ler(R / f"v6_relatorio{SUFIXO}.json")
if rel and isinstance(rel.get("projecao"), dict):
    display(pd.DataFrame({m: v["por_fonte"] for m, v in rel["projecao"].items()}))
'''))
    C.append(md(r'''
**Como ler:** se, na cor por fonte, cada fonte forma uma "ilha" e, dentro dela, real e fake se separam, o modelo carrega muita informação de gravação (já visto na v4: a camada 7 reconhece o banco de origem dos reais com acurácia balanceada 0,80) e a cabeça linear consegue separar real × fake dentro de cada ilha. A projeção da CNN da v3 (opcional) **não** foi feita nesta rodada.
'''))
    C.append(md(r'''
## 5. Análise exploratória

### 5.1 Onde ficam as pistas em cada modelo (AUC por camada)
**O que faz:** para cada uma das 25 saídas de cada modelo, uma regressão logística (C escolhido pela pior fonte) e a AUC em cada fonte de decisão. **Por quê:** os dois modelos têm 24 camadas, e a v5 mostrou que as pistas ficam em lugares diferentes (WavLM mais embaixo).
'''))
    C.append(code(r'''
figura(R / f"v6_e1_camadas{SUFIXO}.png")
figura(R / f"v6_camadas_meta_microsoft{SUFIXO}.png")
t = tabela(R / f"v6_e1_camadas{SUFIXO}.csv",
           ["modelo", "camada", "C (pior fonte)", "AUC pior fonte", "fonte pior", "AUC dfadd", "AUC shiftyspeech",
            "AUC codecfake_plus", "AUC dev_shift", "AUC média (G)", "val EER média por fonte (3), regra v4"])
'''))
    C.append(code(r'''
if e1:
    rp = e1["reproducao_v4"]
    if rp.get("feita"):
        display(Markdown(f"**Reprodução da v4 (XLS-R):** com o C escolhido pela regra da v4, a maior diferença de EER média "
                         f"por fonte por camada contra `results/v4_camadas.csv` foi de **{rp['dif_max_pontos']} ponto** "
                         f"(camada {rp['camada_dif_max']}; limite 0,5) → **{'passou' if rp['passou'] else 'NÃO passou'}**. "
                         f"A referência v4 (camadas 5, 6, 7; C = 0,1) recalculada difere das notas gravadas na v4 em "
                         f"{e1['v4_ref_dif_max_notas_gravadas']} (logit)."))
'''))
    C.append(md(r'''
### 5.2 Os dois modelos erram nos mesmos áudios?
**O que faz:** a correlação das notas (logits) dos dois modelos, por classe, e a tabela 2×2 de acertos (cada modelo no seu limiar A, o ponto de EER da validação). **Por quê:** juntar dois classificadores só ajuda se eles **não** errarem sempre juntos; isso foi medido **antes** de comparar as fusões.
'''))
    C.append(code(r'''
e2 = ler(R / f"v6_e2_resumo{SUFIXO}.json")
if e2:
    display(pd.DataFrame(e2["correlacao"]).round(3))
    figura(R / f"v6_erros_2x2{SUFIXO}.png")
    display(pd.DataFrame({k: v for k, v in e2["erros_2x2"].items() if k != "limiar A (logit)"}).T)
else:
    pendente("o resumo da E2")
'''))
    C.append(md(r'''
## 6. Treinamento: o funil

Nada aqui foi treinado pelo notebook: os scripts treinaram, uma etapa por processo, e gravaram as notas por clipe. Tudo é escolhido na **validação + dev_shift**.

### 6.1 E1 · Meta × Microsoft (melhor cabeça de cada um)
**O que faz:** para cada modelo, a melhor entre as 25 camadas sozinhas, `concat_top3` e `concat_top4` (as 3 ou 4 melhores camadas lado a lado), pela AUC da pior fonte; depois, o bootstrap pareado da diferença. **Regressão logística** (Aula 4): uma camada linear com sigmoide; o C controla a **regularização** L2 (C pequeno = pesos pequenos = menos overfitting).
'''))
    C.append(code(r'''
tabela(R / f"v6_e1_cabecas{SUFIXO}.csv",
       ["modelo", "cabeça", "camadas", "C", "AUC pior fonte", "fonte pior", "AUC média (G)", "acc bal média (G)",
        "EER média (G)", "dev AUC", "val EER média por fonte (3)", "é a melhor do modelo"])
if e1:
    c = e1["bootstrap_wavlm_menos_xlsr"]
    display(Markdown(f"**Bootstrap pareado (Microsoft − Meta), AUC da pior fonte:** {c['dif AUC pior (desafiante - atual)']:+.4f} "
                     f"(IC95 {c['IC95'][0]:+.4f} a {c['IC95'][1]:+.4f}; {c['reamostragens válidas']} reamostragens) → "
                     f"**{'diferença relevante' if c['relevante'] else 'sem diferença relevante a favor da Microsoft'}**."))
'''))
    C.append(md(r'''
### 6.2 E2 · Fusão dos dois modelos
- **(a) média das notas:** cada logit padronizado com a média e o desvio da validação; média simples. Zero parâmetros novos.
- **(c) stacking:** uma regressão logística de **duas entradas** (as duas notas) aprende quanto confiar em cada modelo. Para a nota da validação não ser "dentro da amostra", usa **cross-fitting por fonte**: a nota de cada fonte vem de uma regressão treinada nas outras fontes. ⚠️ Fora da trilha.
- **(b) vetores colados:** os vetores das melhores camadas dos dois modelos lado a lado, cada bloco com o mesmo peso, e uma regressão logística por cima (milhares de pesos).
- **Por que não máximo, votação ou atenção?** Os dois primeiros jogam fora a confiança de cada modelo; o terceiro tem parâmetros demais para a validação que temos.
'''))
    C.append(code(r'''
tabela(R / f"v6_e2_fusao{SUFIXO}.csv", ["sistema", "escolhido", "AUC pior fonte", "fonte pior", "AUC média (G)",
                                        "acc bal média (G)", "dev AUC", "AUC codecfake_plus", "AUC dev_shift"])
if e2:
    display(Markdown("**Pesos do stacking** (entradas padronizadas; modelo 1 = Meta, modelo 2 = Microsoft):"))
    display(pd.DataFrame(e2["stacking_coeficientes"]).round(3))
    display(Markdown(f"**Vencedor da fusão (regra média → stacking → colados): `{e2['vencedor_fusao']}`**"))
'''))
    C.append(md("### 6.3 E3 · Qual front-end?\n**Regra:** só Meta → só Microsoft → fusão; cada opção mais complexa só entra se superar a atual com diferença relevante."))
    C.append(code(r'''
e3 = ler(R / f"v6_e3_frontend{SUFIXO}.json")
if e3:
    display(pd.DataFrame({k: {m: v.get(m) for m in ("AUC pior fonte", "fonte pior", "AUC média (G)", "acc bal média (G)", "dev AUC")}
                          for k, v in e3["metricas"].items()}).T)
    display(Markdown(f"**Vencedor: `{e3['vencedor']}`.** {e3['razao']}"))
else:
    pendente("a decisão da E3")
'''))
    C.append(md(r'''
### 6.4 E4 · Três cabeças básicas, uma de cada vez
- **RL básica:** a melhor camada única (pela pior fonte, com C = 1), C = 1. É o "transfer learning puro": modelo congelado + classificador linear.
- **MLP básico** (Aula de redes neurais): a mesma entrada, uma camada escondida de 256 neurônios com ReLU e **dropout** 0,3 (desliga neurônios ao acaso no treino, uma regularização), AdamW, **early stopping** pela validação (para quando a validação para de melhorar: o remédio clássico contra overfitting). ⚠️ Fora da trilha: o AdamW.
- **CNN básica** (Aula de CNN): em vez de olhar uma camada só, olha **as 25 camadas como uma sequência** e passa um filtro de tamanho 3 ao longo delas (o filtro vê três camadas vizinhas de cada vez, como numa imagem ele veria três pixels vizinhos). ⚠️ Fora da trilha: a convolução anda pelas camadas do modelo, não pelo tempo (o tempo já foi resumido em média e desvio). Ela recebe média **e** desvio (2.048 números por camada), a RL e o MLP só a média.
- **Regra:** RL → MLP → CNN; uma rede só toma o lugar se ganhar com diferença relevante **em cada uma das 3 seeds**.
'''))
    C.append(code(r'''
tabela(R / f"v6_e4_cabecas{SUFIXO}.csv", ["sistema", "seed", "parâmetros", "segundos", "melhor época", "AUC pior fonte",
                                          "fonte pior", "AUC média (G)", "acc bal média (G)", "EER média (G)", "dev AUC",
                                          "AUC dfadd", "AUC shiftyspeech", "AUC codecfake_plus", "AUC dev_shift"])
e4 = ler(R / f"v6_e4_resumo{SUFIXO}.json")
if e4:
    for c in e4["comparacoes"]:
        display(Markdown(f"- **{c['desafiante']} × {c['atual']}:** " + "; ".join(
            f"{r['dif AUC pior (desafiante - atual)']:+.4f} (IC95 {r['IC95'][0]:+.4f} a {r['IC95'][1]:+.4f})"
            + (" relevante" if r["relevante"] else "") for r in c["por seed"])
            + f" → {'toma o lugar' if c['relevante em todas as seeds'] else 'não toma o lugar'}"))
    display(Markdown(f"**Cabeça vencedora: {e4['vencedor']}** (camada da RL básica: {e4['camada_basica']})."))
'''))
    C.append(md(r'''
### 6.5 E5 · Apertar a cabeça vencedora (uma modificação de cada vez)
Cada passo **só fica** se a AUC da pior fonte subir pelo menos 0,003 e a AUC do dev_shift não cair mais que 0,01. Atenção: esta régua não usa bootstrap; passos pequenos podem ser ruído da validação. Na RL: (1) zona de camadas (uma ou três vizinhas), (2) grade ampla de C (a força da regularização), (3) peso igual por fonte de treino, (4) regra do limiar (A = EER global da validação; C = mediana dos limiares de EER por fonte; D = 5% de falso alarme por fonte), escolhida deixando uma fonte de fora de cada vez.
'''))
    C.append(code(r'''
tabela(R / f"v6_e5_passos{SUFIXO}.csv", ["passo", "modificação", "AUC pior antes", "AUC pior depois", "ganho",
                                         "fonte pior depois", "dev AUC antes", "dev AUC depois", "mantida",
                                         "acc bal LOSO A", "acc bal LOSO C", "acc bal LOSO D",
                                         "dev acc bal A", "dev acc bal C", "dev acc bal D"])
e5 = ler(R / f"v6_e5_resumo{SUFIXO}.json")
if e5:
    tabela(R / f"v6_e5_limiar_loso{SUFIXO}.csv")
    c = e5["final_menos_base_bootstrap"]
    display(Markdown(f"**Sistema final:** `{json.dumps(e5['spec_final'], ensure_ascii=False)}`; limiar **{e5['regra_limiar']}** "
                     f"(logits: {json.dumps({k: round(v, 3) for k, v in e5['limiares'].items()})}). "
                     f"Final − básica (bootstrap, só para informar): {c['dif AUC pior (desafiante - atual)']:+.4f} "
                     f"(IC95 {c['IC95'][0]:+.4f} a {c['IC95'][1]:+.4f})."))
'''))
    C.append(md("### 6.6 A tabela do funil\nUma linha por etapa: candidatos, métrica, vencedor, por quê, a alternativa descartada e os números de cada candidato."))
    C.append(code(r'''
figura(R / f"v6_funil{SUFIXO}.png")
if existe(R / f"v6_funil{SUFIXO}.csv"):
    f = pd.read_csv(R / f"v6_funil{SUFIXO}.csv")
    display(Markdown(f.to_markdown(index=False) if hasattr(f, "to_markdown") else f.to_string()))
else:
    pendente("a tabela do funil")
'''))
    C.append(md(r'''
## 7. Avaliação

### 7.1 Finalistas e limiares (gravados antes da E6)
Os três finalistas e os seus limiares foram gravados em `results/v6_finalistas.json` **antes** de qualquer nota de teste. **Limiares:** A = ponto de EER da validação; B = ponto de EER do dev_shift; E5 = o escolhido na E5. Nenhum é ajustado no teste.
'''))
    C.append(code(r'''
fi = ler(R / f"v6_finalistas{SUFIXO}.json")
if fi:
    display(Markdown(f"Gravado em **{fi['gravado_em']}**."))
    display(pd.DataFrame({k: {"descrição": v["descrição"], "limiar principal": v["limiar_principal"],
                              **{f"limiar {L} (logit)": round(t, 3) for L, t in v["limiares_logit"].items()},
                              "AUC pior fonte (val+dev)": round(v["metricas_val_dev"]["AUC pior fonte"], 4),
                              "dev AUC": round(v["metricas_val_dev"].get("dev AUC", float("nan")), 4)}
                          for k, v in fi["finalistas"].items()}).T)
    figura(R / f"v6_limiares_por_fonte_val{SUFIXO}.png")
else:
    pendente("os finalistas")
'''))
    C.append(md(r'''
### 7.2 Teste (E6, avaliado uma vez)
**Atenção:** os seis bancos de teste já tinham sido consultados na v4 e na v5; este é o terceiro olhar e só serve para **mostrar** os finalistas. **Acurácia sozinha engana** em banco desbalanceado (no BRSpeech-DF, 79% é fake: chutar "fake" sempre já acerta 79%), por isso ela vem ao lado da **acurácia trivial** e da **acurácia balanceada** (média do acerto em reais e em fakes; 0,5 = chute). Os intervalos (bootstrap dentro de cada classe) medem só a amostra de teste, não a variação do treino.
'''))
    C.append(code(r'''
te = ler(R / f"v6_teste_resumo{SUFIXO}.json")
if te:
    t = pd.read_csv(R / f"v6_teste_final{SUFIXO}.csv")
    t = t[(t.conjunto == "teste") & t["limiar principal"]]
    pv = t.pivot_table(index=["banco", "tipo"], columns=["sistema", "métrica"], values="valor", sort=False)
    met = ["acurácia", "acurácia trivial", "acurácia balanceada", "AUC", "EER", "recall fake", "falso alarme real"]
    display(Markdown(f"**Cada finalista no seu limiar principal** ({te['limiar_principal']}):"))
    display(pv.reindex(columns=[(s, m) for s in sorted(te["finalistas"]) for m in met]).round(3))
    display(Markdown("**Média dos três bancos principais com duas classes** (In-the-Wild, LlamaPartialSpoof, BRSpeech-DF) e falso alarme do CORAA:"))
    display(pd.DataFrame(te["resumo_finalistas"]).T.round(4))
    figura(R / f"v6_teste_acuracia{SUFIXO}.png")
else:
    pendente("o resultado da E6")
'''))
    C.append(code(r'''
if te:
    figura(R / f"v6_teste_roc_confusao{SUFIXO}.png")
    figura(R / f"v6_teste_limiares{SUFIXO}.png")
    display(Markdown("**Todos os limiares** (A, B e E5), média dos três principais:"))
    display(pd.DataFrame(te["resumo_todos_limiares"]).T.round(4))
    display(Markdown("**Leitura pré-definida** (relevante só com o mesmo sinal nos 3 principais e o IC95 sem o zero em cada um):"))
    display(pd.DataFrame(te["regra_leitura"], index=["resultado"]).T)
    tabela(R / f"v6_teste_diferencas{SUFIXO}.csv")
    display(Markdown(f"**WavLM no teste:** {te['wavlm_no_teste']}."))
    if te.get("colega_v5_teste"):
        c = pd.DataFrame(te["colega_v5_teste"])
        c = c[c.limiar == "A"].pivot_table(index="banco", columns=["sistema", "métrica"], values="valor", sort=False)
        display(Markdown("Números do teste da v5 (**do colega, sem notas por clipe**; limiar A):"))
        display(c.round(3))
'''))
    C.append(md("## 8. Conclusões"))
    C.append(code(r'''
fun = R / f"v6_funil{SUFIXO}.csv"
if existe(fun):
    f = pd.read_csv(fun)
    display(Markdown("**Vencedor de cada etapa (gerado dos arquivos):**\n\n" + "\n".join(
        f"- **{r.etapa}:** {r.vencedor}. {r['por quê']}" for _, r in f.iterrows())))
concl = R / f"v6_conclusao{SUFIXO}.md"
if existe(concl):
    display(Markdown(concl.read_text(encoding="utf-8")))
else:
    display(Markdown("### Conclusão: pendente\n\nO texto da conclusão (o que funcionou, o que não funcionou e por quê) "
                     "é escrito depois de ver os números, em `results/v6_conclusao.md`. As tabelas e os gráficos acima "
                     "já estão completos."))
'''))
    return C


def config_atual():
    nb = nbf.read("notebook.ipynb", as_version=4)
    cel = [c for c in nb.cells if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(cel) == 1, "o notebook.ipynb precisa de exatamente 1 célula com a tag config"
    src = "".join(cel[0].source)
    assert 'VERSAO = "v6"' in src, "a célula config do notebook.ipynb não é a da v6"
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sem-executar", action="store_true")
    ap.add_argument("--log", default=None)
    a, _ = ap.parse_known_args()
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        arq = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, arq)
        sys.stderr = Tee(sys.__stderr__, arq)
    print(f"\n######## {datetime.now().isoformat(timespec='seconds')} | v6_notebook {' '.join(sys.argv[1:])}", flush=True)
    t0 = time.time()
    try:
        suf = ("_quick" + os.environ.get("V6_SUFIXO_EXTRA", "")) if a.quick else ""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3 (env)", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        nb.cells = montar(config_atual())
        erros = 0
        if not a.sem_executar:
            from nbclient import NotebookClient
            if a.quick:
                os.environ["V6_QUICK"] = "1"
            cli = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=True,
                                 resources={"metadata": {"path": str(ROOT)}})
            cli.execute()
            erros = sum(1 for c in nb.cells if c.cell_type == "code"
                        for o in c.get("outputs", []) if o.get("output_type") == "error")
        from src.arquivos import gravar_atomico, salvar_json
        destino = Path(f"results/v6_notebook{suf}.ipynb") if a.quick else Path("notebook.ipynb")
        gravar_atomico(destino, lambda t: nbf.write(nb, str(t)))
        mb = destino.stat().st_size / 2 ** 20
        if not a.quick:
            Path("notebooks/versions").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destino, "notebooks/versions/v6_funil.ipynb")
        print(f"notebook: {destino} ({mb:.2f} MB), {erros} célula(s) com erro, {time.time() - t0:.0f} s", flush=True)
        if erros or mb > LIMITE_MB:
            print(f"STATUS=FALHOU ({erros} erros; {mb:.2f} MB)", flush=True)
            return 1
        salvar_json({"etapa": "notebook", "quick": a.quick, "concluido_em": datetime.now().isoformat(timespec="seconds"),
                     "arquivo": str(destino), "mb": round(mb, 2), "erros": erros,
                     "minutos_neste_processo": round((time.time() - t0) / 60, 2), "pico_memoria_gb_processo_e_workers": None,
                     "conclusao_escrita": Path(f"results/v6_conclusao{suf}.md").exists()},
                    Path(f"results/v6_notebook_concluido{suf}.json"))
        print("STATUS=CONCLUIDO", flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        print("STATUS=FALHOU", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
