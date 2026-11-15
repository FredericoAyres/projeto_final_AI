#!/usr/bin/env python3
"""Monta e executa o notebook da v8 (sistema atual: v8 = R1, fine-tuning parcial; anterior: v7 = T3). Só LÊ do disco.

- Reaproveita o notebook da v7 inteiro (`scripts/nb_v7.py`, que por sua vez reaproveita o da v4), troca a introdução
  e acrescenta as seções da v8 (17 a 22). O `main` do nb_v7 NÃO é chamado (ele copiaria por cima de
  notebooks/versions/v7_final.ipynb, que fica preservado).
- As figuras vêm de `scripts/v8_figuras.py` (results/v8_fig_*.png); as tabelas, de results/v8_*.csv/.json. Os números
  do texto são LIDOS desses arquivos na hora de montar.
- Grava notebook.ipynb e copia para notebooks/versions/v8_finetuning.ipynb.

    python scripts/nb_v8.py [--sem-executar]
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import nb_v7 as v7nb  # noqa: E402  (só as funções; o main dele não é chamado)

import nbformat as nbf  # noqa: E402
import pandas as pd  # noqa: E402

md, code, br, pct, mil = v7nb.md, v7nb.code, v7nb.br, v7nb.pct, v7nb.mil
R = Path("results")
LIMITE_MB = 10.0
ARQUIVO = "notebooks/versions/v8_finetuning.ipynb"


def d4(v):
    return ("+" if v >= 0 else "") + br(v, 4)


class N8:
    """Números da v8, lidos de results/."""

    def __init__(self):
        self.pr = pd.read_csv(R / "v8_principal.csv").set_index("candidato")
        self.tm = pd.read_csv(R / "v8_teste_mestre.csv")
        self.ct = pd.read_csv(R / "v8_controles.csv")
        self.val = pd.read_csv(R / "v8_validacao.csv")
        self.esc = json.loads((R / "v8_escolha.json").read_text(encoding="utf-8"))
        self.tre = {c: json.loads((R / f"v8_treino_{c}.json").read_text(encoding="utf-8"))
                    for c in ("R1_s42", "R1_s123", "R2_s42", "R3_s42")}
        self.porrc = self.esc["AUC_media_por_receita (R1 = média das seeds)"]
        assert self.esc["receita_vencedora_pela_validacao"] == "R1"
        assert (R / "v8_avaliar_concluido.json").exists()

    def p(self, c, col):
        return self.pr.loc[c, col]

    def g(self, c, rec, col="AUC", df=None):
        df = self.tm if df is None else df
        return df[(df.candidato == c) & (df.recorte == rec)].iloc[0][col]

    def dif(self, c):
        return f"{d4(self.p(c, 'dif. média vs R0 (T3)'))} (IC95 pareado {d4(self.p(c, 'dif. média vs R0 IC2.5'))} a {d4(self.p(c, 'dif. média vs R0 IC97.5'))})"


AJUDA_V8 = r'''
# ---- v8: figuras da v8 (results/v8_fig_*.png), embutidas reduzidas
def fig_v8(nome):
    p = R / f"v8_fig_{nome}.png"
    if existe(p):
        display(Image(data=_png_reduzido(p)))
    else:
        display(Markdown(f"> **Não encontrei** `{p}`. Rode `python scripts/v8_figuras.py`."))
'''


def bloco_fig(nome, titulo, mostra, olhar, concluir):
    return [md(f"### {titulo}\n\n**O que ela mostra:** {mostra}\n\n*O código abaixo só exibe a figura gerada por "
               f"`scripts/v8_figuras.py` (`results/v8_fig_{nome}.png`).*"),
            code(f'fig_v8("{nome}")'),
            md(f"**O que olhar:** {olhar}\n\n**O que concluir:** {concluir}")]


# ------------------------------------------------------------------------------------------------ introdução
def introducao(n, v7intro):
    m0, m1 = n.p("R0", "média AUC (3 fontes)"), n.p("R1 (média das seeds)", "média AUC (3 fontes)")
    return md(rf'''
# Detector de voz gerada por IA: documento da v8 (fine-tuning)

> **Sistema atual: v8 (R1).** **Sistema anterior: v7 (T3).** Os dois usam o mesmo caminho: o XLS-R (modelo de fala da Meta) transforma o áudio em números (média no tempo das camadas 5, 6 e 7) e uma camada linear dá a **nota**. Na v7, o XLS-R ficava **congelado** e só a regressão logística aprendia. Na v8, as **camadas 5, 6 e 7 do XLS-R também aprendem** (fine-tuning), começando exatamente da T3.

## 1. Introdução

**O que a v8 responde:** o retorno da apresentação pediu para **aplicar fine-tuning**. A v8 faz isso e mede se ajustar as camadas que a T3 usa melhora a detecção **sem esquecer** os bancos antigos.

**Resultados em 5 linhas** (todos lidos de `results/`; detalhes nas seções 17 a 22):
1. Quatro rodadas de fine-tuning (receitas R1 com 2 seeds, R2 e R3), comparadas com a T3 sem ajuste (R0), com o **mesmo treino, a mesma validação e o mesmo teste mestre da v7**.
2. A **R1** (camadas 5, 6 e 7 + cabeça, cabeça começando na T3) foi escolhida **só pela validação** (AUC média por fonte {br(n.porrc["R1"], 4)} × {br(n.porrc["R0"], 4)} da T3), gravada antes de qualquer teste.
3. No teste mestre, a AUC média (3 fontes) foi de **{br(m0)} (T3)** para **{br(m1)} (R1, média das 2 seeds)**: as duas seeds ganham da T3 com o IC pareado inteiro acima de zero (seed 42: {n.dif("R1_s42")}; seed 123: {n.dif("R1_s123")}). **Ganho pequeno, mas consistente.**
4. **Começar a cabeça pela T3 é o que faz o fine-tuning funcionar:** a R3 (cabeça aleatória) ficou **pior** que a T3 ({n.dif("R3_s42")}). E ajustar só a camada 7 (R2) ganha menos ({n.dif("R2_s42")}).
5. **Sem esquecimento na R1:** nos controles de estúdio ela manteve ou melhorou a T3. **Ressalvas:** o teste mestre não é virgem, na pior fonte (ML-ITW) o ganho não está garantido, há só 2 seeds na R1 e 1 nas outras, e o problema do corte entre ambientes (v7) continua.

**Como ler.** As seções 2 a 9 explicam o caminho e o sistema da v4 (F0); as seções 10 a 16 são a **v7 (T3)**, que é o ponto de partida da v8; as seções **17 a 22 são a v8**. Este notebook **só lê** arquivos já gravados: não treina e não consulta teste nenhum. Tudo o que não foi ensinado na trilha está marcado com "⚠️ Fora da trilha".

<details><summary>Introdução da versão anterior deste documento (v7)</summary>

{v7intro}

</details>
''')


# ------------------------------------------------------------------------------------------------ seções da v8
def secao17():
    return [md(r'''
## 17. v8: o que é o fine-tuning aqui (e por que cada escolha)

**A ideia em uma frase:** a T3 é "XLS-R congelado → média das camadas 5, 6 e 7 → padronização → regressão logística". A v8 escreve **esse mesmo caminho como uma rede só** (a regressão logística vira uma camada linear 3.072 → 1, com os mesmos pesos) e deixa as **camadas 5, 6 e 7 do XLS-R** voltarem a aprender, devagar. No passo zero, a rede dá a mesma nota da T3 (diferença máxima de 0,012 de logit em 48 clipes); a única fonte de mudança é o treino.

**Por que cada escolha faz sentido aqui** (todas ⚠️ Fora da trilha; a Aula 10 usa o modelo pré-treinado **congelado**):

| escolha | por quê, em linguagem simples |
|:--|:--|
| **Fine-tuning de só 3 das 24 camadas** (37,8 M parâmetros) | são as camadas que a T3 já usa; as de cima (8 a 24) nem entram na nota, então o XLS-R é cortado depois da 7. Treinar pouco cabe na placa de vídeo de 6 GB e reduz o risco de "esquecer" o pré-treino. |
| **Cabeça começando na T3 (LP-FT)** | com a cabeça aleatória, os primeiros passos mandam gradientes sem sentido para as camadas e estragam o que elas sabem. Começando de um modelo que já funciona, o ajuste é fino desde o primeiro passo. A **R3** mede esse efeito. |
| **AdamW** (weight decay 0,01) | o otimizador padrão para ajustar transformers: escolhe o tamanho do passo de cada peso sozinho, e o weight decay puxa os pesos de leve para não se afastarem demais. |
| **Taxa menor nas camadas (1e-5) que na cabeça (1e-4)** | as camadas já estão boas e só podem mudar pouco; a cabeça pode mudar mais. |
| **Aquecimento (10%) e queda linear até zero** | começar perto de zero evita um tranco que apagaria o pré-treino; terminar em zero deixa o fim estável. |
| **Corte da norma do gradiente em 1,0** | um lote estranho com gradiente enorme é encolhido, e não estraga o modelo sozinho. |
| **Precisão mista float16** | contas em 16 bits gastam metade da memória e são mais rápidas; os pesos que aprendem ficam em 32 bits. |
| **Sempre em modo `eval()`** | sem dropout e sem o mascaramento do pré-treino: no começo a rede é exatamente a T3. |

**O que fica igual à v7 (comparação justa):** o treino da T3 ({mil} clipes: estúdio + ML-ITW + RADAR zh/zh-tw/ja), a validação (6 fontes), o teste mestre (ML-ITW, In-the-Wild 38 locutores, RADAR en), os controles de estúdio, o pipeline de áudio da v4 e os critérios de escolha e de corte. Protocolo completo, carimbado antes do primeiro treino: `docs/PROTOCOLO_V8.md`. Código: `src/finetune.py` (a rede e o treino) e `scripts/v8.py` (as etapas).
'''.replace("{mil}", "81.288"))]


def secao18(n):
    lin = []
    for c, desc in (("R1_s42", "R1 · camadas 5–7 + cabeça · cabeça = T3 · seed 42"),
                    ("R1_s123", "R1 · camadas 5–7 + cabeça · cabeça = T3 · seed 123"),
                    ("R2_s42", "R2 · só camada 7 + cabeça · cabeça = T3 · seed 42"),
                    ("R3_s42", "R3 · camadas 5–7 + cabeça · cabeça aleatória · seed 42")):
        h = n.tre[c]["historico"]
        lin.append(f"| {desc} | {mil(h[0]['passos'])} | {br(h[0]['perda_media'])} → {br(h[1]['perda_media'])} | "
                   f"{br(sum(e['segundos'] for e in h) / 60, 0)} min | {br(h[0]['gpu_pico_gb'], 2)} GB |")   # chave do json, não aparece no texto
    linhas = "\n".join(lin)
    return [md(rf'''
## 18. v8: receitas e treino

| receita | o que treina | cabeça começa | seeds | para quê |
|:--|:--|:--|:--|:--|
| **R0** | nada (a T3 da v7) | — | — | referência |
| **R1** | camadas 5, 6 e 7 + cabeça | pesos da T3 | 42 e 123 | a receita principal |
| **R2** | só a camada 7 + cabeça | pesos da T3 | 42 | quanto descongelar |
| **R3** | camadas 5, 6 e 7 + cabeça | aleatória | 42 | o começo pela T3 ajuda? |

**Como cada rodada treinou:** 2 épocas; cada época sorteia 20.000 clipes do treino da T3 (10.000 reais e 10.000 falsos), em lotes de 16 clipes de duração parecida; perda de entropia cruzada binária. Os candidatos de cada rodada são o fim da época 1 e o fim da época 2, e vence o de maior AUC média por fonte na **validação**.

| rodada | passos por época | perda média (época 1 → 2) | tempo de treino (2 épocas) | memória de vídeo (pico) |
|:--|:--|:--|:--|:--|
{linhas}

Cada época de treino levou ~12 min (~28 clipes/s) numa placa de vídeo de notebook com 6 GB, e cada validação inteira, ~25 min. A execução (com uma interrupção registrada como adendo no protocolo) está em `docs/PROTOCOLO_V8.md`.
'''),
            *bloco_fig("5_perda_treino", "Figura v8-5 · Perda de treino por época",
                       "a perda média de treino de cada rodada nas duas épocas (rosa = R1, a receita escolhida; cinza = as outras).",
                       f"a R3, com a cabeça aleatória, começa com perda {br(n.tre['R3_s42']['historico'][0]['perda_media'])}, mais que o dobro das outras; as rodadas que começam na T3 já partem perto de 0,1.",
                       "a cabeça aleatória gasta a primeira época só para \"reaprender\" a T3, e nesse caminho mexe nas camadas à toa. Isso antecipa o resultado da R3 no teste (seção 20).")]


def secao19(n):
    v = n.val
    fontes = [c.replace("AUC ", "") for c in v.columns if c.startswith("AUC ") and c != "AUC média por fonte"]
    cab = "| candidato | época | " + " | ".join(fontes) + " | **média** |"
    sep = "|:--|:--|" + "--:|" * (len(fontes) + 1)
    lin = []
    for _, x in v.iterrows():
        ep = "—" if pd.isna(x["época"]) else str(int(x["época"])) + (" ✓" if x["época escolhida"] else "")
        lin.append(f"| {x.candidato} | {ep} | " + " | ".join(br(x[f"AUC {f}"], 4) for f in fontes) +
                   f" | **{br(x['AUC média por fonte'], 4)}** |")
    tab = "\n".join([cab, sep] + lin)
    rc = n.porrc
    return [md(rf'''
## 19. v8: a escolha (só validação)

A escolha foi gravada em `results/v8_escolha.json` às **{n.esc["quando"][11:19]} de {n.esc["quando"][:10]}**, antes de qualquer nota do teste. Critério do protocolo: maior média da AUC por fonte na validação (6 fontes, cada uma pesa igual); na R1, a nota da receita é a **média das duas seeds**.

{tab}

(✓ = época escolhida pela validação em cada rodada.)

**Nota de cada receita:** R0 (T3) {br(rc["R0"], 4)} · **R1 {br(rc["R1"], 4)}** · R2 {br(rc["R2"], 4)} · R3 {br(rc["R3"], 4)}. **Vencedora: R1.** A diferença para a T3 na validação é pequena (+{br(rc["R1"] - rc["R0"], 4)}), o que já dizia que o ganho, se existisse, seria pequeno.

**Onde a R1 ganhou e onde perdeu na validação:** ganhou nas fontes de "mundo real" que estão no treino (ML-ITW e RADAR) e perdeu um pouco no CodecFake+.
'''),
            *bloco_fig("3_validacao_x_teste", "Figura v8-3 · Validação × teste",
                       "cada ponto é uma rodada (na época escolhida): no eixo x, a nota na validação; no eixo y, a média da AUC no teste mestre.",
                       "a ordem das rodadas na validação se repete no teste: R3 < T3 < R2 < R1.",
                       "a validação escolheu bem: a receita que ela apontou (R1) é a melhor no teste, sem que o teste tenha sido usado para escolher nada.")]


def secao20(n):
    lin = []
    for c, nome in (("R0", "R0 = T3 (v7)"), ("R1_s42", "**R1 seed 42**"), ("R1_s123", "**R1 seed 123**"),
                    ("R1 (média das seeds)", "R1 (média das 2 seeds)"), ("R2_s42", "R2 (só camada 7)"),
                    ("R3_s42", "R3 (cabeça aleatória)")):
        ic = f"[{br(n.p(c, 'média IC2.5'))} a {br(n.p(c, 'média IC97.5'))}]"
        dif = "—" if c == "R0" else f"{d4(n.p(c, 'dif. média vs R0 (T3)'))} [{d4(n.p(c, 'dif. média vs R0 IC2.5'))} a {d4(n.p(c, 'dif. média vs R0 IC97.5'))}]"
        lin.append(f"| {nome} | {br(n.p(c, 'média AUC (3 fontes)'))} {ic} | {dif} | {br(n.p(c, 'AUC ml_itw'))} | "
                   f"{br(n.p(c, 'AUC in_the_wild'))} | {br(n.p(c, 'AUC radar_en'))} |")
    tab = "\n".join(lin)
    pior = [n.p(c, "dif. pior vs R0 IC2.5") for c in ("R1_s42", "R1_s123")]
    return [md(rf'''
## 20. v8: resultado no teste mestre (avaliado uma vez)

Cada rodada foi avaliada **uma vez**, na época escolhida pela validação; **todas** são reportadas. IC95 por bootstrap (2.000 repetições, seed 42; por locutor no ML-ITW e no In-the-Wild, por clipe no RADAR). A diferença para a T3 usa **os mesmos sorteios** do bootstrap (IC pareado), o que é bem mais preciso que comparar dois intervalos soltos.

| candidato | média AUC (3 fontes) [IC95] | diferença para a T3 [IC95 pareado] | ML-ITW | In-the-Wild | RADAR en |
|:--|:--|:--|:--|:--|:--|
{tab}

**O que dá para afirmar:**
- **A R1 é melhor que a T3 no teste mestre**, nas duas seeds, com o IC pareado inteiro acima de zero. O ganho é pequeno (~0,004 a 0,005 de AUC), mas não é ruído.
- **Na pior fonte (ML-ITW), o ganho não está garantido:** o IC pareado da pior fonte começa abaixo de zero (seed 42: {br(pior[0], 4)}; seed 123: {br(pior[1], 4)}).
- **O ganho maior está no RADAR en** (de {br(n.p("R0", "AUC radar_en"))} para ~{br(n.p("R1 (média das seeds)", "AUC radar_en"))}), onde o IC é por clipe e, por isso, otimista.
- **A R3 piorou a T3** de forma demonstrada: sem começar pela T3, o fine-tuning atrapalha.
'''),
            *bloco_fig("1_diferenca_vs_t3", "Figura v8-1 · Diferença para a T3 (IC pareado)",
                       "para cada rodada, quanto a média da AUC do teste mestre mudou em relação à T3, com o intervalo de 95%. Linha tracejada = igual à T3.",
                       "as barras rosas (R1) ficam inteiras à direita do zero; a da R3, inteira à esquerda; a da R2 encosta no zero.",
                       "o fine-tuning ajuda **quando começa da T3 e ajusta as 3 camadas**; ajustar uma camada só ajuda pouco, e começar do zero atrapalha."),
            *bloco_fig("2_teste_por_fonte", "Figura v8-2 · Teste mestre por fonte",
                       "a AUC de cada rodada em cada fonte do teste mestre, com IC95. Linha tracejada = T3.",
                       "no ML-ITW e no In-the-Wild os intervalos são largos (poucos locutores) e se sobrepõem; no RADAR en, a R1 se separa da T3.",
                       "o ganho médio vem principalmente do RADAR en; nas fontes de internet real ele é pequeno demais para ser afirmado fonte a fonte."),
            md(r'''
**O que faz:** mostra as tabelas completas da v8 no teste mestre (todas as rodadas, todos os recortes, com AUC, EER e as regras de corte). **Por quê:** os números por trás das figuras, sem escolher nada.
'''),
            code(r'''
pr8 = pd.read_csv(R / "v8_principal.csv")
display(pr8.round(4))
tm8 = pd.read_csv(R / "v8_teste_mestre.csv")
display(tm8[["candidato", "recorte", "época", "regra escolhida (validação)", "n_real", "n_falso", "AUC", "AUC IC2.5",
             "AUC IC97.5", "EER", "bal A'", "bal M", "bal F5"]].round(4))
''')]


def secao21(n):
    ct = n.ct
    lin = []
    for rec, nome in (("asvspoof2019_la", "ASVspoof 2019 LA"), ("asvspoof2021_df", "ASVspoof 2021 DF"),
                      ("llamapartialspoof", "LlamaPartialSpoof"), ("brspeech_df", "BRSpeech-DF (pt-BR)")):
        lin.append(f"| {nome} | " + " | ".join(br(n.g(c, rec, df=ct)) for c in ("R0", "R1_s42", "R1_s123", "R2_s42", "R3_s42")) + " |")
    co = ct[ct.recorte == "coraa"].set_index("candidato")
    fa = " | ".join(pct(co.loc[c, f"FPR {co.loc[c, 'regra escolhida (validação)']}"]) for c in ("R0", "R1_s42", "R1_s123", "R2_s42", "R3_s42"))
    tab = "\n".join(lin)
    return [md(rf'''
## 21. v8: controles de esquecimento (estúdio)

O fine-tuning mexe no XLS-R, então ele poderia "esquecer" o que já sabia. Os testes antigos de estúdio medem isso (AUC):

| controle | R0 = T3 | R1 s42 | R1 s123 | R2 | R3 |
|:--|:--|:--|:--|:--|:--|
{tab}
| CORAA (só reais, pt-BR): % de reais acusados, na regra escolhida | {fa} |

- **A R1 não esqueceu:** melhorou nos dois ASVspoof e no LlamaPartialSpoof; no BRSpeech-DF, uma seed ficou um pouco abaixo e a outra acima da T3.
- **A R3 caiu no português** (BRSpeech-DF), apesar de subir no LlamaPartialSpoof: começar do zero mexe mais no modelo.
- **O falso alarme em português (CORAA) depende da regra de corte de cada rodada**, e a escala da nota mudou com o fine-tuning (seção 22). Não compare essa linha como se fosse a AUC.
'''),
            *bloco_fig("4_controles", "Figura v8-4 · Controles de estúdio",
                       "a AUC de cada rodada nos quatro controles com as duas classes, com IC95. Linha tracejada = T3.",
                       "as bolinhas rosas (R1) ficam na linha da T3 ou à direita dela; a exceção é a seed 42 no BRSpeech-DF, dentro do intervalo.",
                       "ajustar as camadas, começando pela T3 e com taxa pequena, não custou o desempenho nos bancos antigos.")]


def secao22(n):
    esc = n.esc["por_candidato"]
    cortes = " · ".join(f"{c}: {esc[c]['regra']} = {br(esc[c]['cortes'][esc[c]['regra']], 2)}" for c in esc)
    return [md(rf'''
## 22. v8: o que não dá para afirmar e próximos passos

**Ressalvas da v8** (as da v7, na seção 14, continuam valendo):
- **O teste mestre não é virgem.** A T3 já tinha sido avaliada nele (v7), e quem desenhou a v8 conhecia esses números. Nada da v8 foi escolhido olhando o teste, mas só clones próprios, gravados depois, seriam um teste intocado.
- **Poucas seeds.** A R1 tem 2 seeds e as outras, 1. As duas seeds da R1 concordam, mas a variação entre seeds da R2 e da R3 não está medida.
- **Ganho pequeno.** +0,004 a +0,005 de AUC média. Real (IC pareado acima de zero), mas pequeno perto do salto da v7 (+0,20). Na pior fonte (ML-ITW), não está garantido.
- **A escala da nota mudou.** O corte de cada rodada, calculado na validação, mudou bastante ({cortes}). O problema da v7 (um corte único não serve para todos os ambientes) continua.
- **Um único treino curto.** 2 épocas de 20.000 clipes, com taxas fixadas antes; não houve busca de hiperparâmetros (de propósito: cada rodada custa ~1 h e a escolha seria só pela validação).

**Decisão em aberto (não tomada aqui):** qual modelo empacotar como sistema final. A **receita** escolhida é a R1, mas ela tem dois modelos (seed 42 e seed 123). As opções são usar uma seed (escolhida pela validação) ou a média das notas das duas. Isso deve ser decidido **só pela validação** e registrado antes de qualquer novo teste.

**Próximos passos** (somam-se aos da seção 16):
1. **Decidir o modelo final da R1** (seed ou média das duas), pela validação.
2. **Corte por ambiente**, agora sobre a nota da R1 (a escala mudou com o fine-tuning).
3. **Mais seeds** na R1 (e na R2), para medir o acaso do treino.
4. **Fine-tuning com português no treino**, para tirar a validação da cegueira para o idioma do uso final.
5. **Clones próprios como teste intocado**, avaliados uma vez depois de fechar o sistema.
''')]


# ------------------------------------------------------------------------------------------------ montagem
def montar():
    n = N8()
    C = v7nb.montar()
    v7intro = C[0].source.strip()
    v7intro = re.sub(r"^# .*\n", "", v7intro)              # tira o título da v7 (fica só o texto)
    v7intro = v7intro.replace("## 1. Introdução", "**Introdução da v7**")
    C[0] = introducao(n, v7intro)
    # configuração: acrescenta de onde se lê a v8
    ic = [i for i, c in enumerate(C) if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(ic) == 1
    C[ic[0]].source += ('\n\n# ---------------- v8 (fine-tuning parcial do XLS-R a partir da T3) ----------------\n'
                        '# Só leitura. Protocolo: docs/PROTOCOLO_V8.md; código: src/finetune.py e scripts/v8.py.\n'
                        'SISTEMA_ATUAL_V8 = "v8 (R1)"\n'
                        'ARQ_V8 = RESULTS_DIR / "v8_resumo.json"\n'
                        'print(f"documento da v8 | atual: {SISTEMA_ATUAL_V8} | anterior: {SISTEMA_ATUAL}")')
    # results/v4_camadas_concluido.json fica fora do git (só existe onde a v4 rodou): sem ele, a célula mostra
    # "pendente" em vez de quebrar. A saída original está em notebooks/versions/v7_final.ipynb.
    iv4 = v7nb.indice(C, 'mc = ler(R / "v4_camadas_concluido.json")', tipo="code")
    velho = '''mc = ler(R / "v4_camadas_concluido.json")
display(Markdown(f"Ordem das camadas pela validação: {mc['ordem_camadas_val'][:6]}…; melhor camada sozinha: "
                 f"{mc['melhor_camada']} (C = {mc['melhor_camada_C']:g})."))'''
    novo = '''mc = ler(R / "v4_camadas_concluido.json")
if mc:
    display(Markdown(f"Ordem das camadas pela validação: {mc['ordem_camadas_val'][:6]}…; melhor camada sozinha: "
                     f"{mc['melhor_camada']} (C = {mc['melhor_camada_C']:g})."))
else:
    pendente("`results/v4_camadas_concluido.json` (fica fora do git; a saída original está em notebooks/versions/v7_final.ipynb)")'''
    assert C[iv4].source.count(velho) == 1
    C[iv4].source = C[iv4].source.replace(velho, novo)
    ia = v7nb.indice(C, "def pendente(o_que):", tipo="code")
    C[ia] = code(C[ia].source + "\n" + AJUDA_V8.strip("\n"))
    # seção 9: a v8 na tabela do caminho
    v7nb.trocar_texto(C, "## 9. O caminho até aqui (v1 a v7)", "## 9. O caminho até aqui (v1 a v8)")
    m0, m1 = n.p("R0", "média AUC (3 fontes)"), n.p("R1 (média das seeds)", "média AUC (3 fontes)")
    v7nb.trocar_texto(C, "| **sistema atual = T3**, escolhida pela validação |",
                      "| T3, escolhida pela validação (sistema da v7) |\n"
                      f"| **v8** | **fine-tuning** das camadas 5, 6 e 7 do XLS-R, começando da T3 (seções 17 a 22) | média no teste mestre {br(m0)} → {br(m1)} (R1) | **sistema atual = R1**, escolhida pela validação |")
    C += secao17() + secao18(n) + secao19(n) + secao20(n) + secao21(n) + secao22(n)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-executar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for p in ("notebooks/versions/v4_final.ipynb", "notebooks/versions/v7_final.ipynb"):
        assert Path(p).exists(), f"{p} precisa existir (é preservado)"
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (env)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = montar()
    assert sum("config" in c.get("metadata", {}).get("tags", []) for c in nb.cells) == 1
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
    achados = []
    for i, c in enumerate(nb.cells):
        textos = [c.source] + [str(o.get("text", "")) + str(o.get("data", {}).get("text/plain", ""))
                               + str(o.get("data", {}).get("text/html", "")) + str(o.get("data", {}).get("text/markdown", ""))
                               for o in c.get("outputs", [])]
        for t in textos:
            for p in v7nb.PROIBIDOS:
                if re.search(rf"\b{re.escape(p)}\b", t):
                    achados.append((i, p))
    if achados:
        print("ATENÇÃO, nomes proibidos no notebook:", sorted(set(achados)))
    nbf.write(nb, "notebook.ipynb")
    mb = Path("notebook.ipynb").stat().st_size / 2 ** 20
    ok = not (erros or mb > LIMITE_MB or achados)
    if ok and not a.sem_executar:
        shutil.copyfile("notebook.ipynb", ARQUIVO)
    print(f"notebook.ipynb: {mb:.2f} MB, {len(nb.cells)} células, {erros} erro(s), {len(achados)} nome(s) proibido(s), "
          f"{time.time() - t0:.0f} s; cópia em {ARQUIVO}: {'sim' if ok and not a.sem_executar else 'não'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
