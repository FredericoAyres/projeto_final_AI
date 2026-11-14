#!/usr/bin/env python3
"""Monta e executa o notebook da v7 (sistema atual: v7 = T3; sistema anterior: v4 = F0). Só LÊ do disco.

- Reaproveita as seções da v4 de `scripts/nb_final.py` (pipeline, regressão logística, por que regressão, o caminho
  v1 a v6), trocando a introdução, o glossário, os rótulos de "sistema" e as duas últimas seções.
- Acrescenta as seções da v7 (10 a 16). As figuras vêm de `scripts/v7_slides.py` (results/slides/f_*.png) e de
  `scripts/teste_internet_figuras.py` (results/slides/ti_*.png); as tabelas, de results/v7_*.csv/.json e
  results/teste_internet_*.csv/.json. Os números do texto são LIDOS desses arquivos na hora de montar.
- Não roda o `nb_final.py` (ele copiaria por cima de notebooks/versions/v4_final.ipynb, que fica preservado).
- A célula "config" continua a da v4 (os scripts da v4 a executam), com um bloco final que diz de onde se lê a v7.
- As figuras são embutidas reduzidas (até 1.200 px de largura, ~90 dpi, paleta de 256 cores) para o arquivo ficar
  abaixo de 10 MB; as versões em alta estão em results/slides/.
- Grava notebook.ipynb e copia para notebooks/versions/v7_final.ipynb.

    python scripts/nb_v7.py [--sem-executar]
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
import nb_final as base  # noqa: E402  (só as funções; o main dele não é chamado)

import nbformat as nbf  # noqa: E402
import pandas as pd  # noqa: E402

md, code = base.md, base.code
R = Path("results")
LIMITE_MB = 10.0
PROIBIDOS = base.PROIBIDOS + ["Scarlett", "Johansson", "Barack", "Donald", "freay", "Frederico", "ScarlettJohansson"]


# ------------------------------------------------------------------------------------------------ números (lidos)
def br(v, d=3):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def cg(x):
    return f"{x:g}".replace(".", ",")


def pct(v, d=1):
    return br(100 * v, d) + "%"


def mil(n):
    return f"{int(n):,}".replace(",", ".")


class N:
    """Todos os números usados no texto, lidos de results/."""

    def __init__(self):
        self.tm = pd.read_csv(R / "v7_teste_mestre.csv")
        self.pr = pd.read_csv(R / "v7_principal.csv").set_index("receita")
        self.val = pd.read_csv(R / "v7_validacao.csv").set_index("receita")
        self.cv = pd.read_csv(R / "v7_cortes_validacao.csv")
        self.at = pd.read_csv(R / "v7_atalho.csv")
        self.ph = pd.read_csv(R / "v7_poshoc.csv")
        self.dur = pd.read_csv(R / "v7_radar_en_duracao.csv")
        self.cont = pd.read_csv(R / "v7_splits_contagens.csv")
        self.regras = pd.read_csv(R / "v7_splits_mlitw_regras.csv")
        self.sj = json.loads((R / "v7_splits.json").read_text(encoding="utf-8"))
        self.esc = json.loads((R / "v7_escolha.json").read_text(encoding="utf-8"))
        self.tre = json.loads((R / "v7_treino.json").read_text(encoding="utf-8"))
        self.t3 = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))
        self.ti = pd.read_csv(R / "teste_internet_por_fonte.csv").set_index("fonte")
        self.tia = pd.read_csv(R / "teste_internet_auditoria_atalho.csv")
        self.tis = pd.read_csv(R / "teste_internet_sensibilidade.csv")
        self.rep = pd.read_csv(R / "v4final_reproducao.csv").set_index("conjunto")
        assert self.esc["receita_vencedora_pela_validacao"] == "T3"

    def g(self, rc, rec, col="AUC"):
        return self.tm[(self.tm.receita == rc) & (self.tm.recorte == rec)].iloc[0][col]

    def ic(self, rc, rec):
        return f"{br(self.g(rc, rec, 'AUC IC2.5'))} a {br(self.g(rc, rec, 'AUC IC97.5'))}"

    def p(self, rc, col):
        return self.pr.loc[rc, col]


# ------------------------------------------------------------------------------------------------ helpers do notebook
AJUDA_V7 = r'''
# ---- v7: figuras reduzidas (para o arquivo ficar < 10 MB) e leitura dos resultados da v7
from io import BytesIO

from PIL import Image as _PIL

LARGURA_MAX = 1200          # ~90 dpi numa figura de 13,33 pol.; as versões em alta estão em results/slides/


def _png_reduzido(p):
    im = _PIL.open(p).convert("RGB")
    if im.width > LARGURA_MAX:
        im = im.resize((LARGURA_MAX, round(im.height * LARGURA_MAX / im.width)), _PIL.LANCZOS)
    im = im.quantize(colors=256)
    b = BytesIO()
    im.save(b, format="PNG", optimize=True)
    return b.getvalue()


def figura(nome):
    """Figuras da v4 (results/v4final_*.png), embutidas reduzidas."""
    p = R / f"v4final_{nome}.png"
    if existe(p):
        display(Image(data=_png_reduzido(p)))
    else:
        pendente(f"a figura `{p}`")


def slide(nome):
    """Figuras dos slides (results/slides/*.png), embutidas reduzidas."""
    p = SLIDES / f"{nome}.png"
    if existe(p):
        display(Image(data=_png_reduzido(p)))
    else:
        display(Markdown(f"> **Não encontrei** `{p}`. Rode `python scripts/v7_slides.py`."))


T3 = ler(R / "v7_t3_modelo.json")
if T3:
    rg = T3["regressao"]
    display(Markdown(
        f"**Sistema atual: v7 (T3)** (lido de `results/v7_t3_modelo.json`): o mesmo XLS-R congelado, camadas "
        f"**{T3['camadas']}**, {T3['n_numeros']} números por áudio; regressão logística com **C = {rg['C']:g}**, "
        f"solver {rg['solver']}, peso das classes **{rg['class_weight']}**; treinada em {T3['treino']['n']} áudios "
        f"(estúdio + internet real + IA em canal degradado). Pesos em `results/v7_t3_coeficientes.npz`; conferidos contra "
        f"as notas gravadas da v7 (diferença máxima {T3['conferencia']['dif_max_nota_logit']:.1e})."))
'''


def bloco_slide(nome, titulo, mostra, olhar, concluir, origem="scripts/v7_slides.py"):
    return [md(f"### {titulo}\n\n**O que ela mostra:** {mostra}\n\n*O código abaixo só exibe a figura gerada por "
               f"`{origem}` (slide em alta: `results/slides/{nome}.png`).*"),
            code(f'slide("{nome}")'),
            md(f"**O que olhar:** {olhar}\n\n**O que concluir:** {concluir}")]


def substituir(cells, inicio, novo):
    """Troca o texto da célula markdown que começa com `inicio`."""
    achou = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.lstrip().startswith(inicio)]
    assert len(achou) == 1, (inicio, achou)
    cells[achou[0]] = md(novo) if isinstance(novo, str) else novo
    return achou[0]


def indice(cells, trecho, tipo="markdown"):
    achou = [i for i, c in enumerate(cells) if c.cell_type == tipo and trecho in c.source]
    assert len(achou) == 1, (trecho, achou)
    return achou[0]


def trocar_texto(cells, velho, novo):
    n = 0
    for c in cells:
        if velho in c.source:
            c.source = c.source.replace(velho, novo)
            n += 1
    assert n >= 1, velho


# ------------------------------------------------------------------------------------------------ seções
def introducao(n):
    f0_val = n.rep.loc["val", "AUC reajuste"]
    itw = n.g("T0", "in_the_wild")
    m0, m3 = n.p("T0", "média AUC (3 fontes)"), n.p("T3", "média AUC (3 fontes)")
    d3 = n.p("T3", "dif. média vs T0")
    queda = min(n.g("T3", r) - n.g("T0", r) for r in ("llamapartialspoof", "asvspoof2019_la", "asvspoof2021_df", "brspeech_df"))
    li = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw_ingles")].iloc[0]
    lp = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw_portugues")].iloc[0]
    return md(rf'''
# Detector de voz gerada por IA: documento da v7

> **Sistema atual: v7 (T3).** **Sistema anterior: v4 (F0).** Os dois usam o mesmo caminho: um modelo de fala pronto e congelado (o XLS-R, da Meta) transforma cada áudio em números, e uma **regressão logística** treinada por nós dá uma **nota**; acima de um **corte**, a voz é chamada de gerada por IA. O que mudou na v7 foi **com que dados** a regressão é treinada.

## 1. Introdução

**Objetivo.** Dizer se uma voz foi gerada por IA. O treino começou só em inglês de estúdio e audiolivro; os testes incluem vozes de internet, fakes parciais e português.

**Resultados em 5 linhas** (todos lidos de `results/`; detalhes nas seções 10 a 16):
1. O sistema anterior, **v4 (F0)**, ia muito bem no estúdio (validação: AUC {br(f0_val)}) e no banco de internet mais fácil (entrevistas e discursos, In-the-Wild: {br(itw)}), mas **caiu nos dois bancos externos novos**: AUC **{br(n.ti.loc["radar_eval", "AUC"])}** em IA comercial moderna em canal degradado (RADAR) e **{br(n.ti.loc["ml_itw", "AUC"])}** em vídeos reais de redes sociais (ML-ITW).
2. Na v7, treinamos com **parte** desses dois bancos, separada **por locutor** (ML-ITW) e **por idioma** (RADAR, que não informa o locutor), e mantivemos um **teste mestre fixo**, igual para todas as receitas de treino.
3. Entre 5 receitas (T0 a T4), a **T3** (estúdio + internet real + IA moderna em canal degradado) foi escolhida pela **validação** (AUC média por fonte {br(n.val.loc["T3", "AUC média por fonte"])}), gravada antes de abrir o teste.
4. No teste mestre, a AUC média (3 fontes) foi de **{br(m0, 2)} (T0 = F0)** para **{br(m3, 2)} (T3)** ({br(m0)} → {br(m3)}; diferença +{br(d3)}, IC95 +{br(n.p("T3", "dif. média vs T0 IC2.5"))} a +{br(n.p("T3", "dif. média vs T0 IC97.5"))}), **sem piorar o estúdio** (maior queda de AUC nos controles: {br(queda)}).
5. **Ressalvas principais:** o teste mestre **não é virgem** (o F0 já tinha sido medido nesses clipes); no RADAR, o idioma do teste é novo, mas **geradores e canal são os do treino** (não é "gerador novo"); internet real em inglês ({int(li.locutores_real)} / {int(li.locutores_falso)} locutores) e em português ({int(lp.locutores_real)} / {int(lp.locutores_falso)}) é **indício**, não estimativa; **o corte não se transfere entre ambientes** (no português de audiolivro, {pct(n.g("T3", "brspeech_df", "FPR M"), 0)} dos reais são acusados); nada aqui fala de ligações reais.

**Como ler.** As seções 2 a 9 explicam o caminho e o **sistema anterior (v4 = F0)**, que continua valendo para tudo o que não mudou (pipeline, camadas, regressão). As seções 10 a 16 são a **v7**. Antes de cada figura, um parágrafo diz **o que ela mostra**; depois, **o que olhar** e **o que concluir**. Este notebook **só lê** arquivos já gravados: não treina, não extrai e não consulta teste nenhum (`AVALIAR_TESTE_FINAL = False`). As figuras estão em alta resolução em `results/slides/` (lista em `results/slides/INDEX.md`). Tudo o que não foi ensinado na trilha está marcado com "⚠️ Fora da trilha".
''')


def glossario(cells):
    i = indice(cells, "### Glossário em linguagem simples")
    s = cells[i].source
    s = s.replace("| **Peso igual por fonte** | cada fonte conta o mesmo no treino da regressão, para a maior não dominar (testado na v6). |",
                  "| **Peso igual por fonte** | cada fonte conta o mesmo no treino da regressão, para a maior não dominar (testado na v6). Na v7, é a receita **T4**: o estúdio, a internet real e o RADAR somam o mesmo peso (e, dentro de cada um, reais e IA também). |")
    assert "receita **T4**" in s
    s += r'''| **Sistema atual / anterior** | **atual = v7 (T3)**; **anterior = v4 (F0)**. Mesmo pipeline e mesma regressão; muda o treino. |
| **Receita** (T0 a T4) | uma "receita de treino": quais dados entram no treino da regressão. **T0** = o F0 sem retreinar; **T1** = estúdio + internet real (ML-ITW); **T2** = estúdio + IA moderna em canal degradado (RADAR); **T3** = estúdio + os dois; **T4** = T3 com peso igual por fonte. |
| **Teste mestre fixo** | um conjunto de teste escolhido **antes** do treino e **igual para todas as receitas**, para que a comparação seja justa: muda a receita, o teste fica parado. Foi avaliado uma vez por receita. |
| **Visto × não visto no treino** | "visto" = a receita treinou com aquele **tipo** de áudio (por exemplo, outros locutores do mesmo banco, ou outro idioma da mesma fonte); "não visto" = nada daquele banco entrou no treino. Um bom resultado em algo visto vale menos como prova de generalização. |
| **Vazamento** | quando algo do teste "vaza" para o treino ou para as escolhas, e o número do teste fica otimista. Exemplo: **a mesma pessoa** nos dois lados. O modelo pode aprender a reconhecer a **voz da pessoa** (ou o microfone dela) em vez de "real × IA", e acertar no teste por isso. |
| **Locutor atômico** | todo o áudio de um locutor fica **do mesmo lado** (treino, validação ou teste); nunca se divide uma pessoa. |
| **Bloco inglês** | o recorte "só inglês" do teste (RADAR en, internet real em inglês, In-the-Wild e controles de estúdio), para separar o efeito do idioma do efeito do tipo de áudio. |
| **Controle de esquecimento** | os testes antigos de estúdio (LlamaPartialSpoof, ASVspoof, BRSpeech-DF, CORAA): servem para ver se aprender a internet fez o sistema **piorar** onde ele já ia bem. |
| **Calibração do corte** | se a mesma nota quer dizer a mesma coisa em todo lugar. Não quer: a **escala da nota muda entre ambientes** (estúdio, internet, português), e um corte que acusa 6% dos reais num banco pode acusar 75% em outro. |
| **TPR / FPR** | TPR = % dos áudios de IA detectados; FPR = % dos reais acusados como IA (o falso alarme). |
| **IC95** | intervalo de confiança de 95% por bootstrap (sorteios com reposição). No RADAR, que não tem locutor, o IC é por clipe e por isso **otimista** (estreito demais). |
'''
    cells[i] = md(s)


def config_v7():
    src = base.config_v4()
    velho = '''# ---------------- documento final do sistema v4 (F0) ----------------
# Nada abaixo muda o método: só diz de onde o notebook lê. Camadas, C e limiares são LIDOS de results/
# (gravados pela v4 e conferidos por scripts/v4final_figuras.py), não digitados aqui.
DOCUMENTO = "v4 final (F0)"
F0_CABECA = "concat_top3"                       # a cabeça final escolhida pela validação na v4
ARQ_F0 = RESULTS_DIR / "v4final_resumo.json"    # valores confirmados do F0 (scripts/v4final_figuras.py)
print(f"{DOCUMENTO} | AVALIAR_TESTE_FINAL = {AVALIAR_TESTE_FINAL} | lê de {RESULTS_DIR}/")'''
    novo = '''# ---------------- documento da v7: sistema atual = v7 (T3); sistema anterior = v4 (F0) ----------------
# Nada abaixo muda o método: só diz de onde o notebook lê. Camadas, C, receitas e cortes são LIDOS de results/
# (gravados pela v4 e pela v7), não digitados aqui. Protocolo da v7: docs/PROTOCOLO_V7.md.
DOCUMENTO = "v7 (T3)"
SISTEMA_ATUAL = "v7 (T3)"
SISTEMA_ANTERIOR = "v4 (F0)"
F0_CABECA = "concat_top3"                       # a cabeça final escolhida pela validação na v4 (camadas 5, 6, 7)
ARQ_F0 = RESULTS_DIR / "v4final_resumo.json"    # valores confirmados do F0 (scripts/v4final_figuras.py)
ARQ_T3 = RESULTS_DIR / "v7_t3_modelo.json"      # hiperparâmetros e conferência da T3 (scripts/v7_t3_exportar.py)
SPLITS_V7 = "data/processed/splits_v7.parquet"  # divisão congelada da v7 (fora do git; SHA256 no protocolo)
print(f"{DOCUMENTO} | anterior: {SISTEMA_ANTERIOR} | AVALIAR_TESTE_FINAL = {AVALIAR_TESTE_FINAL} | lê de {RESULTS_DIR}/")'''
    assert src.count(velho) == 1
    return src.replace(velho, novo)


def secao10(n):
    ti = n.ti
    rr, ml = ti.loc["radar_eval"], ti.loc["ml_itw"]
    a = n.tia
    meta = a[a.tipo.str.startswith("classificador") & (a.modelo == "regressão logística")]
    rmeta = meta[(meta.fonte == "radar_eval") & meta.atributos.str.startswith("protocolo")]["AUC metadados (fora da dobra)"].iloc[0]
    mmeta = meta[(meta.fonte == "ml_itw") & meta.atributos.str.startswith("protocolo")]["AUC metadados (fora da dobra)"].iloc[0]
    par = n.tis[n.tis["análise"].str.startswith("a) RADAR com durações pareadas")].iloc[0]
    C = [md(rf'''
## 10. A prova externa: o F0 cai

**Por que esta seção existe.** A seção 10 da versão anterior deste documento recomendava: "antes de apresentar um número final como definitivo, avaliar o F0 **uma vez** num banco nunca visto, com o corte declarado antes". Foi o que fizemos em 13/11, com dois bancos novos, por um protocolo congelado antes (`docs/PROTOCOLO_TESTE_INTERNET.md`; resultado em `docs/RESULTADO_TESTE_INTERNET.md`):
- **RADAR eval**: vozes de IA **comerciais e modernas** (de plataformas), passadas por um **canal degradado simulado** (codecs, ruído, música, compressão), em 6 idiomas; {mil(rr.n_real)} reais e {mil(rr.n_falso)} de IA; **sem locutor no arquivo**.
- **ML-ITW**: vídeos **reais de redes sociais** (YouTube, TikTok, X...), reais e de IA, em 14 idiomas; {mil(ml.n_real)} reais e {mil(ml.n_falso)} de IA, {int(ml.locutores_real)} / {int(ml.locutores_falso)} locutores; o rótulo de IA vem do contexto da postagem.

Nada foi treinado nem ajustado nesses dados: modelo, camadas, C e cortes A e B são os da v4.
'''),
         *bloco_slide("f_00_queda_externa", "Figura f_00 · O F0 por tipo de áudio",
                      "a AUC do F0 (sistema anterior) em cada tipo de áudio, com o intervalo de confiança de 95% quando existe. Em cinza, bancos que o projeto já usava (validação de estúdio, entrevistas da internet, fala lida com trechos de IA, português de audiolivro); em preto, os dois bancos novos, avaliados uma vez.",
                      f"as barras cinza ficam entre {br(n.g('T0', 'brspeech_df'))} e {br(n.rep.loc['val', 'AUC reajuste'])}; as pretas, em {br(rr.AUC)} (RADAR) e {br(ml.AUC)} (ML-ITW). O intervalo do ML-ITW é largo ({br(ml['AUC IC2.5'])} a {br(ml['AUC IC97.5'])}) porque é por locutor; o do RADAR é estreito, mas otimista (por clipe).",
                      "o F0 acerta bem o que se parece com o que ele já conhecia e cai nos dois tipos de áudio que nunca viu: IA moderna em canal ruim e vídeos reais de redes sociais. Em palavras: sorteando um áudio real e um de IA, o F0 dá nota maior ao de IA em ~70% das vezes nesses bancos, contra ~98% nas entrevistas da internet."),
         md(rf'''
**O que faz:** mostra a tabela por fonte (AUC, EER, e TPR/FPR nos cortes A e B fixados na v4) e por idioma, lidas de `results/teste_internet_*.csv`. **Por quê:** os números exatos da prova externa.
'''),
         code(r'''
ti = pd.read_csv(R / "teste_internet_por_fonte.csv")
cols = ["fonte", "n_real", "n_falso", "locutores_real", "locutores_falso", "marca", "AUC", "AUC IC2.5", "AUC IC97.5", "EER",
        "TPR A", "FPR A", "TPR B", "FPR B", "corte_EER_da_fonte"]
display(ti[cols].round(3))
tid = pd.read_csv(R / "teste_internet_por_idioma.csv")
display(Markdown("**Por idioma** (RADAR: IC por clipe, otimista; ML-ITW: < 30 locutores por classe = indício):"))
display(tid[["fonte", "valor", "n_real", "n_falso", "locutores_real", "locutores_falso", "marca", "AUC", "AUC IC2.5", "AUC IC97.5"]].round(3))
'''),
         md(rf'''
**Os cortes A e B.** O corte **A** (calibrado no estúdio) pega {pct(rr["TPR A"], 0)} dos falsos do RADAR e {pct(ml["TPR A"], 0)} dos do ML-ITW, mas acusa {pct(rr["FPR A"], 0)} e {pct(ml["FPR A"], 0)} dos reais. O corte **B** (calibrado nas entrevistas da internet) quase não acusa reais ({pct(rr["FPR B"])} e {pct(ml["FPR B"])}), mas **deixa passar {pct(1 - rr["TPR B"], 0)} dos falsos do RADAR e {pct(1 - ml["TPR B"], 0)} dos do ML-ITW**: de cada 4 falsos, 3 passam como reais. O corte que daria o ponto de EER em cada banco fica **abaixo dos dois** (RADAR {br(rr.corte_EER_da_fonte, 2)}; ML-ITW {br(ml.corte_EER_da_fonte, 2)}): nesses áudios, os falsos recebem notas mais baixas do que o F0 espera.
'''),
         *bloco_slide("ti_04_roc", "Figura ti_04 · Onde os cortes A e B caem na curva ROC",
                      "para cada corte possível, a % de reais acusados (eixo x) e a % de falsos detectados (eixo y), nos dois bancos novos, com os pontos dos cortes A e B.",
                      "o ponto do corte B fica embaixo, à esquerda: quase nenhum real acusado, mas poucos falsos detectados.",
                      "um corte escolhido em outro ambiente não serve aqui; é o primeiro sinal de que **a escala da nota muda entre ambientes** (seção 15).",
                      origem="scripts/teste_internet_figuras.py"),
         md(rf'''
**Auditoria de atalho e de duração.** Antes de culpar o modelo, conferimos se a queda vinha de algo trivial:
- **Duração:** no RADAR, os falsos são um pouco mais longos. Pareando reais e falsos por faixa de 0,5 s (mesmo número por faixa; 20 sorteios), a AUC fica em **{br(par.AUC)}** (contra {br(rr.AUC)} no banco original): **a duração não explica a queda**.
- **Atalho por metadados:** uma regressão logística que só vê duração, silêncio nas bordas, volume (RMS), pico e tamanho do arquivo chega a **{br(rmeta)}** no RADAR e **{br(mmeta)}** no ML-ITW. Fica abaixo da regra de alerta (0,7), mas **perto do F0** (a 0,06 a 0,07): nesses bancos, parte da diferença real × IA está em características de gravação, e o F0 não estava muito acima disso.
'''),
         *bloco_slide("ti_05_atalho", "Figura ti_05 · O sistema × um classificador que só vê metadados",
                      "a AUC do F0 ao lado da AUC de um classificador que só olha metadados (duração, silêncio, volume, tamanho), nos dois bancos novos.",
                      "as barras dos metadados ficam abaixo de 0,7, mas perto das do F0.",
                      "não há um atalho forte que explique tudo, mas o F0 quase não passava de \"características de gravação\" nesses bancos. Esse foi o motivo para a v7 tentar ensinar o sistema com esse tipo de áudio.",
                      origem="scripts/teste_internet_figuras.py"),
         *bloco_slide("ti_03_duracao", "Figura ti_03 · A duração não explica a queda",
                      "a AUC do F0 por faixa de duração original e no RADAR com durações pareadas entre reais e falsos.",
                      "as faixas ficam todas perto de 0,7; os clipes curtos são um pouco piores, como na v4.",
                      "a queda é geral, não é efeito de clipes curtos nem da diferença de duração entre as classes.",
                      origem="scripts/teste_internet_figuras.py"),
         md(r'''
**Conclusão da seção 10:** o F0 foi bem onde já tinha sido medido e **caiu nos dois bancos novos** (AUC ~0,7), em todos os idiomas, inclusive o inglês. Os cortes da v4 não servem ali. Duração e metadados não explicam a queda. Esses dois bancos ficaram **consumidos como teste do F0**: qualquer mudança feita olhando para eles os contamina, e por isso a v7 os redividiu com um protocolo próprio (seção 11).
''')]
    return C


def secao11(n):
    c = n.cont
    def cc(b, s, k):
        g = c[(c.bloco == b) & (c.split == s) & (c.classe == k)]
        return int(g.clipes.iloc[0]) if len(g) else 0
    rt = c[(c.bloco == "radar_eval") & (c.split == "treino")].set_index("classe")
    reg = n.regras
    r1 = reg[reg.regra.str.startswith("(1)")].iloc[0]
    r2 = reg[reg.regra.str.startswith("(2)")].iloc[0]
    r4 = reg[reg.regra.str.startswith("(4)")].groupby("split").locutores.sum()
    comp = {(x["bloco"], x["label"]): x["n"] for x in n.t3["treino"]["composicao"]}
    C = [md(rf'''
## 11. v7: o novo desenho dos dados

**Por que mudar.** O F0 só tinha visto inglês de estúdio e audiolivro e caiu nos dois tipos de áudio novos (seção 10). A pergunta da v7 é simples: **se o treino da regressão incluir uma parte desses áudios, o sistema melhora nos outros áudios do mesmo tipo, que ele não viu?** Para responder sem se enganar, três regras, escritas e **congeladas antes do treino** em `docs/PROTOCOLO_V7.md`:
1. **Separar por locutor e por idioma.** No ML-ITW, cada locutor vai inteiro para um lado só (locutor atômico). O RADAR não informa o locutor, então foi separado **por idioma**.
2. **Um teste mestre fixo**, o mesmo para todas as receitas, escolhido antes de treinar.
3. **A receita é escolhida pela validação**, antes de abrir o teste; o teste mostra todas.

**Como cada banco foi dividido** (`docs/PROTOCOLO_V7.md`, seção 2):
- **Estúdio (os dados da v4):** treino = os mesmos 40.000 da v4; validação = os 19.999 da v4 **menos 13** (duplicatas, seção 12) = 19.986. Nada do estúdio vai para o teste mestre.
- **Internet real (ML-ITW), nesta ordem:** (1) locutor com **algum áudio em português** → inteiro no teste mestre ({r1.locutores} locutores, {mil(r1.clipes)} clipes); (2) locutor com **uma classe só** → teste mestre ({r2.locutores} locutores); (3) as **3 pessoas que também estão no In-the-Wild** (figuras públicas; cruzamento por nome) → o lado delas no In-the-Wild (2 na validação, 1 no teste mestre); (4) os demais locutores com as duas classes → **sorteio por locutor** (seed 42), 60% treino / 15% validação / 25% teste mestre, estratificado pelo idioma principal (chinês, inglês, outro): {r4.get("treino", 0)} / {r4.get("validacao", 0)} / {r4.get("teste_mestre", 0)} locutores.
- **IA moderna em canal degradado (RADAR):** treino = chinês (zh), chinês de Taiwan (zh-tw) e japonês (ja); validação = inglês de Singapura (en-sg) e vietnamita (vi); **teste mestre = inglês (en)**. No treino, só a versão com **durações pareadas** (mesmo número de reais e de IA por faixa de 0,5 s, dentro de cada idioma): {mil(rt.loc["real", "clipes_pareados_treino"])} / {mil(rt.loc["falso", "clipes_pareados_treino"])} de {mil(rt.loc["real", "clipes"])} / {mil(rt.loc["falso", "clipes"])}. O **dev do RADAR nunca é usado** (nem foi baixado).
- **Entrevistas e discursos (In-the-Wild):** nunca no treino; os 16 locutores do dev_shift vão para a **validação**; os 38 do `test_final` (a amostra de 10.000 clipes da v4) para o **teste mestre**.
- **Controles de esquecimento:** os 5 testes antigos de estúdio e português da v4 ({mil(cc("controle", "controle", "real"))} reais / {mil(cc("controle", "controle", "falso"))} de IA), fora da métrica principal.

**O ajuste do bloco inglês.** O inglês do RADAR (en) foi reservado **inteiro** para o teste mestre, e os idiomas da validação (en-sg e vi) também ficaram fora do treino. Assim, o RADAR en no teste mede **"idioma novo, com a mesma fonte, os mesmos geradores e o mesmo canal do treino"**. O protocolo define ainda um **bloco inglês** à parte (seção 13): RADAR en (natural e pareado por duração), internet real **só em inglês**, In-the-Wild e os controles de estúdio em inglês.

**O que NÃO entra no treino de nenhuma receita** (pelo protocolo):
- o **In-the-Wild** inteiro (entrevistas e discursos): é o único banco que **nenhuma** receita vê;
- **todo o português**: os locutores do ML-ITW com algum áudio em português vão inteiros para o teste mestre; BRSpeech-DF e CORAA são controles;
- do RADAR: o **inglês (en)**, o **inglês de Singapura (en-sg)** e o **vietnamita (vi)**; o **dev** do RADAR. (O **japonês (ja) entra no treino**, junto com zh e zh-tw.)
- do ML-ITW: os locutores de uma classe só e as 3 pessoas que também estão no In-the-Wild;
- os controles de estúdio e os 13 clipes excluídos.
'''),
         *bloco_slide("f_02_desenho_dos_dados", "Figura f_02 · O desenho dos dados da v7",
                      "para cada tipo de fonte (linhas) e cada lado (colunas: treino, validação, teste mestre), o número de clipes reais / de IA, a proporção (barra azul = real, laranja = IA) e o número de locutores. \"Não entra\" = nada daquela fonte naquele lado.",
                      "o estúdio só aparece no treino e na validação; a internet real e o RADAR aparecem nos três lados; as entrevistas da internet nunca aparecem no treino. A faixa verde embaixo resume as travas: nenhuma pessoa, nenhum arquivo e nenhum áudio idêntico em dois lados.",
                      "o treino da v7 é o do F0 **mais** uma parte dos dois bancos novos; o teste mestre tem outras pessoas (internet real) e outro idioma (RADAR) do que o treino."),
         md(r'''
**O que faz:** mostra as contagens de clipes e locutores por bloco e lado (`results/v7_splits_contagens.csv`) e as regras do ML-ITW (`results/v7_splits_mlitw_regras.csv`; os nomes das 3 pessoas públicas são trocados por uma descrição). **Por quê:** a tabela exata por trás da figura.
'''),
         code(r'''
cont = pd.read_csv(R / "v7_splits_contagens.csv")
t = cont.pivot_table(index=["bloco", "split"], columns="classe", values=["clipes", "locutores", "clipes_pareados_treino",
                                                                        "clipes_pareados_teste"], aggfunc="sum")
display(t.fillna("—"))
reg = pd.read_csv(R / "v7_splits_mlitw_regras.csv")
reg["regra"] = reg.regra.str.replace(r"^\(3\).*$", "(3) mesma pessoa que um locutor do In-the-Wild (figura pública)", regex=True)
display(reg.groupby(["regra", "split"], as_index=False)[["locutores", "clipes", "falsos"]].sum())
'''),
         md(rf'''
**As 5 receitas e a pergunta que cada uma responde** (o modelo é sempre o mesmo: XLS-R congelado, camadas 5, 6, 7, regressão logística; muda só o treino):

| receita | treino | peso | C (pela validação) | pergunta |
|:--|:--|:--|:--|:--|
| **T0** | o F0 sem retreinar (os 40.000 de estúdio) | nenhum | {cg(n.tre["C_escolhido"]["T0"])} (da v4) | quanto o sistema anterior acerta no teste mestre? (a referência) |
| **T1** | estúdio + internet real ({mil(comp[("ml_itw", 0)] + comp[("ml_itw", 1)])} clipes) | classes balanceadas | {cg(n.tre["C_escolhido"]["T1"])} | a internet real ajuda? e ajuda no RADAR, que ela não viu? |
| **T2** | estúdio + IA moderna em canal degradado ({mil(comp[("radar_eval", 0)] + comp[("radar_eval", 1)])} clipes pareados) | classes balanceadas | {cg(n.tre["C_escolhido"]["T2"])} | o RADAR ajuda? e ajuda na internet real, que ele não viu? |
| **T3** | estúdio + os dois ({mil(n.t3["treino"]["n"])} clipes) | classes balanceadas | {cg(n.tre["C_escolhido"]["T3"])} | juntar os dois é melhor que cada um sozinho? |
| **T4** | o mesmo da T3 | **peso igual por fonte** (estúdio, internet real e RADAR somam o mesmo) | {cg(n.tre["C_escolhido"]["T4"])} | impedir que o estúdio (o maior) domine muda alguma coisa? |

O C de cada receita foi escolhido na grade 0,0001 a 10 pela **maior média da AUC por fonte na validação** (6 fontes: DFADD, ShiftySpeech, CodecFake+, In-the-Wild dev, ML-ITW validação, RADAR validação).
''')]
    return C


def secao12(n):
    sj = n.sj["salvaguardas"]
    return [md(rf'''
## 12. Como evitamos vazamento

**A regra da mesma pessoa e do mesmo corpus.** Se a mesma pessoa aparece no treino e no teste, o modelo pode acertar o teste por **reconhecer a voz** (ou o microfone, a sala, o canal daquela pessoa), e não por distinguir real de IA. O número fica otimista sem ninguém perceber. Por isso, na internet real e nas entrevistas, **todo o áudio de um locutor fica de um lado só**, e as pessoas que aparecem nos dois bancos (cruzamento **por nome**: {len(sj["coincidentes"])} figuras públicas) foram tratadas como **a mesma pessoa** e postas no lado delas no In-the-Wild. O mesmo vale para o corpus: nada do dev do RADAR (que repete dados de outros bancos) foi usado.

**As travas automáticas** (`scripts/v7.py splits`; cada uma é um teste no código: se falha, a divisão para e grava `logs/v7_BLOQUEIO.txt`):

| trava | o que verifica | resultado (`results/v7_splits.json`) |
|:--|:--|:--|
| caminho | o mesmo arquivo em dois lados | **{sj["caminhos_em_comum"]}** em comum |
| locutor | a mesma pessoa em dois lados (ML-ITW e In-the-Wild, com as 3 pessoas coincidentes mapeadas) | **{sj["locutores_em_comum"]}** em comum |
| duplicata exata | o vetor do áudio idêntico (hash) entre treino, validação e teste | **parou a 1ª execução**; depois da correção, **{sj["duplicatas_exatas_entre_lados"]}** |
| duplicata quase exata | cosseno ≥ 0,9995 entre treino/validação e teste mestre (internet real e entrevistas) | **{len(sj["quase_duplicatas_pares"])}** pares; o máximo entre lados foi {br(sj["cosseno_maximo_entre_lados"])} |
| dev do RADAR | nenhum clipe do dev | conferido: {sj["dev_do_radar"]} |

**O que a trava achou.** Na 1ª execução, a trava de duplicata exata **parou a divisão**: 12 grupos de áudios **reais do CodecFake+** (o mesmo áudio de referência repetido em pastas de geradores diferentes) tinham vetor idêntico entre o **treino** e a **validação de estúdio da v4**. É herança do plano da v4, não da v7. Como o treino de estúdio é fixo (os 40.000 da v4), os **{len(sj["duplicatas_exatas_excluidas_da_avaliacao"])} clipes da validação** envolvidos foram **excluídos da avaliação** (`split = "excluido"`), e a divisão rodou de novo sem nenhum aviso. Desvio registrado no protocolo: a regra pedia "mover o par para o mesmo lado", o que mudaria o treino da v4.

**O RADAR foi dividido por idioma** porque o arquivo não traz o locutor: não dá para garantir que a mesma pessoa não está em dois idiomas. Separar por idioma é o melhor possível com o que existe, e o IC do RADAR é marcado como **otimista**.

**Protocolo antes do treino.** O protocolo foi congelado às 19:36 de 2026-11-13, **antes** de qualquer treino; a divisão tem SHA256 `{n.t3["splits_v7"]["sha256"][:12]}…` (conferido de novo por `scripts/v7_t3_exportar.py`); a escolha pela validação foi gravada às 19:41:00 (`results/v7_escolha.json`), antes do teste; cada receita foi avaliada no teste mestre **uma vez**.
'''),
            *bloco_slide("f_08_anti_vazamento", "Figura f_08 · As travas contra vazamento",
                         "cada trava automática, o que ela verifica e o resultado, lidos de `results/v7_splits.json`.",
                         "a linha laranja: a única trava que disparou, o que ela achou e como foi resolvido.",
                         "as travas funcionaram (uma delas pegou um problema herdado da v4). O limite que sobra: o cruzamento de pessoas entre bancos foi feito **pelo nome**, e o RADAR não tem locutor."),
            md(r'''
**O que faz:** mostra o bloco de salvaguardas de `results/v7_splits.json`, sem os caminhos e sem os nomes das pessoas. **Por quê:** os números da tabela acima vêm daqui.
'''),
            code(r'''
sj = ler(R / "v7_splits.json")["salvaguardas"]
resumo = {k: v for k, v in sj.items() if k not in ("coincidentes", "duplicatas_exatas_excluidas_da_avaliacao", "quase_duplicatas_pares")}
resumo["pessoas coincidentes entre os bancos (nº) e lado"] = sorted(sum(sj["coincidentes"].values(), []))
resumo["clipes excluídos (duplicatas exatas)"] = len(sj["duplicatas_exatas_excluidas_da_avaliacao"])
resumo["fonte dos excluídos"] = sorted({d["fonte"] for d in sj["duplicatas_exatas_excluidas_da_avaliacao"]})
display(pd.Series(resumo, name="valor").astype(str).to_frame())
''')]


def secao13(n):
    pr, val = n.pr, n.val
    m = {rc: pr.loc[rc, "média AUC (3 fontes)"] for rc in pr.index}
    li = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw_ingles")].iloc[0]
    lp = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw_portugues")].iloc[0]
    at = n.at
    def amet(f, med):
        return at[(at.fonte == f) & (at["medido em"] == med) & (at.como == "treinado no lado de treino")]["AUC metadados"].iloc[0]
    cvm = at[(at.como.str.startswith("validação cruzada"))].set_index("fonte")["AUC metadados"]
    ph = n.ph
    pha = ph[ph["análise"].str.startswith("(a)")].set_index("receita")
    phb = ph[ph["análise"].str.startswith("(b)")].set_index("receita")
    br_t0, br_t3 = n.g("T0", "brspeech_df"), n.g("T3", "brspeech_df")
    ctr = ("llamapartialspoof", "asvspoof2019_la", "asvspoof2021_df", "brspeech_df")
    queda = min(n.g("T3", r) - n.g("T0", r) for r in ctr)
    du = n.dur[n.dur.receita.isin(["T0", "T3"])].pivot(index="faixa", columns="receita", values="AUC")
    n_uma_classe = int(n.regras[n.regras.regra.str.startswith("(2)")].locutores.iloc[0])
    C = [md(rf'''
## 13. Resultados da v7

### 13.1 A escolha pela validação (antes do teste)

A receita vencedora é a de **maior média da AUC por fonte na validação**: T0 {br(val.loc["T0", "AUC média por fonte"])}, T1 {br(val.loc["T1", "AUC média por fonte"])}, T2 {br(val.loc["T2", "AUC média por fonte"])}, **T3 {br(val.loc["T3", "AUC média por fonte"])}**, T4 {br(val.loc["T4", "AUC média por fonte"])}. Venceu a **T3**, gravada às 19:41:00, antes de o teste ser aberto. A T4 ficou a {br(val.loc["T3", "AUC média por fonte"] - val.loc["T4", "AUC média por fonte"])}: seria empate se o protocolo tivesse margem, mas ele não tinha.

**O que faz:** mostra a AUC por fonte na validação e a grade de C de cada receita. **Por quê:** é a decisão, com os números exatos.
'''),
         code(r'''
display(pd.read_csv(R / "v7_validacao.csv").round(3))
gc = pd.read_csv(R / "v7_grade_c.csv")
display(gc.pivot(index="C", columns="receita", values="AUC val média por fonte").round(4))
'''),
         md(rf'''
### 13.2 O teste mestre: a métrica principal

A métrica principal é a **média da AUC** em três fontes do teste mestre: **internet real** (ML-ITW, outros locutores), **entrevistas e discursos** (In-the-Wild, 38 locutores, nunca visto) e **IA moderna em canal degradado** (RADAR en, idioma novo). Também a **pior fonte**.

| receita | média (IC95) | pior fonte | internet real | entrevistas | RADAR en |
|:--|:--|:--|:--|:--|:--|
''' + "\n".join(
            f"| {'**' + rc + '** ★' if rc == 'T3' else rc} | {br(m[rc])} ({br(pr.loc[rc, 'média IC2.5'])} a {br(pr.loc[rc, 'média IC97.5'])}) | "
            f"{br(pr.loc[rc, 'AUC pior fonte'])} | {br(pr.loc[rc, 'AUC ml_itw'])} | {br(pr.loc[rc, 'AUC in_the_wild'])} | {br(pr.loc[rc, 'AUC radar_en'])} |"
            for rc in ("T0", "T1", "T2", "T3", "T4")) + rf'''

★ = escolhida pela validação. **Visto × não visto:** a internet real foi vista por T1, T3 e T4 (outros locutores); o RADAR (outro idioma) por T2, T3 e T4; as entrevistas **nunca**. Diferença T3 − T0 na média: +{br(pr.loc["T3", "dif. média vs T0"])} (IC95 pareado +{br(pr.loc["T3", "dif. média vs T0 IC2.5"])} a +{br(pr.loc["T3", "dif. média vs T0 IC97.5"])}). **Atenção:** o F0 aqui é medido no **teste mestre** (um subconjunto fixo), por isso a internet real dá {br(n.g("T0", "ml_itw"))} e não {br(n.ti.loc["ml_itw", "AUC"])} (banco inteiro, seção 10).
'''),
         *bloco_slide("f_01_antes_depois", "Figura f_01 · Antes e depois: F0 × T3 por tipo de áudio",
                      "para cada tipo de áudio, a AUC do F0 (cinza) e da T3 (vinho) no teste mestre, com IC95. Ponto vinho cheio = tipo de áudio que a T3 **nunca viu** no treino; ponto claro com contorno = tipo visto (outros locutores do mesmo banco, ou outro idioma da mesma fonte).",
                      f"os dois saltos grandes são nos tipos vistos: internet real {br(n.g('T0', 'ml_itw'), 2)} → {br(n.g('T3', 'ml_itw'), 2)} e RADAR en {br(n.g('T0', 'radar_en'), 2)} → {br(n.g('T3', 'radar_en'), 2)}. Nos nunca vistos, as entrevistas ficam onde estavam ({br(n.g('T0', 'in_the_wild'), 2)} → {br(n.g('T3', 'in_the_wild'), 2)}) e o português de audiolivro sobe ({br(br_t0, 2)} → {br(br_t3, 2)}).",
                      "onde o tipo de áudio foi visto, a AUC sobe muito; onde não foi, não piora. Isso é bom, mas **não é a mesma coisa que generalizar para geradores novos** (seção 14)."),
         *bloco_slide("f_03_receitas", "Figura f_03 · As 5 receitas, fonte por fonte",
                      "a AUC de cada receita (T0 a T4) em cada fonte do teste mestre e em dois controles. Barra clara e hachurada = a receita viu aquele tipo de áudio no treino; barra cheia = não viu.",
                      f"T1 é fraca no RADAR ({br(n.g('T1', 'radar_en'), 2)}) e T2 é mais fraca na internet real ({br(n.g('T2', 'ml_itw'), 2)}); T3 e T4 ficam altas nas duas; as entrevistas mudam pouco em todas.",
                      f"**juntar os dois bancos é melhor que cada um sozinho**; T3 e T4 empatam na média ({br(m['T3'])} × {br(m['T4'])}): o peso igual por fonte não mudou nada demonstrável."),
         md(r'''
**O que faz:** mostra a tabela principal (`results/v7_principal.csv`) e o teste mestre completo de cada receita (AUC, IC95, TPR e FPR nas três regras de corte). **Por quê:** os números exatos por trás das figuras.
'''),
         code(r'''
pr = pd.read_csv(R / "v7_principal.csv")
display(pr[["receita", "descrição", "média AUC (3 fontes)", "média IC2.5", "média IC97.5", "pior fonte", "AUC pior fonte",
            "AUC ml_itw", "AUC in_the_wild", "AUC radar_en", "dif. média vs T0", "dif. média vs T0 IC2.5",
            "dif. média vs T0 IC97.5", "vencedora pela validação"]].round(3))
tm = pd.read_csv(R / "v7_teste_mestre.csv")
cols = ["receita", "recorte", "domínio", "n_real", "n_falso", "locutores_real", "locutores_falso", "marca", "AUC",
        "AUC IC2.5", "AUC IC97.5", "regra escolhida (validação)", "TPR A'", "FPR A'", "TPR M", "FPR M", "TPR F5", "FPR F5"]
display(tm[tm.papel == "principal"][cols].round(3))
'''),
         md(rf'''
### 13.3 O sinal de generalização mais limpo

A pergunta mais honesta é: **treinar com um banco novo ajuda no outro banco novo, que a receita não viu?** A T1 (estúdio + internet real) nunca viu o RADAR, e a T2 (estúdio + RADAR) nunca viu a internet real.
- **Treinar só com internet real sobe o RADAR en:** {br(n.g("T0", "radar_en"))} → **{br(n.g("T1", "radar_en"))}** (T1; IC por clipe, otimista).
- **Treinar só com o RADAR sobe a internet real:** {br(n.g("T0", "ml_itw"))} → **{br(n.g("T2", "ml_itw"))}** (T2; IC95 {n.ic("T2", "ml_itw")}, por locutor).
'''),
         *bloco_slide("f_04_sinal_de_generalizacao", "Figura f_04 · Um banco novo ajuda o outro",
                      "setas do F0 para a receita, **só** em tipos de áudio que a receita não viu no treino. Em cima, a T1 (que não viu o RADAR); embaixo, a T2 (que não viu a internet real).",
                      "as duas setas grandes, em negrito: RADAR en com a T1 e internet real com a T2. As demais (entrevistas, estúdio, português) mudam pouco ou sobem um pouco.",
                      "aprender um tipo de áudio \"difícil\" ajudou no outro, que a receita não viu. **Por que** isso acontece (os dois têm canal degradado? os dois têm geradores modernos?) **é hipótese, não testada**."),
         md(rf'''
### 13.4 O bloco inglês

Para separar o efeito do idioma, o protocolo pede um recorte só em inglês. Na T3: RADAR en {br(n.g("T3", "radar_en"))} (natural) e {br(n.g("T3", "radar_en_pareado"))} (durações pareadas); internet real só em inglês {br(n.g("T3", "ml_itw_ingles"))} (IC95 {n.ic("T3", "ml_itw_ingles")}; **indício**: {int(li.locutores_real)} / {int(li.locutores_falso)} locutores real / IA); entrevistas {br(n.g("T3", "in_the_wild"))}; fala lida com trechos de IA {br(n.g("T3", "llamapartialspoof"))}. **Pareado × natural é igual**: a duração não explica o RADAR. Por faixa de duração no RADAR en, os clipes curtos continuam os mais difíceis (T3: < 2 s {br(du.loc["< 2 s", "T3"])}; ≥ 6 s {br(du.loc[">= 6 s", "T3"])}).
'''),
         *bloco_slide("f_06_ingles", "Figura f_06 · O bloco inglês por receita",
                      "uma tabela colorida: linhas = recortes em inglês, colunas = receitas, célula = AUC (mais escuro = maior). Células hachuradas = tipo de áudio visto no treino daquela receita. A coluna da T3 está contornada.",
                      "a linha da internet real em inglês sobe de 0,58 (T0) para 0,94 a 0,96 nas receitas que a viram; as linhas de baixo (estúdio) quase não mudam.",
                      "em inglês, a T3 sobe onde o F0 falhava e não perde onde ele já ia bem; a linha da internet real em inglês tem poucos locutores (indício)."),
         md(rf'''
### 13.5 Controles de estúdio (esquecimento)

Aprender a internet poderia fazer o sistema **esquecer** o estúdio. Não aconteceu: a maior queda de AUC da T3 nos controles foi **{br(queda)}**. O português de audiolivro (BRSpeech-DF) até **sobe**: {br(br_t0)} → {br(br_t3)}. No português real só com reais (CORAA), a % de reais acusados cai de {pct(n.g("T0", "coraa", "FPR A'"))} (F0, regra A') para {pct(n.g("T3", "coraa", "FPR M"))} (T3, regra M).
'''),
         *bloco_slide("f_05_estudio_nao_piorou", "Figura f_05 · O estúdio não piorou",
                      "a AUC do F0 e da T3 nos testes antigos de estúdio e português (nenhum deles entrou no treino) e, à direita, a % de reais do CORAA acusados como IA.",
                      "os pontos de estúdio quase não se mexem; o português de audiolivro anda para a direita; a barra do CORAA diminui.",
                      "não há esquecimento em AUC. Mas a calibração do português piora (seção 15): ordenar bem não é o mesmo que cortar bem."),
         md(rf'''
### 13.6 Atalho por metadados

Repetimos a auditoria de atalho com o desenho da v7: uma regressão logística que só vê **duração, silêncio, volume, pico e tamanho do arquivo**, treinada no lado de treino de cada banco novo. No teste mestre, ela chega a **{br(amet("ml_itw", "teste mestre"))}** na internet real e **{br(amet("radar_eval", "teste mestre"))}** no RADAR en (com validação cruzada dentro do teste mestre: {br(cvm["ml_itw"])} e {br(cvm["radar_eval"])}). Nenhuma passa de 0,7, e as receitas treinadas ficam em 0,94 a 0,96: **os ganhos não são explicados por esses metadados**. Mas, no T0 da internet real ({br(n.g("T0", "ml_itw"))}), os metadados já chegavam praticamente ao nível do modelo.
'''),
         code(r'''
display(pd.read_csv(R / "v7_atalho.csv").drop(columns=["atributos"]).round(3))
'''),
         md(rf'''
### 13.7 Pós-hoc: a internet real dentro do mesmo locutor

**Feito depois de ver o teste** (não muda receita, regra nem escolha). O teste mestre da internet real tem {n_uma_classe} locutores de uma classe só; um modelo poderia acertar "quem é o locutor" em vez de "real × IA". Dois recortes:
- **(a) só locutores com as duas classes:** T0 {br(pha.loc["T0", "AUC"])} → T3 **{br(pha.loc["T3", "AUC"])}** (IC95 {br(pha.loc["T3", "AUC IC2.5"])} a {br(pha.loc["T3", "AUC IC97.5"])}).
- **(b) AUC dentro de cada locutor** (≥ 5 clipes por classe; {int(phb.loc["T3", "locutores_real"])} locutores), média entre locutores: T0 {br(phb.loc["T0", "AUC"])} → T3 **{br(phb.loc["T3", "AUC"])}** (mediana {br(phb.loc["T0", "AUC mediana entre locutores"])} → {br(phb.loc["T3", "AUC mediana entre locutores"])}).

O ganho **se mantém dentro do mesmo locutor**: não é efeito de separar locutores de uma classe só.
'''),
         code(r'''
ph = pd.read_csv(R / "v7_poshoc.csv")
display(ph[["análise", "receita", "locutores_real", "locutores_falso", "AUC", "AUC IC2.5", "AUC IC97.5",
            "AUC mediana entre locutores", "AUC min", "AUC max"]].round(3))
'''),
         md(rf'''
### 13.8 O português, à parte

O protocolo reporta o português **à parte, como indício**:
- **Internet real em português** (ML-ITW, {int(lp.locutores_real)} / {int(lp.locutores_falso)} locutores; nenhum áudio em português entrou no treino): AUC {br(n.g("T0", "ml_itw_portugues"))} (T0) → **{br(n.g("T3", "ml_itw_portugues"))}** (T3), IC95 {n.ic("T3", "ml_itw_portugues")}: largo demais para concluir. Na regra escolhida (M), a T3 pega só **{pct(n.g("T3", "ml_itw_portugues", "TPR M"))}** dos falsos em português (acusa {pct(n.g("T3", "ml_itw_portugues", "FPR M"))} dos reais).
- **Português de audiolivro** (BRSpeech-DF, controle): AUC alta ({br(br_t3)} na T3), mas **{pct(n.g("T3", "brspeech_df", "FPR M"))} dos reais são acusados** no corte da regra M ({pct(n.g("T0", "brspeech_df", "FPR A'"))} no F0, regra A'). A ordem está boa; a escala da nota, não (seção 15).
- **Português real só com reais** (CORAA): {pct(n.g("T3", "coraa", "FPR M"))} de reais acusados na T3 (regra M).

**Conclusão da seção 13:** a T3, escolhida pela validação, sobe a média do teste mestre de {br(m["T0"])} para {br(m["T3"])}; os ganhos estão onde o tipo de áudio foi visto, o que nunca foi visto não piora, um banco novo ajuda o outro, e os metadados não explicam os ganhos. O português continua sem resposta firme, e o corte é o novo ponto fraco.
''')]
    return C


def secao14(n, limitacoes_v4):
    li = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw_ingles")].iloc[0]
    lp = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw_portugues")].iloc[0]
    tmi = n.tm[(n.tm.receita == "T3") & (n.tm.recorte == "ml_itw")].iloc[0]
    return [md(rf'''
## 14. O que não dá para afirmar

- **O teste mestre não é virgem.** O F0 já tinha sido avaliado nesses mesmos clipes (bancos inteiros, 13/11), e as entrevistas da internet (In-the-Wild) foram vistas da v4 à v6. Nada da v7 foi escolhido olhando o teste, mas quem desenhou a v7 já conhecia o resultado agregado do F0. **Só clones próprios, gravados depois, seriam um teste intocado.**
- **O RADAR en mede idioma novo, não gerador novo.** No RADAR, os geradores e o canal do teste (inglês) são os mesmos do treino (chinês e japonês). O "0,96" diz que o sistema aprendeu os falsos do RADAR e os reconhece em outro idioma, não que ele pega geradores que nunca viu.
- **A internet real tem poucos locutores por classe.** No teste mestre inteiro: {int(tmi.locutores_real)} / {int(tmi.locutores_falso)} locutores (real / IA); **só em inglês: {int(li.locutores_real)} / {int(li.locutores_falso)}**, abaixo de 30 por classe: **indício, não estimativa**. Todas as plataformas também ficam abaixo de 30.
- **O português de audiolivro é limpo.** Os reais do BRSpeech-DF vêm de audiolivros (LibriVox, 10 locutores): é fala lida e limpa, longe de uma ligação ou de um áudio de aplicativo. O português "de internet" tem só {int(lp.locutores_real)} / {int(lp.locutores_falso)} locutores.
- **O corte não se transfere entre ambientes.** Com o mesmo corte, a T3 acusa de {pct(min(n.g("T3", r, "FPR M") for r in ("ml_itw", "in_the_wild", "radar_en", "llamapartialspoof", "asvspoof2019_la", "asvspoof2021_df", "coraa")), 0)} a {pct(n.g("T3", "brspeech_df", "FPR M"), 0)} dos reais, conforme o banco (seção 15).
- **Nada sobre ligações ou WhatsApp reais.** Nenhum banco é ligação telefônica real: o RADAR **simula** o canal, e o ML-ITW são vídeos de redes sociais.
- **A causa do ganho cruzado é hipótese.** Por que treinar com o RADAR ajuda a internet real (e vice-versa)? Pode ser o canal degradado, os geradores modernos em comum, ou outra coisa. **Não foi testado.**
- **Uma única divisão e uma única regressão por receita** (seed 42): a variação de sorteio do treino não está medida. As entrevistas no teste mestre são a amostra de 10.000 clipes da v4, não os 21.165 do `test_final`.
- **Estes números não valem como resultado do desafio RADAR**: o desafio pede para não treinar com o eval, e nós treinamos com parte dele (zh, zh-tw, ja). Também não são comparáveis aos da avaliação dos bancos inteiros (seção 10). A licença do ML-ITW não está declarada.

**Limitações que vêm da v4 e continuam valendo** (texto da versão anterior deste documento):

{limitacoes_v4}
'''),
            *bloco_slide("f_09_o_que_nao_afirmamos", "Figura f_09 · O que o teste mostra e o que não mostra",
                         "um quadro com duas colunas: à esquerda, o que os números da v7 sustentam; à direita, o que eles ainda não sustentam.",
                         "cada item da direita é uma ressalva que deve acompanhar qualquer número da v7 apresentado.",
                         "a v7 é um avanço medido com cuidado, não uma prova de que o sistema funciona em golpes reais.")]


def secao15(n):
    esc = n.esc["por_receita"]["T3"]
    cv = n.cv[n.cv.receita == "T3"].set_index("regra")
    rot = {"A'": "A' = ponto de EER da validação agrupada", "M": "M = mediana dos pontos de EER de cada fonte",
           "F5": "F5 = no máximo 5% de reais acusados em cada fonte (mediana entre fontes)"}
    linhas = "\n".join(f"| {rot[k]} | {br(esc['cortes'][k], 3)} | {br(cv.loc[k, 'acurácia balanceada média (deixando uma fonte de fora)'], 4)} |"
                       for k in ("A'", "M", "F5"))
    return [md(rf'''
## 15. Cortes: decisão em aberto

A AUC não depende do corte, mas um sistema de verdade precisa de um. O protocolo calculou **três regras de corte, só com a validação**, e escolheu entre elas por validação cruzada deixando uma fonte de fora (maior acurácia balanceada média):

| regra (calculada na validação da T3) | corte (logit) | acurácia balanceada média, deixando uma fonte de fora |
|:--|:--|:--|
{linhas}

As três **empatam na prática** (diferença de {br(cv["acurácia balanceada média (deixando uma fonte de fora)"].max() - cv["acurácia balanceada média (deixando uma fonte de fora)"].min(), 4)}); a regra M ficou marcada pelo protocolo quase por sorteio. E o problema maior não é qual das três: é que **a escala da nota muda entre ambientes**. A mesma nota quer dizer coisas diferentes no estúdio, na internet e no português, então qualquer corte único acusa muitos reais em algum lugar.

**A escolha da regra (e se haverá um corte por ambiente) para o sistema final fica em aberto, e será decidida depois, só pela validação.** Este documento **não escolhe**: mostra as três.
'''),
            *bloco_slide("f_07_cortes_e_calibracao", "Figura f_07 · A escala da nota e os três cortes",
                         "à esquerda, a nota média dos reais (azul) e dos áudios de IA (laranja) em cada fonte, na T3, com as três linhas de corte da validação; à direita, a % de reais acusados (FPR, azul) e de IA detectada (TPR, laranja) em cada regra (quadrado = A', círculo = M, triângulo = F5). Faixa bege = português.",
                         f"no português de audiolivro, até a nota média dos **reais** fica acima de todos os cortes ({br(n.g('T3', 'brspeech_df', 'nota média real'), 1)}), e {pct(n.g('T3', 'brspeech_df', 'FPR M'), 0)} dos reais são acusados; na internet real em português, a média dos falsos fica perto do corte e só ~{pct(n.g('T3', 'ml_itw_portugues', 'TPR M'), 0)} são pegos; as três regras dão pontos quase iguais em todas as fontes.",
                         "o corte é o novo ponto fraco: as três regras empatam e nenhuma serve para todos os ambientes. Um corte por ambiente (ou calibrar a nota por ambiente) é o próximo passo natural, a decidir só pela validação."),
            md(r'''
**O que faz:** mostra, para a T3, os três cortes da validação e, por fonte do teste mestre e dos controles, o TPR e o FPR em cada regra e a nota média de reais e de IA. **Por quê:** os números da figura, sem escolher regra nenhuma.
'''),
            code(r'''
esc = ler(R / "v7_escolha.json")["por_receita"]["T3"]
display(Markdown(f"**Cortes da T3 (logit), calculados na validação:** " +
                 ", ".join(f"{k} = {v:.3f}" for k, v in esc["cortes"].items()) +
                 f". Acurácia balanceada média (deixando uma fonte de fora): " +
                 ", ".join(f"{k} = {v:.4f}" for k, v in esc["loso"].items()) + "."))
tm = pd.read_csv(R / "v7_teste_mestre.csv")
t3 = tm[tm.receita == "T3"].set_index("recorte")
display(t3[["papel", "n_real", "n_falso", "nota média real", "nota média falso", "TPR A'", "FPR A'", "TPR M", "FPR M",
            "TPR F5", "FPR F5"]].round(3))
''')]


def secao16():
    return [md(r'''
## 16. Próximos passos (não executados)

1. **Corte por ambiente.** Calibrar o corte (ou a nota) separadamente para estúdio, internet e telefone/aplicativo, escolhendo só pela validação. É o problema mais claro que a v7 deixou (seção 15).
2. **Português no treino.** Reais e falsos em português, de bancos que não estão nos testes, e um conjunto de decisão em português, para que as escolhas deixem de ser cegas para o idioma do uso final.
3. **Clones próprios como teste intocado.** Gravar e clonar vozes do grupo (com consentimento), **depois** de fechar o sistema, e avaliar uma única vez: é o único teste realmente virgem que temos como fazer (a pasta `data/raw/clones_proprios` ainda não existe).
4. **O contexto de golpe.** Áudio curto, comprimido, de telefone ou de aplicativo de mensagem; definir o corte pelo custo de cada erro nesse uso (acusar um parente de verdade × deixar passar um golpe).
5. **O sistema final.** Um único pipeline (áudio → XLS-R → regressão da T3 → nota → decisão) empacotado para o grupo, com os pesos de `results/v7_t3_coeficientes.npz`, o corte decidido pela validação e o domínio de uso escrito junto.
''')]


# ------------------------------------------------------------------------------------------------ montagem
def montar():
    n = N()
    C = base.montar()
    # --- introdução e glossário
    C[0] = introducao(n)
    glossario(C)
    # --- configuração (célula única "config")
    ic = [i for i, c in enumerate(C) if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(ic) == 1
    C[ic[0]] = code(config_v7(), tags=["config"])
    trocar_texto(C, "**O que faz:** a célula abaixo (tag `config`) é **a configuração da v4**, mais um bloco curto que diz onde este documento lê os valores do F0.",
                 "**O que faz:** a célula abaixo (tag `config`) é **a configuração da v4** (o pipeline e as camadas não mudaram na v7), mais um bloco curto que diz onde este documento lê os valores do sistema anterior (F0) e do atual (T3).")
    # --- funções de apoio: as da v4 + as da v7 (figuras reduzidas e a T3)
    ia = indice(C, "def pendente(o_que):", tipo="code")
    C[ia] = code(C[ia].source + "\n" + AJUDA_V7.strip("\n"))
    trocar_texto(C, "**O que faz:** carrega pequenas funções para mostrar figuras e tabelas de `results/` e imprime os valores do F0 **confirmados**",
                 "**O que faz:** carrega pequenas funções para mostrar figuras (reduzidas, para o arquivo ficar leve) e tabelas de `results/` e imprime os valores do sistema anterior (F0) e do atual (T3) **confirmados**")
    # --- rótulos de sistema nas seções da v4
    trocar_texto(C, "## 3. Mapa do banco", "## 3. Mapa do banco (o banco da v4; o desenho da v7 está na seção 11)")
    trocar_texto(C, "## 4. O pipeline do sistema v4, passo a passo",
                 "## 4. O pipeline, passo a passo (o mesmo no sistema anterior, v4 (F0), e no atual, v7 (T3))")
    trocar_texto(C, "**O pipeline do F0:**", "**O pipeline (igual no F0 e na T3):**")
    trocar_texto(C, "## 7. Conferências e diagnósticos", "## 7. Conferências e diagnósticos (sistema anterior: v4 (F0))")
    trocar_texto(C, "## 8. Resultados no teste (F0)", "## 8. Resultados no teste da v4 (sistema anterior: v4 (F0))")
    trocar_texto(C, "**Conclusão do teste:**", "**Conclusão do teste da v4 (sistema anterior):**")
    # seção 5: a regressão da T3
    i5 = indice(C, "## 5. A NOSSA PARTE: a regressão logística")
    t3 = n.t3
    comp = {(x["bloco"], x["label"]): x["n"] for x in t3["treino"]["composicao"]}
    C.insert(i5 + 1, md(rf'''
**No sistema atual, v7 (T3), a regressão é a mesma, com três diferenças** (lidas de `results/v7_t3_modelo.json`):
- **Os dados de treino:** {mil(t3["treino"]["n"])} áudios = estúdio ({mil(comp[("estudio", 0)])} reais / {mil(comp[("estudio", 1)])} de IA, os mesmos do F0) + internet real ({mil(comp[("ml_itw", 0)])} / {mil(comp[("ml_itw", 1)])}) + IA moderna em canal degradado ({mil(comp[("radar_eval", 0)])} / {mil(comp[("radar_eval", 1)])}, durações pareadas).
- **O freio:** **C = {cg(t3["regressao"]["C"])}** (mais freio que o F0, C = 0,1), escolhido pela validação na grade 0,0001 a 10.
- **O peso das classes:** `class_weight = "balanced"` (a internet real tem muito mais reais que falsos; com o peso, reais e falsos contam o mesmo no total). A padronização (média e desvio) é recalculada no treino da T3.

Os pesos da T3 (3.072 coeficientes, o intercepto, a média e o desvio da padronização) estão em `results/v7_t3_coeficientes.npz`, conferidos contra as notas gravadas da v7 (diferença máxima {t3["conferencia"]["dif_max_nota_logit"]:.1e} de logit; AUC igual com 3 casas em {len(t3["conferencia"]["auc_3_casas"])} recortes). O texto abaixo descreve o **F0**, que foi onde as escolhas de camada e de C foram feitas.
'''))
    # seção 9: a v7 na tabela do caminho
    trocar_texto(C, "## 9. O caminho até aqui (v1 a v6)", "## 9. O caminho até aqui (v1 a v7)")
    trocar_texto(C, "| v6 | funil: Meta × Microsoft, fusão, RL × MLP × CNN, ajuste fino (pela AUC da pior fonte) | F2: média de acurácia 0,829 | pior que o F0 no teste; o notebook voltou ao F0 |",
                 "| v6 | funil: Meta × Microsoft, fusão, RL × MLP × CNN, ajuste fino (pela AUC da pior fonte) | F2: média de acurácia 0,829 | pior que o F0 no teste; o notebook voltou ao F0 |\n"
                 f"| prova externa | o F0, sem mudança, em dois bancos novos (RADAR e ML-ITW) | AUC {br(n.ti.loc['radar_eval', 'AUC'])} e {br(n.ti.loc['ml_itw', 'AUC'])} | o F0 cai fora do que já conhecia (seção 10) |\n"
                 f"| **v7** | **parte dos bancos novos no treino**, teste mestre fixo, 5 receitas (seções 11 a 13) | média no teste mestre {br(n.p('T0', 'média AUC (3 fontes)'))} → {br(n.p('T3', 'média AUC (3 fontes)'))} (T3) | **sistema atual = T3**, escolhida pela validação |")
    # --- tira as seções 10 (limitações) e 11 (próximos passos) antigas; as limitações viram parte da seção 14
    i10 = indice(C, "## 10. Limitações")
    lim = C[i10].source.split("\n", 2)[2].strip()
    lim = lim.replace("**Recomendação:** antes de apresentar um número final como definitivo, avaliar o F0 **uma vez** num banco nunca visto, com o corte declarado antes.",
                      "(Essa recomendação foi seguida: seção 10.)")
    i11 = indice(C, "## 11. Próximos passos (não executados)")
    for i in sorted([i10, i11], reverse=True):
        del C[i]
    # --- seções novas
    C += secao10(n) + secao11(n) + secao12(n) + secao13(n) + secao14(n, lim) + secao15(n) + secao16()
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-executar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    assert Path("notebooks/versions/v4_final.ipynb").exists(), "o v4_final.ipynb precisa existir (é preservado)"
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (env)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = montar()
    assert sum("config" in c.get("metadata", {}).get("tags", []) for c in nb.cells) == 1
    assert not [c for c in nb.cells if "PREENCHER_" in c.source]
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
            for p in PROIBIDOS:
                if re.search(rf"\b{re.escape(p)}\b", t):
                    achados.append((i, p))
    if achados:
        print("ATENÇÃO, nomes proibidos no notebook:", sorted(set(achados)))
    nbf.write(nb, "notebook.ipynb")
    mb = Path("notebook.ipynb").stat().st_size / 2 ** 20
    ok = not (erros or mb > LIMITE_MB or achados)
    if ok and not a.sem_executar:
        shutil.copyfile("notebook.ipynb", "notebooks/versions/v7_final.ipynb")
    print(f"notebook.ipynb: {mb:.2f} MB, {len(nb.cells)} células, {erros} erro(s), {len(achados)} nome(s) proibido(s), "
          f"{time.time() - t0:.0f} s; cópia em notebooks/versions/v7_final.ipynb: {'sim' if ok and not a.sem_executar else 'não'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
