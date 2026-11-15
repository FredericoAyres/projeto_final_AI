#!/usr/bin/env python3
"""Monta e executa o notebook do SISTEMA PRINCIPAL (v7 = T3), com a v8 (ajuste fino) como anexo NÃO ADOTADO. Só LÊ do disco.

Decisão do grupo (15/11/2026, posterior ao protocolo da v8): o sistema principal é a v7 (T3); a v8 fica como
experimento não adotado. Motivos e números: docs/DECISAO_SISTEMA_PRINCIPAL.md.

- Reaproveita `scripts/nb_v7.py` (seções 1 a 16) e `scripts/nb_v8.py` (seções 17 a 22, que viram o anexo), sem
  alterar nenhum dos dois e sem chamar o `main` deles (que copiariam por cima de notebooks/versions/v7_final.ipynb e
  v8_finetuning.ipynb, preservados).
- Todos os números do texto novo são LIDOS de results/ na hora de montar.
- Grava notebook.ipynb e copia para notebooks/versions/principal_v7.ipynb (só se: 0 erros, < 10 MB e nenhum nome de
  pessoa ou de máquina no texto e nas saídas).

    python scripts/nb_principal.py [--sem-executar]
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
import nb_v7 as v7nb  # noqa: E402  (só as funções)
import nb_v8 as v8nb  # noqa: E402  (só as funções)

import nbformat as nbf  # noqa: E402
import pandas as pd  # noqa: E402

md, code, br, pct, mil, cg = v7nb.md, v7nb.code, v7nb.br, v7nb.pct, v7nb.mil, v7nb.cg
d4 = v8nb.d4
R = Path("results")
LIMITE_MB = 10.0
ARQUIVO = "notebooks/versions/principal_v7.ipynb"
PRESERVADOS = ("notebooks/versions/v4_final.ipynb", "notebooks/versions/v7_final.ipynb",
               "notebooks/versions/v8_finetuning.ipynb")
# nomes de pessoas e de máquinas: a lista do nb_v7 (que inclui a do nb_final) + o que apareceu na v8
PROIBIDOS = v7nb.PROIBIDOS + ["Ayres", "FredericoAyres", "GeForce", "NVIDIA", "Laptop", "placa de vídeo de 6 GB",
                              "6 GB", "8 GB", "GPU de"]


def trocar(cells, velho, novo, n_min=1):
    n = 0
    for c in cells:
        if velho in c.source:
            c.source = c.source.replace(velho, novo)
            n += 1
    assert n >= n_min, velho
    return n


def trocar_re(cells, padrao, novo):
    n = 0
    for c in cells:
        s2, k = re.subn(padrao, novo, c.source, flags=re.S)
        if k:
            c.source, n = s2, n + k
    assert n == 1, (padrao, n)


# ------------------------------------------------------------------------------------------------ números (lidos)
class Nums:
    def __init__(self):
        self.v7 = v7nb.N()
        self.v8 = v8nb.N8()
        self.esc8 = self.v8.esc
        self.pesos = json.loads((R / "v8_r1_pesos.json").read_text(encoding="utf-8"))
        self.t3 = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))
        assert self.esc8["receita_vencedora_pela_validacao"] == "R1"

    def corte(self, c):
        e = self.esc8["por_candidato"][c]
        return e["regra"], e["cortes"][e["regra"]]

    def fpr(self, c, rec):
        """% de reais acusados na regra escolhida pela validação de cada rodada."""
        ct = self.v8.ct
        x = ct[(ct.candidato == c) & (ct.recorte == rec)].iloc[0]
        return x[f"FPR {x['regra escolhida (validação)']}"]

    def med_real(self, c, rec):
        tm = pd.concat([self.v8.tm, self.v8.ct])
        return tm[(tm.candidato == c) & (tm.recorte == rec)].iloc[0]["nota mediana real"]


# ------------------------------------------------------------------------------------------------ textos novos
def intro(n, v7intro_src):
    v7, v8 = n.v7, n.v8
    m0 = v7.p("T0", "média AUC (3 fontes)")
    m3 = v7.p("T3", "média AUC (3 fontes)")
    m8 = v8.p("R1 (média das seeds)", "média AUC (3 fontes)")
    s = v7intro_src
    s = re.sub(r"^# .*\n", "", s.strip() + "\n")
    s = re.sub(r"^> \*\*Sistema atual: v7 \(T3\)\.\*\*.*\n", "", s, flags=re.M)
    s = s.replace("## 1. Introdução", "## 1. Introdução\n\n**Resultados da v7 (o sistema principal)**", 1)
    velho = "As seções 10 a 16 são a **v7**."
    assert velho in s
    s = s.replace(velho, velho + " As **partes A, B e C** (logo depois da seção 16) explicam **o que é a T3**, o **modelo leve** "
                  "(compressão) e o **guia para quem vai construir um site**. O **anexo** (seções 17 a 22) é o **experimento de "
                  "ajuste fino da v8, não adotado**.")
    return md(rf'''
# Detector de voz gerada por IA: o sistema principal (v7, T3)

> **Resumo em 5 linhas**
> 1. **Sistema principal = v7 (T3):** XLS-R 300M **congelado** → média no tempo das camadas 5, 6 e 7 → padronização → **regressão logística** (C = {v7nb.cg(n.t3["regressao"]["C"])}). Os pesos são números no próprio repositório (`results/v7_t3_coeficientes.npz` e `results/v7_t3_modelo.json`).
> 2. **Sistema anterior = v4 (F0):** o mesmo caminho, treinado só com estúdio. No teste mestre da v7, a média da AUC foi de {br(m0)} (F0) para **{br(m3)} (T3)**.
> 3. **Experimento = v8 (ajuste fino dos blocos 5 a 7 do XLS-R), não adotado.** A regra da v8, pela validação, indicou a receita R1; no teste mestre ela deu {br(m8)} (média das 2 sementes), {d4(v8.p("R1 (média das seeds)", "dif. média vs R0 (T3)"))} sobre a T3.
> 4. **Mantivemos a v7 como sistema principal** (decisão do grupo em 15/11/2026, depois do protocolo da v8): ganho pequeno, só 2 sementes, corte instável entre as sementes, pesos fora do repositório, treino muito mais caro e nenhuma melhora confiável no corte em português. Detalhes no anexo e em `docs/DECISAO_SISTEMA_PRINCIPAL.md`.
> 5. **Limites que valem para tudo:** o teste mestre não é virgem; o RADAR en mede idioma novo, não gerador novo; o corte não se transfere entre ambientes (no português de audiolivro, BRSpeech-DF, {pct(n.fpr("R0", "brspeech_df"), 0)} dos reais são acusados no corte da T3); nada aqui fala de ligações ou de WhatsApp reais.

{s}
''')


def anexo_abertura(n):
    v8 = n.v8
    rc = n.esc8["AUC_media_por_receita (R1 = média das seeds)"]
    ra1, c1 = n.corte("R1_s42")
    ra2, c2 = n.corte("R1_s123")
    ra0, c0 = n.corte("R0")
    pj = n.pesos["arquivos"]
    mb = sum(a["MB"] for a in pj.values())
    h = {c: v8.tre[c]["historico"] for c in ("R1_s42", "R1_s123")}
    seg = [sum(e["segundos"] for e in h[c]) / 60 for c in h]
    dif_t3 = f"{n.t3['conferencia']['dif_max_nota_logit']:.1e}".replace(".", ",").replace("e-05", "e-5")
    return md(rf'''
# Anexo: experimento de ajuste fino (v8), não adotado

> **Mantivemos a v7 como sistema principal.** A v8 é reportada abaixo (seções 17 a 22), mas o grupo decidiu, em **15/11/2026**, **não adotá-la**. Essa decisão é **posterior ao protocolo da v8** (`docs/PROTOCOLO_V8.md`) e **não altera** o protocolo nem os resultados.

**O que a regra da v8 escolheu.** Pela validação (`results/v8_escolha.json`, gravado às {n.esc8["quando"][11:19]} de {n.esc8["quando"][:10]}, antes do teste), a vencedora foi a **R1** (camadas 5, 6 e 7 + cabeça, cabeça começando na T3): média da AUC por fonte **{br(rc["R1"], 4)}** contra {br(rc["R0"], 4)} da T3 (R2 {br(rc["R2"], 4)}; R3 {br(rc["R3"], 4)}). **O grupo decidiu não adotá-la**, pelos motivos abaixo.

**Por que mantivemos a v7** (números lidos de `results/`):
1. **O ganho é pequeno.** No teste mestre, a média da AUC subiu {d4(v8.p("R1_s42", "dif. média vs R0 (T3)"))} (semente 42) e {d4(v8.p("R1_s123", "dif. média vs R0 (T3)"))} (semente 123) sobre a T3 ({br(v8.p("R0", "média AUC (3 fontes)"))} → {br(v8.p("R1_s42", "média AUC (3 fontes)"))} / {br(v8.p("R1_s123", "média AUC (3 fontes)"))}). Na pior fonte (ML-ITW), o IC pareado começa abaixo de zero ({br(v8.p("R1_s42", "dif. pior vs R0 IC2.5"), 4)} e {br(v8.p("R1_s123", "dif. pior vs R0 IC2.5"), 4)}).
2. **Só 2 sementes.** As duas concordam entre si, mas 2 sementes não medem a variação do treino; e não há tempo para mais.
3. **O corte e a calibração não ficaram estáveis.** O corte escolhido na validação (logit) foi {ra0} = {br(c0, 2)} na T3, {ra1} = {br(c1, 2)} na semente 42 e {ra2} = {br(c2, 2)} na semente 123. A escala da nota também mudou: a nota mediana dos reais do ASVspoof 2019 foi de {br(n.med_real("R0", "asvspoof2019_la"), 1)} (T3) para {br(n.med_real("R1_s42", "asvspoof2019_la"), 1)} / {br(n.med_real("R1_s123", "asvspoof2019_la"), 1)}.
4. **Os pesos ajustados não estão no repositório.** São ~{br(mb, 0)} MB (2 × {br(list(pj.values())[0]["MB"], 1)} MB), numa release do GitHub; a T3 são {v7nb.mil(n.t3["n_numeros"] + 1)} números (3.072 coeficientes + o intercepto, mais a média e o desvio da padronização) no git.
5. **Custo e reprodução.** A regressão da T3 treina sobre vetores já extraídos e é determinística (solver lbfgs; refeita, a nota bate com a gravada a {dif_t3} de logit). Cada rodada da R1 levou ~{br(seg[0], 0)} min só de treino ({br(seg[1], 0)} min na outra semente), mais a validação inteira (~25 min, pelo `docs/HISTORICO_VERSOES.md`); a fila inteira da v8 levou ~11 h.
6. **O corte em português não melhorou de forma confiável.** % de reais do BRSpeech-DF acusados, na regra de cada rodada: T3 {pct(n.fpr("R0", "brspeech_df"))}; R1 semente 42 {pct(n.fpr("R1_s42", "brspeech_df"))}; semente 123 {pct(n.fpr("R1_s123", "brspeech_df"))}. No CORAA (só reais): {pct(n.fpr("R0", "coraa"))} → {pct(n.fpr("R1_s42", "coraa"))} / {pct(n.fpr("R1_s123", "coraa"))}.

> **Quadro de limitações da v8**
> - O teste mestre **não é virgem** (a T3 já tinha sido avaliada nele, e quem desenhou a v8 conhecia esse número).
> - **2 sementes** na R1 e **1** nas outras receitas: o ganho passa do ruído de sortear o teste, mas com 2 sementes não dá para afirmar que passa do ruído do treino.
> - Ganho **pequeno** e **não garantido na pior fonte** (ML-ITW).
> - A **escala da nota mudou** e o corte variou entre as sementes; um corte único continua sem servir para todos os ambientes.
> - **Um único treino curto** (2 épocas), sem busca de hiperparâmetros.

**O que mudaria a decisão:** mais sementes da R1 (**3 ou mais**) confirmando o ganho sobre a T3 e um **corte estável** entre elas, escolhido só pela validação.
''')


def proximos_finais():
    return md(r'''
# Próximos passos (não executados)

Os passos da seção 16 continuam valendo para o sistema principal (v7, T3): corte por ambiente, português no treino, clones próprios como teste intocado, o contexto de golpe e o empacotamento da T3. Do experimento v8, ficam:
1. **Mais sementes da R1** (3 ou mais), só para saber se o ganho passa do ruído do treino. É o que poderia mudar a decisão.
2. **Um corte que fique estável entre as sementes**, escolhido só pela validação (e, de preferência, por ambiente).
3. **Ajuste fino com português no treino**, junto com um conjunto de decisão em português, já que o corte em português não melhorou.

Nada disso foi executado. Qualquer mudança no sistema principal deve ser decidida só pela validação e registrada em `docs/HISTORICO_VERSOES.md` antes de um novo teste.
''')


# ------------------------------------------------------------------------------------------------ blocos A, B e C
# (15/11/2026) O que é a T3, o modelo leve (compressão) e o guia para quem vai construir um site.
# Todos os números são LIDOS de results/ e docs/ na hora de montar; nada é recalculado.
NOME_RECORTE = {
    "ml_itw": "internet real (ML-ITW)", "in_the_wild": "entrevistas e discursos da internet (In-the-Wild, 38 locutores; nunca visto)",
    "radar_en": "IA moderna em canal degradado, inglês (RADAR en)", "radar_en_pareado": "RADAR en, durações pareadas",
    "ml_itw_ingles": "internet real, só inglês (indício)", "ml_itw_portugues": "internet real, só português (indício)",
    "llamapartialspoof": "falso parcial, estúdio (LlamaPartialSpoof)", "brspeech_df": "português de audiolivro (BRSpeech-DF)",
    "coraa": "português falado, só reais (CORAA)", "asvspoof2019_la": "estúdio (ASVspoof 2019)",
    "asvspoof2021_df": "estúdio (ASVspoof 2021 DF)"}
APARELHO = {"gpu": "placa de vídeo do notebook", "cpu4": "processador do notebook, 4 threads",
            "cpu1": "processador do notebook, 1 thread"}
CMP_LEVE = {"V1": "V1 fp32 x V0 fp32", "V2": "V2 fp16 x V0 fp16", "V3": "V3 int8 x V0 fp32"}


def ms(v):
    return "–" if v is None else (mil(round(v)) if v >= 1000 else br(v, 0))


class NumsLeve:
    """Números do modelo leve (results/modelo_leve_*), da v7 (results/v7_*) e as receitas (docs/PROTOCOLO_V7.md)."""

    def __init__(self):
        self.var = pd.read_csv(R / "modelo_leve_variantes.csv").set_index("variante")
        self.lat = pd.read_csv(R / "modelo_leve_latencia.csv")
        self.eq = pd.read_csv(R / "modelo_leve_equivalencia.csv")
        self.res = json.loads((R / "modelo_leve_resumo.json").read_text(encoding="utf-8"))
        self.t3 = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))
        self.esc = json.loads((R / "v7_escolha.json").read_text(encoding="utf-8"))
        self.aval = json.loads((R / "v7_avaliar_concluido.json").read_text(encoding="utf-8"))
        self.val = pd.read_csv(R / "v7_validacao.csv").set_index("receita")
        tm = pd.read_csv(R / "v7_teste_mestre.csv")
        self.tm3 = tm[tm.receita == "T3"].set_index("recorte")
        self.receitas = self._receitas()
        assert self.esc["receita_vencedora_pela_validacao"] == "T3"
        assert self.t3["regra_de_corte_escolhida_no_protocolo"] == "M" == self.esc["por_receita"]["T3"]["regra"]

    @staticmethod
    def _receitas():
        """As linhas da tabela 'receita | treino | pesos' da seção 3 de docs/PROTOCOLO_V7.md."""
        s = (ROOT / "docs" / "PROTOCOLO_V7.md").read_text(encoding="utf-8")
        s = s[s.index("## 3. Receitas"):s.index("## 4.")]
        out = {}
        for lin in s.splitlines():
            m = re.match(r"^\| \*\*(T\d)\*\* \| (.*?) \| (.*?) \|$", lin)
            if m:
                out[m.group(1)] = (m.group(2), m.group(3))
        assert list(out) == ["T0", "T1", "T2", "T3", "T4"], list(out)
        return out

    def t(self, var, ap, dur):
        x = self.lat[(self.lat.variante == var) & (self.lat.aparelho == APARELHO[ap]) & (self.lat.duracao_s == dur)
                     & (self.lat.parte == "total")]
        return None if x.empty else float(x.iloc[0].mediana_ms)

    def t3d(self, var, ap):
        return " / ".join(ms(self.t(var, ap, d)) for d in (3, 5, 8))

    def cmp(self, var):
        x = self.eq[(self.eq.comparacao == CMP_LEVE[var]) & (self.eq.fonte == "(média das 6 fontes)")]
        return x.iloc[0]

    def corte(self, k):
        return float(self.t3["cortes_da_validacao_logit"][k])


def bloco_a(n, L):
    t3 = L.t3
    linhas = []
    for k, (treino, pesos) in L.receitas.items():
        pesos = re.sub(r"\s*Decisão minha:.*$", "", pesos)
        v = L.val.loc[k, "AUC média por fonte"]
        rot = f"**{k}** ★" if k == "T3" else f"**{k}**"
        linhas.append(f"| {rot} | {treino} | {pesos} | {br(v)} |")
    comp = {(c["bloco"], c["label"]): c["n"] for c in t3["treino"]["composicao"]}
    return [md(rf'''
## A. O que é a T3 (e o que é o "sistema principal")

**T3 é o nome de uma receita de treino**, não de um modelo novo. Na v7, testamos **cinco receitas** (T0 a T4). Todas usam **o mesmo caminho** (o XLS-R congelado, as camadas 5, 6 e 7, a média no tempo e uma regressão logística); o que muda é **com quais áudios a regressão é treinada**. O nome vem só da **ordem da tabela** do protocolo: T0 é a primeira linha, T1 a segunda, e assim por diante.

Tabela lida de `docs/PROTOCOLO_V7.md` (seção 3), com a AUC média por fonte **na validação** (`results/v7_validacao.csv`):

| receita | treinada com | pesos no treino | AUC média na validação |
|:--|:--|:--|--:|
{chr(10).join(linhas)}

★ = a vencedora. "ML-ITW" = internet real (vídeos de redes sociais); "RADAR" = IA moderna em canal degradado (em chinês, chinês de Taiwan e japonês no treino).

**A T3 foi escolhida pela validação, antes do teste.** A regra (escrita no protocolo antes de treinar) era: vence a receita com a maior média da AUC por fonte na validação. A escolha foi gravada em `results/v7_escolha.json` às {L.esc["quando"][11:16]} de {L.esc["quando"][8:10]}/{L.esc["quando"][5:7]}/{L.esc["quando"][:4]}; o teste mestre foi avaliado depois (marcador de conclusão às {L.aval["quando"][11:16]}). A T4 ficou a {br(L.val.loc["T3", "AUC média por fonte"] - L.val.loc["T4", "AUC média por fonte"], 3)} da T3, o que na prática é empate, mas o protocolo não tinha margem de empate.

**O "sistema principal"** é, então, a soma de duas partes:
1. o **modelo de fala da Meta** (XLS-R 300M), **congelado** (nenhum peso dele foi treinado por nós) e, na prática, **cortado** depois da 7ª camada (parte B); e
2. a **regressão logística treinada com a receita T3**: a média e o desvio da padronização mais {mil(t3["n_numeros"])} pesos e 1 intercepto, treinados em {mil(t3["treino"]["n"])} áudios (estúdio {mil(comp[("estudio", 0)])} reais / {mil(comp[("estudio", 1)])} IA; internet real {mil(comp[("ml_itw", 0)])} / {mil(comp[("ml_itw", 1)])}; RADAR {mil(comp[("radar_eval", 0)])} / {mil(comp[("radar_eval", 1)])}), com C = {cg(t3["regressao"]["C"])} e classes balanceadas.

Os pesos estão no próprio repositório: `results/v7_t3_coeficientes.npz` e `results/v7_t3_modelo.json`.

> **Atenção a dois nomes parecidos.** No modelo leve (parte B), as variantes se chamam V0 a V3. A **"variante leve V3 (int8)"** é a versão em inteiros de 8 bits e **não tem nada a ver** com a **"v3" do projeto** (a versão do preenchimento de áudios curtos, seção 9). Neste texto, a variante sempre aparece como "variante leve V3 (int8)".
'''),
            md(r'''**O que faz:** mostra a tabela da validação das cinco receitas e a escolha gravada. **Por quê:** os números acima vêm destes arquivos.'''),
            code(r'''
val = pd.read_csv(R / "v7_validacao.csv").set_index("receita")
display(val[[c for c in val.columns if c.startswith("AUC")]].round(3))
esc = ler(R / "v7_escolha.json")
display(Markdown(f"**Vencedora pela validação:** {esc['receita_vencedora_pela_validacao']} (gravado em {esc['quando']}); "
                 f"critério: {esc['criterio']}."))
''')]


def bloco_b(n, L):
    v, res = L.var, L.res
    reg = res["regra"]
    c = {k: L.cmp(k) for k in ("V1", "V2", "V3")}
    lin_param = " | ".join(mil(v.loc[k, "parametros_total"]) for k in ("V0", "V1", "V2", "V3"))
    lin_mb = " | ".join(mil(round(v.loc[k, "mb_total"])) for k in ("V0", "V1", "V2", "V3"))
    cam = " | ".join(br(v.loc[k, "camadas_executadas_gpu"] if pd.notna(v.loc[k, "camadas_executadas_gpu"])
                        else v.loc[k, "camadas_executadas_cpu4"], 0) for k in ("V0", "V1", "V2", "V3"))
    gpu = " | ".join([L.t3d("V0", "gpu") + " (fp16)", L.t3d("V1", "gpu"), "**" + L.t3d("V2", "gpu") + "**", "–"])
    cpu4 = " | ".join([L.t3d("V0", "cpu4"), L.t3d("V1", "cpu4"), "–", "**" + L.t3d("V3", "cpu4") + "**"])
    cpu1 = " | ".join([L.t3d("V0", "cpu1"), L.t3d("V1", "cpu1"), "–", "**" + L.t3d("V3", "cpu1") + "**"])
    queda = " | ".join(["(referência)"] + [f"{br(c[k].queda_auc_media, 4)} / {br(c[k].maior_queda_auc_fonte, 4)}"
                                           for k in ("V1", "V2", "V3")])
    dif = " | ".join(["–"] + [f"{br(c[k].dif_nota_max, 2)} / {br(c[k].dif_nota_mediana, 2)}" for k in ("V1", "V2", "V3")])
    ac = " | ".join(["(referência)"] + ["sim" if res["aceitacao"][k]["aceita"] else "não" for k in ("V1", "V2", "V3")])
    p0, p1 = v.loc["V0", "param_camadas_transformer"], v.loc["V1", "param_camadas_transformer"]
    n_eq = int(c["V1"].n)
    t = L.t
    return [md(rf'''
## B. Modelo leve (compressão)

**O que é compressão de modelo.** É fazer **o mesmo trabalho com um modelo menor ou mais barato de calcular, sem treinar de novo**. Serve para o sistema **rodar mais rápido e ocupar menos espaço**, inclusive **sem placa de vídeo** (só no processador) e, no futuro, talvez em **aparelhos móveis**. Aqui foram testados três jeitos (protocolo `docs/MODELO_LEVE_PROTOCOLO.md`, congelado antes de medir; resultado em `docs/MODELO_LEVE.md`). ⚠️ Fora da trilha: cortar camadas, meia precisão (fp16) e quantização (int8).

**Por que as camadas 8 a 24 podem sair, e as camadas 1 a 4 não.** O XLS-R é uma **cadeia**: o áudio passa pela camada 1, a saída dela entra na 2, a da 2 entra na 3, e assim por diante. O sistema só lê as saídas das camadas **5, 6 e 7**. Então:
- as camadas **8 a 24** vêm **depois** da 7 e não mudam nada do que lemos. Tirar essas camadas **não muda a nota** (a diferença medida foi zero);
- as camadas **1 a 4** vêm **antes**: a saída da 5 é calculada a partir da saída da 4, que vem da 3, e assim por diante. Sem elas, não existe saída da 5.

Detalhe técnico: o modelo inteiro, do jeito que vem do Hugging Face, **calcula as 24 camadas** mesmo quando só usamos até a 7ª ({br(v.loc["V0", "camadas_executadas_cpu4"], 0)} chamadas de camada por áudio contra {br(v.loc["V1", "camadas_executadas_cpu4"], 0)}, contadas no código). Por isso o corte não só diminui o arquivo: ele também **acelera**. As camadas do transformer caem de {mil(p0)} para {mil(p1)} parâmetros.

**As quatro variantes** (`results/modelo_leve_variantes.csv`, `modelo_leve_latencia.csv`, `modelo_leve_equivalencia.csv`; tempos = **mediana** de 30 medidas por duração, lote de 1 áudio, incluindo ler o arquivo, preparar, o modelo e a regressão):

| | V0 completo | V1 cortado (fp32) | V2 cortado, meia precisão (fp16) | variante leve V3 (int8) |
|:--|--:|--:|--:|--:|
| parâmetros do modelo | {lin_param} |
| tamanho em disco (MB, modelo + regressão) | {lin_mb} |
| camadas do transformer executadas por áudio | {cam} |
| **placa de vídeo do notebook**, ms por áudio de 3 / 5 / 8 s | {gpu} |
| **processador do notebook, 4 threads**, ms (3 / 5 / 8 s) | {cpu4} |
| **processador do notebook, 1 thread**, ms (3 / 5 / 8 s) | {cpu1} |
| queda da AUC: média / pior fonte | {queda} |
| diferença da nota (logit): máxima / mediana | {dif} |
| aceita pela regra? | {ac} |

"–" = não medido nessa combinação (a V2 é para placa de vídeo; a variante leve V3 (int8) é para processador). A V1 e a V2 são comparadas com o V0 na mesma precisão (fp32 e fp16); a variante leve V3 (int8) com o V0 em fp32. A regressão soma {mil(v.loc["V0", "param_regressao"])} parâmetros em todas.
'''),
            md(r'''**O que faz:** mostra as três tabelas do modelo leve como estão em `results/`. **Por quê:** a tabela acima foi montada a partir delas.'''),
            code(r'''
var = pd.read_csv(R / "modelo_leve_variantes.csv")
display(var[["variante", "descricao", "parametros_total", "mb_total", "camadas_executadas_cpu4", "pico_gpu_mb",
             "pico_ram_mb_cpu4", "aceita_pela_regra"]].round(1))
lat = pd.read_csv(R / "modelo_leve_latencia.csv")
display(lat[lat.parte == "total"].pivot_table(index=["variante"], columns=["aparelho", "duracao_s"], values="mediana_ms").round(0))
eq = pd.read_csv(R / "modelo_leve_equivalencia.csv")
display(eq[eq.fonte == "(média das 6 fontes)"][["comparacao", "n", "dif_nota_max", "dif_nota_mediana", "auc_media_ref",
                                                 "auc_media_var", "queda_auc_media", "maior_queda_auc_fonte"]].round(4))
'''),
            *v7nb.bloco_slide("ml_01_tamanho_e_tempo", "Figura ml_01 · Tamanho e tempo de cada variante",
                         "em cima, os parâmetros (milhões) e o tamanho em disco (MB) de cada variante; embaixo, o tempo por áudio de 3, 5 e 8 s na placa de vídeo do notebook (V0, V1, V2) e no processador do notebook com 4 threads (V0, V1 e a variante leve V3 (int8)). Cinza = modelo completo; laranja = aceita pela regra.",
                         "no processador, as barras laranja ficam bem abaixo da cinza em todas as durações; na placa de vídeo, só a V2 (fp16) é claramente mais rápida que o V0 (a V1 em fp32 não é comparável com o V0 em fp16).",
                         "o sistema principal cabe num modelo cerca de 3 vezes menor e, cortado, responde em menos de 1 s por áudio de até 8 s no processador do notebook com 4 threads.",
                         origem="scripts/modelo_leve_resumo.py"),
            *v7nb.bloco_slide("ml_02_equivalencia", "Figura ml_02 · A nota muda?",
                         f"à esquerda, a maior diferença de nota (logit) de cada variante contra o modelo completo, nos {mil(n_eq)} clipes da amostra da validação; à direita, a queda da AUC (média e pior fonte), com os limites da regra (linhas tracejada e pontilhada).",
                         f"a V1 e a V2 ficam em zero nos dois painéis; a variante leve V3 (int8) fica abaixo dos limites da AUC, mas a nota muda até {br(c['V3'].dif_nota_max, 1)} logit. (O título da figura encosta no subtítulo: defeito só de layout.)",
                         "a V1 e a V2 podem usar o corte da T3; a variante leve V3 (int8) não, porque o valor da nota muda.",
                         origem="scripts/modelo_leve_resumo.py"),
            md(rf'''
**Quanto tempo leva para ouvir um áudio e responder "real ou IA"?** Para um áudio de **5 s**, neste notebook (lote de 1, mediana):
- **placa de vídeo do notebook, V2 (cortado, fp16): cerca de {ms(t("V2", "gpu", 5))} ms**;
- **processador do notebook, variante leve V3 (int8): cerca de {ms(t("V3", "cpu4", 5))} ms** com 4 threads e {ms(t("V3", "cpu1", 5))} ms com 1 thread;
- **processador do notebook, V1 (cortado, fp32): cerca de {ms(t("V1", "cpu4", 5))} ms** com 4 threads e {ms(t("V1", "cpu1", 5))} ms com 1 thread;
- para comparar, o modelo completo (V0) no processador: {ms(t("V0", "cpu4", 5))} ms (4 threads) e {ms(t("V0", "cpu1", 5))} ms (1 thread).

O que domina o tempo é o modelo; ler o arquivo e preparar o áudio levou {br(L.lat[(L.lat.parte == "a_leitura_e_preparo") & (L.lat.duracao_s == 5)].mediana_ms.min(), 0)} a {br(L.lat[(L.lat.parte == "a_leitura_e_preparo") & (L.lat.duracao_s == 5)].mediana_ms.max(), 0)} ms, e a média, a padronização e a regressão, menos de 1 ms.

**A regra de aceitação (escrita antes de medir).** Uma variante é aceita se, numa amostra de {mil(n_eq)} clipes da **validação** (6 fontes), a AUC média por fonte cair **no máximo {br(reg["queda_max_auc_media"], 3)}** e **nenhuma fonte** cair mais de **{br(reg["queda_max_auc_por_fonte"], 2)}**. A V1 ainda precisava ser **equivalente**: diferença de nota de no máximo {br(reg["dif_max_nota_V1_logit"], 3)} logit. **As três passaram.** A V1 e a V2 deram **nota idêntica** à do modelo completo (diferença {br(c["V1"].dif_nota_max, 0)}). A variante leve V3 (int8) passou pela regra (queda {br(c["V3"].queda_auc_media, 4)} na média e {br(c["V3"].maior_queda_auc_fonte, 4)} na pior fonte), **mas o valor da nota mudou**: até {br(c["V3"].dif_nota_max, 2)} logit, mediana {br(c["V3"].dif_nota_mediana, 2)}.

**Recomendação (para o grupo decidir):**
- **V2 (cortada, meia precisão) é o modelo leve recomendado** quando houver placa de vídeo: nota idêntica à do modelo completo em fp16, {mil(round(v.loc["V2", "mb_total"]))} MB e ~{ms(t("V2", "gpu", 5))} ms por áudio de 5 s.
- **V1 (cortada, precisão normal) é a referência** (nota idêntica à do modelo completo em fp32) e a **opção para processador** quando a nota precisa ser exatamente a da T3 ({mil(round(v.loc["V1", "mb_total"]))} MB; ~{ms(t("V1", "cpu4", 5))} ms com 4 threads).
- **Variante leve V3 (int8): opção COM RESSALVA.** A AUC cai pouco, mas o **valor da nota muda** (mediana {br(c["V3"].dif_nota_mediana, 2)}, máximo {br(c["V3"].dif_nota_max, 2)} logit). Por isso **o corte da T3 ({br(L.corte("M"), 3)}) não vale nela**: seria preciso escolher um corte novo, só pela validação, **e isso não foi feito**. Ela também **não foi medida no teste mestre**.

**Limites.** A equivalência foi medida em {mil(n_eq)} clipes da **validação**, não no teste mestre nem nos controles. A leitura do arquivo foi medida com o **cache de disco aquecido** (arquivo frio seria mais lento). Os tempos são **de um notebook, não de um celular**: o resultado no processador é só uma **referência**. Na variante leve V3 (int8), não foi verificado se a nota muda de forma **sistemática** (sempre para o mesmo lado, o que um corte novo corrigiria) ou **aleatória** (o que nenhum corte corrige); e ela não foi avaliada no teste mestre.
''')]


def bloco_c(n, L):
    tm = L.tm3
    linhas = []
    for rec, rot in NOME_RECORTE.items():
        if rec not in tm.index:
            continue
        x = tm.loc[rec]
        tpr = "–" if pd.isna(x["TPR M"]) else pct(x["TPR M"])
        linhas.append(f"| {rot} | {x['papel']} | {mil(x['n_real'])} / {mil(x['n_falso'])} | {tpr} | {pct(x['FPR M'])} |")
    k = L.esc["por_receita"]["T3"]["loso"]
    return [md(rf'''
## C. Guia para quem vai construir um site

Esta parte é para quem vai **usar** o sistema principal num site. A versão completa (com checklist e erros comuns) está em `docs/GUIA_PARA_O_SITE.md`, e o **gabarito** em código é `scripts/inferencia_referencia.py`.

### C.1 O caminho do áudio até a resposta

Cada passo foi conferido no código (`src/preproc.py`, `src/ssl.py`, `scripts/modelo_leve.py`, `results/v7_t3_modelo.json`):

```
arquivo de áudio
  │  ler e converter para MONO (média dos canais) e 16 kHz       src/preproc.py: ler_mono_16k
  ▼
cortar o silêncio das bordas (quadros 35 dB abaixo do mais forte)
  │  se sobrar menos de 0,5 s, usa o áudio sem corte              src/preproc.py: preparar_clipe
  ▼
recorte central de no máximo 8 s, SEM completar (áudio curto fica com o próprio tamanho)
  ▼
normalizar o clipe (média 0, variância 1)                         src/ssl.py: normalizar
  ▼
XLS-R 300M (Meta), cortado depois da 7ª camada, congelado         src/ssl.py: representar
  ▼
saídas das camadas 5, 6 e 7 (1.024 números a cada 20 ms)
  ▼
média no tempo de cada uma → 3 × 1.024 números
  ▼
colar as três, nessa ordem → 3.072 números (x)
  ▼
padronizar: (x − mu) / sd, com mu e sd do treino da T3            results/v7_t3_coeficientes.npz
  ▼
nota = soma de (x padronizado × coef) + intercepto   (logit; maior = mais parecido com IA)
  ▼
comparar com o corte → "parece real" ou "parece gerada por IA"
```

### C.2 Arquivos necessários

| o quê | onde | no git? |
|:--|:--|:--|
| pesos da regressão: `coef` ({mil(n.t3["n_numeros"])}), `intercepto` (1), `mu` e `sd` ({mil(n.t3["n_numeros"])} cada), `C`, `camadas` | `results/v7_t3_coeficientes.npz` | sim |
| como usar, hiperparâmetros e cortes da validação | `results/v7_t3_modelo.json` | sim |
| modelo da Meta (completo) | Hugging Face, identificador `{n.t3["modelo_ssl"].split(" ")[0]}` | não (baixado do Hugging Face) |
| modelos leves V1, V2 e variante leve V3 (int8) | `models/modelo_leve_V1_fp32.safetensors`, `models/modelo_leve_V2_fp16.safetensors`, `models/modelo_leve_V3_int8.pt` | **não** (regerar com `python scripts/modelo_leve.py tamanho` ou, só o que faltar, com `python scripts/inferencia_referencia.py`) |

### C.3 O corte

A nota é um número; para dizer "real" ou "IA", é preciso um **corte**. A v7 calculou **três regras de corte, só na validação** (`results/v7_t3_modelo.json`):
- **A'** = ponto de EER da validação inteira: {br(L.corte("A'"), 3)} logit;
- **M** = mediana dos pontos de EER de cada fonte: **{br(L.corte("M"), 3)} logit** ← a que a validação escolheu (acurácia balanceada média deixando uma fonte de fora: {br(k["M"], 4)}, contra {br(k["A'"], 4)} da A' e {br(k["F5"], 4)} da F5);
- **F5** = no máximo 5% de reais acusados em cada fonte (mediana entre fontes): {br(L.corte("F5"), 3)} logit.

As três empatam na prática. **A regra final para o produto é DECISÃO EM ABERTO.**

Com o corte M, no teste mestre e nos controles (`results/v7_teste_mestre.csv`; TPR = % das vozes de IA detectadas; FPR = % das vozes reais acusadas):

| tipo de áudio | papel | reais / IA | TPR (corte M) | FPR (corte M) |
|:--|:--|--:|--:|--:|
{chr(10).join(linhas)}

### C.4 Contrato sugerido para o site

- **Entrada:** um arquivo de áudio de qualquer formato que o ffmpeg leia (convertido para mono 16 kHz antes do modelo; o script de referência lê com o soundfile, ou seja, WAV, FLAC, OGG e MP3, e a conversão pelo ffmpeg **não foi testada**). **O modelo só usa os 8 s centrais** (depois do corte de silêncio). Ideia **NÃO testada** para áudios longos: avaliar janelas de 8 s e combinar as notas.
- **Saída:** a **nota**; um **nível de alerta "baixo / atenção / alto"**, com cortes **ainda a decidir**; e sempre a mensagem **"isto é um alerta, não um veredito"**.
- **O que o site NÃO deve prometer:** que detecta qualquer voz de IA; que funciona em ligações, WhatsApp ou celular; que funciona igual em português; uma "probabilidade de ser IA" (a escala da nota muda entre ambientes, seção 15).

### C.5 Limites honestos

- **Nenhum áudio em português entrou no treino:** o estúdio é inglês; da internet real (ML-ITW), todo locutor com algum áudio em português foi para o teste (regra 1 do protocolo); o RADAR do treino é chinês, chinês de Taiwan e japonês.
- **No português de audiolivro (BRSpeech-DF), cerca de {pct(tm.loc["brspeech_df", "FPR M"], 0)} dos reais são acusados** com o corte M; **no português falado (CORAA), cerca de {pct(tm.loc["coraa", "FPR M"], 0)}**.
- **O corte não se transfere entre ambientes:** com o mesmo corte M, o falso alarme vai de {pct(tm["FPR M"].min())} a {pct(tm["FPR M"].max())} conforme o banco.
- **Nada foi testado em ligações, WhatsApp nem celular.**
- **Privacidade (recomendação, não resultado):** processar o áudio no próprio aparelho sempre que der e **não guardar o áudio sem consentimento**.
'''),
            md(r'''**O que faz:** mostra os cortes da T3 e o TPR/FPR do corte M por tipo de áudio, como estão em `results/`. **Por quê:** os números do guia vêm destes arquivos.'''),
            code(r'''
t3m = ler(R / "v7_t3_modelo.json")
display(Markdown("**Cortes da validação (logit):** " + ", ".join(f"{k} = {v:.3f}" for k, v in t3m["cortes_da_validacao_logit"].items())
                 + f"; regra marcada pelo protocolo: {t3m['regra_de_corte_escolhida_no_protocolo']}."))
tm = pd.read_csv(R / "v7_teste_mestre.csv")
display(tm[tm.receita == "T3"].set_index("recorte")[["papel", "n_real", "n_falso", "TPR M", "FPR M"]].round(3))
''')]


GLOSSARIO_NOVO = r'''| **T3** | o nome da **receita de treino** vencedora da v7 (estúdio + internet real + RADAR). O nome vem só da ordem da tabela (T0, T1, T2, T3, T4). Parte A. |
| **Receita vencedora** | a receita escolhida **pela validação**, antes de abrir o teste. Na v7, a T3. |
| **Sistema principal** | o modelo de fala da Meta (XLS-R 300M) **cortado e congelado** + a **regressão treinada com a receita T3**. É o que um site usaria. |
| **Nota (logit)** | o número que a regressão dá a cada áudio: soma ponderada dos 3.072 números padronizados + intercepto. Pode ser negativa; **quanto maior, mais parece IA**. Não é uma probabilidade calibrada. |
| **Corte (v7)** | o valor da nota acima do qual o áudio é chamado de IA. A v7 calculou três regras na validação: **A'**, **M** (a escolhida) e **F5** (partes C e seção 15). A regra final é decisão em aberto. |
| **Compressão de modelo** | fazer o mesmo trabalho com um modelo menor ou mais rápido, sem treinar de novo (parte B). ⚠️ Fora da trilha. |
| **Variantes leves** (V0 a V3) | **V0** = modelo completo; **V1** = cortado depois da 7ª camada (fp32); **V2** = V1 em meia precisão (fp16); **variante leve V3 (int8)** = V1 com inteiros de 8 bits. ⚠️ Fora da trilha. |
| **fp32 / fp16 / int8** | quantos bytes cada número ocupa: 4 (precisão normal), 2 (meia precisão) e 1 (inteiro de 8 bits). Menos bytes = menor e mais rápido, mas a conta pode mudar um pouco. |
| **⚠️ "V3" ≠ "v3"** | a **variante leve V3 (int8)** do modelo leve **não é** a **v3 do projeto** (a versão do preenchimento de áudios curtos). |
'''


# ------------------------------------------------------------------------------------------------ montagem
def montar():
    n = Nums()
    C = v7nb.montar()                                   # seções 1 a 16 da v7
    C[0] = intro(n, C[0].source)
    # rótulos: "sistema atual" (v7) -> "sistema principal"
    for c in C:
        c.source = (c.source.replace("Sistema atual", "Sistema principal").replace("sistema atual", "sistema principal")
                    .replace("e do atual (T3)", "e do principal (T3)").replace("e no atual, v7 (T3)", "e no principal, v7 (T3)"))
    # glossário: XLS-R congelado só no sistema principal; liberado em parte só no experimento v8
    trocar(C, "| **XLS-R** | um modelo de fala pronto (da Meta), treinado sem rótulos em fala de 128 línguas. Usamos congelado: ele só transforma o áudio em números. |",
           "| **XLS-R** | um modelo de fala pronto (da Meta), treinado sem rótulos em fala de 128 línguas. No sistema principal (v7, T3) ele fica **congelado**: só transforma o áudio em números. Ele é **parcialmente liberado (blocos 5 a 7) só no experimento v8** (anexo). |")
    trocar(C, "| **Sistema principal / anterior** | **atual = v7 (T3)**; **anterior = v4 (F0)**. Mesmo pipeline e mesma regressão; muda o treino. |",
           "| **Sistema principal / anterior / experimento** | **principal = v7 (T3)**; **anterior = v4 (F0)**; **experimento = v8 (ajuste fino), não adotado** (anexo). A v4 e a v7 têm o mesmo pipeline e a mesma regressão; muda o treino. |\n"
           "| **Ajuste fino (fine-tuning)** | deixar parte do modelo pronto voltar a aprender com os nossos dados. Só no experimento v8 (anexo). ⚠️ Fora da trilha. |\n"
           "| **Semente (seed)** | o número que fixa os sorteios de um treino. Treinar de novo com outra semente mostra quanto o resultado muda por acaso. |")
    # configuração (célula única "config"): diz qual é o sistema principal e de onde se lê o anexo
    ic = [i for i, c in enumerate(C) if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(ic) == 1
    assert "AVALIAR_TESTE_FINAL = False" in C[ic[0]].source
    C[ic[0]].source += ('\n\n# ---------------- sistema principal = v7 (T3); anexo: v8 (ajuste fino), não adotado ----------------\n'
                        '# Só leitura. Decisão: docs/DECISAO_SISTEMA_PRINCIPAL.md. Protocolo da v8: docs/PROTOCOLO_V8.md.\n'
                        'SISTEMA_PRINCIPAL = "v7 (T3)"\n'
                        'EXPERIMENTO_NAO_ADOTADO = "v8 (R1, ajuste fino)"\n'
                        'ARQ_V8 = RESULTS_DIR / "v8_resumo.json"\n'
                        'print(f"sistema principal: {SISTEMA_PRINCIPAL} | anexo: {EXPERIMENTO_NAO_ADOTADO}")')
    # mesma proteção do nb_v8: results/v4_camadas_concluido.json pode faltar (fica fora do git)
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
    C[ia] = code(C[ia].source + "\n" + v8nb.AJUDA_V8.strip("\n"))
    # seção 9: a v8 na tabela do caminho, como experimento não adotado
    trocar(C, "## 9. O caminho até aqui (v1 a v7)", "## 9. O caminho até aqui (v1 a v8)")
    v8 = n.v8
    trocar(C, "| **sistema principal = T3**, escolhida pela validação |",
           "| **sistema principal = T3**, escolhida pela validação |\n"
           f"| v8 | ajuste fino das camadas 5, 6 e 7 do XLS-R, começando da T3 (anexo) | média no teste mestre {br(v8.p('R0', 'média AUC (3 fontes)'))} → {br(v8.p('R1 (média das seeds)', 'média AUC (3 fontes)'))} (R1) | experimento **não adotado**; mantivemos a T3 |")

    # ---------------- anexo: seções 17 a 22 da v8, com as correções de texto
    A = (v8nb.secao17() + v8nb.secao18(v8) + v8nb.secao19(v8) + v8nb.secao20(v8)
         + v8nb.secao21(v8) + v8nb.secao22(v8))
    trocar(A, "Treinar pouco cabe na placa de vídeo de 6 GB e reduz",
           "Treinar pouco cabe na memória de uma placa de vídeo pequena e reduz")
    trocar(A, "numa placa de vídeo de notebook com 6 GB, e cada", "na máquina usada, e cada")
    trocar(A, "O ganho é pequeno (~0,004 a 0,005 de AUC), mas não é ruído.",
           "O ganho é pequeno (~0,004 a 0,005 de AUC): ele passa do ruído de sortear o teste, mas com 2 sementes não dá para afirmar que passa do ruído do treino.")
    trocar(A, "Real (IC pareado acima de zero), mas pequeno perto do salto da v7 (+0,20).",
           "O IC pareado fica acima de zero, ou seja, o ganho passa do ruído de sortear o teste, mas com 2 sementes não dá para afirmar que passa do ruído do treino. E é pequeno perto do salto da v7 (+0,20).")
    trocar(A, "baixe da release [`v8-pesos-r1`](https://github.com/FredericoAyres/projeto_final_AI/releases/tag/v8-pesos-r1) do GitHub para `models/`",
           "baixe da release `v8-pesos-r1` do repositório do projeto no GitHub para `models/`")
    trocar(A, "## 22. v8: o que não dá para afirmar e próximos passos", "## 22. v8: o que não dá para afirmar")
    trocar(A, "**Os pesos da R1, para usar o modelo**", "**Os pesos da R1, para quem quiser repetir o experimento**")
    trocar_re(A, r"\*\*Decisão em aberto \(não tomada aqui\):\*\*.*?registrado antes de qualquer novo teste\.",
              "**Decisão (15/11/2026, posterior ao protocolo):** nenhum modelo da R1 foi empacotado. **Mantivemos a v7 como sistema principal**; os motivos estão na abertura deste anexo e em `docs/DECISAO_SISTEMA_PRINCIPAL.md`.")
    trocar_re(A, r"\*\*Próximos passos\*\* \(somam-se aos da seção 16\):.*?avaliados uma vez depois de fechar o sistema\.",
              "**Próximos passos:** ver a última seção deste notebook.")
    # títulos do anexo descem um nível (o anexo é o "capítulo"); figuras idem
    for c in A:
        if c.cell_type == "markdown":
            c.source = re.sub(r"^### ", "#### ", c.source, flags=re.M)
            c.source = re.sub(r"^## (\d+)\. v8:", r"## \1. Anexo · v8:", c.source, flags=re.M)
    ig = v7nb.indice(C, "### Glossário em linguagem simples")
    gs = C[ig].source
    gs = gs.replace("regressão mais simples. || **Sistema principal", "regressão mais simples. |\n| **Sistema principal")
    assert "|| " not in gs
    C[ig].source = gs.rstrip("\n") + "\n" + GLOSSARIO_NOVO
    L = NumsLeve()
    C += bloco_a(n, L) + bloco_b(n, L) + bloco_c(n, L)
    C += [anexo_abertura(n)] + A + [proximos_finais()]
    return C


def textos(c):
    out = [c.source]
    for o in c.get("outputs", []):
        t = o.get("text", "")
        out.append("".join(t) if isinstance(t, list) else str(t))
        for k in ("text/plain", "text/html", "text/markdown"):
            v = o.get("data", {}).get(k, "")
            out.append("".join(v) if isinstance(v, list) else str(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-executar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for p in PRESERVADOS:
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
        for t in textos(c):
            for p in PROIBIDOS:
                if re.search(rf"(?<!\w){re.escape(p)}", t, flags=re.I):
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
