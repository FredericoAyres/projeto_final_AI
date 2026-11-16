#!/usr/bin/env python3
"""Monta e executa o notebook da ENTREGA FINAL: o do sistema principal (v7 = T3, `scripts/nb_principal.py`) mais o que
foi feito depois da v7 (teste externo WhatsApp, teste de estresse, módulo de contexto de golpe, errata, sistema combinado,
limitações). Só LÊ de results/ e docs/: não treina, não baixa, não chama API.

- Parte de `nb_principal.montar()` (que reaproveita `nb_v7.py` e `nb_v8.py`, sem alterá-los e sem chamar o `main` deles).
- Todos os números dos textos novos são LIDOS de results/*.csv|json na hora de montar (nada digitado de memória).
- Grava notebook.ipynb e copia para notebooks/versions/principal_entrega.ipynb (só se: 0 erros, < 10 MB e nenhum nome de
  pessoa ou de máquina no texto e nas saídas). NÃO mexe em principal_v7.ipynb, v4_final, v7_final, v8_finetuning.

    python scripts/nb_entrega.py [--sem-executar]
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
import nb_principal as P  # noqa: E402  (só as funções e o montar())

import nbformat as nbf  # noqa: E402
import pandas as pd  # noqa: E402

v7nb = P.v7nb
md, code, br, pct, mil = v7nb.md, v7nb.code, v7nb.br, v7nb.pct, v7nb.mil
R = Path("results")
LIMITE_MB = 10.0
ARQUIVO = "notebooks/versions/principal_entrega.ipynb"
PRESERVADOS = ("notebooks/versions/v4_final.ipynb", "notebooks/versions/v7_final.ipynb",
               "notebooks/versions/v8_finetuning.ipynb", "notebooks/versions/principal_v7.ipynb")
# lista da tarefa (palavra inteira) + nomes de máquina; além da lista que o nb_principal já usa
def sem_acento_txt(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def _carregar_nomes_locais():
    """Nomes lidos em tempo de execução de nome_local (arquivo local, fora do git); minúsculas e sem acento."""
    arq = Path("data/teste_whatsapp/extraido/coleta/manifesto_geral.csv")
    if not arq.exists():
        print("AVISO: manifesto local da coleta não encontrado; checagem de nomes de pessoas pulada.")
        return []
    col = pd.read_csv(arq, encoding="utf-8-sig")["nome_local"].dropna().astype(str)
    return sorted({sem_acento_txt(t) for v in col for t in re.split(r"[\s_.\-]+", v) if len(t) >= 2 and not t.isdigit()})


NOMES_TAREFA = _carregar_nomes_locais() + ["RTX", "GPU"]


def bloco_fe(nome, titulo, mostra, olhar, concluir):
    """Figura do catálogo da entrega (results/figuras_entrega/slides/fe_*.png), gerada por scripts/figuras_entrega.py."""
    cod = (f'p = R / "figuras_entrega" / "slides" / "{nome}.png"\n'
           'if existe(p):\n    display(Image(data=_png_reduzido(p)))\nelse:\n'
           '    display(Markdown(f"> **Não encontrei** `{p}`. Rode `python scripts/figuras_entrega.py`."))')
    return [md(f"### {titulo}\n\n**O que ela mostra:** {mostra}\n\n*O código abaixo só exibe a figura gerada por `scripts/figuras_entrega.py` "
               f"(versões para slide e para paper em `results/figuras_entrega/`).*"),
            code(cod),
            md(f"**O que olhar:** {olhar}\n\n**O que concluir:** {concluir}")]


def ic(x, col, d=1):
    """'98,0% (93,6 a 100,0)' a partir de uma linha com col, col_lo, col_hi (frações)."""
    return f"{pct(x[col], d)} ({br(100 * x[col + '_lo'], d)} a {br(100 * x[col + '_hi'], d)})"


# ------------------------------------------------------------------------------------------------ números (lidos)
class NE:
    def __init__(self):
        self.det = pd.read_csv(R / "teste_whatsapp_detector.csv")
        self.grp = pd.read_csv(R / "teste_whatsapp_grupos.csv")
        self.par = pd.read_csv(R / "teste_whatsapp_pareado.csv")
        self.ata = pd.read_csv(R / "teste_whatsapp_atalho.csv").set_index("metadados")
        self.notas = pd.read_csv(R / "teste_whatsapp_notas.csv")
        self.est = pd.read_csv(R / "teste_3_ia_notas.csv")
        self.dmet = pd.read_csv(R / "diagnostico_whisper_idioma_metricas.csv")
        self.dgrp = pd.read_csv(R / "diagnostico_whisper_idioma_grupos.csv")
        self.dcomb = pd.read_csv(R / "diagnostico_whisper_combinado.csv")
        self.didi = pd.read_csv(R / "diagnostico_whisper_idioma_idioma.csv")
        self.frase = pd.read_csv(R / "analise_erro_contexto_por_frase.csv")
        self.aval = json.loads((R / "contexto_golpe_avaliacao_resumo.json").read_text(encoding="utf-8"))
        self.t3m = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))
        n = self.notas
        self.n_real = int((n.autenticidade == "real").sum())
        self.n_ia = int((n.autenticidade == "ia").sum())
        self.n_pes = int(n.pessoa_id.nunique())
        self.M = float(self.d("T3", "M", "A").corte_logit)
        fa = n[(n.autenticidade == "real") & (n.z_T3_A >= self.M)]
        self.fa_n, self.fa_pessoas = len(fa), sorted(fa.pessoa_id.unique())
        self.gr = n.grupo.value_counts().to_dict()
        self.wer = pd.read_csv(R / "diagnostico_whisper_idioma_dev_wer.csv").set_index("modo")
        self.div = pd.read_csv(R / "diagnostico_whisper_divisao.csv")
        self.eng = int(self.didi.english.sum())
        self.n_clipes = int(self.didi.n.sum())
        self.n_dev = int(self.didi[self.didi.conjunto.str.startswith("DEV")].n.iloc[0])
        self.n_pes_dev = int((self.div.grupo == "DEV").sum())
        self.n_pes_res = int((self.div.grupo == "RESERVADO").sum())
        assert (self.n_real, self.n_ia, self.n_pes) == (44, 56, 23) and self.eng == 96 and self.n_clipes == 100
        assert self.fa_n == 2 and len(self.fa_pessoas) == 1

    def d(self, modelo, regra, cond):
        x = self.det[(self.det.modelo == modelo) & (self.det.corte_regra == regra) & (self.det.condicao == cond)]
        assert len(x) == 1, (modelo, regra, cond)
        return x.iloc[0]

    def g(self, modelo, regra, cond, grupo):
        x = self.grp[(self.grp.modelo == modelo) & (self.grp.corte_regra == regra) & (self.grp.condicao == cond)
                     & (self.grp.grupo == grupo)]
        return x.iloc[0]

    def p(self, modelo, cond):
        return self.par[(self.par.modelo == modelo) & (self.par.condicao == cond)].iloc[0]

    def dm(self, conj, versao, recorte="todos"):
        x = self.dmet[(self.dmet.conjunto.str.startswith(conj)) & (self.dmet.versao.str.startswith(versao))
                      & (self.dmet.recorte == recorte)]
        assert len(x) == 1, (conj, versao)
        return x.iloc[0]

    def ctx100(self, sufixo=""):
        """Contexto nos 100 clipes (descritivo), a partir de diagnostico_whisper_combinado.csv."""
        t = self.dcomb[(self.dcomb.tabela == "contexto_por_grupo") & (self.dcomb.conjunto == "todos")].set_index("grupo")
        col = "alerta_medio_mais" + sufixo
        pos = ["real_com_golpe", "ia_pedido_dinheiro"]
        neg = ["real_sem_golpe", "ia_neutro"]
        npos, nneg = int(t.loc[pos, "n"].sum()), int(t.loc[neg, "n"].sum())
        tp, fp = int(t.loc[pos, col].sum()), int(t.loc[neg, col].sum())
        return dict(n=npos + nneg, npos=npos, nneg=nneg, tp=tp, fp=fp, acc=(tp + nneg - fp) / (npos + nneg),
                    recall=tp / npos, fa=fp / nneg, trivial=npos / (npos + nneg), t=t)

    def comb(self, conj="todos"):
        t = self.dcomb[(self.dcomb.tabela == "combinado_grupo_x_nivel") & (self.dcomb.conjunto == conj)]
        return t.set_index("grupo")

    def frases(self):
        f = self.frase
        claras = f[(f.tema == "golpe") & (f.recall == 1.0)]
        leves = f[(f.tema == "golpe") & (f.recall == 0.0)]
        cot = f[f.tema == "cotidiano"]
        out = {k: (len(x), int(x.n_clipes.sum()), int(x.alerta_medio_mais.sum()), int(x.alerta_so_alto.sum()))
               for k, x in (("claras", claras), ("leves", leves), ("cotidiano", cot))}
        assert out["claras"][1] == out["claras"][2] and out["leves"][2] == 0 and out["cotidiano"][2] == 0
        return out


# ------------------------------------------------------------------------------------------------ textos novos
def parte_d(e):
    t3a, t3b = e.d("T3", "M", "A"), e.d("T3", "M", "B")
    f0a, f0ab = e.d("F0", "A", "A"), e.d("F0", "A", "B")
    f0ba = e.d("F0", "B", "A")
    lin = []
    for rot, x in (("**T3, corte M** (o do sistema principal)", t3a), ("T3, corte A'", e.d("T3", "A'", "A")),
                   ("T3, corte F5", e.d("T3", "F5", "A")), ("T3, corte M, **Opus** (condição B)", t3b),
                   ("F0, corte A (comparação)", f0a), ("F0, corte B (comparação)", f0ba),
                   ("F0, corte A, Opus (condição B)", f0ab)):
        lin.append(f"| {rot} | {ic(x, 'acc')} | {pct(x['bacc'])} | {ic(x, 'tpr')} | {ic(x, 'fpr')} | {br(x['auc'])} |")
    G = lambda m, r, c, k: e.g(m, r, c, k)  # noqa: E731
    lg = []
    for k, rot in (("real_sem_golpe", "real, conteúdo normal"), ("real_com_golpe", "real, conteúdo de golpe (encenado)"),
                   ("ia_pedido_dinheiro", "IA, pede dinheiro"), ("ia_neutro", "IA, conteúdo neutro")):
        a, b, f = G("T3", "M", "A", k), G("T3", "M", "B", k), G("F0", "A", "A", k)
        lg.append(f"| {rot} | {int(a.n)} | {int(a.acusados_como_ia)} ({pct(a.taxa)}) | {int(b.acusados_como_ia)} ({pct(b.taxa)}) "
                  f"| {int(f.acusados_como_ia)} ({pct(f.taxa)}) |")
    p3a, p3b, p0a = e.p("T3", "A"), e.p("T3", "B"), e.p("F0", "A")
    ata_all, ata_sem = e.ata.loc["todos os metadados"], e.ata.loc["sem a taxa original"]
    ata_dur, ata_ban = e.ata.loc["so duracao"], e.ata.loc["so banda95"]
    return [md(rf'''
---
# Depois da v7: o que foi feito para a entrega final

As partes **D a I** abaixo contam o que veio **depois** da v7: um teste externo com áudios de WhatsApp, um teste de estresse informal, o módulo de contexto de golpe, uma **correção (errata)** que fizemos num resultado e o sistema que junta voz e conteúdo. Todos os números são lidos de `results/` (nada foi treinado, baixado ou consultado para montar este texto).

## D. Teste externo "WhatsApp" (detector de voz)

**O que é o banco.** {e.n_real + e.n_ia} áudios de **{e.n_pes} pessoas**, **todas** com áudio real e áudio de IA: **{e.n_real} reais** (gravações de WhatsApp convertidas para WAV, 48 kHz) e **{e.n_ia} de IA** (clones das mesmas pessoas feitos com um **único** gerador, o Chatterbox, 24 kHz). Tudo em **inglês** (informação da coleta; o Whisper em modo automático classificou {e.eng} dos {e.n_clipes} clipes como inglês, ver a parte G). Os grupos são: {e.gr["real_sem_golpe"]} reais de conversa normal, {e.gr["real_com_golpe"]} reais "com golpe" (gravações encenadas, **não** fraude real), {e.gr["ia_pedido_dinheiro"]} clones que pedem dinheiro e {e.gr["ia_neutro"]} clones neutros. Acertar "tudo é IA" daria só **{pct(e.n_ia / (e.n_real + e.n_ia), 0)}** (a **acurácia trivial**: o que se acerta chamando tudo de IA, a classe mais comum).

**Como foi feito (regras do jogo).** O **protocolo foi escrito antes** de qualquer nota (`docs/PROTOCOLO_TESTE_WHATSAPP.md`, com os cortes e os arquivos carimbados por SHA256), **nada foi treinado, reajustado ou trocado**, e o teste **rodou uma vez** (marcador `results/teste_whatsapp_concluido.json`). O sistema usado é o principal, **sem nenhuma mudança**: os cortes **M**, **A'** e **F5** são os da validação da v7 (o principal é o **M = {br(e.M)} logit**). O **F0** (sistema anterior) entra só para comparar.

**Termos.** *Acurácia* = % de acertos. *Acurácia balanceada* = média do acerto em reais e em IA (não engana quando há mais de uma classe). *TPR* (ou recall) = % das vozes de IA detectadas. *FPR* (falso alarme) = % das vozes reais acusadas como IA. *AUC* = capacidade de pôr as notas de IA acima as dos reais, sem depender de corte (1,0 = perfeito, 0,5 = acaso). *IC95* = faixa plausível do número; aqui por **bootstrap por pessoa** (sorteamos as {e.n_pes} pessoas com reposição, 2.000 vezes), porque os áudios da mesma pessoa não são independentes. *Opus* = o formato de áudio comprimido usado em chamadas e aplicativos de mensagem.

| modelo e corte | acurácia (IC95) | acurácia balanceada | TPR (IC95) | FPR (IC95) | AUC |
|:--|:--|--:|:--|:--|--:|
{chr(10).join(lin)}

**Condição Opus (B).** Os {e.n_ia} clones foram recodificados para **Opus mono de ~16 kbps** (parecido com um áudio de WhatsApp); os reais não mudaram. A T3 foi de {pct(t3a["acc"], 0)} para {pct(t3b["acc"], 0)} de acurácia, e a AUC de {br(t3a["auc"])} para {br(t3b["auc"])}: **o resultado não depende de o clone estar em 24 kHz ou em Opus**. Essa condição **não** responde a outras diferenças entre reais e clones (conteúdo, duração, silêncio).

**Por grupo** (quantos áudios foram chamados de IA; condição A, e Opus na T3):

| grupo | n | T3 (corte M) | T3 (M), clones em Opus | F0 (corte A) |
|:--|--:|--:|--:|--:|
{chr(10).join(lg)}

**Por pessoa.** Para cada pessoa, a nota média dos clones menos a dos reais: na T3, **+{br(p3a.dif_media, 1)} logit** (IC95 {br(p3a.dif_lo, 1)} a {br(p3a.dif_hi, 1)}) e **{int(p3a.ia_maior_que_real)} de {int(p3a.pessoas)} pessoas** com a IA acima do real (a menor diferença é +{br(p3a.dif_min, 1)}); no F0, +{br(p0a.dif_media, 1)} e {int(p0a.ia_maior_que_real)} de {int(p0a.pessoas)} (menor diferença +{br(p0a.dif_min, 2)}). Com Opus, a T3 segue em {int(p3b.ia_maior_que_real)} de {int(p3b.pessoas)}. As **{e.fa_n} falsas acusações** (reais chamados de IA) são **da mesma pessoa** ({e.fa_pessoas[0]}).

**Atalho por metadados (leia o 98% com cuidado).** Treinamos um classificador que **não ouve a voz**: só vê duração, nível de volume, fração de silêncio, taxa de amostragem original e largura de banda (deixando uma pessoa de fora a cada vez). Com a taxa original, ele acerta **tudo** (AUC {br(ata_all.AUC_loo)}), porque os reais são 48 kHz e os clones 24 kHz. **Sem a taxa**, a AUC ainda é **{br(ata_sem.AUC_loo)}** (IC95 {br(ata_sem.auc_lo)} a {br(ata_sem.auc_hi)}); só a duração dá {br(ata_dur.AUC_loo)} e só a largura de banda, {br(ata_ban.AUC_loo)}. **Ressalva:** a AUC do detector ({br(e.d("T3", "M", "A").auc)}) **não é distinguível** de um atalho de formato ou de nível: com estes dados **não dá para afirmar que o T3 "ouve a voz" e não o canal**. A condição Opus enfraquece o atalho da taxa de amostragem, mas não elimina os de nível e silêncio.

**Resumo honesto:** neste banco (um gerador, {e.n_pes} pessoas, inglês), o sistema principal separou real de IA muito bem ({pct(t3a["acc"], 0)} de acurácia contra {pct(t3a["acc_trivial"], 0)} do trivial), **mas o teste não consegue descontar o atalho de formato**.
'''),
            md(r'''**O que faz:** mostra as tabelas do teste WhatsApp como estão em `results/` (detector, grupos, por pessoa e atalho). **Por quê:** os números do texto acima vêm destes arquivos.'''),
            code(r'''
det = pd.read_csv(R / "teste_whatsapp_detector.csv")
display(det[["modelo", "corte_regra", "condicao", "acc", "bacc", "tpr", "fpr", "auc", "n", "acc_trivial"]].round(3))
grp = pd.read_csv(R / "teste_whatsapp_grupos.csv")
display(grp[(grp.corte_regra.isin(["M", "A"])) & (grp.condicao == "A")][["modelo", "corte_regra", "grupo", "n", "acusados_como_ia", "taxa"]].round(3))
display(pd.read_csv(R / "teste_whatsapp_pareado.csv").round(3))
display(pd.read_csv(R / "teste_whatsapp_atalho.csv").round(3))
'''),
            *v7nb.bloco_slide("tw_03_histograma_notas", "Figura tw_03 · As notas de reais e de IA no teste WhatsApp",
                              "o histograma da nota (logit) dos reais e dos clones, na T3, com os cortes.",
                              "as duas montanhas quase não se tocam; os dois reais acima do corte são da mesma pessoa.",
                              "a separação é grande neste banco, mas ela pode vir também de diferenças de formato (atalho acima).",
                              origem="scripts/teste_whatsapp_figuras.py"),
            *bloco_fe("fe_08_whatsapp_detector", "Figura fe_08 · Acurácia e erros no WhatsApp (T3 × F0)",
                      "a acurácia e a acurácia balanceada (esquerda) e a % de IA detectada e de reais acusados (direita), com IC95 por pessoa, para a T3 (corte M) e o F0 (cortes A e B).",
                      "a T3 fica acima do F0 nos dois cortes e acima da linha da acurácia trivial; os intervalos do F0 são largos.",
                      "no WhatsApp deste teste (um gerador, inglês), a T3 foi o melhor dos dois; o atalho de formato (fe_11) ainda não foi descontado."),
            *bloco_fe("fe_10_canal_opus", "Figura fe_10 · Condição Opus (clones recodificados)",
                      "a acurácia e a AUC com os clones como vieram (A) e recodificados em Opus a ~16 kbps (B), na T3 e no F0.",
                      "as barras da T3 quase não mudam entre A e B.",
                      "o resultado da T3 não depende de o clone estar em 24 kHz ou em Opus; outras diferenças entre reais e clones continuam sem controle."),
            *bloco_fe("fe_11_atalho_metadados", "Figura fe_11 · O atalho por metadados",
                      "a AUC do detector (T3 e F0) ao lado da de um classificador que só vê metadados do arquivo (com e sem a taxa original), com IC95.",
                      "os metadados chegam perto do detector, mesmo sem a taxa de amostragem.",
                      "com estes dados não dá para afirmar que o T3 ouve a voz e não o formato do áudio.")]


def parte_e(e):
    s = e.est
    o = s[s.condicao == "original"].set_index("id")
    op = s[s.condicao == "opus16k"].set_index("id")
    det = [i for i in o.index if bool(o.loc[i, "passa_M"])]
    queda = [o.loc[i, "nota_T3"] - op.loc[i, "nota_T3"] for i in o.index]
    assert len(o) == 3 and len(det) == 1 and det == ["ia_3"]
    lin = []
    for i in o.index:
        for rot, x in (("original", o.loc[i]), ("Opus 16 kbps", op.loc[i])):
            lin.append(f"| {i} | {rot} | {br(x.nota_T3, 2)} | {br(x.prob_pct, 1)}% | {'sim' if x['passa_M'] else 'não'} "
                       f"| {'sim' if x['F0_passa_A'] else 'não'} / {'sim' if x['F0_passa_B'] else 'não'} |")
    return [md(rf'''
## E. Teste de estresse informal (3 áudios de um gerador novo)

> **Isto NÃO é um teste oficial.** Não teve protocolo escrito antes, tem **só 3 áudios** e **não permite estimar taxa de acerto**. Serve para ver se o sistema escapa de um gerador de **família diferente** da que conhecemos.

**Origem (como informada).** Três áudios de uma demonstração pública do **Dessa** (Block, versão de pesquisa): um modelo **fala-para-fala**, que conversa em tempo real, treinado só em fala. **Não sabemos** qual trecho ou versão gerou os áudios, nem se o site comprimiu o áudio. Passaram pelo **mesmo caminho do sistema principal**, sem mudar conta nem corte (cortes da T3: A' {br(e.t3m["cortes_da_validacao_logit"]["A'"])}, **M {br(e.t3m["cortes_da_validacao_logit"]["M"])}**, F5 {br(e.t3m["cortes_da_validacao_logit"]["F5"])}). Os 3 áudios ficam **reservados**: não entram em treino, validação nem escolha de corte.

| áudio | versão | nota T3 (logit) | "probabilidade" crua | T3 acusa (corte M)? | F0 acusa (corte A / B)? |
|:--|:--|--:|--:|:--|:--|
{chr(10).join(lin)}

**Resultado: a T3 detectou 1 de 3** ({det[0]}). Os outros dois (ia_1 e ia_2) passaram como se fossem reais. Com Opus a 16 kbps, as notas **caem** de {br(min(queda), 1)} a {br(max(queda), 1)} logit; o {det[0]} continua acusado. Na comparação, o F0 só deixa de acusar o ia_1; o ia_2 só é acusado no corte A, e o ia_3 nos dois. A nota ia_2 original é maior que a de {br(o.loc["ia_2", "pct_reais_abaixo"], 1)}% dos {int(o.loc["ia_2", "n_reais"])} reais do teste WhatsApp (o ia_1 fica acima de {br(o.loc["ia_1", "pct_reais_abaixo"], 1)}%, mas abaixo do corte).

**Leitura honesta.** Um gerador de **família nova escapou** em 2 de 3 áudios: o detector **não generaliza com garantia para geradores novos**. A "probabilidade" da tabela é só a conta crua 1/(1 + e^(−nota)), **sem calibração**; a escala das notas muda entre ambientes. Com 3 áudios não dá para dizer qual a taxa real de detecção. **Próximo passo sugerido:** repetir com 15 a 30 áudios de vários geradores, sem treinar, com protocolo escrito antes.
'''),
            md(r'''**O que faz:** mostra a tabela do teste de estresse como está gravada. **Por quê:** os números acima vêm dela.'''),
            code(r'''
est = pd.read_csv(R / "teste_3_ia_notas.csv")
display(est[["id", "condicao", "nota_T3", "prob_pct", "passa_A'", "passa_M", "passa_F5", "pct_reais_abaixo", "nota_F0", "F0_passa_A", "F0_passa_B"]].round(2))
'''),
            *bloco_fe("fe_12_estresse_3_ia", "Figura fe_12 · Onde caem os 3 áudios do teste de estresse",
                      "a nota da T3 dos reais e dos clones do WhatsApp (pontos) e dos 3 áudios do teste de estresse (losangos), com os três cortes.",
                      "dois dos três losangos ficam entre os reais, à esquerda do corte; só o ia_3 passa do corte.",
                      "um gerador de família nova escapou em 2 de 3 áudios; é um teste informal, sem taxa de acerto.")]


def parte_f(e):
    a = e.aval
    p = a["positivo_medio_ou_alto"]
    pa = a["positivo_so_alto"]
    inj = a["injecao"]
    con = a["consistencia"]
    metas = a["metas_declaradas"]
    pg = p["por_grupo"]
    rec, fa, ac = p["recall_golpes_todos"], p["falso_alarme_legitimos_todos"], p["acuracia_nao_ambiguos"]
    fmt = lambda x, d=1: f"{pct(x['taxa'], d)} ({br(100 * x['ic95'][0], 0)} a {br(100 * x['ic95'][1], 0)})"  # noqa: E731
    return [md(rf'''
## F. Módulo de contexto de golpe (protótipo)

**A ideia.** O detector de voz responde *"a voz parece de IA?"*. Este módulo responde outra pergunta: ***"o que foi dito parece golpe?"***. Ele **não treina nada**. ⚠️ Fora da trilha: transcrição de fala (Whisper) e classificação de texto por instruções a um modelo de linguagem (**LLM**, um modelo que lê e escreve texto), sem exemplos de treino. Código: `src/contexto_golpe.py`; documentação: `docs/CONTEXTO_GOLPE.md`.

**Como funciona (4 passos):** (1) o áudio é preparado (mono, 16 kHz, corta o silêncio das bordas); (2) o **Whisper** (`whisper-large-v3-turbo`, pela API do Groq) **transcreve** a fala em texto, descartando trechos com cara de ruído; (3) um LLM (`openai/gpt-oss-120b`, também pelo Groq) lê o texto, **como dado não confiável** (ordens escritas dentro do texto não são obedecidas), e devolve um **nível de risco**, os **sinais** que viu e a **evidência literal** (a frase exata do texto que justifica cada sinal); (4) o código **confere** a resposta: só aceita campos permitidos, **descarta evidência que não existe no texto**, e uma falha vira `indeterminado`, **nunca** "nenhum".

- **Níveis de risco:** `nenhum`, `baixo`, `medio`, `alto` (e `indeterminado` quando algo falha). **Alerta = médio ou alto.**
- **Sinais** (lista fechada): urgência, pedido de dinheiro ou Pix, senha ou código, link suspeito, falsa identidade, pedido de segredo, ameaça ou medo, pagamento incomum, troca de número ou contato, tentativa de manipulação.
- **Aviso fixo:** *isto é um alerta, não um veredito.*
- **É um protótipo:** o áudio e o texto saem da máquina para um serviço externo (Groq); **nunca** enviamos áudio dos bancos do projeto; a chave só vem de uma variável de ambiente.

**Avaliação com textos inventados** (`docs/PROTOCOLO_CONTEXTO.md` e `docs/AVALIACAO_CONTEXTO.md`). Escrevemos **{a["n_textos"]} textos** à mão (golpes explícitos e escondidos, conversas comuns, "legítimos difíceis" que falam de banco ou golpe sem ser golpe, textos com ordens escondidas para o avaliador, e casos ambíguos), com dados fictícios. O **prompt** (as instruções ao LLM) foi ajustado **só num conjunto de desenvolvimento** (3 versões avaliadas; ficou a revisão r1); depois o **protocolo foi congelado** (hash do prompt, do código e dos textos de teste, às {a["congelado"]["congelado_em"][11:16]} de {a["congelado"]["congelado_em"][8:10]}/{a["congelado"]["congelado_em"][5:7]}/{a["congelado"]["congelado_em"][:4]}) e o teste rodou **uma vez**.

| medida | resultado (IC95) | meta declarada antes |
|:--|:--|:--|
| **recall dos golpes** (alerta médio+) | **{fmt(rec)}** | ≥ {pct(metas["recall_golpes_min"], 0)} |
| · golpes explícitos | {fmt(pg["golpe_explicito"]["recall"])} | |
| · golpes escondidos | {fmt(pg["golpe_escondido"]["recall"])} | |
| **falso alarme** nos legítimos | **{fmt(fa)}** | ≤ {pct(metas["falso_alarme_legitimos_max"], 0)} |
| · legítimos difíceis | {fmt(pg["legitimo_dificil"]["falso_alarme"])} | |
| acurácia (115 textos não ambíguos) | {fmt(ac)} | |
| se o alerta fosse só "alto": recall | {pct(pa["recall_golpes_todos"]["taxa"])} | |
| golpes com ordens escondidas ainda em alerta | {inj["golpes_ainda_pos"]["k"]} de {inj["golpes_ainda_pos"]["n"]} | |
| risco igual em 3 repetições | {con["risco_igual_nas_3"]["k"]} de {con["risco_igual_nas_3"]["n"]} textos | |

Em português simples: com **textos inventados**, o módulo quase não acusa conversa normal e pega a maior parte dos golpes de manual; deixa passar cerca de 1 em cada 5 golpes disfarçados ("salva meu número novo", pedidos indiretos). **Ressalvas:** os textos foram escritos por um modelo de linguagem (os golpes podem ser "de manual", diferentes de golpe real, com ruído); a amostra é pequena (o IC do recall vai de {br(100 * rec["ic95"][0], 0)}% a {br(100 * rec["ic95"][1], 0)}%); o ajuste do prompt foi feito olhando o desenvolvimento; só o texto foi avaliado (**o Whisper não entrou**); e só em **português do Brasil**. É um resultado de laboratório, **não de produto**.
'''),
            md(r'''**O que faz:** mostra o resumo da avaliação do contexto como está em `results/`. **Por quê:** os números acima vêm dele.'''),
            code(r'''
av = ler(R / "contexto_golpe_avaliacao_resumo.json")
p = av["positivo_medio_ou_alto"]
display(pd.DataFrame({k: {"k": v["k"], "n": v["n"], "taxa": round(v["taxa"], 3), "IC95 baixo": round(v["ic95"][0], 3), "IC95 alto": round(v["ic95"][1], 3)}
                      for k, v in p.items() if isinstance(v, dict) and "taxa" in v}).T)
display(Markdown(f"Metas declaradas antes: {av['metas_declaradas']}; textos: {av['n_textos']}; "
                 f"congelado em {av['congelado']['congelado_em']}."))
'''),
            *v7nb.bloco_slide("ctx_01_recall_e_falso_alarme", "Figura ctx_01 · Recall e falso alarme (textos inventados)",
                              "o recall (golpes com alerta) e o falso alarme (legítimos com alerta) do módulo, por tipo de texto, com IC95 e as metas.",
                              "o recall dos golpes explícitos é alto e o dos escondidos é o mais baixo; o falso alarme é pequeno.",
                              "nos textos inventados o módulo cumpre as metas, com IC largo.",
                              origem="scripts/contexto_golpe_figuras.py"),
            *v7nb.bloco_slide("ctx_02_matriz_de_confusao", "Figura ctx_02 · Em que nível cada tipo de texto caiu",
                              "quantos textos de cada tipo caíram em cada nível de risco.",
                              "golpes explícitos concentram-se em 'alto'; os escondidos, em 'médio'; os legítimos, em 'nenhum'.",
                              "o nível 'médio' é o que pega os golpes escondidos; sem ele, o recall cairia para cerca de metade.",
                              origem="scripts/contexto_golpe_figuras.py"),
            *v7nb.bloco_slide("ctx_03_injecao_e_consistencia", "Figura ctx_03 · Ordens escondidas e repetição",
                              "à esquerda, o que aconteceu com textos que trazem ordens ao avaliador; à direita, se o mesmo texto recebe o mesmo risco em 3 repetições.",
                              "as ordens escondidas **não derrubaram o alerta**, mas o módulo raramente percebe que são uma manipulação; as repetições concordaram.",
                              "o módulo resiste a essas ordens e é estável, mas não confie nele para **detectar** manipulação.",
                              origem="scripts/contexto_golpe_figuras.py")]


def parte_g(e):
    c15, cc = e.ctx100("_1511"), e.ctx100("")
    rv15, rvc = e.dm("RESERVADO", "15/11"), e.dm("RESERVADO", "corrigido")
    dv15, dvc = e.dm("DEV", "15/11"), e.dm("DEV", "corrigido")
    didi = e.didi.set_index("conjunto")
    eng = e.eng
    fr = e.frases()
    wer = e.wer
    assert abs(cc["acc"] - 0.50) < 1e-9 and round(cc["recall"], 2) == 0.37
    dg = e.dgrp
    lin = []
    for k, rot in (("real_sem_golpe", "real, conteúdo normal"), ("real_com_golpe", "real, conteúdo de golpe (encenado)"),
                   ("ia_pedido_dinheiro", "IA, pede dinheiro"), ("ia_neutro", "IA, neutro")):
        a = dg[(dg.conjunto.str.startswith("RESERVADO")) & (dg.versao.str.startswith("15/11")) & (dg.grupo == k)].iloc[0]
        b = dg[(dg.conjunto.str.startswith("RESERVADO")) & (dg.versao == "corrigido") & (dg.grupo == k)].iloc[0]
        lin.append(f"| {rot} | {int(a.n)} | {int(a.com_alerta)} | {int(b.com_alerta)} |")
    f_cl, f_le, f_co = fr["claras"], fr["leves"], fr["cotidiano"]
    x = dg[(dg.conjunto.str.startswith("RESERVADO")) & (dg.versao == "corrigido") & (dg.grupo == "real_com_golpe")].iloc[0]
    rcg_nenhum, rcg_n = int(x.risco_nenhum), int(x.n)
    return [md(rf'''
## G. O contexto no banco WhatsApp, e a errata

> ### ⚠️ ERRATA (16/11/2026)
> No teste de 15/11, **forçamos** o Whisper a transcrever em **português** (`idioma="pt"`), mas o áudio do banco é **inglês**. Os textos de 15/11 disseram o contrário ("o Whisper detectou português nos 100 clipes"; "o desvio foi usar `auto`"): **ambos estavam errados**, pois o idioma devolvido apenas repetia o que pedimos. **O resultado do contexto de 15/11 não vale como medida do módulo.** O **detector de voz (T3 e F0), os cortes, a condição Opus e o atalho de metadados não dependem do idioma do Whisper e não mudam.** Os arquivos antigos não foram editados; a lista de trechos afetados está em `docs/ERRATA_TESTE_WHATSAPP.md`.

**A correção.** Com o idioma em modo **automático** (`auto`, o Whisper detecta sozinho), o Whisper classificou **{eng} dos 100 clipes como inglês** (os outros {100 - eng}, como português; é o limite do modo automático em clipe curto ou com sotaque). Nos clones, a transcrição ficou quase perfeita: a **taxa de erro de palavras** (WER; 0 = perfeito) média foi **{br(wer.loc["auto", "wer_medio"])}** com `auto`, contra {br(wer.loc["pt", "wer_medio"])} com o português forçado. **Esses dois WER são do conjunto DEV** ({int(wer.loc["auto", "n_clones"])} clones de {e.n_pes_dev} pessoas), não dos 100 clipes (`docs/RESULTADO_CONTEXTO_WHISPER.md`).

**Protocolo da correção (DEV e RESERVADO).** Antes de qualquer chamada, o protocolo (`docs/PROTOCOLO_CONTEXTO_WHISPER.md`) dividiu as {e.n_pes} pessoas em **DEV** ({e.n_pes_dev} pessoas, {e.n_dev} clipes) e **RESERVADO** ({e.n_pes_res} pessoas, {e.n_clipes - e.n_dev} clipes). O DEV serviu só para **escolher o modo do idioma** (regra fixa: `auto` precisa dar inglês em ≥ 90% e WER não pior que o do inglês forçado + 0,05); depois a escolha foi **congelada** e o **RESERVADO rodou uma vez**. Prompt, LLM, cortes e código não mudaram.

**Resultado no RESERVADO** ({int(rvc["n"])} clipes, positivo = golpe; acurácia trivial {pct(rvc["acc_trivial"])}):

| versão | acurácia (IC95) | recall | falso alarme |
|:--|:--|:--|:--|
| 15/11 (idioma forçado errado) | {ic(rv15, "acc")} | {ic(rv15, "tpr")} | {ic(rv15, "fpr")} |
| **corrigido (`auto`)** | **{ic(rvc, "acc")}** | **{ic(rvc, "tpr")}** | **{ic(rvc, "fpr")}** |

Nos **100 clipes juntos** (descritivo; {e.n_dev} deles foram usados no diagnóstico): acurácia **{pct(cc["acc"], 0)}** (trivial {pct(cc["trivial"], 0)}), **recall {pct(cc["recall"], 0)}**, falso alarme {pct(cc["fa"], 0)} (em 15/11: {pct(c15["acc"], 0)}, {pct(c15["recall"], 0)} e {pct(c15["fa"], 0)}). **O idioma corrigido não melhorou nem piorou de forma demonstrável**: as diferenças são de poucos clipes e os ICs se sobrepõem. **O recall ficou bem abaixo da meta de 85% da avaliação com textos inventados.** Nenhum prompt, modelo ou corte foi mudado por causa disso.

**Por que o recall é baixo? Análise por frase** (`docs/ANALISE_ERRO_CONTEXTO.md`; descritiva, pós-hoc, **não usada para ajustar nada**). Os {f_cl[1] + f_le[1]} clones que pedem dinheiro usam só {f_cl[0] + f_le[0]} roteiros distintos, repetidos entre as pessoas; o mesmo texto recebe sempre o mesmo risco (cada roteiro é "tudo ou nada"):

| roteiros | clipes | alerta médio+ | só "alto" |
|:--|--:|--:|--:|
| **claros** (número novo, urgência, "pague por mim", cartão em viagem) — {f_cl[0]} roteiros | {f_cl[1]} | **{f_cl[2]} de {f_cl[1]}** | {f_cl[3]} |
| **pedidos leves** entre conhecidos (mercado, jantar, conta de luz, empréstimo curto) — {f_le[0]} roteiros | {f_le[1]} | **{f_le[2]} de {f_le[1]}** | {f_le[3]} |
| conversa do dia a dia (clones neutros) — {f_co[0]} roteiros | {f_co[1]} | {f_co[2]} (correto) | {f_co[3]} |

**Por que a acurácia geral engana aqui.** (1) Os **{f_le[1]} clones de "pedido leve"** estão rotulados como "golpe" **por construção** (pedem dinheiro), mas sem urgência, identidade falsa, código ou link: pelas instruções do módulo, "baixo" é o esperado, e parte deles talvez nem seja golpe. (2) Nos **reais "com golpe"**, o rótulo vem da coleta (gravações **encenadas**, não fraude): {rcg_nenhum} de {rcg_n} ficaram em "nenhum" no RESERVADO, e sem ler as falas não dá para dizer o que o módulo perdeu. (3) Com **{pct(cc["trivial"], 0)} dos clipes rotulados como "golpe"**, "chamar tudo de golpe" já acerta {pct(cc["trivial"], 0)}, mais que os {pct(cc["acc"], 0)} do módulo. **Para o contexto, o par recall × falso alarme é a leitura correta, não a acurácia.** O que sustentamos: os **{f_cl[0]} roteiros claramente de golpe foram todos pegos ({f_cl[2]} de {f_cl[1]})**; os perdidos são pedidos leves e gravações encenadas.

Contagem por grupo no RESERVADO (clipes com alerta médio+):

| grupo | n | 15/11 (errado) | corrigido |
|:--|--:|--:|--:|
{chr(10).join(lin)}
'''),
            md(r'''**O que faz:** mostra as tabelas da reavaliação do contexto como estão em `results/`. **Por quê:** os números acima vêm delas.'''),
            code(r'''
m = pd.read_csv(R / "diagnostico_whisper_idioma_metricas.csv")
display(m[m.recorte == "todos"][["conjunto", "versao", "n", "n_pos", "acc", "acc_trivial", "tpr", "fpr", "prec"]].round(3))
f = pd.read_csv(R / "analise_erro_contexto_por_frase.csv")
display(f.groupby("tema")[["n_clipes", "alerta_medio_mais", "alerta_so_alto"]].sum())
display(f[["frase_id", "tema", "n_clipes", "alerta_medio_mais", "alerta_so_alto", "recall"]])
'''),
            *v7nb.bloco_slide("ctxw_01_acuracia_antes_depois", "Figura ctxw_01 · Contexto antes e depois da correção do idioma",
                              "a acurácia, o recall e o falso alarme do contexto com o idioma forçado errado (15/11) e com o corrigido (`auto`), no DEV e no RESERVADO.",
                              "as barras das duas versões são parecidas e a acurácia fica abaixo da linha trivial.",
                              "corrigir o idioma não resolveu o recall baixo; o problema está no julgamento do conteúdo (hipótese, não verificada).",
                              origem="scripts/diagnostico_whisper_idioma_figura.py"),
            *bloco_fe("fe_15_contexto_antes_depois_idioma", "Figura fe_15 · Idioma errado × corrigido, com IC95",
                      "a acurácia, o recall e o falso alarme do contexto com o idioma forçado errado e com o `auto`, no conjunto reservado e no DEV.",
                      "as barras das duas versões são parecidas e os ICs se sobrepõem.",
                      "o idioma não era a causa do recall baixo."),
            *bloco_fe("fe_16_contexto_por_grupo", "Figura fe_16 · Alerta do contexto por grupo (idioma corrigido)",
                      "a % de clipes com alerta médio+ (e só alto) em cada um dos 4 grupos do banco WhatsApp, 100 clipes.",
                      "os clones que pedem dinheiro têm a maior parte dos alertas; os reais com golpe encenado quase não têm.",
                      "o contexto pega parte dos clones que pedem dinheiro e quase nada nas gravações encenadas; esta figura substitui a de 15/11."),
            *bloco_fe("fe_17_roteiros_claros_vs_leves", "Figura fe_17 · O contexto por roteiro",
                      "para cada um dos 16 roteiros dos clones, quantos clipes tiveram alerta médio+.",
                      "os 6 roteiros claros têm alerta em todos os clipes; os 4 pedidos leves, em nenhum.",
                      "o recall baixo vem dos pedidos leves de dinheiro, não dos roteiros claramente de golpe (análise descritiva).")]


def parte_h(e):
    cb = e.comb("todos")
    ordem = (("real_sem_golpe", "real, conteúdo normal"), ("real_com_golpe", "real, conteúdo de golpe (encenado)"),
             ("ia_pedido_dinheiro", "IA, pede dinheiro"), ("ia_neutro", "IA, neutro"))
    lin = []
    for k, rot in ordem:
        x = cb.loc[k]
        lin.append(f"| {rot} | {int(x.n)} | {int(x.baixo_corrigido)} | {int(x.atencao_corrigido)} | {int(x.alto_corrigido)} |")
    ia_baixo = int(cb.loc[["ia_pedido_dinheiro", "ia_neutro"], "baixo_corrigido"].sum())
    real_alto = int(cb.loc[["real_sem_golpe", "real_com_golpe"], "alto_corrigido"].sum())
    assert ia_baixo == 0 and real_alto == 0
    dif = max(abs(cb[c1] - cb[c2]).max() for c1, c2 in (("baixo_corrigido", "baixo_1511"), ("atencao_corrigido", "atencao_1511"),
                                                        ("alto_corrigido", "alto_1511")))
    return [md(rf'''
## H. O sistema combinado: voz × conteúdo

**A regra (congelada no protocolo do teste WhatsApp, antes de ver notas).** Cada áudio recebe dois "sinais": a **voz** (IA se a nota da T3 for ≥ M = {br(e.M)}) e o **conteúdo** (alerta se o contexto der risco médio ou alto). O **nível** sai da soma deles:
- **alto** = os **dois** sinais (voz de IA **e** conteúdo de golpe);
- **atenção** = **um** dos dois;
- **baixo** = **nenhum**.

Nos 100 clipes, com o idioma do Whisper corrigido (`docs/RESULTADO_COMBINADO_CORRIGIDO.md`; recálculo descritivo, sem nova chamada à API):

| grupo | n | baixo | atenção | alto |
|:--|--:|--:|--:|--:|
{chr(10).join(lin)}

**O que se vê:** **nenhuma voz de IA ficou em "baixo"** (a voz sozinha já as marca) e **nenhum áudio real chegou a "alto"**. Os clones que pedem dinheiro ficam em "alto" ou "atenção", conforme o contexto pegue ou não o roteiro; os clones neutros ficam em "atenção"; quase todos os reais ficam em "baixo". Em relação à versão de 15/11, **{int(dif)} clipe(s) no máximo mudam por célula**, dentro do ruído. **Isto não é uma "acurácia"**: são só taxas por célula, e a tabela **não deve ser lida como estimativa de desempenho em golpes reais** (um gerador, 23 pessoas, rótulos "com golpe" encenados).

**Como mostrar ao usuário.** O nível é **um alerta, não um veredito**: "alto" pede cautela e confirmação por outro canal; "atenção" diz que um dos dois sinais acendeu; "baixo" **não prova** que o áudio é real nem que a conversa é segura. O corte da voz (M) **não se transfere entre ambientes** (parte C.5), então os níveis valem só para um ambiente parecido com o do teste.
'''),
            md(r'''**O que faz:** mostra a tabela do combinado (grupo × nível), com a versão corrigida e a de 15/11, como gravada. **Por quê:** os números acima vêm dela.'''),
            code(r'''
cb = pd.read_csv(R / "diagnostico_whisper_combinado.csv")
display(cb[(cb.tabela == "combinado_grupo_x_nivel") & (cb.conjunto == "todos")][["grupo", "n", "baixo_corrigido", "atencao_corrigido", "alto_corrigido", "baixo_1511", "atencao_1511", "alto_1511"]])
'''),
            *bloco_fe("fe_18_combinado_grupo_nivel", "Figura fe_18 · Grupo × nível combinado (idioma corrigido)",
                      "quantos clipes de cada grupo ficaram em cada nível (baixo, atenção, alto), com as contagens.",
                      "nenhuma voz de IA está em 'baixo' e nenhum real chegou a 'alto'.",
                      "o combinado é um alerta, não um veredito; esta figura substitui a de 15/11.")]


def parte_i(e, L):
    br_f = float(L.tm3.loc["brspeech_df", "FPR M"])
    cc = e.ctx100("")
    fr = e.frases()
    n_rot = fr["claras"][0] + fr["leves"][0]
    return [md(rf'''
## I. Limitações e honestidade (quadro único)

> - **Um só gerador no WhatsApp.** Os {e.n_ia} clones são todos do mesmo gerador (Chatterbox). Os resultados não dizem nada sobre outros geradores.
> - **Só {e.n_pes} pessoas** ({e.n_real} reais e {e.n_ia} clones), ICs largos, e os áudios da mesma pessoa não são independentes.
> - **O teste de estresse deu 1 de 3.** Um gerador de família nova (fala-para-fala) escapou em 2 de 3 áudios; 3 áudios **não** estimam taxa nenhuma.
> - **O português não foi validado por nós.** O detector **não viu português no treino** (o treino tem estúdio em inglês, internet real multilíngue e IA em chinês, chinês de Taiwan e japonês), e o Whisper em modo `auto` **não foi testado em português** (o banco é inglês).
> - **O corte não se transfere entre ambientes.** Com o mesmo corte M, no português de audiolivro (BRSpeech-DF), **{pct(br_f, 0)} dos reais são acusados** como IA. O resultado de 98% no WhatsApp vale para aquele ambiente, não para qualquer um.
> - **O contexto só foi validado em inglês** (áudio), e em textos inventados em português do Brasil (só o texto). O recall no banco WhatsApp ({pct(cc["recall"], 0)}) está **abaixo da meta**; o falso alarme ({pct(cc["fa"], 0)}) não é desprezível.
> - **Nada foi testado em ligação ao vivo.** Os áudios do WhatsApp são **gravações** prontas; ligação tem outro canal, atraso e fala em tempo real.
> - **O teste mestre da v7 não é virgem** (o F0 já tinha sido medido nele), e **os {e.n_dev} clipes do DEV** do WhatsApp foram usados no diagnóstico do idioma; os agregados de 15/11 já tinham sido vistos.
> - **Os textos de pedido de dinheiro se repetem** ({n_rot} roteiros nos {e.gr["ia_pedido_dinheiro"]} clones), então o RESERVADO do contexto **não é independente em texto**; e o rótulo "com golpe" dos reais vem da coleta (encenação) e pode não ter pedido de dinheiro.
> - **O atalho de metadados** (parte D) não foi descontado: o 98% pode vir em parte do formato do áudio.
> - **Não comparamos com outros sistemas.** Por isso **não** afirmamos que o nosso é "melhor que o da literatura" **nem** que "generaliza para qualquer gerador".
> - **O contexto depende de um serviço externo** (Groq) e do modelo `gpt-oss-120b` na data do teste; e o áudio sai da máquina (no produto, o ideal é processar no aparelho e pedir consentimento).

**O que sustentamos:** com o sistema principal, num banco de WhatsApp de um gerador e inglês, o detector separou bem real de IA (com a ressalva do atalho); o módulo de contexto pega os roteiros claros de golpe e **perde** pedidos leves e encenações; o nível combinado é um **alerta, não um veredito**.
''')]


def proximos_entrega():
    return md(r'''
# Próximos passos (não executados)

Os passos da seção 16 continuam valendo (corte por ambiente, português no treino, clones próprios como teste intocado, o contexto de golpe e o empacotamento da T3). Depois do teste WhatsApp, do teste de estresse e da correção do contexto, a lista **atual** é:
1. **Mais geradores no teste de estresse:** 15 a 30 áudios de vários geradores comerciais (inclusive fala-para-fala), **sem treinar**, com protocolo escrito antes.
2. **Textos novos para o contexto:** um conjunto novo (e não os de hoje, já vistos) com golpes reais anonimizados e pedidos leves de dinheiro, e **só depois** pensar em mexer no prompt; qualquer ajuste feito com os textos atuais não seria validável.
3. **Voz real em português** (e áudio em português para o contexto): reunir reais e IA em português **fora** dos testes, para validar o detector e o Whisper em `auto`.
4. **Ligação ao vivo:** testar com chamadas de verdade (canal, atraso, fala em tempo real).
5. **Calibração do corte por ambiente** (estúdio, internet, aplicativo de mensagem, ligação), escolhida só pela validação.
6. **ffmpeg e Opus:** liberar a leitura de qualquer formato (m4a, opus, webm) com o ffmpeg e testar a conversão, que hoje **não foi testada**.

Do experimento v8 (anexo) ficam: mais sementes da R1 (3 ou mais), um corte estável entre elas e ajuste fino com português no treino. Nada disso foi executado. Qualquer mudança no sistema principal deve ser decidida **só pela validação** e registrada em `docs/HISTORICO_VERSOES.md` antes de um novo teste.
''')


GLOSSARIO_ENTREGA = r'''| **Idiomas do treino** | o treino da T3 **não** é só inglês: tem estúdio em inglês, internet real multilíngue (ML-ITW, sem os locutores com áudio em português) e IA moderna do RADAR em **chinês, chinês de Taiwan e japonês**. **Não tem português.** |
| **Acurácia trivial** | o que se acerta "chamando tudo da classe mais comum". Toda acurácia deve ser lida ao lado dela. |
| **Recall / TPR · falso alarme / FPR** | recall (TPR) = % dos positivos pegos (IA detectada; golpe com alerta). Falso alarme (FPR) = % dos negativos acusados (real chamado de IA; conversa normal com alerta). |
| **Bootstrap por pessoa** | sortear as pessoas (e não os áudios) com reposição, muitas vezes, para ter uma faixa plausível (IC95) que respeita que áudios da mesma pessoa se parecem. |
| **Opus** | formato de áudio comprimido (usado em aplicativos de mensagem e chamadas). Na condição B, os clones foram recodificados em Opus de ~16 kbps. |
| **Atalho (metadados)** | um jeito de acertar sem ouvir a voz, só pelo formato do arquivo (taxa de amostragem, volume, silêncio). Se um classificador de metadados chega perto do detector, o detector pode estar usando o atalho. |
| **Whisper · LLM** | Whisper = modelo que transcreve fala em texto. LLM = modelo de linguagem que lê um texto e responde por instruções (aqui, para julgar o risco de golpe). ⚠️ Fora da trilha. |
| **Nível de risco / nível combinado** | risco do conteúdo: nenhum, baixo, médio, alto. Combinado com a voz: **alto** = voz de IA e conteúdo de golpe; **atenção** = um dos dois; **baixo** = nenhum. É um **alerta, não um veredito**. |
| **WER** | taxa de erro de palavras da transcrição (0 = perfeita). |
| **Errata** | correção publicada de um resultado que estava errado (aqui, o idioma forçado do Whisper em 15/11). |
| **DEV / RESERVADO** | divisão das 23 pessoas do teste WhatsApp para a correção do contexto: o DEV (8) serve para escolher o modo do idioma; o RESERVADO (15) é avaliado uma vez. |
'''


# ------------------------------------------------------------------------------------------------ montagem
def ajustar_topo(C, e):
    s = C[0].source
    n_trein = ("**Objetivo.** Dizer se uma voz foi gerada por IA. O treino inclui **estúdio em inglês**, **internet real multilíngue** (ML-ITW) "
               "e **IA moderna em canal degradado** (RADAR, em chinês, chinês de Taiwan e japonês), **sem português**; os testes incluem vozes de "
               "internet, fakes parciais e português.")
    velho = "**Objetivo.** Dizer se uma voz foi gerada por IA. O treino começou só em inglês de estúdio e audiolivro; os testes incluem vozes de internet, fakes parciais e português."
    assert velho in s
    s = s.replace(velho, n_trein)
    assert "> **Resumo em 5 linhas**" in s
    s = s.replace("> **Resumo em 5 linhas**", "> **Resumo em 8 linhas**")
    ant = "nada aqui fala de ligações ou de WhatsApp reais."
    assert ant in s
    s = s.replace(ant, "nada foi testado em ligação ao vivo (o WhatsApp foi testado só em gravações, parte D).")
    t3a = e.d("T3", "M", "A")
    p3a = e.p("T3", "A")
    est = e.est
    n_det = int(est[(est.condicao == "original")].passa_M.sum())
    fr = e.frases()
    rvc, rv15 = e.dm("RESERVADO", "corrigido"), e.dm("RESERVADO", "15/11")
    novas = (
        f"\n> 6. **Teste externo WhatsApp (parte D; {e.n_real + e.n_ia} áudios, {e.n_pes} pessoas, um só gerador, inglês; protocolo antes, rodou uma vez):** a T3 (corte M) deu **{pct(t3a['acc'], 0)}** de acurácia (IC95 {br(100 * t3a['acc_lo'], 1)} a {br(100 * t3a['acc_hi'], 1)}; trivial {pct(t3a['acc_trivial'], 0)}), TPR {pct(t3a['tpr'], 0)}, falso alarme {pct(t3a['fpr'], 1)}, AUC {br(t3a['auc'])}, e {int(p3a.ia_maior_que_real)} de {int(p3a.pessoas)} pessoas com a IA acima do real. **Ressalva:** um classificador só de metadados do arquivo chega a AUC {br(e.ata.loc['sem a taxa original', 'AUC_loo'])}, então o teste não descarta o atalho de formato.\n"
        f"> 7. **Teste de estresse informal (parte E; não oficial):** de **3 áudios** de um gerador fala-para-fala novo, a T3 detectou **{n_det} de 3**: geradores de família nova escapam.\n"
        f"> 8. **Contexto de golpe (partes F a H):** em textos inventados, recall **{pct(e.aval['positivo_medio_ou_alto']['recall_golpes_todos']['taxa'], 0)}** e falso alarme **{pct(e.aval['positivo_medio_ou_alto']['falso_alarme_legitimos_todos']['taxa'], 0)}**; no banco WhatsApp, com o idioma corrigido (**errata**: em 15/11 forçamos o Whisper em português por engano), a acurácia foi **{pct(rvc['acc'], 0)}** (trivial {pct(rvc['acc_trivial'], 0)}) e o recall **{pct(rvc['tpr'], 0)}** no conjunto reservado. Os roteiros claros de golpe foram todos pegos ({fr["claras"][2]} de {fr["claras"][1]}); os pedidos leves, nenhum ({fr["leves"][2]} de {fr["leves"][1]}). O sistema combinado é um **alerta, não um veredito**.\n"
    )
    # as 3 linhas entram logo depois da linha 5 do resumo (a que começa por '> 5.')
    i5 = s.index("> 5. **Limites que valem para tudo:**")
    fim5 = s.index("\n", i5)
    s = s[:fim5] + novas.rstrip("\n") + s[fim5:]
    velho2 = "O **anexo** (seções 17 a 22) é o **experimento de ajuste fino da v8, não adotado**."
    assert velho2 in s
    s = s.replace(velho2, "As **partes D a I** (logo depois da parte C) apresentam o que veio **depois da v7**: o teste externo WhatsApp, o teste de estresse informal, "
                  "o módulo de contexto de golpe (com a **errata** do idioma), o sistema combinado voz × conteúdo e o quadro de limitações. " + velho2)
    C[0] = md(s)


def montar():
    C = P.montar()
    e = NE()
    L = P.NumsLeve()
    ajustar_topo(C, e)
    # parte do F0: "o treino é todo em inglês" vale só para o F0 (v4)
    P.trocar(C, "**Conclusão:** o treino é todo em inglês e quase todo de estúdio ou audiolivro;",
             "**Conclusão:** o treino **do F0 (v4)** é todo em inglês e quase todo de estúdio ou audiolivro (na v7, o treino ganhou internet real multilíngue e o RADAR em chinês e japonês, mas **continua sem português**);")
    # guia (C.5): ligação ao vivo x WhatsApp
    P.trocar(C, "- **Nada foi testado em ligações, WhatsApp nem celular.**",
             "- **Nada foi testado em ligação ao vivo nem em celular.** O WhatsApp foi testado só com **gravações** de um gerador (parte D), em inglês.")
    # glossário
    ig = v7nb.indice(C, "### Glossário em linguagem simples")
    C[ig].source = C[ig].source.rstrip("\n") + "\n" + GLOSSARIO_ENTREGA
    # parte nova, antes do anexo
    ia = v7nb.indice(C, "# Anexo: experimento de ajuste fino (v8), não adotado")
    novas = parte_d(e) + parte_e(e) + parte_f(e) + parte_g(e) + parte_h(e) + parte_i(e, L)
    C[ia:ia] = novas
    C[-1] = proximos_entrega()
    return C


def textos(c):
    return P.textos(c)


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
    cfg = [c for c in nb.cells if "config" in c.get("metadata", {}).get("tags", [])][0].source
    assert "AVALIAR_TESTE_FINAL = False" in cfg
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
            for p in P.PROIBIDOS:
                if re.search(rf"(?<!\w){re.escape(p)}", t, flags=re.I):
                    achados.append((i, p))
            for p in NOMES_TAREFA:
                if re.search(rf"\b{re.escape(p)}\b", sem_acento_txt(t), flags=re.I):
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
