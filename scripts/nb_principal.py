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

md, code, br, pct = v7nb.md, v7nb.code, v7nb.br, v7nb.pct
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
    s = s.replace(velho, velho + " O **anexo** (seções 17 a 22) é o **experimento de ajuste fino da v8, não adotado**.")
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
