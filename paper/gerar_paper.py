#!/usr/bin/env python3
"""Gera paper/paper.html e paper/paper.pdf a partir de paper/paper.md (sem instalar nada).

    python paper/gerar_paper.py

- Converte o subconjunto de Markdown usado no paper (títulos, parágrafos, negrito/itálico, listas, tabelas,
  citações, imagens com legenda) num HTML com estilo de impressão.
- O estilo segue o dos papers de referência (NeurIPS/arXiv): Times 10 pt numa coluna estreita, título entre
  dois filetes, resumo centralizado e recuado, seções "1 Introdução", tabelas com só três filetes
  (legenda em cima) e figuras com legenda embaixo. Antes vêm a capa e o sumário, pedidos pelo grupo.
- O PDF é impresso pelo Chrome instalado no computador, em modo sem janela.
⚠️ Fora da trilha: HTML/CSS e impressão pelo navegador.
"""
import html
import re
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MD, HTML, PDF = AQUI / "paper.md", AQUI / "paper.html", AQUI / "paper.pdf"
CHROMES = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
           Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")]

CSS = """
@page { size: A4; margin: 25mm 32mm 27mm 32mm;
        @bottom-center { content: counter(page); font: 10pt "Times New Roman", serif; } }
@page capa { @bottom-center { content: none; } }
html { font-family: "Times New Roman", Times, serif; font-size: 10pt; color: #000; }
body { margin: 0; line-height: 1.22; }
p { margin: 0 0 5.5pt; text-align: justify; hyphens: auto; }
strong { font-weight: bold; }
code { font-family: "Courier New", monospace; font-size: 9pt; }
a { color: inherit; text-decoration: none; }

/* capa */
.capa { page: capa; height: 240mm; display: flex; flex-direction: column; text-align: center; break-after: page; }
.capa .inst { font-size: 11pt; font-variant: small-caps; letter-spacing: .3px; margin-top: 10mm; }
.capa .faixa { margin-top: 55mm; border-top: 4pt solid #000; border-bottom: 1pt solid #000; padding: 14pt 0 16pt; }
.capa h1 { font-size: 22pt; line-height: 1.2; margin: 0; }
.capa .sub { font-size: 13pt; font-style: italic; margin: 10pt 8mm 0; }
.capa .autores { font-size: 12pt; margin-top: 22mm; }
.capa .data { font-size: 11pt; margin-top: auto; }

/* sumário */
.sumario { break-after: page; }
.sumario h2 { font-size: 12pt; margin: 0 0 10pt; }
.sumario ol { list-style: none; padding-left: 0; margin: 0; }
.sumario li { margin: 0 0 3pt; } .sumario li.n2 { font-weight: bold; margin-top: 6pt; }
.sumario li.n3 { padding-left: 18pt; }

/* primeira página, como nos papers */
.cabecalho { text-align: center; }
.cabecalho .faixa { border-top: 4pt solid #000; border-bottom: 1pt solid #000; padding: 10pt 0 11pt; margin-bottom: 16pt; }
.cabecalho h1 { font-size: 17pt; line-height: 1.2; margin: 0; }
.cabecalho .autores { font-size: 10pt; margin-bottom: 4pt; }
.cabecalho .autores b { display: block; }
.cabecalho .inst { font-size: 10pt; margin-bottom: 22pt; }
.resumo h2 { text-align: center; font-size: 12pt; margin: 0 0 6pt; }
.resumo p { margin: 0 12.7mm 5.5pt; }
.resumo { margin-bottom: 14pt; }

/* seções */
h2 { font-size: 12pt; font-weight: bold; margin: 13pt 0 6pt; break-after: avoid; }
h3 { font-size: 10pt; font-weight: bold; margin: 10pt 0 4pt; break-after: avoid; }
h2 .num, h3 .num { display: inline-block; min-width: 18pt; }
h3 .num { min-width: 22pt; }
h2.quebra { break-before: page; }
ul, ol { margin: 0 0 5.5pt; padding-left: 16pt; } li { margin-bottom: 2pt; text-align: justify; }
.equacao { display: flex; align-items: center; margin: 6pt 0 8pt; }
.equacao span:first-child { flex: 1; text-align: center; }

/* tabelas: só três filetes, legenda em cima */
.tabela { margin: 8pt 0 10pt; break-inside: avoid; }
.tabela .legenda { text-align: center; margin: 0 0 4pt; }
table { border-collapse: collapse; margin: 0 auto; font-size: 9pt; }
th, td { padding: 1.5pt 7pt; vertical-align: top; text-align: left; border: 0; }
thead th { border-top: 1pt solid #000; border-bottom: .5pt solid #000; font-weight: bold; }
tbody tr:last-child td { border-bottom: 1pt solid #000; }
td.num, th.num { text-align: right; white-space: nowrap; }
p.legenda { font-size: 9pt; text-align: left; }
p.legenda > em:first-child { font-style: normal; }   /* legenda sem itálico, como nos papers */

/* figuras: legenda embaixo */
figure { margin: 8pt 0 10pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 95mm; }
figure.alta img { max-height: 190mm; }
figcaption { font-size: 9pt; text-align: justify; margin-top: 4pt; }
.rotulo { font-weight: normal; }

/* referências e apêndices */
.referencias p { font-size: 9pt; text-align: left; padding-left: 16pt; text-indent: -16pt; margin-bottom: 3pt; }
.pendente { background: #fff3b0; }
"""


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\[(confirmar:[^\]]+)\]", r'<span class="pendente">[\1]</span>', t)
    t = re.sub(r"\[(Autores: [^\]]+)\]", r'<span class="pendente">[\1]</span>', t)
    return t


def legenda(t):
    """'Figura 1. Texto' -> 'Figura 1: Texto', como nos papers."""
    return re.sub(r"^(<em>)?(Figura|Tabela) (\d+)\.", r"\1\2 \3:", t)


def titulo_html(t):
    """'1. Introdução' -> número separado do nome, sem o ponto (como em '1 Introduction')."""
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$", t)
    if not m:
        return inline(t)
    return f'<span class="num">{m.group(1)}</span>{inline(m.group(2))}'


def ancora(texto):
    return "s-" + re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def tabela(linhas):
    cel = [[c.strip() for c in l.strip().strip("|").split("|")] for l in linhas]
    cab, alin, corpo = cel[0], cel[1], cel[2:]
    num = [a.endswith(":") and not a.startswith(":") or a.startswith("--:") for a in alin]
    th = "".join(f'<th class="{"num" if n else ""}">{inline(c)}</th>' for c, n in zip(cab, num))
    trs = "".join("<tr>" + "".join(f'<td class="{"num" if n else ""}">{inline(c)}</td>' for c, n in zip(r, num)) + "</tr>"
                  for r in corpo)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def converter(texto):
    meta, corpo = {}, texto
    if texto.startswith("---"):
        cab, corpo = texto[3:].split("\n---", 1)
        for l in cab.strip().splitlines():
            k, v = l.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    linhas = corpo.strip("\n").splitlines()
    saida, titulos, i = [], [], 0
    while i < len(linhas):
        l = linhas[i]
        if not l.strip():
            i += 1
            continue
        if l.strip() == "[[SUMARIO]]":
            saida.append("[[SUMARIO]]")
            i += 1
        elif l.startswith("### ") or l.startswith("## "):
            nivel = 3 if l.startswith("### ") else 2
            t = l[nivel + 1:].strip()
            a = ancora(t)
            titulos.append((nivel, t, a))
            if nivel == 2 and t.startswith("Referências"):
                saida.append("[[REF]]")
            cls = ' class="quebra"' if nivel == 2 and (t.startswith("Referências") or t.startswith("Apêndice A")) else ""
            saida.append(f'<h{nivel} id="{a}"{cls}>{titulo_html(t)}</h{nivel}>')
            if nivel == 2 and t.startswith("Apêndice A"):
                saida.append("[[/REF]]")
            i += 1
        elif l.startswith("|"):
            bloco = []
            while i < len(linhas) and linhas[i].startswith("|"):
                bloco.append(linhas[i])
                i += 1
            saida.append(tabela(bloco))
        elif l.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.+)\)\s*$", l)
            alt, src = m.group(1), m.group(2)
            cls = ' class="alta"' if "app_telas" in src else ""
            saida.append(f'<figure{cls}><img src="{html.escape(src)}" alt="{html.escape(alt)}">'
                         f"<figcaption>{legenda(inline(alt))}</figcaption></figure>")
            i += 1
        elif re.match(r"^(- |\d+\. )", l):
            ordenada = bool(re.match(r"^\d+\. ", l))
            itens = []
            while i < len(linhas) and re.match(r"^(- |\d+\. )", linhas[i]):
                itens.append(re.sub(r"^(- |\d+\. )", "", linhas[i]))
                i += 1
            tag = "ol" if ordenada else "ul"
            saida.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in itens) + f"</{tag}>")
        elif l.startswith("> "):
            n_eq = sum(s.startswith('<div class="equacao"') for s in saida) + 1
            saida.append(f'<div class="equacao"><span>{inline(l[2:])}</span><span>({n_eq})</span></div>')
            i += 1
        else:
            par = []
            while i < len(linhas) and linhas[i].strip() and not re.match(r"^(#|\||!\[|- |\d+\. |> |\[\[)", linhas[i]):
                par.append(linhas[i].strip())
                i += 1
            t = " ".join(par)
            if re.match(r"^\*(Tabela|Detalhes)", t):
                saida.append(f'<p class="legenda">{legenda(inline(t))}</p>')
            else:
                saida.append(f"<p>{inline(t)}</p>")

    # tabela + legenda: a legenda vai para cima (como nos papers) e as duas ficam juntas
    junto = []
    for s in saida:
        if s.startswith('<p class="legenda">') and junto and junto[-1].startswith("<table") and "Tabela" in s:
            junto[-1] = f'<div class="tabela">{s}{junto[-1]}</div>'
        else:
            junto.append(s)
    saida = junto

    # referências em fonte menor
    corpo_txt = "".join(saida)
    corpo_txt = corpo_txt.replace("[[REF]]", '<section class="referencias">', 1)
    if "[[/REF]]" in corpo_txt:   # fecha antes do título do Apêndice A
        corpo_txt = re.sub(r'(<h2 id="s-ap[^"]*" class="quebra">.*?</h2>)\[\[/REF\]\]', r"</section>\1", corpo_txt, count=1)

    k = corpo_txt.index("[[SUMARIO]]")
    resumo = '<section class="resumo">' + corpo_txt[:k] + "</section>"
    resto = corpo_txt[k + len("[[SUMARIO]]"):]

    itens = "".join(f'<li class="n{n}"><a href="#{a}">{titulo_html(t)}</a></li>'
                    for n, t, a in titulos if t != "Resumo")
    sumario = f'<nav class="sumario"><h2>Sumário</h2><ol>{itens}</ol></nav>'
    capa = (f'<section class="capa"><div class="inst">{inline(meta.get("instituicao", ""))}</div>'
            f'<div class="faixa"><h1>{inline(meta.get("titulo", ""))}</h1>'
            f'<div class="sub">{inline(meta.get("subtitulo", ""))}</div></div>'
            f'<div class="autores">{inline(meta.get("autores", ""))}</div><div class="data">{inline(meta.get("data", ""))}</div></section>')
    cabecalho = (f'<header class="cabecalho"><div class="faixa"><h1>{inline(meta.get("titulo", ""))}</h1></div>'
                 f'<div class="autores"><b>{inline(meta.get("autores", ""))}</b></div>'
                 f'<div class="inst">{inline(meta.get("instituicao", ""))}</div></header>')
    corpo_html = capa + sumario + cabecalho + resumo + resto
    return (f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>{html.escape(meta.get("titulo", ""))}</title>'
            f"<style>{CSS}</style></head><body>{corpo_html}</body></html>")


def main():
    HTML.write_text(converter(MD.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"gravado {HTML}")
    chrome = next((c for c in CHROMES if c.exists()), None)
    if not chrome:
        sys.exit("Chrome/Edge não encontrado: abra paper.html no navegador e imprima em PDF.")
    subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)
    print(f"gravado {PDF} ({PDF.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
