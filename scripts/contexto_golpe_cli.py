#!/usr/bin/env python3
"""Linha de comando do módulo de contexto de golpe (src/contexto_golpe.py). Protótipo; não treina nada.

Uso (com o env ativo e GROQ_API_KEY definida no ambiente; veja docs/CONTEXTO_GOLPE.md):
    python scripts/contexto_golpe_cli.py --teste-conexao
    python scripts/contexto_golpe_cli.py --texto "Oi mãe, troquei de número..."
    python scripts/contexto_golpe_cli.py --audio data/exemplos_contexto/golpe_1.m4a [--max-seg 120] [--idioma pt]
    python scripts/contexto_golpe_cli.py --smoke
    python scripts/contexto_golpe_cli.py --audio meu.wav --salvar resultado.json   # grava o JSON (com o texto)

Imprime o JSON de saída. NÃO envie áudios dos bancos do projeto (CORAA, ML-ITW, RADAR...): só áudios seus.
Nada é guardado por padrão além dos metadados em logs/contexto_golpe.jsonl (sem áudio e sem texto).
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import contexto_golpe as cg  # noqa: E402

PASTA_EXEMPLOS = ROOT / "data" / "exemplos_contexto"

# Textos INVENTADOS para o teste de fumaça. Só verificam o fluxo, a validação e o formato. NÃO são avaliação.
SMOKE = [
    ("golpe: parente pede Pix", "Oi mãe, sou eu, troquei de número, esse é o meu novo. Tô numa situação difícil, "
     "preciso que você me faça um Pix de mil e quinhentos reais agora, é urgente, depois eu te explico. "
     "Não comenta com o pai, tá?"),
    ("golpe: falso gerente pede código", "Boa tarde, aqui é o gerente da sua agência. Identificamos uma "
     "movimentação suspeita na sua conta e ela vai ser bloqueada em dez minutos. Para cancelar, me informe o código "
     "de seis dígitos que acabou de chegar por SMS no seu celular."),
    ("golpe: entrega retida com link", "Olá, sua encomenda está retida no centro de distribuição por falta de "
     "pagamento da taxa de importação. Acesse o link rastreio-entrega-taxa ponto com e pague hoje, senão o pacote "
     "volta para o remetente."),
    ("normal: banco avisa compra", "Olá, aqui é do seu banco. Estamos ligando só para avisar que foi feita uma "
     "compra de cento e vinte reais no seu cartão no supermercado hoje à tarde. Se foi você, não precisa fazer "
     "nada. Se não reconhecer, ligue para o número que está no verso do seu cartão."),
    ("normal: condomínio", "Bom dia, aqui é da administração do condomínio. Só lembrando que o boleto da taxa "
     "deste mês já está disponível no aplicativo do condomínio e vence no dia dez, como sempre. Qualquer dúvida, "
     "é só passar na portaria."),
    ("normal: amigo combina almoço", "E aí, tudo bem? Vamos almoçar na sexta? Pensei naquele restaurante perto "
     "do trabalho, umas meio-dia e meia. Cada um paga o seu, tranquilo. Me fala se dá pra você."),
]


def imprimir(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def salvar(obj, destino):
    p = Path(destino)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[salvo] {p} (contém a transcrição; não coloque no git)", file=sys.stderr)


def tom_curto():
    """1 s de um tom de 440 Hz, gerado aqui (não é áudio de nenhum banco). Só para testar a conexão do ASR."""
    import numpy as np
    import soundfile as sf
    t = np.arange(cg.SR) / cg.SR
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32"), cg.SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def teste_conexao():
    """Lista os modelos (GET) e faz UMA chamada mínima de cada etapa. Mostra só status e latência."""
    m = cg.modelos()
    rel = {"modelos_configurados": m}
    try:
        ids, ms, lim = cg.listar_modelos()
        rel["listar_modelos"] = {"status": "ok", "latencia_ms": ms, "n_modelos": len(ids), "ids": ids,
                                 "asr_existe": m["asr"] in ids, "llm_existe": m["llm"] in ids, "limites": lim}
    except cg.ErroGroq as e:
        rel["listar_modelos"] = {"status": "erro", "erro": str(e)}
        imprimir(rel)
        return 1
    try:
        tr = cg.transcrever(tom_curto(), "tom.wav", "audio/wav", "pt")
        rel["transcricao"] = {"status": "ok", "latencia_ms": tr["latencia_ms"], "limites": tr["limites"]}
    except cg.ErroGroq as e:
        rel["transcricao"] = {"status": "erro", "erro": str(e)}
    try:
        corpo = {"model": m["llm"], "temperature": 0, "max_tokens": 64, **cg.extras_llm(m["llm"]),
                 "messages": [{"role": "user", "content": "Responda só: ok"}]}
        _, ms, lim = cg._requisitar("POST", "/chat/completions", "teste_llm", json=corpo)
        rel["classificacao"] = {"status": "ok", "latencia_ms": ms, "limites": lim}
    except cg.ErroGroq as e:
        rel["classificacao"] = {"status": "erro", "erro": str(e)}
    imprimir(rel)
    return 0 if all(rel[k]["status"] == "ok" for k in ("listar_modelos", "transcricao", "classificacao")) else 1


def smoke():
    print("TESTE DE FUMAÇA: 6 textos inventados. Só confere o fluxo, a validação e o formato; NÃO é avaliação.\n",
          file=sys.stderr)
    linhas = []
    for rotulo, texto in SMOKE:
        r = cg.analisar_texto(texto)
        sinais = ", ".join(s["tipo"] for s in r["sinais"]) or "-"
        linhas.append((rotulo, r["status"], r["risco_conteudo"], sinais, r["evidencias_descartadas"],
                       r["latencia_ms"]["classificacao"], r["erro"] or ""))
        time.sleep(1)                                    # folga para o limite por minuto
    print("| texto | status | risco | sinais | evid. descartadas | latência (ms) | erro |")
    print("|:--|:--|:--|:--|--:|--:|:--|")
    for lin in linhas:
        print("| " + " | ".join(str(c) for c in lin) + " |")
    audios = sorted(p for p in PASTA_EXEMPLOS.glob("*") if p.is_file() and p.stem in ("golpe_1", "normal_1"))
    if not audios:
        print(f"\nTeste de áudio PENDENTE: não há golpe_1.* nem normal_1.* em {PASTA_EXEMPLOS} "
              "(grave os seus; veja docs/CONTEXTO_GOLPE.md, seção 3).")
        return 0
    print("\n| áudio | status | risco | sinais | suspeita | preparo / transcrição / classificação (ms) | observações | erro |")
    print("|:--|:--|:--|:--|:--|:--|:--|:--|")
    for p in audios:
        r = cg.analisar_audio(p)
        lat = r["latencia_ms"]
        print(f"| {p.name} | {r['status']} | {r['risco_conteudo']} | "
              f"{', '.join(s['tipo'] for s in r['sinais']) or '-'} | {r['transcricao_suspeita']} | "
              f"{lat['preparo']} / {lat['transcricao']} / {lat['classificacao']} | "
              f"{'; '.join(r['observacoes']) or '-'} | {r['erro'] or ''} |")
    return 0


def main():
    for f in (sys.stdout, sys.stderr):                  # console do Windows: não quebrar com acentos
        try:
            f.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", metavar="ARQUIVO", help="áudio seu (wav, flac, ogg, mp3, m4a, webm...)")
    g.add_argument("--texto", help="texto já escrito (pula a transcrição)")
    g.add_argument("--teste-conexao", action="store_true", help="lista os modelos e faz 1 chamada mínima por etapa")
    g.add_argument("--smoke", action="store_true", help="teste de fumaça com 6 textos inventados")
    ap.add_argument("--max-seg", type=float, default=cg.MAX_SEG_PADRAO,
                    help="envia só os primeiros N s (depois do corte de silêncio) dos formatos lidos aqui (padrão 120)")
    ap.add_argument("--idioma", default="pt", help='idioma da transcrição (padrão "pt"; "auto" detecta)')
    ap.add_argument("--max-chars", type=int, default=cg.MAX_CHARS_PADRAO, help="limite de caracteres enviados ao LLM")
    ap.add_argument("--salvar", metavar="DESTINO", help="grava o JSON de saída (com a transcrição) neste arquivo")
    a = ap.parse_args()
    import os
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("GROQ_API_KEY não está definida. No PowerShell: setx GROQ_API_KEY \"<sua chave>\" e abra um terminal "
              "novo (a chave fica só na sua máquina; não a coloque em arquivos do projeto).", file=sys.stderr)
        return 2
    if a.teste_conexao:
        return teste_conexao()
    if a.smoke:
        return smoke()
    r = cg.analisar_texto(a.texto, a.max_chars) if a.texto is not None else \
        cg.analisar_audio(a.audio, idioma=a.idioma, max_seg=a.max_seg, max_chars=a.max_chars)
    imprimir(r)
    if a.salvar:
        salvar(r, a.salvar)
    return 0 if r["status"] != "erro" else 1


if __name__ == "__main__":
    sys.exit(main())
