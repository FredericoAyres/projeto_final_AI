#!/usr/bin/env python3
"""Roda o esboço do app nos áudios de app/exemplos.csv e prepara o mockup (app/mockup/index.html).

    python app/demo.py                          # contexto offline (transcrição digitada + palavras-chave)
    python app/demo.py --contexto groq          # contexto dos colegas (src/contexto_golpe.py): ENVIA o áudio ao
                                                # Groq; precisa de GROQ_API_KEY e do consentimento de quem gravou
    python app/demo.py --exemplos outro.csv     # colunas: id, arquivo, transcricao, titulo, contato, hora

Para cada áudio: voz (T3) -> contexto (transcrição + classificação) -> alerta. Grava:
  app/saida/resultados.json   tudo o que foi calculado
  app/mockup/dados.js         o mesmo, para o mockup abrir direto no navegador (sem servidor)
  app/mockup/audios/          cópia dos áudios, para tocar no mockup (fora do git)
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))

from app import alerta, contexto  # noqa: E402
from app.voz import DetectorVoz  # noqa: E402


def caminho(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    for f in (sys.stdout, sys.stderr):            # console do Windows: não quebrar com acentos
        try:
            f.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exemplos", default=str(APP / "exemplos.csv"))
    ap.add_argument("--variante", choices=("v1", "v2"), default="v1")
    ap.add_argument("--contexto", choices=contexto.MODOS, default="offline")
    a = ap.parse_args()
    with open(a.exemplos, encoding="utf-8") as f:
        exemplos = list(csv.DictReader(f))

    t0 = time.perf_counter()
    voz = DetectorVoz(a.variante)
    print(f"[voz] T3 carregada em {time.perf_counter() - t0:.1f} s ({voz.aparelho})")

    pasta_audios = APP / "mockup" / "audios"
    pasta_audios.mkdir(parents=True, exist_ok=True)
    resultados = []
    for ex in exemplos:
        arq = caminho(ex["arquivo"])
        r_voz = voz.analisar(arq)
        r_ctx = contexto.analisar(arq, a.contexto, ex.get("transcricao"))
        r_alerta = alerta.combinar(r_voz, r_ctx)
        copia = pasta_audios / f"{ex['id']}{arq.suffix.lower()}"
        shutil.copyfile(arq, copia)
        resultados.append({"id": ex["id"], "titulo": ex.get("titulo", ""), "contato": ex.get("contato", ""),
                           "hora": ex.get("hora", ""), "arquivo": ex["arquivo"],
                           "audio_mockup": f"audios/{copia.name}", "voz": r_voz,
                           "contexto": r_ctx, "alerta": r_alerta})
        print(f"\n[{ex['id']}] {ex.get('titulo', '')}")
        print(f"  voz: nota {r_voz['nota']:+.3f} (corte {r_voz['corte']:.3f}) -> "
              f"{'parece IA' if r_voz['parece_ia'] else 'parece real'}")
        print(f"  contexto ({a.contexto}): status {r_ctx['status']}, risco {r_ctx['risco_conteudo']} "
              f"({', '.join(s['tipo'] for s in r_ctx['sinais']) or 'nenhum sinal'})"
              + (f"; erro: {r_ctx['erro']}" if r_ctx.get("erro") else ""))
        print(f"  alerta: {r_alerta['nivel'].upper()} - {r_alerta['titulo']}")

    saida = {"gerado_em": time.strftime("%Y-%m-%d %H:%M"), "exemplos": resultados,
             "modo_contexto": a.contexto,
             "aviso": ("Esboço. Voz: T3 (sistema principal). Contexto: " +
                       ("módulo dos colegas (Groq: Whisper + LLM)." if a.contexto == "groq" else
                        "modo offline (transcrição digitada + palavras-chave), não o módulo dos colegas."))}
    (APP / "saida").mkdir(exist_ok=True)
    (APP / "saida" / "resultados.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    (APP / "mockup" / "dados.js").write_text("window.RESULTADOS = " + json.dumps(saida, ensure_ascii=False, indent=2)
                                            + ";\n", encoding="utf-8")
    print("\ngravado app/saida/resultados.json e app/mockup/dados.js; abra app/mockup/index.html no navegador")


if __name__ == "__main__":
    main()
