#!/usr/bin/env python3
"""Servidor local do esboço: mostra as telas (app/mockup) e analisa áudios enviados pela página.

    python app/servidor.py                    # contexto offline (Whisper local + palavras-chave)
    python app/servidor.py --contexto groq    # contexto dos colegas (src/contexto_golpe.py): ENVIA o áudio ao Groq;
                                              # precisa de GROQ_API_KEY e do consentimento de quem gravou
    depois, abra http://127.0.0.1:8765

Cada áudio enviado passa pelo mesmo caminho do app/demo.py: voz (T3) -> transcrição -> contexto -> alerta.
- Transcrição no modo offline: Whisper large-v3-turbo do cache local do Hugging Face (sem internet). No modo groq,
  quem transcreve é o próprio módulo dos colegas.
- O áudio enviado é gravado em app/mockup/enviados/ (fora do git) só para a página poder tocá-lo. Nada mais é
  gravado: o resultado vive só na página e some ao recarregar.
- Só escuta em 127.0.0.1 (este computador). ⚠️ Fora da trilha: servidor HTTP e transcrição com o Whisper.
"""
import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # Whisper só do cache local, sem baixar nada

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))

from app import alerta, contexto  # noqa: E402
from app.voz import DetectorVoz  # noqa: E402

MOCKUP = APP / "mockup"
ENVIADOS = MOCKUP / "enviados"
FORMATOS = {".wav", ".mp3", ".ogg", ".flac"}         # os que o projeto lê sem ffmpeg (src/preproc.py)
LIMITE_BYTES = 25 * 1024 * 1024
ASR = "openai/whisper-large-v3-turbo"


class Motor:
    """Carrega os modelos uma vez e analisa um arquivo de cada vez."""

    def __init__(self, modo):
        self.modo = modo
        t0 = time.perf_counter()
        self.voz = DetectorVoz("v1")
        self.asr = None
        if modo == "offline":
            import torch
            from transformers import pipeline
            self.asr = pipeline("automatic-speech-recognition", model=ASR, device=0 if torch.cuda.is_available() else -1)
        print(f"[servidor] modelos carregados em {time.perf_counter() - t0:.0f} s (contexto: {modo})", flush=True)

    def transcrever(self, caminho):
        import numpy as np
        from src.preproc import ler_mono_16k
        x, _ = ler_mono_16k(str(caminho))                 # já em 16 kHz
        r = self.asr({"raw": np.asarray(x, dtype=np.float32), "sampling_rate": 16000}, return_timestamps=True)
        return r["text"].strip()

    def analisar(self, caminho):
        r_voz = self.voz.analisar(caminho)
        if self.modo == "offline":
            texto = self.transcrever(caminho)
            if len(texto.split()) < 4:                    # mesma regra do módulo dos colegas: < 4 palavras = sem fala
                r_ctx = {"status": "audio_sem_fala", "transcricao": texto, "risco_conteudo": "indeterminado",
                         "sinais": [], "resumo": "", "erro": None, "modo": "offline"}
            else:
                r_ctx = contexto.analisar(caminho, "offline", texto)
        else:
            r_ctx = contexto.analisar(caminho, "groq")
        return r_voz, r_ctx, alerta.combinar(r_voz, r_ctx)


class Pagina(SimpleHTTPRequestHandler):
    motor = None

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(MOCKUP), **k)

    def _json(self, codigo, obj):
        corpo = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_POST(self):
        if self.path != "/analisar":
            return self._json(404, {"erro": "rota desconhecida"})
        nome = unquote(self.headers.get("X-Nome", "audio"))
        ext = Path(nome).suffix.lower()
        n = int(self.headers.get("Content-Length", 0))
        if ext not in FORMATOS:
            return self._json(400, {"erro": f"formato {ext or '?'} não aceito; use WAV, MP3, OGG ou FLAC"})
        if not 0 < n <= LIMITE_BYTES:
            return self._json(400, {"erro": "arquivo vazio ou maior que 25 MB"})
        ENVIADOS.mkdir(exist_ok=True)
        arq = ENVIADOS / f"{uuid.uuid4().hex[:12]}{ext}"
        arq.write_bytes(self.rfile.read(n))
        try:
            r_voz, r_ctx, r_alerta = self.motor.analisar(arq)
        except Exception as e:                            # noqa: BLE001  (a página mostra a mensagem)
            return self._json(500, {"erro": f"não foi possível analisar o áudio: {e}"})
        self._json(200, {"id": arq.stem, "titulo": f"Áudio enviado: {nome}", "contato": "Áudio enviado",
                         "hora": time.strftime("%H:%M"), "audio_mockup": f"enviados/{arq.name}",
                         "voz": r_voz, "contexto": r_ctx, "alerta": r_alerta})

    def log_message(self, fmt, *args):                    # sem log de cada arquivo estático
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    for f in (sys.stdout, sys.stderr):
        try:
            f.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contexto", choices=contexto.MODOS, default="offline")
    ap.add_argument("--porta", type=int, default=8765)
    a = ap.parse_args()
    Pagina.motor = Motor(a.contexto)
    srv = HTTPServer(("127.0.0.1", a.porta), Pagina)      # um pedido por vez (os modelos não são divididos)
    print(f"[servidor] abra http://127.0.0.1:{a.porta}  (Ctrl+C para parar)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
