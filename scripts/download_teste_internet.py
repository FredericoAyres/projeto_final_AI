#!/usr/bin/env python3
"""Baixa os dois bancos de TESTE "da internet": RADAR Challenge 2026 (so o eval) e ML-ITW.

Uso (na raiz do repo, com o env ativo):
    python scripts/download_teste_internet.py --dry-run
    python scripts/download_teste_internet.py                       # tudo, em data/raw/teste_internet
    python scripts/download_teste_internet.py --dest D:\\dados\\teste_internet
    python scripts/download_teste_internet.py --only radar

Fontes:
  radar  Zenodo 22110815 (CC BY 4.0): audio do EVAL + rotulos + idiomas do eval (md5 conferido).
         O conjunto de desenvolvimento (dev) NAO e baixado: ele vem do LlamaPartialSpoof e do
         LibriTTS, que se sobrepoem aos dados de treino/teste do projeto.
  mlitw  Hugging Face (dataset aberto, sem login/aceite quando conferido): parquets do split eval.

Retomavel: Zenodo por HTTP Range; Hugging Face pelo retomar do huggingface_hub. Marcador .done por
fonte. O arquivo compactado NAO e apagado, a menos que --apagar-compactado. Nenhum token e lido
ou gravado por este script (o huggingface_hub usa o login ja feito, se houver).

Dependencias: requests, huggingface_hub (e tqdm, opcional).
"""
import argparse
import ctypes
import hashlib
import logging
import os
import re
import shutil
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data" / "raw" / "teste_internet"
LOG_FILE = ROOT / "logs" / "download_teste_internet.log"
GB = 1e9
CHUNK = 1 << 20
RETRIES = 8
SPACE_FACTOR = 1.5          # exige 1,5 x o total a baixar de disco livre
HEADERS = {"User-Agent": "download_teste_internet.py (projeto academico)"}

ZENODO_ID = 22110815
RADAR_FILES = ["RADAR2026-eval.tar.gz", "label_RADAR2026-eval.txt", "lang_RADAR2026-eval.txt",
               "README.txt", "LICENSE.txt"]            # dev NUNCA entra aqui
HF_REPO = "nccm2p2/ML-ITW"   # o enunciado citava chlo1/ML-ITW, que nao existe; este tem os 24.529 exemplos esperados
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}"

log = logging.getLogger("teste_internet")


class Fatal(Exception):
    pass


class Manual(Exception):
    """A fonte exige um passo manual do usuario (login/aceite)."""


class Redact(logging.Filter):
    PAT = re.compile(r"hf_[A-Za-z0-9]{10,}|Bearer\s+\S+")

    def filter(self, rec):
        rec.msg, rec.args = self.PAT.sub("[REDACTED]", rec.getMessage()), ()
        return True


def fmt(b):
    return f"{b / GB:.2f} GB" if b >= 1e8 else f"{b / 1e6:.2f} MB"


def dir_size(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.lstat(os.path.join(r, f)).st_size
            except OSError:
                pass
    return t


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- listagem (sem baixar)
def list_radar():
    r = requests.get(f"https://zenodo.org/api/records/{ZENODO_ID}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    d = r.json()
    by = {f["key"]: f for f in d["files"]}
    miss = [k for k in RADAR_FILES if k not in by]
    if miss:
        raise Fatal(f"arquivos nao encontrados no Zenodo {ZENODO_ID}: {miss}")
    lic = (d["metadata"].get("license") or {}).get("id", "?")
    items = [dict(name=k, size=by[k]["size"], md5=by[k]["checksum"].split(":", 1)[1], url=by[k]["links"]["self"])
             for k in RADAR_FILES]
    return items, lic


def list_mlitw():
    """Lista os arquivos do HF. Levanta Manual se exigir login/aceite."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        info = HfApi().dataset_info(HF_REPO, files_metadata=True)
    except GatedRepoError:
        raise Manual(f"{HF_URL} exige aceite dos termos")
    except RepositoryNotFoundError:
        raise Manual(f"{HF_URL} nao encontrado sem login (inexistente, privado ou exige aceite/login)")
    if info.gated:
        raise Manual(f"{HF_URL} e 'gated' (gated={info.gated}): exige aceite dos termos")
    items = [dict(name=s.rfilename, size=s.size or 0,
                  sha256=(s.lfs.sha256 if getattr(s, "lfs", None) else None))
             for s in info.siblings if s.rfilename.endswith(".parquet") or s.rfilename == "README.md"]
    lic = next((t.split(":", 1)[1] for t in info.tags if t.startswith("license:")), "nao declarada no card")
    return items, lic


# --------------------------------------------------------------------------- download (Zenodo)
def download_http(url, dest, size, desc):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return
        if have > size:
            raise Fatal(f"{dest.name} local ({have} B) maior que o remoto ({size} B); nada foi apagado, verifique")
        try:
            hdr = dict(HEADERS, **({"Range": f"bytes={have}-"} if have else {}))
            with requests.get(url, headers=hdr, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                mode = "ab"
                if have and r.status_code != 206:
                    if size > GB:
                        raise Fatal("servidor ignorou o Range (sem 206); nao vou recomecar um arquivo grande")
                    mode, have = "wb", 0
                if have:
                    log.info("  retomando %s a partir de %s", dest.name, fmt(have))
                bar = tqdm(total=size, initial=have, unit="B", unit_scale=True, desc=desc, mininterval=10,
                           leave=False, file=sys.stderr) if tqdm else None
                n, nxt = have, have + (1 << 30)
                try:
                    with open(dest, mode) as f:
                        for ch in r.iter_content(CHUNK):
                            f.write(ch)
                            n += len(ch)
                            if bar:
                                bar.update(len(ch))
                            elif n >= nxt:
                                nxt += 1 << 30
                                log.info("  %s: %.1f GB", desc, n / GB)
                finally:
                    if bar:
                        bar.close()
        except Fatal:
            raise
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if 400 <= code < 500 and code != 429:
                raise Fatal(f"HTTP {code} em {url.split('?')[0]}")
        except requests.RequestException:
            pass
        if dest.exists() and dest.stat().st_size == size:
            return
        wait = min(120, 10 * 2 ** (attempt - 1))
        log.warning("  %s incompleto (tentativa %d/%d); nova tentativa em %ds", dest.name, attempt, RETRIES, wait)
        time.sleep(wait)
    raise Fatal(f"{dest.name}: falhou apos {RETRIES} tentativas")


def safe_extract_tar(archive, out):
    """Extrai so arquivos e pastas; rejeita caminho absoluto, '..', links e dispositivos."""
    out.mkdir(parents=True, exist_ok=True)
    out_r = out.resolve()
    n = total = 0
    with tarfile.open(archive, "r:*") as tf:
        for m in tf:
            p = PurePosixPath(m.name.replace("\\", "/"))
            if p.is_absolute() or re.match(r"^[A-Za-z]:", m.name) or ".." in p.parts:
                raise Fatal(f"caminho inseguro no tar: {m.name!r}; extracao abortada")
            if not (m.isreg() or m.isdir()):
                raise Fatal(f"membro nao regular no tar (link/dispositivo): {m.name!r}; extracao abortada")
            target = (out / m.name).resolve()
            if out_r != target and out_r not in target.parents:
                raise Fatal(f"caminho sai da pasta de destino: {m.name!r}")
            if m.isreg() and target.exists() and target.stat().st_size == m.size:
                n, total = n + 1, total + m.size           # retomada: ja extraido
                continue
            tf.extract(m, out, filter="data")
            if m.isreg():
                if target.stat().st_size != m.size:
                    raise Fatal(f"tamanho diferente apos extrair {m.name}")
                n, total = n + 1, total + m.size
                if n % 20000 == 0:
                    log.info("  extraidos %d arquivos (%s)", n, fmt(total))
    if n == 0:
        raise Fatal("tar sem arquivos")
    log.info("  extraido %s: %d arquivos, %s", archive.name, n, fmt(total))
    return n


# --------------------------------------------------------------------------- fontes
def run_radar(items, target, args):
    target.mkdir(parents=True, exist_ok=True)
    for it in items:
        dest = target / it["name"]
        log.info("  [radar] %s (%s)", it["name"], fmt(it["size"]))
        download_http(it["url"], dest, it["size"], it["name"])
        ok = target / f".md5_ok_{it['name']}"
        if not ok.exists():
            if md5_of(dest) != it["md5"]:
                raise Fatal(f"md5 de {it['name']} nao confere com o Zenodo (arquivo mantido para inspecao)")
            ok.write_text(it["md5"], encoding="utf-8")
            log.info("  md5 ok: %s", it["name"])
    tar = target / "RADAR2026-eval.tar.gz"
    if not (target / ".extraido").exists():
        safe_extract_tar(tar, target)
        (target / ".extraido").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    if args.apagar_compactado:
        tar.unlink()
        log.info("  removido %s (--apagar-compactado)", tar.name)
    (target / ".done").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def run_mlitw(items, target, args):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    target.mkdir(parents=True, exist_ok=True)
    for it in items:
        dest = target / it["name"]
        log.info("  [mlitw] %s (%s)", it["name"], fmt(it["size"]))
        for attempt in range(1, RETRIES + 1):
            try:
                hf_hub_download(HF_REPO, it["name"], repo_type="dataset", local_dir=str(target))
                break
            except (GatedRepoError, RepositoryNotFoundError) as e:
                raise Manual(f"acesso negado em {HF_URL} ({type(e).__name__})")
            except Exception as e:
                if attempt == RETRIES:
                    raise Fatal(f"{it['name']}: falhou apos {RETRIES} tentativas ({type(e).__name__})")
                wait = min(120, 10 * 2 ** (attempt - 1))
                log.warning("  %s falhou (%s); nova tentativa %d/%d em %ds", it["name"], type(e).__name__,
                            attempt, RETRIES, wait)
                time.sleep(wait)
        if dest.stat().st_size != it["size"]:
            raise Fatal(f"{it['name']}: tamanho {dest.stat().st_size} != {it['size']}")
        ok = target / f".sha256_ok_{Path(it['name']).name}"
        if it["sha256"] and not ok.exists():
            if sha256_of(dest) != it["sha256"]:
                raise Fatal(f"sha256 de {it['name']} nao confere com o HF")
            ok.write_text(it["sha256"], encoding="utf-8")
    (target / ".done").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


# --------------------------------------------------------------------------- orquestracao
def pending_bytes(items, target):
    """Bytes que ainda faltam baixar (parcial conta pelo que ja existe)."""
    t = 0
    for it in items:
        p = target / it["name"]
        t += it["size"] - (min(p.stat().st_size, it["size"]) if p.exists() else 0)
    return t


def keep_awake(on):
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="pasta de destino (padrao data/raw/teste_internet)")
    ap.add_argument("--only", choices=["radar", "mlitw", "all"], default="all")
    ap.add_argument("--max-gb", type=float, default=40, help="teto do total a baixar (GB, padrao 40)")
    ap.add_argument("--dry-run", action="store_true", help="lista o que faria, sem baixar nem gravar nada")
    ap.add_argument("--apagar-compactado", action="store_true", help="apaga o .tar.gz do RADAR depois de extrair")
    args = ap.parse_args()
    dest = args.dest.resolve()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.dry_run:
        LOG_FILE.parent.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)
    for h in handlers:
        h.addFilter(Redact())

    sources = {"radar": ("radar_eval", list_radar, run_radar), "mlitw": ("ml_itw", list_mlitw, run_mlitw)}
    chosen = ["radar", "mlitw"] if args.only == "all" else [args.only]
    t0 = time.time()
    log.info("destino: %s", dest)

    # 1) listar (sem baixar)
    plan, manual, status = {}, [], {}
    for s in chosen:
        sub, lister, _ = sources[s]
        if (dest / sub / ".done").exists():
            log.info("[%s] ja concluido (.done); pulando", s)
            status[s] = "ja concluido"
            continue
        try:
            items, lic = lister()
            plan[s] = (items, lic, dest / sub)
        except Manual as e:
            manual.append((s, str(e)))
            status[s] = "MANUAL"
            log.warning("[%s] exige passo manual: %s", s, e)
        except Exception as e:
            status[s] = f"falhou na listagem: {type(e).__name__}: {e}"
            log.error("[%s] listagem falhou: %s: %s", s, type(e).__name__, e)

    log.info("")
    log.info("%-6s %-45s %12s  %s", "fonte", "arquivo", "tamanho", "licenca")
    total = falta = 0
    for s, (items, lic, target) in plan.items():
        for it in items:
            log.info("%-6s %-45s %12s  %s", s, it["name"], fmt(it["size"]), lic)
        total += sum(i["size"] for i in items)
        falta += pending_bytes(items, target)
    log.info("TOTAL a baixar (nominal): %s | ja presente/parcial: %s | falta: %s (teto --max-gb %s)",
             fmt(total), fmt(total - falta), fmt(falta), args.max_gb)

    if total > args.max_gb * GB:
        log.error("total %s passa de --max-gb %s: parei; aumente --max-gb se quiser continuar", fmt(total), args.max_gb)
        sys.exit(2)
    probe = dest if dest.exists() else next(p for p in [dest, *dest.parents] if p.exists())
    free = shutil.disk_usage(probe).free
    need = SPACE_FACTOR * total
    log.info("disco livre em %s: %s | exigido (%.1f x total): %s", probe, fmt(free), SPACE_FACTOR, fmt(need))
    if plan and free < need:
        log.error("disco livre insuficiente; parei")
        sys.exit(2)
    if args.dry_run:
        log.info("DRY-RUN: nada foi baixado ou gravado.")
    else:
        keep_awake(True)
        try:
            for s, (items, lic, target) in plan.items():
                log.info("[%s] baixando para %s", s, target)
                try:
                    sources[s][2](items, target, args)
                    status[s] = f"ok ({fmt(dir_size(target))} em disco)"
                    log.info("[%s] concluido", s)
                except Manual as e:
                    manual.append((s, str(e)))
                    status[s] = "MANUAL"
                    log.warning("[%s] exige passo manual: %s", s, e)
                except KeyboardInterrupt:
                    log.warning("interrompido; reexecute para retomar")
                    raise
                except Exception as e:
                    status[s] = f"FALHOU: {type(e).__name__}: {e}"
                    log.error("[%s] FALHOU: %s: %s (reexecute para retomar)", s, type(e).__name__, e)
        finally:
            keep_awake(False)

    log.info("")
    log.info("RESUMO (%.0f s)", time.time() - t0)
    for s in chosen:
        log.info("  %-6s %s", s, status.get(s, "(dry-run)"))
    if manual:
        log.info("FAZER MANUALMENTE:")
        for s, why in manual:
            log.info("  - %s: %s", s, why)
            if s == "mlitw":
                log.info("      1) abra %s no navegador, entre na sua conta e aceite os termos;", HF_URL)
                log.info("      2) no terminal: huggingface-cli login   (ou: hf auth login), cole o token voce mesmo;")
                log.info("      3) rode de novo: python scripts/download_teste_internet.py --only mlitw")


if __name__ == "__main__":
    main()
