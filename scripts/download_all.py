#!/usr/bin/env python3
"""Baixa, de forma retomavel e segura, os datasets do projeto para data/raw/.

Uso:
    python scripts/download_all.py --dry-run
    python scripts/download_all.py --max-gb 100
    python scripts/download_all.py --only asvspoof2019_la,librispeech

Dependencias: requests (obrigatoria); tqdm e huggingface_hub (opcionais:
barra de progresso e leitura do token do `hf auth login`, nunca impresso).
    pip install requests tqdm huggingface_hub

Todos os downloads (HF, Zenodo, Edinburgh, OpenSLR) usam HTTP Range, entao
retomam de onde pararam. Nada e apagado alem do arquivo compactado, e apenas
depois de extraido e verificado.
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
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

try:
    from tqdm import tqdm
except ImportError:  # barra de progresso e opcional
    tqdm = None

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LOGS = ROOT / "logs"
GB = 1e9
RESERVE = 5 * GB          # folga minima de disco livre
CHUNK = 1 << 20
RETRIES = 8
HEADERS = {"User-Agent": "download_all.py (projeto academico)"}

log = logging.getLogger("download")
SESSION = requests.Session()
MANUAL = []               # (dataset, motivo, link) -> "FAZER MANUALMENTE"
SPACE_SKIPPED = []        # (dataset, motivo)


# --------------------------------------------------------------------------- erros
class Skip(Exception):
    """Dataset pulado. manual=True -> entra em FAZER MANUALMENTE."""

    def __init__(self, reason, manual=True, link=None):
        super().__init__(reason)
        self.reason, self.manual, self.link = reason, manual, link


class AccessDenied(Exception):
    """HTTP 401/403 (login, aceite ou repo gated)."""


class Fatal(Exception):
    """Erro que nao adianta repetir."""


class Redact(logging.Filter):
    """Garantia extra: nunca deixar um token chegar ao log."""
    PAT = re.compile(r"hf_[A-Za-z0-9]{10,}|Bearer\s+\S+")

    def filter(self, rec):
        rec.msg, rec.args = self.PAT.sub("[REDACTED]", rec.getMessage()), ()
        return True


# --------------------------------------------------------------------------- util
def fmt(b):
    return f"{b / GB:.2f} GB"


def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def stem(name):
    return re.sub(r"\.(tar\.(gz|xz|bz2)|tgz|zip|tar)$", "", name)


_token = False


def hf_token():
    """Token do login do HF (so usado em huggingface.co; nunca impresso)."""
    global _token
    if _token is False:
        try:
            from huggingface_hub import get_token
            _token = get_token()
        except Exception:
            _token = None
    return _token


def get(url, headers=None, timeout=(30, 120), **kw):
    h = dict(HEADERS)
    if urlparse(url).hostname == "huggingface.co" and hf_token():
        h["Authorization"] = f"Bearer {hf_token()}"   # requests remove no redirect p/ outro host
    h.update(headers or {})
    r = SESSION.get(url, headers=h, timeout=timeout, **kw)
    if r.status_code in (401, 403):
        r.close()
        raise AccessDenied(f"HTTP {r.status_code} em {url.split('?')[0]}")
    return r


def probe(url):
    """Tamanho total do arquivo (GET Range 0-0, le 1 byte)."""
    with get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=30) as r:
        r.raise_for_status()
        cr = r.headers.get("Content-Range")
        if r.status_code == 206 and cr:
            return int(cr.rsplit("/", 1)[1])
        return int(r.headers["Content-Length"])


@dataclass
class Item:
    key: str                    # id unico dentro do dataset
    url: str
    dest: str                   # caminho relativo ao alvo
    size: int
    md5: str = None
    extract: bool = False
    extract_to: str = None      # relativo ao alvo (None = o proprio alvo)


# --------------------------------------------------------------------------- listagens
def hf_files(repo):
    """{caminho: tamanho} de um dataset do HF (API oficial, paginada)."""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    out = {}
    while url:
        r = get(url, timeout=60)
        if r.status_code == 404:
            raise Skip(f"repositorio {repo} nao encontrado no HF", link=f"https://huggingface.co/datasets/{repo}")
        r.raise_for_status()
        for e in r.json():
            if e.get("type") == "file":
                out[e["path"]] = e["size"]
        url = r.links.get("next", {}).get("url")
    return out


def hf_item(repo, path, size, **kw):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(path)}"
    flat = kw.pop("flat", False)          # arquivos compactados vao direto na raiz do alvo
    return Item(key=path, url=url, dest=Path(path).name if flat else path, size=size, **kw)


def zenodo_files(rec_id):
    r = get(f"https://zenodo.org/api/records/{rec_id}", timeout=60)
    r.raise_for_status()
    d = r.json()
    files = {f["key"]: dict(size=f["size"], md5=f["checksum"].split(":", 1)[1], url=f["links"]["self"])
             for f in d["files"]}
    return d, files


def zen_item(name, f, **kw):
    return Item(key=name, url=f["url"], dest=name, size=f["size"], md5=f["md5"], **kw)


def cap(items, cap_gb, name):
    tot = sum(i.size for i in items)
    if tot > cap_gb * GB:
        raise Skip(f"selecao de {name} ({fmt(tot)}) excede o teto de {cap_gb} GB", manual=False)
    return items


# --------------------------------------------------------------------------- resolvers
def r_asvspoof2019_la():
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
    return [Item("LA.zip", url, "LA.zip", probe(url), extract=True)]


def r_librispeech():
    page = get("https://www.openslr.org/12/", timeout=60)
    page.raise_for_status()
    items = []
    for n in ("train-clean-100", "dev-clean", "test-clean"):
        cands = list(dict.fromkeys(re.findall(r'href="(https?://[^"]*/resources/12/%s\.tar\.gz)"' % n, page.text)))
        found = None
        for u in cands:
            try:
                found = (u, probe(u))
                break
            except Exception as e:
                log.warning("librispeech: mirror %s falhou (%s)", urlparse(u).hostname, type(e).__name__)
        if not found:
            raise Skip(f"link de {n} nao confirmado em https://www.openslr.org/12", link="https://www.openslr.org/12")
        items.append(Item(f"{n}.tar.gz", found[0], f"{n}.tar.gz", found[1], extract=True))
    return items


def r_in_the_wild():
    repo, f = "mueller91/In-The-Wild", "release_in_the_wild.zip"
    files = hf_files(repo)
    if f not in files:
        raise Skip(f"{f} nao esta no repo {repo}", link=f"https://huggingface.co/datasets/{repo}")
    return [hf_item(repo, f, files[f], extract=True, flat=True)]


def r_brspeech_df():
    repo = "AKCIT-Deepfake/BRSpeech-DF"
    files = hf_files(repo)
    # config 'default' (bonafide + spoof juntos), split test. spoof/ e bonafide/ duplicam o mesmo conteudo.
    sel = sorted(p for p in files if re.fullmatch(r"data/test-\d{5}-of-\d{5}\.parquet", p))
    if not sel:
        raise Skip("shards data/test-*.parquet nao encontrados", link=f"https://huggingface.co/datasets/{repo}")
    log.info("brspeech_df: split test = %d shards (data/test-*), %s", len(sel), fmt(sum(files[p] for p in sel)))
    return [hf_item(repo, p, files[p]) for p in sel]


def r_dfadd():
    repo = "isjwdu/DFADD"
    files = hf_files(repo)
    zips = sorted(p for p in files if re.fullmatch(r"DATASET_[^/]+\.zip", p))
    if not any("VCTK_BONAFIDE" in p for p in zips) or len(zips) < 2:
        raise Skip(f"zips de falsos/VCTK_BONAFIDE nao confirmados em {repo}", link=f"https://huggingface.co/datasets/{repo}")
    log.info("dfadd: %s", ", ".join(zips))
    # cada zip vai para a sua propria subpasta (evita colisao de nomes entre subconjuntos)
    return [hf_item(repo, p, files[p], extract=True, flat=True, extract_to=stem(p)) for p in zips]


def r_asvspoof2021_df():
    d, files = zenodo_files(4835108)
    parts = sorted(k for k in files if re.fullmatch(r"ASVspoof2021_DF_eval_part\d+\.tar\.gz", k))
    if not parts:
        raise Skip("partes DF nao encontradas no Zenodo 4835108", link="https://zenodo.org/records/4835108")
    part = parts[0]   # UMA das 4 partes
    log.info("asvspoof2021_df: parte escolhida = %s (%s)", part, fmt(files[part]["size"]))
    keys = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
    return [zen_item(part, files[part], extract=True),
            Item("DF-keys-full.tar.gz", keys, "DF-keys-full.tar.gz", probe(keys), extract=True)]


def r_llamapartialspoof():
    try:   # DOI 10.5281/zenodo.14214149 -> registro Zenodo 14214149
        d, files = zenodo_files(14214149)
        log.info("llamapartialspoof: Zenodo %s", d.get("doi", "?"))
        return [zen_item(k, f, extract=k.endswith(".tgz")) for k, f in sorted(files.items())]
    except (requests.RequestException, KeyError) as e:
        log.warning("llamapartialspoof: Zenodo falhou (%s); usando espelho HF", type(e).__name__)
    repo = "HaoY0001/LlamaPartialSpoof"
    files = hf_files(repo)
    keep = [p for p in files if "/" not in p and p != ".gitattributes"]
    return [hf_item(repo, p, files[p], extract=p.endswith(".tgz"), flat=True) for p in sorted(keep)]


def r_shiftyspeech():
    repo = "ash56/ShiftySpeech"
    files = hf_files(repo)
    # 3 familias de vocoder (GAN / difusao / Fourier-iSTFT), dominio 'audiobook' (ingles) + real do mesmo dominio
    wanted = [f"Vocoders/{v}/{v}_audiobook_flac.tar.gz" for v in ("hifigan", "wavegrad", "vocos")]
    wanted.append("real_data_flac/real_data_audiobook_flac.tar.gz")
    miss = [w for w in wanted if w not in files]
    if miss:
        raise Skip(f"arquivos nao encontrados no repo: {miss}", link=f"https://huggingface.co/datasets/{repo}")
    log.info("shiftyspeech: escolhidos -> hifigan (GAN), wavegrad (difusao), vocos (Fourier/iSTFT) "
             "[audiobook] + real_data_audiobook")
    items = [hf_item(repo, w, files[w], extract=True, flat=True, extract_to=stem(Path(w).name)) for w in wanted]
    return cap(items, 25, "shiftyspeech")


def r_codecfake_plus():
    repo = "CodecFake/CodecFake_Plus_Dataset"
    files = hf_files(repo)
    wanted = ["CodecFake_plus_CoSG.tar.xz", "CoSG_labels.txt", "CoRS_labels.txt"]
    miss = [w for w in wanted if w not in files]
    if miss:
        raise Skip(f"arquivos nao encontrados: {miss}", link=f"https://huggingface.co/datasets/{repo}")
    parts = sorted(p for p in files if p.startswith("Codecfake_plus_CoRS.part"))
    MANUAL.append(("codecfake_plus (CoRS)",
                   f"CoRS e um unico .tar.xz dividido em {len(parts)} partes de ~25 GB: nenhuma parte sozinha e extraivel "
                   f"e uma parte ja estoura o teto de 25 GB. Baixados: CoSG + labels. Para o CoRS, baixe todas as partes e "
                   f"junte com `cat`.", f"https://huggingface.co/datasets/{repo}"))
    log.info("codecfake_plus: escolhidos -> CoSG (tar.xz) + CoSG_labels + CoRS_labels")
    items = [hf_item(repo, w, files[w], flat=True, extract=w.endswith(".xz")) for w in wanted]
    return cap(items, 25, "codecfake_plus")


def r_parlaspoof_br():
    repo = "freds0/ParlaSpoof-BR"
    files = hf_files(repo)      # 404/401/403 -> pulado (inexistente ou gated)
    shards = [p for p in files if p.startswith("data/") and p.endswith(".parquet")]
    raise Skip(f"repo acessivel, mas os {len(shards)} shards ({fmt(sum(files[p] for p in shards))}) sao todos 'train-*' e "
               f"misturam os subconjuntos (coluna 'subset' so dentro do parquet): nao da para separar o conjunto de "
               f"avaliacao por arquivo. Dica: metadata/metadata_complete.parquet (87 MB) lista o subset de cada audio.",
               link=f"https://huggingface.co/datasets/{repo}")


def r_sonar():
    link = "https://github.com/Jessegator/SONAR"
    try:
        txt = get("https://raw.githubusercontent.com/Jessegator/SONAR/master/README.md", timeout=30).text
        m = re.search(r"https://drive\.google\.com/drive/folders/[^\s)]+", txt)
        link = m.group(0) if m else link
    except Exception:
        pass
    raise Skip("README aponta so para uma pasta do Google Drive (sem URL direta de arquivo confirmavel); "
               "baixar pelo navegador", link=link)


def manual(reason, link=None):
    def f():
        raise Skip(reason, link=link)
    return f


def r_coraa():
    repo = "gabrielrstan/CORAA-v1.1"
    files = hf_files(repo)
    wanted = ["test.zip", "metadata_test_final.csv"]   # split de teste inteiro (~2.4 GB) < teto de 5 GB
    miss = [w for w in wanted if w not in files]
    if miss:
        raise Skip(f"arquivos nao encontrados: {miss}", link=f"https://huggingface.co/datasets/{repo}")
    items = [hf_item(repo, w, files[w], flat=True, extract=w.endswith(".zip")) for w in wanted]
    return cap(items, 5, "coraa")


# ordem pedida pelo usuario
DATASETS = [
    ("asvspoof2019_la", "train_val_en/asvspoof2019_la", r_asvspoof2019_la),
    ("librispeech", "real_en/librispeech", r_librispeech),
    ("in_the_wild", "test_generalization/in_the_wild", r_in_the_wild),
    ("brspeech_df", "test_ptbr/brspeech_df", r_brspeech_df),
    ("dfadd", "train_val_en/dfadd", r_dfadd),
    ("asvspoof2021_df", "test_generalization/asvspoof2021_df", r_asvspoof2021_df),
    ("llamapartialspoof", "test_generalization/llamapartialspoof", r_llamapartialspoof),
    ("shiftyspeech", "train_val_en/shiftyspeech", r_shiftyspeech),
    ("codecfake_plus", "train_val_en/codecfake_plus", r_codecfake_plus),
    ("parlaspoof_br", "test_ptbr/parlaspoof_br", r_parlaspoof_br),
    ("sonar", "test_generalization/sonar", r_sonar),
    ("coraa", "test_ptbr/coraa", r_coraa),
    # nunca baixados por este script
    ("spoofceleb", "test_generalization/spoofceleb",
     manual("gated: exige aceite e e-mail institucional", "https://huggingface.co/datasets/jungjee/spoofceleb")),
    ("mlaad_pt", "test_ptbr/mlaad_pt", manual("pulado por decisao do projeto (so a pasta fake/pt)")),
    ("fakebraccent", "test_ptbr/fakebraccent", manual("pulado por decisao do projeto")),
    ("vctk", "real_en/vctk", manual("URL a confirmar no Edinburgh DataShare", "https://datashare.ed.ac.uk/")),
    ("common_voice", None, manual("exige login e aceite de termos", "https://commonvoice.mozilla.org/datasets")),
    ("voxceleb", None, manual("exige cadastro e aceite de termos", "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")),
]


# --------------------------------------------------------------------------- download
class Bar:
    def __init__(self, total, initial, desc):
        self.n, self.next = initial, initial + (1 << 30)
        self.bar = tqdm(total=total, initial=initial, unit="B", unit_scale=True, unit_divisor=1024,
                        desc=desc, mininterval=10, leave=False, file=sys.stderr) if tqdm else None
        self.desc = desc

    def update(self, k):
        self.n += k
        if self.bar:
            self.bar.update(k)
        elif self.n >= self.next:
            self.next += 1 << 30
            log.info("  %s: %.1f GB", self.desc, self.n / GB)

    def close(self):
        if self.bar:
            self.bar.close()


def download(item, target):
    dest = target / item.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, RETRIES + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == item.size:
            return
        if have > item.size:
            raise Fatal(f"{dest.name} local ({have} B) maior que o remoto ({item.size} B); nada foi apagado, verifique")
        try:
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with get(item.url, headers=hdr, stream=True) as r:
                if r.status_code == 404:
                    raise Fatal(f"404 em {item.url.split('?')[0]}")
                r.raise_for_status()
                mode = "ab"
                if have and r.status_code != 206:
                    if item.size > GB:
                        raise Fatal("servidor ignorou o Range (sem 206); nao vou recomecar um arquivo grande")
                    mode, have = "wb", 0
                if have:
                    log.info("  retomando %s a partir de %s", dest.name, fmt(have))
                bar = Bar(item.size, have, dest.name)
                try:
                    with open(dest, mode) as f:
                        for chunk in r.iter_content(CHUNK):
                            f.write(chunk)
                            bar.update(len(chunk))
                finally:
                    bar.close()
        except (Fatal, AccessDenied):
            raise
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if 400 <= code < 500 and code != 429:
                raise Fatal(f"HTTP {code} em {item.url.split('?')[0]}")
            last = e
        except requests.RequestException as e:    # queda de rede, timeout, conexao cortada
            last = e
        if dest.exists() and dest.stat().st_size == item.size:
            return
        wait = min(120, 10 * 2 ** (attempt - 1))
        log.warning("  %s incompleto (tentativa %d/%d, %s); nova tentativa em %ds",
                    dest.name, attempt, RETRIES, type(last).__name__ if last else "stream curto", wait)
        time.sleep(wait)
    raise Fatal(f"{dest.name}: falhou apos {RETRIES} tentativas")


def md5_ok(path, expected):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest() == expected


def extract_and_verify(archive, out):
    """Extrai com checagem de caminho e confere o tamanho de cada arquivo extraido."""
    out.mkdir(parents=True, exist_ok=True)
    total = n = 0
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for i in zf.infolist():
                if i.is_dir():
                    zf.extract(i, out)
                    continue
                p = Path(zf.extract(i, out))
                if p.stat().st_size != i.file_size:
                    raise Fatal(f"tamanho diferente apos extrair {i.filename}")
                total, n = total + i.file_size, n + 1
    else:
        with tarfile.open(archive, "r:*") as tf:
            for m in tf:
                tf.extract(m, out, filter="data")      # bloqueia path traversal/links perigosos
                if m.isreg():
                    p = out / m.name
                    if not p.exists() or p.stat().st_size != m.size:
                        raise Fatal(f"tamanho diferente apos extrair {m.name}")
                    total, n = total + m.size, n + 1
    if n == 0 or total < 0.5 * archive.stat().st_size:
        raise Fatal(f"extracao suspeita: {n} arquivos, {fmt(total)} (arquivo compactado: {fmt(archive.stat().st_size)})")
    log.info("  extraido %s: %d arquivos, %s", archive.name, n, fmt(total))


def load_done(target):
    f = target / ".items_done"
    return set(f.read_text(encoding="utf-8").splitlines()) if f.exists() else set()


def process(item, target):
    log.info("  [%s] %s (%s)", target.name, item.key, fmt(item.size))
    download(item, target)
    dest = target / item.dest
    if item.md5 and not md5_ok(dest, item.md5):
        raise Fatal(f"md5 de {dest.name} nao confere com o Zenodo (arquivo mantido para inspecao)")
    if item.extract:
        extract_and_verify(dest, target / item.extract_to if item.extract_to else target)
        dest.unlink()                                    # unico apagamento: o compactado ja verificado
        log.info("  removido %s", dest.name)
    with open(target / ".items_done", "a", encoding="utf-8") as f:
        f.write(item.key + "\n")


# --------------------------------------------------------------------------- orquestracao
def check_space(name, pending, target, max_gb, sim_used, sim_free_used):
    remaining = peak = 0
    for it in pending:
        p = target / it.dest
        remaining += it.size - (p.stat().st_size if p.exists() else 0)
    peak = max((it.size for it in pending if it.extract), default=0)   # arquivo + conteudo extraido ao mesmo tempo
    need = remaining + peak
    used = dir_size(RAW) + sim_used
    free = shutil.disk_usage(RAW).free - sim_free_used
    if used + need > max_gb * GB:
        raise Skip(f"nao cabe no limite: usado {fmt(used)} + necessario {fmt(need)} > {max_gb} GB", manual=False)
    if free < need + RESERVE:
        raise Skip(f"disco livre insuficiente: livre {fmt(free)} < necessario {fmt(need)} + folga {fmt(RESERVE)}", manual=False)
    return remaining


def run_dataset(name, rel, resolver, args, state):
    target = RAW / rel if rel else None
    if target and (target / ".done").exists():
        log.info("[%s] ja concluido (.done); pulando", name)
        return (name, "ok", dir_size(target), "ja concluido (.done)")
    try:
        items = resolver()
        if target is None:
            raise Skip("sem pasta de destino")
        done = load_done(target)
        pending = [i for i in items if i.key not in done]
        remaining = check_space(name, pending, target, args.max_gb, state["sim_used"], state["sim_used"])
        if args.dry_run:
            log.info("[%s] DRY-RUN: %d arquivo(s) pendente(s), baixar %s -> %s", name, len(pending), fmt(remaining), rel)
            for it in pending:
                p = target / it.dest
                have = p.stat().st_size if p.exists() else 0
                log.info("    - %-48s %9s%s%s", it.key, fmt(it.size),
                         f"  (retoma de {fmt(have)})" if have else "",
                         f"  [extrai{' em ' + it.extract_to + '/' if it.extract_to else ''}, apaga compactado]" if it.extract else "")
                log.info("        %s", it.url.split("?")[0])
            state["sim_used"] += remaining
            return (name, "previsto", dir_size(target) if target.exists() else 0, f"baixaria {fmt(remaining)}")
        target.mkdir(parents=True, exist_ok=True)
        failed = []
        for it in pending:
            try:
                process(it, target)
            except (Skip, AccessDenied):
                raise
            except Exception as e:
                log.error("  FALHOU %s: %s: %s", it.key, type(e).__name__, e)
                failed.append(it.key)
        if failed:
            return (name, "falhou", dir_size(target), f"{len(failed)} arquivo(s): {', '.join(failed)} (reexecute para retomar)")
        (target / ".done").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        log.info("[%s] concluido", name)
        return (name, "ok", dir_size(target), f"{len(items)} arquivo(s)")
    except Skip as s:
        log.warning("[%s] PULADO: %s", name, s.reason)
        (MANUAL if s.manual else SPACE_SKIPPED).append((name, s.reason, s.link) if s.manual else (name, s.reason))
        return (name, "pulado", dir_size(target) if target and target.exists() else 0, s.reason)
    except AccessDenied as e:
        log.warning("[%s] PULADO (acesso negado/login/aceite): %s", name, e)
        MANUAL.append((name, f"acesso negado (login/aceite/gated): {e}", None))
        return (name, "pulado", dir_size(target) if target and target.exists() else 0, "acesso negado")
    except Exception as e:
        log.error("[%s] FALHOU: %s: %s", name, type(e).__name__, e)
        return (name, "falhou", dir_size(target) if target and target.exists() else 0, f"{type(e).__name__}: {e}")


def build_summary(rows, args):
    w = max(len(r[0]) for r in rows)
    lines = [f"RESUMO {'(DRY-RUN) ' if args.dry_run else ''}- {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             f"{'dataset':<{w}}  {'status':<9}  {'em disco':>10}  detalhe", "-" * (w + 40)]
    for n, st, sz, det in rows:
        lines.append(f"{n:<{w}}  {st:<9}  {fmt(sz):>10}  {det[:110]}")
    lines += ["", f"Total em data/raw: {fmt(dir_size(RAW))} (limite --max-gb {args.max_gb})", ""]
    if SPACE_SKIPPED:
        lines.append("PULADOS POR ESPACO (reexecute com --max-gb maior ou libere disco):")
        lines += [f"  - {n}: {r}" for n, r in SPACE_SKIPPED] + [""]
    lines.append("FAZER MANUALMENTE:")
    seen = set()
    for n, r, link in MANUAL:
        if (n, r) not in seen:
            seen.add((n, r))
            lines.append(f"  - {n}: {r}" + (f"\n      {link}" if link else ""))
    return "\n".join(lines)


def keep_awake(on):
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)


def main():
    names = [d[0] for d in DATASETS]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="lista separada por virgula: " + ",".join(names))
    ap.add_argument("--dry-run", action="store_true", help="lista o que faria, sem baixar nem gravar nada")
    ap.add_argument("--max-gb", type=float, default=100, help="limite total em data/raw (GB, padrao 100)")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    if only and (bad := [s for s in only if s not in names]):
        ap.error(f"dataset(s) desconhecido(s): {bad}. Validos: {names}")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.dry_run:
        LOGS.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(LOGS / "download.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)
    for h in handlers:
        h.addFilter(Redact())

    if not args.dry_run:
        keep_awake(True)
    rows, state = [], {"sim_used": 0}
    try:
        for name, rel, res in DATASETS:
            if only and name not in only:
                continue
            rows.append(run_dataset(name, rel, res, args, state))
    except KeyboardInterrupt:
        log.warning("interrompido pelo usuario; reexecute para retomar")
    finally:
        if not args.dry_run:
            keep_awake(False)
        if rows:
            text = build_summary(rows, args)
            print("\n" + text)
            if not args.dry_run:
                (LOGS / "download_summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
