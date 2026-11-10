"""Gravação atômica de arquivos e identificação de dados e código (execução; não muda nenhum resultado).

Gravação atômica: escreve num arquivo temporário na mesma pasta e só então o renomeia para o nome
final (`os.replace`). Se o processo cair no meio, o arquivo final ou não existe ou é a versão
anterior inteira; nunca fica pela metade.
"""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch


def _temporario(destino):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{destino.name}.", suffix=".tmp", dir=destino.parent)
    os.close(fd)
    return Path(tmp)


def _trocar(tmp, destino):
    try:
        os.replace(tmp, destino)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def gravar_atomico(destino, escrever):
    """Chama `escrever(caminho_temporario)` e renomeia o temporário para `destino`."""
    tmp = _temporario(destino)
    try:
        escrever(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _trocar(tmp, destino)


def salvar_torch(obj, destino):
    gravar_atomico(destino, lambda t: torch.save(obj, t))


def salvar_csv(df, destino):
    gravar_atomico(destino, lambda t: df.to_csv(t, index=False))


def salvar_parquet(df, destino):
    gravar_atomico(destino, lambda t: df.to_parquet(t, index=False))


def salvar_json(obj, destino):
    texto = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    gravar_atomico(destino, lambda t: Path(t).write_text(texto, encoding="utf-8"))


def sha256_arquivo(caminho, bloco=1 << 20):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while b := f.read(bloco):
            h.update(b)
    return h.hexdigest()


def sha256_texto(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def id_config(config):
    """Identificador curto (16 hex) de um dicionário de configuração (ordem das chaves não importa)."""
    return sha256_texto(json.dumps(config, sort_keys=True, ensure_ascii=False, default=str))[:16]


def identificar_codigo(raiz, extras=()):
    """Commit do git, arquivos com mudança não commitada e o sha256 de cada arquivo de código usado."""
    raiz = Path(raiz)

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=raiz, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    arquivos = sorted((raiz / "src").glob("*.py")) + [raiz / e for e in extras]
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "mudancas_nao_commitadas": [l for l in (git("status", "--porcelain") or "").splitlines() if l],
            "sha256": {str(a.relative_to(raiz)).replace("\\", "/"): sha256_arquivo(a)
                       for a in arquivos if a.exists()}}
