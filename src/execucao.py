"""Utilidades de execução por script (v4): log duplicado em arquivo e a célula de configuração do notebook.

Mesma ideia de scripts/treinar_experimento.py (v3), que não foi alterado: os hiperparâmetros NÃO são
copiados nos scripts; eles executam a célula com a tag "config" do notebook.
"""
import json
from pathlib import Path


class Tee:
    """Escreve no console e num arquivo de log ao mesmo tempo (UTF-8; o console troca o que não sabe mostrar)."""

    def __init__(self, console, arquivo):
        self.console, self.arquivo = console, arquivo

    def write(self, s):
        try:
            if self.console is not None:
                self.console.write(s)
        except (UnicodeEncodeError, OSError, ValueError):
            try:
                enc = getattr(self.console, "encoding", None) or "ascii"
                self.console.write(s.encode(enc, "replace").decode(enc))
            except (OSError, ValueError):
                pass
        self.arquivo.write(s)
        return len(s)

    def isatty(self):                 # bibliotecas (transformers, tqdm) perguntam isso ao stdout
        return False

    @property
    def encoding(self):
        return "utf-8"

    def flush(self):
        try:
            if self.console is not None:
                self.console.flush()
        except (OSError, ValueError):
            pass
        self.arquivo.flush()


def celula_config(notebook):
    cel = [c for c in json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
           if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(cel) == 1, f"esperava 1 célula com a tag 'config' em {notebook}, achei {len(cel)}"
    return "".join(cel[0]["source"])


def executar_config(src, quick, notebook):
    """Executa a célula de configuração com QUICK_DO_SCRIPT = `quick` e devolve o namespace."""
    ns = {"__name__": "config_do_notebook", "QUICK_DO_SCRIPT": quick}
    exec(compile(src, f"{notebook} (célula config)", "exec"), ns)
    assert ns["QUICK"] == quick, "a célula de configuração não respeitou o modo QUICK pedido"
    return ns
