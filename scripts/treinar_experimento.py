#!/usr/bin/env python3
"""Treina UM experimento do notebook atual num processo próprio (toda a memória volta ao terminar).

Os hiperparâmetros NÃO são copiados aqui: o script executa a célula de configuração do
notebook.ipynb (a célula com a tag "config", seção 2) e usa o mesmo código de src/experimento.py
que o notebook usa. Depois, o notebook (com REUSAR = True) só lê o checkpoint e o histórico.

Uso (na raiz do repo, com o env ativo):
    python scripts/treinar_experimento.py --exp v1r
    python scripts/treinar_experimento.py --exp v1r --quick     # teste rápido (QUICK = True)

Se o checkpoint E o histórico do experimento já existem, não treina de novo (como REUSAR no
notebook). Um checkpoint sem histórico é de um treino interrompido e é sobrescrito.
"""
import os

# Antes de importar numpy: o OpenBLAS reserva ~675 MB por processo com as 22 threads do PC.
# O processo principal não usa álgebra linear na CPU (o treino roda na GPU).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.memoria import GB, Amostrador  # noqa: E402


def celula_config(notebook):
    cel = [c for c in json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
           if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(cel) == 1, f"esperava 1 célula com a tag 'config' em {notebook}, achei {len(cel)}"
    return "".join(cel[0]["source"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="id do experimento (chave de EXPERIMENTOS no notebook)")
    ap.add_argument("--quick", action="store_true", help="roda com QUICK = True")
    ap.add_argument("--notebook", default="notebook.ipynb")
    a = ap.parse_args()

    amostrador = Amostrador()
    amostrador.start()
    t0 = time.time()
    src = celula_config(a.notebook)
    if a.quick:
        assert "QUICK = False" in src
        src = src.replace("QUICK = False", "QUICK = True", 1)
    ns = {"__name__": "config_do_notebook"}
    exec(compile(src, f"{a.notebook} (célula config)", "exec"), ns)   # define HP, EXPERIMENTOS, caminhos...
    if a.exp not in ns["EXPERIMENTOS"]:
        sys.exit(f"experimento '{a.exp}' não existe; opções: {list(ns['EXPERIMENTOS'])}")

    ckpt, hist_csv = ns["caminho_ckpt"](a.exp), ns["caminho_hist"](a.exp)
    if ckpt.exists() and hist_csv.exists():
        print(f"{a.exp}: {ckpt} e {hist_csv} já existem; nada a fazer (apague os dois para treinar de novo).")
        return
    if ckpt.exists():
        print(f"{a.exp}: {ckpt} existe SEM histórico (treino interrompido): será sobrescrito.")

    from src.experimento import ler_manifest, loader_validacao, preparar_treino, treinar_experimento
    from src.model import LogMelPadronizado

    hp = ns["HP"]
    df = ler_manifest()
    pool_tr, grupos, cotas, val_df = preparar_treino(df, hp)
    del df
    val_loader = loader_validacao(val_df, hp)
    logmel = LogMelPadronizado(hp["SAMPLE_RATE"], hp["N_FFT"], hp["WIN_LENGTH"], hp["HOP_LENGTH"],
                               hp["N_MELS"]).to(ns["DEVICE"])
    print(f"treino: {len(pool_tr)} áudios | validação: {len(val_df)} | NUM_WORKERS = {hp['NUM_WORKERS']}")
    treinar_experimento(a.exp, ns["EXPERIMENTOS"][a.exp], hp, pool_tr, grupos, cotas, val_df, val_loader,
                        logmel, ns["DEVICE"], ckpt, hist_csv, ns["VERSAO"],
                        log=lambda s: print(s, flush=True))
    amostrador.parar = True
    print(f"\n{a.exp} concluído em {(time.time() - t0) / 60:.1f} min | pico de memória reservada (commit) "
          f"deste processo + workers: {amostrador.pico_proc / 1024:.1f} GB | pico do sistema: "
          f"{amostrador.pico_sis:.1f} GB")


if __name__ == "__main__":
    main()
