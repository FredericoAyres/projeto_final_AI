#!/usr/bin/env python3
"""Exporta os pesos da R1 da v8 (as 2 seeds, na época escolhida pela validação) para compartilhar, e confere.

- Grava só o que o fine-tuning mudou (camadas 5, 6, 7 do XLS-R + a cabeça linear; 37,8 M números) em float16, no
  formato safetensors (⚠️ Fora da trilha: o formato padrão do Hugging Face para pesos; ao contrário do .pt, abrir o
  arquivo não executa código). O resto do modelo é o XLS-R 300M original (baixado do Hugging Face) e a
  padronização da T3 (results/v7_t3_coeficientes.npz, no git).
- Confere: as notas com os pesos exportados x as notas gravadas na validação (data/processed/v8_notas), num
  sorteio fixo de clipes da validação. Não toca no teste.
- Saídas: models/v8_R1_s{seed}_pesos.safetensors (fora do git; ~76 MB cada) e results/v8_r1_pesos.json (no git:
  SHA256, tamanho e a conferência).

    python scripts/v8_exportar.py [--n 256]

Para usar os pesos em outro computador (com o repositório e o env):
    from scripts.v8_exportar import carregar
    modelo = carregar(42)            # DetectorFT pronto na GPU; notas com src.finetune.pontuar
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import v8  # noqa: E402  (chdir para a raiz, caminhos e construir)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEEDS = (42, 123)


def arq_pesos(seed):
    return v8.MODELOS / f"v8_R1_s{seed}_pesos.safetensors"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def exportar(seed, ep):
    import torch
    from safetensors.torch import save_file
    e = torch.load(v8.arq_ckpt("R1", seed, ep), map_location="cpu", weights_only=False)
    assert e["receita"] == "R1" and int(e["seed"]) == seed
    pesos = {k: v.to(torch.float16).contiguous() for k, v in e["treinavel"].items()}
    meta = {"versao": "v8", "receita": "R1", "seed": str(seed), "epoca": str(ep),
            "base": "facebook/wav2vec2-xls-r-300m cortado na camada 7; camadas 5, 6, 7 + cabeça treináveis",
            "padronizacao": "results/v7_t3_coeficientes.npz (mu, sd)", "dtype_original": "float32"}
    save_file(pesos, str(arq_pesos(seed)), metadata=meta)
    return sum(v.numel() for v in pesos.values())


def carregar(seed, dispositivo="cuda", arquivo=None):
    """DetectorFT da R1 com os pesos exportados (o XLS-R original + a padronização da T3 + os pesos da v8)."""
    import torch
    from safetensors.torch import load_file
    model = v8.construir("R1", seed, dispositivo)
    pesos = {k: v.to(torch.float32) for k, v in load_file(str(arquivo or arq_pesos(seed))).items()}
    faltam, sobram = model.load_state_dict(pesos, strict=False)
    assert not sobram, sobram
    treinaveis = {n for n, p in model.named_parameters() if p.requires_grad}
    assert treinaveis == set(pesos), "os pesos exportados não cobrem as camadas treináveis"
    return model.eval()


def conferir(seed, ep, n, nw):
    from src.finetune import pontuar
    v = v8.validacao(v8.splits())
    rng = np.random.default_rng(v8.SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(v.label.values == c), n // 2, replace=False) for c in (0, 1)]))
    sub = v.iloc[idx].reset_index(drop=True)
    gravadas = sub[["linha"]].merge(pd.read_parquet(v8.arq_val("R1", seed, ep)), on="linha", how="left").z.values
    model = carregar(seed)
    z, _ = pontuar(model, v8.fonte(sub), v8.dur(sub), v8.hp(), nw, v8.POR_LOTE_AVAL, log=lambda *a, **k: None)
    d = np.abs(z - gravadas)
    del model
    import torch
    torch.cuda.empty_cache()
    return {"n_clipes": int(len(sub)), "dif_max_logit": float(d.max()), "dif_mediana_logit": float(np.median(d)),
            "correlacao": float(np.corrcoef(z, gravadas)[0, 1]),
            "mesma_decisao_com_corte_0": float(np.mean((z > 0) == (gravadas > 0)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="clipes da validação para a conferência")
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args()
    esc = json.loads((v8.R / "v8_escolha.json").read_text(encoding="utf-8"))
    saida = {"o_que_e": "pesos da R1 da v8 (camadas 5, 6, 7 do XLS-R + cabeça), float16, safetensors; "
                        "a época de cada seed é a escolhida pela validação (results/v8_escolha.json)",
             "como_usar": "from scripts.v8_exportar import carregar; modelo = carregar(42)",
             "precisa_tambem": ["facebook/wav2vec2-xls-r-300m (Hugging Face)", "results/v7_t3_coeficientes.npz"],
             "arquivos": {}}
    for seed in SEEDS:
        ep = int(esc["epoca_escolhida"][f"R1_s{seed}"])
        n = exportar(seed, ep)
        p = arq_pesos(seed)
        conf = conferir(seed, ep, a.n, a.num_workers)
        saida["arquivos"][p.name] = {"seed": seed, "epoca": ep, "n_parametros": int(n),
                                     "MB": round(p.stat().st_size / 2 ** 20, 1), "sha256": sha256(p),
                                     "conferencia_validacao": conf}
        print(p, saida["arquivos"][p.name], flush=True)
    v8.salvar_json(saida, v8.R / "v8_r1_pesos.json")
    print("gravado results/v8_r1_pesos.json")


if __name__ == "__main__":
    main()
