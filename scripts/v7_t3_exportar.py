#!/usr/bin/env python3
"""Exporta os pesos da receita T3 da v7 (o sistema atual) para um arquivo pequeno e estável em results/.

Não treina nada. Os pesos da T3 já foram gravados pela etapa `treinar` da v7 em `results/v7_parcial/T3.npz`
(média e desvio da padronização, coeficientes, intercepto e C; pasta fora do git). Este script:
  1. confere o SHA256 de data/processed/splits_v7.parquet contra o do protocolo (docs/PROTOCOLO_V7.md);
  2. confere a composição do treino guardada no npz contra a divisão congelada (mesmos blocos, classes e contagens);
  3. recalcula as notas de TODOS os 249.187 clipes com esses pesos e compara com as notas gravadas
     (data/processed/v7_notas.parquet, coluna z_T3): diferença máxima <= 1e-4;
  4. recalcula a AUC por recorte do teste mestre, dos controles e da validação e compara, com 3 casas, com
     results/v7_teste_mestre.csv e results/v7_validacao.csv;
  5. só se tudo bater, grava results/v7_t3_coeficientes.npz e results/v7_t3_modelo.json.
Se algo não bater, NÃO grava os pesos e escreve logs/v7_pesos_BLOQUEIO.txt.

    python scripts/v7_t3_exportar.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

R = Path("results")
DP = Path("data/processed")
SPLITS = DP / "splits_v7.parquet"
SHA_PROTOCOLO = "24d3aea93654711a80ea1482799c5de3ec180ec5e336c719935e98b7f7a54c63"
ORIGEM = R / "v7_parcial" / "T3.npz"
SAIDA_NPZ = R / "v7_t3_coeficientes.npz"
SAIDA_JSON = R / "v7_t3_modelo.json"
BLOQUEIO = Path("logs") / "v7_pesos_BLOQUEIO.txt"
CAMADAS = [5, 6, 7]
TOL = 1e-4


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def bloquear(motivo):
    BLOQUEIO.parent.mkdir(exist_ok=True)
    BLOQUEIO.write_text(f"{datetime.now().isoformat(timespec='seconds')}\n{motivo}\n", encoding="utf-8")
    print("BLOQUEIO:", motivo)
    sys.exit(4)


def pontuar(X, linhas, mu, sd, coef, b, lote=20000):
    # igual a scripts/v7.py: pontuar (float32, padronização, produto, intercepto)
    z = np.empty(len(linhas))
    for i in range(0, len(linhas), lote):
        idx = linhas[i:i + lote]
        z[i:i + lote] = ((np.asarray(X[idx], dtype=np.float32) - mu) / sd) @ coef + b
    return z


def main():
    sha = sha256(SPLITS)
    if sha != SHA_PROTOCOLO:
        bloquear(f"SHA256 do splits_v7.parquet {sha} != protocolo {SHA_PROTOCOLO}")
    if not ORIGEM.exists():
        bloquear(f"{ORIGEM} não existe; seria preciso reajustar a T3 (não feito aqui)")
    z = np.load(ORIGEM, allow_pickle=True)
    mu, sd, coef = z["mu"], z["sd"], z["coef"]
    b, C, n_treino = float(z["b"]), float(z["C"]), int(z["n_treino"])
    composicao = json.loads(str(z["composicao"]))
    grade = json.loads(str(z["grade"]))

    sp = pd.read_parquet(SPLITS)
    tr = sp[(sp.split == "treino") & sp.bloco.isin(["estudio", "ml_itw", "radar_eval"])
            & ((sp.bloco != "radar_eval") | sp.pareado_treino)]
    comp_div = tr.groupby(["bloco", "label"]).size().rename("n").reset_index().to_dict("records")
    comp_div = [{"bloco": c["bloco"], "label": int(c["label"]), "n": int(c["n"])} for c in comp_div]
    if comp_div != composicao or len(tr) != n_treino:
        bloquear(f"composição do treino diferente: npz {composicao} × divisão {comp_div}")
    esc = json.loads((R / "v7_escolha.json").read_text(encoding="utf-8"))
    tre = json.loads((R / "v7_treino.json").read_text(encoding="utf-8"))
    if abs(tre["C_escolhido"]["T3"] - C) > 0 or esc["receita_vencedora_pela_validacao"] != "T3":
        bloquear(f"C do npz ({C}) ou receita vencedora não batem com results/v7_treino.json / v7_escolha.json")

    X = np.load(DP / "v7_vetores" / "X.npy", mmap_mode="r")
    meta = json.loads((DP / "v7_vetores" / "concluido.json").read_text(encoding="utf-8"))
    assert meta["camadas"] == CAMADAS and X.shape[1] == 3072
    notas = pd.read_parquet(DP / "v7_notas.parquet", columns=["linha", "bloco", "fonte", "split", "label", "idioma", "z_T3"])
    zt = pontuar(X, notas.linha.values, mu, sd, coef, b)
    dif = float(np.abs(zt - notas.z_T3.values).max())
    notas["z_novo"] = zt

    # AUC com 3 casas: teste mestre e controles (v7_teste_mestre.csv) e validação por fonte (v7_validacao.csv)
    tm = pd.read_csv(R / "v7_teste_mestre.csv")
    tm = tm[(tm.receita == "T3") & tm.AUC.notna()].set_index("recorte").AUC
    recortes = {
        "ml_itw": (notas.bloco == "ml_itw") & (notas.split == "teste_mestre"),
        "in_the_wild": (notas.bloco == "in_the_wild") & (notas.split == "teste_mestre"),
        "radar_en": (notas.bloco == "radar_eval") & (notas.split == "teste_mestre"),
        "ml_itw_portugues": None,
    }
    conf = []
    for rc, m in recortes.items():
        if m is None or rc not in tm.index:
            continue
        g = notas[m]
        conf.append({"recorte": rc, "AUC recalculada": roc_auc_score(g.label, g.z_novo), "AUC gravada": tm[rc]})
    for f in ("llamapartialspoof", "brspeech_df", "asvspoof2019_la", "asvspoof2021_df"):
        g = notas[(notas.split == "controle") & (notas.fonte == f)]
        conf.append({"recorte": f, "AUC recalculada": roc_auc_score(g.label, g.z_novo), "AUC gravada": tm[f]})
    va = pd.read_csv(R / "v7_validacao.csv").set_index("receita").loc["T3"]
    vl = notas[notas.split == "validacao"].copy()
    vl["chave"] = np.where(vl.bloco == "estudio", vl.fonte, vl.bloco)
    for k, g in vl.groupby("chave"):
        if g.label.nunique() == 2:
            conf.append({"recorte": f"validação {k}", "AUC recalculada": roc_auc_score(g.label, g.z_novo),
                         "AUC gravada": float(va[f"AUC {k}"])})
    conf = pd.DataFrame(conf)
    conf["igual com 3 casas"] = conf["AUC recalculada"].round(3) == conf["AUC gravada"].round(3)
    print(conf.round(4).to_string(index=False))
    print(f"diferença máxima de nota (249.187 clipes): {dif:.2e}")
    if dif > TOL or not conf["igual com 3 casas"].all():
        bloquear(f"não reproduziu: diferença máxima {dif:.2e}; recortes diferentes: "
                 f"{conf[~conf['igual com 3 casas']].recorte.tolist()}")

    np.savez(SAIDA_NPZ, coef=coef.astype(np.float32), intercepto=np.float64(b), mu=mu.astype(np.float32),
             sd=sd.astype(np.float32), C=np.float64(C), camadas=np.array(CAMADAS))
    info = {
        "quando": datetime.now().isoformat(timespec="seconds"),
        "sistema": "v7 (T3): estúdio + ML-ITW (treino) + RADAR (treino, pareado por duração)",
        "origem_dos_pesos": f"{ORIGEM.as_posix()} (gravado pela etapa 'treinar' de scripts/v7.py às {tre['quando']}); não foi reajustado",
        "como_usar": "nota = ((x - mu) / sd) @ coef + intercepto; x = média no tempo das saídas 5, 6 e 7 do XLS-R 300M "
                     "(facebook/wav2vec2-xls-r-300m, congelado), concatenadas nessa ordem (3.072 números), com o pipeline de "
                     "áudio da v4; nota em logit, maior = mais falso",
        "modelo_ssl": "facebook/wav2vec2-xls-r-300m (congelado)",
        "camadas": CAMADAS,
        "n_numeros": int(len(coef)),
        "regressao": {"tipo": "LogisticRegression (scikit-learn)", "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4,
                      "C": C, "class_weight": "balanced", "sample_weight": None, "penalidade": "L2"},
        "padronizacao": "média e desvio (sem pesos) do treino da T3; desvio 0 trocado por 1",
        "grade_C": [g["C"] for g in grade],
        "C_escolhido_por": "maior média da AUC por fonte na validação (6 fontes), docs/PROTOCOLO_V7.md seção 3",
        "iteracoes_do_C_escolhido": next(g["iterações"] for g in grade if g["C"] == C),
        "treino": {"n": n_treino, "composicao": composicao},
        "splits_v7": {"arquivo": SPLITS.as_posix(), "sha256": sha, "sha256_do_protocolo": SHA_PROTOCOLO},
        "cortes_da_validacao_logit": esc["por_receita"]["T3"]["cortes"],
        "regra_de_corte_escolhida_no_protocolo": esc["por_receita"]["T3"]["regra"],
        "aviso_sobre_cortes": "a regra de corte do sistema final ainda está em aberto (decidir só pela validação); "
                              "as três regras empatam na validação e a calibração muda entre ambientes",
        "conferencia": {"dif_max_nota_logit": dif, "tolerancia": TOL, "n_clipes": int(len(notas)),
                        "auc_3_casas": conf.round(4).to_dict("records")},
    }
    SAIDA_JSON.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"gravado: {SAIDA_NPZ} ({SAIDA_NPZ.stat().st_size / 1024:.0f} kB) e {SAIDA_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
