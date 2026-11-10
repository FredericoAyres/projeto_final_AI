#!/usr/bin/env python3
"""v5: compara XLS-R (v5_xlsr) e WavLM Large (v5_wavlm) pela regra do pré-registro. Só validação e dev_shift.

Uso (na raiz, com o env ativo, depois de `rodar_v4.ps1 -Modelo xlsr` e `-Modelo wavlm`):
    python scripts/comparar_v5.py

Grava (results/, só arquivos pequenos): v5_comparacao.json (decisão e bootstrap), v5_comparacao_camadas.csv,
v5_comparacao_fontes.csv, v5_comparacao_geradores.csv, v5_comparacao_duracao.csv, v5_comparacao_locutores.csv.
O teste final NÃO é lido aqui.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.arquivos import salvar_csv, salvar_json  # noqa: E402
from src.comparacao import (acuracias_no_limiar, bootstrap_pareado_por_locutor, eer_por_gerador,  # noqa: E402
                            juntar_notas, metricas_por_grupo, por_duracao)
from src.metrics import eer_e_limiar, eer_media_por_fonte  # noqa: E402

R = ROOT / "results"
A, B = "xlsr", "wavlm"           # a = referência (Meta), b = candidato (Microsoft)


def ler(nome):
    return json.loads((R / nome).read_text(encoding="utf-8"))


def notas_cabeca(modelo, cabeca, conj):
    """Notas (logits) da cabeça no conjunto. Cabeças de camada única vêm da tabela de camadas."""
    if cabeca.startswith("rl_camada_"):
        l = int(cabeca.rsplit("_", 1)[1])
        d = pd.read_parquet(R / f"v5_{modelo}_notas_camadas_{conj}.parquet")
        return d.drop(columns=[c for c in d.columns if c.startswith("z_L")]).assign(z=d[f"z_L{l:02d}"].values)
    if cabeca.startswith("soma_"):
        cabeca = f"{cabeca}_ensemble"      # cabeça com 3 seeds: a média dos logits das 3 (a nota de "uma cabeça")
    return pd.read_parquet(R / f"v5_{modelo}_notas_{cabeca}_{conj}.parquet")


def main():
    for m in (A, B):
        if not (R / f"v5_{m}_concluido.json").exists():
            sys.exit(f"falta results/v5_{m}_concluido.json: rode scripts\\rodar_v4.ps1 -Modelo {m}")
    res = {m: ler(f"v5_{m}_resumo.json") for m in (A, B)}
    out = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "a": A, "b": B, "sistemas": {}}

    # ---- sistemas comparados: a cabeça final de cada modelo (decide) e "só a última camada" (só reportado)
    sistemas = {"final": {m: res[m]["cabeça_final_pela_validação"] for m in (A, B)},
                "ultima_camada": {m: "rl_camada_24" for m in (A, B)}}
    tabelas = {}
    for sis, cab in sistemas.items():
        nv = juntar_notas(notas_cabeca(A, cab[A], "val"), notas_cabeca(B, cab[B], "val"), (A, B))
        nd = juntar_notas(notas_cabeca(A, cab[A], "dev_shift"), notas_cabeca(B, cab[B], "dev_shift"), (A, B))
        info = {"cabeças": cab}
        lim = {}
        for m in (A, B):
            z = f"z_{m}"
            lim[z] = eer_e_limiar(nv.label.values, nv[z].values)[1]            # limiar: ponto de EER da validação
            eer_mf, por_fonte = eer_media_por_fonte(nv.label.values, nv[z].values, nv.source_dataset.values)
            info[m] = {"val EER média por fonte": eer_mf, "val EER por fonte": por_fonte,
                       "dev AUC": float(roc_auc_score(nd.label.values, nd[z].values)),
                       "dev EER": eer_e_limiar(nd.label.values, nd[z].values)[0], "limiar_val_logit": lim[z],
                       **{f"dev {k}": v for k, v in acuracias_no_limiar(nd.label.values, nd[z].values, lim[z]).items()}}
        info["dev acurácia trivial"] = float(max(nd.label.mean(), 1 - nd.label.mean()))
        info["bootstrap_dev_por_locutor"] = bootstrap_pareado_por_locutor(
            nd.label.values, nd[f"z_{A}"].values, nd[f"z_{B}"].values, nd.speaker_id.values)
        out["sistemas"][sis] = info
        tabelas[sis] = (nv, nd, lim)

    # ---- decisão (pré-registro b), só com a cabeça final
    f = out["sistemas"]["final"]
    bs = f["bootstrap_dev_por_locutor"]
    val_b_melhor = f[B]["val EER média por fonte"] < f[A]["val EER média por fonte"]
    dev_b_melhor = bs["dif AUC (b - a)"] > 0 and not bs["dif AUC IC95 inclui zero"]
    dev_b_pior = bs["dif AUC (b - a)"] < 0 and not bs["dif AUC IC95 inclui zero"]
    if val_b_melhor and dev_b_melhor:
        decisao = f"{B} MELHOR (validação e dev_shift, com o intervalo do dev_shift sem o zero)"
    elif (not val_b_melhor) and dev_b_pior:
        decisao = f"{B} PIOR (validação e dev_shift, com o intervalo do dev_shift sem o zero): segue o {A}"
    else:
        decisao = f"EMPATE pela regra (validação e dev_shift não concordam ou o intervalo inclui zero): segue o {A}"
    out["decisao"] = {"val_wavlm_menor_eer": bool(val_b_melhor), "dev_wavlm_melhor_demonstrado": bool(dev_b_melhor),
                      "dev_wavlm_pior_demonstrado": bool(dev_b_pior), "texto": decisao}

    # ---- análise descritiva (pré-registro d, e): não muda a decisão
    nv, nd, lim = tabelas["final"]
    zc = [f"z_{A}", f"z_{B}"]
    fontes = metricas_por_grupo(nv, zc, "source_dataset")
    geradores = eer_por_gerador(nv, zc)
    duracao = por_duracao(nd, zc, lim)
    locutores = metricas_por_grupo(nd, zc, "speaker_id", lim)
    ca, cb = (pd.read_csv(R / f"v5_{m}_camadas.csv") for m in (A, B))
    cols = ["camada", "C", "val EER média por fonte", "dev AUC", "dev EER", "dev acurácia balanceada"]
    camadas = ca[cols].merge(cb[cols], on="camada", suffixes=(f" {A}", f" {B}"))
    at = {m: ler(f"v5_{m}_atalhos.json") for m in (A, B)}
    out["atalhos_b_fonte_dos_reais"] = {m: {"camada": at[m]["b_fonte_por_camada"]["camada"],
                                            "acurácia balanceada": at[m]["b_fonte_por_camada"]["acurácia balanceada"]}
                                        for m in (A, B)}
    out["atalhos_b_fonte_dos_reais"]["estatisticas"] = at[A]["b_fonte_por_estatisticas"]["acurácia balanceada"]
    for nome, t in (("camadas", camadas), ("fontes", fontes), ("geradores", geradores), ("duracao", duracao),
                    ("locutores", locutores)):
        salvar_csv(t, R / f"v5_comparacao_{nome}.csv")
    salvar_json(out, R / "v5_comparacao.json")

    # ---- impressão
    for sis, info in out["sistemas"].items():
        print(f"\n== {sis}: {info['cabeças']}")
        for m in (A, B):
            i = info[m]
            print(f"  {m:6s} val EER m/f {100 * i['val EER média por fonte']:.2f}% | dev AUC {i['dev AUC']:.4f} "
                  f"EER {100 * i['dev EER']:.2f}% | acc.bal. {i['dev acurácia balanceada']:.3f} "
                  f"(0,5: {i['dev acurácia balanceada (0,5)']:.3f}) | acc. {i['dev acurácia']:.3f} (trivial {info['dev acurácia trivial']:.3f})")
        b = info["bootstrap_dev_por_locutor"]
        print(f"  dif AUC ({B} - {A}) {b['dif AUC (b - a)']:+.4f}, IC95 [{b['dif AUC IC95'][0]:+.4f}, {b['dif AUC IC95'][1]:+.4f}]"
              f" | dif EER {100 * b['dif EER (b - a)']:+.2f} pp, IC95 [{100 * b['dif EER IC95'][0]:+.2f}, {100 * b['dif EER IC95'][1]:+.2f}]"
              f" | {b['n_locutores']} locutores, {b['n_reamostragens']} reamostragens")
    print(f"\nDECISÃO: {decisao}")
    pd.set_option("display.width", 200)
    for nome, t in (("fontes (validação)", fontes), ("geradores (validação)", geradores),
                    ("duração (dev_shift)", duracao), ("locutores (dev_shift)", locutores)):
        print(f"\n-- {nome}\n{t.round(3).to_string(index=False)}")
    print("\n-- atalhos (b), fonte dos reais:", json.dumps(out["atalhos_b_fonte_dos_reais"], ensure_ascii=False))


if __name__ == "__main__":
    main()
