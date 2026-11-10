"""Teste final da v4 (avaliado UMA vez): amostra, plano de extração, sistemas S0 a S3 e métricas.

Pré-registro: docs/HISTORICO_VERSOES.md, seção "v4 - teste final: pré-registro". Nada aqui escolhe
sistema, camada, C ou limiar olhando o teste: tudo o que é escolhido usa a validação (ou o dev_shift,
no limiar B), e o teste só é medido.

Sistemas:
- S0: a CNN da v3 com menor EER média por fonte na validação (janelas de 4 s, passo 2 s, média dos logits);
- S1: XLS-R congelado, só a saída 24 (média no tempo) + regressão logística (C pela validação);
- S2: a cabeça final da v4 (concat_top3), reajustada com o mesmo C e conferida contra as notas gravadas;
- S3: regressão logística de 2 entradas (logit do S2 e do S0, padronizados pela validação), treinada só
  na validação.

Bootstrap: reamostragem com reposição DENTRO de cada classe (o nº de reais e de fakes não muda), as
mesmas reamostragens para todos os sistemas (comparação pareada). O intervalo mede só a amostra de
teste, não a variação do treino. ⚠️ Fora da trilha: o bootstrap.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metrics import acuracia_trivial, acuracias, eer_e_limiar, metricas

# bancos do teste final (pré-registro): (nome, tipo)
BANCOS = [("in_the_wild", "principal"), ("llamapartialspoof", "principal"), ("brspeech_df", "principal"),
          ("coraa", "só real"), ("asvspoof2019_la", "secundário"), ("asvspoof2021_df", "secundário")]
# QUICK: bancos de mentira tirados da VALIDAÇÃO (nunca do teste), só para exercitar o código
BANCOS_QUICK = [("quick_dfadd", "principal"), ("quick_shifty", "principal"), ("quick_codec", "principal"),
                ("quick_libri", "só real")]
_FONTE_QUICK = {"quick_dfadd": "dfadd", "quick_shifty": "shiftyspeech", "quick_codec": "codecfake_plus",
                "quick_libri": "librispeech"}
SISTEMAS = ("S0", "S1", "S2", "S3")
DESCRICAO = {"S0": "CNN da v3 (log-mel, janelas)", "S1": "XLS-R congelado, só a camada 24 + regressão logística",
             "S2": "v4 final (concat_top3: camadas 5, 6, 7)", "S3": "v4 + CNN (regressão de 2 entradas na validação)"}
LIMIARES = {"A": "ponto de EER da validação", "B": "ponto de EER do dev_shift"}


# ---------------------------------------------------------------- amostra e plano
def amostra_teste_idx(df, max_por_classe, seed, quick=False, excluir=()):
    """Índices do manifest da amostra de teste, banco a banco e classe a classe.

    Mesma regra de src/experimento.py: amostra_teste (até `max_por_classe` por classe e banco,
    `random_state=seed`, In-the-Wild só test_final); devolve (DataFrame com idx_manifest e banco).
    QUICK: bancos de mentira da VALIDAÇÃO (fora de `excluir`), sem tocar no teste.
    """
    partes = []
    if not quick:
        tf = df[(df.split == "test") & (df.eval_subset == "test_final")]
        for banco, _ in BANCOS:
            for lab in (0, 1):
                g = tf[(tf.source_dataset == banco) & (tf.label == lab)]
                if len(g):
                    partes.append(pd.DataFrame({"idx_manifest": g.sample(min(len(g), max_por_classe),
                                                                         random_state=seed).index.values,
                                                "banco": banco}))
    else:
        v = df[(df.split == "val") & ~df.index.isin(list(excluir))]
        for banco, tipo in BANCOS_QUICK:
            for lab in ((0,) if tipo == "só real" else (0, 1)):
                g = v[(v.source_dataset == _FONTE_QUICK[banco]) & (v.label == lab)]
                partes.append(pd.DataFrame({"idx_manifest": g.sample(min(len(g), max_por_classe),
                                                                     random_state=seed).index.values,
                                            "banco": banco}))
    return pd.concat(partes, ignore_index=True)


def montar_plano_teste(amostra, tam_shard):
    """Plano no formato de src/cache_v4.py (idx_manifest, conjunto, shard, cortar_silencio, ordem).
    Conjunto = "teste_<banco>"; cada banco em shards de `tam_shard`, na ordem da amostra."""
    linhas = []
    for banco, g in amostra.groupby("banco", sort=False):
        idx = g.idx_manifest.values
        for s, i in enumerate(range(0, len(idx), tam_shard)):
            linhas.append(pd.DataFrame({"idx_manifest": idx[i:i + tam_shard], "conjunto": f"teste_{banco}",
                                        "shard": s, "cortar_silencio": True}))
    plano = pd.concat(linhas, ignore_index=True)
    plano["ordem"] = np.arange(len(plano))
    return plano


def carregar_ou_criar_plano_teste(amostra, tam_shard, cache_dir, cfg):
    """Cria o plano na 1ª vez; nas seguintes, confere que nada mudou (senão para, sem apagar nada)."""
    from src.arquivos import salvar_json, salvar_parquet
    cache_dir = Path(cache_dir)
    arq, arq_cfg = cache_dir / "plano.parquet", cache_dir / "plano.json"
    plano = montar_plano_teste(amostra, tam_shard)
    if arq.exists():
        salvo = json.loads(arq_cfg.read_text(encoding="utf-8"))
        if salvo != json.loads(json.dumps(cfg, default=str)):
            raise RuntimeError(f"{arq_cfg} é de outra configuração. Nada foi apagado.\nsalvo: {salvo}\nagora: {cfg}")
        antigo = pd.read_parquet(arq)
        if not antigo[["idx_manifest", "conjunto", "shard"]].equals(plano[["idx_manifest", "conjunto", "shard"]]):
            raise RuntimeError(f"{arq} não bate com a amostra recalculada. Nada foi apagado.")
        return antigo
    cache_dir.mkdir(parents=True, exist_ok=True)
    salvar_parquet(plano, arq)
    salvar_json(cfg, arq_cfg)
    return plano


# ---------------------------------------------------------------- escolha do S0 (pela validação)
def escolher_s0(results_dir, sufixo_v3=""):
    """O marcador da v3 com MENOR EER média por fonte na validação. Devolve (nome, tabela)."""
    linhas = []
    for p in sorted(Path(results_dir).glob(f"v3_*_s*_concluido{sufixo_v3}.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("quick") != bool(sufixo_v3):
            continue
        linhas.append({"experimento": j["experimento"], "val EER média por fonte": j["val_eer_media_por_fonte"],
                       "dev_shift AUC (janelas)": j["dev_shift_janelas"]["AUC"], "condição": j["condicao"],
                       "seed": j["seed"], "checkpoint": j["arquivos"]["checkpoint"]})
    tab = pd.DataFrame(linhas).sort_values("val EER média por fonte").reset_index(drop=True)
    return tab.iloc[0]["experimento"], tab


# ---------------------------------------------------------------- métricas
def metricas_banco(y, z, limiar):
    """As métricas do pré-registro num banco, no `limiar` (mesma escala de `z`)."""
    y = np.asarray(y).astype(int)
    m = metricas(y, z, limiar)
    a = acuracias(y, z, limiar)
    duas = m["n real"] > 0 and m["n fake"] > 0
    return {"acurácia": a["acurácia"], "acurácia balanceada": a["acurácia balanceada"] if duas else np.nan,
            "acurácia trivial": acuracia_trivial(y), "AUC": m["AUC"], "EER": m["EER"],
            "recall fake": m["recall fake"], "falso alarme real": m["falso alarme"],
            "n real": m["n real"], "n fake": m["n fake"]}


def reamostras(y, n_boot, seed):
    """(n_boot, n) índices: reamostragem com reposição dentro de cada classe (mesmo nº por classe)."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    grupos = [np.nonzero(y == c)[0] for c in (0, 1) if (y == c).any()]
    return np.stack([np.concatenate([rng.choice(g, len(g), replace=True) for g in grupos])
                     for _ in range(n_boot)])


def _bal(y, z, limiar):
    pred = z >= limiar
    return 0.5 * ((~pred[y == 0]).mean() + pred[y == 1].mean())


def bootstrap_banco(y, notas, limiares, idx):
    """Para cada sistema: AUC e acurácia balanceada (por limiar) em cada reamostragem; no banco só real,
    o falso alarme. `notas` = {sistema: z}; `limiares` = {sistema: {"A": t, "B": t}}. Devolve
    {sistema: {métrica: array(n_boot)}}."""
    y = np.asarray(y).astype(int)
    duas = (y == 0).any() and (y == 1).any()
    out = {}
    for s, z in notas.items():
        z = np.asarray(z)
        r = {}
        if duas:
            r["AUC"] = np.array([roc_auc_score(y[i], z[i]) for i in idx])
            for L, t in limiares[s].items():
                r[f"acurácia balanceada|{L}"] = np.array([_bal(y[i], z[i], t) for i in idx])
        else:
            for L, t in limiares[s].items():
                r[f"falso alarme real|{L}"] = np.array([(z[i] >= t).mean() for i in idx])
        out[s] = r
    return out


def ic(a, nivel=0.95):
    lo = (1 - nivel) / 2
    return float(np.quantile(a, lo)), float(np.quantile(a, 1 - lo))


def limiares_de(z_val, y_val, z_dev, y_dev):
    return {"A": eer_e_limiar(y_val, z_val)[1], "B": eer_e_limiar(y_dev, z_dev)[1]}


# ---------------------------------------------------------------- gráfico
NOME_CURTO = {"in_the_wild": "In-the-Wild", "llamapartialspoof": "LlamaPartial", "brspeech_df": "BRSpeech-DF",
              "asvspoof2019_la": "ASVspoof19", "asvspoof2021_df": "ASVspoof21 DF"}
CORES = {"S0": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a", "S3": "#eda100"}


def grafico(tab, bancos):
    """Acurácia balanceada por banco e sistema (limiares A e B) e falso alarme do banco só real, com o
    intervalo bootstrap. `tab` = results/v4_teste_final.csv; devolve a figura."""
    import matplotlib.pyplot as plt
    tipo = dict(bancos)
    duas = [b for b, t in bancos if t != "só real"]
    so_real = [b for b, t in bancos if t == "só real"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=80, gridspec_kw={"width_ratios": [2.2, 2.2, 1]})
    w = 0.8 / len(SISTEMAS)
    x = np.arange(len(duas))
    for ax, L in zip(axes[:2], ("A", "B")):
        for i, s in enumerate(SISTEMAS):
            g = tab[(tab.sistema == s) & (tab.limiar == L) & (tab["métrica"] == "acurácia balanceada")
                    & tab.banco.isin(duas)].set_index("banco").loc[duas]
            ax.bar(x + (i - 1.5) * w, g.valor, w * 0.92, color=CORES[s], label=f"{s}: {DESCRICAO[s]}",
                   yerr=[g.valor - g["IC 2,5%"], g["IC 97,5%"] - g.valor], error_kw={"lw": 1, "capsize": 2})
        ax.axhline(0.5, color="#52514e", ls="--", lw=1)
        ax.set_xticks(x, [f"{NOME_CURTO.get(b, b)}\n({tipo[b]})" for b in duas], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("acurácia balanceada")
        ax.set_title(f"limiar {L}: {LIMIARES[L]}", fontsize=10)
    if so_real:
        for i, s in enumerate(SISTEMAS):
            g = tab[(tab.sistema == s) & (tab.banco == so_real[0]) & (tab["métrica"] == "falso alarme real")
                    ].set_index("limiar").loc[["A", "B"]]
            axes[2].bar(np.arange(2) + (i - 1.5) * w, 100 * g.valor, w * 0.92, color=CORES[s],
                        yerr=[100 * (g.valor - g["IC 2,5%"]), 100 * (g["IC 97,5%"] - g.valor)],
                        error_kw={"lw": 1, "capsize": 2})
        axes[2].set_xticks(np.arange(2), ["limiar A", "limiar B"], fontsize=8)
        axes[2].set_ylim(0, 100)
        axes[2].set_ylabel("% de reais acusados")
        axes[2].set_title(f"{so_real[0]}: falso alarme\n(só reais; menor é melhor)", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Teste final da v4 (barras de erro: intervalo bootstrap de 95%, só da amostra de teste)", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    return fig
