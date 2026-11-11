#!/usr/bin/env python3
"""Figuras e tabelas do documento final do sistema v4 (F0). Só DIAGNÓSTICO: nada aqui escolhe modelo.

- Lê os caches existentes (data/processed/v4_cache e v4_cache_teste); não extrai nada, não baixa nada.
- Reajusta a cabeça F0 (= S2 da v4: camadas 5, 6, 7, média no tempo, padronizadas com o treino; regressão
  logística C = 0,1, lbfgs, max_iter 1000, tol 1e-4) e CONFERE que reproduz as notas gravadas na v4
  (results/v4_teste_notas.parquet, coluna z_S2). Se não reproduzir, para sem desenhar nada.
- As regressões extras (curva de aprendizado, caminho do C) são só para os gráficos G6 e G7: não mudam
  camada, C nem limiar de nada. Os bancos de teste só aparecem em figuras marcadas como "só visualização".
- Grava results/v4final_*.png (dpi 100) e results/v4final_*.csv, e results/v4final_resumo.json.
- Limite de memória: o processo se encerra sozinho se a memória privada passar de 10 GB.

    python scripts/v4final_figuras.py
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.analysis import FAIXAS_DURACAO, faixa_duracao  # noqa: E402
from src.cache_v4 import Conjunto  # noqa: E402
from src.metrics import acuracia_trivial, acuracias, eer_e_limiar, eer_media_por_fonte, sigmoide  # noqa: E402

warnings.filterwarnings("ignore")
LIMITE_GB = 10.0
R = Path("results")
DPI = 100
T0 = time.time()

# ---------------- cores (paleta de referência; real/fake iguais às das versões anteriores) ----------------
REAL_C, FAKE_C = "#2a78d6", "#eb6834"
COR_A, COR_B = "#0b0b0b", "#8a8984"
COR_TREINO, COR_VAL, COR_DEV = "#6250d6", "#1baf7a", "#eda100"
COR_F0, COR_S0, COR_S1, COR_TRIV = "#1baf7a", "#6250d6", "#eb6834", "#b8b7b2"
COR_META, COR_MS = "#2a78d6", "#e87ba4"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": "#8a8984", "axes.titlesize": 10, "legend.frameon": False})
NOME = {"in_the_wild": "In-the-Wild", "llamapartialspoof": "LlamaPartialSpoof", "brspeech_df": "BRSpeech-DF",
        "coraa": "CORAA (só real)", "asvspoof2019_la": "ASVspoof 2019 (sec.)", "asvspoof2021_df": "ASVspoof 2021 DF (sec.)",
        "dfadd": "DFADD", "shiftyspeech": "ShiftySpeech", "codecfake_plus": "CodecFake+", "librispeech": "LibriSpeech (só real)",
        "val": "validação (geral)", "dev_shift": "dev_shift"}
BANCOS = ["in_the_wild", "llamapartialspoof", "brspeech_df", "coraa", "asvspoof2019_la", "asvspoof2021_df"]
PRINCIPAIS = ["in_the_wild", "llamapartialspoof", "brspeech_df"]
FONTES_VAL = ["dfadd", "shiftyspeech", "codecfake_plus", "librispeech"]


# ---------------- vigia de memória ----------------
def memoria_gb():
    eu = psutil.Process()
    tot = 0
    for p in [eu] + eu.children(recursive=True):
        try:
            tot += p.memory_info().private
        except psutil.Error:
            pass
    return tot / 2 ** 30


PICO = [0.0]


def vigia():
    while True:
        m = memoria_gb()
        PICO[0] = max(PICO[0], m)
        if m > LIMITE_GB:
            print(f"\n!!! memória privada {m:.1f} GB > {LIMITE_GB} GB: encerrando o processo.", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=vigia, daemon=True).start()


def log(s):
    print(f"[{(time.time() - T0) / 60:5.1f} min | {memoria_gb():4.1f} GB] {s}", flush=True)


def salvar_fig(fig, nome):
    p = R / f"v4final_{nome}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"figura {p} ({p.stat().st_size / 1024:.0f} KB)")


def salvar_csv(df, nome):
    p = R / f"v4final_{nome}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    log(f"tabela {p} ({len(df)} linhas)")


def ler_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ======================================================================== 1. o que é o F0 (lido do disco)
res_v4 = ler_json(R / "v4_resumo.json")
mc = ler_json(R / "v4_camadas_concluido.json")
cab = [r for r in mc["concat"] if r["cabeça"] == "concat_top3"][0]
assert res_v4["cabeça_final_pela_validação"] == "concat_top3"
CAMADAS, C_F0 = list(cab["camadas"]), float(cab["C"])
tres = ler_json(R / "v4_teste_final_resumo.json")
LIM = {"A": float(tres["limiares_logit"]["S2"]["A"]), "B": float(tres["limiares_logit"]["S2"]["B"])}
RL_MAX_ITER = 1000          # célula config da v4 (notebooks/versions/v4_ssl.ipynb)
log(f"F0: camadas {CAMADAS}, C = {C_F0}, limiares (logit) A = {LIM['A']:.4f}, B = {LIM['B']:.4f}")

try:            # qual classe o AutoModel escolhe para o XLS-R (sem carregar os pesos)
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    classe_auto = MODEL_MAPPING_NAMES.get("wav2vec2")
except Exception as e:   # noqa: BLE001
    classe_auto = f"não encontrei ({e})"
cfg_hf = None
for p in Path(os.path.expanduser("~/.cache/huggingface/hub")).glob("models--facebook--wav2vec2-xls-r-300m/snapshots/*/config.json"):
    cfg_hf = json.loads(p.read_text(encoding="utf-8"))
modelo_info = {"classe_AutoModel_para_wav2vec2": classe_auto,
               "config_architectures": (cfg_hf or {}).get("architectures"),
               "config_num_hidden_layers": (cfg_hf or {}).get("num_hidden_layers"),
               "config_hidden_size": (cfg_hf or {}).get("hidden_size")}
log(f"modelo: {modelo_info}")

# ======================================================================== 2. dados (mmap)
z_pad = np.load(R / "v4_parcial" / "padronizacao.npz")
MU, SD = z_pad["mu"], z_pad["sd"]


def abrir(cache, nome):
    c = Conjunto(cache, nome)
    c.ok = c.meta.erro_leitura.fillna(0).values == 0
    return c


tr, va, dv = (abrir("data/processed/v4_cache", n) for n in ("treino", "val", "dev_shift"))
te = {b: abrir("data/processed/v4_cache_teste", f"teste_{b}") for b in BANCOS}
assert tr.ok.all() and int(z_pad["n"]) == len(tr)
conjs = {"val": va, "dev_shift": dv, **te}
metas = {k: c.meta[c.ok].reset_index(drop=True) for k, c in conjs.items()}
log("tamanhos: treino %d | " % len(tr) + " | ".join(f"{k} {len(m)}" for k, m in metas.items()))


def X(c, cams):
    return np.hstack([(c.camada(l)[c.ok] - MU[l]) / SD[l] for l in cams])


ytr = tr.meta.label.values.astype(int)
Xtr = X(tr, CAMADAS)
Xs = {k: X(c, CAMADAS) for k, c in conjs.items()}
ys = {k: m.label.values.astype(int) for k, m in metas.items()}
log(f"matrizes prontas: treino {Xtr.shape}")

# ======================================================================== 3. F0 reajustado e conferido
t = time.time()
f0 = LogisticRegression(C=C_F0, max_iter=RL_MAX_ITER, tol=1e-4).fit(Xtr, ytr)
log(f"F0 reajustado em {time.time() - t:.1f} s ({int(f0.n_iter_[0])} iterações)")
Z = {k: f0.decision_function(x) for k, x in Xs.items()}
notas = pd.read_parquet(R / "v4_teste_notas.parquet")
repro = []
for k, m in metas.items():
    g = notas[notas.banco == k].set_index("idx_manifest").loc[m.idx_manifest.values]
    zg = g.z_S2.values
    duas = len(np.unique(ys[k])) == 2
    repro.append({"conjunto": k, "n": len(m), "dif. máx. de nota (logit)": float(np.abs(Z[k] - zg).max()),
                  "AUC reajuste": round(roc_auc_score(ys[k], Z[k]), 3) if duas else np.nan,
                  "AUC gravada (v4)": round(roc_auc_score(ys[k], zg), 3) if duas else np.nan})
repro = pd.DataFrame(repro)
print(repro.to_string(index=False))
lim_re = {"A": eer_e_limiar(ys["val"], Z["val"])[1], "B": eer_e_limiar(ys["dev_shift"], Z["dev_shift"])[1]}
assert repro["dif. máx. de nota (logit)"].max() < 1e-3, "o reajuste NÃO reproduziu a v4: parei sem desenhar"
assert abs(lim_re["A"] - LIM["A"]) < 1e-4 and abs(lim_re["B"] - LIM["B"]) < 1e-4, (lim_re, LIM)
salvar_csv(repro, "reproducao")
w, b0 = f0.coef_[0].astype(np.float64), float(f0.intercept_[0])
np.savez(R / "v4final_f0_coeficientes.npz", coef=w, intercepto=b0, camadas=np.array(CAMADAS), C=C_F0,
         limiar_A=LIM["A"], limiar_B=LIM["B"])

# ======================================================================== G1 plano de decisão exato
wh = w / np.linalg.norm(w)
rng = np.random.default_rng(42)
sub = rng.choice(len(Xtr), 20000, replace=False)
Xc = Xtr[sub].astype(np.float64)
media_tr = Xc.mean(0)
Xc = Xc - media_tr
Xc = Xc - np.outer(Xc @ wh, wh)                  # remove a componente na direção de w
_, sv, vt = np.linalg.svd(Xc, full_matrices=False)
v1 = vt[0]
var_exp = float(sv[0] ** 2 / (sv ** 2).sum())
log(f"G1: |v1·ŵ| = {abs(v1 @ wh):.2e} (tem de ser ~0); variância do complemento explicada por v1: {var_exp:.3f}")
del Xc


def eixo_y(x):
    return (x - media_tr) @ v1


def amostra_idx(n, maximo=6000, seed=42):
    r = np.random.default_rng(seed)
    return np.arange(n) if n <= maximo else np.sort(r.choice(n, maximo, replace=False))


def painel_plano(ax, z, yv, lab, titulo, xlim):
    idx = amostra_idx(len(z))
    ordem = np.random.default_rng(0).permutation(idx)
    cor = np.where(lab[ordem] == 1, FAKE_C, REAL_C)
    ax.scatter(z[ordem], yv[ordem], c=cor, s=3, alpha=0.45, linewidths=0, rasterized=True)
    for L, est in (("A", "--"), ("B", ":")):
        ax.axvline(LIM[L], color=COR_A if L == "A" else COR_B, ls=est, lw=1.4)
    nr, nf = int((lab == 0).sum()), int((lab == 1).sum())
    ax.set_title(f"{titulo}\n{nr:,} reais · {nf:,} falsos (mostrados: {len(idx):,})".replace(",", "."), fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_xlabel("nota da regressão (w·x + b, logit)")


def legenda_plano(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=REAL_C, label="real"),
         Line2D([], [], ls="", marker="o", color=FAKE_C, label="falso (IA)"),
         Line2D([], [], color=COR_A, ls="--", label=f"limiar A (EER da validação) = {LIM['A']:.2f}"),
         Line2D([], [], color=COR_B, ls=":", label=f"limiar B (EER do dev_shift) = {LIM['B']:.2f}")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=9)


todos_z = np.concatenate([Z[k] for k in Z])
xlim = tuple(np.percentile(todos_z, [0.5, 99.5]) + np.array([-2, 2]))
mv = metas["val"]
paineis = [(f, (mv.source_dataset == f).values) for f in FONTES_VAL] + [("val", None), ("dev_shift", None)]
fig, axs = plt.subplots(2, 3, figsize=(14, 8.2), sharey=True)
for ax, (nome, msk) in zip(axs.ravel(), paineis):
    if nome == "dev_shift":
        z, yy, lab = Z["dev_shift"], eixo_y(Xs["dev_shift"]), ys["dev_shift"]
    else:
        m = np.ones(len(mv), bool) if msk is None else msk
        z, yy, lab = Z["val"][m], eixo_y(Xs["val"][m]), ys["val"][m]
    painel_plano(ax, z, yy, lab, NOME[nome] + ("" if nome in ("val", "dev_shift") else " (validação)"), xlim)
for ax in axs[:, 0]:
    ax.set_ylabel("PC1 fora da direção w")
fig.suptitle("G1 · Plano de decisão exato do F0: a fronteira é a reta vertical nota = limiar "
             "(validação e dev_shift)", fontsize=11)
legenda_plano(fig)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
salvar_fig(fig, "g1_plano_decisao")

fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True)
for ax, bnc in zip(axs.ravel(), ["in_the_wild", "llamapartialspoof", "brspeech_df", "coraa"]):
    painel_plano(ax, Z[bnc], eixo_y(Xs[bnc]), ys[bnc], NOME[bnc], xlim)
for ax in axs[:, 0]:
    ax.set_ylabel("PC1 do complemento (o mesmo eixo do G1)")
fig.suptitle("G1-teste · SÓ VISUALIZAÇÃO, NÃO DECIDE NADA: os bancos de teste no mesmo plano "
             "(cabeça e limiares fixados antes do teste)", fontsize=11, color="#b4232a")
legenda_plano(fig)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
salvar_fig(fig, "g1_plano_decisao_teste")

# ======================================================================== G2 histogramas da nota
ordem_g2 = ["val", "dev_shift"] + BANCOS
fig, axs = plt.subplots(2, 4, figsize=(15, 6.6))
bins = np.linspace(*xlim, 70)
for ax, k in zip(axs.ravel(), ordem_g2):
    for lab, cor, nm in ((0, REAL_C, "real"), (1, FAKE_C, "falso")):
        zz = Z[k][ys[k] == lab]
        if len(zz):
            ax.hist(np.clip(zz, *xlim), bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=cor)
            ax.hist(np.clip(zz, *xlim), bins=bins, density=True, histtype="step", lw=1.2, color=cor,
                    label=f"{nm} (n={len(zz):,})".replace(",", "."))
    ax.axvline(LIM["A"], color=COR_A, ls="--", lw=1.3)
    ax.axvline(LIM["B"], color=COR_B, ls=":", lw=1.5)
    titulo = NOME[k] + (" · teste (só mostra)" if k in BANCOS else "")
    ax.set_title(titulo, fontsize=9)
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("nota (logit)")
fig.suptitle(f"G2 · Distribuição da nota do F0, real × falso (tracejada: limiar A = {LIM['A']:.2f}; "
             f"pontilhada: limiar B = {LIM['B']:.2f}). Densidade por classe; valores fora do eixo somados nas pontas.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
salvar_fig(fig, "g2_histogramas")

# ======================================================================== G3 projeção dos vetores crus
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

MELHOR = int(res_v4["melhor_camada"])
iv = amostra_idx(len(va.meta), 3000, seed=1)
idv = amostra_idx(len(dv.meta), 3000, seed=2)
lab3 = np.concatenate([va.meta.label.values[iv], dv.meta.label.values[idv]])
dom3 = np.array(["validação"] * len(iv) + ["dev_shift"] * len(idv))
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
proj_info = {}
for row, cam in enumerate([MELHOR, 24]):
    Xr = np.vstack([va.camada(cam)[iv], dv.camada(cam)[idv]]).astype(np.float32)   # crus: sem padronizar
    pca = PCA(n_components=50, random_state=42).fit(Xr)
    P = pca.transform(Xr)
    t1 = time.time()
    T = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(P)
    log(f"G3: camada {cam}: t-SNE em {time.time() - t1:.0f} s")
    proj_info[f"camada {cam}"] = {"PCA variância PC1+PC2": float(pca.explained_variance_ratio_[:2].sum())}
    for col, (E, nome_e) in enumerate([(P[:, :2], "PCA"), (T, "t-SNE")]):
        o = np.random.default_rng(0).permutation(len(E))
        ax = axs[row, 2 * col]
        ax.scatter(E[o, 0], E[o, 1], c=np.where(lab3[o] == 1, FAKE_C, REAL_C), s=3, alpha=0.5, linewidths=0, rasterized=True)
        ax.set_title(f"camada {cam} · {nome_e} · cor = real (azul) × falso (laranja)", fontsize=9)
        ax = axs[row, 2 * col + 1]
        ax.scatter(E[o, 0], E[o, 1], c=np.where(dom3[o] == "dev_shift", "#008300", "#b8b7b2"), s=3, alpha=0.5,
                   linewidths=0, rasterized=True)
        ax.set_title(f"camada {cam} · {nome_e} · cor = validação (cinza) × dev_shift (verde)", fontsize=9)
    for ax in axs[row]:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"G3 · Os vetores CRUS (sem a regressão) da melhor camada ({MELHOR}) e da última (24): "
             "3.000 clipes da validação + 3.000 do dev_shift", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
salvar_fig(fig, "g3_projecao")

# ======================================================================== G4 AUC e EER por camada
cam = pd.read_csv(R / "v4_camadas.csv")
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axs[0]
ax.plot(cam.camada, cam["val AUC"], "-o", ms=4, lw=2, color=COR_VAL, label="validação")
ax.plot(cam.camada, cam["dev AUC"], "-o", ms=4, lw=2, color=COR_DEV, label="dev_shift")
ax.set_ylabel("AUC (maior é melhor)")
ax = axs[1]
ax.plot(cam.camada, 100 * cam["val EER média por fonte"], "-o", ms=4, lw=2, color=COR_VAL, label="validação (EER média por fonte)")
ax.plot(cam.camada, 100 * cam["dev EER"], "-o", ms=4, lw=2, color=COR_DEV, label="dev_shift (EER)")
ax.set_ylabel("EER % (menor é melhor)")
for ax in axs:
    for c_ in CAMADAS:
        ax.axvspan(c_ - 0.4, c_ + 0.4, color=COR_F0, alpha=0.15, lw=0)
    ax.axvline(24, color=COR_S1, ls="--", lw=1)
    ax.set_xlabel("saída do XLS-R (0 = entrada do transformer; 24 = última camada)")
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=8)
axs[0].set_title("AUC por camada (uma regressão logística por camada; C pela validação)")
axs[1].set_title("EER por camada · faixas verdes: camadas do F0 (5, 6, 7) · tracejada: só a 24 (S1)")
fig.tight_layout()
salvar_fig(fig, "g4_camadas")
salvar_csv(cam[["camada", "C", "val EER média por fonte", "val AUC", "dev AUC", "dev EER", "dev acurácia balanceada"]]
           .assign(**{"no F0": cam.camada.isin(CAMADAS)}), "g4_camadas")

# ======================================================================== G5 peso de cada camada no F0
blocos = np.split(w, len(CAMADAS))
g5 = pd.DataFrame({"camada": CAMADAS, "norma L2 dos 1.024 pesos": [float(np.linalg.norm(x)) for x in blocos],
                   "soma |pesos|": [float(np.abs(x).sum()) for x in blocos]})
g5["parte de Σw² (%)"] = 100 * g5["norma L2 dos 1.024 pesos"] ** 2 / (g5["norma L2 dos 1.024 pesos"] ** 2).sum()
g5["parte de Σ|w| (%)"] = 100 * g5["soma |pesos|"] / g5["soma |pesos|"].sum()
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].bar([f"camada {c_}" for c_ in CAMADAS], g5["parte de Σ|w| (%)"], color=COR_F0, width=0.55)
for i, v in enumerate(g5["parte de Σ|w| (%)"]):
    axs[0].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
axs[0].set_ylabel("% da soma dos |pesos|")
axs[0].set_title("Quanto cada camada pesa na regressão final (entradas padronizadas)")
for c_, x, cor in zip(CAMADAS, blocos, ["#2a78d6", "#1baf7a", "#eda100"]):
    axs[1].hist(np.abs(x), bins=50, histtype="step", lw=1.5, color=cor, label=f"camada {c_}")
axs[1].set_xlabel("|peso| de cada um dos 1.024 números")
axs[1].set_ylabel("quantos números")
axs[1].set_title("Distribuição dos pesos dentro de cada camada")
axs[1].legend()
fig.suptitle(f"G5 · Peso total de cada camada no F0 (intercepto b = {b0:.2f}; {len(w)} pesos)", fontsize=11)
fig.tight_layout()
salvar_fig(fig, "g5_pesos_camadas")
salvar_csv(g5, "g5_pesos_camadas")


# ======================================================================== métricas auxiliares
def bal(y, z, t):
    return acuracias(y, z, t)["acurácia balanceada"]


def avaliar_modelo(m, Xtr_, ytr_):
    zt, zv, zd = m.decision_function(Xtr_), m.decision_function(Xs["val"]), m.decision_function(Xs["dev_shift"])
    tA = eer_e_limiar(ys["val"], zv)[1]
    return {"treino acc. bal. (lim. A)": bal(ytr_, zt, tA), "val acc. bal. (lim. A)": bal(ys["val"], zv, tA),
            "dev acc. bal. (lim. A)": bal(ys["dev_shift"], zd, tA),
            "treino AUC": roc_auc_score(ytr_, zt), "val AUC": roc_auc_score(ys["val"], zv),
            "dev AUC": roc_auc_score(ys["dev_shift"], zd),
            "val EER média por fonte": eer_media_por_fonte(ys["val"], zv, metas["val"].source_dataset.values)[0],
            "dev EER": eer_e_limiar(ys["dev_shift"], zd)[0], "limiar A (logit)": tA,
            "iterações": int(m.n_iter_[0])}


# ======================================================================== G6 curva de aprendizado
linhas = []
idx0, idx1 = np.nonzero(ytr == 0)[0], np.nonzero(ytr == 1)[0]
r6 = np.random.default_rng(42)
p0, p1 = r6.permutation(idx0), r6.permutation(idx1)
for n in (2000, 5000, 10000, 20000, 40000):
    sel = np.sort(np.concatenate([p0[: n // 2], p1[: n // 2]]))
    t = time.time()
    m = f0 if n == len(Xtr) else LogisticRegression(C=C_F0, max_iter=RL_MAX_ITER, tol=1e-4).fit(Xtr[sel], ytr[sel])
    linhas.append({"clipes de treino": n, **avaliar_modelo(m, Xtr[sel], ytr[sel]), "segundos": round(time.time() - t, 1)})
    log(f"G6: n = {n}: {linhas[-1]}")
g6 = pd.DataFrame(linhas)
salvar_csv(g6, "g6_curva_aprendizado")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for col, cor, nm in (("treino", COR_TREINO, "treino (os próprios clipes)"), ("val", COR_VAL, "validação"),
                     ("dev", COR_DEV, "dev_shift (vozes de internet)")):
    axs[0].plot(g6["clipes de treino"], g6[f"{col} acc. bal. (lim. A)"], "-o", lw=2, ms=5, color=cor, label=nm)
    axs[1].plot(g6["clipes de treino"], g6[f"{col} AUC"], "-o", lw=2, ms=5, color=cor, label=nm)
for ax, nm in zip(axs, ["acurácia balanceada (limiar A de cada modelo)", "AUC"]):
    ax.set_xscale("log")
    ax.set_xticks(g6["clipes de treino"], [f"{n // 1000} mil" for n in g6["clipes de treino"]])
    ax.minorticks_off()
    ax.set_xlabel("clipes de treino (metade real, metade falso)")
    ax.set_ylabel(nm)
    ax.legend(fontsize=8)
fig.suptitle("G6 · Curva de aprendizado do F0 (mesmas camadas e C = 0,1; só o nº de clipes muda; diagnóstico)", fontsize=11)
fig.tight_layout()
salvar_fig(fig, "g6_curva_aprendizado")

# ======================================================================== G7 caminho do C
GRADE_V4 = [1e-3, 1e-2, 1e-1, 1.0]
grade7 = sorted(set([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]))
linhas = []
for Cc in grade7:
    t = time.time()
    m = f0 if Cc == C_F0 else LogisticRegression(C=Cc, max_iter=RL_MAX_ITER, tol=1e-4).fit(Xtr, ytr)
    linhas.append({"C": Cc, "na grade da v4": Cc in GRADE_V4, **avaliar_modelo(m, Xtr, ytr),
                   "norma de w": float(np.linalg.norm(m.coef_)), "segundos": round(time.time() - t, 1)})
    log(f"G7: C = {Cc:g}: val EER/fonte {100 * linhas[-1]['val EER média por fonte']:.2f}% dev AUC "
        f"{linhas[-1]['dev AUC']:.4f} ({linhas[-1]['segundos']} s, {linhas[-1]['iterações']} it.)")
g7 = pd.DataFrame(linhas)
salvar_csv(g7, "g7_caminho_c")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(g7.C, 100 * g7["val EER média por fonte"], "-o", lw=2, ms=5, color=COR_VAL, label="validação (EER média por fonte)")
axs[0].plot(g7.C, 100 * g7["dev EER"], "-o", lw=2, ms=5, color=COR_DEV, label="dev_shift (EER)")
axs[0].set_ylabel("EER % (menor é melhor)")
for col, cor, nm in (("treino", COR_TREINO, "treino"), ("val", COR_VAL, "validação"), ("dev", COR_DEV, "dev_shift")):
    axs[1].plot(g7.C, g7[f"{col} AUC"], "-o", lw=2, ms=5, color=cor, label=nm)
axs[1].set_ylabel("AUC")
axs[2].plot(g7.C, g7["norma de w"], "-o", lw=2, ms=5, color=COR_TREINO)
axs[2].set_ylabel("tamanho dos pesos ‖w‖")
axs[2].set_yscale("log")
for ax in axs:
    ax.set_xscale("log")
    ax.axvline(C_F0, color=COR_F0, lw=6, alpha=0.25)
    ax.set_xlabel("C (escala log; C pequeno = freio forte)")
    for Cc in GRADE_V4:
        ax.axvline(Cc, color="#d8d7d2", lw=0.8, ls="--", zorder=0)
    if ax is not axs[2]:
        ax.legend(fontsize=8)
axs[2].set_title("C maior → pesos maiores (menos freio)", fontsize=9)
fig.suptitle("G7 · Caminho do C (faixa verde: C = 0,1 do F0; tracejadas: a grade da v4). As outras regressões são só diagnóstico.",
             fontsize=11)
fig.tight_layout()
salvar_fig(fig, "g7_caminho_c")

# ======================================================================== G8 calibração
linhas = []
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
bordas = np.linspace(0, 1, 11)
for ax, k in zip(axs, ["val", "dev_shift"]):
    p = sigmoide(Z[k])
    yk = ys[k]
    cx, cy, cn = [], [], []
    for i in range(10):
        msk = (p >= bordas[i]) & ((p < bordas[i + 1]) if i < 9 else (p <= 1))
        if msk.sum():
            cx.append(p[msk].mean()); cy.append(yk[msk].mean()); cn.append(int(msk.sum()))
            linhas.append({"conjunto": k, "faixa de prob.": f"{bordas[i]:.1f} a {bordas[i + 1]:.1f}",
                           "prob. média prevista": p[msk].mean(), "fração real de falsos": yk[msk].mean(), "n": int(msk.sum())})
    ax.plot([0, 1], [0, 1], color="#b8b7b2", ls="--", lw=1, label="calibração perfeita")
    ax.plot(cx, cy, "-o", lw=2, color=COR_VAL if k == "val" else COR_DEV, ms=5, label="F0")
    for x_, y_, n_ in zip(cx, cy, cn):
        ax.annotate(f"{n_:,}".replace(",", "."), (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#52514e")
    brier = float(np.mean((p - yk) ** 2))
    ax.set_title(f"{NOME[k]} ({100 * yk.mean():.0f}% falsos) · Brier {brier:.3f}", fontsize=9)
    ax.set_xlabel("probabilidade prevista de ser falso")
    ax.set_ylabel("fração que é falsa de verdade")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("G8 · Calibração do F0 (números: clipes em cada faixa de 0,1). ⚠️ Fora da trilha.", fontsize=11)
fig.tight_layout()
salvar_fig(fig, "g8_calibracao")
salvar_csv(pd.DataFrame(linhas), "g8_calibracao")

# ======================================================================== G9 ROC e matrizes de confusão (teste)
cores_b = {"in_the_wild": "#2a78d6", "llamapartialspoof": "#eb6834", "brspeech_df": "#1baf7a",
           "asvspoof2019_la": "#6250d6", "asvspoof2021_df": "#e87ba4"}
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for bnc, cor in cores_b.items():
    fpr, tpr, _ = roc_curve(ys[bnc], Z[bnc])
    ax.plot(fpr, tpr, lw=2, color=cor, label=f"{NOME[bnc]} · AUC {roc_auc_score(ys[bnc], Z[bnc]):.3f}")
    for L, mk in (("A", "o"), ("B", "s")):
        pr = Z[bnc] >= LIM[L]
        ax.plot(pr[ys[bnc] == 0].mean(), pr[ys[bnc] == 1].mean(), mk, ms=8, color=cor, mec="white", mew=1.5)
ax.plot([0, 1], [0, 1], color="#b8b7b2", ls="--", lw=1)
ax.set_xlabel("falso alarme (reais acusados)")
ax.set_ylabel("recall de falso (falsos pegos)")
ax.set_title("G9 · ROC por banco de teste (só mostra; nada é escolhido)\n● limiar A   ■ limiar B", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
salvar_fig(fig, "g9_roc")

linhas = []
fig, axs = plt.subplots(2, 6, figsize=(17, 7))
for j, bnc in enumerate(BANCOS):
    for i, L in enumerate(("A", "B")):
        pr = (Z[bnc] >= LIM[L]).astype(int)
        mat = np.zeros((2, 2), int)
        for a in (0, 1):
            for p_ in (0, 1):
                mat[a, p_] = int(((ys[bnc] == a) & (pr == p_)).sum())
        linhas.append({"banco": bnc, "limiar": L, "real→real": mat[0, 0], "real→falso (falso alarme)": mat[0, 1],
                       "falso→real (passou)": mat[1, 0], "falso→falso": mat[1, 1]})
        ax = axs[i, j]
        lin_tot = mat.sum(1, keepdims=True).clip(min=1)
        frac = mat / lin_tot
        ax.imshow(frac, cmap="Greens", vmin=0, vmax=1)
        for a in (0, 1):
            for p_ in (0, 1):
                if lin_tot[a, 0] > 1 or mat[a].sum() > 0:
                    ax.text(p_, a, f"{mat[a, p_]:,}\n({100 * frac[a, p_]:.0f}%)".replace(",", "."), ha="center", va="center",
                            fontsize=8, color="white" if frac[a, p_] > 0.6 else "#0b0b0b")
        ax.set_xticks([0, 1], ["previsto real", "previsto falso"], fontsize=7)
        ax.set_yticks([0, 1], ["real", "falso"], fontsize=7)
        ax.grid(False)
        ax.set_title(f"{NOME[bnc]}\nlimiar {L}", fontsize=8)
fig.suptitle("G9 · Matrizes de confusão do F0 no teste (linhas = verdade; % por linha). Só mostra.", fontsize=11)
fig.tight_layout(h_pad=3)
salvar_fig(fig, "g9_confusao")
salvar_csv(pd.DataFrame(linhas), "g9_confusao")

# ======================================================================== tabela do teste (seção 8) e G10
tf = pd.read_csv(R / "v4_teste_final.csv")
tt = tf[tf.conjunto == "teste"]
mets = ["acurácia", "acurácia balanceada", "acurácia trivial", "AUC", "EER", "recall fake", "falso alarme real"]
linhas = []
for s, nome_s in (("S2", "F0 (v4 final)"), ("S1", "S1 (só camada 24)"), ("S0", "S0 (CNN da v3)")):
    for bnc in BANCOS:
        for L in ("A", "B"):
            g = tt[(tt.sistema == s) & (tt.banco == bnc) & (tt.limiar == L)].set_index("métrica")
            r = {"sistema": nome_s, "banco": bnc, "limiar": L}
            for mt in mets:
                r[mt] = g.loc[mt, "valor"]
            for mt in ("acurácia balanceada", "AUC", "falso alarme real"):
                r[f"{mt} IC95"] = f"{g.loc[mt, 'IC 2,5%']:.3f} a {g.loc[mt, 'IC 97,5%']:.3f}" if np.isfinite(g.loc[mt, "IC 2,5%"]) else ""
            r["n real"], r["n fake"] = int(g["n real"].iloc[0]), int(g["n fake"].iloc[0])
            linhas.append(r)
t8 = pd.DataFrame(linhas)
salvar_csv(t8, "teste_por_banco")
# conferência: métricas do reajuste = as gravadas
for bnc in PRINCIPAIS:
    a_ = acuracias(ys[bnc], Z[bnc], LIM["A"])["acurácia"]
    g_ = t8[(t8.sistema == "F0 (v4 final)") & (t8.banco == bnc) & (t8.limiar == "A")]["acurácia"].iloc[0]
    assert abs(a_ - g_) < 1e-9, (bnc, a_, g_)

fig, axs = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
xb = np.arange(len(BANCOS))
for ax, L in zip(axs, ("A", "B")):
    for i, (nome_s, cor) in enumerate((("F0 (v4 final)", COR_F0), ("S0 (CNN da v3)", COR_S0))):
        g = t8[(t8.sistema == nome_s) & (t8.limiar == L)].set_index("banco").loc[BANCOS]
        ax.bar(xb + (i - 1) * 0.27, g["acurácia"], 0.25, color=cor, label=nome_s)
        for x_, v in zip(xb + (i - 1) * 0.27, g["acurácia"]):
            ax.text(x_, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    g = t8[(t8.sistema == "F0 (v4 final)") & (t8.limiar == L)].set_index("banco").loc[BANCOS]
    ax.bar(xb + 0.27, g["acurácia trivial"], 0.25, color=COR_TRIV, label="chute da classe comum")
    for x_, v in zip(xb + 0.27, g["acurácia trivial"]):
        ax.text(x_, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xb, [NOME[b_] for b_ in BANCOS], fontsize=8, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"limiar {L} ({'EER da validação' if L == 'A' else 'EER do dev_shift'})")
    ax.set_ylabel("acurácia")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=9)
fig.suptitle("G10 · Acurácia por banco de teste: F0 × CNN antiga (S0) × chute da classe comum "
             "(CORAA: só reais, acurácia = 1 − falso alarme)", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
salvar_fig(fig, "g10_acuracia_bancos")

# ======================================================================== G11 falso alarme e fake que passa
linhas = []
grupos = [(f, "val", (metas["val"].source_dataset == f).values) for f in FONTES_VAL] + \
         [("dev_shift", "dev_shift", None)] + [(b_, b_, None) for b_ in BANCOS]
for nome_g, k, msk in grupos:
    m = np.ones(len(ys[k]), bool) if msk is None else msk
    for L in ("A", "B"):
        pr = Z[k][m] >= LIM[L]
        yk = ys[k][m]
        linhas.append({"grupo": nome_g, "conjunto": "validação" if k == "val" else ("dev_shift" if k == "dev_shift" else "teste"),
                       "limiar": L, "falso alarme (reais acusados)": pr[yk == 0].mean() if (yk == 0).any() else np.nan,
                       "falsos que passam (1 − recall)": 1 - pr[yk == 1].mean() if (yk == 1).any() else np.nan,
                       "n real": int((yk == 0).sum()), "n fake": int((yk == 1).sum())})
g11 = pd.DataFrame(linhas)
salvar_csv(g11, "g11_erros")
fig, axs = plt.subplots(2, 1, figsize=(15, 7.4), sharex=True)
nomes_g = [g_[0] for g_ in grupos]
xg = np.arange(len(nomes_g))
for ax, L in zip(axs, ("A", "B")):
    g = g11[g11.limiar == L].set_index("grupo").loc[nomes_g]
    ax.bar(xg - 0.2, 100 * g["falso alarme (reais acusados)"], 0.38, color=REAL_C, label="falso alarme: real acusado de ser falso")
    ax.bar(xg + 0.2, 100 * g["falsos que passam (1 − recall)"], 0.38, color=FAKE_C, label="falso que passa como real")
    for x_, v in zip(xg - 0.2, g["falso alarme (reais acusados)"]):
        if np.isfinite(v):
            ax.text(x_, 100 * v + 1, f"{100 * v:.0f}", ha="center", fontsize=7)
    for x_, v in zip(xg + 0.2, g["falsos que passam (1 − recall)"]):
        if np.isfinite(v):
            ax.text(x_, 100 * v + 1, f"{100 * v:.0f}", ha="center", fontsize=7)
    ax.axvline(3.5, color="#b8b7b2", lw=1); ax.axvline(4.5, color="#b8b7b2", lw=1)
    ax.set_ylabel("% de erro")
    ax.set_ylim(0, 105)
    ax.set_title(f"limiar {L} = {LIM[L]:.2f} ({'EER da validação' if L == 'A' else 'EER do dev_shift'})", fontsize=9)
axs[0].legend(fontsize=8, loc="upper left")
axs[1].set_xticks(xg, [NOME[n_] for n_ in nomes_g], rotation=20, fontsize=8, ha="right")
fig.suptitle("G11 · Os dois erros do F0, por fonte (validação), dev_shift e banco de teste (só mostra)", fontsize=11)
fig.tight_layout()
salvar_fig(fig, "g11_erros")

# ======================================================================== G12 AUC por duração original
linhas = []
for k in ["dev_shift"] + [b_ for b_ in BANCOS if b_ != "coraa"]:
    fx = np.asarray(faixa_duracao(metas[k].duration_s.values))
    for f in FAIXAS_DURACAO:
        m = fx == f
        yk = ys[k][m]
        duas = m.any() and len(np.unique(yk)) == 2
        linhas.append({"conjunto": k, "faixa": f, "n": int(m.sum()), "n real": int((yk == 0).sum()),
                       "n fake": int((yk == 1).sum()), "AUC": roc_auc_score(yk, Z[k][m]) if duas else np.nan})
g12 = pd.DataFrame(linhas)
salvar_csv(g12, "g12_duracao")
fig, ax = plt.subplots(figsize=(12, 4.2))
ks = list(dict.fromkeys(g12.conjunto))
xk = np.arange(len(ks))
for i, (f, cor) in enumerate(zip(FAIXAS_DURACAO, ["#eda100", "#1baf7a", "#2a78d6"])):
    g = g12[g12.faixa == f].set_index("conjunto").loc[ks]
    ax.bar(xk + (i - 1) * 0.27, g.AUC, 0.25, color=cor, label=f"duração original {f}")
    for x_, v, n_ in zip(xk + (i - 1) * 0.27, g.AUC, g.n):
        ax.text(x_, (v if np.isfinite(v) else 0.5) + 0.005, f"{v:.3f}\nn={n_}" if np.isfinite(v) else f"—\nn={n_}",
                ha="center", fontsize=6.5)
ax.set_xticks(xk, [NOME[k_] + ("" if k_ == "dev_shift" else "\n(teste, só mostra)") for k_ in ks], fontsize=8)
ax.set_ylim(0.5, 1.04)
ax.set_ylabel("AUC")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.set_title("G12 · AUC do F0 por faixa de duração ORIGINAL do áudio (antes do corte de silêncio e do limite de 8 s)")
fig.tight_layout()
salvar_fig(fig, "g12_duracao")

# ======================================================================== G13 recall por gerador (validação)
mvf = metas["val"]
fk = (mvf.label == 1).values
linhas = []
for (fonte, gen), g in mvf[fk].groupby(["source_dataset", "generator"]):
    zz = Z["val"][g.index.values]
    linhas.append({"fonte": fonte, "gerador": gen, "n fakes": len(g),
                   "recall (lim. A)": float((zz >= LIM["A"]).mean()), "recall (lim. B)": float((zz >= LIM["B"]).mean())})
g13 = pd.DataFrame(linhas).sort_values(["fonte", "recall (lim. A)"]).reset_index(drop=True)
salvar_csv(g13, "g13_geradores")
fig, ax = plt.subplots(figsize=(10, 0.28 * len(g13) + 1.6))
yy = np.arange(len(g13))
cor_f = {"dfadd": "#2a78d6", "shiftyspeech": "#1baf7a", "codecfake_plus": "#eb6834"}
ax.barh(yy, 100 * g13["recall (lim. A)"], 0.6, color=[cor_f[f] for f in g13.fonte])
ax.plot(100 * g13["recall (lim. B)"], yy, "|", ms=10, mew=2, color="#0b0b0b", label="limiar B")
ax.set_yticks(yy, [f"{NOME[f]} · {gn} (n={n})" for f, gn, n in zip(g13.fonte, g13.gerador, g13["n fakes"])], fontsize=7.5)
ax.set_xlabel("% dos falsos pegos (recall); barra = limiar A, traço = limiar B")
ax.set_xlim(0, 103)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("G13 · Recall do F0 por gerador na validação (cor = fonte; n pequeno = número instável)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
salvar_fig(fig, "g13_geradores")

# ======================================================================== G14 Meta × Microsoft por camada
c5 = pd.read_csv(R / "v5_comparacao_camadas.csv")
c6 = pd.read_csv(R / "v6_e1_camadas.csv")
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
axs[0].plot(c5.camada, 100 * c5["val EER média por fonte xlsr"], "-o", ms=4, lw=2, color=COR_META, label="Meta (XLS-R 300M)")
axs[0].plot(c5.camada, 100 * c5["val EER média por fonte wavlm"], "-o", ms=4, lw=2, color=COR_MS, label="Microsoft (WavLM Large)")
axs[0].set_ylabel("EER média por fonte na validação (%)")
axs[0].set_title("v5 (David Chung): validação", fontsize=9)
axs[1].plot(c5.camada, c5["dev AUC xlsr"], "-o", ms=4, lw=2, color=COR_META, label="Meta")
axs[1].plot(c5.camada, c5["dev AUC wavlm"], "-o", ms=4, lw=2, color=COR_MS, label="Microsoft")
axs[1].set_ylabel("AUC no dev_shift")
axs[1].set_title("v5 (David Chung): dev_shift", fontsize=9)
for mdl, cor, nm in (("xlsr", COR_META, "Meta"), ("wavlm", COR_MS, "Microsoft")):
    g = c6[c6.modelo == mdl]
    axs[2].plot(g.camada, g["AUC pior fonte"], "-o", ms=4, lw=2, color=cor, label=nm)
axs[2].set_ylabel("AUC da pior fonte (val + dev_shift)")
axs[2].set_title("v6 (nesta máquina, mesmos clipes)", fontsize=9)
for ax in axs:
    ax.set_xlabel("camada")
    ax.set_xticks(range(0, 25, 4))
    ax.legend(fontsize=8)
    for c_ in CAMADAS:
        ax.axvspan(c_ - 0.4, c_ + 0.4, color=COR_F0, alpha=0.12, lw=0)
fig.suptitle("G14 · Onde estão as pistas em cada modelo: AUC/EER por camada, Meta × Microsoft "
             "(v5 feita pelo colega David Chung; faixas verdes: camadas do F0)", fontsize=11)
fig.tight_layout()
salvar_fig(fig, "g14_meta_microsoft")
salvar_csv(c5.merge(c6.pivot(index="camada", columns="modelo", values="AUC pior fonte")
                    .rename(columns={"xlsr": "v6 AUC pior fonte xlsr", "wavlm": "v6 AUC pior fonte wavlm"}).reset_index(),
                    on="camada"), "g14_meta_microsoft")

# ======================================================================== tabelas das seções 6 e 9
c4 = pd.read_csv(R / "v4_cabecas.csv")
c4 = c4[c4["cabeça"] != "atalho_estatisticas"]
t6a = c4[["cabeça", "tipo", "seed", "camadas", "camada", "C", "val EER média por fonte", "dev AUC", "dev EER",
          "dev acurácia balanceada"]].copy()
t6a.insert(0, "rodada", "v4 (regra: menor EER média por fonte na validação)")
e4 = pd.read_csv(R / "v6_e4_cabecas.csv")
t6b = e4[["sistema", "cabeça", "seed", "parâmetros", "segundos", "AUC pior fonte", "fonte pior", "AUC codecfake_plus",
          "AUC dev_shift", "val EER média por fonte (3)"]].copy()
t6b.insert(0, "rodada", "v6 E4 (regra: desafiante; o mais simples fica se não houver diferença relevante)")
salvar_csv(t6a, "cabecas_v4")
salvar_csv(t6b, "cabecas_v6_e4")

v6t = pd.read_csv(R / "v6_teste_final.csv")
v6t = v6t[v6t.conjunto == "teste"]
linhas = []
for s in ("F0", "F1", "F2"):
    for bnc in BANCOS:
        for L in ("A", "B"):
            g = v6t[(v6t.sistema == s) & (v6t.banco == bnc) & (v6t.limiar == L)].set_index("métrica")
            if len(g):
                linhas.append({"sistema": s, "descrição": g["descrição"].iloc[0], "banco": bnc, "limiar": L,
                               **{mt: g.loc[mt, "valor"] for mt in ("acurácia", "acurácia balanceada", "acurácia trivial",
                                                                     "AUC", "falso alarme real") if mt in g.index}})
salvar_csv(pd.DataFrame(linhas), "f0_f1_f2")

# ======================================================================== resumo
resumo = {"F0": {"descrição": "XLS-R 300M congelado (Meta), média no tempo das saídas 5, 6 e 7, padronizadas com o "
                              "treino, concatenadas (3.072 números) + regressão logística",
                 "camadas": CAMADAS, "C": C_F0, "solver": "lbfgs (padrão do scikit-learn)", "max_iter": RL_MAX_ITER,
                 "tol": 1e-4, "class_weight": "nenhum (o treino já é 50/50)", "n_treino": int(len(Xtr)),
                 "n_pesos": int(len(w)), "intercepto": b0, "iterações": int(f0.n_iter_[0]),
                 "limiar_A_logit": LIM["A"], "limiar_A_prob": float(sigmoide(LIM["A"])),
                 "limiar_B_logit": LIM["B"], "limiar_B_prob": float(sigmoide(LIM["B"]))},
          "reproducao": repro.to_dict("records"), "modelo_hf": modelo_info,
          "g1": {"cos(v1, w)": float(v1 @ wh), "variância do complemento explicada por v1": var_exp,
                 "PCA ajustado em": "20.000 clipes do treino (seed 42), depois de remover a direção w"},
          "g3": proj_info, "melhor_camada_v4": MELHOR,
          "controle_rotulos_embaralhados": ler_json(R / "v4_atalhos.json")["d_controle_rotulos_embaralhados"],
          "minutos": round((time.time() - T0) / 60, 1), "pico_memoria_gb": round(PICO[0], 2)}
(R / "v4final_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"FIM. pico de memória {PICO[0]:.2f} GB")
