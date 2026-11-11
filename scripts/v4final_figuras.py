#!/usr/bin/env python3
"""Figuras e tabelas do documento final do sistema v4 (F0), no formato de apresentação. Só DIAGNÓSTICO e
APRESENTAÇÃO: nada aqui escolhe modelo, camada, C nem limiar.

- Lê os caches existentes (data/processed/v4_cache e v4_cache_teste); não extrai nada, não baixa nada.
- Reajusta a cabeça F0 (= S2 da v4: camadas 5, 6, 7, média no tempo, padronizadas com o treino; regressão
  logística C = 0,1, lbfgs, max_iter 1000, tol 1e-4) e CONFERE que reproduz as notas gravadas na v4
  (results/v4_teste_notas.parquet, coluna z_S2). Se não reproduzir, para sem desenhar nada.
- As regressões extras (curva de aprendizado, caminho do C) são só para os gráficos: não mudam camada, C nem
  limiar de nada. Os bancos de teste só aparecem para MOSTRAR (a cabeça e os cortes foram fixados antes).
- Lê 2 áudios de data/raw (um real e um gerado por IA, da validação) só para a figura didática do log-mel.
- Grava:
    results/slides/NN_nome.png (13,33 x 7,5 pol., dpi 180, para os slides);
    results/v4final_<nome>.png (o mesmo desenho em dpi 110, embutido no notebook);
    results/v4final_*.csv, results/v4final_resumo.json, results/v4final_f0_coeficientes.npz;
    results/slides/grupos_bancos.csv, hiperparametros_versoes.csv e INDEX.md.
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
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.analysis import FAIXAS_DURACAO, faixa_duracao  # noqa: E402
from src.cache_v4 import Conjunto  # noqa: E402
from src.metrics import acuracias, eer_e_limiar, eer_media_por_fonte, sigmoide  # noqa: E402

warnings.filterwarnings("ignore")
LIMITE_GB = 10.0
R = Path("results")
SLIDES = R / "slides"
SLIDES.mkdir(exist_ok=True)
DPI_SLIDE, DPI_NB = 180, 110
TAM = (13.33, 7.5)
T0 = time.time()

# ---------------- estilo (paleta pedida; uma cor por significado) ----------------
REAL_C, IA_C, NEUTRO = "#2B63C9", "#E0662C", "#5E6670"
META, MS = "#1F8A7A", "#6A4FC4"
FUNDO, TEXTO, GRADE = "#FBFAF6", "#22272E", "#E8E5DC"
REAL_CLARO, IA_CLARO, META_CLARO = "#9DB8EA", "#F2B394", "#8CCBBF"
ESTUDIO_C, INTERNET_C, TREINO_C = NEUTRO, "#A23B72", "#B4BAC1"     # domínio: estúdio (cinza) × internet (vinho)
CHUTE_C = "#CDD1D6"
COR_PAPEL = {"treino": "#3E4651", "validação": "#8C949E", "dev_shift": INTERNET_C, "teste principal": "#B07D12",
             "teste de falso alarme": "#2E7D32", "teste secundário": "#C9CDD2"}
COR_GRUPO = {"in_the_wild": INTERNET_C, "llamapartialspoof": "#B07D12", "brspeech_df": "#2E7D32"}

_fontes = {f.name for f in font_manager.fontManager.ttflist}
FONTE = "IBM Plex Sans" if "IBM Plex Sans" in _fontes else "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONTE, "font.size": 14, "text.color": TEXTO, "axes.labelcolor": TEXTO,
    "axes.titlesize": 16, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "legend.frameon": False, "figure.facecolor": FUNDO, "axes.facecolor": FUNDO,
    "savefig.facecolor": FUNDO, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA0A6", "xtick.color": "#4A5058", "ytick.color": "#4A5058", "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.titlepad": 10})

NOME = {"in_the_wild": "In-the-Wild", "llamapartialspoof": "LlamaPartialSpoof", "brspeech_df": "BRSpeech-DF",
        "coraa": "CORAA", "asvspoof2019_la": "ASVspoof 2019", "asvspoof2021_df": "ASVspoof 2021 DF",
        "dfadd": "DFADD", "shiftyspeech": "ShiftySpeech", "codecfake_plus": "CodecFake+", "librispeech": "LibriSpeech",
        "val": "validação", "dev_shift": "dev_shift"}
GRUPO = {"val": "Estúdio", "dev_shift": "Internet", "in_the_wild": "Internet", "llamapartialspoof": "Fakes parciais",
         "brspeech_df": "Português", "coraa": "Português, só real", "asvspoof2019_la": "Secundário",
         "asvspoof2021_df": "Secundário"}
DETALHE = {"val": "validação", "dev_shift": "dev_shift", "in_the_wild": "teste: In-the-Wild",
           "llamapartialspoof": "LlamaPartialSpoof", "brspeech_df": "BRSpeech-DF", "coraa": "CORAA",
           "asvspoof2019_la": "ASVspoof 2019", "asvspoof2021_df": "ASVspoof 2021 DF"}
BANCOS = ["in_the_wild", "llamapartialspoof", "brspeech_df", "coraa", "asvspoof2019_la", "asvspoof2021_df"]
PRINCIPAIS = ["in_the_wild", "llamapartialspoof", "brspeech_df"]
FONTES_VAL = ["dfadd", "shiftyspeech", "codecfake_plus", "librispeech"]
INDEX = []          # uma linha por figura dos slides (results/slides/INDEX.md)


def br(v, d=2):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def mil(n):
    return f"{int(n):,}".replace(",", ".")


def pct(v, d=0):
    return br(100 * v, d) + "%"


def virgula(ax, eixo="y", d=2):
    f = FuncFormatter(lambda v, _: br(v, d))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(f)


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


# ---------------- moldura comum das figuras ----------------
def quadro(titulo, subtitulo, nrows=1, ncols=1, top=0.74, bottom=0.12, left=0.075, right=0.98,
           wspace=0.25, hspace=0.45, **kw):
    """Figura 13,33 x 7,5 com TÍTULO (a conclusão) e SUBTÍTULO (o que olhar) no alto, à esquerda."""
    fig, axs = plt.subplots(nrows, ncols, figsize=TAM, squeeze=False, **kw)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=wspace, hspace=hspace)
    fig.text(0.012, 0.975, titulo, fontsize=24, fontweight="bold", va="top", color=TEXTO)
    fig.text(0.012, 0.905, subtitulo, fontsize=16, va="top", color=NEUTRO)
    return fig, axs


def legenda(fig, handles, y=0.855, ncol=None, x=0.012, loc="upper left"):
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x, y), ncol=ncol or len(handles), fontsize=14,
               frameon=False, handlelength=1.6, columnspacing=1.6, borderaxespad=0)


def titulo_painel(ax, grupo, detalhe=None, pad=30):
    ax.set_title(grupo, loc="left", fontsize=16, fontweight="bold", pad=pad)
    if detalhe:
        ax.text(0, 1.03, detalhe, transform=ax.transAxes, fontsize=13, color=NEUTRO, va="bottom")


def salvar(fig, slide=None, nb=None, mostra="", olhar=""):
    if slide:
        p = SLIDES / f"{slide}.png"
        fig.savefig(p, dpi=DPI_SLIDE)
        INDEX.append({"figura": f"{slide}.png", "o que mostra": mostra, "o que olhar": olhar})
        log(f"slide {p} ({p.stat().st_size / 1024:.0f} KB)")
    if nb:
        p = R / f"v4final_{nb}.png"
        fig.savefig(p, dpi=DPI_NB)
        log(f"notebook {p} ({p.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


def salvar_csv(df, nome):
    p = R / f"v4final_{nome}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    log(f"tabela {p} ({len(df)} linhas)")


def ler_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def h_ponto(cor, rotulo):
    return Line2D([], [], ls="", marker="o", ms=11, color=cor, label=rotulo)


def h_caixa(cor, rotulo, **kw):
    return Patch(facecolor=cor, edgecolor=kw.pop("edgecolor", "none"), label=rotulo, **kw)


# ======================================================================== 1. o que é o F0 (lido do disco)
res_v4 = ler_json(R / "v4_resumo.json")
mc = ler_json(R / "v4_camadas_concluido.json")
cab = [r for r in mc["concat"] if r["cabeça"] == "concat_top3"][0]
assert res_v4["cabeça_final_pela_validação"] == "concat_top3"
CAMADAS, C_F0 = list(cab["camadas"]), float(cab["C"])
tres = ler_json(R / "v4_teste_final_resumo.json")
LIM = {"A": float(tres["limiares_logit"]["S2"]["A"]), "B": float(tres["limiares_logit"]["S2"]["B"])}
RL_MAX_ITER = 1000          # célula config da v4 (notebooks/versions/v4_ssl.ipynb)
log(f"F0: camadas {CAMADAS}, C = {C_F0}, limiares (logit) A = {LIM['A']:.4f}, B = {LIM['B']:.4f}; fonte {FONTE}")
CORTE = {"A": f"Corte A (calibrado no estúdio) = {br(LIM['A'])}",
         "B": f"Corte B (calibrado na internet) = {br(LIM['B'])}"}
EST_CORTE = {"A": dict(color=TEXTO, ls="--", lw=2.2), "B": dict(color=NEUTRO, ls=":", lw=2.6)}

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

# eixo y do plano (G1 / figura 03): a direção de maior variação DEPOIS de tirar a direção de w
wh = w / np.linalg.norm(w)
rng = np.random.default_rng(42)
sub = rng.choice(len(Xtr), 20000, replace=False)
Xc = Xtr[sub].astype(np.float64)
media_tr = Xc.mean(0)
Xc = Xc - media_tr
Xc = Xc - np.outer(Xc @ wh, wh)
_, sv, vt = np.linalg.svd(Xc, full_matrices=False)
v1 = vt[0]
var_exp = float(sv[0] ** 2 / (sv ** 2).sum())
del Xc


def eixo_y(x):
    return (x - media_tr) @ v1


def amostra_idx(n, maximo=6000, seed=42):
    r = np.random.default_rng(seed)
    return np.arange(n) if n <= maximo else np.sort(r.choice(n, maximo, replace=False))


todos_z = np.concatenate([Z[k] for k in Z])
XLIM = tuple(np.percentile(todos_z, [0.5, 99.5]) + np.array([-2, 2]))


def n_classes(y):
    return int((y == 0).sum()), int((y == 1).sum())


def detalhe_n(k, y, extra=""):
    nr, nf = n_classes(y)
    partes = [f"{mil(nr)} reais"] + ([f"{mil(nf)} IA"] if nf else [])
    return f"({DETALHE[k]}) · " + " · ".join(partes) + extra


def linhas_corte(ax, rotular=False, ymax=None):
    for L in ("A", "B"):
        ax.axvline(LIM[L], **EST_CORTE[L], zorder=3)
        if rotular:
            ax.text(LIM[L], 1.0, f" {L}", transform=ax.get_xaxis_transform(), fontsize=14, fontweight="bold",
                    color=EST_CORTE[L]["color"], ha="left", va="top")


# ======================================================================== 01 composição do banco
log("01: composição do banco")
man = pd.read_parquet("data/processed/manifest.parquet",
                      columns=["source_dataset", "split", "label", "eval_subset", "role"])


def papel_de(r):
    if r.split == "train":
        return "treino"
    if r.split == "val":
        return "validação"
    if r.split == "test" and r.eval_subset == "dev_shift":
        return "dev_shift"
    if r.split == "test" and r.source_dataset in PRINCIPAIS:
        return "teste principal"
    if r.split == "test" and r.source_dataset == "coraa":
        return "teste de falso alarme"
    if r.split == "test":
        return "teste secundário"
    return "excluído"


man["papel"] = [papel_de(r) for r in man.itertuples(index=False)]
cont = man.groupby(["papel", "label"]).size().unstack(fill_value=0)
ORDEM_PAPEL = ["treino", "validação", "dev_shift", "teste principal", "teste de falso alarme", "teste secundário"]
USADOS = {"treino": len(tr), "validação": len(metas["val"]), "dev_shift": len(metas["dev_shift"]),
          "teste principal": sum(len(metas[b]) for b in PRINCIPAIS), "teste de falso alarme": len(metas["coraa"]),
          "teste secundário": len(metas["asvspoof2019_la"]) + len(metas["asvspoof2021_df"])}
n_excl = int(cont.loc["excluído"].sum())
assert n_excl == 27535, n_excl

# tabela grupos_bancos.csv (descrições tiradas de docs/audit_report.md e docs/dataset_overview.md)
INFO_BANCO = {
    "asvspoof2019_la": ("Estúdio e leitura (treino) / Secundário (teste)",
                        "Desafio ASVspoof 2019, acesso lógico: falas do VCTK lidas em estúdio, com locutores anonimizados, e versões geradas por 19 sistemas (A01 a A19).",
                        "inglês", "2019 (do nome do banco)", "antigos (docs/dataset_overview.md: 'os antigos do ASVspoof 2019')"),
    "dfadd": ("Geradores novos", "Falas do VCTK gravadas em estúdio e as mesmas frases geradas por 5 TTS (GradTTS, MatchaTTS, NaturalSpeech2, PflowTTS, StyleTTS2).",
              "inglês", "não documentado", "novos (agrupamento do projeto)"),
    "shiftyspeech": ("Geradores novos", "Leitura de audiolivro (LibriSpeech) e a mesma fala ressintetizada por 3 vocoders (hifigan, vocos, wavegrad).",
                     "inglês", "não documentado", "novos (agrupamento do projeto)"),
    "codecfake_plus": ("Geradores novos", "Amostras de páginas de demonstração de 17 TTS baseados em codec neural (só a parte CoSG), com os reais das mesmas demos.",
                       "inglês", "não documentado", "novos (agrupamento do projeto)"),
    "librispeech": ("Estúdio e leitura", "Leitura de audiolivros (train-clean-100); só vozes reais.", "inglês",
                    "não documentado", "— (só real)"),
    "in_the_wild": ("Internet", "Falas de 54 figuras públicas coletadas na internet, reais e falsas; o gerador dos falsos não é informado.",
                    "inglês", "não documentado", "não documentado"),
    "llamapartialspoof": ("Fakes parciais", "Reais do LibriTTS e falsos de 6 TTS, inteiros ou parciais (trechos gerados dentro de uma fala real).",
                          "inglês", "não documentado", "não documentado"),
    "asvspoof2021_df": ("Secundário", "Desafio ASVspoof 2021, deepfake: ataques gerados e comprimidos por codecs; só a parte 1 de 4.",
                        "inglês", "2021 (do nome do banco)", "não documentado"),
    "brspeech_df": ("Português", "Reais de audiolivro (LibriVox, segundo o card) e falsos de 5 TTS; mistura pt-BR e pt-PT sem indicação por arquivo.",
                    "português", "não documentado", "não documentado"),
    "coraa": ("Português", "Corpus de reconhecimento de fala em português brasileiro (entrevistas, aulas, TEDx); só vozes reais.",
              "português (pt-BR)", "não documentado", "— (só real)"),
}
linhas = []
for (bnc, papel), g in man[man.papel != "excluído"].groupby(["source_dataset", "papel"]):
    grp, oque, idioma, ano, geracao = INFO_BANCO[bnc]
    if bnc == "asvspoof2019_la":
        grp = "Estúdio e leitura" if papel == "treino" else "Secundário"
    nr, nf = int((g.label == 0).sum()), int((g.label == 1).sum())
    linhas.append({"banco": NOME[bnc], "grupo": grp, "o que é": oque,
                   "real/IA": "real e IA" if nr and nf else ("só real" if nr else "só IA"), "idioma": idioma,
                   "ano": ano, "geração dos geradores": geracao, "papel": papel, "nº real": nr, "nº IA": nf})
gb = pd.DataFrame(linhas)
gb["_o"] = gb.papel.map({p: i for i, p in enumerate(ORDEM_PAPEL)})
gb = gb.sort_values(["_o", "banco"]).drop(columns="_o").reset_index(drop=True)
gb.to_csv(SLIDES / "grupos_bancos.csv", index=False, encoding="utf-8")
log(f"tabela {SLIDES / 'grupos_bancos.csv'} ({len(gb)} linhas)")

fig = plt.figure(figsize=TAM)
fig.text(0.012, 0.975, "Treinamos em estúdio e testamos na internet e em português", fontsize=24,
         fontweight="bold", va="top")
fig.text(0.012, 0.905, f"Quantos áudios há em cada papel (todo o banco) e de onde vêm; {mil(n_excl)} áudios "
         "excluídos ficaram de fora", fontsize=16, va="top", color=NEUTRO)
ax = fig.add_axes([0.20, 0.10, 0.33, 0.66])
yy = np.arange(len(ORDEM_PAPEL))[::-1]
reais = np.array([cont.loc[p, 0] for p in ORDEM_PAPEL])
ias = np.array([cont.loc[p, 1] for p in ORDEM_PAPEL])
ax.barh(yy, reais, 0.62, color=REAL_C)
ax.barh(yy, ias, 0.62, left=reais, color=IA_C)
for y_, r_, f_, p in zip(yy, reais, ias, ORDEM_PAPEL):
    ax.text(r_ + f_ + 4000, y_ + 0.12, f"{mil(r_)} reais · {mil(f_)} IA" if f_ else f"{mil(r_)} reais",
            va="center", fontsize=13)
    ax.text(r_ + f_ + 4000, y_ - 0.24, f"usados: {mil(USADOS[p])}", va="center", fontsize=13, color=NEUTRO)
ax.set_yticks(yy, ORDEM_PAPEL)
for lbl, p in zip(ax.get_yticklabels(), ORDEM_PAPEL):
    lbl.set_color(COR_PAPEL[p] if p != "teste secundário" else NEUTRO)
    lbl.set_fontweight("bold")
ax.set_xlim(0, 4.6e5)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1000)} mil" if v else "0"))
ax.set_xlabel("nº de áudios")
ax.grid(axis="y", visible=False)
ax.set_title("(a) áudios por papel", loc="left", fontsize=16, fontweight="bold", pad=12)
legenda(fig, [h_caixa(REAL_C, "real"), h_caixa(IA_C, "gerada por IA")], y=0.855, x=0.012)
# (b) uma faixa por banco
axb = fig.add_axes([0.575, 0.06, 0.415, 0.72])
axb.set_axis_off()
axb.set_xlim(0, 1)
FAIXAS = [("ASVspoof 2019", "Estúdio e leitura", ["treino", "teste secundário"], "real + IA · inglês · geradores antigos"),
          ("LibriSpeech", "Estúdio e leitura", ["treino", "validação"], "só real · inglês"),
          ("DFADD", "Geradores novos", ["treino", "validação"], "real + IA · inglês"),
          ("ShiftySpeech", "Geradores novos", ["treino", "validação"], "real + IA · inglês"),
          ("CodecFake+", "Geradores novos", ["treino", "validação"], "real + IA · inglês"),
          ("In-the-Wild", "Internet", ["dev_shift", "teste principal"], "real + IA · inglês"),
          ("LlamaPartialSpoof", "Fakes parciais", ["teste principal"], "real + IA · inglês"),
          ("BRSpeech-DF", "Português", ["teste principal"], "real + IA · português"),
          ("CORAA", "Português", ["teste de falso alarme"], "só real · português"),
          ("ASVspoof 2021 DF", "Secundário", ["teste secundário"], "real + IA · inglês")]
n_f = len(FAIXAS)
axb.set_ylim(-0.5, n_f - 0.3)
axb.text(0, n_f - 0.25, "(b) cada banco: grupo, conteúdo e papel", fontsize=16, fontweight="bold", va="bottom")
for i, (nm, grp, papeis, desc) in enumerate(FAIXAS):
    y_ = n_f - 1 - i
    if i % 2 == 0:
        axb.add_patch(FancyBboxPatch((0, y_ - 0.45), 1, 0.9, boxstyle="square,pad=0", fc="#F1EFE8", ec="none"))
    axb.text(0.01, y_ + 0.12, grp, fontsize=14, fontweight="bold", va="center")
    axb.text(0.01, y_ - 0.22, f"({nm})", fontsize=13, color=NEUTRO, va="center")
    axb.text(0.37, y_ - 0.22, desc, fontsize=13, color=TEXTO, va="center")
    x_ = 0.37
    for p in papeis:
        largura = 0.0165 * len(p) + 0.03
        axb.add_patch(FancyBboxPatch((x_, y_ + 0.0), largura, 0.3, boxstyle="round,pad=0.01,rounding_size=0.05",
                                     fc=COR_PAPEL[p], ec="none"))
        axb.text(x_ + largura / 2, y_ + 0.15, p, fontsize=13, ha="center", va="center",
                 color=TEXTO if p == "teste secundário" else "white")
        x_ += largura + 0.02
salvar(fig, "01_composicao_banco", "01_composicao_banco",
       "Quantos áudios reais e gerados por IA há em cada papel (treino, validação, dev_shift, testes) e de que banco vem cada um.",
       "O treino e a validação são inglês de estúdio ou leitura; a internet (dev_shift) decide entre versões; os testes são internet, fakes parciais e português.")
del man

# ======================================================================== 02 log-mel (2 áudios da validação)
log("02: log-mel")
import torch  # noqa: E402
import torchaudio  # noqa: E402

from src.preproc import ler_mono_16k  # noqa: E402

mv_all = pd.read_parquet("data/processed/manifest.parquet",
                         columns=["path", "label", "source_dataset", "split", "duration_s", "generator"])
dv_ = mv_all[(mv_all.split == "val") & (mv_all.source_dataset == "dfadd")].copy()
del mv_all
dv_["frase"] = dv_.path.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
dv_.loc[dv_.label == 1, "frase"] = dv_.loc[dv_.label == 1, "frase"].str.rsplit("_", n=1).str[0]
par = dv_[(dv_.label == 0) & (dv_.duration_s >= 4.5) & (dv_.duration_s <= 7)].merge(
    dv_[(dv_.label == 1) & (dv_.duration_s >= 4.2)], on="frase", suffixes=("_r", "_f")).sort_values(["frase", "generator_f"])
par = par.iloc[int(np.random.default_rng(42).integers(len(par)))]
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=64)
spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, power=2.0)
fig, axs = quadro("Como o áudio vira uma imagem para a CNN das versões 1 a 3",
                  "Mesma frase e mesmo locutor (VCTK): à esquerda, a voz real; à direita, a gerada por IA. "
                  "4 s do meio de cada áudio", 3, 2, top=0.77, bottom=0.08, left=0.10, right=0.875, hspace=0.62, wspace=0.12)
ims = []
for col, (lado, cor, nome) in enumerate((("r", REAL_C, "Voz real"), ("f", IA_C, f"Gerada por IA ({par['generator_f']})"))):
    x, _ = ler_mono_16k(par[f"path_{lado}"])
    meio = (len(x) - 64000) // 2
    x = x[max(meio, 0): max(meio, 0) + 64000]
    tt = np.arange(len(x)) / 16000
    ax = axs[0, col]
    ax.plot(tt, x.numpy(), color=cor, lw=0.5)
    titulo_painel(ax, nome, "1) forma de onda: a pressão do ar a cada instante", pad=26)
    ax.set_xlim(0, 4)
    ax.set_yticks([])
    ax.grid(False)
    S = 10 * torch.log10(spec(x) + 1e-10).numpy()
    ax = axs[1, col]
    ax.imshow(S, origin="lower", aspect="auto", extent=[0, 4, 0, 8], cmap="magma", vmin=S.max() - 80, vmax=S.max())
    ax.text(0, 1.03, "2) espectrograma: 257 faixas de frequência iguais", transform=ax.transAxes, fontsize=13,
            color=NEUTRO, va="bottom")
    ax.grid(False)
    M = torch.log(mel(x) + 1e-6).numpy()
    ax = axs[2, col]
    ims.append(ax.imshow(M, origin="lower", aspect="auto", extent=[0, 4, 0, 64], cmap="magma"))
    ax.text(0, 1.03, "3) log-mel: 64 faixas como o ouvido; brilho em escala log", transform=ax.transAxes,
            fontsize=13, color=NEUTRO, va="bottom")
    ax.grid(False)
    ax.set_xlabel("tempo (s)")
axs[1, 0].set_ylabel("frequência\n(kHz)", fontsize=14)
axs[2, 0].set_ylabel("faixa mel\n(0 a 63)", fontsize=14)
axs[0, 0].set_ylabel("amplitude", fontsize=14)
for ax in axs[:, 1]:
    ax.set_yticklabels([])
for ax in axs.ravel():
    virgula(ax, "x", 1)
fig.text(0.887, 0.40, "graves\nembaixo,\nagudos\nem cima\n\nbrilho =\nintensi-\ndade", fontsize=13, color=NEUTRO, va="center")
salvar(fig, "02_log_mel", "02_log_mel",
       "Um áudio real e um gerado por IA (mesma frase, validação): forma de onda, espectrograma e log-mel de 64 faixas.",
       "O log-mel é a 'imagem' que a CNN das versões 1 a 3 recebia; o sistema final (v4) lê a onda direto com o XLS-R.")
LOGMEL_INFO = {"real": par["path_r"].rsplit("/", 1)[-1], "IA": par["path_f"].rsplit("/", 1)[-1],
               "gerador": par["generator_f"]}

# ======================================================================== 03 plano da regressão (2 painéis)
log("03: plano da regressão")


def painel_plano(ax, z, yv, lab, mostrar_regioes=True, s=6):
    idx = amostra_idx(len(z))
    ordem = np.random.default_rng(0).permutation(idx)
    ax.scatter(z[ordem], yv[ordem], c=np.where(lab[ordem] == 1, IA_C, REAL_C), s=s, alpha=0.45, linewidths=0,
               rasterized=True)
    linhas_corte(ax)
    ax.set_xlim(*XLIM)
    if mostrar_regioes:
        ax.axvspan(LIM["A"], LIM["B"], color="#ECE9E0", zorder=0, lw=0)
        tr_ = ax.get_xaxis_transform()
        ax.text((XLIM[0] + LIM["A"]) / 2, 0.97, "lado real\n(nos dois cortes)", transform=tr_, ha="center",
                va="top", fontsize=13, color=REAL_C, fontweight="bold")
        ax.text((LIM["B"] + XLIM[1]) / 2, 0.97, "lado IA\n(nos dois cortes)", transform=tr_, ha="center",
                va="top", fontsize=13, color=IA_C, fontweight="bold")
    return len(idx)


fig, axs = quadro("Nas vozes da internet, parte dos reais passa do corte",
                  "Cada ponto é um áudio; quanto mais à direita, mais a regressão acha que é IA", 1, 2,
                  top=0.67, bottom=0.13, left=0.10, wspace=0.12, sharey=True)
for ax, k in zip(axs[0], ["val", "dev_shift"]):
    nmost = painel_plano(ax, Z[k], eixo_y(Xs[k]), ys[k])
    titulo_painel(ax, GRUPO[k], detalhe_n(k, ys[k], f" (mostrados: {mil(nmost)})"))
    ax.set_xlabel("nota da regressão (quanto maior, mais parece IA)")
axs[0, 0].set_ylabel("variação entre os áudios\n(só espalha os pontos)")
legenda(fig, [h_ponto(REAL_C, "real"), h_ponto(IA_C, "gerada por IA"),
              Line2D([], [], **EST_CORTE["A"], label=CORTE["A"]), Line2D([], [], **EST_CORTE["B"], label=CORTE["B"]),
              h_caixa("#ECE9E0", "faixa: IA pelo corte A, real pelo B")],
        y=0.85, ncol=3)
salvar(fig, "03_plano_regressao", "03_plano_regressao",
       "A nota da regressão (eixo x) de cada áudio no estúdio (validação) e na internet (dev_shift), com os cortes A e B.",
       "No estúdio quase todo real fica à esquerda do corte A; na internet, uma cauda de reais passa dele, e o corte B (mais à direita) os recupera.")

# G1 de 6 painéis (mantido no notebook) e G1 nos bancos de teste
mv = metas["val"]
paineis = [(f, (mv.source_dataset == f).values) for f in FONTES_VAL] + [("val", None), ("dev_shift", None)]
fig, axs = quadro("Cada fonte do estúdio fica bem separada; na internet, não",
                  "Mesmo desenho da figura anterior, fonte por fonte da validação; a fronteira é a reta vertical do corte",
                  2, 3, top=0.73, bottom=0.10, left=0.09, hspace=0.62, wspace=0.10, sharey=True)
for ax, (nome, msk) in zip(axs.ravel(), paineis):
    if nome == "dev_shift":
        z, yy_, lab = Z["dev_shift"], eixo_y(Xs["dev_shift"]), ys["dev_shift"]
        titulo_painel(ax, "Internet", detalhe_n("dev_shift", lab), pad=26)
    else:
        m = np.ones(len(mv), bool) if msk is None else msk
        z, yy_, lab = Z["val"][m], eixo_y(Xs["val"][m]), ys["val"][m]
        nr, nf = n_classes(lab)
        if nome == "val":
            titulo_painel(ax, "Estúdio (todas as fontes)", detalhe_n("val", lab), pad=26)
        else:
            titulo_painel(ax, f"Estúdio: {NOME[nome]}",
                          f"(validação) · {mil(nr)} reais" + (f" · {mil(nf)} IA" if nf else ""), pad=26)
    painel_plano(ax, z, yy_, lab, mostrar_regioes=False, s=4)
for ax in axs[1]:
    ax.set_xlabel("nota da regressão")
for ax in axs[:, 0]:
    ax.set_ylabel("variação\n(só espalha)")
legenda(fig, [h_ponto(REAL_C, "real"), h_ponto(IA_C, "gerada por IA"),
              Line2D([], [], **EST_CORTE["A"], label=CORTE["A"]), Line2D([], [], **EST_CORTE["B"], label=CORTE["B"])],
        y=0.85, ncol=4)
salvar(fig, nb="g1_plano_decisao")

fig, axs = quadro("Nos bancos de teste, os erros são do mesmo tipo",
                  "Só visualização, não decide nada: a regressão e os cortes foram fixados antes do teste",
                  1, 4, top=0.72, bottom=0.13, left=0.07, wspace=0.10, sharey=True)
for ax, bnc in zip(axs[0], ["in_the_wild", "llamapartialspoof", "brspeech_df", "coraa"]):
    painel_plano(ax, Z[bnc], eixo_y(Xs[bnc]), ys[bnc], mostrar_regioes=False, s=4)
    nr, nf = n_classes(ys[bnc])
    titulo_painel(ax, GRUPO[bnc].replace(", só real", ""),
                  f"({NOME[bnc]}) · {mil(nr)} reais" + (f" · {mil(nf)} IA" if nf else ""))
    ax.set_xlabel("nota da regressão")
axs[0, 0].set_ylabel("variação (só espalha)")
legenda(fig, [h_ponto(REAL_C, "real"), h_ponto(IA_C, "gerada por IA"),
              Line2D([], [], **EST_CORTE["A"], label=CORTE["A"]), Line2D([], [], **EST_CORTE["B"], label=CORTE["B"])],
        y=0.85, ncol=4)
salvar(fig, nb="g1_plano_decisao_teste")

# ======================================================================== 04 distribuição da nota (2 x 3)
log("04: distribuição da nota")
fig, axs = quadro("O corte do estúdio fica cedo demais para a internet e o português",
                  "Distribuição da nota por classe em cada grupo (altura = proporção dos áudios de cada classe)",
                  2, 3, top=0.73, bottom=0.10, left=0.04, right=0.985, hspace=0.62, wspace=0.08)
bins = np.linspace(*XLIM, 70)
for ax, k in zip(axs.ravel(), ["val", "dev_shift", "in_the_wild", "llamapartialspoof", "brspeech_df", "coraa"]):
    for lab, cor in ((0, REAL_C), (1, IA_C)):
        zz = Z[k][ys[k] == lab]
        if len(zz):
            ax.hist(np.clip(zz, *XLIM), bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=cor)
            ax.hist(np.clip(zz, *XLIM), bins=bins, density=True, histtype="step", lw=1.6, color=cor)
    linhas_corte(ax, rotular=True)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    titulo_painel(ax, GRUPO[k] if k != "in_the_wild" else "Internet (teste)", detalhe_n(k, ys[k]), pad=26)
for ax in axs[1]:
    ax.set_xlabel("nota da regressão")
legenda(fig, [h_caixa(REAL_C, "real"), h_caixa(IA_C, "gerada por IA"),
              Line2D([], [], **EST_CORTE["A"], label=CORTE["A"]), Line2D([], [], **EST_CORTE["B"], label=CORTE["B"])],
        y=0.85, ncol=4)
salvar(fig, "04_distribuicao_nota", "04_distribuicao_nota",
       "Histograma da nota de reais e de gerados por IA em seis grupos: estúdio, internet (dev_shift e teste), fakes parciais, português e português só real.",
       "No estúdio as duas montanhas quase não se tocam; na internet e no português os reais se deslocam para a direita e passam do corte A.")

# ======================================================================== 05 camadas
log("05: camadas")
cam = pd.read_csv(R / "v4_camadas.csv")
fig, axs = quadro("Quais camadas do modelo separam melhor real de IA?",
                  "Uma regressão por camada do XLS-R; faixa verde: as camadas usadas no sistema (5, 6 e 7)",
                  1, 2, top=0.68, bottom=0.13, wspace=0.22)
ax = axs[0, 0]
ax.plot(cam.camada, 100 * cam["val EER média por fonte"], "-o", ms=6, lw=2.6, color=ESTUDIO_C)
ax.plot(cam.camada, 100 * cam["dev EER"], "-o", ms=6, lw=2.6, color=INTERNET_C)
ax.set_ylabel("erro (%) · menor é melhor")
titulo_painel(ax, "Erro no ponto de erro igual", "(reais acusados = IA que passa)", pad=26)
ax = axs[0, 1]
ax.plot(cam.camada, cam["val AUC"], "-o", ms=6, lw=2.6, color=ESTUDIO_C)
ax.plot(cam.camada, cam["dev AUC"], "-o", ms=6, lw=2.6, color=INTERNET_C)
ax.set_ylabel("acerto na ordenação (AUC)")
virgula(ax)
titulo_painel(ax, "Chance de dar nota maior à IA", "(1 = sempre; 0,5 = cara ou coroa)", pad=26)
for ax in axs[0]:
    for c_ in CAMADAS:
        ax.axvspan(c_ - 0.45, c_ + 0.45, color=META, alpha=0.16, lw=0)
    ax.set_xlabel("camada do XLS-R (0 = entrada; 24 = última)")
    ax.set_xticks(range(0, 25, 2))
axs[0, 0].annotate("última camada (24)", xy=(24, 100 * cam.loc[cam.camada == 24, "val EER média por fonte"].iloc[0]),
                   xytext=(15.5, 17), fontsize=13, arrowprops=dict(arrowstyle="->", color=NEUTRO), color=NEUTRO)
axs[0, 0].text(6, 100 * cam["dev EER"].max() * 0.92, "5, 6, 7", ha="center", fontsize=14, color=META, fontweight="bold")
legenda(fig, [Line2D([], [], color=ESTUDIO_C, lw=3, marker="o", label="estúdio (validação)"),
              Line2D([], [], color=INTERNET_C, lw=3, marker="o", label="internet (dev_shift)"),
              h_caixa(META, "camadas usadas", alpha=0.3)], y=0.85)
salvar(fig, "05_camadas", "05_camadas",
       "Erro e AUC de uma regressão por camada do XLS-R, no estúdio (validação) e na internet (dev_shift).",
       "As camadas 0 a 3 quase não servem na internet; da 5 em diante tudo passa de 0,97; as do meio (5 a 7) têm o menor erro.")
salvar_csv(cam[["camada", "C", "val EER média por fonte", "val AUC", "dev AUC", "dev EER", "dev acurácia balanceada"]]
           .assign(**{"no F0": cam.camada.isin(CAMADAS)}), "05_camadas")

# ======================================================================== 06 vetores (t-SNE por camada) + G3 antigo
log("06: t-SNE por camada")
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

MELHOR = int(res_v4["melhor_camada"])
iv = amostra_idx(len(va.meta), 3000, seed=1)
idv = amostra_idx(len(dv.meta), 3000, seed=2)
lab_v, lab_d = va.meta.label.values[iv], dv.meta.label.values[idv]
CAMS6 = [5, 6, 7, 24]


def tsne(M):
    P = PCA(n_components=50, random_state=42).fit_transform(M)
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(P)


emb = {}
pca_g3 = {}
for c_ in CAMS6:
    Mv = va.camada(c_)[iv].astype(np.float32)
    Md = dv.camada(c_)[idv].astype(np.float32)
    t1 = time.time()
    emb[(c_, "val")] = tsne(Mv)
    emb[(c_, "mix")] = tsne(np.vstack([Mv, Md]))
    emb[(c_, "dev")] = tsne(Md)
    if c_ in (MELHOR, 24):
        pca = PCA(n_components=50, random_state=42).fit(np.vstack([Mv, Md]))
        pca_g3[c_] = (pca.transform(np.vstack([Mv, Md]))[:, :2], float(pca.explained_variance_ratio_[:2].sum()))
    log(f"06: camada {c_}: 3 t-SNE em {time.time() - t1:.0f} s")

fig, axs = quadro("Nas camadas do meio, real e IA formam grupos; a origem também",
                  "Vetores do modelo sem a regressão, achatados em 2D (t-SNE); cada ponto é um áudio",
                  3, 4, top=0.75, bottom=0.03, left=0.13, right=0.99, hspace=0.32, wspace=0.06)
lab_mix = np.concatenate([np.zeros(len(iv)), np.ones(len(idv))])
LINHAS6 = [("val", "Estúdio\nreal × IA", lambda: np.where(lab_v == 1, IA_C, REAL_C)),
           ("mix", "Estúdio +\ninternet\n(origem)", lambda: np.where(lab_mix == 1, INTERNET_C, "#B4BAC1")),
           ("dev", "Internet\nreal × IA", lambda: np.where(lab_d == 1, IA_C, REAL_C))]
for i, (chave, rot, cores) in enumerate(LINHAS6):
    cor = cores()
    for j, c_ in enumerate(CAMS6):
        ax = axs[i, j]
        E = emb[(c_, chave)]
        o = np.random.default_rng(0).permutation(len(E))
        ax.scatter(E[o, 0], E[o, 1], c=cor[o], s=3, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        if i == 0:
            ax.set_title(f"camada {c_}" + (" (última)" if c_ == 24 else ""), fontsize=16, fontweight="bold")
    axs[i, 0].text(-0.08, 0.5, rot, transform=axs[i, 0].transAxes, ha="right", va="center", fontsize=14,
                   fontweight="bold")
legenda(fig, [h_ponto(REAL_C, "real"), h_ponto(IA_C, "gerada por IA"), h_ponto("#B4BAC1", "estúdio (3.000 áudios)"),
              h_ponto(INTERNET_C, "internet (3.000 áudios)")], y=0.855)
salvar(fig, "06_vetores", "06_vetores",
       "Os vetores das camadas 5, 6, 7 e 24, sem a regressão, achatados em 2D: só estúdio (real × IA), estúdio + internet (origem) e só internet (real × IA).",
       "Nas camadas do meio as cores real/IA já se agrupam; na linha do meio, a separação mais forte é estúdio × internet; na 24 tudo se mistura mais.")

# G3 antigo (PCA e t-SNE; camada 7 e 24; 3.000 + 3.000), mantido no notebook
lab3 = np.concatenate([lab_v, lab_d])
fig, axs = quadro("Sem a regressão, o que mais aparece é a origem da gravação",
                  "Camada 7 e camada 24; PCA e t-SNE; 3.000 áudios do estúdio + 3.000 da internet",
                  2, 4, top=0.75, bottom=0.03, left=0.10, right=0.99, hspace=0.30, wspace=0.06)
proj_info = {}
for row, c_ in enumerate([MELHOR, 24]):
    P2, var2 = pca_g3[c_]
    proj_info[f"camada {c_}"] = {"PCA variância PC1+PC2": var2}
    for col, (E, nome_e) in enumerate([(P2, "PCA"), (emb[(c_, "mix")], "t-SNE")]):
        o = np.random.default_rng(0).permutation(len(E))
        for k_, cor in enumerate([np.where(lab3 == 1, IA_C, REAL_C), np.where(lab_mix == 1, INTERNET_C, "#B4BAC1")]):
            ax = axs[row, 2 * col + k_]
            ax.scatter(E[o, 0], E[o, 1], c=cor[o], s=3, alpha=0.55, linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if row == 0:
                ax.set_title(f"{nome_e} · cor = " + ("real × IA" if k_ == 0 else "origem"), fontsize=14)
    axs[row, 0].text(-0.06, 0.5, f"camada {c_}", transform=axs[row, 0].transAxes, ha="right", va="center",
                     fontsize=16, fontweight="bold", rotation=90)
legenda(fig, [h_ponto(REAL_C, "real"), h_ponto(IA_C, "gerada por IA"), h_ponto("#B4BAC1", "estúdio"),
              h_ponto(INTERNET_C, "internet")], y=0.855)
salvar(fig, nb="g3_projecao")
del emb

# ======================================================================== G5 peso de cada camada no F0 (notebook)
blocos = np.split(w, len(CAMADAS))
g5 = pd.DataFrame({"camada": CAMADAS, "norma L2 dos 1.024 pesos": [float(np.linalg.norm(x)) for x in blocos],
                   "soma |pesos|": [float(np.abs(x).sum()) for x in blocos]})
g5["parte de Σw² (%)"] = 100 * g5["norma L2 dos 1.024 pesos"] ** 2 / (g5["norma L2 dos 1.024 pesos"] ** 2).sum()
g5["parte de Σ|w| (%)"] = 100 * g5["soma |pesos|"] / g5["soma |pesos|"].sum()
fig, axs = quadro("A regressão usa as três camadas quase por igual",
                  f"Peso total de cada camada na regressão final ({mil(len(w))} pesos; entradas padronizadas)",
                  1, 2, top=0.76, bottom=0.13, wspace=0.25)
cores5 = [META, "#4FA89B", "#8CCBBF"]
axs[0, 0].bar([f"camada {c_}" for c_ in CAMADAS], g5["parte de Σ|w| (%)"], color=cores5, width=0.55)
for i, v in enumerate(g5["parte de Σ|w| (%)"]):
    axs[0, 0].text(i, v + 0.8, br(v, 1) + "%", ha="center", fontsize=14)
axs[0, 0].set_ylabel("% da soma dos |pesos|")
axs[0, 0].grid(axis="x", visible=False)
for c_, x, cor in zip(CAMADAS, blocos, cores5):
    axs[0, 1].hist(np.abs(x), bins=50, histtype="step", lw=2, color=cor, label=f"camada {c_}")
axs[0, 1].set_xlabel("tamanho de cada peso (|peso|)")
axs[0, 1].set_ylabel("quantos pesos")
virgula(axs[0, 1], "x", 3)
axs[0, 1].legend(loc="upper right")
salvar(fig, nb="g5_pesos_camadas")
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


# ======================================================================== 08 curva de aprendizado (G6)
log("08: curva de aprendizado")
linhas = []
idx0, idx1 = np.nonzero(ytr == 0)[0], np.nonzero(ytr == 1)[0]
r6 = np.random.default_rng(42)
p0, p1 = r6.permutation(idx0), r6.permutation(idx1)
for n in (2000, 5000, 10000, 20000, 40000):
    sel = np.sort(np.concatenate([p0[: n // 2], p1[: n // 2]]))
    t = time.time()
    m = f0 if n == len(Xtr) else LogisticRegression(C=C_F0, max_iter=RL_MAX_ITER, tol=1e-4).fit(Xtr[sel], ytr[sel])
    linhas.append({"clipes de treino": n, **avaliar_modelo(m, Xtr[sel], ytr[sel]), "segundos": round(time.time() - t, 1)})
g6 = pd.DataFrame(linhas)
salvar_csv(g6, "g6_curva_aprendizado")
fig, axs = quadro("Mais áudios de estúdio não melhoraram a internet",
                  "A mesma regressão (C = 0,1) treinada com 2 mil a 40 mil áudios (metade real, metade IA); diagnóstico",
                  1, 2, top=0.68, bottom=0.13, wspace=0.22)
for col, cor, ls in (("treino", TREINO_C, "--"), ("val", ESTUDIO_C, "-"), ("dev", INTERNET_C, "-")):
    axs[0, 0].plot(g6["clipes de treino"], g6[f"{col} acc. bal. (lim. A)"], ls, marker="o", lw=2.6, ms=7, color=cor)
    axs[0, 1].plot(g6["clipes de treino"], g6[f"{col} AUC"], ls, marker="o", lw=2.6, ms=7, color=cor)
titulo_painel(axs[0, 0], "Acerto equilibrado", "(média do acerto em reais e em IA, no corte A de cada modelo)", pad=26)
titulo_painel(axs[0, 1], "Acerto na ordenação (AUC)", "(chance de dar nota maior à IA)", pad=26)
axs[0, 0].set_ylabel("acerto equilibrado")
axs[0, 1].set_ylabel("AUC")
for ax in axs[0]:
    ax.set_xscale("log")
    ax.set_xticks(g6["clipes de treino"], [f"{n // 1000} mil" for n in g6["clipes de treino"]])
    ax.minorticks_off()
    ax.set_xlabel("nº de áudios de treino")
    ax.axvline(40000, color=META, lw=8, alpha=0.18)
    ax.text(40000, 0.55, "usado  \n(40 mil)  ", transform=ax.get_xaxis_transform(), ha="right", fontsize=13, color=META)
from matplotlib.ticker import MultipleLocator  # noqa: E402
axs[0, 0].yaxis.set_major_locator(MultipleLocator(0.02)); virgula(axs[0, 0], d=2)
axs[0, 1].yaxis.set_major_locator(MultipleLocator(0.004)); virgula(axs[0, 1], d=3)
legenda(fig, [Line2D([], [], color=TREINO_C, lw=3, ls="--", marker="o", label="treino (os próprios áudios)"),
              Line2D([], [], color=ESTUDIO_C, lw=3, marker="o", label="estúdio (validação)"),
              Line2D([], [], color=INTERNET_C, lw=3, marker="o", label="internet (dev_shift)")], y=0.85)
salvar(fig, "08_curva_aprendizado", "08_curva_aprendizado",
       "Acerto da mesma regressão treinada com 2 mil a 40 mil áudios, no próprio treino, no estúdio e na internet.",
       "O treino fica em 100%; o estúdio melhora pouco; a internet não melhora (AUC 0,993 com 2 mil, 0,983 com 40 mil).")

# ======================================================================== 07 caminho do C (G7)
log("07: caminho do C")
GRADE_V4 = [1e-3, 1e-2, 1e-1, 1.0]
grade7 = sorted(set([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]))
linhas = []
for Cc in grade7:
    t = time.time()
    m = f0 if Cc == C_F0 else LogisticRegression(C=Cc, max_iter=RL_MAX_ITER, tol=1e-4).fit(Xtr, ytr)
    linhas.append({"C": Cc, "na grade da v4": Cc in GRADE_V4, **avaliar_modelo(m, Xtr, ytr),
                   "norma de w": float(np.linalg.norm(m.coef_)), "segundos": round(time.time() - t, 1)})
g7 = pd.DataFrame(linhas)
salvar_csv(g7, "g7_caminho_c")
fig, axs = quadro("Com mais freio, o erro na internet caiu; o estúdio prefere C = 0,1",
                  "A mesma regressão com 11 valores de C; diagnóstico feito depois de olhar a internet: não muda o sistema",
                  1, 2, top=0.68, bottom=0.15, wspace=0.22)
ax = axs[0, 0]
ax.plot(g7.C, 100 * g7["val EER média por fonte"], "-o", lw=2.6, ms=7, color=ESTUDIO_C)
ax.plot(g7.C, 100 * g7["dev EER"], "-o", lw=2.6, ms=7, color=INTERNET_C)
ax.set_ylabel("erro (%) · menor é melhor")
titulo_painel(ax, "Erro no ponto de erro igual", "(estúdio: média por fonte; internet: dev_shift)", pad=26)
ax = axs[0, 1]
ax.plot(g7.C, g7["norma de w"], "-o", lw=2.6, ms=7, color=TEXTO)
ax.set_yscale("log")
ax.set_ylabel("tamanho dos pesos (norma)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1) if v < 10 else f"{v:.0f}"))
titulo_painel(ax, "Os pesos encolhem quando o freio aumenta", "(freio forte = pesos pequenos = regressão mais simples)",
              pad=26)
for ax in axs[0]:
    ax.set_xscale("log")
    ax.axvline(C_F0, color=META, lw=10, alpha=0.2)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10], ["0,0001", "0,001", "0,01", "0,1", "1", "10"])
    ax.minorticks_off()
    ax.set_xlabel("freio de regularização:  forte ←   valor de C   → fraco")
axs[0, 0].text(C_F0 * 1.15, 0.03, "C usado (0,1)", transform=axs[0, 0].get_xaxis_transform(), fontsize=13,
               color=META, va="bottom", fontweight="bold")
legenda(fig, [Line2D([], [], color=ESTUDIO_C, lw=3, marker="o", label="estúdio (validação)"),
              Line2D([], [], color=INTERNET_C, lw=3, marker="o", label="internet (dev_shift)"),
              h_caixa(META, "C usado no sistema (0,1)", alpha=0.3)], y=0.85)
salvar(fig, "07_c_freio", "07_c_freio",
       "Erro no estúdio e na internet, e o tamanho dos pesos, para 11 valores do freio de regularização (C).",
       "O estúdio tem o menor erro no C usado (0,1); a internet melhora com freio mais forte (C = 0,0001: erro 2,7% contra 5,9%); diagnóstico, não muda o sistema.")

# ======================================================================== G8 calibração (notebook)
linhas = []
fig, axs = quadro("A nota ordena bem, mas não vale como probabilidade na internet",
                  "Áudios agrupados pela probabilidade prevista; no eixo y, quantos são de fato gerados por IA",
                  1, 2, top=0.68, bottom=0.13, wspace=0.18)
bordas = np.linspace(0, 1, 11)
for ax, k in zip(axs[0], ["val", "dev_shift"]):
    p = sigmoide(Z[k])
    yk = ys[k]
    cx, cy, cn = [], [], []
    for i in range(10):
        msk = (p >= bordas[i]) & ((p < bordas[i + 1]) if i < 9 else (p <= 1))
        if msk.sum():
            cx.append(p[msk].mean()); cy.append(yk[msk].mean()); cn.append(int(msk.sum()))
            linhas.append({"conjunto": k, "faixa de prob.": f"{bordas[i]:.1f} a {bordas[i + 1]:.1f}",
                           "prob. média prevista": p[msk].mean(), "fração real de falsos": yk[msk].mean(), "n": int(msk.sum())})
    ax.plot([0, 1], [0, 1], color=CHUTE_C, ls="--", lw=2)
    ax.plot(cx, cy, "-o", lw=2.6, color=ESTUDIO_C if k == "val" else INTERNET_C, ms=7)
    for x_, y_, n_ in (zip(cx, cy, cn)):
        if n_ >= 100:
            ax.annotate(mil(n_), (x_, y_), textcoords="offset points", xytext=((-58, -4) if x_ > 0.9 else (6, 8)), fontsize=13, color=NEUTRO)
    titulo_painel(ax, GRUPO[k], f"({NOME[k]}) · {pct(yk.mean())} gerados por IA; números = áudios na faixa", pad=26)
    ax.set_xlabel("probabilidade prevista de ser IA")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    virgula(ax, "x", 1); virgula(ax, "y", 1)
axs[0, 0].set_ylabel("fração que é IA de verdade")
legenda(fig, [Line2D([], [], color=CHUTE_C, ls="--", lw=3, label="probabilidade perfeita"),
              Line2D([], [], color=ESTUDIO_C, lw=3, marker="o", label="estúdio"),
              Line2D([], [], color=INTERNET_C, lw=3, marker="o", label="internet")], y=0.85)
salvar(fig, nb="g8_calibracao")
salvar_csv(pd.DataFrame(linhas), "g8_calibracao")

# ======================================================================== tabela do teste (seção 8)
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
for bnc in PRINCIPAIS:      # conferência: métricas do reajuste = as gravadas
    a_ = acuracias(ys[bnc], Z[bnc], LIM["A"])["acurácia"]
    g_ = t8[(t8.sistema == "F0 (v4 final)") & (t8.banco == bnc) & (t8.limiar == "A")]["acurácia"].iloc[0]
    assert abs(a_ - g_) < 1e-9, (bnc, a_, g_)


def v8(sist, bnc, L, mt):
    return float(t8[(t8.sistema == sist) & (t8.banco == bnc) & (t8.limiar == L)][mt].iloc[0])


# conferência: a média dos 3 principais bate com o resumo gravado pela v4
media_res = {(r["sistema"], r["limiar"]): r for r in tres["resumo"]}
for s, sist in (("S2", "F0 (v4 final)"), ("S0", "S0 (CNN da v3)")):
    for L in ("A", "B"):
        m_ = np.mean([v8(sist, b, L, "acurácia") for b in PRINCIPAIS])
        chave = [k_ for k_ in media_res if k_[0] == s and k_[1] == L]
        if chave:
            r_ = media_res[chave[0]]
            gravada = [v for k_, v in r_.items() if "acurácia" in k_ and "média" in k_ and "balanceada" not in k_]
            if gravada:
                assert abs(gravada[0] - m_) < 1e-3, (s, L, gravada[0], m_)

# ======================================================================== 09 acurácia por grupo (G10)
log("09: acurácia por grupo")
SERIES9 = [("F0 (v4 final)", "A", META, "sistema v4, corte A", None),
           ("F0 (v4 final)", "B", META_CLARO, "sistema v4, corte B", None),
           ("S0 (CNN da v3)", "A", NEUTRO, "CNN antiga (v3), corte A", None),
           (None, None, CHUTE_C, "chute da classe comum", "//")]
fig = plt.figure(figsize=TAM)
fig.text(0.012, 0.975, "O sistema novo acerta mais que a CNN antiga em todos os grupos", fontsize=24,
         fontweight="bold", va="top")
fig.text(0.012, 0.905, "Acurácia no teste (avaliado uma vez); em português o chute 'sempre IA' já acerta 79%",
         fontsize=16, va="top", color=NEUTRO)
ax = fig.add_axes([0.07, 0.17, 0.62, 0.55])
axc = fig.add_axes([0.77, 0.17, 0.21, 0.55])
xg = np.arange(4)
larg = 0.2
dados9 = []
for i, (sist, L, cor, rotulo, hatch) in enumerate(SERIES9):
    if sist:
        vals = [v8(sist, b, L, "acurácia") for b in PRINCIPAIS]
    else:
        vals = [v8("F0 (v4 final)", b, "A", "acurácia trivial") for b in PRINCIPAIS]
    vals.append(float(np.mean(vals)))
    dados9.append({"série": rotulo, **{NOME[b]: v for b, v in zip(PRINCIPAIS, vals[:3])}, "média dos 3": vals[3]})
    xs_ = xg + (i - 1.5) * larg
    ax.bar(xs_, vals, larg * 0.92, color=cor, hatch=hatch, edgecolor="white" if hatch else "none", label=rotulo)
    for x_, v in zip(xs_, vals):
        ax.text(x_, v + 0.015, pct(v), ha="center", fontsize=13, rotation=0)
ax.set_xticks(xg, ["Internet", "Fakes parciais", "Português", "Média dos 3"])
for x_, nm in zip(xg, ["(In-the-Wild)", "(LlamaPartialSpoof)", "(BRSpeech-DF)", ""]):
    ax.annotate(nm, xy=(x_, 0), xycoords=("data", "axes fraction"), xytext=(0, -38), textcoords="offset points",
                ha="center", fontsize=13, color=NEUTRO)
ax.set_ylim(0, 1.1)
ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
ax.set_ylabel("acurácia (% de acertos)")
ax.grid(axis="x", visible=False)
ax.set_title("Bancos com reais e gerados por IA", loc="left", fontsize=16, fontweight="bold", pad=12)
vc = [1 - v8("F0 (v4 final)", "coraa", "A", "falso alarme real"), 1 - v8("F0 (v4 final)", "coraa", "B", "falso alarme real"),
      1 - v8("S0 (CNN da v3)", "coraa", "A", "falso alarme real")]
axc.bar(np.arange(3), vc, 0.62, color=[META, META_CLARO, NEUTRO])
for i, v in enumerate(vc):
    axc.text(i, v + 0.015, pct(v, 1), ha="center", fontsize=13)
axc.set_xticks(np.arange(3), ["v4\ncorte A", "v4\ncorte B", "CNN\ncorte A"], fontsize=13)
axc.set_ylim(0, 1.1)
axc.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
axc.grid(axis="x", visible=False)
axc.set_title("Português, só real", loc="left", fontsize=16, fontweight="bold", pad=34)
axc.text(0, 1.03, f"(CORAA) · {mil(v8('F0 (v4 final)', 'coraa', 'A', 'n real'))} reais", transform=axc.transAxes,
         fontsize=13, color=NEUTRO)
axc.set_ylabel("reais aceitos como reais")
h9 = [h_caixa(c, r_, hatch=h, edgecolor="white" if h else "none") for _, _, c, r_, h in SERIES9]
legenda(fig, h9, y=0.85)
salvar(fig, "09_acuracia_grupos", "09_acuracia_grupos",
       "Acurácia no teste por grupo: sistema v4 nos cortes A e B, CNN antiga e o chute da classe comum; à direita, % de reais do CORAA aceitos.",
       "Na internet a CNN fica abaixo do chute e o v4 vai a 84% (A) e 93% (B); no português (BRSpeech-DF) o v4 passa pouco do chute; no CORAA, o corte A aceita 83% dos reais.")
salvar_csv(pd.DataFrame(dados9), "09_acuracia_grupos")

# ======================================================================== 10 erros por grupo (G11)
log("10: erros por grupo")
GRUPOS10 = [("Estúdio", "(validação)", "val"), ("Internet", "(dev_shift)", "dev_shift"),
            ("Internet", "(teste: In-the-Wild)", "in_the_wild"), ("Fakes parciais", "(LlamaPartialSpoof)", "llamapartialspoof"),
            ("Português", "(BRSpeech-DF)", "brspeech_df"), ("Português, só real", "(CORAA)", "coraa")]
linhas = []
for grp, det, k in GRUPOS10:
    for L in ("A", "B"):
        pr = Z[k] >= LIM[L]
        yk = ys[k]
        linhas.append({"grupo": f"{grp} {det}", "conjunto": k, "corte": L,
                       "reais acusados como IA": pr[yk == 0].mean() if (yk == 0).any() else np.nan,
                       "IA que passa sem alarme": 1 - pr[yk == 1].mean() if (yk == 1).any() else np.nan,
                       "n real": int((yk == 0).sum()), "n IA": int((yk == 1).sum())})
# as fontes da validação (para o notebook; mesma conta do G11 antigo)
for f in FONTES_VAL:
    msk = (metas["val"].source_dataset == f).values
    for L in ("A", "B"):
        pr = Z["val"][msk] >= LIM[L]
        yk = ys["val"][msk]
        linhas.append({"grupo": f"Estúdio: {NOME[f]} (validação)", "conjunto": "val", "corte": L,
                       "reais acusados como IA": pr[yk == 0].mean() if (yk == 0).any() else np.nan,
                       "IA que passa sem alarme": 1 - pr[yk == 1].mean() if (yk == 1).any() else np.nan,
                       "n real": int((yk == 0).sum()), "n IA": int((yk == 1).sum())})
g10 = pd.DataFrame(linhas)
salvar_csv(g10, "10_erros_por_grupo")
fig, axs = quadro("Um corte acusa reais; o outro deixa passar gerados por IA",
                  "% de erro de cada tipo, por grupo, nos dois cortes (barra escura: corte A; clara: corte B)",
                  1, 2, top=0.75, bottom=0.08, left=0.215, right=0.985, wspace=0.10, sharey=True)
yg = np.arange(len(GRUPOS10))[::-1]
for ax, col, (cA, cB), tit in ((axs[0, 0], "reais acusados como IA", (REAL_C, REAL_CLARO), "Reais acusados como IA"),
                               (axs[0, 1], "IA que passa sem alarme", (IA_C, IA_CLARO), "Gerados por IA que passam sem alarme")):
    for d_, L, cor in ((0.19, "A", cA), (-0.19, "B", cB)):
        g = g10[(g10.corte == L)].set_index("grupo").loc[[f"{a} {b}" for a, b, _ in GRUPOS10]]
        v = g[col].values
        ax.barh(yg + d_, np.nan_to_num(100 * v), 0.36, color=cor)
        for y_, vv in zip(yg + d_, v):
            if np.isfinite(vv):
                ax.text(100 * vv + 1, y_, pct(vv), va="center", fontsize=13)
            elif L == "A":
                ax.text(1, y_ - 0.19, "não se aplica (só real)", va="center", fontsize=13, color=NEUTRO)
    ax.set_xlim(0, 80)
    ax.set_xlabel("% de erro")
    ax.grid(axis="y", visible=False)
    ax.set_title(tit, loc="left", fontsize=16, fontweight="bold")
axs[0, 0].set_yticks(yg, [a for a, _, _ in GRUPOS10])
for y_, (_, det, _) in zip(yg, GRUPOS10):
    axs[0, 0].annotate(det, xy=(0, y_), xycoords=("axes fraction", "data"), xytext=(-8, -17),
                       textcoords="offset points", ha="right", fontsize=13, color=NEUTRO)
legenda(fig, [h_caixa(REAL_C, "reais acusados, corte A"), h_caixa(REAL_CLARO, "corte B"),
              h_caixa(IA_C, "IA que passa, corte A"), h_caixa(IA_CLARO, "corte B")], y=0.855)
salvar(fig, "10_erros_por_grupo", "10_erros_por_grupo",
       "Os dois erros do sistema por grupo: reais acusados como IA e gerados por IA que passam, nos cortes A e B.",
       "O corte A acusa muitos reais na internet e no português; o corte B quase não acusa reais, mas deixa passar metade dos fakes parciais.")

# ======================================================================== 11 duração (G12)
log("11: duração")
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
fig, axs = quadro("Áudios curtos são os mais difíceis, sobretudo em português",
                  "Acerto na ordenação por duração original do áudio, nos três bancos principais do teste",
                  1, 1, top=0.74, bottom=0.17, left=0.08)
ax = axs[0, 0]
CORES_DUR = ["#C9D9F2", "#7FA2DE", REAL_C]
xk = np.arange(3)
for i, (f, cor) in enumerate(zip(FAIXAS_DURACAO, CORES_DUR)):
    g = g12[g12.faixa == f].set_index("conjunto").loc[PRINCIPAIS]
    xs_ = xk + (i - 1) * 0.27
    ax.bar(xs_, g.AUC - 0.5, 0.25, bottom=0.5, color=cor)
    for x_, v, n_ in zip(xs_, g.AUC, g.n):
        ax.text(x_, v + 0.006, br(v, 2), ha="center", fontsize=14, fontweight="bold")
        if v - 0.5 < 0.08:      # barra baixa: o nº de áudios vai acima do valor
            ax.text(x_, v + 0.03, f"{mil(n_)}\náudios", ha="center", va="bottom", fontsize=13)
        else:
            ax.text(x_, 0.505, f"{mil(n_)}\náudios", ha="center", va="bottom", fontsize=13,
                    color="white" if cor == REAL_C else TEXTO)
ax.set_xticks(xk, [GRUPO[b] for b in PRINCIPAIS])
for x_, b in zip(xk, PRINCIPAIS):
    ax.annotate(f"({NOME[b]})", xy=(x_, 0), xycoords=("data", "axes fraction"), xytext=(0, -38),
                textcoords="offset points", ha="center", fontsize=13, color=NEUTRO)
ax.set_ylim(0.5, 1.04)
virgula(ax)
ax.axhline(0.5, color=TEXTO, lw=1)
ax.set_ylabel("acerto na ordenação (AUC)\n0,5 = cara ou coroa")
ax.grid(axis="x", visible=False)
legenda(fig, [h_caixa(c, f"duração original {f}") for f, c in zip(FAIXAS_DURACAO, CORES_DUR)], y=0.85)
salvar(fig, "11_duracao", "11_duracao",
       "AUC do sistema por faixa de duração original do áudio (< 2 s, 2 a 4 s, ≥ 4 s) nos três bancos principais, com o nº de áudios.",
       "Em todos os bancos os curtos são os piores; no português abaixo de 2 s a ordenação fica perto de cara ou coroa (0,53).")

# ======================================================================== 12 Meta × Microsoft (G14)
log("12: Meta × Microsoft")
c5 = pd.read_csv(R / "v5_comparacao_camadas.csv")
c6 = pd.read_csv(R / "v6_e1_camadas.csv")
fig, axs = quadro("Os dois modelos guardam as pistas em camadas diferentes",
                  "v5: o WavLM da Microsoft no lugar do XLS-R da Meta, com o mesmo pipeline e os mesmos áudios",
                  1, 2, top=0.68, bottom=0.13, wspace=0.22)
ax = axs[0, 0]
ax.plot(c5.camada, 100 * c5["val EER média por fonte xlsr"], "-o", ms=6, lw=2.6, color=META)
ax.plot(c5.camada, 100 * c5["val EER média por fonte wavlm"], "-o", ms=6, lw=2.6, color=MS)
ax.set_ylabel("erro (%) · menor é melhor")
titulo_painel(ax, "Erro no estúdio", "(validação, média por fonte)", pad=26)
ax = axs[0, 1]
ax.plot(c5.camada, c5["dev AUC xlsr"], "-o", ms=6, lw=2.6, color=META)
ax.plot(c5.camada, c5["dev AUC wavlm"], "-o", ms=6, lw=2.6, color=MS)
ax.set_ylabel("acerto na ordenação (AUC)")
virgula(ax)
titulo_painel(ax, "Acerto na internet", "(dev_shift)", pad=26)
for ax in axs[0]:
    for c_ in CAMADAS:
        ax.axvspan(c_ - 0.45, c_ + 0.45, color=META, alpha=0.12, lw=0)
    ax.set_xlabel("camada (0 = entrada; 24 = última)")
    ax.set_xticks(range(0, 25, 4))
legenda(fig, [Line2D([], [], color=META, lw=3, marker="o", label="Meta (XLS-R 300M, 128 línguas)"),
              Line2D([], [], color=MS, lw=3, marker="o", label="Microsoft (WavLM Large, só inglês)"),
              h_caixa(META, "camadas usadas no sistema", alpha=0.25)], y=0.85)
salvar(fig, "12_meta_microsoft", "12_meta_microsoft",
       "Erro no estúdio e AUC na internet de uma regressão por camada, nos dois modelos prontos: XLS-R (Meta) e WavLM (Microsoft).",
       "No WavLM as melhores camadas são as de baixo (3 e 4) e as de cima pioram muito; no XLS-R as do meio (5 a 9) ficam todas boas.")
salvar_csv(c5.merge(c6.pivot(index="camada", columns="modelo", values="AUC pior fonte")
                    .rename(columns={"xlsr": "v6 AUC pior fonte xlsr", "wavlm": "v6 AUC pior fonte wavlm"}).reset_index(),
                    on="camada"), "g14_meta_microsoft")

# ======================================================================== 13 secundários
log("13: secundários")
SEC = ["asvspoof2019_la", "asvspoof2021_df"]
fig, axs = quadro("Nos testes secundários o sistema quase não erra",
                  "Secundários: podem ter locutores em comum com o treino (VCTK), então medem só geradores novos",
                  1, 2, top=0.74, bottom=0.17, wspace=0.22)
for ax, (tit, col, Ls) in zip(axs[0], (("Acurácia", "acurácia", ("A", "B")), ("Acerto na ordenação (AUC)", "AUC", ("A",)))):
    xs0 = np.arange(len(SEC))
    series = [("F0 (v4 final)", L) for L in Ls] + [("S0 (CNN da v3)", "A")]
    cores = {("F0 (v4 final)", "A"): META, ("F0 (v4 final)", "B"): META_CLARO, ("S0 (CNN da v3)", "A"): NEUTRO}
    larg = 0.8 / len(series)
    for i, (sist, L) in enumerate(series):
        vals = [v8(sist, b, L, col) for b in SEC]
        xs_ = xs0 + (i - (len(series) - 1) / 2) * larg
        ax.bar(xs_, vals, larg * 0.9, color=cores[(sist, L)])
        for x_, v in zip(xs_, vals):
            ax.text(x_, v + 0.015, pct(v) if col == "acurácia" else br(v, 3), ha="center", fontsize=13)
    ax.set_xticks(xs0, [NOME[b] for b in SEC])
    for x_, b in zip(xs0, SEC):
        nr, nf = int(v8("F0 (v4 final)", b, "A", "n real")), int(v8("F0 (v4 final)", b, "A", "n fake"))
        ax.annotate(f"{mil(nr)} reais · {mil(nf)} IA", xy=(x_, 0), xycoords=("data", "axes fraction"), xytext=(0, -38),
                    textcoords="offset points", ha="center", fontsize=13, color=NEUTRO)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title(tit, loc="left", fontsize=16, fontweight="bold")
    if col == "acurácia":
        ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    else:
        virgula(ax, d=1)
legenda(fig, [h_caixa(META, "sistema v4, corte A"), h_caixa(META_CLARO, "sistema v4, corte B"),
              h_caixa(NEUTRO, "CNN antiga (v3), corte A")], y=0.85)
salvar(fig, "13_secundarios", "13_secundarios",
       "Acurácia (cortes A e B) e AUC nos dois testes secundários (ASVspoof 2019 e 2021 DF), sistema v4 e CNN antiga.",
       "Números altos, mas esses bancos podem ter locutores em comum com o treino: valem para geradores novos, não para pessoas novas.")

# ======================================================================== 14 ROC dos 3 principais (+ G9 confusão)
log("14: ROC")
fig, axs = quadro("Em inglês a curva fica colada no canto; em português, não",
                  "Para cada corte possível: reais acusados (x) e gerados por IA detectados (y); ● = corte A, ■ = corte B",
                  1, 1, top=0.78, bottom=0.11, left=0.10, right=0.50)
ax = axs[0, 0]
hs = []
for bnc in PRINCIPAIS:
    fpr, tpr, _ = roc_curve(ys[bnc], Z[bnc])
    ax.plot(100 * fpr, 100 * tpr, lw=3, color=COR_GRUPO[bnc])
    for L, mk in (("A", "o"), ("B", "s")):
        pr = Z[bnc] >= LIM[L]
        ax.plot(100 * pr[ys[bnc] == 0].mean(), 100 * pr[ys[bnc] == 1].mean(), mk, ms=12, color=COR_GRUPO[bnc],
                mec="white", mew=2)
    hs.append(Line2D([], [], color=COR_GRUPO[bnc], lw=3,
                     label=f"{GRUPO[bnc]} ({NOME[bnc]}): AUC {br(roc_auc_score(ys[bnc], Z[bnc]), 3)}"))
ax.plot([0, 100], [0, 100], color=CHUTE_C, ls="--", lw=2)
ax.text(62, 55, "cara ou coroa", rotation=33, fontsize=13, color=NEUTRO)
ax.set_xlabel("reais acusados como IA (%)")
ax.set_ylabel("gerados por IA detectados (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect("equal")
fig.legend(handles=hs, loc="center left", bbox_to_anchor=(0.50, 0.45), fontsize=14, frameon=False)
salvar(fig, "14_roc", "14_roc",
       "Curva ROC do sistema nos três bancos principais do teste, com os pontos dos cortes A e B.",
       "Internet e fakes parciais ficam perto do canto superior esquerdo (bom); o português fica bem abaixo; os pontos A e B mudam de lugar em cada banco.")

linhas = []
fig, axs = quadro("Os dois cortes trocam um erro pelo outro",
                  "Matrizes de confusão no teste (linha = verdade; % por linha); só mostra, não decide nada",
                  2, 4, top=0.74, bottom=0.07, left=0.08, right=0.99, hspace=0.55, wspace=0.35)
for j, bnc in enumerate(["in_the_wild", "llamapartialspoof", "brspeech_df", "coraa"]):
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
        nl = 1 if bnc == "coraa" else 2
        ax.imshow(frac[:nl], cmap="Greys", vmin=0, vmax=1.4)
        for a in range(nl):
            for p_ in (0, 1):
                ax.text(p_, a, f"{mil(mat[a, p_])}\n({pct(frac[a, p_])})", ha="center", va="center", fontsize=13,
                        color="white" if frac[a, p_] > 0.7 else TEXTO)
        ax.set_xticks([0, 1], ["dito real", "dito IA"], fontsize=13)
        ax.set_yticks(range(nl), ["real", "IA"][:nl], fontsize=13)
        ax.grid(False)
        ax.set_title(f"{GRUPO[bnc].replace(', só real', '')} · corte {L}", fontsize=14, loc="left")
        if i == 0:
            ax.text(0, 1.28, f"({NOME[bnc]})", transform=ax.transAxes, fontsize=13, color=NEUTRO)
salvar(fig, nb="g9_confusao")
salvar_csv(pd.DataFrame(linhas), "g9_confusao")

# ======================================================================== G13 recall por gerador (notebook)
mvf = metas["val"]
fk = (mvf.label == 1).values
linhas = []
for (fonte, gen), g in mvf[fk].groupby(["source_dataset", "generator"]):
    zz = Z["val"][g.index.values]
    linhas.append({"fonte": fonte, "gerador": gen, "n fakes": len(g),
                   "recall (lim. A)": float((zz >= LIM["A"]).mean()), "recall (lim. B)": float((zz >= LIM["B"]).mean())})
g13 = pd.DataFrame(linhas).sort_values(["fonte", "recall (lim. A)"]).reset_index(drop=True)
salvar_csv(g13, "g13_geradores")
fig, axs = quadro("Os falsos de codec neural são os que mais escapam na validação",
                  "% dos gerados por IA detectados, por gerador (barra: corte A; traço: corte B); n pequeno = número instável",
                  1, 1, top=0.80, bottom=0.08, left=0.33, right=0.97)
ax = axs[0, 0]
yy = np.arange(len(g13))
cor_f = {"dfadd": NEUTRO, "shiftyspeech": "#9AA1A9", "codecfake_plus": IA_C}
ax.barh(yy, 100 * g13["recall (lim. A)"], 0.7, color=[cor_f[f] for f in g13.fonte])
ax.plot(100 * g13["recall (lim. B)"], yy, "|", ms=12, mew=2.5, color=TEXTO)
ax.set_yticks(yy, [f"{NOME[f]} · {gn} (n={n})" for f, gn, n in zip(g13.fonte, g13.gerador, g13["n fakes"])], fontsize=13)
ax.set_xlabel("% detectados")
ax.set_xlim(0, 103)
ax.grid(axis="y", visible=False)
legenda(fig, [h_caixa(NEUTRO, "DFADD"), h_caixa("#9AA1A9", "ShiftySpeech"), h_caixa(IA_C, "CodecFake+"),
              Line2D([], [], ls="", marker="|", ms=14, mew=2.5, color=TEXTO, label="corte B")], y=0.86)
salvar(fig, nb="g13_geradores")

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

# ======================================================================== tabela de hiperparâmetros (seção 9)
ep6 = {s: json.loads(v)["xlsr"] for s, v in zip(e4.sistema, e4["melhor época"]) if isinstance(v, str)}
mlp_ep = ", ".join(str(ep6[f"MLP básico s{s}"]) for s in (42, 123, 2026))
cnn_ep = ", ".join(str(ep6[f"CNN básico s{s}"]) for s in (42, 123, 2026))
NAO_DOC, NAO_MED = "não documentado", "não medido"
HPV = [
    {"versão": "v1 (baseline)",
     "entrada": "16 kHz mono; recorte de 4 s (aleatório no treino, central na validação); áudio curto repetido; log-mel 64 faixas (n_fft 512, janela 400, passo 160), padronizado por áudio",
     "arquitetura": "CNN: 4 blocos Conv 3×3 + BatchNorm + ReLU + MaxPool (16/32/64/128 filtros), Dropout, Linear(1); 110.433 parâmetros",
     "épocas (máx. / escolhida)": "30 / 16 (parou na 21; early stopping, paciência 5, pela EER global da validação)",
     "lote": "64 (25.000 reais + 25.000 IA por época)", "otimizador e taxa": "Adam, 0,001",
     "regularização": "dropout 0,3; sem weight decay; early stopping", "augmentation": "nenhuma (de propósito)",
     "seeds": "42", "camadas do XLS-R": "—",
     "por que esse valor": "4 s: mediana de duração da maioria dos bancos; 64 faixas: entrada pequena para uma CNN pequena; lote 64 e Adam 0,001: Aula 6; dropout 0,3: regularização moderada da Aula 7; paciência 5: early stopping da Aula 7; sem augmentation: primeiro medir o modelo puro",
     "efeito observado": "validação EER 1,3%; In-the-Wild AUC 0,211 (ordena ao contrário); CORAA 41,6% de reais acusados",
     "fonte": "docs/HISTORICO_VERSOES.md (v1); notebooks/versions/v1_baseline.ipynb, seção 2"},
    {"versão": "v2 (v1r, v2a, v2b)",
     "entrada": "igual à v1; teste por janelas de 4 s com passo de 2 s (média dos logits)",
     "arquitetura": "igual à v1", "épocas (máx. / escolhida)": "30 / 13, 20, 17 (v1r, v2a, v2b; checkpoint pela EER média por fonte)",
     "lote": "64", "otimizador e taxa": "Adam, 0,001", "regularização": "dropout 0,3; sem weight decay; early stopping (paciência 5)",
     "augmentation": "v1r: nenhuma; v2a: volume (RMS −26 dB e ganho −12 a +6 dB, chance 0,5 cada); v2b: v2a + ruído branco/rosa/marrom (SNR 5 a 30 dB), banda de telefone (8 kHz) e mu-law 8 bits, chance 0,3 cada",
     "seeds": "42 (uma execução por experimento)", "camadas do XLS-R": "—",
     "por que esse valor": "RMS −26 dB: nível típico de fala (EDA da v1); SNR 5 a 30 dB: de ruído forte a quase limpo; 8 kHz e mu-law: telefonia; chance 0,3: ~1/3 dos exemplos continua limpo; checkpoint pela EER média por fonte: a validação é 79% DFADD",
     "efeito observado": "dev_shift AUC 0,325 / 0,386 / 0,389: sem ganho além do acaso; v2b robusto às degradações que viu",
     "fonte": "docs/HISTORICO_VERSOES.md (v2); notebooks/versions/v2_augmentation.ipynb, seção 2"},
    {"versão": "v3 (repeat × zero)",
     "entrada": "igual à v1r; áudio < 4 s repetido (repeat) ou completado com zeros à direita (zero)",
     "arquitetura": "igual à v1", "épocas (máx. / escolhida)": "30 / repeat 9, 12, 9; zero 7, 19, 11 (seeds 42, 123, 2026)",
     "lote": "64", "otimizador e taxa": "Adam, 0,001", "regularização": "dropout 0,3; sem weight decay; early stopping (paciência 5)",
     "augmentation": "nenhuma", "seeds": "42, 123 e 2026 (3 por condição)", "camadas do XLS-R": "—",
     "por que esse valor": "3 seeds: medir o acaso do treino (a v2 não separava efeito de acaso); recorte determinístico: as duas condições veem os mesmos trechos",
     "efeito observado": "padding sem efeito demonstrado; AUC do dev_shift de 0,146 a 0,378 só mudando a seed (a régua do acaso)",
     "fonte": "docs/HISTORICO_VERSOES.md (v3, pré-registro e fechamento); notebooks/versions/v3_padding.ipynb"},
    {"versão": "v4 (F0, sistema final)",
     "entrada": "16 kHz mono; corte de silêncio nas bordas (35 dB; se sobrar < 0,5 s, não corta); até 8 s do meio, sem completar; sem log-mel (o XLS-R lê a onda)",
     "arquitetura": "XLS-R 300M da Meta congelado (24 camadas, 1.024 números a cada 20 ms); média no tempo das camadas 5, 6 e 7 (3.072 números) + regressão logística (3.073 parâmetros)",
     "épocas (máx. / escolhida)": f"não se aplica: lbfgs até 1.000 iterações (convergiu em {int(f0.n_iter_[0])})",
     "lote": "não se aplica (lbfgs usa os 40.000 áudios de treino de uma vez)",
     "otimizador e taxa": "lbfgs do scikit-learn (tolerância 0,0001), sem taxa de aprendizado",
     "regularização": "L2 com C = 0,1 (grade 0,001 / 0,01 / 0,1 / 1)", "augmentation": "nenhuma",
     "seeds": "42 (dados); a regressão é determinística", "camadas do XLS-R": "5, 6 e 7",
     "por que esse valor": "C = 0,1 e as camadas 5, 6, 7: menor EER média por fonte na validação (5,92%); 8 s: o dobro dos 4 s, ~90% dos áudios cabem inteiros; 40.000 áudios: o maior treino em que o teto por fonte ainda vale",
     "efeito observado": "dev_shift AUC 0,983; In-the-Wild 0,979; ganho sobre só a última camada não demonstrado; corte de silêncio sem efeito (ablação); com mais freio o dev_shift melhora (diagnóstico, figura 07)",
     "fonte": "docs/HISTORICO_VERSOES.md (v4); notebooks/versions/v4_ssl.ipynb, seção 2; results/v4_*"},
    {"versão": "v6 (F2, saída do funil)",
     "entrada": "igual à v4", "arquitetura": "XLS-R congelado, só a camada 11 + regressão logística (1.025 parâmetros)",
     "épocas (máx. / escolhida)": "não se aplica (lbfgs)", "lote": "não se aplica",
     "otimizador e taxa": "lbfgs do scikit-learn", "regularização": "L2 com C = 1; pesos iguais por fonte de treino",
     "augmentation": "nenhuma", "seeds": "42 (dados; bootstrap 2.000 reamostragens)", "camadas do XLS-R": "11",
     "por que esse valor": "camada 11 e C = 1: maior AUC da pior fonte (validação + dev_shift); pesos por fonte: +0,006 na pior fonte, passou a régua de +0,003",
     "efeito observado": "pior que a v4 no teste (média da acurácia 0,829 × 0,851; BRSpeech-DF AUC 0,744 × 0,842)",
     "fonte": "docs/HISTORICO_VERSOES.md (v6); notebooks/versions/v6_funil.ipynb, seção 2; results/v6_funil.csv"},
    {"versão": "v6 (E4: MLP desafiante)",
     "entrada": "a mesma da RL básica (camada 11, média)", "arquitetura": "MLP: 1 camada escondida de 256, ReLU, dropout; 262.657 parâmetros",
     "épocas (máx. / escolhida)": f"30 / {mlp_ep} (seeds 42, 123, 2026; paciência 5)", "lote": "256",
     "otimizador e taxa": "AdamW, 0,001", "regularização": "dropout 0,3; weight decay 0,001; early stopping",
     "augmentation": "nenhuma", "seeds": "42, 123 e 2026", "camadas do XLS-R": "11",
     "por que esse valor": "a MLP da v4, sem ajuste novo; o early stopping evita overfitting",
     "efeito observado": "AUC da pior fonte 0,902 a 0,912 × 0,904 da RL: nenhuma seed com diferença relevante",
     "fonte": "docs/HISTORICO_VERSOES.md (v6, E4); results/v6_e4_cabecas.csv"},
    {"versão": "v6 (E4: CNN desafiante)",
     "entrada": "as 25 saídas do XLS-R (média e desvio: 25 × 2.048)", "arquitetura": "CNN ao longo das camadas: Conv1d(2048→64, k=3) + BatchNorm + ReLU, Conv1d(64→64, k=3) + ReLU, média, Linear; 405.825 parâmetros",
     "épocas (máx. / escolhida)": f"30 / {cnn_ep} (seeds 42, 123, 2026; paciência 5)", "lote": "256",
     "otimizador e taxa": "AdamW, 0,001", "regularização": "weight decay 0,001; early stopping",
     "augmentation": "nenhuma", "seeds": "42, 123 e 2026", "camadas do XLS-R": "todas (0 a 24)",
     "por que esse valor": "uma CNN pequena (2 convoluções) para não ter muito mais parâmetros que o MLP",
     "efeito observado": "AUC da pior fonte 0,906 a 0,910 × 0,904 da RL: nenhuma seed com diferença relevante",
     "fonte": "docs/HISTORICO_VERSOES.md (v6, pré-registro E4); results/v6_e4_cabecas.csv"},
]
pd.DataFrame(HPV).to_csv(SLIDES / "hiperparametros_versoes.csv", index=False, encoding="utf-8")
log(f"tabela {SLIDES / 'hiperparametros_versoes.csv'} ({len(HPV)} linhas)")

# ======================================================================== INDEX.md e resumo
idx_md = ["# Figuras para os slides (sistema v4 = F0)", "",
          "Geradas por `scripts/v4final_figuras.py` (13,33 × 7,5 pol., dpi 180). Corte A = calibrado no estúdio "
          f"(nota {br(LIM['A'])}); corte B = calibrado na internet (nota {br(LIM['B'])}).", "",
          "| figura | o que mostra | o que olhar |", "|:--|:--|:--|"]
idx_md += [f"| `{r['figura']}` | {r['o que mostra']} | {r['o que olhar']} |" for r in sorted(INDEX, key=lambda r: r["figura"])]
idx_md += ["", "Tabelas: `grupos_bancos.csv` (banco, grupo, o que é, real/IA, idioma, ano, papel, nº de áudios) e "
           "`hiperparametros_versoes.csv` (hiperparâmetros de v1, v2, v3, v4 e v6)."]
(SLIDES / "INDEX.md").write_text("\n".join(idx_md) + "\n", encoding="utf-8")

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
          "g3": proj_info, "melhor_camada_v4": MELHOR, "log_mel": LOGMEL_INFO, "fonte_das_figuras": FONTE,
          "controle_rotulos_embaralhados": ler_json(R / "v4_atalhos.json")["d_controle_rotulos_embaralhados"],
          "minutos": round((time.time() - T0) / 60, 1), "pico_memoria_gb": round(PICO[0], 2)}
(R / "v4final_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"FIM. pico de memória {PICO[0]:.2f} GB")
