#!/usr/bin/env python3
"""v7: teste mestre fixo e 5 receitas (T0 a T4), só regressão logística sobre os vetores já extraídos.

Regras: docs/PROTOCOLO_V7.md (congelado ANTES de treinar). Nada é extraído nem baixado: os vetores vêm
dos caches da v4 (`data/processed/v4_cache`, `v4_cache_teste`) e do teste da internet
(`data/processed/teste_internet_cache`). Só a média no tempo das saídas 5, 6 e 7 do XLS-R (3.072 números).

Etapas (uma por processo; cada uma retomável e determinística):
    python scripts/v7.py vetores    # junta os vetores (camadas 5, 6, 7) -> data/processed/v7_vetores/
    python scripts/v7.py splits     # data/processed/splits_v7.parquet + salvaguardas (assert; senão BLOQUEIO)
    python scripts/v7.py treinar    # T1 a T4 (grade de C pela validação) + T0 -> data/processed/v7_notas.parquet
    python scripts/v7.py escolher   # SÓ validação: C, regra de corte, receita vencedora -> results/v7_escolha.json
    python scripts/v7.py avaliar    # teste mestre (uma vez) + controles + diagnósticos -> results/v7_*.csv/json
    python scripts/v7.py figuras    # results/slides/v7_*.png
    --quick: subamostras pequenas, saídas com sufixo _quick (só para testar o código; números não são lidos)
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import teste_internet as ti  # noqa: E402  (reaproveita: Vigia, sem_suspensao, carregar_f0, boot_pesos, metricas_ponderadas)

R = Path("results")
LOGS = Path("logs")
DP = Path("data/processed")
STATUS = LOGS / "v7_status.txt"
BLOQUEIO = LOGS / "v7_BLOQUEIO.txt"
LIMITE_GB = 10.0
SEED = 42
N_BOOT = 2000
CAMADAS = (5, 6, 7)
GRADE_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MIN_POR_CLASSE = 20
MIN_LOCUTORES = 30
COS_QUASE = 0.9995
LARGURA_FAIXA_S = 0.5
RECEITAS = ("T0", "T1", "T2", "T3", "T4")
DESCR = {"T0": "F0 sem retreinar", "T1": "estúdio + ML-ITW", "T2": "estúdio + RADAR",
         "T3": "estúdio + ML-ITW + RADAR", "T4": "T3, peso igual por fonte"}
BLOCOS_TREINO = {"T1": ("estudio", "ml_itw"), "T2": ("estudio", "radar_eval"),
                 "T3": ("estudio", "ml_itw", "radar_eval"), "T4": ("estudio", "ml_itw", "radar_eval")}
RADAR_LADO = {"zh": "treino", "zh-tw": "treino", "ja": "treino", "en-sg": "validacao", "vi": "validacao",
              "en": "teste_mestre"}             # AJUSTE 1 da tarefa
COINCIDENTES = {"mlitw:ScarlettJohansson": "itw:Scarlett Johansson", "mlitw:Trump": "itw:Donald Trump",
                "mlitw:obama": "itw:Barack Obama"}
CONTROLES = ("llamapartialspoof", "brspeech_df", "coraa", "asvspoof2019_la", "asvspoof2021_df")
PRINCIPAIS = ("ml_itw", "in_the_wild", "radar_en")          # AJUSTE 2
REGRAS = ("A'", "M", "F5")

QUICK = False


def sfx():
    return "_quick" if QUICK else ""


def agora():
    return datetime.now().isoformat(timespec="seconds")


def escrever(p, txt):
    from src.arquivos import gravar_atomico
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    gravar_atomico(Path(p), lambda t: Path(t).write_text(txt, encoding="utf-8"))


def salvar_json(obj, p):
    escrever(p, json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def status(etapa, msg):
    linha = f"[{agora()}] v7{' QUICK' if QUICK else ''} | etapa {etapa} | {msg} | pico memória " \
            f"{ti.VIGIA.pico if ti.VIGIA else float('nan'):.2f} GB (limite {LIMITE_GB})\n"
    print(linha.strip(), flush=True)
    alvo = STATUS if not QUICK else LOGS / "v7_status_quick.txt"
    # append (e não troca atômica): no Windows, um leitor com o arquivo aberto faz a troca falhar
    for tentativa in range(5):
        try:
            with open(alvo, "a", encoding="utf-8") as f:
                f.write(linha)
            break
        except OSError:
            time.sleep(1)


def pasta_vetores():
    return DP / "v7_vetores"


def arq_splits():
    return DP / "splits_v7.parquet"          # o mesmo no QUICK (a divisão não depende do treino)


def arq_notas():
    return DP / f"v7_notas{sfx()}.parquet"


# ======================================================================== vetores
FONTES_CACHE = [("data/processed/v4_cache", "treino", "v4"), ("data/processed/v4_cache", "val", "v4"),
                ("data/processed/v4_cache", "dev_shift", "v4")] + \
               [("data/processed/v4_cache_teste", f"teste_{b}", "v4") for b in
                ("in_the_wild",) + CONTROLES] + \
               [("data/processed/teste_internet_cache", "radar_eval", "internet"),
                ("data/processed/teste_internet_cache", "ml_itw", "internet")]
ESPERADO = {"treino": 40000, "val": 19999, "dev_shift": 10614, "radar_eval": 102725, "ml_itw": 24529}
COLS_META = ["idx_manifest", "label", "dur_original_s", "silencio_inicio_s", "silencio_fim_s", "rms_db", "pico_db",
             "banda_hz", "erro_leitura"]


def etapa_vetores(a):
    from src.cache_v4 import Conjunto
    pasta = pasta_vetores()
    if (pasta / "concluido.json").exists():
        status("vetores", "já pronto (concluido.json); nada a fazer")
        return 0
    pasta.mkdir(parents=True, exist_ok=True)
    man4 = pd.read_parquet(DP / "manifest.parquet")
    mani = pd.read_parquet(DP / "manifest_teste_internet.parquet")
    metas, blocos, n_tot = [], [], 0
    for cache, conj, qual in FONTES_CACHE:
        c = Conjunto(cache, conj)
        n = len(c)
        if conj in ESPERADO:
            assert n == ESPERADO[conj], (conj, n)
        m = c.meta[COLS_META].copy()
        assert (m.erro_leitura.fillna(0) == 0).all(), f"{conj}: há clipes ilegíveis"
        X = np.hstack([c.camada(l).astype(np.float16) for l in CAMADAS])     # float16 do cache: conversão exata
        assert X.shape == (n, 3 * 1024) and np.isfinite(X).all(), (conj, X.shape)
        man = man4 if qual == "v4" else mani
        info = man.loc[m.idx_manifest.values]
        m["manifesto"] = qual
        m["conjunto_cache"] = conj
        m["caminho"] = info.path.values
        m["fonte"] = info.source_dataset.values
        m["locutor"] = info.speaker_id.values
        m["idioma"] = info.language.values
        m["gerador"] = info.generator.values
        m["duracao_s"] = info.duration_s.values
        m["eval_subset"] = info.eval_subset.values
        m["split_manifesto"] = info.split.values
        m["original_split"] = info.original_split.values
        if qual == "internet":
            m["plataforma"] = info.plataforma.values
            m["tamanho_bytes"] = info.tamanho_bytes.values
        else:
            m["plataforma"] = None
            m["tamanho_bytes"] = np.nan
        assert (m.label.values == info.label.values).all(), conj
        metas.append(m)
        blocos.append(X)
        n_tot += n
        status("vetores", f"{conj}: {n} clipes")
    meta = pd.concat(metas, ignore_index=True)
    X = np.concatenate(blocos)
    del blocos
    meta["linha"] = np.arange(len(meta))
    meta["hash_vetor"] = [hashlib.sha1(X[i].tobytes()).hexdigest() for i in range(len(X))]
    np.save(pasta / "X.npy", X)
    meta.to_parquet(pasta / "meta.parquet", index=False)
    salvar_json({"quando": agora(), "n": int(len(meta)), "camadas": list(CAMADAS), "dtype": "float16",
                 "por_conjunto": meta.groupby("conjunto_cache").size().to_dict()}, pasta / "concluido.json")
    status("vetores", f"CONCLUÍDO: {len(meta)} clipes x {X.shape[1]}")
    return 0


def carregar_vetores(mmap=True):
    pasta = pasta_vetores()
    assert (pasta / "concluido.json").exists(), "rode a etapa 'vetores'"
    X = np.load(pasta / "X.npy", mmap_mode="r" if mmap else None)
    meta = pd.read_parquet(pasta / "meta.parquet")
    return X, meta


# ======================================================================== splits
def bloquear(msg):
    escrever(BLOQUEIO, f"{agora()}\n{msg}\n")
    status("splits", f"BLOQUEIO: {msg}")
    raise SystemExit(4)


def checar(cond, msg):
    if not cond:
        bloquear(msg)


def parear(df, chave_extra, rng):
    """Índices (do df) pareados: em cada (chave_extra, faixa de 0,5 s), o mesmo nº de reais e falsos."""
    faixa = np.floor(df.duracao_s.values / LARGURA_FAIXA_S).astype(int)
    g = pd.DataFrame({"i": df.index.values, "k": df[chave_extra].values if chave_extra else 0, "f": faixa,
                      "y": df.label.values})
    sel = []
    for (_, _), h in g.groupby(["k", "f"], sort=True):
        r, f = h[h.y == 0].i.values, h[h.y == 1].i.values
        n = min(len(r), len(f))
        if n:
            sel += [rng.choice(r, n, replace=False), rng.choice(f, n, replace=False)]
    return np.sort(np.concatenate(sel))


def grupo_idioma(langs):
    s = pd.Series(langs).value_counts()
    top = s.index[0]
    return top if top in ("chinese", "english") else "outro"


def etapa_splits(a):
    X, meta = carregar_vetores()
    d = meta.copy()
    d["split"] = None
    d["bloco"] = None
    d["regra"] = None
    d["pareado_treino"] = False
    d["pareado_teste"] = False
    d["pessoa"] = None
    cc = d.conjunto_cache
    # ---------------- estúdio (v4)
    d.loc[cc == "treino", ["split", "bloco", "regra"]] = ["treino", "estudio", "v4 treino (40.000, seed 42)"]
    d.loc[cc == "val", ["split", "bloco", "regra"]] = ["validacao", "estudio", "v4 validação (19.999)"]
    for b in CONTROLES:
        d.loc[cc == f"teste_{b}", ["split", "bloco", "regra"]] = ["controle", "controle", "teste antigo de estúdio"]
    # ---------------- In-the-Wild
    d.loc[cc == "dev_shift", ["split", "bloco", "regra"]] = ["validacao", "in_the_wild", "dev_shift (16 locutores)"]
    d.loc[cc == "teste_in_the_wild", ["split", "bloco", "regra"]] = ["teste_mestre", "in_the_wild",
                                                                     "test_final (38 locutores)"]
    itw = d.bloco == "in_the_wild"
    d.loc[itw, "pessoa"] = d.loc[itw, "locutor"]
    # ---------------- RADAR
    rad = cc == "radar_eval"
    checar((d.loc[rad, "original_split"] == "eval").all() and d.loc[rad, "caminho"].str.contains("RADAR2026-eval").all()
           and not d.loc[rad, "caminho"].str.contains("RADAR2026-dev").any(), "há clipe que não é do eval do RADAR")
    d.loc[rad, "bloco"] = "radar_eval"
    d.loc[rad, "split"] = d.loc[rad, "idioma"].map(RADAR_LADO)
    d.loc[rad, "regra"] = "RADAR por idioma: " + d.loc[rad, "idioma"]
    checar(d.loc[rad, "split"].notna().all(), "idioma do RADAR sem lado")
    rng = np.random.default_rng(SEED)
    tr = d[rad & (d.split == "treino")]
    d.loc[parear(tr, "idioma", rng), "pareado_treino"] = True
    te = d[rad & (d.split == "teste_mestre")]
    d.loc[parear(te, None, np.random.default_rng(SEED)), "pareado_teste"] = True
    # ---------------- ML-ITW (locutor atômico)
    ml = cc == "ml_itw"
    d.loc[ml, "bloco"] = "ml_itw"
    m = d[ml]
    loc = m.groupby("locutor").agg(ncls=("label", "nunique"), langs=("idioma", list))
    com_pt = set(m[m.idioma == "portuguese"].locutor)
    lado, regra = {}, {}
    itw_lado = d[itw].groupby("locutor").split.agg(lambda s: set(s))
    for l in loc.index:
        if l in com_pt:
            lado[l], regra[l] = "teste_mestre", "(1) tem áudio em português"
        elif l in COINCIDENTES:
            alvo = itw_lado.get(COINCIDENTES[l])
            checar(alvo is not None and len(alvo) == 1, f"coincidente {l}: lado no In-the-Wild = {alvo}")
            lado[l], regra[l] = next(iter(alvo)), f"(3) mesma pessoa que {COINCIDENTES[l]} no In-the-Wild"
        elif loc.loc[l, "ncls"] == 1:
            lado[l], regra[l] = "teste_mestre", "(2) uma classe só"
    resto = [l for l in loc.index if l not in lado]
    est = pd.Series({l: grupo_idioma(loc.loc[l, "langs"]) for l in resto})
    rng = np.random.default_rng(SEED)
    for g in sorted(est.unique()):
        ls = np.array(sorted(est[est == g].index))
        ls = ls[rng.permutation(len(ls))]
        n = len(ls)
        ntr, nva = int(round(0.60 * n)), int(round(0.15 * n))
        for i, l in enumerate(ls):
            lado[l] = "treino" if i < ntr else ("validacao" if i < ntr + nva else "teste_mestre")
            regra[l] = f"(4) sorteio estratificado por locutor (grupo de idioma '{g}', seed 42, 60/15/25)"
    d.loc[ml, "split"] = d.loc[ml, "locutor"].map(lado)
    d.loc[ml, "regra"] = d.loc[ml, "locutor"].map(regra)
    d.loc[ml, "pessoa"] = d.loc[ml, "locutor"].map(lambda l: COINCIDENTES.get(l, l))
    checar(d.split.notna().all(), "clipe sem lado")

    # ---------------- salvaguardas
    lados = ("treino", "validacao", "teste_mestre")
    S = {s: d[d.split == s] for s in lados}
    log = {}
    for i, s1 in enumerate(lados):
        for s2 in lados[i + 1:]:
            inter = set(S[s1].caminho) & set(S[s2].caminho)
            checar(not inter, f"caminhos em comum entre {s1} e {s2}: {list(inter)[:5]}")
            pi = set(S[s1].pessoa.dropna()) & set(S[s2].pessoa.dropna())
            checar(not pi, f"locutores (pessoa) em comum entre {s1} e {s2}: {sorted(pi)[:10]}")
    log["caminhos_em_comum"] = 0
    log["locutores_em_comum"] = 0
    for l, alvo in COINCIDENTES.items():
        lm = set(d[d.locutor == l].split)
        li = set(d[d.locutor == alvo].split)
        checar(len(lm) == 1 and lm == li, f"coincidente {l}: ML-ITW {lm} x In-the-Wild {li}")
    log["coincidentes"] = {l: sorted(set(d[d.locutor == l].split)) for l in COINCIDENTES}
    # duplicatas exatas (hash dos bytes do vetor) entre lados.
    # Achado na 1ª execução (antes de qualquer treino): 12 grupos de reais do CodecFake+ (CoSG) com vetor idêntico
    # entre o treino e a validação de ESTÚDIO da v4 (o mesmo áudio de referência em pastas de geradores diferentes).
    # Como o treino é fixo (os 40.000 da v4), o membro que NÃO está no treino sai da avaliação (split "excluido").
    h = d[d.split.isin(lados)].groupby("hash_vetor").split.agg(lambda s: set(s))
    cruz = h[h.map(len) > 1]
    excl = d.hash_vetor.isin(cruz[cruz.map(lambda s: "treino" in s)].index) & d.split.isin(["validacao", "teste_mestre"])
    log["duplicatas_exatas_excluidas_da_avaliacao"] = d.loc[excl, ["caminho", "split", "fonte", "label"]].to_dict("records")
    d.loc[excl, "regra"] = d.loc[excl, "regra"] + " | excluído: vetor idêntico a um clipe do treino"
    d.loc[excl, "split"] = "excluido"
    h = d[d.split.isin(lados)].groupby("hash_vetor").split.nunique()
    dup = h[h > 1]
    log["duplicatas_exatas_entre_lados"] = int(len(dup))
    hc = d.groupby("hash_vetor").split.agg(lambda s: set(s))
    ctrl = hc[hc.map(lambda s: "controle" in s and len(s) > 1)]
    log["duplicatas_exatas_controle_x_outros_lados (informativo)"] = int(len(ctrl))
    checar(len(dup) == 0, f"{len(dup)} vetores idênticos em lados diferentes (ex.: "
           f"{d[d.hash_vetor.isin(dup.index[:3])][['caminho', 'split']].to_dict('records')})")
    # quase duplicatas (cosseno >= 0,9995) no ML-ITW e no In-the-Wild: treino/validação x teste_mestre
    f0 = ti.carregar_f0()
    mu = np.concatenate([f0["mu"][l] for l in CAMADAS]).astype(np.float32)
    sd = np.concatenate([f0["sd"][l] for l in CAMADAS]).astype(np.float32)
    viz = d.bloco.isin(["ml_itw", "in_the_wild"])
    A_i = d[viz & d.split.isin(["treino", "validacao"])].linha.values
    B_i = d[viz & (d.split == "teste_mestre")].linha.values

    def norm(idx):
        Z = (np.asarray(X[np.sort(idx)], dtype=np.float32) - mu) / sd
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        return np.sort(idx), Z
    A_i, ZA = norm(A_i)
    B_i, ZB = norm(B_i)
    pares, cos_max = [], -1.0
    for i in range(0, len(ZA), 2000):
        S_ = ZA[i:i + 2000] @ ZB.T
        cos_max = max(cos_max, float(S_.max()))
        r, c = np.nonzero(S_ >= COS_QUASE)
        pares += [(int(A_i[i + a]), int(B_i[b]), float(S_[a, b])) for a, b in zip(r, c)]
    log["quase_duplicatas_criterio"] = (f"cosseno >= {COS_QUASE} entre vetores padronizados pela média e desvio do "
                                        f"treino da v4 (os do F0); {len(A_i)} clipes de treino/validação x {len(B_i)} "
                                        f"do teste mestre (ML-ITW e In-the-Wild)")
    log["cosseno_maximo_entre_lados"] = cos_max
    movidos = []
    if pares:
        # regra (protocolo): o par vai para o mesmo lado; para manter o locutor atômico, move-se o LOCUTOR inteiro
        # do clipe de treino/validação para o teste mestre (a pessoa sai do treino/validação).
        for a_, b_, cs in pares:
            ra, rb = d.loc[a_], d.loc[b_]
            movidos.append({"clipe_treino_val": ra.caminho, "lado_antes": ra.split, "locutor": ra.locutor,
                            "clipe_teste": rb.caminho, "locutor_teste": rb.locutor, "cosseno": cs})
        for loc_ in sorted({m_["locutor"] for m_ in movidos}):
            pes = d.loc[d.locutor == loc_, "pessoa"].iloc[0]
            sel = (d.pessoa == pes) & d.bloco.isin(["ml_itw", "in_the_wild"])
            d.loc[sel, "split"] = "teste_mestre"
            d.loc[sel, "regra"] = d.loc[sel, "regra"] + " | movido ao teste mestre (quase duplicata)"
    log["quase_duplicatas_pares"] = movidos
    # reconferir depois de mover
    S = {s: d[d.split == s] for s in lados}
    for i, s1 in enumerate(lados):
        for s2 in lados[i + 1:]:
            checar(not (set(S[s1].pessoa.dropna()) & set(S[s2].pessoa.dropna())), "locutores em comum após mover")
            checar(not (set(S[s1].caminho) & set(S[s2].caminho)), "caminhos em comum após mover")
    log["dev_do_radar"] = "nenhum clipe (todos com original_split = eval e caminho RADAR2026-eval)"

    # ---------------- contagens
    cont = []
    for (bl, sp), g in d.groupby(["bloco", "split"], sort=True):
        for lab in (0, 1):
            h_ = g[g.label == lab]
            cont.append({"bloco": bl, "split": sp, "classe": "real" if lab == 0 else "falso", "clipes": int(len(h_)),
                         "locutores": int(h_.locutor.nunique()) if bl in ("ml_itw", "in_the_wild") else None,
                         "clipes_pareados_treino": int(h_.pareado_treino.sum()) if bl == "radar_eval" else None,
                         "clipes_pareados_teste": int(h_.pareado_teste.sum()) if bl == "radar_eval" else None})
    cont = pd.DataFrame(cont)
    regras_ml = d[d.bloco == "ml_itw"].groupby(["regra", "split"]).agg(locutores=("locutor", "nunique"),
                                                                       clipes=("label", "size"),
                                                                       falsos=("label", "sum")).reset_index()
    vml = cont[(cont.bloco == "ml_itw") & (cont.split == "validacao")]
    aviso = [f"validação do ML-ITW com {int(r.locutores)} locutores {r.classe} (< 10)" for r in vml.itertuples()
             if r.locutores < 10]
    log["avisos"] = aviso
    out = d[["linha", "caminho", "fonte", "bloco", "locutor", "pessoa", "idioma", "label", "plataforma", "gerador",
             "split", "regra", "pareado_treino", "pareado_teste", "duracao_s", "conjunto_cache", "manifesto",
             "idx_manifest", "eval_subset", "hash_vetor"]].copy()
    out["plataforma_ou_gerador"] = out.plataforma.where(out.plataforma.notna(), out.gerador)
    from src.arquivos import salvar_csv, salvar_parquet, sha256_arquivo
    salvar_parquet(out, arq_splits())
    sha = sha256_arquivo(arq_splits())
    salvar_csv(cont, R / f"v7_splits_contagens{sfx()}.csv")
    salvar_csv(regras_ml, R / f"v7_splits_mlitw_regras{sfx()}.csv")
    salvar_json({"quando": agora(), "arquivo": str(arq_splits()), "sha256": sha, "n": int(len(out)),
                 "por_split": out.groupby("split").size().to_dict(), "salvaguardas": log}, R / f"v7_splits{sfx()}.json")
    status("splits", f"CONCLUÍDO: {len(out)} clipes; SHA256 {sha}; avisos {aviso}; quase duplicatas {len(movidos)}")
    return 0


# ======================================================================== treino
def chave_val(sp):
    """Fonte de avaliação na validação: estúdio por banco; os outros por bloco."""
    return np.where(sp.bloco == "estudio", sp.fonte, sp.bloco)


def auc_val_por_fonte(y, z, fontes):
    from sklearn.metrics import roc_auc_score
    out = {}
    for f in sorted(set(fontes)):
        m = fontes == f
        if (y[m] == 0).sum() >= MIN_POR_CLASSE and (y[m] == 1).sum() >= MIN_POR_CLASSE:
            out[f] = float(roc_auc_score(y[m], z[m]))
    return out


def pesos_t4(blocos, y):
    """Cada bloco soma o mesmo peso total; dentro dele, reais e falsos somam o mesmo. Média 1."""
    w = np.zeros(len(y))
    nb = len(set(blocos))
    for b in set(blocos):
        for c in (0, 1):
            m = (blocos == b) & (y == c)
            w[m] = (len(y) / nb / 2) / m.sum()
    return w


def pontuar(X, linhas, mu, sd, coef, b, lote=20000):
    z = np.empty(len(linhas))
    for i in range(0, len(linhas), lote):
        idx = linhas[i:i + lote]
        z[i:i + lote] = ((np.asarray(X[idx], dtype=np.float32) - mu) / sd) @ coef + b
    return z


def etapa_treinar(a):
    from sklearn.linear_model import LogisticRegression

    from src.arquivos import salvar_csv
    X, _ = carregar_vetores()
    sp = pd.read_parquet(arq_splits())
    parc = R / f"v7_parcial{sfx()}"
    parc.mkdir(parents=True, exist_ok=True)
    val = sp[sp.split == "validacao"]
    if QUICK:
        val = val.groupby(chave_val(val)).sample(frac=0.1, random_state=SEED)
    lv, yv, fv = val.linha.values, val.label.values.astype(int), chave_val(val)
    # ---- T0: F0 do disco
    f0 = ti.carregar_f0()
    mu0 = np.concatenate([f0["mu"][l] for l in CAMADAS]).astype(np.float32)
    sd0 = np.concatenate([f0["sd"][l] for l in CAMADAS]).astype(np.float32)
    modelos = {"T0": {"mu": mu0, "sd": sd0, "coef": f0["coef"].astype(np.float32), "b": f0["intercepto"], "C": f0["C"]}}
    grade = []
    zv0 = pontuar(X, lv, mu0, sd0, modelos["T0"]["coef"], modelos["T0"]["b"])
    a0 = auc_val_por_fonte(yv, zv0, fv)
    grade.append({"receita": "T0", "C": f0["C"], **{f"AUC val {k}": v for k, v in a0.items()},
                  "AUC val média por fonte": float(np.mean(list(a0.values()))), "iterações": None,
                  "convergiu": None, "n_treino": 40000, "segundos": 0.0, "escolhido": True})
    status("treinar", f"T0: AUC val média por fonte {grade[-1]['AUC val média por fonte']:.4f}")
    for rc in ("T1", "T2", "T3", "T4"):
        arq = parc / f"{rc}.npz"
        tr = sp[(sp.split == "treino") & sp.bloco.isin(BLOCOS_TREINO[rc])
                & ((sp.bloco != "radar_eval") | sp.pareado_treino)]
        if QUICK:
            tr = tr.groupby(["bloco", "label"]).sample(n=300, random_state=SEED)
        tr = tr.sort_values("linha")
        lt, yt, bt = tr.linha.values, tr.label.values.astype(int), tr.bloco.values
        if arq.exists():
            z = np.load(arq, allow_pickle=True)
            modelos[rc] = {k: z[k] for k in ("mu", "sd", "coef")} | {"b": float(z["b"]), "C": float(z["C"])}
            grade += json.loads(str(z["grade"]))
            status("treinar", f"{rc}: já pronto ({arq}); C = {modelos[rc]['C']}")
            continue
        Xt = np.asarray(X[lt], dtype=np.float32)
        mu = Xt.mean(0, dtype=np.float64).astype(np.float32)
        sd = Xt.std(0, dtype=np.float64).astype(np.float32)
        sd[sd == 0] = 1.0
        Xt -= mu
        Xt /= sd
        w = pesos_t4(bt, yt) if rc == "T4" else None
        cw = None if rc == "T4" else "balanced"
        linhas_g, melhor = [], None
        for C in GRADE_C:
            t0 = time.time()
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, class_weight=cw)
                lr.fit(Xt, yt, sample_weight=w)
            conv = not any("converge" in str(x.message).lower() for x in avisos)
            coef, b = lr.coef_.ravel().astype(np.float32), float(lr.intercept_[0])
            zv = pontuar(X, lv, mu, sd, coef, b)
            av = auc_val_por_fonte(yv, zv, fv)
            media = float(np.mean(list(av.values())))
            linhas_g.append({"receita": rc, "C": C, **{f"AUC val {k}": v for k, v in av.items()},
                             "AUC val média por fonte": media, "iterações": int(lr.n_iter_[0]), "convergiu": conv,
                             "n_treino": int(len(lt)), "segundos": round(time.time() - t0, 1), "escolhido": False})
            if melhor is None or media > melhor[0]:
                melhor = (media, C, coef, b)
            status("treinar", f"{rc} C={C:g}: AUC val média por fonte {media:.4f}; {int(lr.n_iter_[0])} iterações; "
                   f"convergiu {conv}; {time.time() - t0:.0f} s; n treino {len(lt)}")
        for g_ in linhas_g:
            g_["escolhido"] = g_["C"] == melhor[1]
        grade += linhas_g
        modelos[rc] = {"mu": mu, "sd": sd, "coef": melhor[2], "b": melhor[3], "C": melhor[1]}
        np.savez(arq, mu=mu, sd=sd, coef=melhor[2], b=melhor[3], C=melhor[1],
                 grade=json.dumps(linhas_g), n_treino=len(lt),
                 composicao=json.dumps(tr.groupby(["bloco", "label"]).size().rename("n").reset_index().to_dict("records")))
        del Xt
    salvar_csv(pd.DataFrame(grade), R / f"v7_grade_c{sfx()}.csv")
    # ---- notas de todos os clipes (treino incluso, marcado pelo split)
    notas = sp[["linha", "caminho", "fonte", "bloco", "split", "label", "locutor", "idioma", "plataforma",
                "pareado_treino", "pareado_teste", "duracao_s"]].copy()
    todas = sp.linha.values
    for rc in RECEITAS:
        mo = modelos[rc]
        notas[f"z_{rc}"] = pontuar(X, todas, mo["mu"], mo["sd"], mo["coef"], mo["b"])
    # conferência do T0 contra as notas gravadas (v4 e teste da internet)
    conf = {}
    if not QUICK:
        ant = pd.read_parquet(DP / "teste_internet_notas.parquet")[["idx_manifest", "fonte", "z_F0"]]
        meta = pd.read_parquet(pasta_vetores() / "meta.parquet")[["linha", "idx_manifest", "conjunto_cache"]]
        n2 = notas.merge(meta, on="linha")
        for f in ("radar_eval", "ml_itw"):
            g = n2[n2.conjunto_cache == f].merge(ant[ant.fonte == f], on="idx_manifest")
            conf[f] = float(np.abs(g.z_T0 - g.z_F0).max())
        v4 = pd.read_parquet(R / "v4_teste_notas.parquet")
        mapa = {"val": "val", "dev_shift": "dev_shift", **{f"teste_{b}": b for b in ("in_the_wild",) + CONTROLES}}
        for cc_, banco in mapa.items():
            g = n2[n2.conjunto_cache == cc_].merge(v4[v4.banco == banco][["idx_manifest", "z_S2"]], on="idx_manifest")
            conf[cc_] = float(np.abs(g.z_T0 - g.z_S2).max())
        checar(max(conf.values()) <= 1e-4, f"T0 não reproduz as notas gravadas do F0: {conf}")
    from src.arquivos import salvar_parquet
    salvar_parquet(notas, arq_notas())
    salvar_json({"quando": agora(), "C_escolhido": {rc: float(modelos[rc]["C"]) for rc in RECEITAS},
                 "conferencia_T0_dif_max_logit": conf}, R / f"v7_treino{sfx()}.json")
    status("treinar", f"CONCLUÍDO: notas de {len(notas)} clipes; C {[(rc, float(modelos[rc]['C'])) for rc in RECEITAS]}; "
           f"conferência T0 {conf}")
    return 0


# ======================================================================== escolher (SÓ validação)
def cortes_de(y, z, fontes):
    """As 3 regras na validação dada. fontes: chave de fonte por clipe."""
    from src.metrics import eer_e_limiar
    t = {"A'": eer_e_limiar(y, z)[1]}
    eers, f5 = [], []
    for f in sorted(set(fontes)):
        m = fontes == f
        nr, nf = (y[m] == 0).sum(), (y[m] == 1).sum()
        if nr >= MIN_POR_CLASSE and nf >= MIN_POR_CLASSE:
            eers.append(eer_e_limiar(y[m], z[m])[1])
        if nr >= MIN_POR_CLASSE:
            f5.append(float(np.quantile(z[m][y[m] == 0], 0.95)))
    t["M"] = float(np.median(eers))
    t["F5"] = float(np.median(f5))
    return t


def bal(y, z, t):
    p = z >= t
    return 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())


def etapa_escolher(a):
    from src.arquivos import salvar_csv
    notas = pd.read_parquet(arq_notas())
    v = notas[notas.split == "validacao"].copy()          # SÓ validação
    assert set(v.split) == {"validacao"}
    v["fv"] = chave_val(v)
    y, fv = v.label.values.astype(int), v.fv.values
    duas = [f for f in sorted(set(fv)) if ((y[fv == f] == 0).sum() >= MIN_POR_CLASSE and (y[fv == f] == 1).sum() >= MIN_POR_CLASSE)]
    linhas_v, linhas_c, esc = [], [], {}
    for rc in RECEITAS:
        z = v[f"z_{rc}"].values
        av = auc_val_por_fonte(y, z, fv)
        linhas_v.append({"receita": rc, "descrição": DESCR[rc], **{f"AUC {k}": x for k, x in av.items()},
                         "AUC média por fonte": float(np.mean(list(av.values()))),
                         "AUC pior fonte": float(min(av.values())), "pior fonte": min(av, key=av.get)})
        cortes = cortes_de(y, z, fv)
        loso = {}
        for r in REGRAS:
            accs = {}
            for f in duas:
                fora = fv != f
                t = cortes_de(y[fora], z[fora], fv[fora])[r]
                accs[f] = bal(y[fv == f], z[fv == f], t)
            loso[r] = float(np.mean(list(accs.values())))
            linhas_c.append({"receita": rc, "regra": r, "corte (validação inteira)": cortes[r],
                             "acurácia balanceada média (deixando uma fonte de fora)": loso[r],
                             **{f"bal. fora: {f}": x for f, x in accs.items()}})
        r_esc = max(REGRAS, key=lambda r: loso[r])
        esc[rc] = {"regra": r_esc, "cortes": cortes, "loso": loso}
    tv = pd.DataFrame(linhas_v)
    venc = tv.loc[tv["AUC média por fonte"].idxmax(), "receita"]
    salvar_csv(tv, R / f"v7_validacao{sfx()}.csv")
    salvar_csv(pd.DataFrame(linhas_c), R / f"v7_cortes_validacao{sfx()}.csv")
    salvar_json({"quando": agora(), "fontes_validacao_com_duas_classes": duas,
                 "receita_vencedora_pela_validacao": venc,
                 "criterio": "maior média da AUC por fonte na validação (cada fonte pesa igual; >= 20 clipes por classe)",
                 "por_receita": esc, "tabela_validacao": json.loads(tv.to_json(orient="records", force_ascii=False))},
                R / f"v7_escolha{sfx()}.json")
    status("escolher", f"CONCLUÍDO: vencedora pela validação {venc}; regras {[(k, e['regra']) for k, e in esc.items()]}")
    return 0


# ======================================================================== avaliar (teste mestre: UMA vez)
def ic(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def medir(g, rc, cortes, unidade, n_boot):
    """Métricas de um recorte para a receita rc. Devolve (linha, array de AUC do bootstrap ou None)."""
    from sklearn.metrics import roc_auc_score

    from src.metrics import eer_e_limiar
    y = g.label.values.astype(int)
    z = g[f"z_{rc}"].values.astype(np.float64)
    nr, nf = int((y == 0).sum()), int((y == 1).sum())
    L = {"n_real": nr, "n_falso": nf}
    if unidade == "locutor":
        lr_, lf_ = g[g.label == 0].locutor.nunique(), g[g.label == 1].locutor.nunique()
        L.update({"locutores_real": int(lr_), "locutores_falso": int(lf_),
                  "marca": "estimativa" if min(lr_, lf_) >= MIN_LOCUTORES else "indício (< 30 locutores por classe)",
                  "ic_unidade": "locutor"})
    else:
        L.update({"marca": "IC otimista (por clipe, sem locutor)" if g.bloco.iloc[0] == "radar_eval" else "IC por clipe",
                  "ic_unidade": "clipe"})
    calcula = nr >= MIN_POR_CLASSE and nf >= MIN_POR_CLASSE
    if calcula:
        eer, _ = eer_e_limiar(y, z)
        L.update({"AUC": float(roc_auc_score(y, z)), "EER": float(eer)})
    elif nf == 0:
        L["marca"] = "só reais (vale o FPR)"
    else:
        L["marca"] = f"só contagem (< {MIN_POR_CLASSE} clipes numa classe)"
    for r, t in cortes.items():
        p = z >= t
        L[f"TPR {r}"] = float(p[y == 1].mean()) if nf else np.nan
        L[f"FPR {r}"] = float(p[y == 0].mean()) if nr else np.nan
        L[f"bal {r}"] = 0.5 * (L[f"TPR {r}"] + 1 - L[f"FPR {r}"]) if (nr and nf) else np.nan
    L["nota média real"] = float(z[y == 0].mean()) if nr else np.nan
    L["nota média falso"] = float(z[y == 1].mean()) if nf else np.nan
    L["nota mediana real"] = float(np.median(z[y == 0])) if nr else np.nan
    L["nota mediana falso"] = float(np.median(z[y == 1])) if nf else np.nan
    aucs = None
    grupo = g.locutor.astype(str).values if unidade == "locutor" else np.arange(len(g))
    if n_boot and calcula:
        acc = {}
        for W in ti.boot_pesos(y, grupo, n_boot, SEED):
            for k, x in ti.metricas_ponderadas(y, z, W, cortes).items():
                acc.setdefault(k, []).append(x)
        for k, x in acc.items():
            lo, hi = ic(np.concatenate(x))
            k2 = k.replace("acurácia balanceada", "bal")
            L[f"{k2} IC2.5"], L[f"{k2} IC97.5"] = lo, hi
        aucs = np.concatenate(acc["AUC"])
    elif n_boot and nf == 0 and nr:
        acc = {r: [] for r in cortes}
        for W in ti.boot_pesos(y, grupo, n_boot, SEED):
            for r, t in cortes.items():
                acc[r].append(W @ (z >= t).astype(np.float32) / W.sum(1))
        for r in cortes:
            L[f"FPR {r} IC2.5"], L[f"FPR {r} IC97.5"] = ic(np.concatenate(acc[r]))
    return L, aucs


def visto(rc, recorte):
    if recorte.startswith("in_the_wild"):
        return "nunca visto"
    if recorte.startswith("ml_itw"):
        return "visto (ML-ITW no treino, outros locutores)" if rc in ("T1", "T3", "T4") else "não visto"
    if recorte.startswith("radar"):
        return ("idioma novo, mas fonte, gerador e canal já vistos" if rc in ("T2", "T3", "T4") else "não visto")
    return "controle (estúdio)"


def etapa_avaliar(a):
    from src.arquivos import salvar_csv
    marcador = R / f"v7_avaliar_concluido{sfx()}.json"
    if marcador.exists() and not a.forcar:
        status("avaliar", f"já avaliado ({marcador}); o teste mestre é avaliado UMA vez. Nada feito.")
        return 0
    esc = json.loads((R / f"v7_escolha{sfx()}.json").read_text(encoding="utf-8"))
    notas = pd.read_parquet(arq_notas())
    n_boot = 50 if QUICK else N_BOOT
    tm = notas[notas.split == "teste_mestre"]
    ct = notas[notas.split == "controle"]
    if QUICK:
        tm = tm.groupby(["bloco", "label"]).sample(frac=0.05, random_state=SEED)
        ct = ct.groupby(["fonte", "label"]).sample(frac=0.05, random_state=SEED)
    rad_en = tm[tm.bloco == "radar_eval"]
    ml = tm[tm.bloco == "ml_itw"]
    recortes = {  # nome -> (DataFrame, unidade do IC, papel)
        "ml_itw": (ml, "locutor", "principal"),
        "in_the_wild": (tm[tm.bloco == "in_the_wild"], "locutor", "principal"),
        "radar_en": (rad_en, "clipe", "principal"),
        "radar_en_pareado": (rad_en[rad_en.pareado_teste], "clipe", "inglês"),
        "ml_itw_ingles": (ml[ml.idioma == "english"], "locutor", "inglês"),
        "ml_itw_portugues": (ml[ml.idioma == "portuguese"], "locutor", "à parte (indício)"),
    }
    for b in CONTROLES:
        recortes[b] = (ct[ct.fonte == b], "clipe", "controle")
    t0 = time.time()
    linhas, AUCS = [], {}
    for rc in RECEITAS:
        e = esc["por_receita"][rc]
        cortes = {r: e["cortes"][r] for r in REGRAS}
        for nome, (g, un, papel) in recortes.items():
            L, aucs = medir(g, rc, cortes, un, n_boot)
            linhas.append({"receita": rc, "descrição": DESCR[rc], "recorte": nome, "papel": papel,
                           "domínio": visto(rc, nome), "regra escolhida (validação)": e["regra"], **L})
            AUCS[(rc, nome)] = aucs
        status("avaliar", f"{rc}: {len(recortes)} recortes ({(time.time() - t0) / 60:.1f} min)")
    tab = pd.DataFrame(linhas)
    salvar_csv(tab, R / f"v7_teste_mestre{sfx()}.csv")
    # ---- métrica principal: média da AUC por fonte em PRINCIPAIS e a pior fonte (IC pelo bootstrap; pareado)
    princ = []
    for rc in RECEITAS:
        M = np.vstack([AUCS[(rc, s)] for s in PRINCIPAIS])
        pt = tab[(tab.receita == rc) & tab.recorte.isin(PRINCIPAIS)].set_index("recorte").loc[list(PRINCIPAIS), "AUC"]
        M0 = np.vstack([AUCS[("T0", s)] for s in PRINCIPAIS])
        d_media = M.mean(0) - M0.mean(0)
        d_pior = M.min(0) - M0.min(0)
        princ.append({"receita": rc, "descrição": DESCR[rc], "média AUC (3 fontes)": float(pt.mean()),
                      "média IC2.5": ic(M.mean(0))[0], "média IC97.5": ic(M.mean(0))[1],
                      "pior fonte": pt.idxmin(), "AUC pior fonte": float(pt.min()),
                      "pior IC2.5": ic(M.min(0))[0], "pior IC97.5": ic(M.min(0))[1],
                      **{f"AUC {s}": float(pt[s]) for s in PRINCIPAIS},
                      "dif. média vs T0": float(pt.mean() - tab[(tab.receita == "T0") & tab.recorte.isin(PRINCIPAIS)].AUC.mean()),
                      "dif. média vs T0 IC2.5": ic(d_media)[0], "dif. média vs T0 IC97.5": ic(d_media)[1],
                      "dif. pior vs T0 IC2.5": ic(d_pior)[0], "dif. pior vs T0 IC97.5": ic(d_pior)[1],
                      "vencedora pela validação": rc == esc["receita_vencedora_pela_validacao"]})
    tp = pd.DataFrame(princ)
    salvar_csv(tp, R / f"v7_principal{sfx()}.csv")
    # ---- bloco inglês (AJUSTE 4)
    ing = ["radar_en", "radar_en_pareado", "ml_itw_ingles", "in_the_wild", "llamapartialspoof", "asvspoof2019_la",
           "asvspoof2021_df"]
    tb = tab[tab.recorte.isin(ing)].copy()
    tb["secundário"] = tb.recorte.isin(["asvspoof2019_la", "asvspoof2021_df"])
    salvar_csv(tb, R / f"v7_bloco_ingles{sfx()}.csv")
    salvar_csv(tab[tab.papel == "controle"], R / f"v7_controles{sfx()}.csv")
    # ---- RADAR en por faixa de duração
    from sklearn.metrics import roc_auc_score
    fx = []
    faixas = [("< 2 s", 0, 2), ("2 a 4 s", 2, 4), ("4 a 6 s", 4, 6), (">= 6 s", 6, 1e9)]
    for rc in RECEITAS:
        for nome, lo, hi in faixas:
            g = rad_en[(rad_en.duracao_s >= lo) & (rad_en.duracao_s < hi)]
            y = g.label.values
            ok = (y == 0).sum() >= MIN_POR_CLASSE and (y == 1).sum() >= MIN_POR_CLASSE
            fx.append({"receita": rc, "faixa": nome, "n_real": int((y == 0).sum()), "n_falso": int((y == 1).sum()),
                       "AUC": float(roc_auc_score(y, g[f"z_{rc}"])) if ok else np.nan})
    salvar_csv(pd.DataFrame(fx), R / f"v7_radar_en_duracao{sfx()}.csv")
    # ---- diagnóstico: por idioma (RADAR, todos os lados) e por plataforma (ML-ITW teste)
    dg = []
    rad_all = notas[notas.bloco == "radar_eval"]
    for rc in RECEITAS:
        for lang, g in rad_all.groupby("idioma"):
            y = g.label.values
            dg.append({"receita": rc, "tipo": "RADAR idioma", "valor": lang, "lado": g.split.iloc[0],
                       "n_real": int((y == 0).sum()), "n_falso": int((y == 1).sum()),
                       "AUC": float(roc_auc_score(y, g[f"z_{rc}"])),
                       "obs": "dentro da amostra de treino (T2 a T4)" if g.split.iloc[0] == "treino" and rc in ("T2", "T3", "T4") else ""})
        for plat, g in ml.groupby("plataforma"):
            y = g.label.values
            nr, nf = (y == 0).sum(), (y == 1).sum()
            lr_, lf_ = g[g.label == 0].locutor.nunique(), g[g.label == 1].locutor.nunique()
            dg.append({"receita": rc, "tipo": "ML-ITW plataforma (teste mestre)", "valor": plat, "lado": "teste_mestre",
                       "n_real": int(nr), "n_falso": int(nf), "locutores_real": int(lr_), "locutores_falso": int(lf_),
                       "AUC": float(roc_auc_score(y, g[f"z_{rc}"])) if min(nr, nf) >= MIN_POR_CLASSE else np.nan,
                       "obs": "estimativa" if min(lr_, lf_) >= MIN_LOCUTORES else "indício (< 30 locutores por classe)"})
    salvar_csv(pd.DataFrame(dg), R / f"v7_diagnostico{sfx()}.csv")
    # ---- auditoria de atalho (só metadados) nos conjuntos novos
    aud = auditoria_atalho(notas, tab)
    salvar_csv(aud, R / f"v7_atalho{sfx()}.csv")
    salvar_json({"quando": agora(), "n_boot": n_boot, "seed": SEED, "principal": json.loads(tp.to_json(orient="records", force_ascii=False)),
                 "escolha_validacao": esc["receita_vencedora_pela_validacao"],
                 "regras_escolhidas": {k: v["regra"] for k, v in esc["por_receita"].items()},
                 "minutos": round((time.time() - t0) / 60, 1)}, R / f"v7_resumo{sfx()}.json")
    salvar_json({"quando": agora()}, marcador)
    status("avaliar", f"CONCLUÍDO em {(time.time() - t0) / 60:.1f} min")
    return 0


def auditoria_atalho(notas, tab):
    """Classificador só com metadados: treinado no lado de treino de cada fonte nova, medido na validação e
    no teste mestre; e validação cruzada em 5 partes dentro do teste mestre (por locutor no ML-ITW)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    meta = pd.read_parquet(pasta_vetores() / "meta.parquet")[["linha", "dur_original_s", "silencio_inicio_s",
                                                              "silencio_fim_s", "rms_db", "pico_db", "tamanho_bytes"]]
    d = notas.merge(meta, on="linha")
    d["fracao_silencio"] = (d.silencio_inicio_s + d.silencio_fim_s) / d.dur_original_s
    d["tamanho_kb"] = d.tamanho_bytes / 1024
    cols = ["dur_original_s", "fracao_silencio", "rms_db", "pico_db", "tamanho_kb"]
    out = []
    for bl in ("ml_itw", "radar_eval"):
        g = d[d.bloco == bl]
        tr = g[(g.split == "treino") & ((bl != "radar_eval") | g.pareado_treino)]
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        est.fit(tr[cols].values, tr.label.values)
        alvos = {"validação": g[g.split == "validacao"], "teste mestre": g[g.split == "teste_mestre"]}
        if bl == "radar_eval":
            alvos["teste mestre pareado"] = alvos["teste mestre"][alvos["teste mestre"].pareado_teste]
        for nome, h in alvos.items():
            p = est.predict_proba(h[cols].values)[:, 1]
            out.append({"fonte": bl, "como": "treinado no lado de treino", "medido em": nome, "n": int(len(h)),
                        "AUC metadados": float(roc_auc_score(h.label.values, p)),
                        **{f"AUC {rc}": float(roc_auc_score(h.label.values, h[f"z_{rc}"].values)) for rc in RECEITAS}})
        h = alvos["teste mestre"]
        if bl == "ml_itw":
            cv, grp = StratifiedGroupKFold(5, shuffle=True, random_state=SEED), h.locutor.astype(str).values
        else:
            cv, grp = StratifiedKFold(5, shuffle=True, random_state=SEED), None
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), h[cols].values,
                              h.label.values, cv=cv, groups=grp, method="predict_proba")[:, 1]
        out.append({"fonte": bl, "como": "validação cruzada 5 partes dentro do teste mestre", "medido em": "teste mestre",
                    "n": int(len(h)), "AUC metadados": float(roc_auc_score(h.label.values, p)),
                    **{f"AUC {rc}": float(roc_auc_score(h.label.values, h[f"z_{rc}"].values)) for rc in RECEITAS}})
    a = pd.DataFrame(out)
    a["atributos"] = ", ".join(cols)
    a["marca (> 0,7)"] = np.where(a["AUC metadados"] > 0.7, "sinal de atalho", "sem sinal forte de atalho")
    return a


# ======================================================================== main
def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["vetores", "splits", "treinar", "escolher", "avaliar", "figuras"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--forcar", action="store_true", help="refaz 'avaliar' mesmo com marcador (só em erro de código)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    QUICK = a.quick
    if a.log:
        from src.execucao import Tee
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        f = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.__stdout__, f), Tee(sys.__stderr__, f)
    print(f"\n######## {agora()} | {' '.join(sys.argv)} | pid {os.getpid()}", flush=True)
    ti.VIGIA = ti.Vigia(LIMITE_GB)
    ti.VIGIA.start()
    ti.sem_suspensao(True)
    t0 = time.time()
    try:
        if a.etapa == "figuras":
            import v7_figuras
            return v7_figuras.main(QUICK)
        return {"vetores": etapa_vetores, "splits": etapa_splits, "treinar": etapa_treinar,
                "escolher": etapa_escolher, "avaliar": etapa_avaliar}[a.etapa](a)
    except SystemExit as e:
        print(f"STATUS=BLOQUEIO ({e})", flush=True)
        return 4
    except BaseException:
        traceback.print_exc()
        status(a.etapa, "FALHOU (ver log)")
        print("STATUS=FALHOU", flush=True)
        return 1
    finally:
        ti.sem_suspensao(False)
        ti.VIGIA.parar = True
        print(f"PICO_MEMORIA_GB={ti.VIGIA.pico:.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
