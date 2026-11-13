#!/usr/bin/env python3
"""Avaliação ÚNICA do modelo v4 (F0) congelado nos bancos de teste da internet (RADAR eval e ML-ITW).

Regra: docs/PROTOCOLO_TESTE_INTERNET.md (congelado antes de qualquer avaliação). Nada aqui treina,
reajusta ou escolhe modelo, camada, C ou corte: o F0 é lido do disco (coeficientes da v4) e validado
contra as notas gravadas da v4 ANTES de ser usado.

Etapas (uma por processo; retomável):
    python scripts/teste_internet.py validar                 # F0 do disco reproduz as notas da v4? (senão, BLOQUEIO)
    python scripts/teste_internet.py extrair [--quick] [--vazao N] [--interromper-apos K] [--cache-sufixo S]
    python scripts/teste_internet.py notas   [--quick]       # aplica o F0 -> data/processed/teste_internet_notas.parquet
    python scripts/teste_internet.py metricas [--quick]      # métricas por fonte e recorte, IC95, sensibilidade, atalho
    (figuras: scripts/teste_internet_figuras.py)

Extração: o pipeline da v4 sem mudança (src/preproc.py, src/ssl.py, src/extracao_v4.py, src/cache_v4.py):
mono, 16 kHz, corte de silêncio 35 dB (desfeito se sobrar < 0,5 s), até 8 s do meio, sem completar, fp16,
os hiperparâmetros lidos da célula "config" do notebook.ipynb (a da v4). Shards de 2.000 clipes com
marcador, 4 workers não persistentes, OPENBLAS_NUM_THREADS=1, vigia de memória (encerra acima de 10 GB de
commit privado do processo + workers) e status em logs/avaliacao_status.txt a cada shard.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("results")
LOGS = Path("logs")
RAIZ_AUDIO = ROOT / "data/raw/teste_internet"
MANIFEST = Path("data/processed/manifest_teste_internet.parquet")
STATUS = LOGS / "avaliacao_status.txt"
BLOQUEIO = LOGS / "avaliacao_BLOQUEIO.txt"
LIMITE_GB = 10.0
NUM_WORKERS = 4
ORCAMENTO_H = 3.3   # era 3,0; +0,3 h autorizado em 13/11 (o contador correu durante a espera do Windows)
FONTES = ("radar_eval", "ml_itw")

# ---------------- F0 congelado (protocolo, seção 1) ----------------
CAMADAS = (5, 6, 7)
C_F0 = 0.1
CORTE_A, CORTE_B = -0.0395, 8.2184          # logit; fixados na validação da v4 (lidos e conferidos no npz)

# ---------------- avaliação (protocolo + pedido) ----------------
N_BOOT = 2000                                # protocolo: 1.000; pedido: 2.000 (desvio registrado)
SEED = 42
MIN_POR_CLASSE = 20                          # "poucos áudios": abaixo disso por classe, só contagem (sem AUC/EER)
MIN_LOCUTORES = 30                           # por classe; abaixo disso, "indício, não estimativa"
SOBREPOSTOS = ("ScarlettJohansson", "Trump", "obama")   # possíveis sobreposições com o In-the-Wild (por nome)
N_REP_PAREADO = 20
LARGURA_FAIXA_S = 0.5


# ======================================================================== utilidades
def agora():
    return datetime.now().isoformat(timespec="seconds")


def ler_json(p, padrao=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else padrao


def escrever(p, txt):
    from src.arquivos import gravar_atomico
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    gravar_atomico(Path(p), lambda t: Path(t).write_text(txt, encoding="utf-8"))


class Vigia(threading.Thread):
    """Commit privado deste processo + workers a cada 0,5 s; guarda o pico e encerra acima do limite."""

    def __init__(self, limite_gb=LIMITE_GB, ao_estourar=None):
        super().__init__(daemon=True)
        self.limite, self.pico, self.pico_sis, self.parar = limite_gb, 0.0, 0.0, False
        self.ao_estourar = ao_estourar
        self.detalhe = {}

    def run(self):
        from src.memoria import commit_sistema, processos
        while not self.parar:
            try:
                ps = processos()
                gb = sum(c for *_, c in ps) / 1024
                if gb > self.pico:
                    self.detalhe = {"principal_gb": round(sum(c for p_, *_, c in ps if p_ == "principal") / 1024, 2),
                                    "workers_gb": [round(c / 1024, 2) for p_, *_, c in ps if p_ == "worker"]}
                self.pico = max(self.pico, gb)
                self.pico_sis = max(self.pico_sis, commit_sistema()[0] / 2 ** 30)
                if gb > self.limite:
                    msg = f"memória {gb:.2f} GB > {self.limite} GB (processo + workers; {self.detalhe}): encerrando"
                    print("\n!!! " + msg, flush=True)
                    if self.ao_estourar:
                        try:
                            self.ao_estourar(msg)
                        except Exception:   # noqa: BLE001
                            pass
                    os._exit(3)
            except Exception:   # noqa: BLE001
                pass
            time.sleep(0.5)


VIGIA = None


def sem_suspensao(ligar=True):
    """Impede a suspensão do Windows enquanto o processo roda (como scripts/v6.py e v5_teste.py)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if ligar else 0x80000000)
    except Exception:   # noqa: BLE001
        pass


def manifest():
    """Manifesto de teste com caminho absoluto (cópia em memória; o arquivo não é alterado)."""
    df = pd.read_parquet(MANIFEST)
    assert len(df) == 127_254, len(df)
    assert pd.RangeIndex(len(df)).equals(df.index)
    macos = df.path.str.contains(r"(?:^|[\\/])\._", regex=True)
    assert not macos.any(), f"{int(macos.sum())} arquivos ._* no manifesto"
    df = df.copy()
    df["path_rel"] = df.path
    df["path"] = [str(RAIZ_AUDIO / p) for p in df.path_rel]
    return df


def config_v4():
    import contextlib
    import io

    from src.execucao import celula_config, executar_config
    src = celula_config("notebook.ipynb")
    with contextlib.redirect_stdout(io.StringIO()):
        ns = executar_config(src, False, "notebook.ipynb")
    return ns


def carregar_f0():
    """Pesos do F0 (results/v4final_f0_coeficientes.npz) + padronização do treino da v4
    (results/v4_parcial/padronizacao.npz: o npz do F0 não traz média e desvio)."""
    z = np.load(R / "v4final_f0_coeficientes.npz")
    chaves = sorted(z.files)
    p = np.load(R / "v4_parcial" / "padronizacao.npz")
    f0 = {"coef": z["coef"].astype(np.float64), "intercepto": float(z["intercepto"]),
          "camadas": [int(c) for c in z["camadas"]], "C": float(z["C"]),
          "limiar_A": float(z["limiar_A"]), "limiar_B": float(z["limiar_B"]),
          "mu": p["mu"], "sd": p["sd"], "n_treino_padronizacao": int(p["n"]), "chaves_npz": chaves,
          "chaves_padronizacao": sorted(p.files)}
    assert f0["camadas"] == list(CAMADAS) and abs(f0["C"] - C_F0) < 1e-12, (f0["camadas"], f0["C"])
    assert f0["coef"].shape == (3072,), f0["coef"].shape
    assert abs(f0["limiar_A"] - CORTE_A) < 1e-4 and abs(f0["limiar_B"] - CORTE_B) < 1e-4, (f0["limiar_A"], f0["limiar_B"])
    return f0


def aplicar_f0(f0, conj, ok=None):
    """Logit do F0 para os clipes de um `Conjunto` (src/cache_v4.py), na ordem do meta."""
    ok = np.ones(len(conj), bool) if ok is None else ok
    X = np.hstack([(conj.camada(l)[ok] - f0["mu"][l]) / f0["sd"][l] for l in f0["camadas"]])
    return X @ f0["coef"] + f0["intercepto"]


# ======================================================================== validar
def etapa_validar(a):
    from sklearn.metrics import roc_auc_score

    from src.cache_v4 import Conjunto
    from src.metrics import eer_e_limiar
    f0 = carregar_f0()
    print(f"F0: chaves do npz {f0['chaves_npz']}; padronização {f0['chaves_padronizacao']} "
          f"(n treino {f0['n_treino_padronizacao']}); intercepto {f0['intercepto']:.6f}", flush=True)
    notas = pd.read_parquet(R / "v4_teste_notas.parquet")
    conjs = {"val": ("data/processed/v4_cache", "val"), "dev_shift": ("data/processed/v4_cache", "dev_shift")}
    for b in ("in_the_wild", "llamapartialspoof", "brspeech_df", "coraa", "asvspoof2019_la", "asvspoof2021_df"):
        conjs[b] = ("data/processed/v4_cache_teste", f"teste_{b}")
    linhas, Z, Y = [], {}, {}
    for k, (cache, nome) in conjs.items():
        c = Conjunto(cache, nome)
        ok = c.meta.erro_leitura.fillna(0).values == 0
        m = c.meta[ok].reset_index(drop=True)
        z = aplicar_f0(f0, c, ok)
        g = notas[notas.banco == k].set_index("idx_manifest").loc[m.idx_manifest.values]
        y = m.label.values.astype(int)
        duas = len(np.unique(y)) == 2
        Z[k], Y[k] = z, y
        linhas.append({"conjunto": k, "n": int(len(m)), "dif_max_logit": float(np.abs(z - g.z_S2.values).max()),
                       "AUC_disco": round(float(roc_auc_score(y, z)), 3) if duas else None,
                       "AUC_gravada_v4": round(float(roc_auc_score(y, g.z_S2.values)), 3) if duas else None})
        print(linhas[-1], flush=True)
    lim = {"A": eer_e_limiar(Y["val"], Z["val"])[1], "B": eer_e_limiar(Y["dev_shift"], Z["dev_shift"])[1]}
    ok_dif = all(r["dif_max_logit"] <= 1e-4 for r in linhas)
    ok_auc = all(r["AUC_disco"] == r["AUC_gravada_v4"] for r in linhas)
    ok_lim = abs(lim["A"] - CORTE_A) < 1e-3 and abs(lim["B"] - CORTE_B) < 1e-3
    res = {"quando": agora(), "fonte_dos_pesos": "results/v4final_f0_coeficientes.npz (coef, intercepto, camadas, C, "
           "limiares) + results/v4_parcial/padronizacao.npz (mu, sd do treino da v4)",
           "chaves_npz": f0["chaves_npz"], "faltava_padronizacao_no_npz": "mu" not in f0["chaves_npz"],
           "conjuntos": linhas, "limiares_recalculados": lim, "cortes_do_protocolo": {"A": CORTE_A, "B": CORTE_B},
           "criterio": "AUC com 3 casas igual e diferença máxima de nota <= 1e-4 em todos os conjuntos",
           "aprovado": bool(ok_dif and ok_auc and ok_lim)}
    escrever(R / "teste_internet_validacao_f0.json", json.dumps(res, ensure_ascii=False, indent=1))
    if not res["aprovado"]:
        escrever(BLOQUEIO, f"{agora()}\nO F0 lido do disco NÃO reproduziu as notas da v4.\n"
                 + json.dumps(res, ensure_ascii=False, indent=1))
        print("F0 NÃO reproduz a v4: BLOQUEIO gravado\nSTATUS=BLOQUEIO", flush=True)
        return 4
    print(f"F0 validado: dif. máx. {max(r['dif_max_logit'] for r in linhas):.2e}; limiares {lim}\nSTATUS=OK", flush=True)
    return 0


# ======================================================================== extração
def plano_de(df, modo, n_vazao=500):
    """(plano, tamanho do shard). Conjunto = fonte; shards na ordem do manifesto."""
    if modo == "quick":           # 200 clipes: 50 de cada fonte e classe
        partes = [df[(df.fonte == f) & (df.label == l)].sample(50, random_state=SEED) for f in FONTES for l in (0, 1)]
        sel, tam = pd.concat(partes).sort_index(), 50
    elif modo == "vazao":         # 500 clipes ao acaso de todo o banco (vazão realista)
        sel, tam = df.sample(n_vazao, random_state=SEED).sort_index(), 10 ** 9
    else:
        sel, tam = df, 2000
    linhas = []
    for f in FONTES:
        idx = sel[sel.fonte == f].index.values
        for s, i in enumerate(range(0, len(idx), tam)):
            linhas.append(pd.DataFrame({"idx_manifest": idx[i:i + tam], "conjunto": f, "shard": s,
                                        "cortar_silencio": True}))
    plano = pd.concat(linhas, ignore_index=True)
    plano["ordem"] = np.arange(len(plano))
    return plano


def caminho_cache(a):
    base = {"full": "teste_internet_cache", "quick": "teste_internet_cache_quick",
            "vazao": "teste_internet_cache_vazao"}[a.modo]
    return Path("data/processed") / (base + (a.cache_sufixo or ""))


def progresso(cache, plano):
    from src.cache_v4 import shard_pronto
    blocos = plano.drop_duplicates(["conjunto", "shard"])[["conjunto", "shard"]]
    prontos = [(c, s) for c, s in blocos.itertuples(index=False) if shard_pronto(cache, c, s)]
    n_por = plano.groupby(["conjunto", "shard"]).size()
    clipes = int(sum(n_por.loc[(c, s)] for c, s in prontos))
    erros = []
    for c, s in prontos:
        m = pd.read_parquet(Path(cache) / c / f"shard_{int(s):05d}" / "meta.parquet", columns=["idx_manifest", "erro_leitura"])
        erros += m.idx_manifest[m.erro_leitura.fillna(0) > 0].tolist()
    return len(prontos), len(blocos), clipes, len(plano), erros


def status_extracao(cache, plano, orc, t_proc, clipes_ini, extra=""):
    feitos, total, clipes, n_tot, erros = progresso(cache, plano)
    seg = time.time() - t_proc
    vaz = (clipes - clipes_ini) / seg if seg > 0 and clipes > clipes_ini else float("nan")
    resta_min = (n_tot - clipes) / vaz / 60 if vaz == vaz and vaz > 0 else float("nan")
    txt = (f"AVALIACAO TESTE INTERNET (F0) | atualizado {agora()}\n"
           f"etapa: extração | cache: {cache}\n"
           f"shards prontos: {feitos} de {total} | clipes prontos: {clipes} de {n_tot}\n"
           f"vazão neste processo: {vaz:.1f} clipes/s | estimativa para terminar: {resta_min:.1f} min\n"
           f"tempo de extração acumulado (orçamento {ORCAMENTO_H} h): {orc.gasto() / 3600:.2f} h\n"
           f"pico de memória (processo + workers): {VIGIA.pico if VIGIA else float('nan'):.2f} GB (limite {LIMITE_GB}) | "
           f"pico do sistema {VIGIA.pico_sis if VIGIA else float('nan'):.1f} GB\n"
           f"clipes ilegíveis até agora: {len(erros)} {erros[:20]}\n"
           f"para retomar: python scripts/teste_internet.py extrair (pula os shards com concluido.json)\n{extra}")
    escrever(STATUS if a_modo_global == "full" else LOGS / f"avaliacao_status_{a_modo_global}.txt", txt)
    return feitos, total


a_modo_global = "full"


def aquecer(model, hp):
    """Uma passada com RUÍDO SINTÉTICO no MAIOR balde antes da extração (execução, não método).

    Medido nesta máquina (sonda com áudio sintético): se o 1º lote é de um balde pequeno, o alocador da GPU
    cresce aos pedaços e, no Windows, a memória reservada do processo principal vai a 8,6-9,2 GB; se o 1º
    lote é o maior balde, ele reserva os blocos grandes uma vez e o principal fica em ~6,9 GB. O modelo
    está congelado (eval, inference_mode): a passada não muda nada nem é gravada.
    """
    import math

    import torch

    from src.preproc import SR
    from src.ssl import representar
    s = int(math.ceil(hp["MAX_S"]))
    linhas = int(min(hp["MAX_CLIPES_LOTE"], hp["MAX_AMOSTRAS_LOTE"] // (s * SR)))
    g = torch.Generator().manual_seed(0)
    clipes = [torch.randn(int(s * SR * 0.9), generator=g) for _ in range(linhas)]
    representar(model, clipes, "cuda", hp["DTYPE"], forma=(linhas, s * SR))
    torch.cuda.synchronize()
    print(f"aquecimento: 1 lote sintético {linhas} x {s} s; memória do processo "
          f"{VIGIA.pico if VIGIA else float('nan'):.2f} GB (pico até aqui)", flush=True)


def etapa_extrair(a):
    global a_modo_global
    import torch

    from src.cache_v4 import Orcamento, limpar_temporarios
    from src.extracao_v4 import extrair
    from src.ssl import carregar_ssl
    a_modo_global = a.modo
    ns = config_v4()
    hp, cfg = dict(ns["HP"]), dict(ns["CFG_EXTRACAO"])
    df = manifest()
    plano = plano_de(df, a.modo, a.vazao or 500)
    cache = caminho_cache(a)
    cache.mkdir(parents=True, exist_ok=True)
    cfg_plano = {"extracao": cfg, "modo": a.modo, "n": int(len(plano)), "manifest": str(MANIFEST),
                 "idx_sha": int(pd.util.hash_pandas_object(plano[["idx_manifest", "shard"]], index=False).sum() % (2 ** 61)),
                 "hp": {k: hp[k] for k in ("MODELO_SSL", "DTYPE", "TOP_DB", "MIN_S", "MAX_S", "MAX_AMOSTRAS_LOTE",
                                            "MAX_CLIPES_LOTE", "BALDE_S")}}
    salvo = ler_json(cache / "plano.json")
    if salvo is not None and salvo != json.loads(json.dumps(cfg_plano, default=str)):
        raise RuntimeError(f"{cache}/plano.json é de outra configuração; nada foi apagado.\n{salvo}\n{cfg_plano}")
    if salvo is None:
        plano.to_parquet(cache / "plano.parquet", index=False)
        escrever(cache / "plano.json", json.dumps(cfg_plano, ensure_ascii=False, indent=1, default=str))
    n_tmp = limpar_temporarios(cache)
    print(f"plano: {len(plano)} clipes, {plano.groupby(['conjunto', 'shard']).ngroups} shards; "
          f"temporários descartados: {n_tmp}", flush=True)
    print(f"pipeline (célula config da v4): {cfg} | lote {hp['MAX_AMOSTRAS_LOTE']} amostras / {hp['MAX_CLIPES_LOTE']} "
          f"clipes | baldes de {hp['BALDE_S']} s", flush=True)
    orc = Orcamento(cache / "estado.json", ORCAMENTO_H)
    t_proc = time.time()
    feitos0 = progresso(cache, plano)[2]
    n_shards = [0]

    def log(s):
        print(s, flush=True)
        if s.strip().startswith("shard "):
            n_shards[0] += 1
            status_extracao(cache, plano, orc, t_proc, feitos0)
            if a.interromper_apos and n_shards[0] >= a.interromper_apos:
                print(f"!!! interrupção simulada (queda à força) depois de {n_shards[0]} shard(s)", flush=True)
                os._exit(9)
    status_extracao(cache, plano, orc, t_proc, feitos0, "carregando o modelo...")
    t_modelo = time.time()
    model = carregar_ssl(hp["MODELO_SSL"], "cuda", hp["DTYPE"])
    aquecer(model, hp)
    t_modelo = time.time() - t_modelo
    t_ext = time.time()
    try:
        r = extrair(df, plano, model, hp, cache, orc, cfg, log=log, num_workers=a.num_workers)
    finally:
        orc.fechar()
    t_ext = time.time() - t_ext
    feitos, total = status_extracao(cache, plano, orc, t_proc, feitos0)
    r.update({"segundos_carregar_modelo": round(t_modelo, 1), "segundos_extracao_neste_processo": round(t_ext, 1),
              "pico_memoria_gb": round(VIGIA.pico, 2), "pico_detalhe": VIGIA.detalhe, "pico_sistema_gb": round(VIGIA.pico_sis, 1),
              "gpu_pico_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "num_workers": a.num_workers})
    print("resultado:", json.dumps(r, ensure_ascii=False), flush=True)
    if a.modo == "vazao":
        escrever(R / "teste_internet_vazao.json", json.dumps({"quando": agora(), "n": int(len(plano)), **r,
                 "clipes_por_s_com_carga": round(len(plano) / (t_modelo + t_ext), 2)}, ensure_ascii=False, indent=1))
    if feitos < total:
        print(f"extração incompleta ({r.get('parou_por')}); rode de novo.\nSTATUS=INCOMPLETO", flush=True)
        return 2
    _, _, clipes, _, erros = progresso(cache, plano)
    escrever(cache / "extracao_concluida.json", json.dumps({"quando": agora(), "clipes": clipes, "ilegiveis": erros,
             "horas_extracao_acumuladas": round(orc.gasto() / 3600, 3), "ultima_passada": r}, ensure_ascii=False, indent=1))
    status_extracao(cache, plano, orc, t_proc, feitos0, "EXTRAÇÃO CONCLUÍDA\n")
    print("extração CONCLUÍDA\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ======================================================================== notas
def arq_notas(quick):
    return Path("data/processed") / ("teste_internet_notas_quick.parquet" if quick else "teste_internet_notas.parquet")


def etapa_notas(a):
    from src.cache_v4 import Conjunto
    a.modo = "quick" if a.quick else "full"
    cache = caminho_cache(a)
    if not (cache / "extracao_concluida.json").exists():
        raise RuntimeError(f"extração não concluída em {cache}")
    assert ler_json(R / "teste_internet_validacao_f0.json", {}).get("aprovado"), "F0 não validado (rode 'validar')"
    f0 = carregar_f0()
    df = manifest()
    partes = []
    for f in FONTES:
        c = Conjunto(cache, f)
        m = c.meta.reset_index(drop=True)
        ok = m.erro_leitura.fillna(0).values == 0
        z = np.full(len(m), np.nan)
        z[ok] = aplicar_f0(f0, c, ok)
        extra = df.loc[m.idx_manifest.values, ["path_rel", "fonte", "language", "plataforma", "canais", "codec",
                                               "extensao", "tamanho_bytes"]].reset_index(drop=True)
        extra = extra.rename(columns={"path_rel": "path_relativo", "sample_rate": "sr_manifesto"})
        p = pd.concat([m.drop(columns=["path"]), extra], axis=1)
        p["z_F0"] = z
        p["ilegivel"] = ~ok
        partes.append(p)
    notas = pd.concat(partes, ignore_index=True)
    assert notas.idx_manifest.is_unique
    from src.arquivos import salvar_parquet
    salvar_parquet(notas, arq_notas(a.quick))
    n_il = int(notas.ilegivel.sum())
    print(f"notas: {len(notas)} clipes ({n_il} ilegíveis) -> {arq_notas(a.quick)}; NaN entre legíveis: "
          f"{int(np.isnan(notas.z_F0[~notas.ilegivel]).sum())}\nSTATUS=CONCLUIDO", flush=True)
    return 0


# ======================================================================== métricas
def boot_pesos(y, grupo, B, seed, chunk=250):
    """Gera blocos (b, n) de pesos: em cada classe, sorteia com reposição as unidades (`grupo`: locutor ou
    clipe) daquela classe; peso do clipe = nº de vezes que a sua unidade saiu. Mesma sequência aleatória
    qualquer que seja o tamanho do bloco."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    classes = []
    for c in (0, 1):
        idx = np.nonzero(y == c)[0]
        if len(idx):
            cod, uniq = pd.factorize(pd.Series(np.asarray(grupo)[idx]).astype(str))
            classes.append((idx, cod, len(uniq)))
    feitos = 0
    while feitos < B:
        b = min(chunk, B - feitos)
        W = np.zeros((b, len(y)), np.float32)
        for i in range(b):
            for idx, cod, k in classes:
                W[i, idx] = np.bincount(rng.integers(0, k, k), minlength=k)[cod]
        feitos += b
        yield W


def metricas_ponderadas(y, z, W, limiares):
    """Para cada linha de pesos: AUC, EER e, em cada limiar, TPR, FPR e acurácia balanceada."""
    from src.v6_metricas import _auc_ponderada
    y = np.asarray(y).astype(int)
    out = {"AUC": _auc_ponderada(y, z, W)}
    ordem = np.argsort(z, kind="mergesort")
    zs, ys = z[ordem], y[ordem]
    inicio = np.concatenate([[0], np.nonzero(np.diff(zs) != 0)[0] + 1])
    Ws = W[:, ordem].astype(np.float64)
    Gp = np.add.reduceat(Ws * (ys == 1), inicio, axis=1)
    Gn = np.add.reduceat(Ws * (ys == 0), inicio, axis=1)
    tp = np.cumsum(Gp[:, ::-1], axis=1)[:, ::-1]         # fakes com nota >= limiar do grupo
    fp = np.cumsum(Gn[:, ::-1], axis=1)[:, ::-1]
    tpr, fpr = tp / tp[:, :1], fp / fp[:, :1]
    fnr = 1 - tpr
    j = np.argmin(np.abs(fnr - fpr), axis=1)
    r = np.arange(len(j))
    out["EER"] = (fpr[r, j] + fnr[r, j]) / 2
    Wp, Wn = (W * (y == 1)).astype(np.float64), (W * (y == 0)).astype(np.float64)
    for L, t in limiares.items():
        pos = (z >= t).astype(np.float64)
        tpr_t = Wp @ pos / Wp.sum(1)
        fpr_t = Wn @ pos / Wn.sum(1)
        out[f"TPR {L}"], out[f"FPR {L}"] = tpr_t, fpr_t
        out[f"acurácia balanceada {L}"] = 0.5 * (tpr_t + 1 - fpr_t)
    return out


def ic(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (np.nan, np.nan)


def avaliar_recorte(m, fonte, recorte, valor, n_boot, ic_unidade):
    """Uma linha de métricas para o recorte `m` (DataFrame com label, z_F0, speaker_id)."""
    from sklearn.metrics import roc_auc_score

    from src.metrics import acuracia_trivial, eer_e_limiar
    leg = m[~m.ilegivel]
    y, z = leg.label.values.astype(int), leg.z_F0.values.astype(np.float64)
    nr, nf = int((y == 0).sum()), int((y == 1).sum())
    linha = {"fonte": fonte, "recorte": recorte, "valor": valor, "n_real": nr, "n_falso": nf,
             "ilegiveis_excluidos": int(m.ilegivel.sum()), "chute_classe_comum": acuracia_trivial(y) if len(y) else np.nan}
    if fonte == "ml_itw":
        lr = leg[leg.label == 0].speaker_id.nunique()
        lf = leg[leg.label == 1].speaker_id.nunique()
        linha.update({"locutores_real": int(lr), "locutores_falso": int(lf)})
        marca = "estimativa" if min(lr, lf) >= MIN_LOCUTORES else "indício, não estimativa (< 30 locutores por classe)"
        linha["ic_unidade"] = "locutor (estratificado por classe)"
    else:
        linha.update({"locutores_real": np.nan, "locutores_falso": np.nan})
        marca = "IC otimista (sem locutor no arquivo)"
        linha["ic_unidade"] = "clipe (estratificado por classe)"
    if ic_unidade == "clipe":
        linha["ic_unidade"] = "clipe (estratificado por classe)"
    calcula = nr >= MIN_POR_CLASSE and nf >= MIN_POR_CLASSE
    if not calcula:
        marca = f"só contagem (menos de {MIN_POR_CLASSE} clipes em uma classe)"
    linha["marca"] = marca
    lim = {"A": CORTE_A, "B": CORTE_B}
    if calcula:
        eer, t_eer = eer_e_limiar(y, z)
        lim["EER da fonte"] = t_eer
        linha.update({"AUC": float(roc_auc_score(y, z)), "EER": eer, "corte_EER_da_fonte": t_eer})
    for L, t in lim.items():
        pred = z >= t
        tpr = float(pred[y == 1].mean()) if nf else np.nan
        fpr = float(pred[y == 0].mean()) if nr else np.nan
        linha[f"acurácia {L}"] = float((pred == y).mean()) if len(y) else np.nan
        linha[f"acurácia balanceada {L}"] = 0.5 * (tpr + 1 - fpr) if (nr and nf) else np.nan
        linha[f"TPR {L}"], linha[f"FPR {L}"] = tpr, fpr
    if calcula and n_boot:
        grupo = leg.speaker_id.astype(str).values if ic_unidade == "locutor" else np.arange(len(leg))
        acc = {}
        for W in boot_pesos(y, grupo, n_boot, SEED):
            for k, v in metricas_ponderadas(y, z, W, {"A": CORTE_A, "B": CORTE_B}).items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            lo, hi = ic(np.concatenate(v))
            linha[f"{k} IC2.5"], linha[f"{k} IC97.5"] = lo, hi
        linha["n_boot"] = n_boot
    return linha


def faixa_dur(d):
    return np.where(d < 2, "< 2 s", np.where(d < 4, "2 a 4 s", ">= 4 s"))


def etapa_metricas(a):
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import pearsonr, spearmanr

    from src.arquivos import salvar_csv
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    n_boot = 50 if a.quick else N_BOOT
    out_dir = (R / "teste_internet_quick") if a.quick else R
    out_dir.mkdir(exist_ok=True)

    def arq(nome, ext="csv"):
        return out_dir / f"teste_internet_{nome}.{ext}"
    notas = pd.read_parquet(arq_notas(a.quick))
    notas["speaker_id"] = notas.speaker_id.astype("string")
    unidade = {"radar_eval": "clipe", "ml_itw": "locutor"}

    def log(s):
        print(f"[{(time.time() - t0) / 60:5.1f} min | {VIGIA.pico:4.1f} GB] {s}", flush=True)

    # ---------------- ETAPA 2: por fonte, idioma e plataforma
    por_fonte, por_idioma, por_plat = [], [], []
    for f in FONTES:
        m = notas[notas.fonte == f]
        por_fonte.append(avaliar_recorte(m, f, "inteiro", "todos", n_boot, unidade[f]))
        log(f"{f} inteiro")
        for lang, g in m.groupby("language", sort=True):
            por_idioma.append(avaliar_recorte(g, f, "idioma", lang, n_boot, unidade[f]))
        log(f"{f} por idioma")
        if f == "ml_itw":
            for plat, g in m.groupby("plataforma", sort=True):
                por_plat.append(avaliar_recorte(g, f, "plataforma", plat, n_boot, unidade[f]))
            log(f"{f} por plataforma")
    tf, ti, tp = pd.DataFrame(por_fonte), pd.DataFrame(por_idioma), pd.DataFrame(por_plat)
    salvar_csv(tf, arq("por_fonte"))
    salvar_csv(ti, arq("por_idioma"))
    salvar_csv(tp, arq("por_plataforma"))

    # ---------------- ETAPA 3a/b: sensibilidade
    sens = []
    leg = notas[~notas.ilegivel]
    # (a) RADAR com durações pareadas (faixas de 0,5 s, mesmo nº por classe e faixa, 20 repetições)
    rad = leg[leg.fonte == "radar_eval"].reset_index(drop=True)
    faixa = np.floor(rad.duration_s.values / LARGURA_FAIXA_S).astype(int)
    rng = np.random.default_rng(SEED)
    aucs, bal_a, bal_b, n_par, med = [], [], [], [], []
    grupos = {k: (np.nonzero((faixa == k) & (rad.label.values == 0))[0], np.nonzero((faixa == k) & (rad.label.values == 1))[0])
              for k in np.unique(faixa)}
    for _ in range(N_REP_PAREADO):
        sel = []
        for k, (ir, i_f) in grupos.items():
            n = min(len(ir), len(i_f))
            if n:
                sel += [rng.choice(ir, n, replace=False), rng.choice(i_f, n, replace=False)]
        sel = np.concatenate(sel)
        y, z = rad.label.values[sel].astype(int), rad.z_F0.values[sel]
        aucs.append(roc_auc_score(y, z))
        for L, t, lst in (("A", CORTE_A, bal_a), ("B", CORTE_B, bal_b)):
            p = z >= t
            lst.append(0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean()))
        n_par.append(len(sel))
        d = rad.duration_s.values[sel]
        med.append((float(np.median(d[y == 0])), float(np.median(d[y == 1]))))
    sens.append({"análise": "a) RADAR com durações pareadas", "recorte": "faixas de 0,5 s, mesmo nº por classe",
                 "n_real": n_par[0] // 2, "n_falso": n_par[0] // 2, "AUC": float(np.mean(aucs)),
                 "AUC min": float(np.min(aucs)), "AUC max": float(np.max(aucs)),
                 "acurácia balanceada A": float(np.mean(bal_a)), "acurácia balanceada A min": float(np.min(bal_a)),
                 "acurácia balanceada A max": float(np.max(bal_a)), "acurácia balanceada B": float(np.mean(bal_b)),
                 "acurácia balanceada B min": float(np.min(bal_b)), "acurácia balanceada B max": float(np.max(bal_b)),
                 "repetições": N_REP_PAREADO, "mediana duração real/falso (s)": f"{med[0][0]:.2f} / {med[0][1]:.2f}",
                 "nota": "intervalo = mínimo e máximo das 20 repetições (variação do sorteio, não IC)"})
    log("sensibilidade (a) pareado")
    for f in FONTES:
        m = notas[notas.fonte == f].copy()
        m["faixa"] = faixa_dur(m.duration_s.values)
        for fx in ("< 2 s", "2 a 4 s", ">= 4 s"):
            g = m[m.faixa == fx]
            if len(g):
                sens.append({"análise": "a) AUC por faixa de duração original", **avaliar_recorte(g, f, "faixa de duração", fx, n_boot, unidade[f])})
    log("sensibilidade (a) faixas")
    # (b) ML-ITW: só locutores com as duas classes; sem os possíveis sobrepostos com o In-the-Wild
    ml = notas[notas.fonte == "ml_itw"]
    cls = ml[~ml.ilegivel].groupby("speaker_id").label.nunique()
    ambos = set(cls[cls == 2].index)
    nome_loc = ml.speaker_id.astype(str).str.split(":", n=1).str[-1]      # ids vêm como "mlitw:<nome>"
    sobre = nome_loc.isin(SOBREPOSTOS).values
    presentes = sorted(set(nome_loc[sobre]))
    assert len(presentes) == len(SOBREPOSTOS), f"locutores sobrepostos achados: {presentes}"
    sub = {"inteiro (referência)": ml,
           f"só locutores com as duas classes ({len(ambos)} de {ml.speaker_id.nunique()})": ml[ml.speaker_id.isin(ambos)],
           f"sem os {len(presentes)} possíveis sobrepostos com o In-the-Wild": ml[~sobre],
           "só com as duas classes e sem os sobrepostos": ml[ml.speaker_id.isin(ambos).values & ~sobre]}
    for nome, g in sub.items():
        sens.append({"análise": "b) ML-ITW por locutor", **avaliar_recorte(g, "ml_itw", "locutores", nome, n_boot, "locutor")})
    log("sensibilidade (b)")
    ts = pd.DataFrame(sens)
    salvar_csv(ts, arq("sensibilidade"))

    # ---------------- ETAPA 3c: auditoria de atalho (só metadados)
    leg = leg.copy()
    leg["fracao_silencio"] = (leg.silencio_inicio_s + leg.silencio_fim_s) / leg.dur_original_s
    leg["taxa_original"] = leg.sample_rate.astype(float)
    leg["canais_n"] = leg.canais.astype(float)
    leg["tamanho_kb"] = leg.tamanho_bytes.astype(float) / 1024
    conjuntos_attr = {
        "pedido: duração, taxa, canais, fração de silêncio, RMS, pico":
            ["dur_original_s", "taxa_original", "canais_n", "fracao_silencio", "rms_db", "pico_db"],
        "protocolo: + tamanho do arquivo": ["dur_original_s", "taxa_original", "canais_n", "fracao_silencio", "rms_db",
                                            "pico_db", "tamanho_kb"],
        "só duração": ["dur_original_s"],
    }
    aud = []
    for f in FONTES:
        g = leg[leg.fonte == f].reset_index(drop=True)
        y = g.label.values.astype(int)
        auc_f0 = float(roc_auc_score(y, g.z_F0.values))
        if f == "ml_itw":
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            grupos_cv = g.speaker_id.astype(str).values
            como = "5 partes agrupadas por locutor (StratifiedGroupKFold)"
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            grupos_cv = None
            como = "5 partes por clipe (StratifiedKFold)"
        for nome, cols in conjuntos_attr.items():
            Xm = g[cols].values.astype(np.float64)
            constantes = [c for c in cols if g[c].nunique() <= 1]
            for modelo, est in (("regressão logística", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                                ("árvore rasa (profundidade 3)", DecisionTreeClassifier(max_depth=3, random_state=SEED))):
                p = cross_val_predict(est, Xm, y, cv=cv, groups=grupos_cv, method="predict_proba")[:, 1]
                auc_m = float(roc_auc_score(y, p))
                aud.append({"fonte": f, "tipo": "classificador só com metadados", "atributos": nome, "modelo": modelo,
                            "validação cruzada": como, "colunas constantes (sem informação)": ", ".join(constantes),
                            "AUC metadados (fora da dobra)": auc_m, "AUC F0": auc_f0,
                            "marca": ("confundida pelo canal ou pela duração" if auc_m > 0.7 else "sem sinal forte de atalho")
                            if modelo == "regressão logística" else "(secundário)",
                            "n": int(len(g))})
        for lab, nome_cl in ((0, "real"), (1, "falso")):
            h = g[g.label == lab]
            rp = pearsonr(h.z_F0.values, h.dur_original_s.values)[0]
            rs = spearmanr(h.z_F0.values, h.dur_original_s.values)[0]
            aud.append({"fonte": f, "tipo": f"correlação nota F0 × duração, só {nome_cl}", "atributos": "dur_original_s",
                        "modelo": "Pearson / Spearman", "correlação Pearson": float(rp), "correlação Spearman": float(rs),
                        "AUC F0": auc_f0, "n": int(len(h))})
    log("auditoria de atalho")
    ta = pd.DataFrame(aud)
    salvar_csv(ta, arq("auditoria_atalho"))

    # ---------------- ETAPA 3d: contexto (teste da v4, sem recalcular)
    v4 = pd.read_csv(R / "v4_teste_final.csv")
    v4 = v4[(v4.sistema == "S2") & (v4.conjunto == "teste") & v4.banco.isin(["in_the_wild", "llamapartialspoof", "brspeech_df"])]
    ctx = []
    for b, g in v4.groupby("banco", sort=False):
        d = {"banco": b}
        for _, r in g.iterrows():
            if r["métrica"] in ("AUC", "EER", "acurácia trivial", "n real", "n fake") and r.limiar == "A":
                d[r["métrica"]] = r.valor
                if r["métrica"] == "AUC":
                    d["AUC IC"] = [r["IC 2,5%"], r["IC 97,5%"]]
            elif r["métrica"] in ("acurácia balanceada", "recall fake", "falso alarme real", "acurácia"):
                d[f"{r['métrica']} {r.limiar}"] = r.valor
        d["n real"], d["n fake"] = int(g["n real"].iloc[0]), int(g["n fake"].iloc[0])
        ctx.append(d)

    # ---------------- resumo
    def limpa(df_):
        return json.loads(df_.to_json(orient="records", force_ascii=False))
    resumo = {"quando": agora(), "quick": bool(a.quick), "protocolo": "docs/PROTOCOLO_TESTE_INTERNET.md",
              "modelo": {"nome": "F0 = v4 (XLS-R 300M, camadas 5, 6, 7, média no tempo, padronização, RL C = 0,1)",
                         "cortes_logit": {"A": CORTE_A, "B": CORTE_B}},
              "validacao_f0": ler_json(R / "teste_internet_validacao_f0.json"),
              "extracao": ler_json(Path("data/processed") / ("teste_internet_cache_quick" if a.quick else "teste_internet_cache")
                                   / "extracao_concluida.json"),
              "n_boot": n_boot, "seed": SEED, "min_por_classe_para_AUC": MIN_POR_CLASSE,
              "min_locutores_por_classe": MIN_LOCUTORES,
              "por_fonte": limpa(tf), "sensibilidade": limpa(ts), "auditoria_atalho": limpa(ta),
              "contexto_v4_teste_final_S2": ctx, "minutos": round((time.time() - t0) / 60, 1),
              "pico_memoria_gb": round(VIGIA.pico, 2)}
    escrever(arq("resumo", "json"), json.dumps(resumo, ensure_ascii=False, indent=1, default=str))
    log(f"métricas CONCLUÍDAS -> {out_dir}")
    print("STATUS=CONCLUIDO", flush=True)
    return 0


# ======================================================================== main
def main():
    global VIGIA
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["validar", "extrair", "notas", "metricas"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vazao", type=int, default=None)
    ap.add_argument("--interromper-apos", type=int, default=None)
    ap.add_argument("--cache-sufixo", default="")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    a.modo = "vazao" if a.vazao else ("quick" if a.quick else "full")
    if a.log:
        from src.execucao import Tee
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        f = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.__stdout__, f), Tee(sys.__stderr__, f)
    print(f"\n######## {agora()} | {' '.join(sys.argv)} | pid {os.getpid()}", flush=True)

    def estourou(msg):
        if a.etapa == "extrair":
            alvo = STATUS if a.modo == "full" else LOGS / f"avaliacao_status_{a.modo}.txt"
            antigo = alvo.read_text(encoding="utf-8") if alvo.exists() else ""
            escrever(alvo, antigo + f"\nPAROU {agora()}: {msg}\n")
    VIGIA = Vigia(LIMITE_GB, estourou)
    VIGIA.start()
    t0 = time.time()
    sem_suspensao(True)
    try:
        return {"validar": etapa_validar, "extrair": etapa_extrair, "notas": etapa_notas,
                "metricas": etapa_metricas}[a.etapa](a)
    except BaseException:
        traceback.print_exc()
        print("STATUS=FALHOU", flush=True)
        return 1
    finally:
        sem_suspensao(False)
        VIGIA.parar = True
        print(f"PICO_MEMORIA_GB={VIGIA.pico:.2f} (processo + workers) | sistema {VIGIA.pico_sis:.1f} GB | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
