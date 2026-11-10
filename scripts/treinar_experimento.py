#!/usr/bin/env python3
"""Treina UM experimento do notebook atual num processo próprio (toda a memória volta ao terminar).

Os hiperparâmetros NÃO são copiados aqui: o script executa a célula de configuração do
notebook.ipynb (a célula com a tag "config", seção 2) e usa o mesmo código de src/ que o notebook usa.

v3 (uso normal, na raiz do repo, com o env ativo):
    python scripts/treinar_experimento.py --condicao repeat --seed 42
    python scripts/treinar_experimento.py --condicao zero --seed 42 --quick      # teste rápido
    opções: --num-workers N (padrão: NUM_WORKERS da configuração, 4) | --log logs/arquivo.log

Etapas de uma execução (cada uma grava o seu arquivo de forma atômica e é pulada se já terminou):
    1. treino (retomável por época: models/<exp>_estado.pt) -> models/<exp>_cnn.pt + results/<exp>_historico.csv
    2. notas da validação  -> results/<exp>_notas_val.parquet
    3. notas do dev_shift  -> results/<exp>_notas_dev_shift.parquet
    4. MARCADOR de conclusão, gravado por último -> results/<exp>_concluido.json
Só o marcador diz que a execução terminou. Rodar o mesmo comando de novo continua de onde parou;
com o marcador presente, não faz nada. O teste final NUNCA é avaliado aqui.
Com --quick, todos os arquivos levam o sufixo _quick (não valem como resultado).

Versões antigas (v2): --exp <id> --notebook notebooks/versions/v2_augmentation.ipynb [--quick]
treina como antes (só a etapa 1, sem marcador).
"""
import os

# Antes de importar numpy: o OpenBLAS reserva ~675 MB por processo com as 22 threads do PC.
# O processo principal não usa álgebra linear na CPU (o treino roda na GPU).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.memoria import Amostrador  # noqa: E402


class Tee:
    """Escreve no console e num arquivo de log ao mesmo tempo (UTF-8; o console troca o que não sabe mostrar)."""

    def __init__(self, console, arquivo):
        self.console, self.arquivo = console, arquivo

    def write(self, s):
        try:
            self.console.write(s)
        except UnicodeEncodeError:
            self.console.write(s.encode(self.console.encoding or "ascii", "replace").decode(self.console.encoding or "ascii"))
        self.arquivo.write(s)
        return len(s)

    def flush(self):
        self.console.flush()
        self.arquivo.flush()


def celula_config(notebook):
    cel = [c for c in json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
           if "config" in c.get("metadata", {}).get("tags", [])]
    assert len(cel) == 1, f"esperava 1 célula com a tag 'config' em {notebook}, achei {len(cel)}"
    return "".join(cel[0]["source"])


def executar_config(src, quick, notebook):
    """Executa a célula de configuração. QUICK é passado por QUICK_DO_SCRIPT (a célula lê essa variável);
    células antigas (v2), que não a leem, recebem a troca de texto antiga."""
    ns = {"__name__": "config_do_notebook", "QUICK_DO_SCRIPT": quick}
    if "QUICK_DO_SCRIPT" not in src and quick:
        assert "QUICK = False" in src
        src = src.replace("QUICK = False", "QUICK = True", 1)
    exec(compile(src, f"{notebook} (célula config)", "exec"), ns)
    assert ns["QUICK"] == quick, "a célula de configuração não respeitou o modo QUICK pedido"
    return ns


def legado(a, ns, hp, amostrador, t0):
    """Fluxo da v2: só treina (sem marcador, sem notas)."""
    from src.experimento import ler_manifest, loader_validacao, preparar_treino, treinar_experimento
    from src.model import LogMelPadronizado
    ckpt, hist_csv = ns["caminho_ckpt"](a.exp), ns["caminho_hist"](a.exp)
    if ckpt.exists() and hist_csv.exists():
        print(f"{a.exp}: {ckpt} e {hist_csv} já existem; nada a fazer (apague os dois para treinar de novo).")
        return 0
    df = ler_manifest()
    pool_tr, grupos, cotas, val_df = preparar_treino(df, hp)
    del df
    logmel = LogMelPadronizado(hp["SAMPLE_RATE"], hp["N_FFT"], hp["WIN_LENGTH"], hp["HOP_LENGTH"],
                               hp["N_MELS"]).to(ns["DEVICE"])
    treinar_experimento(a.exp, ns["EXPERIMENTOS"][a.exp], hp, pool_tr, grupos, cotas, val_df,
                        loader_validacao(val_df, hp), logmel, ns["DEVICE"], ckpt, hist_csv, ns["VERSAO"],
                        log=lambda s: print(s, flush=True))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condicao", choices=["repeat", "zero"], help="política de áudio curto (v3)")
    ap.add_argument("--seed", type=int, help="seed de treino (v3)")
    ap.add_argument("--exp", help="id do experimento (alternativa a --condicao/--seed; versões antigas)")
    ap.add_argument("--quick", action="store_true", help="teste rápido (arquivos com sufixo _quick)")
    ap.add_argument("--num-workers", type=int, default=None, help="workers do DataLoader (padrão: o da configuração)")
    ap.add_argument("--notebook", default="notebook.ipynb")
    ap.add_argument("--log", default=None, help="também grava a saída neste arquivo (acrescenta)")
    a = ap.parse_args()

    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        arq = open(a.log, "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, arq)
        sys.stderr = Tee(sys.__stderr__, arq)
    print(f"\n######## {datetime.now().isoformat(timespec='seconds')} | {' '.join(sys.argv)}", flush=True)

    amostrador = Amostrador()
    amostrador.start()
    t0 = time.time()
    try:
        return rodar(a, amostrador, t0)
    except BaseException:
        traceback.print_exc()
        print("STATUS=FALHOU", flush=True)
        return 1
    finally:
        amostrador.parar = True
        print(f"PICO_MEMORIA_GB={amostrador.pico_proc / 1024:.2f} (este processo + workers) | "
              f"pico do sistema {amostrador.pico_sis:.1f} GB | {(time.time() - t0) / 60:.1f} min neste processo",
              flush=True)


def rodar(a, amostrador, t0):
    src_cfg = celula_config(a.notebook)
    ns = executar_config(src_cfg, a.quick, a.notebook)
    hp = dict(ns["HP"])
    if a.num_workers is not None:
        hp["NUM_WORKERS"] = a.num_workers

    if a.condicao or a.seed is not None:
        if not (a.condicao and a.seed is not None):
            sys.exit("use --condicao e --seed juntos")
        exp = f"{ns['VERSAO']}_{a.condicao}_s{a.seed}"
    elif a.exp:
        exp = a.exp
    else:
        sys.exit("informe --condicao e --seed (ou --exp)")
    if exp not in ns["EXPERIMENTOS"]:
        sys.exit(f"experimento '{exp}' não existe; opções: {list(ns['EXPERIMENTOS'])}")
    cfg = ns["EXPERIMENTOS"][exp]
    if a.condicao:
        assert cfg["short_audio_mode"] == a.condicao and cfg["seed"] == a.seed, cfg
    if "caminho_marcador" not in ns:
        a.exp = exp
        return legado(a, ns, hp, amostrador, t0)

    from src.arquivos import (id_config, identificar_codigo, salvar_json, salvar_parquet, sha256_arquivo,
                              sha256_texto)
    from src.experimento import (amostra_dev_shift, calcular_notas, carregar_modelo, ler_manifest,
                                 loader_validacao, modo_curto, preparar_treino, seed_treino,
                                 treinar_experimento)
    from src.metrics import eer_e_limiar, metricas, sigmoide
    from src.model import LogMelPadronizado

    marcador = ns["caminho_marcador"](exp)
    if marcador.exists():
        print(f"{exp}: já concluído ({marcador}); nada a fazer.")
        print("STATUS=JA_CONCLUIDO")
        return 0

    ckpt, hist_csv, estado = ns["caminho_ckpt"](exp), ns["caminho_hist"](exp), ns["caminho_estado"](exp)
    arq_val, arq_dev = ns["caminho_notas"](exp, "val"), ns["caminho_notas"](exp, "dev_shift")
    device = ns["DEVICE"]
    manifest = ROOT / "data/processed/manifest.parquet"
    manifest_sha = sha256_arquivo(manifest)
    # o que define o MÉTODO desta execução (NUM_WORKERS não entra: não muda o método, ver HISTORICO h)
    metodo = {"experimento": exp, "versao": ns["VERSAO"], "quick": a.quick, "cfg": cfg,
              "hp": {k: v for k, v in hp.items() if k != "NUM_WORKERS"}, "manifest_sha256": manifest_sha}
    cid = id_config(metodo)
    codigo = identificar_codigo(ROOT, extras=["scripts/treinar_experimento.py"])
    codigo["celula_config_sha256"] = sha256_texto(src_cfg)
    print(f"{exp} | condição {modo_curto(cfg)} | seed de treino {seed_treino(cfg, hp)} | QUICK {a.quick} | "
          f"NUM_WORKERS {hp['NUM_WORKERS']} | config_id {cid} | commit {codigo['commit']}"
          + (" (com mudanças não commitadas)" if codigo["mudancas_nao_commitadas"] else ""), flush=True)

    df = ler_manifest(manifest)
    logmel = LogMelPadronizado(hp["SAMPLE_RATE"], hp["N_FFT"], hp["WIN_LENGTH"], hp["HOP_LENGTH"],
                               hp["N_MELS"]).to(device)
    pool_tr, grupos, cotas, val_df = preparar_treino(df, hp)
    dev_df = amostra_dev_shift(df, hp)
    del df

    # ---- etapa 1: treino (pula se o histórico já existe e o checkpoint é desta configuração)
    treino_ok = False
    if hist_csv.exists() and ckpt.exists():
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if ck["config"].get("config_id") == cid:
            treino_ok = True
            print(f"etapa 1: treino já concluído ({hist_csv}); pulando.", flush=True)
        else:
            sys.exit(f"{ckpt} existe com outra configuração ({ck['config'].get('config_id')} ≠ {cid}). "
                     "Nada foi apagado; confira à mão.")
    if not treino_ok:
        print(f"treino: {len(pool_tr)} áudios | validação: {len(val_df)}", flush=True)
        treinar_experimento(exp, cfg, hp, pool_tr, grupos, cotas, val_df, None, logmel, device, ckpt,
                            hist_csv, ns["VERSAO"], log=lambda s: print(s, flush=True),
                            estado_path=estado, config_id=cid,
                            extra_config={"manifest_sha256": manifest_sha, "codigo": codigo})

    # ---- etapas 2 e 3: notas por áudio (janelas e central) na validação e no dev_shift
    model, ck, modo = carregar_modelo(ckpt, device)
    assert modo == modo_curto(cfg) and ck["config"].get("config_id") == cid
    notas = {}
    for nome, amostra, arq in [("val", val_df, arq_val), ("dev_shift", dev_df, arq_dev)]:
        if arq.exists():
            salvo = pd_read(arq)
            if (salvo["path"].tolist() == amostra["path"].tolist()
                    and salvo["config_id"].eq(cid).all() and salvo["short_audio_mode"].eq(modo).all()):
                notas[nome] = salvo
                print(f"etapa {nome}: notas já calculadas ({arq}); pulando.", flush=True)
                continue
            print(f"etapa {nome}: {arq} é de outra configuração/amostra; recalculando.", flush=True)
        t = time.time()
        out = calcular_notas(model, logmel, amostra, hp, modo, device)
        out = out.assign(experimento=exp, short_audio_mode=modo, seed=seed_treino(cfg, hp), config_id=cid)
        salvar_parquet(out, arq)
        notas[nome] = out
        print(f"etapa {nome}: {len(out)} áudios em {(time.time() - t) / 60:.1f} min -> {arq}", flush=True)

    # ---- etapa 4: resumo e marcador (por último, atômico)
    import pandas as pd
    hist = pd.read_csv(hist_csv)
    i = hist["EER média por fonte (val)"].idxmin()
    retomadas = []
    if estado.exists():
        import torch
        retomadas = torch.load(estado, map_location="cpu", weights_only=False).get("retomadas", [])
    v, d = notas["val"], notas["dev_shift"]
    limiar = eer_e_limiar(v.label.values, v.z_janelas.values)[1]     # mesmo procedimento nos dois braços
    m_dev = metricas(d.label.values, d.z_janelas.values, limiar)
    m_dev_c = metricas(d.label.values, d.z_central.values, eer_e_limiar(v.label.values, v.z_central.values)[1])
    resumo = {
        "experimento": exp, "versao": ns["VERSAO"], "condicao": modo, "seed": seed_treino(cfg, hp),
        "quick": a.quick, "status": "concluido", "concluido_em": datetime.now().isoformat(timespec="seconds"),
        "melhor_epoca": int(hist.loc[i, "época"]), "epocas": int(len(hist)),
        "minutos_treino": round(float(hist["segundos"].sum()) / 60, 2),
        "minutos_neste_processo": round((time.time() - t0) / 60, 2),
        "pico_memoria_gb_processo_e_workers": round(amostrador.pico_proc / 1024, 2),
        "pico_memoria_gb_sistema": round(amostrador.pico_sis, 2),
        "val_eer_media_por_fonte": float(hist.loc[i, "EER média por fonte (val)"]),
        "val_eer_global_central": float(hist.loc[i, "EER global (val)"]),
        "val_eer_por_fonte": {c.replace("EER val ", ""): float(hist.loc[i, c]) for c in hist.columns
                              if c.startswith("EER val ")},
        "limiar_janelas_logit": float(limiar), "limiar_janelas_prob": float(sigmoide(limiar)),
        "dev_shift_janelas": {k: (None if pd.isna(x) else float(x)) for k, x in m_dev.items()},
        "dev_shift_central": {k: (None if pd.isna(x) else float(x)) for k, x in m_dev_c.items()},
        "treino_retomado_nas_epocas": retomadas,
        "config_id": cid, "metodo": metodo, "num_workers": hp["NUM_WORKERS"], "codigo": codigo,
        "arquivos": {"checkpoint": str(ckpt), "historico": str(hist_csv), "notas_val": str(arq_val),
                     "notas_dev_shift": str(arq_dev)},
    }
    salvar_json(resumo, marcador)
    if estado.exists():
        estado.unlink()       # o estado só serve para retomar o treino; o marcador já foi gravado
    print(f"\n{exp} CONCLUÍDO | melhor época {resumo['melhor_epoca']} | EER média/fonte (val) "
          f"{100 * resumo['val_eer_media_por_fonte']:.2f}% | dev_shift (janelas) EER {100 * m_dev['EER']:.1f}% "
          f"AUC {m_dev['AUC']:.3f} | marcador: {marcador}", flush=True)
    print("STATUS=CONCLUIDO")
    return 0


def pd_read(arq):
    import pandas as pd
    return pd.read_parquet(arq)


if __name__ == "__main__":
    sys.exit(main())
