#!/usr/bin/env python3
"""Avaliação do módulo de contexto de golpe com textos INVENTADOS (protocolo: docs/PROTOCOLO_CONTEXTO.md).

Não treina nada. Só texto (a transcrição do Whisper não entra aqui). Etapas:
    python scripts/contexto_golpe_avaliar.py dev --modelo openai/gpt-oss-120b --revisao r0
    python scripts/contexto_golpe_avaliar.py congelar --revisao rN        # grava results/contexto_golpe_congelado.json
    python scripts/contexto_golpe_avaliar.py teste                        # UMA vez; recusa se já existir
    python scripts/contexto_golpe_avaliar.py consistencia                 # 20 textos (seed 42) x 3 execuções
    python scripts/contexto_golpe_avaliar.py resumo                       # results/contexto_golpe_avaliacao_resumo.json
    python scripts/contexto_golpe_avaliar.py figuras                      # results/slides/ctx_0*.png

Ritmo: uma chamada a cada 10 s (~6 por minuto; o gpt-oss-120b tem 8.000 tokens/min e cada análise gasta ~1.100).
Para sozinho se o limite diário restante (x-ratelimit-remaining-requests) cair abaixo de 100.
As "pistas_ocultas" dos textos NUNCA são enviadas ao modelo: só o campo "texto".
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src import contexto_golpe as cg  # noqa: E402

PASTA = ROOT / "docs" / "avaliacao_contexto"
DEV, TESTE = PASTA / "textos_dev.jsonl", PASTA / "textos_teste.jsonl"
R = ROOT / "results"
CSV_DEV, CSV_TESTE = R / "contexto_golpe_dev.csv", R / "contexto_golpe_avaliacao.csv"
CSV_CONS = R / "contexto_golpe_consistencia.csv"
CONGELADO = R / "contexto_golpe_congelado.json"
RESUMO = R / "contexto_golpe_avaliacao_resumo.json"
INTERVALO_S = 12.0
MIN_RESTANTE_DIA = 100
POS = ("medio", "alto")
NIVEIS = ("nenhum", "baixo", "medio", "alto", "indeterminado")
GRUPOS = ("golpe_explicito", "golpe_escondido", "injecao_golpe", "legitimo_comum", "legitimo_dificil",
          "injecao_legitimo", "ambiguo")
NOME_GRUPO = {"golpe_explicito": "Golpes explícitos", "golpe_escondido": "Golpes escondidos",
              "injecao_golpe": "Golpes com injeção", "legitimo_comum": "Legítimas comuns",
              "legitimo_dificil": "Legítimas difíceis", "injecao_legitimo": "Legítimas com injeção",
              "ambiguo": "Ambíguas (fora da acurácia)"}

# ---------------------------------------------------------------------------------------- contagem de chamadas
ESTADO = {"http": 0, "r429": 0, "restante_dia": None}
_req_original = cg.requests.request


def _req_contado(*a, **k):
    r = _req_original(*a, **k)
    ESTADO["http"] += 1
    if r.status_code == 429:
        ESTADO["r429"] += 1
    v = r.headers.get("x-ratelimit-remaining-requests")
    if v is not None:
        try:
            ESTADO["restante_dia"] = int(v)
        except ValueError:
            pass
    return r


cg.requests.request = _req_contado


def ler(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_arquivo(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_texto(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def grupo(l):
    if l["contem_injecao"]:
        return f"injecao_{l['rotulo']}"
    if l["rotulo"] == "ambiguo":
        return "ambiguo"
    return f"{l['rotulo']}_{l['dificuldade']}"


def prompt_da_revisao(rev):
    return (PASTA / f"prompt_{rev}.txt").read_text(encoding="utf-8")


def rodar(textos, rotulo_execucao, extra=None):
    """Analisa cada texto com cg.analisar_texto, no ritmo de INTERVALO_S. Devolve a lista de linhas."""
    import pandas as pd  # noqa: F401
    linhas = []
    for i, l in enumerate(textos):
        if ESTADO["restante_dia"] is not None and ESTADO["restante_dia"] < MIN_RESTANTE_DIA:
            print(f"PARADO: limite diário restante {ESTADO['restante_dia']} < {MIN_RESTANTE_DIA}", flush=True)
            break
        t0 = time.perf_counter()
        h0, q0 = ESTADO["http"], ESTADO["r429"]
        r = cg.analisar_texto(l["texto"])            # só o texto vai ao modelo
        lin = {"id": l["id"], "rotulo": l["rotulo"], "categoria": l["categoria"], "dificuldade": l["dificuldade"],
               "contem_injecao": l["contem_injecao"], "grupo": grupo(l), "risco_previsto": r["risco_conteudo"],
               "sinais": ";".join(s["tipo"] for s in r["sinais"]),
               "evidencias": " | ".join(s["evidencia"] for s in r["sinais"]),
               "n_evidencias_descartadas": r["evidencias_descartadas"], "manipulacao_detectada": r["manipulacao_detectada"],
               "latencia_ms": r["latencia_ms"]["classificacao"], "status": r["status"], "erro": r["erro"] or "",
               "observacoes": " | ".join(r["observacoes"]), "resumo": r["resumo"], "modelo": r["modelos"]["llm"],
               "execucao": rotulo_execucao, "n_http": ESTADO["http"] - h0, "n_429": ESTADO["r429"] - q0,
               "quando": time.strftime("%Y-%m-%d %H:%M:%S")}
        if extra:
            lin.update(extra)
        linhas.append(lin)
        print(f"[{i + 1}/{len(textos)}] {l['id']:10s} {grupo(l):18s} -> {r['risco_conteudo']:13s} "
              f"{lin['sinais'] or '-'} ({lin['latencia_ms']} ms){' ERRO ' + lin['erro'] if lin['erro'] else ''}",
              flush=True)
        espera = INTERVALO_S - (time.perf_counter() - t0)
        if espera > 0 and i < len(textos) - 1:
            time.sleep(espera)
    return linhas


def anexar_csv(linhas, caminho):
    import pandas as pd
    df = pd.DataFrame(linhas)
    if caminho.exists():
        df = pd.concat([pd.read_csv(caminho, keep_default_na=False), df], ignore_index=True)
    df.to_csv(caminho, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------------------- métricas
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def taxa(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "taxa": (k / n) if n else None, "ic95": [lo, hi]}


def metricas(df, positivos=POS):
    """df com colunas rotulo, grupo, risco_previsto. Ambíguos ficam fora da acurácia."""
    pos = df.risco_previsto.isin(positivos)
    neg = df.risco_previsto.isin([n for n in ("nenhum", "baixo", "medio", "alto") if n not in positivos])
    na = df.rotulo != "ambiguo"
    golpe, leg = df.rotulo == "golpe", df.rotulo == "legitimo"
    acerto = (golpe & pos) | (leg & neg)
    out = {"acuracia_nao_ambiguos": taxa((acerto & na).sum(), na.sum()),
           "recall_golpes_todos": taxa((golpe & pos).sum(), golpe.sum()),
           "falso_alarme_legitimos_todos": taxa((leg & pos).sum(), leg.sum()),
           "precisao": taxa((golpe & pos).sum(), (na & pos).sum()),
           "indeterminados": int((df.risco_previsto == "indeterminado").sum())}
    rec = out["recall_golpes_todos"]["taxa"] or 0
    fa = out["falso_alarme_legitimos_todos"]["taxa"] or 0
    out["acuracia_balanceada"] = (rec + (1 - fa)) / 2
    por = {}
    for g in GRUPOS:
        m = df.grupo == g
        if not m.any():
            continue
        if g.startswith("golpe") or g == "injecao_golpe":
            por[g] = {"recall": taxa((m & pos).sum(), m.sum())}
        elif g == "ambiguo":
            por[g] = {"marcados_positivo": taxa((m & pos).sum(), m.sum())}
        else:
            por[g] = {"falso_alarme": taxa((m & pos).sum(), m.sum())}
        por[g]["niveis"] = {n: int((m & (df.risco_previsto == n)).sum()) for n in NIVEIS}
    out["por_grupo"] = por
    return out


def imprimir_metricas(mt, titulo):
    f = lambda d: f"{d['taxa']:.3f} ({d['k']}/{d['n']}; IC95 {d['ic95'][0]:.2f}-{d['ic95'][1]:.2f})"
    print(f"\n== {titulo}")
    print("acurácia (não ambíguos):", f(mt["acuracia_nao_ambiguos"]))
    print("recall dos golpes:", f(mt["recall_golpes_todos"]))
    print("falso alarme nos legítimos:", f(mt["falso_alarme_legitimos_todos"]))
    print("precisão:", f(mt["precisao"]), "| acurácia balanceada:", round(mt["acuracia_balanceada"], 3),
          "| indeterminados:", mt["indeterminados"])
    for g, d in mt["por_grupo"].items():
        k = [x for x in d if x != "niveis"][0]
        print(f"  {g:18s} {k:18s} {f(d[k])}  níveis {d['niveis']}")


# ---------------------------------------------------------------------------------------- etapas
def etapa_dev(a):
    import pandas as pd
    os.environ["GROQ_LLM_MODEL"] = a.modelo
    prompt = prompt_da_revisao(a.revisao)
    cg.PROMPT_SISTEMA = prompt
    textos = ler(DEV)
    if CSV_DEV.exists():
        ant = pd.read_csv(CSV_DEV, keep_default_na=False)
        if ((ant.modelo == a.modelo) & (ant.revisao == a.revisao)).any():
            sys.exit(f"já existe execução dev para {a.modelo} {a.revisao}")
    print(f"DEV: {len(textos)} textos, modelo {a.modelo}, prompt {a.revisao} (sha256 {sha256_texto(prompt)[:16]}...)")
    linhas = rodar(textos, f"dev_{a.revisao}", {"revisao": a.revisao, "sha256_prompt": sha256_texto(prompt)})
    df = anexar_csv(linhas, CSV_DEV)
    sub = df[(df.modelo == a.modelo) & (df.revisao == a.revisao)]
    imprimir_metricas(metricas(sub), f"DEV {a.modelo} {a.revisao}")
    print(f"\nchamadas HTTP nesta execução: {ESTADO['http']} | 429: {ESTADO['r429']} | "
          f"restante no dia: {ESTADO['restante_dia']}")


def parametros():
    m = cg.modelos()["llm"]
    return {"modelo_llm": m, "temperature": 0, "seed": cg.SEED, "max_tokens": cg.MAX_TOKENS_LLM,
            "response_format": "json_object", **cg.extras_llm(m), "max_chars": cg.MAX_CHARS_PADRAO,
            "timeout_s": cg.TIMEOUT_S, "max_retentativas": cg.MAX_RETENTATIVAS}


def etapa_congelar(a):
    if os.environ.get("GROQ_LLM_MODEL"):
        sys.exit("tire GROQ_LLM_MODEL do ambiente: o teste usa o padrão gravado em src/contexto_golpe.py")
    if cg.PROMPT_SISTEMA != prompt_da_revisao(a.revisao):
        sys.exit(f"o PROMPT_SISTEMA de src/contexto_golpe.py não é igual a prompt_{a.revisao}.txt")
    if CONGELADO.exists():
        sys.exit("já congelado")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    c = {"congelado_em": time.strftime("%Y-%m-%d %H:%M:%S (UTC%z)"), "commit_base": commit, "revisao": a.revisao,
         "sha256_prompt_sistema": sha256_texto(cg.PROMPT_SISTEMA), "sha256_textos_teste": sha256_arquivo(TESTE),
         "sha256_src_contexto_golpe": sha256_arquivo(ROOT / "src" / "contexto_golpe.py"),
         "parametros": parametros()}
    CONGELADO.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(c, ensure_ascii=False, indent=2))


def conferir_congelado():
    if os.environ.get("GROQ_LLM_MODEL"):
        sys.exit("tire GROQ_LLM_MODEL do ambiente")
    c = json.loads(CONGELADO.read_text(encoding="utf-8"))
    atual = {"sha256_prompt_sistema": sha256_texto(cg.PROMPT_SISTEMA), "sha256_textos_teste": sha256_arquivo(TESTE),
             "sha256_src_contexto_golpe": sha256_arquivo(ROOT / "src" / "contexto_golpe.py"), "parametros": parametros()}
    for k, v in atual.items():
        if c[k] != v:
            sys.exit(f"DIFERENTE do congelado: {k}")
    return c


def etapa_teste(a):
    c = conferir_congelado()
    if CSV_TESTE.exists():
        sys.exit("o teste já foi rodado (uma vez só); nada foi feito")
    textos = ler(TESTE)
    print(f"TESTE: {len(textos)} textos, congelado em {c['congelado_em']}, modelo {c['parametros']['modelo_llm']}")
    linhas = rodar(textos, "teste", {"revisao": c["revisao"]})
    anexar_csv(linhas, CSV_TESTE)
    print(f"\nchamadas HTTP: {ESTADO['http']} | 429: {ESTADO['r429']} | restante no dia: {ESTADO['restante_dia']}")


def sorteio_consistencia():
    import numpy as np
    ids = sorted(l["id"] for l in ler(TESTE))
    rng = np.random.default_rng(42)
    return sorted(ids[i] for i in rng.choice(len(ids), size=20, replace=False))


def etapa_consistencia(a):
    c = conferir_congelado()
    if not CSV_TESTE.exists():
        sys.exit("rode o teste antes")
    if CSV_CONS.exists():
        sys.exit("a consistência já foi rodada")
    escolhidos = set(sorteio_consistencia())
    textos = [l for l in ler(TESTE) if l["id"] in escolhidos]
    todas = []
    for k in (1, 2, 3):                      # em rodadas, para espalhar no tempo
        print(f"\nCONSISTÊNCIA: rodada {k}/3 ({len(textos)} textos)")
        todas += rodar(textos, f"consistencia_{k}", {"revisao": c["revisao"], "rodada": k})
        if ESTADO["restante_dia"] is not None and ESTADO["restante_dia"] < MIN_RESTANTE_DIA:
            break
        time.sleep(INTERVALO_S)
    anexar_csv(todas, CSV_CONS)
    print(f"\nchamadas HTTP: {ESTADO['http']} | 429: {ESTADO['r429']} | restante no dia: {ESTADO['restante_dia']}")


def etapa_resumo(a):
    import pandas as pd
    c = json.loads(CONGELADO.read_text(encoding="utf-8"))
    df = pd.read_csv(CSV_TESTE, keep_default_na=False)
    mt, mt_alto = metricas(df), metricas(df, positivos=("alto",))
    imprimir_metricas(mt, "TESTE (positivo = medio ou alto)")
    imprimir_metricas(mt_alto, "TESTE (positivo = só alto)")
    inj = df[df.grupo == "injecao_golpe"]
    ok_inj = inj.risco_previsto.isin(POS) & (inj.manipulacao_detectada.astype(str) == "True")
    leg_inj = df[df.grupo == "injecao_legitimo"]
    res = {"congelado": c, "n_textos": len(df), "positivo_medio_ou_alto": mt, "positivo_so_alto": mt_alto,
           "injecao": {"golpes_ainda_pos_e_manipulacao": taxa(ok_inj.sum(), len(inj)),
                       "golpes_ainda_pos": taxa(inj.risco_previsto.isin(POS).sum(), len(inj)),
                       "golpes_manipulacao_detectada": taxa((inj.manipulacao_detectada.astype(str) == "True").sum(), len(inj)),
                       "legitimos_com_injecao_marcados_pos": taxa(leg_inj.risco_previsto.isin(POS).sum(), len(leg_inj)),
                       "legitimos_com_injecao_manipulacao": taxa(
                           (leg_inj.manipulacao_detectada.astype(str) == "True").sum(), len(leg_inj))},
           "evidencias": {"descartadas_total": int(df.n_evidencias_descartadas.astype(int).sum()),
                          "textos_com_descarte": int((df.n_evidencias_descartadas.astype(int) > 0).sum()),
                          "sinais_validos_total": int(sum(len(s.split(";")) for s in df.sinais if s))},
           "latencia_ms": {"mediana": float(df.latencia_ms.astype(float).median()),
                           "p90": float(df.latencia_ms.astype(float).quantile(0.9)),
                           "max": float(df.latencia_ms.astype(float).max())},
           "chamadas": {"teste_http": int(df.n_http.astype(int).sum()), "teste_429": int(df.n_429.astype(int).sum()),
                        "erros": int((df.erro != "").sum())},
           "metas_declaradas": {"recall_golpes_min": 0.85, "falso_alarme_legitimos_max": 0.15}}
    if CSV_CONS.exists():
        cs = pd.read_csv(CSV_CONS, keep_default_na=False)
        base = df.set_index("id").risco_previsto
        por = cs.groupby("id").risco_previsto.agg(list)
        iguais3 = sum(len(set(v)) == 1 and len(v) == 3 for v in por)
        iguais4 = sum(len(set(v + [base[i]])) == 1 and len(v) == 3 for i, v in por.items())
        pos_igual = sum(len({x in POS for x in v}) == 1 and len(v) == 3 for v in por)
        res["consistencia"] = {"n_textos": int(len(por)), "ids": list(por.index),
                               "risco_igual_nas_3": taxa(iguais3, len(por)),
                               "risco_igual_nas_3_e_no_teste": taxa(iguais4, len(por)),
                               "decisao_pos_igual_nas_3": taxa(pos_igual, len(por)),
                               "riscos": {i: v for i, v in por.items()},
                               "chamadas_http": int(cs.n_http.astype(int).sum()), "r429": int(cs.n_429.astype(int).sum())}
    if CSV_DEV.exists():
        dv = pd.read_csv(CSV_DEV, keep_default_na=False)
        res["dev"] = {f"{m} {r}": {"acuracia_balanceada": round(metricas(g)["acuracia_balanceada"], 4),
                                   "recall": metricas(g)["recall_golpes_todos"]["taxa"],
                                   "falso_alarme": metricas(g)["falso_alarme_legitimos_todos"]["taxa"],
                                   "chamadas_http": int(g.n_http.astype(int).sum()), "r429": int(g.n_429.astype(int).sum())}
                      for (m, r), g in dv.groupby(["modelo", "revisao"], sort=False)}
    RESUMO.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\ngravado {RESUMO}")


def main():
    for f in (sys.stdout, sys.stderr):
        try:
            f.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="etapa", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--modelo", required=True)
    d.add_argument("--revisao", required=True)
    cgl = sub.add_parser("congelar")
    cgl.add_argument("--revisao", required=True)
    for n in ("teste", "consistencia", "resumo", "figuras"):
        sub.add_parser(n)
    a = ap.parse_args()
    if a.etapa == "figuras":
        sys.path.insert(0, str(ROOT / "scripts"))
        import contexto_golpe_figuras
        return contexto_golpe_figuras.main()
    {"dev": etapa_dev, "congelar": etapa_congelar, "teste": etapa_teste, "consistencia": etapa_consistencia,
     "resumo": etapa_resumo}[a.etapa](a)


if __name__ == "__main__":
    main()
