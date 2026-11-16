"""Diagnostico SOMENTE LEITURA das transcricoes do teste WhatsApp (DEV x RESERVADO por pessoa_id).
Etapas: `dividir` (passo 0, nao abre transcricao) e `diagnostico` (so DEV). Nao chama API nem modelo.
RESERVADO: nunca abrir transcricao nem calcular qualidade de transcricao."""
import hashlib, json, random, re, sys, datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIV = ROOT / "results/diagnostico_whisper_divisao.csv"
MAN = ROOT / "data/processed/manifest_whatsapp.parquet"


def dividir():
    m = pd.read_parquet(MAN)
    gp = m.groupby("pessoa_id")["grupo"].agg(lambda s: set(s)).to_dict()
    pessoas = sorted(gp)
    rng = random.Random(42)
    for _ in range(100000):
        dev = rng.sample(pessoas, 8)
        c = lambda g: sum(g in gp[p] for p in dev)
        if c("real_sem_golpe") >= 2 and c("real_com_golpe") >= 4 and c("ia_neutro") >= 2 and c("ia_pedido_dinheiro") >= 2:
            break
    else:
        raise SystemExit("sem divisao valida")
    pd.DataFrame({"pessoa_id": pessoas, "grupo": ["DEV" if p in dev else "RESERVADO" for p in pessoas]}).to_csv(DIV, index=False)
    print("SHA256", hashlib.sha256(DIV.read_bytes()).hexdigest(), datetime.datetime.now().isoformat(timespec="seconds"))
    print(pd.read_csv(DIV).grupo.value_counts().to_dict())


def _p(t):
    t = re.sub(r"[^\w\s]", " ", str(t).lower())
    return t.split()


def _wer(ref, hip):
    r, h = _p(ref), _p(hip)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def diagnostico():
    import difflib
    div = pd.read_csv(DIV)
    m = pd.read_parquet(MAN)
    dev_p = set(div[div.grupo == "DEV"].pessoa_id)
    res_p = set(div[div.grupo == "RESERVADO"].pessoa_id)
    dev = set(m[m.pessoa_id.isin(dev_p)].id)
    proib = set(m[m.pessoa_id.isin(res_p)].id)
    ctx = {}
    for l in open(ROOT / "data/processed/tw_contexto.jsonl", encoding="utf8"):
        i = json.loads(l)["id"] if '"id"' in l else None
        if i not in dev:                      # filtra ANTES de usar qualquer campo
            continue
        assert i not in proib
        ctx[i] = json.loads(l)["res"]
    assert set(ctx) <= dev and not (set(ctx) & proib)
    g = pd.read_csv(ROOT / "data/teste_whatsapp/extraido/coleta/manifesto_geral.csv", usecols=["pessoa_id", "arquivo", "tema", "texto"])
    notas = pd.read_csv(ROOT / "results/teste_whatsapp_notas.csv", usecols=["id", "pessoa_id", "grupo", "ctx_risco", "alerta_medio"])
    notas = notas[notas.id.isin(dev)]
    d = m[m.id.isin(dev)][["id", "pessoa_id", "grupo", "caminho"]].merge(notas[["id", "ctx_risco", "alerta_medio"]], on="id")
    d["idioma"] = d.id.map(lambda i: ctx[i]["idioma"]); d["n_pal"] = d.id.map(lambda i: len(_p(ctx[i]["transcricao"])))
    d["obs"] = d.id.map(lambda i: len(ctx[i]["observacoes"])); d["susp"] = d.id.map(lambda i: ctx[i]["transcricao_suspeita"])
    # texto verdadeiro dos clones: casa pelo nome do arquivo
    g["k"] = g.arquivo.astype(str).map(lambda a: Path(a).stem)
    d["k"] = d.caminho.map(lambda a: Path(a).stem)
    d = d.merge(g[["k", "texto"]], on="k", how="left")
    cl = d[d.grupo.str.startswith("ia_")].copy()
    cl["wer"] = [_wer(r.texto, ctx[r.id]["transcricao"]) for r in cl.itertuples()]
    cl["sim"] = [difflib.SequenceMatcher(None, " ".join(_p(r.texto)), " ".join(_p(ctx[r.id]["transcricao"]))).ratio() for r in cl.itertuples()]
    print("n clones com texto verdadeiro:", cl.texto.notna().sum(), "de", len(cl))
    print(cl.groupby("grupo")[["wer", "sim"]].agg(["mean", "median", "min", "max"]).round(3).to_string())
    print("idioma clones:", cl.idioma.value_counts().to_dict())
    ing = [i for i in cl.id if ctx[i]["idioma"].lower().startswith("en")]
    print("clones com idioma detectado ingles:", len(ing))
    re_ = d[d.grupo.str.startswith("real_")]
    print("reais DEV: idioma", re_.idioma.value_counts().to_dict(), "| n_pal media", round(re_.n_pal.mean(), 1),
          "| <4 palavras:", int((re_.n_pal < 4).sum()), "de", len(re_), "| suspeitas:", int(re_.susp.sum()), "| com observacoes:", int((re_.obs > 0).sum()))
    rng = random.Random(42)
    for gr in ("real_com_golpe", "real_sem_golpe"):
        ids = sorted(re_[re_.grupo == gr].id)
        for i in rng.sample(ids, 2):
            print(f"[{gr}] {i} ({ctx[i]['idioma']}, {d.set_index('id').n_pal[i]} pal, risco={ctx[i]['risco_conteudo']}):", ctx[i]["transcricao"][:200].replace(chr(10), " "))
    pd_ = cl[cl.grupo == "ia_pedido_dinheiro"].copy()
    pd_["alerta"] = pd_.alerta_medio.astype(bool)
    print(pd_[["id", "wer", "ctx_risco", "alerta"]].round(3).to_string(index=False))
    print(pd_.groupby("alerta").wer.agg(["count", "mean", "median"]).round(3))
    print(re_.groupby("grupo").ctx_risco.value_counts().to_string())
    out = d[["id", "pessoa_id", "grupo", "ctx_risco", "alerta_medio", "idioma", "n_pal", "susp", "obs"]].merge(cl[["id", "wer", "sim"]], on="id", how="left")
    out.to_csv(ROOT / "results/diagnostico_whisper_dev.csv", index=False)


if __name__ == "__main__":
    {"dividir": dividir, "diagnostico": diagnostico}[sys.argv[1]]()
