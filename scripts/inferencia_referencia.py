#!/usr/bin/env python3
"""Inferência de REFERÊNCIA do sistema principal (v7, T3) com os modelos leves. Não treina nada.

Para quem vai construir o site: este script é o "gabarito". Um áudio entra, sai a nota (logit) e a
classificação pelo corte. Se a sua implementação der a mesma nota que este script para o mesmo arquivo,
ela está certa. Guia: docs/GUIA_PARA_O_SITE.md.

O caminho (o mesmo da T3; nenhuma conta foi reescrita, tudo vem de src/):
    áudio -> mono 16 kHz (src/preproc.py: ler_mono_16k)
          -> corte do silêncio nas bordas (35 dB; se sobrar < 0,5 s, sem corte) e recorte central de no
             máximo 8 s, sem completar (src/preproc.py: preparar_clipe)
          -> normalização do clipe (média 0, variância 1) e XLS-R 300M cortado depois da 7ª camada
             (src/ssl.py: representar)
          -> média no tempo das saídas 5, 6 e 7 (1.024 números cada) -> 3.072 números
          -> padronização com mu e sd do treino da T3 -> nota = soma ponderada + intercepto
             (results/v7_t3_coeficientes.npz)
          -> comparação com o corte (padrão: regra M da validação, results/v7_t3_modelo.json)

Variantes (docs/MODELO_LEVE.md):
    v1  cortado, fp32 (padrão; roda em qualquer processador). Nota idêntica à do modelo completo.
    v2  cortado, fp16 (precisa de placa de vídeo). Nota idêntica à do modelo completo em fp16.
    v3  cortado, int8 (processador). A nota MUDA (até ~4,5 logit na amostra): o corte da T3 NÃO vale nela.

Uso:
    python scripts/inferencia_referencia.py caminho/do/audio.wav
    python scripts/inferencia_referencia.py caminho/do/audio.wav --variante v2 --corte 0.5
    python scripts/inferencia_referencia.py --verificar            # 20 clipes da amostra da validação
    python scripts/inferencia_referencia.py --verificar 50 --variante v3

Leitura de arquivo: soundfile (WAV, FLAC, OGG, MP3) com o miniaudio de reserva para FLAC. Outros formatos
(m4a, aac, opus em webm...) precisam ser convertidos antes (por exemplo, com o ffmpeg); isso não foi testado.

Arquivos lidos: models/modelo_leve_V{1,2,3}_*.{safetensors,pt} (fora do git; se faltar, é gerado aqui a partir
do modelo da Meta no cache local do Hugging Face, sem baixar nada e sem treinar), results/v7_t3_coeficientes.npz,
results/v7_t3_modelo.json. Para --verificar: data/processed/modelo_leve_amostra/ e data/processed/v7_notas.parquet.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.preproc import ler_mono_16k, preparar_clipe  # noqa: E402
from src.ssl import NOME_PADRAO, representar  # noqa: E402

R, DP, MODELS = ROOT / "results", ROOT / "data" / "processed", ROOT / "models"
CAMADAS = (5, 6, 7)
ARQ = {"v1": MODELS / "modelo_leve_V1_fp32.safetensors", "v2": MODELS / "modelo_leve_V2_fp16.safetensors",
       "v3": MODELS / "modelo_leve_V3_int8.pt"}
AMOSTRA = DP / "modelo_leve_amostra"
NOTAS_T3 = DP / "v7_notas.parquet"
SPLITS = DP / "splits_v7.parquet"
TOL_FP32 = 1e-3
# coluna de referência de cada variante em data/processed/modelo_leve_amostra/notas.parquet (scripts/modelo_leve.py)
REF_AMOSTRA = {"v1": ("z_V0_fp32", "modelo completo fp32 (V0)"), "v2": ("z_V1_fp16", "V2 fp16 gravada"),
               "v3": ("z_V3_int8", "V3 int8 gravada")}


# --------------------------------------------------------------------------------------------- pesos da T3
def pesos_t3():
    z = np.load(R / "v7_t3_coeficientes.npz")
    assert tuple(int(c) for c in z["camadas"]) == CAMADAS, z["camadas"]
    return (z["mu"].astype(np.float32), z["sd"].astype(np.float32), z["coef"].astype(np.float32),
            float(z["intercepto"]))


def corte_m():
    t3 = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))
    return float(t3["cortes_da_validacao_logit"]["M"])


def nota(x, mu, sd, coef, b):
    """x: (3072,) ou (n, 3072). Mesma conta de scripts/v7.py: pontuar (float32)."""
    return (((np.asarray(x, dtype=np.float32) - mu) / sd) @ coef + b).astype(np.float64)


# --------------------------------------------------------------------------------------------- modelos
def esqueleto_cortado():
    """XLS-R 300M com só as 7 primeiras camadas do transformer e o layer norm final trocado por identidade,
    montado a partir do config.json do cache local (sem pesos, sem baixar nada)."""
    from transformers import AutoConfig, Wav2Vec2Model
    cfg = AutoConfig.from_pretrained(NOME_PADRAO, local_files_only=True)
    cfg.num_hidden_layers = max(CAMADAS)
    m = Wav2Vec2Model(cfg)
    m.encoder.layer_norm = torch.nn.Identity()   # no modelo inteiro, só a 24ª saída passa por ele (docs/MODELO_LEVE_PROTOCOLO.md)
    return m.eval()


def gerar_arquivo(variante):
    """Gera models/modelo_leve_V*.* a partir do modelo da Meta no cache local, como scripts/modelo_leve.py (tamanho),
    mas sem gravar nada em results/. Não treina nada."""
    sys.path.insert(0, str(ROOT / "scripts"))
    cwd = os.getcwd()
    try:
        import modelo_leve as ml                       # faz os.chdir(ROOT); voltamos depois
    finally:
        os.chdir(cwd)
    from safetensors.torch import save_file
    MODELS.mkdir(exist_ok=True)
    nome = {"v1": "V1", "v2": "V2", "v3": "V3"}[variante]
    print(f"[modelo] {ARQ[variante].name} não existe; gerando a partir de {NOME_PADRAO} (cache local)...", flush=True)
    m1, _ = ml.montar("V1", "cpu")
    sd1 = {k: v.contiguous() for k, v in m1.state_dict().items()}
    if nome == "V1":
        save_file(sd1, str(ARQ["v1"]))
    elif nome == "V2":
        save_file({k: v.half().contiguous() for k, v in sd1.items()}, str(ARQ["v2"]))
    else:
        m3, _ = ml.montar("V3", "cpu")
        torch.save(m3.state_dict(), ARQ["v3"])
    print(f"[modelo] gravado {ARQ[variante]}", flush=True)


def carregar(variante, aparelho):
    """Devolve (modelo, dtype em texto) já no aparelho, em modo de avaliação, sem gradiente."""
    from safetensors.torch import load_file
    if not ARQ[variante].exists():
        gerar_arquivo(variante)
    m = esqueleto_cortado()
    if variante in ("v1", "v2"):
        sd = load_file(str(ARQ[variante]))
        if variante == "v2":
            m = m.half()
        m.load_state_dict(sd, strict=True)
        dtype = "float16" if variante == "v2" else "float32"
    else:
        eng = torch.backends.quantized.supported_engines
        torch.backends.quantized.engine = "onednn" if "onednn" in eng else torch.backends.quantized.engine
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        m.load_state_dict(torch.load(ARQ["v3"], weights_only=False), strict=True)   # arquivo gerado por nós
        dtype = "float32"
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(aparelho).eval(), dtype


def escolher_aparelho(variante, pedido):
    if variante == "v3":
        if pedido == "cuda":
            sys.exit("A v3 (int8) só roda no processador (--aparelho cpu).")
        return "cpu"
    if variante == "v2":
        if not torch.cuda.is_available():
            sys.exit("A v2 (fp16) precisa de placa de vídeo (CUDA), e nenhuma foi encontrada. Use --variante v1.")
        return "cuda"
    return pedido or "cpu"


# --------------------------------------------------------------------------------------------- uma nota
def vetor_de_clipe(model, clipe, aparelho, dtype):
    """(3072,) = média no tempo das saídas 5, 6 e 7 do XLS-R cortado, via src/ssl.py: representar (lote de 1)."""
    media, _, _ = representar(model, [clipe], device=aparelho, dtype=dtype)    # (1, 8, 1024): entrada + 7 camadas
    assert media.shape[1] == max(CAMADAS) + 1, media.shape
    return torch.cat([media[0, k] for k in CAMADAS]).cpu().numpy()


def nota_de_arquivo(caminho, model, aparelho, dtype, pesos):
    x, sr = ler_mono_16k(str(caminho))
    clipe, est = preparar_clipe(x, sr)                  # padrões da v4: top_db=35, min_s=0,5, max_s=8
    v = vetor_de_clipe(model, clipe, aparelho, dtype)
    return float(nota(v, *pesos)), clipe, est


# --------------------------------------------------------------------------------------------- verificação
def resolver(p):
    """Igual a scripts/modelo_leve.py: radar_eval e ml_itw têm caminho relativo a data/raw/teste_internet."""
    p = str(p)
    for q in (ROOT / p, ROOT / "data" / "raw" / "teste_internet" / p):
        if q.exists():
            return q
    return ROOT / p


def verificar(n, variante, aparelho, model, dtype, pesos):
    import pandas as pd
    t3 = pd.read_parquet(NOTAS_T3, columns=["linha", "z_T3"]).set_index("linha").z_T3 if NOTAS_T3.exists() else None
    if (AMOSTRA / "amostra.parquet").exists():
        meta = pd.read_parquet(AMOSTRA / "amostra.parquet")
        notas = pd.read_parquet(AMOSTRA / "notas.parquet")
        assert (notas.linha.values == meta.linha.values).all()
        flat = np.load(AMOSTRA / "clipes.npy", mmap_mode="r")
        off = np.load(AMOSTRA / "offsets.npy")
        origem = "data/processed/modelo_leve_amostra/ (2.000 clipes da validação)"
    else:
        sp = pd.read_parquet(SPLITS)
        meta = sp[sp.split == "validacao"].reset_index(drop=True)
        notas, flat, off = None, None, None
        origem = "data/processed/splits_v7.parquet (validação)"
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(meta), size=min(n, len(meta)), replace=False))
    print(f"Verificação: {len(idx)} clipes de {origem}; variante {variante} em {aparelho} ({dtype}).")
    linhas = []
    for i in idx:
        r = meta.iloc[i]
        z, clipe, est = nota_de_arquivo(resolver(r.caminho), model, aparelho, dtype, pesos)
        lin = {"linha": int(r.linha), "fonte": r.get("fonte_av", r.get("fonte")), "label": int(r.label), "nota": z,
               "dur_entrada_s": est["dur_entrada_s"]}
        if flat is not None:
            guardado = np.asarray(flat[off[i]:off[i + 1]])
            lin["dif_clipe"] = (float(np.abs(guardado - clipe.numpy()).max()) if guardado.shape[0] == clipe.numel()
                                else float("inf"))
            col, _ = REF_AMOSTRA[variante]
            lin["ref_amostra"] = float(notas.iloc[i][col])
        if t3 is not None:
            lin["z_T3_gravada"] = float(t3.loc[int(r.linha)])
        linhas.append(lin)
    df = pd.DataFrame(linhas)
    print()
    ok = True
    if "dif_clipe" in df:
        dc = df.dif_clipe.max()
        print(f"Pré-processamento: maior diferença entre o clipe preparado agora e o guardado na amostra = {dc:.2e}"
              f" ({'igual' if dc == 0 else 'DIFERENTE'}).")
        d = (df.nota - df.ref_amostra).abs()
        col, rotulo = REF_AMOSTRA[variante]
        print(f"Nota x {rotulo} (coluna {col} de modelo_leve_amostra/notas.parquet): diferença máxima {d.max():.2e} logit, "
              f"mediana {d.median():.2e}.")
        if variante == "v3":
            print("  (a V3 gravada foi calculada em lotes de até 8 clipes com enchimento; o int8 dinâmico calcula a escala"
                  " de cada camada a partir do lote inteiro, então a nota da V3 também muda com o lote. Aqui o lote é 1.)")
        if variante == "v1":
            ok = bool(d.max() <= TOL_FP32)
            print(f"  -> critério fp32 (<= {TOL_FP32:g} logit): {'PASSOU' if ok else 'NÃO PASSOU'}")
    if "z_T3_gravada" in df:
        d3 = (df.nota - df.z_T3_gravada).abs()
        print(f"Nota x nota da T3 gravada (data/processed/v7_notas.parquet, z_T3): diferença máxima {d3.max():.2e} logit, "
              f"mediana {d3.median():.2e}.")
        if variante == "v3":
            print("  (informativo: na V3 a nota muda de valor; ver docs/MODELO_LEVE.md, seção 4.)")
        else:
            print("  (informativo: a T3 original foi pontuada a partir de vetores extraídos em fp16 na placa de vídeo e"
                  " guardados em fp16; por isso a diferença esperada aqui é da ordem de centésimos, não de milésimos.)")
        mesma = ((df.nota > corte_m()) == (df.z_T3_gravada > corte_m())).mean()
        print(f"  Mesma decisão pelo corte M ({corte_m():.3f}) que a T3 gravada: {100 * mesma:.0f}% dos clipes.")
    print()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))
    return ok


# --------------------------------------------------------------------------------------------- main
def main():
    for f in (sys.stdout, sys.stderr):            # console do Windows (cp1252): não quebrar com acentos
        try:
            f.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("audio", nargs="?", help="caminho do arquivo de áudio")
    ap.add_argument("--variante", choices=("v1", "v2", "v3"), default="v1")
    ap.add_argument("--corte", type=float, default=None, help="corte em logit (padrão: regra M da validação da T3)")
    ap.add_argument("--aparelho", choices=("cpu", "cuda"), default=None, help="só para a v1 (padrão: cpu)")
    ap.add_argument("--verificar", type=int, nargs="?", const=20, default=None, metavar="N",
                    help="compara N clipes (padrão 20) com as notas gravadas")
    a = ap.parse_args()
    if a.audio is None and a.verificar is None:
        ap.print_help()
        return 2
    aparelho = escolher_aparelho(a.variante, a.aparelho)
    t0 = time.perf_counter()
    model, dtype = carregar(a.variante, aparelho)
    pesos = pesos_t3()
    print(f"[modelo] variante {a.variante} carregada em {time.perf_counter() - t0:.1f} s ({aparelho}, {dtype}).")
    if a.verificar is not None:
        return 0 if verificar(a.verificar, a.variante, aparelho, model, dtype, pesos) else 1
    corte = corte_m() if a.corte is None else a.corte
    t1 = time.perf_counter()
    z, clipe, est = nota_de_arquivo(a.audio, model, aparelho, dtype, pesos)
    dt = time.perf_counter() - t1
    ia = z > corte
    print(f"arquivo: {a.audio}")
    print(f"duração original {est['dur_original_s']:.2f} s; usada {est['dur_entrada_s']:.2f} s "
          f"(corte de silêncio {'aplicado' if est['corte_aplicado'] else 'não aplicado'}; taxa do arquivo {est['sr_arquivo']:.0f} Hz)")
    print(f"nota (logit; maior = mais parecido com voz de IA): {z:.4f}")
    print(f"corte: {corte:.4f} ({'regra M da validação da T3' if a.corte is None else 'informado em --corte'})")
    print(f"classificação: {'PARECE GERADA POR IA' if ia else 'parece real'}  ({dt * 1000:.0f} ms para ler, preparar e pontuar)")
    if a.variante == "v3" and a.corte is None:
        print("ATENÇÃO: na v3 (int8) o valor da nota muda; o corte da T3 não foi validado para ela. Escolha um corte novo "
              "pela validação antes de usar esta classificação.")
    print("Isto é um alerta, não um veredito.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
