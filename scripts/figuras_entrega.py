#!/usr/bin/env python3
"""Catálogo de figuras da entrega final (paper e slides). Só LÊ results/*.csv|json já gravados; não treina, não baixa, não chama API.

Saídas em results/figuras_entrega/: slides/ (1920x1080, fonte >= 16 pt), paper/ (6,5 pol de largura, 300 dpi, fonte >= 8 pt),
INDEX.md e folha_resumo.png. Cada figura: fe_NN_nome.png. Todos os textos das figuras estão no dicionário TX (em português),
para poder traduzir depois. Nenhum nome de pessoa nem de máquina: só pessoa_id e ids de áudio.

    python scripts/figuras_entrega.py
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "figuras_entrega"
(OUT / "slides").mkdir(parents=True, exist_ok=True)
(OUT / "paper").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------- estilo (5 cores, uma por conceito)
C_DET = "#0072B2"     # detector de voz
C_CTX = "#E69F00"     # contexto (conteúdo)
C_REAL = "#009E73"    # áudio real
C_IA = "#D55E00"      # voz de IA
C_CONG = "#7F7F7F"    # congelado / não adotado / referência
C_GRID = "#DDDDDD"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "axes.facecolor": "white",
                     "figure.facecolor": "white", "axes.edgecolor": "#444444", "axes.grid": True, "grid.color": C_GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "hatch.linewidth": 0.8})

# ---------------------------------------------------------------------------------------------- textos (todos aqui)
TX = {
    "rot": {"auc": "AUC", "acc": "acurácia", "bacc": "acurácia balanceada", "tpr": "IA detectada (TPR)", "fpr": "real acusado (falso alarme)",
            "real": "real", "ia": "IA", "corte": "corte", "trivial": "acurácia trivial", "recall": "recall", "fa": "falso alarme",
            "meta": "meta", "n": "n"},
    "recortes": {"ml_itw": "internet real\n(ML-ITW)", "in_the_wild": "entrevistas\n(In-the-Wild)", "radar_en": "IA moderna,\ncanal degradado\n(RADAR en)"},
    "fonte_dados": "Fonte: ",
    "fe_01": {"titulo": "O sistema final junta duas perguntas, a voz e o conteúdo, num alerta de três níveis",
              "nota": "Congelado = pesos, cortes, prompt e modelo congelados antes do teste; exceto o idioma auto, fixado em 16/11 (antes só do conjunto reservado do WhatsApp). O alerta é um aviso, não um veredito.",
              "corte_aberto": "corte final do produto:\ndecisão em aberto", "idioma_aviso": "padrão do código ainda 'pt':\npassar auto explicitamente",
              "audio": "Áudio", "det": "Detector de voz", "det2": "XLS-R congelado\n(camadas 5, 6, 7)\n+ regressão T3\n→ nota",
              "corte": "Nota ≥\ncorte M?", "voz": "voz de IA?", "tr": "Transcrição", "tr2": "Whisper\n(idioma auto)",
              "jg": "Julgamento do conteúdo", "jg2": "modelo de linguagem\n→ risco: nenhum,\nbaixo, médio, alto",
              "ct": "conteúdo\nde golpe?\n(risco médio\nou alto)", "comb": "Combinação", "comb2": "alto = os dois sinais\natenção = um\nbaixo = nenhum",
              "alerta": "Alerta:\nbaixo · atenção · alto", "frase": "alerta, não veredito", "cong": "congelado"},
    "fe_02": {"titulo": "Do primeiro modelo ao sistema final: cada passo respondeu a uma limitação do anterior",
              "nota": "Linha do tempo em ordem de trabalho; as versões antigas foram medidas em testes diferentes (ver fe_03).",
              "marcos": [("v1 a v3", "CNN sobre o espectrograma; mais dados e padding de áudio curto não resolveram a internet real", C_CONG),
                         ("v4", "XLS-R congelado + regressão logística (F0): sistema anterior", C_DET),
                         ("v5 e v6", "outro modelo de fala, fusão e cabeças mais complexas: nada melhorou; voltamos ao F0", C_CONG),
                         ("prova externa", "F0 em áudio novo (RADAR, ML-ITW): AUC cai para {auc_radar} e {auc_itw}", C_IA),
                         ("v7", "parte dos bancos novos no treino: receita T3 = sistema principal", C_DET),
                         ("v8", "ajuste fino do XLS-R: ganho de {dv8} de AUC; não adotada", C_CONG),
                         ("modelo leve", "corte das camadas: {mb_ratio}× menor, nota idêntica", C_DET),
                         ("contexto", "Whisper + modelo de linguagem: recall {rec_txt}, falso alarme {fa_txt} em textos inventados", C_CTX),
                         ("testes finais", "WhatsApp ({acc_wa} de acurácia, um gerador); estresse: {n_det} de 3; contexto no WhatsApp: recall {rec_wa}", C_IA)]},
    "fe_03": {"titulo": "Cada versão foi medida num teste diferente; por isso mostramos um painel por teste, sem ligar as barras",
              "nota": "Painéis NÃO comparáveis entre si: bancos, métricas e cortes diferentes. v1 a v3 só têm medida em dados que já conhecíamos.",
              "p1": "Teste da v4\n(acurácia média, 3 bancos)", "p2": "Teste mestre da v7\n(AUC média, 3 fontes)", "p3": "Teste WhatsApp\n(AUC, 100 áudios)",
              "chute": "chute", "cnn": "CNN\nantiga (v3)", "v4a": "v4\ncorte A", "v4b": "v4\ncorte B", "f0": "F0", "t3": "T3", "r1": "R1\n(v8, não adotada)"},
    "fe_04": {"titulo": "Treinar com áudio de internet e de IA comercial com canal degradado (RADAR) subiu a AUC média de {m0} para {m3} no teste mestre",
              "nota": "IC95 por locutor (ML-ITW e In-the-Wild) e por clipe (RADAR en, intervalo otimista). O teste mestre não é virgem (o F0 já tinha sido medido nele). RADAR en mede idioma novo, não gerador novo.",
              "f0": "F0 (sem retreinar)", "t3": "T3"},
    "fe_05": {"titulo": "A validação escolheu a T3; no teste mestre, T3 e T4 empatam em média ({m3} e {m4})",
              "nota": "IC95 por locutor. A T3 foi escolhida pela validação, antes de abrir o teste. A T4 tem a pior fonte um pouco melhor.",
              "media": "AUC média (3 fontes)", "pior": "AUC da pior fonte", "esc": "escolhida"},
    "fe_06": {"titulo": "O ajuste fino da v8 subiu a AUC em {dv8}, mas o corte escolhido variou de {cmin} a {cmax}; não foi adotado",
              "nota": "2 sementes só; IC95 por locutor (ML-ITW, In-the-Wild) e por clipe (RADAR en, otimista). Cinza = não adotado. O corte é o da regra escolhida pela validação de cada rodada.",
              "auc": "AUC média (3 fontes)", "corte": "corte escolhido (logit)", "r0": "T3", "r1a": "R1\nsemente 42", "r1b": "R1\nsemente 123"},
    "fe_07": {"titulo": "Cortar as camadas sobrantes deixa o modelo {mb_ratio}× menor com a mesma nota; a AUC média praticamente não muda",
              "nota": "Tempos: mediana de 30 medidas, áudio de 5 s, lote de 1, notebook. Placa de vídeo (rótulo acima da barra): completo em fp16 e cortado em fp32 (precisões diferentes); V2 fp16 cortado: 10 ms. int8 muda o valor da nota (não usa o corte da T3). Validação, 2.000 clipes.",
              "mb": "tamanho em disco (MB)", "tempo": "tempo por áudio de 5 s (ms)", "auc": "AUC média na validação",
              "v": ["completo", "cortado\nfp32", "cortado\nfp16", "cortado\nint8"], "cpu": "processador, 4 threads", "gpu": "placa de vídeo", "prec": {"V0": "fp16", "V1": "fp32", "V2": "fp16"}},
    "fe_08": {"titulo": "No WhatsApp, a T3 acertou {acc} dos áudios, contra {accF0} do F0 no melhor corte",
              "nota": "Um só gerador, 23 pessoas, só inglês; IC95 por pessoa. Acurácia trivial = {triv}. O atalho de formato não foi descontado (fe_11).",
              "t3": "T3 (corte M)", "f0a": "F0 (corte A)", "f0b": "F0 (corte B)", "p1": "acertos", "p2": "erros por tipo de áudio"},
    "fe_09": {"titulo": "As notas da T3 separam os reais dos clones, e os 2 reais acima do corte são da mesma pessoa",
              "nota": "100 áudios de 23 pessoas, um só gerador. A separação pode vir também do formato do arquivo (fe_11).",
              "x": "nota da T3 (logit; maior = mais parecido com IA)", "y": "número de áudios", "corte": "corte M = {M}"},
    "fe_10": {"titulo": "Recodificar os clones em Opus (~16 kbps) quase não mudou a T3: acurácia {a} → {b}",
              "nota": "Os reais não mudam. A condição não responde a outras diferenças (conteúdo, duração, silêncio). IC95 por pessoa.",
              "A": "como veio", "B": "clones em Opus", "t3": "T3 (corte M)", "f0": "F0 (corte A)"},
    "fe_11": {"titulo": "Um classificador que nem ouve a voz chega a AUC {sem}; o 98% do detector não descarta o atalho de formato",
              "nota": "Metadados: duração, volume, silêncio, taxa original e largura de banda; validação deixando uma pessoa de fora. IC95 por pessoa.",
              "det": "detector T3", "detf0": "detector F0", "todos": "metadados\n(todos)", "sem": "metadados\n(sem taxa original)", "dur": "só duração", "ban": "só largura\nde banda"},
    "fe_12": {"titulo": "Teste de estresse (teste informal, 3 áudios): a T3 detectou {n} de 3 áudios de um gerador de família nova",
              "cortes_txt": "A', M e F5 quase coincidem\n({a} / {m} / {f5})",
              "nota": "Teste informal e não oficial: 3 áudios, sem protocolo prévio; não estima taxa de acerto. Cortes da T3: A', M e F5.",
              "x": "nota da T3 (logit)", "reais": "reais ({nr})", "clones": "clones ({ni})", "est": "estresse\n(3 áudios)", "cortes": "cortes"},
    "fe_13": {"titulo": "Com o mesmo corte, {br} dos reais do audiolivro em português são acusados, contra {co} na fala espontânea",
              "nota": "T3, corte M. Nenhum áudio em português entrou no treino. O corte não se transfere entre ambientes. IC95 (por clipe nos controles).",
              "y": "reais acusados como IA (falso alarme)", "pt": "português", "outro": "outros ambientes",
              "rec": {"asvspoof2019_la": "estúdio\n(ASVspoof 2019)", "asvspoof2021_df": "estúdio\n(ASVspoof 2021)", "llamapartialspoof": "falso parcial\n(Llama)",
                      "ml_itw": "internet real\n(ML-ITW)", "in_the_wild": "entrevistas\n(In-the-Wild)", "radar_en": "RADAR\n(en)",
                      "coraa": "fala espontânea\n(CORAA, pt)", "brspeech_df": "audiolivro\n(BRSpeech-DF, pt)"}},
    "fe_14": {"titulo": "Em textos inventados, o contexto pegou {rec} dos golpes e acusou {fa} das conversas legítimas",
              "nota": "130 textos escritos por nós (só texto, sem áudio; só pt-BR). Prompt ajustado em outro conjunto; teste uma vez. IC95. Linhas = metas fixadas antes.",
              "grupos": {"golpe_explicito": "golpe explícito", "golpe_escondido": "golpe escondido", "injecao_golpe": "golpe com ordens ao avaliador",
                         "legitimo_comum": "legítimo comum", "legitimo_dificil": "legítimo difícil", "injecao_legitimo": "legítimo com ordens ao avaliador"},
              "x1": "golpes com alerta (recall)", "x2": "legítimos com alerta (falso alarme)", "meta": "meta"},
    "fe_15": {"titulo": "Corrigir o idioma do Whisper não mudou o resultado: recall {a} → {b} no conjunto reservado",
              "nota": "Reservado (15 pessoas, 60 clipes; avaliado uma vez) e DEV (8 pessoas, 40 clipes; usado no diagnóstico). IC95 por pessoa. Só inglês.",
              "pt": "pt forçado (15/11, errado)", "auto": "auto (corrigido)", "p1": "Reservado", "p2": "DEV", "m": {"acc": "acurácia", "tpr": "recall", "fpr": "falso alarme"},
              "trivial": "trivial {t}"},
    "fe_16": {"titulo": "No banco WhatsApp, o contexto deu alerta em {a} dos clones que pedem dinheiro e em {b} dos reais com golpe encenado",
              "nota": "Alerta = risco médio ou alto, idioma auto, 100 clipes (descritivo; 40 usados no diagnóstico). Um só gerador; rótulo 'com golpe' vem da coleta.",
              "g": {"real_sem_golpe": "real,\nconversa normal", "real_com_golpe": "real,\ngolpe encenado", "ia_pedido_dinheiro": "IA,\npede dinheiro", "ia_neutro": "IA,\nneutro"},
              "y": "clipes com alerta (%)", "alto": "só alto", "medio": "médio ou alto"},
    "fe_17": {"titulo": "O contexto pegou {c} dos roteiros claros de golpe e {l} dos pedidos leves de dinheiro",
              "nota": "Análise descritiva por roteiro (clones, idioma auto); não usada para ajustar nada. Cada roteiro é tudo ou nada: mesmo texto, mesmo risco.",
              "classe": {"claras": "roteiros claros de golpe", "leves": "pedidos leves de dinheiro", "cotidiano": "conversa do dia a dia"},
              "x": "clipes com alerta médio+ (de n clipes do roteiro)", "r": "roteiro"},
    "fe_18": {"titulo": "Nenhuma voz de IA ficou em 'baixo' e nenhum real chegou a 'alto'",
              "nota": "Contagem de clipes, idioma auto, 100 clipes. Não é acurácia; um só gerador, 23 pessoas, rótulo 'com golpe' encenado. Alerta, não veredito.",
              "g": {"real_sem_golpe": "real, conversa normal", "real_com_golpe": "real, golpe encenado", "ia_pedido_dinheiro": "IA, pede dinheiro", "ia_neutro": "IA, neutro"},
              "n": ["baixo", "atenção", "alto"], "x": "nível do alerta combinado"},
    "fe_19": {"titulo": "O que a evidência sustenta e o que ainda não sabemos",
              "nota": "Não comparamos com outros sistemas; por isso não afirmamos nada sobre a literatura.",
              "sust": "O que sustenta", "nao": "O que ainda não sabemos",
              "s": ["T3: AUC média {m3} no teste mestre (IC95 {lo} a {hi}), escolhida pela validação",
                    "WhatsApp: {acc} de acurácia, {pes} de {pes} pessoas com a IA acima do real",
                    "Contexto: pega os roteiros claros ({c}); recall {rec} em textos inventados",
                    "Nenhuma IA em 'baixo' no combinado; correção do idioma documentada (errata)"],
              "n": ["Um só gerador no WhatsApp; só {pes} pessoas",
                    "Teste de estresse: {nd} de 3 (gerador de família nova escapou)",
                    "Só inglês: português não validado (corte acusa {br} dos reais no audiolivro)",
                    "Atalho de formato não descontado (AUC {sem} só com metadados)",
                    "Nada testado em ligação ao vivo; contexto: recall {rw} no WhatsApp",
                    "Sem comparação com outros sistemas"]},
    "fe_20": {"titulo": "Resumo: o detector separa bem em ambientes parecidos com o treino; o alerta combinado é só um aviso",
              "nota": "Ressalvas: teste mestre não virgem; WhatsApp = um gerador, 23 pessoas, só inglês; estresse informal (3 áudios); sem ligação ao vivo.",
              "k1": "T3 no teste mestre\n(AUC média)", "k2": "T3 no WhatsApp\n(acurácia)", "k3": "teste de estresse\n(gerador novo)", "k4": "combinado: IA em\n'baixo'",
              "s1": "IC95 {lo} a {hi}", "s2": "IC95 {lo} a {hi}; atalho de formato\nnão descontado", "s3": "informal; 3 áudios", "s4": "de {n} vozes de IA; reais em\n'alto': {ra}"},
}

# índice: nome -> (arquivo de dados, o que mostra, equivalente em results/slides, uso)
INDEX = {
    "fe_01_fluxo_sistema": ("(diagrama; sem dados)", "O caminho do áudio até o alerta: detector de voz, contexto, combinação.", "nenhum equivalente (v8_0_esquema é só da v8)", "ambos"),
    "fe_02_linha_do_tempo": ("v7_principal.csv, v8_principal.csv, modelo_leve_variantes.csv, teste_whatsapp_detector.csv, contexto_golpe_avaliacao_resumo.json, teste_3_ia_notas.csv, diagnostico_whisper_idioma_metricas.csv, ti_*", "Os marcos do projeto, uma frase por marco.", "nenhum", "ambos"),
    "fe_03_evolucao_versoes": ("v4final_09_acuracia_grupos.csv, v7_principal.csv, v8_principal.csv, teste_whatsapp_detector.csv", "Evolução do detector em painéis (um por teste, não comparáveis).", "parcial: 09_acuracia_grupos, f_01_antes_depois", "ambos"),
    "fe_04_f0_vs_t3_por_fonte": ("v7_teste_mestre.csv, v7_principal.csv", "AUC por fonte no teste mestre, F0 x T3, com IC.", "equivalente: v7_01_auc_por_fonte, f_01_antes_depois", "ambos (já existe parecida)"),
    "fe_05_receitas_t0_t4": ("v7_principal.csv", "Média e pior fonte de T0 a T4, T3 destacada.", "equivalente: v7_02_media_principal, f_03_receitas", "ambos (já existe parecida)"),
    "fe_06_t3_vs_v8": ("v8_principal.csv, v8_escolha.json", "T3 x R1: ganho pequeno e corte instável; não adotado.", "equivalente: v8_1_diferenca_vs_t3, v8_3_validacao_x_teste", "ambos"),
    "fe_07_modelo_leve": ("modelo_leve_variantes.csv, modelo_leve_latencia.csv, modelo_leve_equivalencia.csv", "Tamanho, tempo e AUC das variantes do modelo leve.", "equivalente: ml_01_tamanho_e_tempo, ml_02_equivalencia", "ambos (já existe parecida)"),
    "fe_08_whatsapp_detector": ("teste_whatsapp_detector.csv", "Acurácia, acurácia balanceada, TPR e FPR no WhatsApp, T3 x F0.", "parcial: tw_01 (traz o contexto errado)", "ambos"),
    "fe_09_whatsapp_notas": ("teste_whatsapp_notas.csv", "Notas da T3, reais x clones, com o corte M.", "equivalente: tw_03_histograma_notas", "ambos (já existe parecida)"),
    "fe_10_canal_opus": ("teste_whatsapp_detector.csv", "Condição A x B (clones em Opus): acurácia e AUC.", "nenhum", "paper"),
    "fe_11_atalho_metadados": ("teste_whatsapp_atalho.csv", "AUC do detector x classificador só de metadados.", "parcial: ti_05_atalho (outro banco)", "ambos"),
    "fe_12_estresse_3_ia": ("teste_3_ia_notas.csv, teste_whatsapp_notas.csv, v7_t3_modelo.json", "Onde caem os 3 áudios do estresse entre reais e clones.", "nenhum", "ambos"),
    "fe_13_portugues_corte": ("v7_teste_mestre.csv, v7_controles.csv", "Reais acusados com o mesmo corte em cada ambiente.", "equivalente: f_07_cortes_e_calibracao, v7_04_regras_corte", "ambos"),
    "fe_14_contexto_textos_inventados": ("contexto_golpe_avaliacao_resumo.json", "Recall e falso alarme do contexto por tipo de texto.", "equivalente: ctx_01_recall_e_falso_alarme", "ambos (já existe parecida)"),
    "fe_15_contexto_antes_depois_idioma": ("diagnostico_whisper_idioma_metricas.csv", "Contexto com idioma errado x corrigido, com IC.", "equivalente: ctxw_01_acuracia_antes_depois", "ambos (já existe parecida)"),
    "fe_16_contexto_por_grupo": ("diagnostico_whisper_combinado.csv", "Alerta do contexto por grupo no WhatsApp (corrigido).", "substitui tw_02 (contexto errado)", "ambos"),
    "fe_17_roteiros_claros_vs_leves": ("analise_erro_contexto_por_frase.csv", "Alerta por roteiro: claros 26/26, leves 0/20, dia a dia 0/10.", "nenhum", "ambos"),
    "fe_18_combinado_grupo_nivel": ("diagnostico_whisper_combinado.csv", "Grupo x nível combinado, com contagens.", "substitui tw_05 (versão de 15/11)", "ambos"),
    "fe_19_limitacoes": ("vários (números lidos dos CSV)", "O que a evidência sustenta e o que não sabemos.", "equivalente: f_09_o_que_nao_afirmamos", "ambos"),
    "fe_20_resumo_final": ("v7_principal.csv, teste_whatsapp_detector.csv, teste_3_ia_notas.csv, diagnostico_whisper_combinado.csv", "Quatro números-chave do sistema final e as ressalvas.", "nenhum", "slide (e paper, como figura 1)"),
}
JA_EXISTE = ("ml_01_tamanho_e_tempo, ml_02_equivalencia, v7_01 a v7_06, f_00 a f_09, ctx_01 a ctx_03, ctxw_01, tw_03_histograma_notas "
             "(não refeitas; equivalentes indicados na coluna acima)")


# ---------------------------------------------------------------------------------------------- utilidades
def f(v, d=2, sinal=False):
    s = f"{v:.{d}f}".replace(".", ",").replace("-", "−")
    return ("+" + s if sinal and v > 0 else s)


def p(v, d=0):
    return f(100 * v, d) + "%"


def fmt(txt, **k):
    return txt.format(**k)


def F(st, k):
    """tamanho de fonte: nunca abaixo do mínimo do modo (16 pt nos slides, 8 pt no paper)."""
    return max(st["fs"] * k, st["min"])


MODOS = {
    "slides": dict(w=19.2, h=10.8, dpi=100, fs=17, tfs=27, nfs=16, wrap=72, wrapn=130, xl=1.15, lw=3.0, ms=14, lm=0.07, rm=0.98, min=16, pt=0.45),
    "paper": dict(w=6.5, h=3.6, dpi=300, fs=8.5, tfs=10, nfs=8, wrap=64, wrapn=105, xl=0.55, lw=1.3, ms=5, lm=0.1, rm=0.98, min=8, pt=0.22),
}


def quadro(fig, st, titulo, nota, nr=1, nc=1, hr=None, wr=None, wspace=0.3, hspace=0.35, extra_top=0.0, left=None):
    W, H = fig.get_size_inches()
    tl = textwrap.wrap(titulo, st["wrap"])
    nl = textwrap.wrap(nota, st["wrapn"]) if nota else []
    th = len(tl) * st["tfs"] * 1.35 / 72 + 0.2 + extra_top + (st["pt"] if nr * nc > 1 else 0)
    nh = len(nl) * st["nfs"] * 1.35 / 72 + 0.12
    fig.text(0.012, 1 - 0.12 / H, "\n".join(tl), va="top", ha="left", fontsize=st["tfs"], fontweight="bold", color="#111111")
    if nl:
        fig.text(0.012, 0.1 / H, "\n".join(nl), va="bottom", ha="left", fontsize=st["nfs"], color="#444444")
    gs = dict(top=1 - th / H, bottom=(nh + st["xl"]) / H, left=left if left is not None else st["lm"], right=st["rm"], wspace=wspace, hspace=hspace)
    if hr:
        gs["height_ratios"] = hr
    if wr:
        gs["width_ratios"] = wr
    axs = fig.subplots(nr, nc, squeeze=False, gridspec_kw=gs)
    for a in axs.ravel():
        a.tick_params(labelsize=st["fs"], length=st["lw"] * 1.2)
    return axs


def barras(ax, x, v, lo=None, hi=None, cores=None, st=None, w=0.6, hatch=None, lab=None, fmt_=None, horiz=False, ymax=None):
    v = np.asarray(v, float)
    cores = cores if cores is not None else [C_DET] * len(v)
    err = None
    if lo is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        err = [np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)]
    hs = hatch if hatch is not None else [None] * len(v)
    for i in range(len(v)):
        kw = dict(color=cores[i], edgecolor="#222222", linewidth=0.8, hatch=hs[i])
        if horiz:
            ax.barh(x[i], v[i], height=w, **kw)
        else:
            ax.bar(x[i], v[i], width=w, **kw)
    if err is not None:
        if horiz:
            ax.errorbar(v, x, xerr=err, fmt="none", ecolor="#111111", capsize=st["lw"] * 1.2, elinewidth=st["lw"] * 0.6)
        else:
            ax.errorbar(x, v, yerr=err, fmt="none", ecolor="#111111", capsize=st["lw"] * 1.2, elinewidth=st["lw"] * 0.6)
    if lab is not None:
        for i in range(len(v)):
            top = (hi[i] if hi is not None else v[i])
            if horiz:
                ax.text(top + (ymax or 1) * 0.015, x[i], lab[i], va="center", ha="left", fontsize=st["fs"], fontweight="bold")
            else:
                ax.text(x[i], top + (ymax or 1) * 0.015, lab[i], ha="center", va="bottom", fontsize=st["fs"], fontweight="bold")


def legpatch(ax, color=None, hatch=None, edgecolor="#222222", label=None):
    """entrada de legenda (retângulo de tamanho zero) com a mesma cor e hachura das barras."""
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=color, hatch=hatch, edgecolor=edgecolor, label=label))


def ic3(row, col):
    return row[col], row[col + "_lo"], row[col + "_hi"]


# ---------------------------------------------------------------------------------------------- dados
class D:
    def __init__(self):
        rd = lambda n: pd.read_csv(R / n)  # noqa: E731
        self.tm, self.ctl = rd("v7_teste_mestre.csv"), rd("v7_controles.csv")
        self.pr7, self.pr8 = rd("v7_principal.csv").set_index("receita"), rd("v8_principal.csv").set_index("candidato")
        self.esc8 = json.loads((R / "v8_escolha.json").read_text(encoding="utf-8"))
        self.var, self.lat, self.eq = rd("modelo_leve_variantes.csv").set_index("variante"), rd("modelo_leve_latencia.csv"), rd("modelo_leve_equivalencia.csv")
        self.det, self.ata, self.notas = rd("teste_whatsapp_detector.csv"), rd("teste_whatsapp_atalho.csv").set_index("metadados"), rd("teste_whatsapp_notas.csv")
        self.par = rd("teste_whatsapp_pareado.csv")
        self.est = rd("teste_3_ia_notas.csv")
        self.dmet, self.dcomb, self.frase = rd("diagnostico_whisper_idioma_metricas.csv"), rd("diagnostico_whisper_combinado.csv"), rd("analise_erro_contexto_por_frase.csv")
        self.aval = json.loads((R / "contexto_golpe_avaliacao_resumo.json").read_text(encoding="utf-8"))
        self.t3m = json.loads((R / "v7_t3_modelo.json").read_text(encoding="utf-8"))
        self.v4 = rd("v4final_09_acuracia_grupos.csv").set_index("série")
        ti = rd("teste_internet_por_fonte.csv").set_index("fonte")
        self.f0ext = {"radar": float(ti.loc["radar_eval", "AUC"]), "itw": float(ti.loc["ml_itw", "AUC"])}
        self.cortes = self.t3m["cortes_da_validacao_logit"]
        self.M = self.cortes["M"]

    def d(self, modelo, regra, cond):
        x = self.det[(self.det.modelo == modelo) & (self.det.corte_regra == regra) & (self.det.condicao == cond)]
        assert len(x) == 1
        return x.iloc[0]

    def dm(self, conj, versao, rec="todos"):
        x = self.dmet[(self.dmet.conjunto.str.startswith(conj)) & (self.dmet.versao.str.startswith(versao)) & (self.dmet.recorte == rec)]
        assert len(x) == 1
        return x.iloc[0]

    def comb(self):
        return self.dcomb[(self.dcomb.tabela == "combinado_grupo_x_nivel") & (self.dcomb.conjunto == "todos")].set_index("grupo")

    def ctxg(self):
        return self.dcomb[(self.dcomb.tabela == "contexto_por_grupo") & (self.dcomb.conjunto == "todos")].set_index("grupo")


GRUPOS = ["real_sem_golpe", "real_com_golpe", "ia_pedido_dinheiro", "ia_neutro"]


# ---------------------------------------------------------------------------------------------- figuras
def caixa(ax, x, y, w, h, txt, cor, st, cong=False, fs=None, tc="#111111", bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=cor, lw=st["lw"] * (1.0 if not cong else 0.9),
                       ls="--" if cong else "-")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs or F(st, 0.92), color=tc, fontweight="bold" if bold else "normal")


def seta(ax, x0, y0, x1, y1, st, cor="#333333"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=cor, lw=st["lw"] * 0.7))


def fe_01(fig, st, d):
    t = TX["fe_01"]
    ax = quadro(fig, st, t["titulo"], t["nota"], left=0.02)[0, 0]
    ax.axis("off")
    ax.grid(False)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1)
    k = F(st, 1.0)
    NN = "\n\n"
    caixa(ax, 0.0, 0.38, 0.09, 0.24, t["audio"], C_CONG, st, bold=True, fs=k)
    caixa(ax, 0.13, 0.62, 0.23, 0.34, t["det"] + NN + t["det2"], C_DET, st, cong=True, fs=k)
    caixa(ax, 0.41, 0.62, 0.14, 0.34, t["corte"], C_DET, st, cong=True, fs=k)
    caixa(ax, 0.13, 0.06, 0.15, 0.34, t["tr"] + NN + t["tr2"], C_CTX, st, cong=True, fs=k)
    caixa(ax, 0.32, 0.06, 0.23, 0.34, t["jg"] + NN + t["jg2"], C_CTX, st, cong=True, fs=k)
    seta(ax, 0.09, 0.54, 0.13, 0.79, st)
    seta(ax, 0.09, 0.46, 0.13, 0.23, st)
    seta(ax, 0.36, 0.79, 0.41, 0.79, st)
    seta(ax, 0.28, 0.23, 0.32, 0.23, st)
    caixa(ax, 0.60, 0.62, 0.13, 0.34, t["voz"], C_DET, st, fs=k)
    caixa(ax, 0.60, 0.06, 0.13, 0.34, t["ct"], C_CTX, st, fs=k)
    seta(ax, 0.55, 0.79, 0.60, 0.79, st)
    seta(ax, 0.55, 0.23, 0.60, 0.23, st)
    caixa(ax, 0.79, 0.34, 0.21, 0.32, t["comb"] + NN + t["comb2"], C_CONG, st, fs=k)
    seta(ax, 0.73, 0.79, 0.85, 0.66, st)
    seta(ax, 0.73, 0.2, 0.83, 0.34, st)
    ax.text(0.99, 0.28, t["alerta"], ha="right", va="top", fontsize=k, fontweight="bold")
    ax.text(0.99, 0.08, "“" + t["frase"] + "”", ha="right", va="top", fontsize=k, style="italic", color="#444444")
    ax.text(0.0, 0.0, "- - -  " + t["cong"], fontsize=k, color="#444444", va="bottom")
    ax.text(0.48, 0.60, t["corte_aberto"], ha="center", va="top", fontsize=F(st, 0.85), style="italic", color=C_DET)
    ax.text(0.215, 0.42, t["idioma_aviso"], ha="center", va="bottom", fontsize=F(st, 0.85), style="italic", color="#8a5a00")


def fe_02(fig, st, d):
    t = TX["fe_02"]
    pr7, v8 = d.pr7, d.pr8
    rec = d.aval["positivo_medio_ou_alto"]
    wa = d.d("T3", "M", "A")
    cc = ctx100(d)
    est_det = int(d.est[(d.est.condicao == "original")].passa_M.sum())
    var = d.var
    val = dict(auc_radar=f(d.f0ext["radar"], 2), auc_itw=f(d.f0ext["itw"], 2),
               dv8=f(float(v8.loc["R1 (média das seeds)", "média AUC (3 fontes)"] - v8.loc["R0", "média AUC (3 fontes)"]), 3),
               mb_ratio=f(var.loc["V0", "mb_total"] / var.loc["V1", "mb_total"], 1), rec_txt=p(rec["recall_golpes_todos"]["taxa"]),
               fa_txt=p(rec["falso_alarme_legitimos_todos"]["taxa"]), acc_wa=p(wa.acc), n_det=est_det, rec_wa=p(cc["recall"]))
    ax = quadro(fig, st, t["titulo"], t["nota"])[0, 0]
    ax.axis("off")
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(t["marcos"])
    ax.plot([0.12, 0.12], [0.02, 0.98], color="#333333", lw=st["lw"] * 0.6, zorder=1)
    for i, (rot, txt, cor) in enumerate(t["marcos"]):
        y = 0.96 - i * (0.94 / (n - 1))
        ax.plot(0.12, y, "o", ms=st["ms"], color=cor, mec="#222222", zorder=3)
        ax.text(0.10, y, rot, ha="right", va="center", fontsize=st["fs"], fontweight="bold")
        ax.text(0.15, y, textwrap.fill(txt.format(**val), 78 if st is MODOS["slides"] else 70), ha="left", va="center", fontsize=F(st, 0.95))


def ctx100(d):
    t = d.ctxg()
    pos, neg = ["real_com_golpe", "ia_pedido_dinheiro"], ["real_sem_golpe", "ia_neutro"]
    npos, nneg = int(t.loc[pos, "n"].sum()), int(t.loc[neg, "n"].sum())
    tp, fp = int(t.loc[pos, "alerta_medio_mais"].sum()), int(t.loc[neg, "alerta_medio_mais"].sum())
    return dict(recall=tp / npos, fa=fp / nneg, acc=(tp + nneg - fp) / (npos + nneg))


def fe_03(fig, st, d):
    t = TX["fe_03"]
    ax = quadro(fig, st, t["titulo"], t["nota"], 1, 3, wspace=0.35, wr=[1.15, 1, 1.05])
    a1, a2, a3 = ax[0]
    v4 = d.v4["média dos 3"]
    ch = float(v4["chute da classe comum"])
    xs = [0, 1, 2, 3]
    vals = [ch, v4["CNN antiga (v3), corte A"], v4["sistema v4, corte A"], v4["sistema v4, corte B"]]
    barras(a1, xs, vals, cores=[C_CONG, C_CONG, C_DET, C_DET], hatch=["//", "//", None, None], st=st, lab=[p(x) for x in vals], ymax=1)
    a1.set_xticks(xs, [t["chute"], t["cnn"], t["v4a"], t["v4b"]], fontsize=F(st, 0.8))
    a1.set_ylim(0, 1.12)
    a1.set_title(t["p1"], fontsize=st["fs"], fontweight="bold", loc="left")
    r = d.pr7
    xs = [0, 1]
    vals = [r.loc["T0", "média AUC (3 fontes)"], r.loc["T3", "média AUC (3 fontes)"]]
    lo = [r.loc["T0", "média IC2.5"], r.loc["T3", "média IC2.5"]]
    hi = [r.loc["T0", "média IC97.5"], r.loc["T3", "média IC97.5"]]
    v8 = d.pr8.loc["R1 (média das seeds)"]
    xs, vals, lo, hi = [0, 1, 2], vals + [v8["média AUC (3 fontes)"]], lo + [v8["média IC2.5"]], hi + [v8["média IC97.5"]]
    barras(a2, xs, vals, lo, hi, cores=[C_CONG, C_DET, C_CONG], hatch=["", "", "//"], st=st, lab=[f(x, 3) for x in vals], ymax=1)
    a2.set_xticks(xs, [t["f0"], t["t3"], t["r1"]], fontsize=F(st, 0.8))
    a2.set_ylim(0.5, 1.1)
    a2.set_title(t["p2"], fontsize=st["fs"], fontweight="bold", loc="left")
    x1, x2 = d.d("F0", "A", "A"), d.d("T3", "M", "A")
    vals = [x1.auc, x2.auc]
    barras(a3, [0, 1], vals, [x1.auc_lo, x2.auc_lo], [x1.auc_hi, x2.auc_hi], cores=[C_CONG, C_DET], st=st, lab=[f(x, 3) for x in vals], ymax=1)
    a3.set_xticks([0, 1], [t["f0"], t["t3"]], fontsize=F(st, 0.8))
    a3.set_ylim(0.5, 1.1)
    a3.set_title(t["p3"], fontsize=st["fs"], fontweight="bold", loc="left")


def fe_04(fig, st, d):
    t = TX["fe_04"]
    m0, m3 = d.pr7.loc["T0", "média AUC (3 fontes)"], d.pr7.loc["T3", "média AUC (3 fontes)"]
    ax = quadro(fig, st, fmt(t["titulo"], m0=f(m0, 2), m3=f(m3, 2)), t["nota"])[0, 0]
    recs = ["ml_itw", "in_the_wild", "radar_en"]
    tm = d.tm[d.tm.papel == "principal"]
    xs = np.arange(len(recs))
    w = 0.36
    for j, (rc, cor, rot, hh) in enumerate((("T0", C_CONG, t["f0"], "//"), ("T3", C_DET, t["t3"], None))):
        x = tm[tm.receita == rc].set_index("recorte").loc[recs]
        barras(ax, xs + (j - 0.5) * w, x["AUC"], x["AUC IC2.5"], x["AUC IC97.5"], cores=[cor] * 3, hatch=[hh] * 3, st=st, w=w * 0.95,
               lab=[f(v, 2) for v in x["AUC"]], ymax=1)
        legpatch(ax, color=cor, hatch=hh, label=rot, edgecolor="#222222")
    ax.set_xticks(xs, [TX["recortes"][r] for r in recs], fontsize=F(st, 0.9))
    ax.set_ylim(0.4, 1.12)
    ax.set_ylabel(TX["rot"]["auc"], fontsize=st["fs"])
    ax.set_ylim(0.4, 1.28)
    ax.legend(fontsize=st["fs"], loc="upper left", frameon=False, ncol=2)


def fe_05(fig, st, d):
    t = TX["fe_05"]
    r = d.pr7
    ax = quadro(fig, st, fmt(t["titulo"], m3=f(r.loc["T3", "média AUC (3 fontes)"], 3), m4=f(r.loc["T4", "média AUC (3 fontes)"], 3)), t["nota"], 1, 2, wspace=0.12)[0]
    rec = ["T0", "T1", "T2", "T3", "T4"]
    venc = r.index[r["vencedora pela validação"]][0]
    for a, (col, lo, hi, tit) in zip(ax, (("média AUC (3 fontes)", "média IC2.5", "média IC97.5", t["media"]), ("AUC pior fonte", "pior IC2.5", "pior IC97.5", t["pior"]))):
        cores = [C_DET if k == venc else C_CONG for k in rec]
        barras(a, np.arange(5), r.loc[rec, col], r.loc[rec, lo], r.loc[rec, hi], cores=cores, hatch=[None if k == venc else "//" for k in rec], st=st,
               lab=[f(v, 2) for v in r.loc[rec, col]], ymax=1)
        a.set_xticks(np.arange(5), [k + (f"\n({t['esc']})" if k == venc else "") for k in rec], fontsize=F(st, 0.85))
        a.set_ylim(0.4, 1.12)
        a.set_title(tit, fontsize=st["fs"], fontweight="bold", loc="left")


def fe_06(fig, st, d):
    t = TX["fe_06"]
    c = ["R0", "R1_s42", "R1_s123"]
    pr = d.pr8
    cut = [d.esc8["por_candidato"][k]["cortes"][d.esc8["por_candidato"][k]["regra"]] for k in c]
    dv8 = pr.loc["R1 (média das seeds)", "média AUC (3 fontes)"] - pr.loc["R0", "média AUC (3 fontes)"]
    ax = quadro(fig, st, fmt(t["titulo"], dv8=f(dv8, 3), cmin=f(min(cut), 2), cmax=f(max(cut), 2)), t["nota"], 1, 2, wspace=0.3)[0]
    cores = [C_DET, C_CONG, C_CONG]
    hs = [None, "//", "//"]
    barras(ax[0], np.arange(3), pr.loc[c, "média AUC (3 fontes)"], pr.loc[c, "média IC2.5"], pr.loc[c, "média IC97.5"], cores=cores, hatch=hs, st=st,
           lab=[f(v, 3) for v in pr.loc[c, "média AUC (3 fontes)"]], ymax=1)
    ax[0].set_ylim(0.9, 1.02)
    ax[0].set_title(t["auc"], fontsize=st["fs"], fontweight="bold", loc="left")
    barras(ax[1], np.arange(3), cut, cores=cores, hatch=hs, st=st, lab=[f(v, 2) for v in cut], ymax=1)
    ax[1].axhline(0, color="#222222", lw=1)
    ax[1].set_ylim(min(cut) - 0.15, max(cut) + 0.25)
    ax[1].set_title(t["corte"], fontsize=st["fs"], fontweight="bold", loc="left")
    for a in ax:
        a.set_xticks(np.arange(3), [t["r0"], t["r1a"], t["r1b"]], fontsize=F(st, 0.85))


def fe_07(fig, st, d):
    t = TX["fe_07"]
    v = d.var
    ratio = v.loc["V0", "mb_total"] / v.loc["V1", "mb_total"]
    ax = quadro(fig, st, fmt(t["titulo"], mb_ratio=f(ratio, 1)), t["nota"], 1, 3, wspace=0.38)[0]
    nomes = ["V0", "V1", "V2", "V3"]
    cor = [C_CONG, C_DET, C_DET, C_DET]
    hs = ["//", None, None, None]
    mb = [v.loc[k, "mb_total"] for k in nomes]
    barras(ax[0], np.arange(4), mb, cores=cor, hatch=hs, st=st, lab=[f(x, 0) for x in mb], ymax=max(mb))
    ax[0].set_title(t["mb"], fontsize=st["fs"], fontweight="bold", loc="left")
    ax[0].set_ylim(0, max(mb) * 1.15)
    L = d.lat[(d.lat.parte == "total") & (d.lat.duracao_s == 5)]
    g = lambda var, ap: L[(L.variante == var) & (L.aparelho.str.startswith(ap))].mediana_ms  # noqa: E731
    xs, vals, cs = [], [], []
    for i, k in enumerate(nomes):
        c4 = g(k, "processador do notebook, 4")
        if len(c4):
            xs.append(i - 0.2), vals.append(float(c4.iloc[0])), cs.append("cpu")
        gp = g(k, "placa")
        if len(gp):
            xs.append(i + 0.2), vals.append(float(gp.iloc[0])), cs.append("gpu:" + k)
    for x, vv, c_ in zip(xs, vals, cs):
        ax[1].bar(x, vv, width=0.38, color=C_DET if c_ == "cpu" else C_CTX, hatch=None if c_ == "cpu" else "//", edgecolor="#222222")
        ax[1].text(x, vv * 1.02, f(vv, 0), ha="center", va="bottom", fontsize=F(st, 0.8), fontweight="bold")
        if c_ != "cpu":  # precisão de cada barra da placa de vídeo (não são comparáveis entre si)
            ax[1].text(x, vv * 1.02 + max(vals) * 0.045, t["prec"][c_[4:]], ha="center", va="bottom", fontsize=F(st, 0.7), color="#8a5a00", fontweight="bold")
    legpatch(ax[1], color=C_DET, label=t["cpu"], edgecolor="#222222")
    legpatch(ax[1], color=C_CTX, hatch="//", label=t["gpu"], edgecolor="#222222")
    ax[1].legend(fontsize=F(st, 0.8), frameon=False, loc="upper right")
    ax[1].set_ylim(0, max(vals) * 1.2)
    ax[1].set_title(t["tempo"], fontsize=st["fs"], fontweight="bold", loc="left")
    e = d.eq[d.eq.fonte == "(média das 6 fontes)"].set_index("comparacao")
    a0 = float(e.loc["V1 fp32 x V0 fp32", "auc_media_ref"])
    aucs = [a0] + [float(e.loc[k, "auc_media_var"]) for k in ("V1 fp32 x V0 fp32", "V2 fp16 x V0 fp16", "V3 int8 x V0 fp32")]
    barras(ax[2], np.arange(4), aucs, cores=cor, hatch=hs, st=st, lab=[f(x, 4) for x in aucs], ymax=1)
    ax[2].set_ylim(0.9, 1.0)
    ax[2].set_title(t["auc"], fontsize=st["fs"], fontweight="bold", loc="left")
    for a in ax:
        a.set_xticks(np.arange(4), t["v"], fontsize=F(st, 0.75))


def fe_08(fig, st, d):
    t = TX["fe_08"]
    t3, fa, fb = d.d("T3", "M", "A"), d.d("F0", "A", "A"), d.d("F0", "B", "A")
    melhor = max(fa.acc, fb.acc)
    ax = quadro(fig, st, fmt(t["titulo"], acc=p(t3.acc), accF0=p(melhor)), fmt(t["nota"], triv=p(t3.acc_trivial)), 1, 2, wspace=0.22, wr=[1, 1.2])[0]
    mods = [(t["t3"], t3, C_DET, None), (t["f0a"], fa, C_CONG, "//"), (t["f0b"], fb, C_CONG, "xx")]
    w = 0.26
    for i, (m, k) in enumerate((("acc", TX["rot"]["acc"]), ("bacc", TX["rot"]["bacc"]))):
        for j, (rot, x, cor, h) in enumerate(mods):
            v, lo, hi = ic3(x, m)
            barras(ax[0], [i + (j - 1) * w], [v], [lo], [hi], cores=[cor], hatch=[h], st=st, w=w * 0.95, lab=[p(v)], ymax=1)
    for j, (rot, x, cor, h) in enumerate(mods):
        legpatch(ax[0], color=cor, hatch=h, edgecolor="#222222", label=rot)
    ax[0].axhline(t3.acc_trivial, color="#222222", ls=":", lw=st["lw"] * 0.5)
    ax[0].set_xticks([0, 1], [TX["rot"]["acc"], TX["rot"]["bacc"]], fontsize=F(st, 0.9))
    ax[0].set_ylim(0, 1.2)
    ax[1].legend(handles=ax[0].patches[-3:] + [Line2D([], [], color="#222222", ls=":", lw=st["lw"] * 0.6, label=TX["rot"]["trivial"] + " " + p(t3.acc_trivial))],
                 fontsize=F(st, 0.8), frameon=False, loc="upper right")
    ax[0].set_title(t["p1"], fontsize=st["fs"], fontweight="bold", loc="left")
    for i, (m, k) in enumerate((("tpr", TX["rot"]["tpr"]), ("fpr", TX["rot"]["fpr"]))):
        for j, (rot, x, cor, h) in enumerate(mods):
            v, lo, hi = ic3(x, m)
            barras(ax[1], [i + (j - 1) * w], [v], [lo], [hi], cores=[cor], hatch=[h], st=st, w=w * 0.95, lab=[p(v)], ymax=1)
    ax[1].set_xticks([0, 1], [TX["rot"]["tpr"].replace(" (", "\n("), TX["rot"]["fpr"].replace(" (", "\n(")], fontsize=F(st, 0.85))
    ax[1].set_ylim(0, 1.2)
    ax[1].set_title(t["p2"], fontsize=st["fs"], fontweight="bold", loc="left")


def fe_09(fig, st, d):
    t = TX["fe_09"]
    ax = quadro(fig, st, t["titulo"], t["nota"])[0, 0]
    n = d.notas
    bins = np.linspace(n.z_T3_A.min() - 0.5, n.z_T3_A.max() + 0.5, 28)
    ax.hist(n[n.autenticidade == "real"].z_T3_A, bins=bins, color=C_REAL, alpha=0.95, edgecolor="#222222", label=f"{TX['rot']['real']} ({(n.autenticidade == 'real').sum()})")
    ax.hist(n[n.autenticidade == "ia"].z_T3_A, bins=bins, color=C_IA, alpha=0.85, edgecolor="#222222", hatch="//", label=f"{TX['rot']['ia']} ({(n.autenticidade == 'ia').sum()})")
    ax.axvline(d.M, color="#111111", lw=st["lw"], ls="--")
    ax.text(d.M, ax.get_ylim()[1] * 0.95, " " + fmt(t["corte"], M=f(d.M, 2)), fontsize=st["fs"], va="top")
    ax.set_xlabel(t["x"], fontsize=st["fs"])
    ax.set_ylabel(t["y"], fontsize=st["fs"])
    ax.legend(fontsize=st["fs"], frameon=False, loc="upper right", bbox_to_anchor=(1, 0.8))


def fe_10(fig, st, d):
    t = TX["fe_10"]
    a, b = d.d("T3", "M", "A"), d.d("T3", "M", "B")
    ax = quadro(fig, st, fmt(t["titulo"], a=p(a.acc), b=p(b.acc)), t["nota"], 1, 2, wspace=0.25)[0]
    mods = [(t["t3"], "T3", "M", C_DET, None), (t["f0"], "F0", "A", C_CONG, "//")]
    for k, (m, tit) in enumerate((("acc", TX["rot"]["acc"]), ("auc", TX["rot"]["auc"]))):
        for j, (rot, mo, rg, cor, h) in enumerate(mods):
            for i, cond in enumerate(("A", "B")):
                x = d.d(mo, rg, cond)
                v, lo, hi = ic3(x, m)
                xx = i + (j - 0.5) * 0.34
                barras(ax[k], [xx], [v], [lo], [hi], cores=[cor], hatch=[h], st=st, w=0.32, lab=[p(v) if m == "acc" else f(v, 3)], ymax=1)
        ax[k].set_xticks([0, 1], [t["A"], t["B"]], fontsize=st["fs"])
        ax[k].set_ylim(0.5, 1.3 if k == 0 else 1.12)
        ax[k].set_title(tit, fontsize=st["fs"], fontweight="bold", loc="left")
    for rot, mo, rg, cor, h in mods:
        legpatch(ax[0], color=cor, hatch=h, edgecolor="#222222", label=rot)
    ax[0].legend(fontsize=F(st, 0.8), frameon=False, loc="upper left", ncol=2)


def fe_11(fig, st, d):
    t = TX["fe_11"]
    a = d.ata
    sem = a.loc["sem a taxa original", "AUC_loo"]
    ax = quadro(fig, st, fmt(t["titulo"], sem=f(sem, 3)), t["nota"])[0, 0]
    itens = [(t["det"], "DETECTOR T3 (condicao A)", C_DET, None), (t["detf0"], "DETECTOR F0 (condicao A)", C_CONG, "//"),
             (t["todos"], "todos os metadados", C_CTX, None), (t["sem"], "sem a taxa original", C_CTX, None),
             (t["dur"], "so duracao", C_CTX, None), (t["ban"], "so banda95", C_CTX, None)]
    xs = np.arange(len(itens))
    v = [a.loc[k, "AUC_loo"] for _, k, _, _ in itens]
    lo = [a.loc[k, "auc_lo"] for _, k, _, _ in itens]
    hi = [a.loc[k, "auc_hi"] for _, k, _, _ in itens]
    barras(ax, xs, v, lo, hi, cores=[c for *_, c, _ in itens], hatch=[h for *_, h in itens], st=st, lab=[f(x, 3) for x in v], ymax=1)
    ax.set_xticks(xs, [i[0] for i in itens], fontsize=F(st, 0.78))
    ax.axhline(0.5, color="#222222", ls=":", lw=1)
    ax.set_ylim(0.4, 1.12)
    ax.set_ylabel(TX["rot"]["auc"], fontsize=st["fs"])


def fe_12(fig, st, d):
    t = TX["fe_12"]
    est = d.est[d.est.condicao == "original"].set_index("id")
    n = int(est.passa_M.sum())
    ax = quadro(fig, st, fmt(t["titulo"], n=n), t["nota"], left=0.17 if st["min"] == 16 else 0.2)[0, 0]
    nt = d.notas
    rng = np.random.default_rng(0)
    nr, ni = int((nt.autenticidade == "real").sum()), int((nt.autenticidade == "ia").sum())
    for y, key, cor, h in ((2, "real", C_REAL, "o"), (1, "ia", C_IA, "s")):
        x = nt[nt.autenticidade == key].z_T3_A.values
        ax.scatter(x, y + rng.uniform(-0.17, 0.17, len(x)), s=st["ms"] ** 2 * 0.35, color=cor, alpha=0.8, marker=h, edgecolor="#222222", linewidth=0.4)
    for i, (idn, r) in enumerate(est.iterrows()):
        ax.scatter([r.nota_T3], [0], s=st["ms"] ** 2 * 1.5, color="white", edgecolor="#111111", linewidth=st["lw"] * 0.8, marker="D", zorder=5)
        ax.annotate(idn, (r.nota_T3, 0), xytext=(0, 14 if st is MODOS["paper"] else 24), textcoords="offset points", ha="center", fontsize=F(st, 0.85), fontweight="bold")
    for k, ls in (("A'", ":"), ("M", "--"), ("F5", "-.")):
        ax.axvline(d.cortes[k], color="#111111", lw=st["lw"] * 0.6, ls=ls)
    ax.text(d.cortes["M"] + 0.3, 2.62, fmt(t["cortes_txt"], a=f(d.cortes["A'"], 3), m=f(d.cortes["M"], 3), f5=f(d.cortes["F5"], 3)),
            fontsize=F(st, 0.85), va="center", ha="left")
    ax.set_yticks([2, 1, 0], [fmt(t["reais"], nr=nr), fmt(t["clones"], ni=ni), t["est"]], fontsize=st["fs"])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel(t["x"], fontsize=st["fs"])
    ax.grid(axis="y", visible=False)


def fe_13(fig, st, d):
    t = TX["fe_13"]
    tm = d.tm[(d.tm.receita == "T3")].drop_duplicates("recorte").set_index("recorte")
    ordem = ["asvspoof2019_la", "asvspoof2021_df", "llamapartialspoof", "ml_itw", "in_the_wild", "radar_en", "coraa", "brspeech_df"]
    pt = {"coraa", "brspeech_df"}
    v = [tm.loc[k, "FPR M"] for k in ordem]
    lo = [tm.loc[k, "FPR M IC2.5"] for k in ordem]
    hi = [tm.loc[k, "FPR M IC97.5"] for k in ordem]
    ax = quadro(fig, st, fmt(t["titulo"], br=p(tm.loc["brspeech_df", "FPR M"]), co=p(tm.loc["coraa", "FPR M"])), t["nota"], left=0.3 if st["min"] == 16 else 0.33)[0, 0]
    ys = np.arange(len(ordem))[::-1]
    barras(ax, ys, v, lo, hi, cores=[C_IA if k in pt else C_CONG for k in ordem], hatch=[None if k in pt else "//" for k in ordem], st=st, w=0.62,
           lab=[p(x) for x in v], horiz=True, ymax=1)
    ax.set_yticks(ys, [t["rec"][k].replace("\n", " ") for k in ordem], fontsize=F(st, 0.9))
    ax.set_xlim(0, 0.95)
    ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", "25", "50", "75%"])
    ax.set_xlabel(t["y"], fontsize=st["fs"])
    ax.grid(axis="y", visible=False)
    legpatch(ax, color=C_IA, edgecolor="#222222", label=t["pt"])
    legpatch(ax, color=C_CONG, hatch="//", edgecolor="#222222", label=t["outro"])
    ax.legend(fontsize=F(st, 0.9), frameon=False, loc="center right")


def fe_14(fig, st, d):
    t = TX["fe_14"]
    p_ = d.aval["positivo_medio_ou_alto"]
    pg = p_["por_grupo"]
    rec, fa = p_["recall_golpes_todos"], p_["falso_alarme_legitimos_todos"]
    ax = quadro(fig, st, fmt(t["titulo"], rec=p(rec["taxa"]), fa=p(fa["taxa"])), t["nota"], 1, 2, wspace=0.75, left=0.19 if st["min"] == 16 else 0.2)[0]
    gs = [("golpe_explicito", "recall", 0), ("golpe_escondido", "recall", 0), ("injecao_golpe", "recall", 0),
          ("legitimo_comum", "falso_alarme", 1), ("legitimo_dificil", "falso_alarme", 1), ("injecao_legitimo", "falso_alarme", 1)]
    metas = d.aval["metas_declaradas"]
    for k, a in enumerate(ax):
        itens = [g for g in gs if g[2] == k]
        xs = np.arange(len(itens))[::-1]
        v = [pg[g][m]["taxa"] for g, m, _ in itens]
        lo = [pg[g][m]["ic95"][0] for g, m, _ in itens]
        hi = [pg[g][m]["ic95"][1] for g, m, _ in itens]
        barras(a, xs, v, lo, hi, cores=[C_CTX] * len(itens), hatch=[None if k == 0 else "//"] * len(itens), st=st, w=0.55,
               lab=[f"{p(x)} ({pg[g][m]['k']}/{pg[g][m]['n']})" for x, (g, m, _) in zip(v, itens)], horiz=True, ymax=1)
        a.set_yticks(xs, [t["grupos"][g].replace("com ordens ao avaliador", "com ordens\nao avaliador") for g, _, _ in itens], fontsize=F(st, 0.85))
        a.set_xlim(0, 1.45)
        a.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"], fontsize=F(st, 0.85))
        mv = metas["recall_golpes_min"] if k == 0 else metas["falso_alarme_legitimos_max"]
        a.axvline(mv, color="#111111", ls="--", lw=st["lw"] * 0.6)
        a.text(mv, len(itens) - 0.45, f" {t['meta']} {p(mv)}", fontsize=F(st, 0.8), va="bottom")
        a.set_xlabel(t["x1"] if k == 0 else t["x2"], fontsize=F(st, 0.9))
    ax[1].tick_params(labelleft=True)


def fe_15(fig, st, d):
    t = TX["fe_15"]
    rv15, rvc = d.dm("RESERVADO", "15/11"), d.dm("RESERVADO", "corrigido")
    ax = quadro(fig, st, fmt(t["titulo"], a=p(rv15.tpr), b=p(rvc.tpr)), t["nota"], 1, 2, wspace=0.2)[0]
    for a, conj, tit in ((ax[0], "RESERVADO", t["p1"]), (ax[1], "DEV", t["p2"])):
        x15, xc = d.dm(conj, "15/11"), d.dm(conj, "corrigido")
        for i, m in enumerate(("acc", "tpr", "fpr")):
            for j, (x, cor, h) in enumerate(((x15, C_CONG, "//"), (xc, C_CTX, None))):
                v, lo, hi = ic3(x, m)
                barras(a, [i + (j - 0.5) * 0.36], [v], [lo], [hi], cores=[cor], hatch=[h], st=st, w=0.34, lab=[p(v)], ymax=1)
        a.axhline(x15.acc_trivial, color="#111111", ls=":", lw=st["lw"] * 0.6)
        a.text(2.45, x15.acc_trivial + 0.015, fmt(t["trivial"], t=p(x15.acc_trivial)), fontsize=F(st, 0.75), ha="right", va="bottom")
        a.set_xticks([0, 1, 2], [t["m"]["acc"], t["m"]["tpr"], t["m"]["fpr"]], fontsize=F(st, 0.9))
        a.set_ylim(0, 1.2)
        a.set_title(tit, fontsize=st["fs"], fontweight="bold", loc="left")
    legpatch(ax[0], color=C_CONG, hatch="//", edgecolor="#222222", label=t["pt"])
    legpatch(ax[0], color=C_CTX, edgecolor="#222222", label=t["auto"])
    ax[0].legend(fontsize=F(st, 0.75), frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93))


def fe_16(fig, st, d):
    t = TX["fe_16"]
    g = d.ctxg()
    ia, rc = g.loc["ia_pedido_dinheiro"], g.loc["real_com_golpe"]
    ax = quadro(fig, st, fmt(t["titulo"], a=f"{int(ia.alerta_medio_mais)} de {int(ia.n)}", b=f"{int(rc.alerta_medio_mais)} de {int(rc.n)}"), t["nota"])[0, 0]
    xs = np.arange(4)
    for j, (col, cor, h, rot) in enumerate((("alerta_medio_mais", C_CTX, None, t["medio"]), ("alerta_so_alto", C_CTX, "//", t["alto"]))):
        v = [g.loc[k, col] / g.loc[k, "n"] for k in GRUPOS]
        lab = [f"{int(g.loc[k, col])}/{int(g.loc[k, 'n'])}" for k in GRUPOS]
        barras(ax, xs + (j - 0.5) * 0.36, v, cores=[cor] * 4, hatch=[h] * 4, st=st, w=0.34, lab=lab, ymax=1)
        legpatch(ax, color=cor, hatch=h, edgecolor="#222222", label=rot)
    ax.set_xticks(xs, [t["g"][k] for k in GRUPOS], fontsize=F(st, 0.9))
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6], ["0", "20", "40", "60%"])
    ax.set_ylabel(t["y"], fontsize=st["fs"])
    ax.legend(fontsize=st["fs"], frameon=False, loc="upper left")


def fe_17(fig, st, d):
    t = TX["fe_17"]
    fr = d.frase.copy()
    cl = fr[(fr.tema == "golpe") & (fr.recall == 1.0)]
    le = fr[(fr.tema == "golpe") & (fr.recall == 0.0)]
    co = fr[fr.tema == "cotidiano"]
    ax = quadro(fig, st, fmt(t["titulo"], c=f"{int(cl.alerta_medio_mais.sum())} de {int(cl.n_clipes.sum())}", l=f"{int(le.alerta_medio_mais.sum())} de {int(le.n_clipes.sum())}"),
                t["nota"], left=0.07)[0, 0]
    grupos = [("claras", cl, C_DET, None), ("leves", le, C_CTX, "//"), ("cotidiano", co, C_CONG, "xx")]
    y = 0
    ticks, labs = [], []
    for nome, df, cor, h in grupos:
        for _, r in df.sort_values("frase_id").iterrows():
            ax.barh(y, r.n_clipes, color="white", edgecolor="#222222", lw=0.8, height=0.7)
            ax.barh(y, r.alerta_medio_mais, color=cor, hatch=h, edgecolor="#222222", lw=0.8, height=0.7)
            ax.text(r.n_clipes + 0.1, y, f"{int(r.alerta_medio_mais)}/{int(r.n_clipes)}", va="center", fontsize=F(st, 0.8))
            ticks.append(y), labs.append(r.frase_id)
            y += 1
        ax.text(7.6, y - len(df) / 2 - 0.5, f"{TX['fe_17']['classe'][nome]}\n{int(df.alerta_medio_mais.sum())} de {int(df.n_clipes.sum())}", va="center", ha="left",
                fontsize=F(st, 0.85), fontweight="bold", color="#111111")
        y += 0.6
    ax.set_yticks(ticks, labs, fontsize=F(st, 0.75))
    ax.invert_yaxis()
    ax.set_xlim(0, 12.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel(t["x"], fontsize=F(st, 0.85), loc="left")
    ax.grid(axis="y", visible=False)


def fe_18(fig, st, d):
    t = TX["fe_18"]
    cb = d.comb()
    M = np.array([[cb.loc[g, "baixo_corrigido"], cb.loc[g, "atencao_corrigido"], cb.loc[g, "alto_corrigido"]] for g in GRUPOS], float)
    ia_baixo = int(M[2:, 0].sum())
    real_alto = int(M[:2, 2].sum())
    assert ia_baixo == 0 and real_alto == 0
    ax = quadro(fig, st, t["titulo"], t["nota"], left=0.3 if st["min"] == 16 else 0.38)[0, 0]
    ax.grid(False)
    frac = M / M.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=st["fs"] * 1.4, fontweight="bold", color="white" if frac[i, j] > 0.55 else "#111111")
    ax.set_xticks(range(3), t["n"], fontsize=st["fs"])
    ax.set_yticks(range(4), [f"{t['g'][g]} (n={int(M[i].sum())})" for i, g in enumerate(GRUPOS)], fontsize=st["fs"])
    ax.set_xlabel(t["x"], fontsize=st["fs"])
    for s in ax.spines.values():
        s.set_visible(False)


def fe_19(fig, st, d):
    t = TX["fe_19"]
    r = d.pr7.loc["T3"]
    ctx = d.aval["positivo_medio_ou_alto"]["recall_golpes_todos"]["taxa"]
    wa = d.d("T3", "M", "A")
    pes = int(d.par[(d.par.modelo == "T3") & (d.par.condicao == "A")].pessoas.iloc[0])
    pareado = d.par[(d.par.modelo == "T3") & (d.par.condicao == "A")].iloc[0]
    fr = d.frase
    cl = fr[(fr.tema == "golpe") & (fr.recall == 1.0)]
    c = f"{int(cl.alerta_medio_mais.sum())} de {int(cl.n_clipes.sum())}"
    cc = ctx100(d)
    tm = d.tm[(d.tm.receita == "T3")].drop_duplicates("recorte").set_index("recorte")
    k = dict(m3=f(r["média AUC (3 fontes)"], 3), lo=f(r["média IC2.5"], 3), hi=f(r["média IC97.5"], 3), acc=p(wa.acc), pes=pes, c=c, rec=p(ctx), nd=int(d.est[d.est.condicao == "original"].passa_M.sum()),
             br=p(tm.loc["brspeech_df", "FPR M"]), sem=f(d.ata.loc["sem a taxa original", "AUC_loo"], 3), rw=p(cc["recall"]))
    k["pes"] = int(pareado.ia_maior_que_real)
    ax = quadro(fig, st, t["titulo"], t["nota"], 1, 2, wspace=0.06)[0]
    for a, tit, itens, cor, mk in ((ax[0], t["sust"], t["s"], C_DET, "✓"), (ax[1], t["nao"], t["n"], C_IA, "?")):
        a.axis("off")
        a.grid(False)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.add_patch(FancyBboxPatch((0.0, 0.0), 1, 0.9, boxstyle="round,pad=0.005,rounding_size=0.03", fc="white", ec=cor, lw=st["lw"]))
        a.text(0.5, 0.95, tit, ha="center", va="bottom", fontsize=st["fs"] * 1.1, fontweight="bold", color=cor if cor != C_IA else "#8a3a00")
        y = 0.84
        wrapw = 40 if st["min"] == 16 else 34
        for s_ in itens:
            txt = textwrap.fill(fmt(s_, **k), wrapw)
            nl = txt.count(chr(10)) + 1
            a.text(0.04, y, mk, fontsize=F(st, 1.1), fontweight="bold", va="top", color=cor)
            a.text(0.10, y, txt, fontsize=F(st, 0.92), va="top")
            y -= 0.05 + nl * (0.055 if st["min"] == 16 else 0.058)


def fe_20(fig, st, d):
    t = TX["fe_20"]
    r = d.pr7.loc["T3"]
    wa = d.d("T3", "M", "A")
    est = d.est[d.est.condicao == "original"]
    nd = int(est.passa_M.sum())
    cb = d.comb()
    ia_n = int(cb.loc[["ia_pedido_dinheiro", "ia_neutro"], "n"].sum())
    ia_b = int(cb.loc[["ia_pedido_dinheiro", "ia_neutro"], "baixo_corrigido"].sum())
    ra = int(cb.loc[["real_sem_golpe", "real_com_golpe"], "alto_corrigido"].sum())
    ax = quadro(fig, st, t["titulo"], t["nota"], 2, 2, wspace=0.12, hspace=0.18)
    itens = [(t["k1"], f(r["média AUC (3 fontes)"], 3), fmt(t["s1"], lo=f(r["média IC2.5"], 3), hi=f(r["média IC97.5"], 3)), C_DET),
             (t["k2"], p(wa.acc), fmt(t["s2"], lo=p(wa.acc_lo), hi=p(wa.acc_hi)), C_DET),
             (t["k3"], f"{nd} de {len(est)}", t["s3"], C_IA),
             (t["k4"], f"{ia_b} de {ia_n}", fmt(t["s4"], n=ia_n, ra=ra), C_CTX)]
    for a, (rot, big, sub, cor) in zip(ax.ravel(), itens):
        a.axis("off")
        a.grid(False)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.005,rounding_size=0.04", fc="white", ec=cor, lw=st["lw"]))
        a.text(0.5, 0.86, rot, ha="center", va="top", fontsize=F(st, 0.95), fontweight="bold")
        a.text(0.5, 0.48, big, ha="center", va="center", fontsize=st["fs"] * 3.0, fontweight="bold", color=cor if cor != C_CTX else "#8a5a00")
        a.text(0.5, 0.14, sub, ha="center", va="center", fontsize=F(st, 0.8), color="#333333")


FIGS = [("fe_01_fluxo_sistema", fe_01, 4.4), ("fe_02_linha_do_tempo", fe_02, 4.6), ("fe_03_evolucao_versoes", fe_03, 3.4), ("fe_04_f0_vs_t3_por_fonte", fe_04, 3.4),
        ("fe_05_receitas_t0_t4", fe_05, 3.4), ("fe_06_t3_vs_v8", fe_06, 3.4), ("fe_07_modelo_leve", fe_07, 3.5), ("fe_08_whatsapp_detector", fe_08, 3.7),
        ("fe_09_whatsapp_notas", fe_09, 3.4), ("fe_10_canal_opus", fe_10, 3.4), ("fe_11_atalho_metadados", fe_11, 3.5), ("fe_12_estresse_3_ia", fe_12, 3.4),
        ("fe_13_portugues_corte", fe_13, 3.6), ("fe_14_contexto_textos_inventados", fe_14, 3.8), ("fe_15_contexto_antes_depois_idioma", fe_15, 3.6),
        ("fe_16_contexto_por_grupo", fe_16, 3.5), ("fe_17_roteiros_claros_vs_leves", fe_17, 4.6), ("fe_18_combinado_grupo_nivel", fe_18, 3.4),
        ("fe_19_limitacoes", fe_19, 4.2), ("fe_20_resumo_final", fe_20, 4.4)]


def renderizar(nome, fn, hp, d, modos=("slides", "paper")):
    for modo in modos:
        st = dict(MODOS[modo])
        w, h = st["w"], (st["h"] if modo == "slides" else hp)
        fig = plt.figure(figsize=(w, h), dpi=st["dpi"])
        fn(fig, st, d)
        fig.savefig(OUT / modo / f"{nome}.png", dpi=st["dpi"], facecolor="white")
        plt.close(fig)


def folha(nomes):
    from PIL import Image, ImageDraw
    ncol, tw = 4, 480
    th = 270
    nl = (len(nomes) + ncol - 1) // ncol
    im = Image.new("RGB", (ncol * tw, nl * (th + 30)), "white")
    dr = ImageDraw.Draw(im)
    for i, n in enumerate(nomes):
        a = Image.open(OUT / "slides" / f"{n}.png").convert("RGB").resize((tw - 10, th - 5))
        x, y = (i % ncol) * tw + 5, (i // ncol) * (th + 30) + 28
        im.paste(a, (x, y))
        dr.text((x, y - 22), n, fill="black")
    im.save(OUT / "folha_resumo.png")


def indice(feitas, puladas):
    L = ["# Figuras da entrega (paper e slides)", "",
         "Geradas por `scripts/figuras_entrega.py`, só a partir de `results/*.csv|json`. Cada figura tem duas saídas: `slides/` (1920×1080, fonte ≥ 16 pt) e `paper/` (6,5 pol, 300 dpi, fonte ≥ 8 pt). Miniaturas: `folha_resumo.png`. Cores: azul = detector, laranja = contexto, verde = real, vermelho = IA, cinza = congelado/não adotado/referência.", "",
         "| figura | o que mostra | dados | equivalente já existente em results/slides | onde usar |", "|:--|:--|:--|:--|:--|"]
    for n in feitas:
        dados, mostra, eq, uso = INDEX[n]
        L.append(f"| `{n}.png` | {mostra} | {dados} | {eq} | {uso} |")
    L += ["", "**Já existem e não foram refeitas:** " + JA_EXISTE + ".", ""]
    if puladas:
        L += ["## Figuras puladas", ""] + [f"- `{n}`: {m}" for n, m in puladas]
    else:
        L += ["## Figuras puladas", "", "Nenhuma: todos os dados necessários existem em `results/`."]
    L += ["", "**Observação:** `tw_01`, `tw_02`, `tw_04` e `tw_05` (15/11) trazem o contexto com o idioma do Whisper forçado errado; os substitutos corrigidos são `fe_08`, `fe_16` e `fe_18` (ver `docs/ERRATA_TESTE_WHATSAPP.md`)."]
    (OUT / "INDEX.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    """Sem argumentos: todas as figuras, slides e paper, mais folha e INDEX. Com --so fe_01,fe_04 (e --modos slides): só essas, sem folha nem INDEX."""
    args = sys.argv[1:]
    so = args[args.index("--so") + 1].split(",") if "--so" in args else None
    modos = tuple(args[args.index("--modos") + 1].split(",")) if "--modos" in args else ("slides", "paper")
    d = D()
    feitas, puladas = [], []
    for nome, fn, hp in FIGS:
        if so is not None and nome[:5] not in so:
            continue
        try:
            renderizar(nome, fn, hp, d, modos)
            feitas.append(nome)
            print("ok", nome)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            puladas.append((nome, f"erro ao gerar: {type(e).__name__}: {e}"))
    if so is not None:
        print(len(feitas), "figuras (seleção; folha e INDEX não refeitos)")
        return 0 if not puladas else 1
    folha(feitas)
    indice(feitas, puladas)
    print(len(feitas), "figuras;", len(puladas), "puladas")
    return 0 if not puladas else 1


if __name__ == "__main__":
    sys.exit(main())
