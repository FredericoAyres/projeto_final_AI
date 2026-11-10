"""Modelo de fala auto-supervisionado CONGELADO como extrator de características (v4: XLS-R 300M;
v5: XLS-R 300M e WavLM Large, `microsoft/wavlm-large`, ~94 mil horas de inglês).

⚠️ Fora da trilha: o modelo auto-supervisionado (wav2vec 2.0 / XLS-R, pré-treinado sem rótulos em
~436 mil horas de fala em 128 línguas) e o pacote `transformers`. Usar um modelo pré-treinado e congelado
como extrator de características é a ideia de transfer learning da Aula 10 (lá, com um modelo de imagem).

O que sai: as 25 "saídas escondidas" do modelo (`output_hidden_states=True`): a entrada do transformer
(índice 0) e a saída de cada uma das 24 camadas. Cada saída tem um vetor de 1024 números a cada 20 ms.
Resumimos cada saída de cada áudio pela MÉDIA e pelo DESVIO no tempo, só nos quadros válidos.
"""
import torch

NOME_PADRAO = "facebook/wav2vec2-xls-r-300m"
N_SAIDAS, DIM = 25, 1024
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def carregar_ssl(nome=NOME_PADRAO, device="cuda", dtype="float16"):
    """Carrega o modelo congelado (eval, sem gradiente) no `dtype` pedido. Lê só do cache local.

    `AutoModel` escolhe a classe pelo `config.json` do modelo: `Wav2Vec2Model` para o XLS-R (a mesma da v4)
    e `WavLMModel` para o WavLM Large (v5). Os dois têm 24 camadas de 1024 números a cada 20 ms.
    """
    from transformers import AutoModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()      # avisos de pesos do pré-treino que não usamos (quantizador)
    m = AutoModel.from_pretrained(nome, local_files_only=True, torch_dtype=DTYPES[dtype])
    n = m.config.num_hidden_layers + 1
    assert (n, m.config.hidden_size) == (N_SAIDAS, DIM), f"{nome}: {n} saídas de {m.config.hidden_size} (esperado 25 x 1024)"
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


def normalizar(x):
    """Média zero e variância um no clipe inteiro (o que o `Wav2Vec2FeatureExtractor` faz, do_normalize)."""
    x = x.float()
    return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)


@torch.inference_mode()
def representar(model, clipes, device="cuda", dtype="float16", forma=None):
    """Média e desvio no tempo das 25 saídas para cada clipe da lista `clipes` (tamanhos diferentes).

    - Cada clipe é normalizado sozinho (média 0, variância 1) ANTES do padding.
    - O lote é completado com zeros à direita só para caber num tensor; a `attention_mask` diz ao
      modelo quais amostras são de verdade, e média e desvio usam SÓ os quadros válidos de cada clipe.
    - `forma=(linhas, amostras)`: completa o lote até esse formato fixo (linhas extras = 25 ms de zeros,
      descartadas no fim). Motivo (execução, não método): no Windows, cada formato novo de lote faz a GPU
      carregar mais variantes de kernels na memória da CPU (medido: formatos variados levaram o processo
      a 14 GB reservados e o deixaram 2,4x mais lento; um formato fixo, 4,5 GB). Ver src/extracao_v4.py.
    Devolve (media, desvio) em float32, cada um (lote, 25, 1024), e o nº de quadros válidos (lote,).
    """
    n = torch.tensor([c.numel() for c in clipes])
    linhas, amostras = (len(clipes), int(n.max())) if forma is None else forma
    assert linhas >= len(clipes) and amostras >= int(n.max()), (forma, len(clipes), int(n.max()))
    lote = torch.zeros(linhas, amostras)
    mask = torch.zeros(linhas, amostras, dtype=torch.long)
    mask[len(clipes):, :400] = 1                 # linhas extras: 25 ms de silêncio (descartadas)
    for i, c in enumerate(clipes):
        lote[i, : c.numel()] = normalizar(c)
        mask[i, : c.numel()] = 1
    n = torch.cat([n, torch.full((linhas - len(clipes),), 400)])
    out = model(lote.to(device, DTYPES[dtype]), attention_mask=mask.to(device), output_hidden_states=True)
    nq = model._get_feat_extract_output_lengths(n).to(device)          # quadros válidos de cada clipe
    T = out.hidden_states[0].shape[1]
    valido = (torch.arange(T, device=device)[None, :] < nq[:, None]).float()[:, :, None]   # (B, T, 1)
    cont = valido.sum(dim=1)                                            # (B, 1)
    medias, desvios = [], []
    for h in out.hidden_states:                                         # 25 x (B, T, 1024)
        h = h.float()
        mu = (h * valido).sum(dim=1) / cont
        var = (((h - mu[:, None, :]) ** 2) * valido).sum(dim=1) / cont
        medias.append(mu)
        desvios.append(var.clamp_min(0).sqrt())
    k = len(clipes)
    return torch.stack(medias, 1)[:k], torch.stack(desvios, 1)[:k], nq.cpu()[:k]
