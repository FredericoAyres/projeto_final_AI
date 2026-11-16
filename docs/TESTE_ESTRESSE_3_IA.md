# Teste de estresse informal: 3 áudios de IA (15 a 16/11/2026)

**Status:** teste informal de estresse. **Não é teste oficial**, não teve protocolo escrito antes, tem só 3 áudios e **não permite estimar taxa de acerto**. Nada foi treinado, ajustado ou trocado a partir dele.

## Origem (como informada)
Demonstração do Dessa (Block, research preview): modelo fala-para-fala, full-duplex, treinado do zero só em fala, sem clonagem de voz declarada (https://block.xyz/dessa). Não afirmamos nada além do que a página diz. **Não sabemos** qual trecho ou versão exata do modelo gerou os áudios, nem se o site comprimiu o áudio.

## Áudios
3 WAV de 10,4 a 12,5 s (48 kHz, mono), ids `ia_1` a `ia_3`. Mesmo caminho de `scripts/inferencia_referencia.py` (variante v1, fp32): mono 16 kHz, corte de silêncio, recorte central de 8 s, XLS-R camadas 5, 6, 7, regressão da T3. Nenhuma conta foi reescrita; nenhum corte foi alterado.

## Resultado (`results/teste_3_ia_notas.csv`)
Cortes congelados da T3: A' 0,027; M 0,233; F5 0,211.

| id | versão | nota T3 | prob. crua | A' / M / F5 | F0 (corte A / B) |
|---|---|---|---|---|---|
| ia_1 | original | -0,60 | 35,4% | não / não / não | não / não |
| ia_1 | Opus 16 kbps | -1,18 | 23,6% | não / não / não | sim / não |
| ia_2 | original | -3,38 | 3,3% | não / não / não | sim / não |
| ia_2 | Opus 16 kbps | -5,79 | 0,3% | não / não / não | sim / não |
| ia_3 | original | 2,72 | 93,8% | sim / sim / sim | sim / sim |
| ia_3 | Opus 16 kbps | 1,81 | 85,9% | sim / sim / sim | sim / sim |

- **T3 (corte M):** só o `ia_3` é acusado; `ia_1` e `ia_2` passam como reais. **1 de 3.**
- **Opus 16 kbps:** as notas caem de 0,6 a 2,4 logit; o `ia_3` continua acusado (1,81).
- **F0 (comparação, original):** `ia_1` não acusado em nenhum corte; `ia_2` só no corte A; `ia_3` nos dois.
- **Posição entre os reais do teste WhatsApp (44):** `ia_2` fica abaixo de 36% deles (maior que 63,6%); `ia_1` fica acima de 95% (95,5%), mas abaixo do corte. `ia_3` acima de 100%.

## Leitura
É um gerador de família nova (fala-para-fala), fora do treino. O resultado mostra o limite já declarado: o detector **não generaliza para geradores novos**. A "porcentagem" é a probabilidade crua do modelo (1/(1+e^-nota)), **sem calibração**; a escala das notas muda entre ambientes.

## Regras daqui em diante
Esses 3 áudios ficam **reservados como teste de estresse**: não entram em treino, validação nem escolha de corte. Arquivos em `data/teste_3_ia/` (fora do git).

## Próximo passo sugerido
Repetir com 15 a 30 áudios de vários geradores comerciais, sem treinar, com protocolo escrito antes.
