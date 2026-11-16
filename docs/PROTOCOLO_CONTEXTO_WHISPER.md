# Protocolo: reavaliação do módulo de contexto com o idioma corrigido (teste WhatsApp)

Escrito em **2026-11-16 10:05 (UTC-0300)**, **antes de qualquer chamada à API**. Nenhum nome de pessoa e nenhuma transcrição neste arquivo.

## Arquivos congelados (SHA256)

| Arquivo | SHA256 |
|---|---|
| `docs/avaliacao_contexto/prompt_r1.txt` (fim de linha LF, como em `results/contexto_golpe_congelado.json`) | `3f848dbf65e294475e86ee963ab09103f7b83853bec979f14fbad872bf6de503` |
| `src/contexto_golpe.py` (LF) | `d0327e1b3c177aba0b25d307199ea6c4900c7ce3d17e09254382780e15d81dfb` |
| `results/diagnostico_whisper_divisao.csv` | `77d1b1b8cb54113a8eb4936c75fd366b3c01d4d15dd9a6775c5c2e189ea18a9e` |
| `data/processed/manifest_whatsapp.parquet` | `0fcac35d27fc0cadcbfeb8c99395b8a478dfc758a0d1e5cea9bf7c360f655b29` |

(No disco do Windows os dois arquivos de texto estão em CRLF; os SHA256 acima são com LF, os mesmos do congelamento de 15/11.)

## Motivo

`scripts/teste_whatsapp.py` chamou `analisar_audio(..., idioma="pt")`: o Whisper foi **forçado** a português. O áudio do banco é inglês. O relatório de 15/11 dizia "idioma=auto" e "o Whisper detectou português nos 100 clipes"; isso era efeito do parâmetro forçado, não detecção. Portanto o resultado de contexto de 15/11 (acurácia 53%) **é inválido como medida do módulo**.

## Princípio

O módulo não deve ser obrigado a um idioma. Se a detecção automática funcionar, ótimo; se não, o módulo é declarado **somente inglês** e isso é uma limitação a escrever. Não alteramos prompt, LLM, cortes nem `src/contexto_golpe.py`. Nunca se usa a função de **tradução** do Whisper, só a transcrição.

## Limite do que o banco permite validar

Todos os áudios do banco são em **inglês**. Logo só dá para validar o módulo em inglês. O comportamento do modo `auto` em **português não foi validado por nós**.

## Fase DEV (8 pessoas, 40 clipes; 24 clones e 16 reais)

Só o Whisper (`whisper-large-v3-turbo`, `verbose_json`, `temperature 0`, mesmos filtros de segmento de `src/contexto_golpe.py`), sem o LLM, nos modos:

- **auto**: o parâmetro `language` é omitido (conferido no código: `transcrever()` só põe `language` em `dados` se `idioma != "auto"`);
- **en**: `language=en`;
- **pt**: reaproveita as transcrições já existentes de 15/11, só como comparação (nenhuma chamada nova).

Medidas: (1) WER dos 24 clones do DEV contra o texto do manifesto (minúsculas, sem pontuação, mesma conta de `scripts/diagnostico_whisper.py`); (2) idioma detectado no modo auto (clones e reais); (3) concordância auto × en (mesmo texto normalizado, e WER de um contra o outro).

### Regra de escolha (fixa agora)

Usar **auto** se **(a)** o Whisper detectar `english` em pelo menos **90%** dos 40 clipes do DEV no modo auto **e (b)** o WER médio dos clones no auto for no máximo **0,05 acima** do WER médio dos clones no modo en. Caso contrário, usar **en** e declarar o módulo **"somente inglês"** (limitação a escrever).

Se o **WER mediano** dos clones no modo escolhido ficar **acima de 0,15**, testar **UMA** variante com `whisper-large-v3` nesse mesmo modo e escolher o modelo de **menor WER médio** dos clones (diferença menor que 0,01: fica o `whisper-large-v3-turbo`). Se o WER mediano ficar até 0,15, não testar.

### Congelamento

A escolha (modo, modelo, hora, SHA256 dos arquivos acima) é gravada em `results/diagnostico_whisper_congelado.json` **antes** de tocar no RESERVADO.

## Fase RESERVADO (15 pessoas, 60 clipes)

Pipeline completo (Whisper no modo congelado + julgamento com o prompt r1 congelado e `openai/gpt-oss-120b`, intervalo de 12 s entre clipes), **uma única vez**; o marcador `results/diagnostico_whisper_reservado_concluido.json` impede repetir. É **proibido ler ou imprimir transcrições do RESERVADO**; só métricas agregadas, calculadas depois do congelamento. Se vier 429 persistente: parar e avisar, sem descartar clipes (a retomada reaproveita as chamadas já feitas).

Depois do RESERVADO, o mesmo pipeline roda no DEV com o modo congelado, e é reportado **separado**, marcado "usado no diagnóstico".

## Métricas

As mesmas do teste de 15/11: acurácia (ao lado da trivial), recall, falso alarme, precisão, por grupo. IC95 por bootstrap de pessoas (2.000 repetições, seed 42) e Wilson como complemento. Positivo = conteúdo de golpe (`real_com_golpe` e `ia_pedido_dinheiro`); alerta = risco **médio ou alto**; `indeterminado` e `audio_sem_fala` = sem alerta, e contados à parte. **Principal = RESERVADO.** Comparação, nos mesmos conjuntos e clipes, com a versão de 15/11 (`results/teste_whatsapp_notas.csv`, derivada de `results/teste_whatsapp_contexto.csv`).

## Ressalvas (a repetir no resultado)

- 23 pessoas no total (15 no RESERVADO): ICs largos.
- Os 10 textos de pedido de dinheiro se repetem entre as pessoas: o RESERVADO **não é independente em texto**.
- O rótulo "com golpe" dos áudios reais vem da coleta e pode não conter pedido de dinheiro.
- Clones de pedido de dinheiro nem sempre são golpe.
- Os agregados dos 100 clipes (com o idioma errado) já tinham sido vistos em 15/11; o RESERVADO não é virgem em agregado.
