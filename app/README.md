# Escuta Segura: esboço do app

Esboço para a apresentação. **Não é um app pronto.** A pessoa manda um áudio recebido (de WhatsApp ou de uma ligação gravada) e o app junta duas perguntas:

1. **A voz parece gerada por IA?** Quem responde é a **T3 (v7)**, o sistema principal, chamada pelo gabarito `scripts/inferencia_referencia.py` (ver `docs/GUIA_PARA_O_SITE.md`).
2. **O que foi dito parece golpe?** Quem responde é o **módulo de contexto dos colegas**, `src/contexto_golpe.py` (ver `docs/CONTEXTO_GOLPE.md`): o Whisper transcreve e um LLM classifica, os dois pela API do Groq.

```
áudio ──► app/voz.py (T3) ────────────────────────────► voz com sinais de IA?  ┐
      └─► app/contexto.py ─► src/contexto_golpe.py ───► risco do conteúdo      ┘──► app/alerta.py ──► baixo / atenção / ALTO
```

## A regra do alerta

É a regra congelada pelos colegas (`docs/RESULTADO_COMBINADO_CORRIGIDO.md`):
- a voz conta como **IA** se a nota da T3 for maior ou igual ao corte M da validação;
- o conteúdo conta como **suspeito** se o risco for `medio` ou `alto`.

| voz \ conteúdo | não suspeito (`nenhum`, `baixo`) | suspeito (`medio`, `alto`) | `indeterminado` (falha, sem fala) |
|:--|:--|:--|:--|
| **sem sinais de IA** | baixo | atenção | não avaliado |
| **com sinais de IA** | atenção | **ALTO: alerta de golpe** | atenção |

"Indeterminado" **nunca** conta como "não suspeito", e o LLM nunca rebaixa o alerta da voz (`docs/CONTEXTO_GOLPE.md`, seção 6).

## Como rodar

```bash
env\Scripts\activate
python app/demo.py                    # contexto offline (sem internet, sem chave)
python app/demo.py --contexto groq    # contexto de verdade, o módulo dos colegas
```

O `demo.py` calcula os exemplos fixos de `app/exemplos.csv`. Para **ver as telas e analisar áudios novos**, ligue o servidor local:

```bash
python app/servidor.py                    # contexto offline: Whisper local + palavras-chave (sem internet)
python app/servidor.py --contexto groq    # contexto dos colegas (envia o áudio ao Groq)
```

Depois, abra `http://127.0.0.1:8765`. No final da página, o campo **"Analise um áudio"** recebe um arquivo WAV, MP3, OGG ou FLAC, de até 25 MB.
- O servidor roda o mesmo caminho dos exemplos: voz (T3), transcrição, contexto e alerta. O resultado vira um celular novo, igual aos outros.
- No modo offline, a transcrição é feita pelo Whisper large-v3-turbo, a partir do cache local do Hugging Face.
- O áudio fica em `app/mockup/enviados/` (fora do git), só para a página poder tocá-lo. O celular novo some ao recarregar a página.
- O servidor só aceita conexões deste computador (127.0.0.1).

Abrindo `app/mockup/index.html` direto, os exemplos aparecem, mas o campo de envio não funciona.

### Abrir pelo celular

```bash
python app/servidor.py --rede
```

O servidor mostra o endereço para o celular, algo como `http://10.102.14.252:8765`. O celular precisa estar **na mesma rede** que o computador.
- Enquanto o servidor estiver ligado com `--rede`, qualquer aparelho dessa rede consegue abrir a página e enviar áudios.
- **Wi-Fi da faculdade costuma bloquear** a conexão entre aparelhos, e no Windows ele aparece como rede "Pública", em que o firewall barra conexões de entrada. O jeito mais simples: ligar o **roteador do celular** e conectar o computador nele.
- Na primeira vez, o Windows pergunta se libera o Python no firewall. Isso é você quem decide, na janela do próprio Windows.

**QR code para a apresentação:** com o servidor ligado com `--rede`, abra `http://127.0.0.1:8765/qr.html` no computador. A página desenha o QR com o endereço da rede **atual**. O QR é desenhado pela biblioteca qrcodejs (⚠️ Fora da trilha), carregada do cdnjs, então o computador precisa de internet.
- **O endereço muda de rede para rede.** Gere o QR no dia, já conectado à rede da apresentação (por exemplo, o roteador do celular).
- Ou mostre a própria página `qr.html` na tela, em vez de um print no slide.
- Um QR tirado hoje, no Wi-Fi da faculdade, não vai funcionar em outra rede.

**Modo `groq`:** é o módulo de verdade.
- Precisa da `GROQ_API_KEY` no ambiente. Para configurar, veja `docs/CONTEXTO_GOLPE.md`, seção 2. A chave nunca vai para arquivo.
- **Envia o áudio a um serviço externo.** Só use com o consentimento de quem gravou e nunca com áudios dos bancos do projeto.
- O idioma é `auto`, como foi congelado em `results/diagnostico_whisper_congelado.json`.

**Modo `offline`:** é só uma reserva para quando não houver internet ou chave.
- Usa a transcrição digitada em `app/exemplos.csv` e uma lista de palavras-chave, com regras escolhidas à mão e não medidas: nenhum tipo de sinal dá risco `nenhum`; 1 dá `baixo`; 2 dão `medio`; 3 ou mais dão `alto`.
- **Ele erra fácil:** palavras como "códigos" ou "fraude" num assunto inocente já disparam um sinal.

## Os exemplos

Ficam em `app/exemplos_audio/`, **fora do git**, porque são vozes de pessoas reais e clones delas, que não devem ser publicados. As colunas de `app/exemplos.csv` são `id`, `arquivo`, `transcricao` (usada só no modo offline), `titulo`, `contato` e `hora`.

São quatro, todos do zip da coleta do grupo, um para cada célula da regra:

| arquivo no app | origem no zip | voz | conteúdo | alerta esperado |
|:--|:--|:--|:--|:--|
| `real_normal_telefone.wav` | `coleta/wav`, pessoa p009, 1º áudio, com filtro de telefone | real | recado do dia a dia | baixo |
| `real_golpe_telefone.wav` | `coleta/wav`, pessoa p003, 2º áudio, com filtro de telefone | real | roteiro "filho com número novo pedindo dinheiro" | atenção |
| `clone_neutro_telefone.wav` | `coleta/clonados_neutros_nano/p001_neutro01.wav`, com filtro de telefone | clone (chatterbox-nano) | assunto comum (cozinhar macarrão) | atenção |
| `clone_golpe.wav` | `coleta/clonados_nano/p014_clone01.wav`, sem filtro | clone (chatterbox-nano) | cartão recusado em viagem, pedindo dinheiro para o hotel ("send me enough to cover the room") | **alto** |

**Filtro de telefone (celulares 1 a 3):** `app/telefone.py` refaz em Python o canal G.711 de `golpe_dataset/voz/03_processar.py` (não há ffmpeg aqui): passa-alta de 300 Hz e passa-baixa de 3.400 Hz (biquads do torchaudio), reamostragem para 8 kHz e μ-law de 8 bits. O celular 4 fica com o áudio original. Nota da T3, sem filtro → com filtro (o alerta não mudou em nenhum):

| celular | sem filtro | com filtro |
|:--|--:|--:|
| 1 (real, recado) | −2,93 | −1,23 |
| 2 (real, roteiro de golpe) | −3,48 | −0,80 |
| 3 (clone, assunto comum) | +6,97 | +9,96 |
| 4 (clone, pedido de dinheiro; o celular usa o original) | +7,32 | +7,68 |

Nos dois reais, o filtro aproximou a nota do corte (+0,23), sem cruzá-lo; nos clones, afastou. São 4 áudios: não dá para concluir nada sobre o efeito do canal em geral.

As transcrições dos reais vieram do Whisper local; as dos clones, do `manifesto.csv` da coleta.

Os exemplos foram escolhidos pelo tipo de áudio, antes de ver a nota da T3 do app. O celular 4 já usou o `p003_clone02` e o `p010_clone02`. Agora usa o `p014_clone01`, por escolha do grupo depois de ouvir os áudios. Isto é uma demonstração, não uma avaliação. A avaliação na coleta está em `docs/RESULTADO_TESTE_WHATSAPP.md`.

## Para trocar uma peça

| peça | arquivo | o que manter |
|:--|:--|:--|
| detector de voz | `app/voz.py` | a saída de `DetectorVoz.analisar`: `nota`, `corte`, `parece_ia`, `chance_ia` |
| contexto | `app/contexto.py` → `src/contexto_golpe.py` | o contrato de `docs/CONTEXTO_GOLPE.md`, seção 4 (`status`, `transcricao`, `risco_conteudo`, `sinais` [{`tipo`, `evidencia`}], `resumo`...) |
| regra do alerta | `app/alerta.py` | a tabela acima; mudar só se o grupo mudar a regra congelada |

## Cuidados

- **A decisão da voz** compara a nota da T3 com o corte M. A barra azul usa a sigmoide da nota só para desenhar, e a calibração dessa porcentagem não foi conferida.
- **É um alerta, não um veredito.** O detector erra, e o golpe pode vir com voz humana.
- **⚠️ Fora da trilha:** o mockup (HTML/CSS/JavaScript), as expressões regulares do modo offline e o Whisper/LLM do módulo de contexto.

## Arquivos

| arquivo | o que é | no git? |
|:--|:--|:--|
| `voz.py`, `contexto.py`, `alerta.py` | as três peças do app | sim |
| `telefone.py` | filtro de telefone (G.711) aplicado aos exemplos 1 a 3 | sim |
| `demo.py`, `exemplos.csv` | roda os exemplos e prepara o mockup | sim |
| `servidor.py` | serve as telas e analisa áudios enviados pela página | sim |
| `mockup/qr.html` | QR code com o endereço do servidor na rede atual (para o celular) | sim |
| `mockup/index.html`, `mockup/dados.js` | as telas e os resultados (com as transcrições) | sim |
| `saida/resultados.json` | os resultados completos | sim |
| `exemplos_audio/`, `mockup/audios/`, `mockup/enviados/` | os áudios, as cópias para o mockup e os enviados pela página | não |
