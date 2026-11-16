# Errata do teste externo WhatsApp (16/11/2026)

`scripts/teste_whatsapp.py` (linha 186) chamou `cg.analisar_audio(..., idioma="pt")`: o Whisper foi **forçado** a português. O áudio do banco é inglês. Vários textos de 15/11 repetem consequências disso. Os arquivos antigos **não foram editados**; esta errata aponta cada trecho. Reavaliação: `docs/RESULTADO_CONTEXTO_WHISPER.md` (protocolo: `docs/PROTOCOLO_CONTEXTO_WHISPER.md`).

## Os quatro erros

**(1) "O Whisper detectou português nos 100 clipes" — era forçado, não detectado.** O campo `idioma` devolvido pelo Groq reflete o idioma pedido. Na verdade, com detecção automática, 96 dos 100 clipes (DEV + RESERVADO: 38/40 + 58/60) saem como inglês.
- `docs/RESULTADO_TESTE_WHATSAPP.md`, linha 7 (título da seção 1) e linha 9 ("O Whisper detectou **português nos 100 clipes** ... A alegação da coleta ('tudo em inglês') **não se confirma**").
- `docs/PROTOCOLO_TESTE_WHATSAPP.md`, linha 82 ("o Whisper detectou **português nos 100 clipes**").
- `docs/HISTORICO_VERSOES.md`, linha 979 ("o Whisper detectou **português nos 100 clipes** ... pendência: conferir as transcrições").
- `results/teste_whatsapp_resumo.json`: `idiomas_detectados` e `"nao_ingles": 100`; coluna `idioma` de `results/teste_whatsapp_notas.csv` (todos "Portuguese").

**(2) O desvio "idioma=auto" — na verdade foi `pt`.**
- `docs/PROTOCOLO_TESTE_WHATSAPP.md`, linha 56 ("Será passado **`idioma="auto"`**").
- `docs/RESULTADO_TESTE_WHATSAPP.md`, linha 56 (desvio 1: "a chamada usou `idioma="auto"` em vez do padrão `"pt"`") e linha 69 (item (b): "O Whisper rodou com `idioma="auto"`").

**(3) A ressalva "o T3 nunca viu português" para este banco — não se aplica.** O áudio é inglês. (Continua verdadeira para português em geral: ver `docs/GUIA_PARA_O_SITE.md`.)
- `docs/RESULTADO_TESTE_WHATSAPP.md`, linha 9 ("O detector (T3) **nunca viu português no treino** ... este é um teste em português") e linha 64 ("idioma português, sem português no treino do detector").
- `docs/HISTORICO_VERSOES.md`, linha 980 ("o T3 nunca viu português no treino").
- O resultado do **detector** (98%, AUC ~1,0) não depende do idioma do Whisper e não é afetado por este erro.

**(4) O resultado de contexto de 15/11 (acurácia 53%) é inválido como medida do módulo.** Foi obtido com o Whisper forçado a português em áudio inglês (nos 100 clipes: 53%, recall 39,7%, falso alarme 11,1%).
- `docs/RESULTADO_TESTE_WHATSAPP.md`, linhas 18, 21 e 40 (acurácia 53%, "abaixo da trivial", comparação com 87,7% de recall dos textos inventados) e 68 (item (a)).
- `docs/HISTORICO_VERSOES.md`, linha 974.
- `results/teste_whatsapp_contexto.csv`, `results/teste_whatsapp_contexto_grupos.csv`, colunas `ctx_*` de `results/teste_whatsapp_notas.csv`.
- Figuras `results/slides/tw_01_acuracia_dois_modelos.png`, `tw_02_taxas_por_grupo.png`, `tw_04_matriz_contexto.png`, `tw_05_grupo_por_nivel.png` (partes de contexto) e suas linhas em `results/slides/INDEX.md` (72 a 76).
- Substituto: a tabela corrigida de `docs/RESULTADO_CONTEXTO_WHISPER.md`.

## Outro trecho desatualizado (não é o mesmo erro)

`docs/CONTEXTO_GOLPE.md`, linhas 22 e 220, dizem que o módulo usa idioma "pt" e só foi avaliado em português do Brasil / sem fala real. Para o site, ver a recomendação de idioma no resultado da reavaliação.

## O que não muda

Detector (T3, F0), cortes, condição B (Opus), atalho de metadados e o combinado em voz: nenhum depende do idioma do Whisper. O "combinado" usa o alerta de contexto de 15/11 e deve ser relido com cautela.
