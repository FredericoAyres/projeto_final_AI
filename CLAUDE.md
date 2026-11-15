# CLAUDE.md

Projeto final da liga Insper AI: detector de voz gerada por IA (treino em inglês, teste também em pt-BR).

## Antes de qualquer coisa
- **Leia `docs/HISTORICO_VERSOES.md` inteiro antes de propor ou rodar qualquer versão.** Ao fechar uma versão, registre-a lá.
- Ambiente: Windows, Python 3.13; GPU RTX 4060 8 GB numa máquina e RTX 4050 Laptop 6 GB na outra (a v8 rodou na de 6 GB). Ative o env antes de qualquer comando: `env\Scripts\activate`.
- Memória: o limite é o commit do Windows (~34 GB, ~26 GB já usados pelo sistema). Cada worker do DataLoader reserva ~1 GB. Treino completo: um experimento por processo (`scripts/treinar_experimento.py`), fora desta sessão (ver `docs/HISTORICO_VERSOES.md`, seções g e h).

## Regras do jogo (anti-vazamento)
- Bancos de teste nunca entram no treino nem na escolha de hiperparâmetros.
- Decisões entre versões usam **validação + dev_shift** (`eval_subset == "dev_shift"` no manifest).
- O teste final (`eval_subset == "test_final"`) é avaliado **uma única vez por versão**, e TODAS as versões são reportadas, sem escolher a melhor pelo teste.
- Agrupar na mesma versão o que der para fazer junto, para ganhar tempo, mantendo comparações baratas (ablações) dentro da versão para entender o efeito de cada parte. (Substitui a regra antiga "uma mudança por experimento", a partir da v4.)
- O split train/val é por locutor.
- Nunca inverter a nota do modelo com base no teste.

## Métricas
- **EER e AUC continuam sendo as métricas principais** (não dependem de limiar nem da proporção real/fake).
- Acurácia e acurácia balanceada (média do recall de real e de fake) também são reportadas, com o limiar fixado na validação (ponto de EER) e com 0,5, sempre ao lado da acurácia trivial ("sempre a classe majoritária"). **Acurácia sozinha engana em banco desbalanceado:** no dev_shift (81% real), chamar tudo de real já acerta 81%.

## Fluxo de git
- Depois de cada commit que der certo, o usuário dá merge no `main`.

## Onde fica cada coisa
- `data/processed/manifest.parquet`: uma linha por áudio (a única fonte da verdade sobre os dados); gerado por `scripts/build_manifest.py`.
- `src/`: código reutilizável (dataset, augment, sampling, model, metrics, train, analysis). Não duplique código nos notebooks.
- `notebook.ipynb`: SÓ a versão atual. Ao fechar uma versão, copie para `notebooks/versions/vN_nome.ipynb` (com as saídas) antes de começar a próxima.
- `results/`: `baseline_results.csv` (v1) e `experiments.csv` (v2 e v3); da v4 em diante, `results/v4_*.csv`/`.json` (`v4_camadas.csv`, `v4_cabecas.csv`, `v4_resumo.json`...) e `v3_acuracias.csv`; históricos e notas por áudio ficam fora do git.
- `data/processed/v4_cache/`: características da v4 em shards (fora do git); gerado por `scripts/v4.py extrair`. Execução da v4: `scripts/vigia_v4.ps1` → `scripts/rodar_v4.ps1` → `scripts/v4.py <etapa>`. Teste final da v4 (já avaliado, uma vez): `scripts/vigia_v4_teste.ps1` → `scripts/rodar_v4_teste.ps1` → `scripts/v4_teste.py <etapa>`; resultados em `results/v4_teste_*`, cache em `data/processed/v4_cache_teste/` (fora do git).
- **Sistema principal = v7 (T3)** (decisão do grupo de 15/11/2026, em `docs/DECISAO_SISTEMA_PRINCIPAL.md`): XLS-R 300M congelado, camadas 5, 6, 7, média no tempo, padronização e regressão logística (C = 0,01). Pesos no git: `results/v7_t3_coeficientes.npz` e `results/v7_t3_modelo.json` (como usar). Notebook da raiz gerado por `scripts/nb_principal.py` (reaproveita `scripts/nb_v7.py` e `scripts/nb_v8.py`), cópia em `notebooks/versions/principal_v7.ipynb`.
- v7: protocolo `docs/PROTOCOLO_V7.md`, resultado `docs/RESULTADO_V7.md`; código `scripts/v7.py`, `scripts/v7_figuras.py`, `scripts/v7_poshoc.py`, `scripts/v7_slides.py`, `scripts/v7_t3_exportar.py`; notebook `scripts/nb_v7.py` → `notebooks/versions/v7_final.ipynb`; tabelas `results/v7_*`; figuras dos slides `results/slides/` (lista em `results/slides/INDEX.md`).
- v8 (experimento não adotado; fine-tuning parcial do XLS-R a partir da T3): protocolo `docs/PROTOCOLO_V8.md`; código `src/finetune.py` e `scripts/v8.py <etapa>` (`fila` roda tudo); figuras `scripts/v8_figuras.py` (`--slides` para `results/slides/v8_*`); notebook `scripts/nb_v8.py`; pesos da R1 por `scripts/v8_exportar.py` em `models/v8_R1_s*_pesos.safetensors` (fora do git; baixar da release `v8-pesos-r1` do GitHub; SHA256 em `results/v8_r1_pesos.json`). Áudios preparados em `data/processed/v8_audio/` (~28,5 GB) e notas por clipe em `data/processed/v8_notas/` (fora do git). Notebook arquivado: `notebooks/versions/v8_finetuning.ipynb`.
- `models/vN_*.pt`: checkpoints (fora do git).
- Modelo leve (fase 1; compressão do sistema principal, sem treino): protocolo `docs/MODELO_LEVE_PROTOCOLO.md`, resultado `docs/MODELO_LEVE.md`; código `scripts/modelo_leve.py <etapa>` e `scripts/modelo_leve_resumo.py`; tabelas `results/modelo_leve_*`; figuras `results/slides/ml_01_tamanho_e_tempo.png` e `ml_02_equivalencia.png`. Modelos em `models/modelo_leve_V1_fp32.safetensors`, `V2_fp16.safetensors` e `V3_int8.pt` (fora do git; regerar com `python scripts/modelo_leve.py tamanho`). Amostra e notas em `data/processed/modelo_leve_amostra/` (fora do git). A "variante leve V3 (int8)" não é a v3 do projeto.
- Guia para o site: `docs/GUIA_PARA_O_SITE.md` (caminho do áudio, arquivos, cortes, contrato, limites, checklist). Script de inferência de referência: `scripts/inferencia_referencia.py <áudio> [--variante v1|v2|v3] [--corte X]` e `--verificar [N]` (compara com as notas gravadas). No notebook da raiz: partes A (T3), B (modelo leve) e C (guia).
- Módulo de contexto de golpe (15/11/2026; protótipo separado do detector de voz; transcreve com o Whisper e julga o conteúdo com `openai/gpt-oss-120b`, ambos pela API do Groq; a chave só vem de `GROQ_API_KEY` no ambiente; nunca enviar áudio dos bancos do projeto): código `src/contexto_golpe.py`, `scripts/contexto_golpe_cli.py` (`--teste-conexao`, `--texto`, `--audio`, `--smoke`); documentação `docs/CONTEXTO_GOLPE.md`. Avaliação com textos inventados: protocolo `docs/PROTOCOLO_CONTEXTO.md` (congelado; o teste já foi avaliado, uma vez), resultado `docs/AVALIACAO_CONTEXTO.md`, textos e prompts em `docs/avaliacao_contexto/`, código `scripts/contexto_golpe_avaliar.py <etapa>` e `scripts/contexto_golpe_figuras.py`, tabelas `results/contexto_golpe_*` (fora do git pelo `results/*`: precisam de `git add -f`), figuras `results/slides/ctx_0*.png`. Log só de metadados em `logs/contexto_golpe.jsonl` (fora do git). Áudios gravados pelo grupo em `data/exemplos_contexto/` (fora do git).
- `docs/`: `HISTORICO_VERSOES.md`, `DECISAO_SISTEMA_PRINCIPAL.md`, `PROTOCOLO_V7.md`, `RESULTADO_V7.md`, `PROTOCOLO_V8.md`, `audit_report.md`, `dataset_overview.md`, `datasets.yaml`.

## Não faça
- Não mexa em `data/raw/` (nada é modificado, movido ou apagado lá).
- Não faça commit nem push sem o usuário pedir.

## Estilo
- Texto em português, direto e fácil de entender.
- Use só o que a trilha ensinou (https://trilhas-insperai.vercel.app/trainees); marque com "⚠️ Fora da trilha" tudo o que vier de fora.
- Formato do notebook: ver `docs/HISTORICO_VERSOES.md`, seção "Formato do notebook".
- Se houver dúvida de decisão, liste no final em vez de decidir em silêncio.
