# CLAUDE.md

Projeto final da liga Insper AI: detector de voz gerada por IA (treino em inglês, teste também em pt-BR).

## Antes de qualquer coisa
- **Leia `docs/HISTORICO_VERSOES.md` inteiro antes de propor ou rodar qualquer versão.** Ao fechar uma versão, registre-a lá.
- Ambiente: Windows, Python 3.13, RTX 4060 8 GB. Ative o env antes de qualquer comando: `env\Scripts\activate`.
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
- `models/vN_*.pt`: checkpoints (fora do git).
- `docs/`: `HISTORICO_VERSOES.md`, `audit_report.md`, `dataset_overview.md`, `datasets.yaml`.

## Não faça
- Não mexa em `data/raw/` (nada é modificado, movido ou apagado lá).
- Não faça commit nem push sem o usuário pedir.

## Estilo
- Texto em português, direto e fácil de entender.
- Use só o que a trilha ensinou (https://trilhas-insperai.vercel.app/trainees); marque com "⚠️ Fora da trilha" tudo o que vier de fora.
- Formato do notebook: ver `docs/HISTORICO_VERSOES.md`, seção "Formato do notebook".
- Se houver dúvida de decisão, liste no final em vez de decidir em silêncio.
