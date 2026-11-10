# Roda a v4 inteira, uma etapa por processo Python, pulando o que já tem marcador de conclusão.
#
# Uso (PowerShell, na raiz do repositório):
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1            # execução oficial
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v4.ps1 -Quick     # teste rápido (arquivos _quick)
# Normalmente é chamado pelo scripts\vigia_v4.ps1, que tenta de novo se algo cair.
#
# - Ativa o env; 1 thread de álgebra linear por processo; HF_HUB_OFFLINE=1 (o modelo já está no cache local).
# - Workers do DataLoader: variável V4_NUM_WORKERS (padrão 4).
# - Etapas: extracao -> camadas -> somas -> atalhos -> resumo; no fim, "concluir" grava results\v4_concluido.json.
#   Cada etapa é pulada se results\v4_<etapa>_concluido.json existe; uma etapa incompleta continua de onde parou.
# - Se uma etapa falhar, registra em logs\v4_erros.txt e termina com código diferente de zero (as seguintes
#   dependem dela). Rodar de novo continua de onde parou.
# - Logs: logs\v4_<etapa>.log (saída do Python), logs\v4_rodar.log (este script), logs\v4_resumo.txt
#   (atualizado depois de cada etapa), logs\v4_etapa.txt (etapa atual, lido pelo vigia).
# - Na execução oficial, depois do marcador de conclusão, executa o notebook.ipynb (só lê do disco; falha não
#   desfaz a conclusão).
# - Encerra workers do DataLoader órfãos (pai já morto) no início e depois de cada etapa.
# - Impede a suspensão do Windows enquanto roda e restaura no fim. Nunca avalia o teste final.
# - v5 (-Modelo xlsr ou -Modelo wavlm): mesmos passos, arquivos com prefixo v5_<modelo> (results\v5_wavlm_*,
#   logs\v5_wavlm_*, cache data\processed\v5_cache_wavlm); o modelo vai para o notebook pela variável V5_MODELO.
#   Na v5 o notebook NÃO é executado aqui (ele compara os dois modelos; roda depois que os dois terminarem).
param([switch]$Quick, [ValidateSet("", "xlsr", "wavlm")][string]$Modelo = "")

$ErrorActionPreference = "Continue"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz

$Ativar = Join-Path $Raiz "env\Scripts\Activate.ps1"
if (Test-Path $Ativar) { . $Ativar }
$Py = Join-Path $Raiz "env\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Host "ERRO: não achei $Py"; exit 1 }
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
$NumWorkers = 4
$P = "v4"
if ($Modelo) { $P = "v5_$Modelo"; $env:V5_MODELO = $Modelo }
if ($env:V4_NUM_WORKERS) { $NumWorkers = [int]$env:V4_NUM_WORKERS }

$Suf = ""
$QArgs = @()
if ($Quick) { $Suf = "_quick" + $env:V4_SUFIXO_EXTRA; $QArgs = @("--quick") }   # extra: só testes
$Logs = Join-Path $Raiz "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$LogRodar = Join-Path $Logs "$($P)_rodar$Suf.log"
$ArqEtapa = Join-Path $Logs "$($P)_etapa$Suf.txt"
$ArqErros = Join-Path $Logs "$($P)_erros$Suf.txt"

function Log($msg) {
    $linha = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $msg"
    Write-Host $linha
    Add-Content -Path $LogRodar -Value $linha -Encoding UTF8
}

function Etapa-Atual($nome) {
    $tmp = "$ArqEtapa.tmp"
    Set-Content -Path $tmp -Value "$nome | desde $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
    Move-Item -Force $tmp $ArqEtapa
}

function Marcador($etapa) { Join-Path $Raiz "results\$($P)_$($etapa)_concluido$Suf.json" }

function Limpar-Orfaos {
    # Workers do DataLoader cujo processo principal já morreu (ex.: queda ou interrupção à força) podem
    # ficar vivos segurando ~1 GB cada. Só encerra processos python "spawn_main(parent_pid=N)" com N morto.
    try {
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
            if ($_.CommandLine -match 'spawn_main\(parent_pid=(\d+)') {
                $pai = [int]$Matches[1]
                if (-not (Get-Process -Id $pai -ErrorAction SilentlyContinue)) {
                    Log "encerrando worker órfão PID $($_.ProcessId) (pai $pai já terminou)"
                    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                }
            }
        }
    } catch { Log "aviso: limpeza de órfãos falhou: $_" }
}

function Atualizar-Resumo {
    & $Py scripts\v4.py status @QArgs | Out-Null
}

Add-Type -Namespace Win32 -Name EnergiaV4 -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.EnergiaV4]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

$Etapas = @(
    [pscustomobject]@{ Nome = "extracao"; Cmd = "extrair" },
    [pscustomobject]@{ Nome = "camadas"; Cmd = "camadas" },
    [pscustomobject]@{ Nome = "somas"; Cmd = "somas" },
    [pscustomobject]@{ Nome = "atalhos"; Cmd = "atalhos" },
    [pscustomobject]@{ Nome = "resumo"; Cmd = "resumo" }
)
$Codigo = 0
try {
    Log "rodar_v4 começou (PID $PID, prefixo $P, NUM_WORKERS = $NumWorkers, Quick = $Quick)"
    Limpar-Orfaos
    Atualizar-Resumo
    foreach ($e in $Etapas) {
        if (Test-Path (Marcador $e.Nome)) {
            Log "[$($e.Nome)] já concluída: pulando."
            continue
        }
        Etapa-Atual $e.Nome
        $log = Join-Path $Logs "$($P)_$($e.Nome)$Suf.log"
        Log "[$($e.Nome)] começando | log: $log"
        $t0 = Get-Date
        & $Py scripts\v4.py $e.Cmd @QArgs --num-workers $NumWorkers --log $log
        $c = $LASTEXITCODE
        $min = ((Get-Date) - $t0).TotalMinutes
        Limpar-Orfaos
        Atualizar-Resumo
        if ($c -eq 0 -and (Test-Path (Marcador $e.Nome))) {
            Log ("[{0}] concluída em {1:N1} min" -f $e.Nome, $min)
        } else {
            Log ("[{0}] FALHOU (código {1}) depois de {2:N1} min; ver {3} e {4}" -f $e.Nome, $c, $min, $log, $ArqErros)
            Add-Content -Path $ArqErros -Encoding UTF8 -Value "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($e.Nome): código $c (últimas linhas de $log)"
            if (Test-Path $log) { Get-Content $log -Tail 20 -Encoding UTF8 | Add-Content -Path $ArqErros -Encoding UTF8 }
            $Codigo = 10
            break
        }
    }
    if ($Codigo -eq 0) {
        Etapa-Atual "concluir"
        & $Py scripts\v4.py concluir @QArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $Raiz "results\$($P)_concluido$Suf.json"))) {
            Log "concluir: falhou (código $LASTEXITCODE)"
            $Codigo = 11
        } else {
            Log "$P concluída: results\$($P)_concluido$Suf.json gravado"
            if (-not $Quick -and -not $Modelo) {
                # o notebook só LÊ do disco; executá-lo aqui deixa as saídas prontas (não altera resultado nenhum).
                # Se falhar, o notebook fica como estava e a v4 continua concluída.
                Etapa-Atual "notebook"
                $logNb = Join-Path $Logs "v4_notebook.log"
                Log "executando o notebook (nbconvert --inplace) | log: $logNb"
                & $Py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb *> $logNb
                if ($LASTEXITCODE -eq 0) { Log "notebook executado" } else { Log "aviso: o notebook não rodou (código $LASTEXITCODE); ver $logNb" }
            }
        }
    }
} catch {
    Log "ERRO inesperado no rodar_v4: $_"
    $Codigo = 12
} finally {
    Atualizar-Resumo
    Etapa-Atual "fim (código $Codigo)"
    [void][Win32.EnergiaV4]::SetThreadExecutionState($ES_CONTINUOUS)
    Log "rodar_v4 terminou com código $Codigo"
}
exit $Codigo
