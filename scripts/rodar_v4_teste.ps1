# Roda o TESTE FINAL da v4 (avaliado uma vez), uma etapa por processo Python, pulando o que já tem marcador.
#
# Uso (PowerShell, na raiz do repositório):
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v4_teste.ps1            # execução oficial
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v4_teste.ps1 -Quick     # teste rápido (bancos de mentira da validação)
# Normalmente é chamado pelo scripts\vigia_v4_teste.ps1, que tenta de novo se algo cair.
#
# - Mesmo padrão do rodar_v4.ps1: env, 1 thread de álgebra linear, HF_HUB_OFFLINE=1, V4_NUM_WORKERS (padrão 4).
# - Etapas: extracao -> cnn -> avaliar; no fim, "concluir" grava results\v4_teste_final_concluido.json (por último).
#   Cada etapa é pulada se results\v4_teste_<etapa>_concluido.json existe; uma etapa incompleta continua de onde parou.
# - Logs: logs\v4_teste_<etapa>.log, logs\v4_teste_rodar.log, logs\v4_teste_resumo.txt, logs\v4_teste_etapa.txt.
# - Encerra workers órfãos do DataLoader; impede a suspensão do Windows enquanto roda.
param([switch]$Quick)

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
if ($env:V4_NUM_WORKERS) { $NumWorkers = [int]$env:V4_NUM_WORKERS }

$Suf = ""
$QArgs = @()
if ($Quick) { $Suf = "_quick" + $env:V4_SUFIXO_EXTRA; $QArgs = @("--quick") }
$Logs = Join-Path $Raiz "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$LogRodar = Join-Path $Logs "v4_teste_rodar$Suf.log"
$ArqEtapa = Join-Path $Logs "v4_teste_etapa$Suf.txt"
$ArqErros = Join-Path $Logs "v4_teste_erros$Suf.txt"

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

function Marcador($etapa) { Join-Path $Raiz "results\v4_teste_$($etapa)_concluido$Suf.json" }

function Limpar-Orfaos {
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
    & $Py scripts\v4_teste.py status @QArgs | Out-Null
}

Add-Type -Namespace Win32 -Name EnergiaV4Teste -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.EnergiaV4Teste]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

$Etapas = @(
    [pscustomobject]@{ Nome = "extracao"; Cmd = "extrair" },
    [pscustomobject]@{ Nome = "cnn"; Cmd = "cnn" },
    [pscustomobject]@{ Nome = "avaliar"; Cmd = "avaliar" }
)
$Codigo = 0
try {
    Log "rodar_v4_teste começou (PID $PID, NUM_WORKERS = $NumWorkers, Quick = $Quick)"
    Limpar-Orfaos
    Atualizar-Resumo
    foreach ($e in $Etapas) {
        if (Test-Path (Marcador $e.Nome)) {
            Log "[$($e.Nome)] já concluída: pulando."
            continue
        }
        Etapa-Atual $e.Nome
        $log = Join-Path $Logs "v4_teste_$($e.Nome)$Suf.log"
        Log "[$($e.Nome)] começando | log: $log"
        $t0 = Get-Date
        & $Py scripts\v4_teste.py $e.Cmd @QArgs --num-workers $NumWorkers --log $log
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
        & $Py scripts\v4_teste.py concluir @QArgs
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $Raiz "results\v4_teste_final_concluido$Suf.json"))) {
            Log "concluir: falhou (código $LASTEXITCODE)"
            $Codigo = 11
        } else {
            Log "teste final concluído: results\v4_teste_final_concluido$Suf.json gravado"
        }
    }
} catch {
    Log "ERRO inesperado no rodar_v4_teste: $_"
    $Codigo = 12
} finally {
    Atualizar-Resumo
    Etapa-Atual "fim (código $Codigo)"
    [void][Win32.EnergiaV4Teste]::SetThreadExecutionState($ES_CONTINUOUS)
    Log "rodar_v4_teste terminou com código $Codigo"
}
exit $Codigo
