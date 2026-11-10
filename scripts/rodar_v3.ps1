# Roda as seis execuções da v3, cada uma num processo Python separado, na ordem intercalada por seed.
#
# Uso (PowerShell, na raiz do repositório, com o Claude Code e de preferência o VS Code fechados):
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v3.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v3.ps1 -NumWorkers 3
#
# - Pula as execuções que já têm marcador de conclusão (results\v3_<cond>_s<seed>_concluido.json).
# - Uma execução incompleta continua de onde parou (o script Python retoma o treino pela última época salva).
# - Se uma execução falhar, registra o erro e segue para a próxima.
# - Log de cada execução: logs\v3_<cond>_s<seed>.log. Resumo atualizado depois de cada uma: logs\v3_resumo.txt.
# - Impede a suspensão do Windows enquanto roda (SetThreadExecutionState) e restaura no fim.
# - Nunca avalia o teste final.
param([int]$NumWorkers = 4)

$ErrorActionPreference = "Continue"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz

# ---- ambiente: env ativado e 1 thread de álgebra linear por processo (memória; ver HISTORICO_VERSOES.md, seção h)
$Ativar = Join-Path $Raiz "env\Scripts\Activate.ps1"
if (Test-Path $Ativar) { . $Ativar }
$Py = Join-Path $Raiz "env\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Host "ERRO: não achei $Py"; exit 1 }
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force (Join-Path $Raiz "logs") | Out-Null

# ---- ordem intercalada por seed
$Execucoes = @()
foreach ($s in 42, 123, 2026) { foreach ($c in "repeat", "zero") { $Execucoes += [pscustomobject]@{ Cond = $c; Seed = $s; Id = "${c}_s$s" } } }

$Resumo = Join-Path $Raiz "logs\v3_resumo.txt"
$Status = @{}
$Minutos = @{}

function Marcador($e) { Join-Path $Raiz "results\v3_$($e.Id)_concluido.json" }

function Pct($x) { if ($null -eq $x) { "-" } else { "{0:N2}%" -f (100 * [double]$x) } }

function Escrever-Resumo {
    $linhas = @("Resumo v3 (atualizado em $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))",
                "Valores de validação/dev_shift lidos do marcador de cada execução. Teste final NÃO avaliado.",
                "",
                ("{0,-16} {1,-26} {2,8} {3,10} {4,18} {5,16} {6,12}" -f "id", "status", "minutos", "pico (GB)",
                 "val EER méd/fonte", "dev_shift EER", "dev_shift AUC"))
    foreach ($e in $Execucoes) {
        $st = if ($Status.ContainsKey($e.Id)) { $Status[$e.Id] } else { "pendente" }
        $min = if ($Minutos.ContainsKey($e.Id)) { "{0:N1}" -f $Minutos[$e.Id] } else { "-" }
        $pico = "-"; $val = "-"; $deer = "-"; $dauc = "-"
        $m = Marcador $e
        if (Test-Path $m) {
            $j = Get-Content $m -Raw -Encoding UTF8 | ConvertFrom-Json
            $pico = "{0:N2}" -f $j.pico_memoria_gb_processo_e_workers
            $val = Pct $j.val_eer_media_por_fonte
            $deer = Pct $j.dev_shift_janelas.EER
            $dauc = if ($null -eq $j.dev_shift_janelas.AUC) { "-" } else { "{0:N3}" -f $j.dev_shift_janelas.AUC }
            if ($min -eq "-") { $min = "{0:N1}*" -f $j.minutos_treino }
        } else {
            $log = Join-Path $Raiz "logs\v3_$($e.Id).log"
            if (Test-Path $log) {
                $p = Select-String -Path $log -Pattern "PICO_MEMORIA_GB=([0-9.]+)" -Encoding UTF8 | Select-Object -Last 1
                if ($p) { $pico = $p.Matches[0].Groups[1].Value + " (log)" }
            }
        }
        $linhas += ("{0,-16} {1,-26} {2,8} {3,10} {4,18} {5,16} {6,12}" -f $e.Id, $st, $min, $pico, $val, $deer, $dauc)
    }
    $linhas += ""
    $linhas += "minutos: tempo deste processo nesta rodada; com * = minutos de treino do marcador (execução de rodada anterior)."
    $linhas += "pico: memória reservada (commit) do processo Python + workers. EER/AUC do dev_shift: avaliação por janelas."
    $tmp = "$Resumo.tmp"
    Set-Content -Path $tmp -Value $linhas -Encoding UTF8
    Move-Item -Force $tmp $Resumo
}

# ---- impede a suspensão do Windows (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) e restaura no fim
Add-Type -Namespace Win32 -Name Energia -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.Energia]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

try {
    Write-Host "v3: $($Execucoes.Count) execuções, NUM_WORKERS = $NumWorkers. Resumo em $Resumo"
    foreach ($e in $Execucoes) {
        if (Test-Path (Marcador $e)) {
            $Status[$e.Id] = "pulado (já concluído)"
            Write-Host "[$($e.Id)] já concluído: pulando."
            Escrever-Resumo
            continue
        }
        $log = Join-Path $Raiz "logs\v3_$($e.Id).log"
        Write-Host "`n[$($e.Id)] começando $(Get-Date -Format 'HH:mm:ss') | log: $log"
        $t0 = Get-Date
        & $Py scripts\treinar_experimento.py --condicao $e.Cond --seed $e.Seed --num-workers $NumWorkers --log $log
        $codigo = $LASTEXITCODE
        $Minutos[$e.Id] = ((Get-Date) - $t0).TotalMinutes
        if ($codigo -eq 0 -and (Test-Path (Marcador $e))) {
            $Status[$e.Id] = "concluído"
        } else {
            $Status[$e.Id] = "FALHOU (código $codigo)"
            $erros = Join-Path $Raiz "logs\v3_erros.txt"
            Add-Content -Path $erros -Encoding UTF8 -Value "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($e.Id): código $codigo (últimas linhas de $log)"
            if (Test-Path $log) { Get-Content $log -Tail 15 -Encoding UTF8 | Add-Content -Path $erros -Encoding UTF8 }
            Write-Host "[$($e.Id)] FALHOU (código $codigo); detalhes em $erros. Seguindo para a próxima."
        }
        Escrever-Resumo
    }
} finally {
    [void][Win32.Energia]::SetThreadExecutionState($ES_CONTINUOUS)
    Escrever-Resumo
    $pend = @($Execucoes | Where-Object { -not (Test-Path (Marcador $_)) } | ForEach-Object { $_.Id })
    Write-Host "`n==== fim $(Get-Date -Format 'HH:mm:ss')"
    Get-Content $Resumo -Encoding UTF8 | Write-Host
    if ($pend.Count -eq 0) {
        Write-Host "`nTodas as execuções têm marcador de conclusão. Próximo passo: abrir o notebook (AVALIAR_TESTE_FINAL continua False)."
    } else {
        Write-Host "`nPendentes ($($pend.Count)): $($pend -join ', '). Rode o mesmo comando de novo para continuar de onde parou."
    }
}
