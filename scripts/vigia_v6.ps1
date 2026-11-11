# Vigia da v6: roda scripts\rodar_v6.ps1 sozinho e tenta de novo se algo cair (execução noturna sem supervisão).
# Cópia do scripts\vigia_v4.ps1 com os nomes da v6 e as regras desta rodada.
#
# Lançar (na raiz do repositório), num processo oculto que sobrevive ao terminal que o lançou:
#     Start-Process powershell.exe -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File scripts\vigia_v6.ps1"
# Conferir:  Get-Content logs\v6_status.txt   |   Get-Content logs\v6_resumo.txt   |   logs\v6_PROGRESSO.md
#
# Regras:
# - Instância única: logs\v6_vigia.pid guarda o PID; se ele estiver vivo e for um PowerShell, não cria outro.
# - Impede a suspensão do Windows durante toda a execução e restaura ao final.
# - Antes de cada tentativa, confere a folga de memória reservada (Limite de confirmação - Bytes confirmados).
#   Se for menor que -FolgaGB (11), espera em pedaços de -EsperaMemoriaMin (2) até -MaxEsperaMemoriaMin (10);
#   depois roda mesmo assim.
# - Tentativas 1 e 2 com V6_NUM_WORKERS=4; da 3ª em diante, 2. Se o rodar_v6.ps1 terminar com erro, espera
#   -EsperaFalhaMin (2) e tenta de novo, até -MaxTentativas (8) ou até passar -MaxHoras (7) desde o início.
#   Tudo é retomável, então o progresso não se perde.
# - logs\v6_status.txt (a cada -IntervaloStatusS segundos): STATUS (AGUARDANDO_MEMORIA, RODANDO, CONCLUIDO ou
#   FALHOU), tentativa, etapa atual, hora, folga de memória. No fim, logs\v6_FIM.txt; se não concluir,
#   também logs\v6_BLOQUEIO.txt com o motivo (últimos erros).
param(
    [switch]$Quick,
    [double]$FolgaGB = 11,
    [double]$EsperaMemoriaMin = 2,
    [double]$MaxEsperaMemoriaMin = 10,
    [double]$EsperaFalhaMin = 2,
    [int]$MaxTentativas = 8,
    [double]$MaxHoras = 7,
    [int]$IntervaloStatusS = 60
)

$ErrorActionPreference = "Continue"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz
$Suf = ""
if ($Quick) { $Suf = "_quick" + $env:V6_SUFIXO_EXTRA }   # V6_SUFIXO_EXTRA: só testes do QUICK
$Logs = Join-Path $Raiz "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$ArqPid = Join-Path $Logs "v6_vigia$Suf.pid"
$ArqStatus = Join-Path $Logs "v6_status$Suf.txt"
$ArqFim = Join-Path $Logs "v6_FIM$Suf.txt"
$ArqBloqueio = Join-Path $Logs "v6_BLOQUEIO$Suf.txt"
$LogVigia = Join-Path $Logs "v6_vigia$Suf.log"
$ArqEtapa = Join-Path $Logs "v6_etapa$Suf.txt"
$ArqErros = Join-Path $Logs "v6_erros$Suf.txt"
$ArqConcluido = Join-Path $Raiz "results\v6_concluido$Suf.json"
$ArqResumo = Join-Path $Logs "v6_resumo$Suf.txt"
$Inicio = Get-Date

function Log($msg) {
    $linha = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | vigia $PID | $msg"
    Add-Content -Path $LogVigia -Value $linha -Encoding UTF8
}

# ---------------- instância única
if (Test-Path $ArqPid) {
    $velho = 0
    [void][int]::TryParse(((Get-Content $ArqPid -ErrorAction SilentlyContinue) | Select-Object -First 1), [ref]$velho)
    if ($velho -gt 0 -and $velho -ne $PID) {
        $p = Get-Process -Id $velho -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -match "powershell|pwsh") {
            Log "já existe um vigia vivo (PID $velho); este não faz nada."
            exit 0
        }
    }
}
Set-Content -Path $ArqPid -Value $PID -Encoding ASCII

# ---------------- memória reservada (commit)
$script:Contadores = $null
function Folga-GB {
    try {
        if ($null -eq $script:Contadores) {
            $set = Get-Counter -ListSet "Mem*" -ErrorAction Stop | Where-Object { @($_.Paths | Where-Object { $_ -match 'Limit' }).Count -ge 1 } | Select-Object -First 1
            $lim = @($set.Paths | Where-Object { $_ -match 'Limit' })[0]
            $com = @($set.Paths | Where-Object { $_ -notmatch 'Limit' -and $_ -notmatch '%' -and $_ -match 'Commit|onfirm' })[0]
            if (-not $lim -or -not $com) { throw "contadores de memória não encontrados" }
            $script:Contadores = @($lim, $com)
        }
        $s = (Get-Counter -Counter $script:Contadores -ErrorAction Stop).CounterSamples
        return [math]::Round(($s[0].CookedValue - $s[1].CookedValue) / 1GB, 2)
    } catch {
        $os = Get-CimInstance Win32_OperatingSystem
        return [math]::Round($os.FreeVirtualMemory / 1MB, 2)      # KB -> GB
    }
}

function Etapa-Lida {
    try { if (Test-Path $ArqEtapa) { return ((Get-Content $ArqEtapa -Encoding UTF8) | Select-Object -First 1) } } catch { }
    return "-"
}

$script:Tentativa = 0
$script:Workers = "-"
$script:Folga = "-"
function Status($st, $etapa) {
    $txt = @(
        "STATUS=$st",
        "TENTATIVA=$($script:Tentativa) de $MaxTentativas",
        "ETAPA=$etapa",
        "ATUALIZADO=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        ("DESDE=$($Inicio.ToString('yyyy-MM-dd HH:mm:ss')) ({0:N2} h de no máximo $MaxHoras h)" -f ((Get-Date) - $Inicio).TotalHours),
        "FOLGA_MEMORIA_GB=$($script:Folga) (mínimo pedido: $FolgaGB)",
        "NUM_WORKERS=$($script:Workers)",
        "PID_VIGIA=$PID"
    )
    $tmp = "$ArqStatus.tmp"
    Set-Content -Path $tmp -Value $txt -Encoding UTF8
    Move-Item -Force $tmp $ArqStatus
}

Add-Type -Namespace Win32 -Name EnergiaVigiaV6 -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.EnergiaVigiaV6]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

function Dormir($segundos, $st, $etapa) {
    $fim = (Get-Date).AddSeconds($segundos)
    while ((Get-Date) -lt $fim) {
        [void][Win32.EnergiaVigiaV6]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
        Status $st $etapa
        $resta = ($fim - (Get-Date)).TotalSeconds
        Start-Sleep -Seconds ([math]::Max(1, [math]::Min($IntervaloStatusS, $resta)))
    }
}

$Final = "FALHOU"
$Motivo = ""
try {
    Log "vigia começou (Quick = $Quick, FolgaGB = $FolgaGB, MaxTentativas = $MaxTentativas, MaxHoras = $MaxHoras)"
    for ($t = 1; $t -le $MaxTentativas; $t++) {
        $script:Tentativa = $t
        if (Test-Path $ArqConcluido) { $Final = "CONCLUIDO"; break }
        if (((Get-Date) - $Inicio).TotalHours -gt $MaxHoras) { $Motivo = "passou de $MaxHoras h desde o início"; Log $Motivo; break }

        $esperou = 0
        $script:Folga = Folga-GB
        while ($script:Folga -lt $FolgaGB -and $esperou -lt $MaxEsperaMemoriaMin) {
            Log "folga de memória $($script:Folga) GB < $FolgaGB GB; esperando $EsperaMemoriaMin min (já esperou $esperou de $MaxEsperaMemoriaMin)"
            Dormir ($EsperaMemoriaMin * 60) "AGUARDANDO_MEMORIA" "antes da tentativa $t (folga $($script:Folga) GB; esperou $esperou min)"
            $esperou += $EsperaMemoriaMin
            $script:Folga = Folga-GB
        }
        if ($script:Folga -lt $FolgaGB) { Log "folga ainda $($script:Folga) GB depois de $esperou min; rodando mesmo assim" }

        $script:Workers = if ($t -le 2) { 4 } else { 2 }
        $env:V6_NUM_WORKERS = "$($script:Workers)"
        $args_ = "-NoProfile -ExecutionPolicy Bypass -File `"$Raiz\scripts\rodar_v6.ps1`""
        if ($Quick) { $args_ += " -Quick" }
        Log "tentativa $t : lançando rodar_v6 (NUM_WORKERS = $($script:Workers), folga $($script:Folga) GB)"
        Set-Content -Path $ArqEtapa -Value "iniciando a tentativa $t | desde $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
        $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $args_ -WorkingDirectory $Raiz -WindowStyle Hidden -PassThru
        $null = $proc.Handle
        Log "tentativa $t : rodar_v6 PID $($proc.Id)"
        while (-not $proc.WaitForExit($IntervaloStatusS * 1000)) {
            [void][Win32.EnergiaVigiaV6]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
            Status "RODANDO" (Etapa-Lida)
        }
        $proc.WaitForExit()
        $codigo = $proc.ExitCode
        Log "tentativa $t : rodar_v6 terminou com código $codigo"
        if ($codigo -eq 0 -and (Test-Path $ArqConcluido)) { $Final = "CONCLUIDO"; break }
        $Motivo = "rodar_v6 terminou com código $codigo na tentativa $t"
        if ($t -lt $MaxTentativas) {
            Log "tentativa $t falhou (código $codigo); nova tentativa em $EsperaFalhaMin min"
            Dormir ($EsperaFalhaMin * 60) "RODANDO" "espera de $EsperaFalhaMin min antes da tentativa $($t + 1) (a $t terminou com código $codigo)"
        }
    }
} catch {
    Log "ERRO inesperado no vigia: $_"
    $Motivo = "erro inesperado no vigia: $_"
} finally {
    $etapa = if ($Final -eq "CONCLUIDO") { "fim" } else { "fim sem conclusão (ver logs\v6_erros$Suf.txt e logs\v6_vigia$Suf.log)" }
    Status $Final $etapa
    $linha1 = "v6 $Final em $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | tentativas usadas: $($script:Tentativa) de $MaxTentativas | " + ("{0:N2} h desde o início" -f ((Get-Date) - $Inicio).TotalHours)
    $fim = @($linha1, "")
    if (Test-Path $ArqResumo) { $fim += Get-Content $ArqResumo -Encoding UTF8 }
    Set-Content -Path $ArqFim -Value $fim -Encoding UTF8
    if ($Final -ne "CONCLUIDO") {
        $b = @("v6 BLOQUEADA em $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Motivo", "", "Últimos erros (logs\v6_erros$Suf.txt):")
        if (Test-Path $ArqErros) { $b += Get-Content $ArqErros -Tail 60 -Encoding UTF8 }
        Set-Content -Path $ArqBloqueio -Value $b -Encoding UTF8
    }
    Log "vigia terminou: $Final"
    [void][Win32.EnergiaVigiaV6]::SetThreadExecutionState($ES_CONTINUOUS)
    Remove-Item -Force $ArqPid -ErrorAction SilentlyContinue
}
