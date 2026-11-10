# Vigia do TESTE FINAL da v4: roda scripts\rodar_v4_teste.ps1 sozinho e tenta de novo se algo cair (mesmo padrão do vigia_v4.ps1; marcador final results\v4_teste_final_concluido.json).
#
# Lançar (na raiz do repositório), num processo oculto que sobrevive ao terminal que o lançou:
#     Start-Process powershell.exe -WindowStyle Hidden -ArgumentList "-ExecutionPolicy Bypass -File scripts\vigia_v4_teste.ps1"
# Conferir:  Get-Content logs\v4_teste_status.txt   e   Get-Content logs\v4_teste_resumo.txt
#
# Regras:
# - Instância única: logs\v4_teste_vigia.pid guarda o PID; se ele estiver vivo e for um PowerShell, não cria outro.
# - Impede a suspensão do Windows durante toda a execução (SetThreadExecutionState) e restaura ao final.
# - Antes de cada tentativa, confere a folga de memória reservada (Limite de confirmação - Bytes confirmados,
#   via Get-Counter; se o Get-Counter falhar, via Win32_OperatingSystem). Se for menor que -FolgaGB (14),
#   espera -EsperaMemoriaMin (5) e confere de novo, até -MaxEsperaMemoriaMin (60); depois tenta mesmo assim.
# - Tentativas 1 e 2 com V4_NUM_WORKERS=4; da 3ª em diante, 2. Se o rodar_v4_teste.ps1 terminar com erro ou sem
#   o marcador de conclusão, espera -EsperaFalhaMin (10) e tenta de novo, até -MaxTentativas (8).
#   Tudo é retomável, então o progresso não se perde.
# - logs\v4_teste_status.txt (a cada mudança e a cada -IntervaloStatusS segundos): STATUS (AGUARDANDO_MEMORIA,
#   RODANDO, CONCLUIDO ou FALHOU), tentativa, etapa atual, hora da última atualização, horas de extração gastas.
# - No fim, grava logs\v4_teste_FIM.txt com o resultado resumido.
param(
    [switch]$Quick,
    [double]$FolgaGB = 14,
    [double]$EsperaMemoriaMin = 5,
    [double]$MaxEsperaMemoriaMin = 60,
    [double]$EsperaFalhaMin = 10,
    [int]$MaxTentativas = 8,
    [int]$IntervaloStatusS = 60
)

$ErrorActionPreference = "Continue"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz
$Suf = ""
if ($Quick) { $Suf = "_quick" + $env:V4_SUFIXO_EXTRA }   # V4_SUFIXO_EXTRA: só testes do QUICK
$Logs = Join-Path $Raiz "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$ArqPid = Join-Path $Logs "v4_teste_vigia$Suf.pid"
$ArqStatus = Join-Path $Logs "v4_teste_status$Suf.txt"
$ArqFim = Join-Path $Logs "v4_teste_FIM$Suf.txt"
$LogVigia = Join-Path $Logs "v4_teste_vigia$Suf.log"
$ArqEtapa = Join-Path $Logs "v4_teste_etapa$Suf.txt"
$ArqEstado = Join-Path $Raiz "results\v4_teste_estado$Suf.json"
$ArqConcluido = Join-Path $Raiz "results\v4_teste_final_concluido$Suf.json"
$ArqResumo = Join-Path $Logs "v4_teste_resumo$Suf.txt"

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

function Horas-Extracao {
    try {
        if (Test-Path $ArqEstado) { return (Get-Content $ArqEstado -Raw -Encoding UTF8 | ConvertFrom-Json).horas_extracao }
    } catch { }
    return 0
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
        "HORAS_EXTRACAO=$(Horas-Extracao)",
        "FOLGA_MEMORIA_GB=$($script:Folga) (mínimo pedido: $FolgaGB)",
        "NUM_WORKERS=$($script:Workers)",
        "PID_VIGIA=$PID"
    )
    $tmp = "$ArqStatus.tmp"
    Set-Content -Path $tmp -Value $txt -Encoding UTF8
    Move-Item -Force $tmp $ArqStatus
}

Add-Type -Namespace Win32 -Name EnergiaVigiaTeste -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.EnergiaVigiaTeste]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

function Dormir($segundos, $st, $etapa) {
    # dorme em pedaços, atualizando o status (e mantendo o PC acordado)
    $fim = (Get-Date).AddSeconds($segundos)
    while ((Get-Date) -lt $fim) {
        [void][Win32.EnergiaVigiaTeste]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
        Status $st $etapa
        $resta = ($fim - (Get-Date)).TotalSeconds
        Start-Sleep -Seconds ([math]::Max(1, [math]::Min($IntervaloStatusS, $resta)))
    }
}

$Final = "FALHOU"
try {
    Log "vigia começou (Quick = $Quick, FolgaGB = $FolgaGB, MaxTentativas = $MaxTentativas)"
    for ($t = 1; $t -le $MaxTentativas; $t++) {
        $script:Tentativa = $t
        if (Test-Path $ArqConcluido) { $Final = "CONCLUIDO"; break }

        # folga de memória
        $esperou = 0
        $script:Folga = Folga-GB
        while ($script:Folga -lt $FolgaGB -and $esperou -lt $MaxEsperaMemoriaMin) {
            Log "folga de memória $($script:Folga) GB < $FolgaGB GB; esperando $EsperaMemoriaMin min (já esperou $esperou de $MaxEsperaMemoriaMin)"
            Dormir ($EsperaMemoriaMin * 60) "AGUARDANDO_MEMORIA" "antes da tentativa $t (folga $($script:Folga) GB; esperou $esperou min)"
            $esperou += $EsperaMemoriaMin
            $script:Folga = Folga-GB
        }
        if ($script:Folga -lt $FolgaGB) { Log "folga ainda $($script:Folga) GB depois de $esperou min; tentando mesmo assim" }

        $script:Workers = if ($t -le 2) { 4 } else { 2 }
        $env:V4_NUM_WORKERS = "$($script:Workers)"
        $args_ = "-NoProfile -ExecutionPolicy Bypass -File `"$Raiz\scripts\rodar_v4_teste.ps1`""
        if ($Quick) { $args_ += " -Quick" }
        Log "tentativa $t : lançando rodar_v4_teste (NUM_WORKERS = $($script:Workers), folga $($script:Folga) GB)"
        Set-Content -Path $ArqEtapa -Value "iniciando a tentativa $t | desde $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Encoding UTF8
        $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $args_ -WorkingDirectory $Raiz -WindowStyle Hidden -PassThru
        $null = $proc.Handle          # sem isso, ExitCode pode vir vazio
        Log "tentativa $t : rodar_v4_teste PID $($proc.Id)"
        while (-not $proc.WaitForExit($IntervaloStatusS * 1000)) {
            [void][Win32.EnergiaVigiaTeste]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
            Status "RODANDO" (Etapa-Lida)
        }
        $proc.WaitForExit()
        $codigo = $proc.ExitCode
        Log "tentativa $t : rodar_v4_teste terminou com código $codigo"
        if ($codigo -eq 0 -and (Test-Path $ArqConcluido)) { $Final = "CONCLUIDO"; break }
        if ($t -lt $MaxTentativas) {
            Log "tentativa $t falhou (código $codigo, marcador de conclusão ausente); nova tentativa em $EsperaFalhaMin min"
            Dormir ($EsperaFalhaMin * 60) "RODANDO" "espera de $EsperaFalhaMin min antes da tentativa $($t + 1) (a $t terminou com código $codigo)"
        }
    }
} catch {
    Log "ERRO inesperado no vigia: $_"
} finally {
    $etapa = if ($Final -eq "CONCLUIDO") { "fim" } else { "fim sem conclusão (ver logs\v4_teste_erros$Suf.txt e logs\v4_teste_vigia$Suf.log)" }
    Status $Final $etapa
    $fim = @("v4 (teste final) $Final em $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | tentativas usadas: $($script:Tentativa) de $MaxTentativas | horas de extração: $(Horas-Extracao)", "")
    if (Test-Path $ArqResumo) { $fim += Get-Content $ArqResumo -Encoding UTF8 }
    Set-Content -Path $ArqFim -Value $fim -Encoding UTF8
    Log "vigia terminou: $Final"
    [void][Win32.EnergiaVigiaTeste]::SetThreadExecutionState($ES_CONTINUOUS)
    Remove-Item -Force $ArqPid -ErrorAction SilentlyContinue
}
