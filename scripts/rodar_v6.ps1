# Roda a v6 inteira, uma etapa por processo Python, pulando o que já tem marcador de conclusão.
#
# Uso (PowerShell, na raiz do repositório):
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v6.ps1            # execução oficial
#     powershell -ExecutionPolicy Bypass -File scripts\rodar_v6.ps1 -Quick     # teste rápido (arquivos _quick)
# Normalmente é chamado pelo scripts\vigia_v6.ps1, que tenta de novo se algo cair.
#
# - Ativa o env; 1 thread de álgebra linear por processo; HF_HUB_OFFLINE=1 (os modelos já estão no cache local).
# - Workers do DataLoader: variável V6_NUM_WORKERS (padrão 4). Um processo pesado por vez.
# - Etapas (marcador results\v6_<etapa>_concluido.json): medir -> extracao -> e1 -> e2 -> e3 -> e4 -> e5 ->
#   finalistas -> teste_extracao -> teste (E6, scripts\v6_teste.py, só depois dos finalistas) -> relatorio -> notebook.
#   Uma etapa com marcador é pulada; uma etapa cuja dependência falta é pulada e registrada; se uma etapa
#   falhar, registra em logs\v6_erros.txt e SEGUE para as que não dependem dela (o relatório e o notebook
#   sempre rodam, com o que houver). O relatório e o notebook são refeitos quando há marcador mais novo.
# - Logs: logs\v6_<etapa>.log, logs\v6_rodar.log, logs\v6_resumo.txt e logs\v6_PROGRESSO.md (a cada etapa),
#   logs\v6_etapa.txt (etapa atual, lido pelo vigia). Orçamento de tempo por etapa: se estourar, registra.
# - Encerra workers do DataLoader órfãos; impede a suspensão do Windows. Código 0 só com tudo concluído.
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
if ($env:V6_NUM_WORKERS) { $NumWorkers = [int]$env:V6_NUM_WORKERS }

$Suf = ""
$QArgs = @()
if ($Quick) { $Suf = "_quick" + $env:V6_SUFIXO_EXTRA; $QArgs = @("--quick") }   # extra: só testes
$Logs = Join-Path $Raiz "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$LogRodar = Join-Path $Logs "v6_rodar$Suf.log"
$ArqEtapa = Join-Path $Logs "v6_etapa$Suf.txt"
$ArqErros = Join-Path $Logs "v6_erros$Suf.txt"

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

function Marcador($etapa) { Join-Path $Raiz "results\v6_$($etapa)_concluido$Suf.json" }

function Limpar-Orfaos {
    # Workers do DataLoader cujo processo principal já morreu podem ficar vivos segurando ~1 GB cada.
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

function Atualizar-Resumo { & $Py scripts\v6.py status @QArgs | Out-Null }

Add-Type -Namespace Win32 -Name EnergiaV6 -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Win32.EnergiaV6]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

# Nome, script, comando, dependências, orçamento (min), obrigatória para concluir, refazer se houver marcador mais novo
$Etapas = @(
    [pscustomobject]@{ Nome = "medir";          Script = "scripts\v6.py";          Cmd = "medir";      Deps = @();                 Min = 15;  Obrig = $false; Refaz = $false },
    [pscustomobject]@{ Nome = "extracao";       Script = "scripts\v6.py";          Cmd = "extrair";    Deps = @();                 Min = 180; Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "e1";             Script = "scripts\v6.py";          Cmd = "e1";         Deps = @("extracao");       Min = 40;  Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "e2";             Script = "scripts\v6.py";          Cmd = "e2";         Deps = @("e1");             Min = 20;  Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "e3";             Script = "scripts\v6.py";          Cmd = "e3";         Deps = @("e2");             Min = 10;  Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "e4";             Script = "scripts\v6.py";          Cmd = "e4";         Deps = @("e3");             Min = 60;  Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "e5";             Script = "scripts\v6.py";          Cmd = "e5";         Deps = @("e4");             Min = 60;  Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "finalistas";     Script = "scripts\v6.py";          Cmd = "finalistas"; Deps = @("e5");             Min = 5;   Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "teste_extracao"; Script = "scripts\v6_teste.py";    Cmd = "extrair";    Deps = @("finalistas");     Min = 120; Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "teste";          Script = "scripts\v6_teste.py";    Cmd = "avaliar";    Deps = @("teste_extracao"); Min = 30;  Obrig = $true;  Refaz = $false },
    [pscustomobject]@{ Nome = "relatorio";      Script = "scripts\v6.py";          Cmd = "relatorio";  Deps = @("e1");             Min = 20;  Obrig = $true;  Refaz = $true },
    [pscustomobject]@{ Nome = "notebook";       Script = "scripts\v6_notebook.py"; Cmd = "";           Deps = @();                 Min = 15;  Obrig = $true;  Refaz = $true }
)
$Codigo = 0
$Falhas = @()
try {
    Log "rodar_v6 começou (PID $PID, NUM_WORKERS = $NumWorkers, Quick = $Quick)"
    Limpar-Orfaos
    Atualizar-Resumo
    foreach ($e in $Etapas) {
        $m = Marcador $e.Nome
        if (Test-Path $m) {
            $refazer = $false
            if ($e.Refaz) {
                $t = (Get-Item $m).LastWriteTime
                $mais_novo = @(Get-ChildItem (Join-Path $Raiz "results") -Filter "v6_*_concluido$Suf.json" |
                    Where-Object { $_.Name -notmatch "relatorio|notebook" -and $_.LastWriteTime -gt $t })
                if ($e.Nome -eq "notebook") {
                    $rel = Marcador "relatorio"
                    if ((Test-Path $rel) -and (Get-Item $rel).LastWriteTime -gt $t) { $mais_novo += @($rel) }
                }
                if ($mais_novo.Count -gt 0) { $refazer = $true }
            }
            if (-not $refazer) { Log "[$($e.Nome)] já concluída: pulando."; continue }
            Log "[$($e.Nome)] há resultados mais novos que ela: refazendo"
            Move-Item -Force $m "$m.antigo"
        }
        $falta = @($e.Deps | Where-Object { -not (Test-Path (Marcador $_)) })
        if ($falta.Count -gt 0) {
            Log "[$($e.Nome)] pulada: falta $($falta -join ', ')"
            continue
        }
        Etapa-Atual $e.Nome
        $log = Join-Path $Logs "v6_$($e.Nome)$Suf.log"
        Log "[$($e.Nome)] começando | log: $log"
        $t0 = Get-Date
        if ($e.Cmd) {
            & $Py $e.Script $e.Cmd @QArgs --num-workers $NumWorkers --log $log
        } else {
            & $Py $e.Script @QArgs --log $log
        }
        $c = $LASTEXITCODE
        $min = ((Get-Date) - $t0).TotalMinutes
        Limpar-Orfaos
        Atualizar-Resumo
        if ($min -gt $e.Min) { Log ("[{0}] ESTOUROU o orçamento: {1:N1} min (previsto {2} min); seguindo" -f $e.Nome, $min, $e.Min) }
        if ($c -eq 0 -and (Test-Path $m)) {
            Log ("[{0}] concluída em {1:N1} min" -f $e.Nome, $min)
            if (Test-Path "$m.antigo") { Remove-Item -Force "$m.antigo" -ErrorAction SilentlyContinue }
        } else {
            Log ("[{0}] FALHOU (código {1}) depois de {2:N1} min; ver {3} e {4}" -f $e.Nome, $c, $min, $log, $ArqErros)
            Add-Content -Path $ArqErros -Encoding UTF8 -Value "==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($e.Nome): código $c (últimas linhas de $log)"
            if (Test-Path $log) { Get-Content $log -Tail 25 -Encoding UTF8 | Add-Content -Path $ArqErros -Encoding UTF8 }
            if ((Test-Path "$m.antigo") -and -not (Test-Path $m)) { Move-Item -Force "$m.antigo" $m }   # mantém a versão anterior
            $Falhas += $e.Nome
            if ($e.Obrig) { $Codigo = 10 }
        }
    }
    if ($Codigo -eq 0) {
        $faltam = @($Etapas | Where-Object { $_.Obrig -and -not (Test-Path (Marcador $_.Nome)) } | ForEach-Object { $_.Nome })
        if ($faltam.Count -gt 0) { Log "incompleta: faltam $($faltam -join ', ')"; $Codigo = 11 }
        else {
            Etapa-Atual "concluir"
            & $Py scripts\v6.py concluir @QArgs | Out-Null
            if (Test-Path (Join-Path $Raiz "results\v6_concluido$Suf.json")) { Log "v6 CONCLUÍDA: results\v6_concluido$Suf.json" }
            else { Log "concluir falhou"; $Codigo = 12 }
        }
    }
} catch {
    Log "ERRO inesperado no rodar_v6: $_"
    $Codigo = 13
} finally {
    Atualizar-Resumo
    Etapa-Atual "fim (código $Codigo; falhas: $($Falhas -join ', '))"
    [void][Win32.EnergiaV6]::SetThreadExecutionState($ES_CONTINUOUS)
    Log "rodar_v6 terminou com código $Codigo"
}
exit $Codigo
