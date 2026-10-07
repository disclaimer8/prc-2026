param([ValidateSet('Install','Run')][string]$Mode = 'Install')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'PowerShell 7 required' }
$Root = 'C:/Users/admin/prc-2026'
$Audit = Join-Path $Root 'analysis/adsb_stage_audit_20261007'
$Bundle = Join-Path $Audit 'portable_inputs'
$Output = Join-Path $Audit 'results'
$Execution = Join-Path $Audit 'execution'
$Task = 'PRC-ADS-B-Stage-Audit-20261007'
$Registration = 'f2fc89603be4a1521860a98b2adb641d392a3e09'
$ManifestSHA = '3bc8613751d0118e3bdf5bfed7326fc168bdabeef17a05c6a50b288cdf76794c'

function Record([string]$Name, $Value) {
    $Path = Join-Path $Execution $Name
    $Stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $Bytes = [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 8))
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally { $Stream.Dispose() }
}

if ((Get-FileHash (Join-Path $Bundle 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ManifestSHA) { throw 'Manifest differs' }
if ($Mode -eq 'Install') {
    if (Get-ScheduledTask -TaskName $Task -ErrorAction SilentlyContinue) { throw 'Task already exists; no retry' }
    if ((Test-Path $Output) -or (Test-Path $Execution)) { throw 'Audit execution already exists; no retry' }
    $Watcher = Get-ScheduledTask -TaskName 'PRC-Ally-VRAM-After-Training-20261006'
    if ($Watcher.State -eq 'Running' -or @($Watcher.Triggers).Where({$null -ne $_}).Count -gt 0) { throw 'Old reboot watcher active or scheduled' }
    $WatcherProcesses = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'ally_vram_finish.ps1.*-Mode Watch' })
    if ($WatcherProcesses.Count -gt 0) { throw 'Old reboot watcher process active' }
    New-Item -ItemType Directory -Path $Execution | Out-Null
    Record 'installed.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); Registration=$Registration; ManifestSHA=$ManifestSHA;
        LauncherSHA=(Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();
        OldWatcherState=[string]$Watcher.State; OldWatcherTriggerCount=0; OldWatcherProcessCount=0; Task=$Task }
    $Action = New-ScheduledTaskAction -Execute (Join-Path $Root 'runtime/PowerShell-7.6.6/pwsh.exe') -Argument ('-NoProfile -NonInteractive -File "' + $PSCommandPath + '" -Mode Run') -WorkingDirectory $Root
    $Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    $Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $Task -Action $Action -Principal $Principal -Settings $Settings | Out-Null
    Start-ScheduledTask -TaskName $Task
    Get-ScheduledTask -TaskName $Task | Select-Object TaskName,State | ConvertTo-Json
    exit 0
}

if (Test-Path $Output) { throw 'Output exists; no retry' }
Record 'started.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); PID=$PID; Registration=$Registration; ManifestSHA=$ManifestSHA; PowerShell=$PSVersionTable.PSVersion.ToString() }
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class ADSBAuditAwake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
$ExitCode = 1
try {
    if ([ADSBAuditAwake]::SetThreadExecutionState(2147483649) -eq 0) { throw 'Temporary sleep request failed' }
    & (Join-Path $Root '.venv-prc/Scripts/python.exe') -u (Join-Path $Bundle 'adsb_stage_audit_20261007.py') run --bundle $Bundle --output $Output --registration-commit $Registration --input-manifest-sha256 $ManifestSHA *> (Join-Path $Execution 'worker.log')
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) { throw "Audit worker exit $ExitCode; no retry" }
    if (-not (Test-Path (Join-Path $Output 'result.json'))) { throw 'No atomic completed result' }
    Record 'completed.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); ExitCode=$ExitCode; ResultSHA=(Get-FileHash (Join-Path $Output 'result.json') -Algorithm SHA256).Hash.ToLowerInvariant() }
} catch {
    Record 'failed.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); ExitCode=$ExitCode; Error=$_.Exception.Message }
    $ExitCode = 1
} finally {
    [ADSBAuditAwake]::SetThreadExecutionState(2147483648) | Out-Null
}
exit $ExitCode
