param([ValidateSet('Check','Watch','Verify')][string]$Mode = 'Check')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'PowerShell 7 required' }
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Run = Join-Path $ProjectRoot 'data/cat_ally_20261006_v2'
$Record = Join-Path $ProjectRoot 'data/ally_vram_change_20261006'
$Registration = '9c0d087a0dfc96bbb57e7a343ad984418b3c08a6'

function Write-Record([string]$Name, $Value) {
    $path = Join-Path $Record $Name
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 6))
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}

function Read-Memory {
    $os = Get-CimInstance Win32_OperatingSystem
    $gpu = @(Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Video' |
        ForEach-Object { Get-ItemProperty (Join-Path $_.PSPath '0000') -ErrorAction SilentlyContinue } |
        Where-Object { $_.'HardwareInformation.qwMemorySize' } |
        ForEach-Object { [long]$_.'HardwareInformation.qwMemorySize' })
    [pscustomobject]@{ UTC=[DateTime]::UtcNow.ToString('o'); VisibleKiB=$os.TotalVisibleMemorySize;
        FreeKiB=$os.FreePhysicalMemory; Boot=$os.LastBootUpTime; GPUBytes=$gpu }
}

function Readiness {
    $pending = Get-Content (Join-Path $Record 'pending.json') -Raw | ConvertFrom-Json
    if (-not $pending.ReadbackPass -or $pending.PendingRaw -ne 65794) { throw 'Unconfirmed target setting' }
    $python = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'")
    if ($python.Count) { return 'training_or_other_python_running' }
    $reportPath = Join-Path $Run 'report.json'
    if (-not (Test-Path $reportPath)) { throw 'No full training report; no reboot' }
    $report = Get-Content $reportPath -Raw | ConvertFrom-Json
    if ($report.registration -ne $Registration -or $report.rows -ne 344419 -or
        $report.protocol -ne 'cat-ally-windows-exact-qualification-v2') { throw 'Unexpected training report' }
    foreach ($pair in @(@('prediction.npz','prediction_sha256'), @('model.cbm','model_sha256'))) {
        $actual = (Get-FileHash (Join-Path $Run $pair[0]) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $report.($pair[1])) { throw "Artifact hash differs: $($pair[0])" }
    }
    $power = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus
    $battery = Get-CimInstance Win32_Battery
    if (-not $power.PowerOnline -or $battery.EstimatedChargeRemaining -lt 20) { return 'waiting_for_power' }
    return 'ready'
}

if ($Mode -eq 'Check') {
    [pscustomobject]@{ Readiness=(Readiness); Memory=(Read-Memory) } | ConvertTo-Json -Depth 4
    exit 0
}

if ($Mode -eq 'Verify') {
    $memory = Read-Memory
    $before = Get-Content (Join-Path $Record 'before.json') -Raw | ConvertFrom-Json
    $pass = $memory.VisibleKiB -ge 14680064 -and $memory.GPUBytes -contains 1073741824 -and
        ([datetime]$memory.Boot -gt [datetime]$before.BeforeBoot)
    Write-Record 'postboot.json' @{ Pass=$pass; Memory=$memory; TargetGPUBytes=1073741824 }
    schtasks.exe /Change /TN PRC-Ally-VRAM-Verify-20261006 /DISABLE | Out-Null
    if (-not $pass) { throw 'Postboot memory check did not pass; do not retry or change settings' }
    exit 0
}

Write-Record 'watch_started.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); PID=$PID;
    WorkerSHA=(Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant(); Registration=$Registration }
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class PRCAllyAwake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
try {
    if ([PRCAllyAwake]::SetThreadExecutionState(2147483649) -eq 0) { throw 'Temporary sleep request failed' }
    $deadline = [DateTime]::UtcNow.AddHours(8)
    while ((Readiness) -ne 'ready') {
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Eight-hour wait expired; no reboot' }
        Start-Sleep -Seconds 30
    }
    $backup = Join-Path $Record 'training_backup'
    New-Item -ItemType Directory $backup | Out-Null
    foreach ($name in @('manifest.json','invocation.json','report.json','prediction.npz','model.cbm','fit.log')) {
        $source = Join-Path $Run $name
        $dest = Join-Path $backup $name
        Copy-Item $source $dest
        if ((Get-FileHash $source).Hash -ne (Get-FileHash $dest).Hash) { throw "Backup differs: $name" }
    }
    if ((Readiness) -ne 'ready') { throw 'Readiness changed during backup; no reboot' }
    $wmi = Get-CimInstance -Namespace root/wmi -ClassName AsusAtkWmi_WMNB
    $read = Invoke-CimMethod -InputObject $wmi -MethodName DSTS -Arguments @{ Device_ID=[uint32]0x000600C1 }
    if ([uint32]$read.device_status -ne [uint32]0x10102) { throw 'GPU setting changed; no reboot' }
    Write-Record 'reboot_requested.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); Memory=(Read-Memory);
        TrainingReport=(Get-Content (Join-Path $backup 'report.json') -Raw | ConvertFrom-Json);
        Backup=$backup; Forced=$false }
    shutdown.exe /r /t 0
    if ($LASTEXITCODE -ne 0) { throw "Shutdown request failed: $LASTEXITCODE" }
} catch {
    Write-Record 'watch_failed.json' @{ UTC=[DateTime]::UtcNow.ToString('o'); Error=$_.Exception.Message }
    throw
} finally {
    [PRCAllyAwake]::SetThreadExecutionState(2147483648) | Out-Null
}
