$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$source = Join-Path $ProjectRoot 'scripts/ally_vram_finish.ps1'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$function = @($ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Readiness'}, $true))
if ($function.Count -ne 1) { throw 'Readiness function not found exactly once' }
Invoke-Expression $function[0].Extent.Text
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('prc-vram-test-' + [guid]::NewGuid())
$Run = Join-Path $fixture 'run'; $Record = Join-Path $fixture 'record'
New-Item -ItemType Directory $Run,$Record | Out-Null
$Registration = '9c0d087a0dfc96bbb57e7a343ad984418b3c08a6'
$script:HasPython = $true; $script:PowerOnline = $true
function Get-CimInstance {
    param($ClassName, $Filter, $Namespace)
    if ($ClassName -eq 'Win32_Process') { if ($script:HasPython) { [pscustomobject]@{ProcessId=123} }; return }
    if ($ClassName -eq 'BatteryStatus') { return [pscustomobject]@{PowerOnline=$script:PowerOnline} }
    if ($ClassName -eq 'Win32_Battery') { return [pscustomobject]@{EstimatedChargeRemaining=50} }
    throw 'Unexpected CIM request in guard test'
}
function Must-Fail {
    $failed=$false
    try { $null=Readiness } catch { $failed=$true }
    if (-not $failed) { throw 'Unsafe readiness accepted' }
}
'{"ReadbackPass":true,"PendingRaw":65794}' | Set-Content (Join-Path $Record 'pending.json')
if ((Readiness) -ne 'training_or_other_python_running') { throw 'Active process guard failed' }
$script:HasPython=$false
Must-Fail # No report must never allow a reboot.
'fixture-prediction' | Set-Content (Join-Path $Run 'prediction.npz')
'fixture-model' | Set-Content (Join-Path $Run 'model.cbm')
$report=@{registration=$Registration;rows=344419;protocol='cat-ally-windows-exact-qualification-v2';
    prediction_sha256=(Get-FileHash (Join-Path $Run 'prediction.npz')).Hash.ToLowerInvariant();
    model_sha256=(Get-FileHash (Join-Path $Run 'model.cbm')).Hash.ToLowerInvariant()}
$report | ConvertTo-Json | Set-Content (Join-Path $Run 'report.json')
if ((Readiness) -ne 'ready') { throw 'Valid fixture rejected' }
$script:PowerOnline=$false
if ((Readiness) -ne 'waiting_for_power') { throw 'Power guard failed' }
$script:PowerOnline=$true
'corruption' | Add-Content (Join-Path $Run 'prediction.npz')
Must-Fail # Corrupt artifact must never allow a reboot.
$report.registration='0000000000000000000000000000000000000000'
$report | ConvertTo-Json | Set-Content (Join-Path $Run 'report.json')
Must-Fail
'{"ReadbackPass":false,"PendingRaw":65797}' | Set-Content (Join-Path $Record 'pending.json')
Must-Fail
Write-Output '7 REBOOT GUARD CHECKS PASSED; no shutdown or firmware methods called'
