param([Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{40}$')][string]$Registration)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'PowerShell 7 is required' }
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot '.venv-prc/Scripts/python.exe'
$Worker = Join-Path $PSScriptRoot 'cat_ally_windows.py'
$Bundle = Join-Path $ProjectRoot 'data/cat_laptop_inputs_20261005'
$Output = Join-Path $ProjectRoot 'data/cat_ally_20261006'
$Log = Join-Path $ProjectRoot 'data/cat_ally_launcher_20261006.log'
if (Test-Path $Log) { throw 'Launcher log already exists; automatic retry is forbidden' }
& $Python -u $Worker run --bundle $Bundle --out $Output --registration-commit $Registration *> $Log
exit $LASTEXITCODE
