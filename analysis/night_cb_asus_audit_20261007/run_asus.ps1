param([Parameter(Mandatory=$true)][string]$SourceCommit,
      [Parameter(Mandatory=$true)][string]$ManifestSHA)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'PowerShell7 required' }
$Root = 'C:/Users/admin/prc-2026'
$Audit = Join-Path $Root 'analysis/night_cb_asus_audit_20261007'
$Bundle = Join-Path $Audit 'inputs'
$Output = Join-Path $Audit 'results'
if (Test-Path $Output) { throw 'Output exists; no retry' }
if ((Get-FileHash (Join-Path $Bundle 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ManifestSHA) { throw 'Manifest SHA differs' }
& (Join-Path $Root '.venv-prc/Scripts/python.exe') -u (Join-Path $Bundle 'night_cb_asus_audit_20261007.py') run --bundle $Bundle --output $Output --source-commit $SourceCommit --manifest-sha256 $ManifestSHA *> (Join-Path $Audit 'worker.log')
$Code = $LASTEXITCODE
if ($Code -ne 0) { throw "Audit worker failed exit=$Code; no retry" }
if (-not (Test-Path (Join-Path $Output 'result.json'))) { throw 'Missing successful result' }
@{ SourceCommit=$SourceCommit; ManifestSHA=$ManifestSHA; PowerShell=$PSVersionTable.PSVersion.ToString(); UTC=[DateTime]::UtcNow.ToString('o'); ExitCode=$Code; ResultSHA=(Get-FileHash (Join-Path $Output 'result.json') -Algorithm SHA256).Hash.ToLowerInvariant() } | ConvertTo-Json | Set-Content -Path (Join-Path $Audit 'terminal.json') -Encoding utf8
Get-Content (Join-Path $Audit 'terminal.json')
