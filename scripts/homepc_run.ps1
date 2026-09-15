# Home PC (Windows, Python 3.14): run one experiment detached from the ssh session at below-normal priority.
#   pwsh -File scripts/homepc_run.ps1 <name> <python args...>     log: data/<name>.log
param([string]$Name, [Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
$root = "C:\Users\HiMaxym\prc-2026"
$vpy = "$root\.venv\Scripts\python.exe"
$cmd = "cd /d $root && set POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage&& set PYTHONIOENCODING=utf-8&& `"$vpy`" $($PyArgs -join ' ') > data\$Name.log 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $cmd"
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -Priority 7 -ExecutionTimeLimit (New-TimeSpan -Hours 12) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "prc-$Name" -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "prc-$Name"
"started prc-$Name"
