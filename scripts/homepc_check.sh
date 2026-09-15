#!/bin/zsh
# Home PC: free memory, python processes and the tail of a fold log (UTF-8 file round trip).  scripts/homepc_check.sh e29
S=${TMPDIR:-/tmp}
cat > $S/chk.ps1 <<PS
\$o = "C:\Users\HiMaxym\logs.txt"
\$os = Get-CimInstance Win32_OperatingSystem
"free RAM GB \$([math]::Round(\$os.FreePhysicalMemory/1MB,1)); free commit GB \$([math]::Round(\$os.FreeVirtualMemory/1MB,1))" | Out-File \$o -Encoding utf8
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, PriorityClass, @{n='CPU_s';e={[math]::Round(\$_.CPU)}}, @{n='Priv_GB';e={[math]::Round(\$_.PrivateMemorySize64/1GB,1)}} | Out-String | Out-File \$o -Append -Encoding utf8
Get-Content C:\Users\HiMaxym\prc-2026\data\\$1_folds.log -ErrorAction SilentlyContinue | Select-String "^saved|Error|Traceback|failed" | Out-File \$o -Append -Encoding utf8
Get-Content C:\Users\HiMaxym\prc-2026\data\\$1.log -Tail 10 -ErrorAction SilentlyContinue | Out-File \$o -Append -Encoding utf8
PS
scp -q $S/chk.ps1 home-pc-lan:C:/Users/HiMaxym/ && ssh home-pc-lan 'pwsh -NoProfile -ExecutionPolicy Bypass -File C:/Users/HiMaxym/chk.ps1' >/dev/null 2>&1
scp -q home-pc-lan:C:/Users/HiMaxym/logs.txt $S/l.txt && grep -v "^\s*$" $S/l.txt | cut -c1-200
