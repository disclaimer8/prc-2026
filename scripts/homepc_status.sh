#!/bin/zsh
# Home PC: print the tail of an experiment log and python processes (reads through a UTF-8 file, console output is garbled).
#   scripts/homepc_status.sh e29
S=${TMPDIR:-/tmp}
cat > $S/hp_status.ps1 <<PS
\$root = "C:\Users\HiMaxym\prc-2026"
\$out = "C:\Users\HiMaxym\hp_status.txt"
"== $1 log" | Out-File \$out -Encoding utf8
Get-Content "\$root\data\\$1.log" -Tail 12 -ErrorAction SilentlyContinue | Out-File \$out -Append -Encoding utf8
"== folds" | Out-File \$out -Append -Encoding utf8
Get-Content "\$root\data\\$1_folds.log" -ErrorAction SilentlyContinue | Select-String "^saved|Error|Traceback" | Out-File \$out -Append -Encoding utf8
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, PriorityClass, @{n='CPU_s';e={[math]::Round(\$_.CPU)}}, @{n='WS_GB';e={[math]::Round(\$_.WorkingSet64/1GB,1)}} | Out-String | Out-File \$out -Append -Encoding utf8
PS
scp -q $S/hp_status.ps1 home-pc-lan:C:/Users/HiMaxym/ && ssh home-pc-lan 'pwsh -NoProfile -ExecutionPolicy Bypass -File C:/Users/HiMaxym/hp_status.ps1' >/dev/null 2>&1
scp -q home-pc-lan:C:/Users/HiMaxym/hp_status.txt $S/hp_status.txt && cat $S/hp_status.txt
