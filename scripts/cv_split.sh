#!/bin/zsh
# Run a six-fold CV across both machines: folds 1,3,5 here and 2,4,6 on home-pc, then collect.
# Halves the wall time of every experiment — the folds are independent and share no state.
#   scripts/cv_split.sh <CV_OUT dir> [env assignments...]
#   scripts/cv_split.sh data/cv_preds_e56 PROXY_ISO=1 REGIME_TOL=10
set -euo pipefail
cd "${0:a:h}/.."
out=$1; shift
env_pairs="$@"
mkdir -p $out
echo "[$(date +%H:%M)] local folds 1,3,5 -> $out"
git bundle create /tmp/prc.bundle master >/dev/null 2>&1
scp -q /tmp/prc.bundle home-pc:prc.bundle
cat > /tmp/cv_pc.ps1 <<PS
[Console]::OutputEncoding=[Text.Encoding]::UTF8
\$root = "C:\Users\HiMaxym\prc-2026"
Set-Location \$root
git pull -q C:\Users\HiMaxym\prc.bundle master 2>&1 | Out-Null
\$ps = @"
Set-Location \$root
\`\$env:POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR = 'load_as_storage'; \`\$env:PYTHONIOENCODING = 'utf-8'
\`\$env:FEATURES = 'data/features_e28.parquet'; \`\$env:DROP_PREFIX = 'day_,tq_,wxr_,month'
\`\$env:FOLDS = '2,4,6'; \`\$env:CV_OUT = '$out'
$(for kv in ${(z)env_pairs}; do echo "\`\$env:${kv%%=*} = '${kv#*=}'"; done)
New-Item -ItemType Directory -Force -Path $out | Out-Null
& '\$root\.venv\Scripts\python.exe' src/cv_fold_preds_nested.py *> data\cv_split_pc.log
"@
Set-Content -Path "\$root\data\run_cv_split.ps1" -Value \$ps -Encoding utf8
\$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "\`"C:\Program Files\PowerShell\7\pwsh.exe\`" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File \$root\data\run_cv_split.ps1"; CurrentDirectory = \$root }
"home-pc folds 2,4,6 pid \$(\$r.ProcessId)"
PS
scp -q /tmp/cv_pc.ps1 home-pc:cv_pc.ps1
ssh home-pc 'pwsh -NoProfile -ExecutionPolicy Bypass -File cv_pc.ps1' 2>&1 | strings | grep -v '^\s*$' || true
env FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month FOLDS=1,3,5 CV_OUT=$out ${(z)env_pairs} .venv/bin/python src/cv_fold_preds_nested.py
echo "[$(date +%H:%M)] local folds done, waiting for home-pc"
for i in $(seq 1 60); do
  scp -q "home-pc:prc-2026/$out/*.npz" $out/ 2>/dev/null || true
  n=$(ls $out/fold_*.npz 2>/dev/null | wc -l | tr -d " ")
  [[ $n -eq 6 ]] && { echo "[$(date +%H:%M)] all six folds collected in $out"; exit 0; }
  sleep 60
done
echo "[$(date +%H:%M)] only $(ls $out/fold_*.npz 2>/dev/null | wc -l | tr -d ' ') folds after an hour — check home-pc"
exit 1
