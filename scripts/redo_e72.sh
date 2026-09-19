#!/bin/zsh
# E72 re-run, 19.09. The night queue produced a verdict on three folds: features_x_e28.parquet existed only
# on the Mac, so home-pc's folds 2,4,6 failed instantly on a missing file and cv_split.sh's collector simply
# timed out. Copy the table over, let both arms fill the missing folds, and refuse to judge on fewer than six.
set -uo pipefail
cd "${0:a:h}/.."
say() { echo "[$(date +%H:%M)] $*" | tee -a data/redo_e72.log }

say "copying features_x_e28.parquet to home-pc"
scp -q data/features_x_e28.parquet home-pc:prc-2026/data/features_x_e28.parquet || { say "copy FAILED"; exit 1; }
say "copied"

for arm in ctrl cand; do
  if [[ $arm == ctrl ]]; then out=data/cv_preds_e72_ctrl; drop=day_,tq_,wxr_,month,x_
  else out=data/cv_preds_e72; drop=day_,tq_,wxr_,month; fi
  n=$(ls $out/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
  if [[ $n -eq 6 ]]; then say "$arm already complete"; continue; fi
  say "$arm: $n/6 folds, filling the rest"
  zsh scripts/cv_split.sh $out PROXY_SCALE=1 FEATURES=data/features_x_e28.parquet DROP_PREFIX=$drop >> data/redo_e72.log 2>&1
done

.venv/bin/python - <<'PY' 2>&1 | grep -v Warning | tee -a data/redo_e72.log
import glob, sys
sys.path.insert(0, "src")
n_c = len(glob.glob("data/cv_preds_e72_ctrl/fold_*.npz"))
n_k = len(glob.glob("data/cv_preds_e72/fold_*.npz"))
if n_c != 6 or n_k != 6:
    sys.exit(f"REFUSING TO JUDGE: control has {n_c} folds, candidate {n_k}. A verdict on fewer than six is what produced last night's REJECT.")
from fold_metrics import compare
d, t, s, tr, l, n, st = compare("data/cv_preds_e72_ctrl", "data/cv_preds_e72", lb_rmse=248.3325)
ok = n == 6 and -d >= 0.3 and l >= 4 and -s >= 0.3 and -tr >= 0.3
print(f"DECISION E72 (re-run, six folds): {'ACCEPT' if ok else 'REJECT'} (lower in {l}/6; decided {d:+.2f} s, "
      f"strict {st:+.2f}, all {t:+.2f}, y<=7200 {s:+.2f}, trimmed {tr:+.2f})")
PY
say "done"
