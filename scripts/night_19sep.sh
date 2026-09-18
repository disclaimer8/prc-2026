#!/bin/zsh
# Overnight queue for 18->19.09. Each stage logs separately and every stage is resumable: re-running the
# script skips work whose artefacts already exist. Nothing here touches data/adsblol or the accepted
# feature table, so the v80 submission waiting for 00:05 UTC is unaffected.
set -uo pipefail
cd "${0:a:h}/.."
say() { echo "[$(date +%H:%M)] $*" | tee -a data/night_19sep.log }

say "=== 1/6 re-extract ground features at 1 kt (E71) ==="
if [[ $(ls data/adsblol_kt1/offblock5_*.parquet 2>/dev/null | wc -l | tr -d ' ') -lt 420 ]]; then
  zsh scripts/reextract_kt1.sh >> data/night_19sep.log 2>&1
else
  say "kt1 extraction already complete, skipping"
fi

say "=== 2/6 E71: movement threshold 3 kt vs 1 kt, at the E65 stage ==="
if [[ ! -f data/cv_preds_e71/fold_6_12.npz ]]; then
  .venv/bin/python src/e71_onset_threshold.py > data/e71_cv.log 2>&1
  tail -3 data/e71_cv.log | tee -a data/night_19sep.log
else
  say "E71 already done"
fi

say "=== 3/6 rebuild the feature table with the two new columns ==="
if [[ ! -f data/features_x_e28.parquet ]]; then
  FEATURES_OUT=data/features_x.parquet .venv/bin/python src/features.py > data/features_x.log 2>&1
  E28_IN=data/features_x.parquet E28_OUT=data/features_x_e28.parquet \
    .venv/bin/python src/e28_forward_neighbours.py build >> data/features_x.log 2>&1
  say "features_x_e28.parquet built"
else
  say "feature table already built"
fi

say "=== 4/6 E72 control: same table, the two new columns dropped ==="
if [[ ! -f data/cv_preds_e72_ctrl/fold_6_12.npz ]]; then
  zsh scripts/cv_split.sh data/cv_preds_e72_ctrl PROXY_SCALE=1 \
    FEATURES=data/features_x_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month,x_ >> data/night_19sep.log 2>&1
else
  say "E72 control already done"
fi

say "=== 5/6 E72 candidate: x_plan_res, x_hprox_med, x_hprox_n added ==="
if [[ ! -f data/cv_preds_e72/fold_6_12.npz ]]; then
  zsh scripts/cv_split.sh data/cv_preds_e72 PROXY_SCALE=1 \
    FEATURES=data/features_x_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month >> data/night_19sep.log 2>&1
  .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from fold_metrics import compare
d,t,s,tr,l,n,st = compare('data/cv_preds_e72_ctrl','data/cv_preds_e72', lb_rmse=248.7737)
ok = n==6 and -d>=0.3 and l>=4 and -s>=0.3 and -tr>=0.3
print(f'DECISION E72: {\"ACCEPT\" if ok else \"REJECT\"} (folds {n}, lower in {l}; decided {d:+.2f} s, strict {st:+.2f}, all {t:+.2f}, y<=7200 {s:+.2f}, trimmed {tr:+.2f})')
" 2>&1 | grep -v Warning | tee -a data/night_19sep.log
else
  say "E72 candidate already done"
fi

say "=== 6/6 E73 (MB3): regressor trained only on the rows it serves ==="
if [[ ! -f data/cv_preds_e73_ctrl/fold_6_12.npz ]]; then
  zsh scripts/cv_split.sh data/cv_preds_e73_ctrl PROXY_SCALE=1 >> data/night_19sep.log 2>&1
fi
if [[ ! -f data/cv_preds_e73/fold_6_12.npz ]]; then
  zsh scripts/cv_split.sh data/cv_preds_e73 PROXY_SCALE=1 SERVED_ONLY=1 >> data/night_19sep.log 2>&1
  .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from fold_metrics import compare
d,t,s,tr,l,n,st = compare('data/cv_preds_e73_ctrl','data/cv_preds_e73', lb_rmse=248.7737)
ok = n==6 and -d>=0.3 and l>=4 and -s>=0.3 and -tr>=0.3
print(f'DECISION E73: {\"ACCEPT\" if ok else \"REJECT\"} (folds {n}, lower in {l}; decided {d:+.2f} s, strict {st:+.2f}, all {t:+.2f}, y<=7200 {s:+.2f}, trimmed {tr:+.2f})')
" 2>&1 | grep -v Warning | tee -a data/night_19sep.log
fi

say "=== night queue finished ==="
