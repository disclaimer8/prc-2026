#!/bin/zsh
# Chain the AOBT_3 abstinence measurement: wait for the control arm, run the candidate arm, then compare.
# The control arm (scripts/cv_split.sh data/cv_preds_noaobt_ctrl PROXY_SCALE=1) is already running.
set -uo pipefail
cd "${0:a:h}/.."
for i in $(seq 1 120); do
  n=$(ls data/cv_preds_noaobt_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d " ")
  [[ $n -eq 6 ]] && break
  sleep 60
done
n=$(ls data/cv_preds_noaobt_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d " ")
if [[ $n -ne 6 ]]; then echo "[$(date +%H:%M)] control arm stuck at $n/6 folds — stopping"; exit 1; fi
echo "[$(date +%H:%M)] control arm complete, starting the candidate arm"
zsh scripts/cv_split.sh data/cv_preds_noaobt PROXY_SCALE=1 NO_AOBT=1 FEATURES=data/features_noaobt_e28.parquet
echo "[$(date +%H:%M)] candidate arm complete, comparing"
.venv/bin/python src/e68_noaobt_cost.py
