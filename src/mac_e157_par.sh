#!/bin/bash
# Second E157 CV worker (journal 160): folds 6 and 5 next to mac_e157.sh (which runs 2 -> 3 -> 4 and skips saved folds),
# then CatBoost E51 seeds resume from 205 once mac_e157.sh has written its verdict.
set -uo pipefail
cd ~/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=7 POLARS_MAX_THREADS=7
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
for F in 6 5; do
  echo "[$(date +%H:%M)] par worker fold $F" >> data/e157.log
  FOLDS=$F CV_OUT=data/cv_preds_e157 FEATURES=data/features_e157.parquet $PY src/cv_fold_preds_nested.py >> data/e157_cv_par.log 2>&1
done
while pgrep -f "[m]ac_e157.sh" > /dev/null; do sleep 120; done
START_SEED=205 bash src/mac_catseeds_e51.sh
