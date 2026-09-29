#!/bin/bash
# E130 / Hypothesis 5 (journal 151): raw-target regressor blended into the regular candidate; LOBT-anchored variant after it.
# Waits for the hetzner production control CV (hz_prodctrl.sh) to free cores 6-11.
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/e130.log
until grep -q "done" data/prodctrl_hz.log 2>/dev/null && ! pgrep -f "[h]z_prodctrl" > /dev/null; do sleep 60; done
for MODE in raw lobt; do
  echo "[$(date +%H:%M)] E130 $MODE regressor CV" >> $L
  E130_MODE=$MODE CV_OUT=data/cv_e130$MODE $PY src/e130_raw_reg_cv.py 2>&1 | grep -v -i warn >> $L || { echo "[$(date +%H:%M)] E130 $MODE FAILED" >> $L; continue; }
  for C in data/cv_preds_e116prod data/cv_preds_e116prod_hz; do
    [[ -e $C/fold_6_12.npz ]] || continue
    O=data/cv_preds_e130${MODE}_$(basename $C | sed 's/cv_preds_//')
    echo "[$(date +%H:%M)] blend $MODE vs $C -> $O" >> $L
    CTRL=$C E130=data/cv_e130$MODE CV_OUT=$O $PY src/e130_blend_cv.py 2>&1 | grep -v -i warn >> $L
    PYTHONPATH=src $PY src/fold_metrics.py $C $O 2>&1 | tail -9 >> $L
    PYTHONPATH=src $PY src/post_e14.py $C $O 2>&1 | tail -8 >> $L
  done
done
echo "[$(date +%H:%M)] done" >> $L
