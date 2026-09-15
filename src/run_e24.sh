#!/bin/zsh
# E24: out-of-fold predictions of the full model without `month`, then the pre-registered decision.
cd "${0:a:h}/.."
export FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_,month CV_OUT=data/cv_preds_nomonth
echo "[$(date +%H:%M)] fold preds without month"
.venv/bin/python src/cv_fold_preds.py 2>&1 | grep -E "^saved|Error|Traceback"
.venv/bin/python src/e24_month_summary.py
echo "[$(date +%H:%M)] done"
