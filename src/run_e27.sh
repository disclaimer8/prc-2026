#!/bin/zsh
# E27: build features with record ordering, out-of-fold predictions of the canonical model (no month), decision.
# Rerun 15.09 17:55: the first run used a train.py with a foreign 7200 s clip of the final prediction (invalid).
cd "${0:a:h}/.."
[[ -f data/features_e27.parquet ]] || .venv/bin/python src/e27_record_order.py build 2>&1 | grep -v Warning
FEATURES=data/features_e27.parquet DROP_PREFIX=day_,tq_,wxr_,month CV_OUT=data/cv_preds_e27 .venv/bin/python src/cv_fold_preds.py 2>&1 | grep -E "^saved|Error|Traceback"
.venv/bin/python src/e27_record_order.py summary
echo "[$(date +%H:%M)] done"
