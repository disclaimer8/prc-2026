#!/bin/zsh
# E28: forward neighbour windows, out-of-fold predictions of the canonical model (no month), decision. Log: data/e28.log
cd "${0:a:h}/.."
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py || { echo "ABORT: uncommitted changes in model code"; exit 1; }
[[ -f data/features_e28.parquet ]] || .venv/bin/python src/e28_forward_neighbours.py build 2>&1 | grep -v Warning
FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month CV_OUT=data/cv_preds_e28 .venv/bin/python src/cv_fold_preds.py 2>&1 | grep -E "^saved|Error|Traceback"
.venv/bin/python src/e28_forward_neighbours.py summary
echo "[$(date +%H:%M)] done"
