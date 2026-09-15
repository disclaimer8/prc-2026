#!/bin/zsh
# E32: control (E28 model + E30 stage, OOF), features with ADS-B, OOF of the canonical model on them, decision. Log: data/e32.log
cd "${0:a:h}/.."
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/e25_adsb_stage.py src/e30_adsb_stage_v2.py || { echo "ABORT: uncommitted changes in model code"; exit 1; }
E30_BASE_DIR=data/cv_preds_e28 E30_SAVE_DIR=data/cv_preds_e28_e30 .venv/bin/python src/e30_adsb_stage_v2.py cv 2>&1 | grep -E "DECISION"
.venv/bin/python src/e32_adsb_in_model.py build 2>&1 | grep -v Warning
FEATURES=data/features_e32.parquet DROP_PREFIX=day_,tq_,wxr_,month CV_OUT=data/cv_preds_e32 .venv/bin/python src/cv_fold_preds.py 2>&1 | grep -E "^saved|Error|Traceback"
E30_BASE_DIR=data/cv_preds_e32 E30_SAVE_DIR=data/cv_preds_e32_e30 .venv/bin/python src/e30_adsb_stage_v2.py cv 2>&1 | grep -E "DECISION"
.venv/bin/python src/e32_adsb_in_model.py summary
echo "[$(date +%H:%M)] done"
