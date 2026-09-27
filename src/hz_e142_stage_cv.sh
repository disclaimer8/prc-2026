#!/bin/bash
# E142 (journal 103): cand_e + cand_f of src/e134_stage_cv.py (5-seed stage), then post_e14 against the 1-seed ctrl.
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
rm -f data/e142.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e142.failed' ERR
echo "[$(date +%H:%M)] E142 stage CV (cand_e, cand_f)"
E134_ONLY=cand_e,cand_f .venv/bin/python src/e134_stage_cv.py > data/e142_stage.log 2>&1
tail -3 data/e142_stage.log
for c in e142_ctrl5 e142_hd5; do
  echo "[$(date +%H:%M)] post_e14 $c"
  PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e134_ctrl data/cv_preds_$c 2>&1 | tail -6 || true
done
echo "[$(date +%H:%M)] done"
