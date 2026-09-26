#!/bin/bash
# E134 stage CV on hetzner; starts after the E137 chain has written its "done" line.
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
until grep -q '\] done' data/e137_chain.log 2>/dev/null || [ -f data/e137.failed ]; do sleep 120; done
rm -f data/e134.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e134.failed' ERR
echo "[$(date +%H:%M)] E134 stage CV"
.venv/bin/python src/e134_stage_cv.py > data/e134_stage.log 2>&1
tail -4 data/e134_stage.log
for c in a b c; do
  echo "[$(date +%H:%M)] post_e14 cand_$c"
  PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e134_ctrl $( [ $c = c ] && echo data/cv_preds_e140 || echo data/cv_preds_e134_$c ) 2>&1 | tail -6 || true
done
echo "[$(date +%H:%M)] done"
