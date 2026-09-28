#!/bin/bash
# E150 (journal 127): stage E70 (HEAD + E141, 1 seed) trained also on rows with |y - base| > 3600, target still clipped to +-1800;
# compared with cand_d (the same stage with the filter), then post_e14.
set -Eeuo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
rm -f data/e150.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e150.failed' ERR
echo "[$(date +%H:%M)] E150 stage CV (cand_i)"
E134_ONLY=cand_i .venv/bin/python src/e134_stage_cv.py > data/e150_stage.log 2>&1
grep -E "^fold|DECISION" data/e150_stage.log
PYTHONPATH=src .venv/bin/python -c "
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare('data/cv_preds_e141', 'data/cv_preds_e150_keep')
print(f'cand_i vs cand_d: decided {dec:+.3f} s, all {tot:+.3f}, y<=7200 {small:+.3f}, trimmed {trim:+.3f}, lower in {lower} of {n}')
"
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e141 data/cv_preds_e150_keep 2>&1 | tail -2 || true
echo "[$(date +%H:%M)] done"
