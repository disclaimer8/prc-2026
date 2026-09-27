#!/bin/bash
# E147 (journal 120): stage E70 without the drifting reception-density features (5 seeds), compared with cand_f (E142 HEAD+E141 x5).
set -Eeuo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
rm -f data/e147.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e147.failed' ERR
echo "[$(date +%H:%M)] E147 stage CV (cand_g)"
E134_ONLY=cand_g .venv/bin/python src/e134_stage_cv.py > data/e147_stage.log 2>&1
grep -E "^fold|DECISION" data/e147_stage.log
echo "[$(date +%H:%M)] cand_g vs cand_f (the cost of dropping the features on 2025)"
PYTHONPATH=src .venv/bin/python -c "
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare('data/cv_preds_e142_hd5', 'data/cv_preds_e147_nodrift5')
print(f'cand_g vs cand_f: decided {dec:+.3f} s, y<=7200 {small:+.3f}, trimmed {trim:+.3f}, lower in {lower} of {n}')
"
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e142_hd5 data/cv_preds_e147_nodrift5 2>&1 | tail -2 || true
echo "[$(date +%H:%M)] done"
