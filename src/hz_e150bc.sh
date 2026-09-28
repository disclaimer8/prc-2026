#!/bin/bash
# E150b (stage E70 keep-all, contradicting rows at weight 0.25) and E150c (E64 keep-all), journal 128.
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
L=data/e150bc_chain.log
echo "[$(date +%H:%M)] E150b" >> $L
E134_ONLY=cand_j .venv/bin/python src/e134_stage_cv.py > data/e150b_stage.log 2>&1 || echo "[$(date +%H:%M)] E150b FAILED" >> $L
grep -E "^fold|DECISION" data/e150b_stage.log >> $L
PYTHONPATH=src .venv/bin/python -c "
from fold_metrics import compare
for d, n in (('data/cv_preds_e150b_w025', 'E150b'), ('data/cv_preds_e150_keep', 'E150')):
    dec, tot, small, trim, lower, k, strict = compare('data/cv_preds_e141', d)
    print(f'{n} vs cand_d: decided {dec:+.3f} s, all {tot:+.3f}, y<=7200 {small:+.3f}, trimmed {trim:+.3f}, lower in {lower} of {k}')
" >> $L 2>&1
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e141 data/cv_preds_e150b_w025 2>&1 | tail -1 >> $L || true
echo "[$(date +%H:%M)] E150c" >> $L
PYTHONPATH=src .venv/bin/python src/e150c_e64_keep_cv.py 2>&1 | grep -E "^fold|DECISION" >> $L || echo "[$(date +%H:%M)] E150c FAILED" >> $L
echo "[$(date +%H:%M)] done" >> $L
