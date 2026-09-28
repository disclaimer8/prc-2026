#!/bin/bash
# E150d (journal 129): E150 with the target clipped to +-3600; compared with E150 and cand_d.
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
L=data/e150d_chain.log
echo "[$(date +%H:%M)] E150d" >> $L
E134_ONLY=cand_k .venv/bin/python src/e134_stage_cv.py > data/e150d_stage.log 2>&1 || echo "[$(date +%H:%M)] E150d FAILED" >> $L
grep -E "^fold|DECISION" data/e150d_stage.log >> $L
PYTHONPATH=src .venv/bin/python -c "
from fold_metrics import compare
for a, b, n in (('data/cv_preds_e141', 'data/cv_preds_e150d_clip3600', 'E150d vs cand_d'), ('data/cv_preds_e150_keep', 'data/cv_preds_e150d_clip3600', 'E150d vs E150')):
    dec, tot, small, trim, lower, k, strict = compare(a, b)
    print(f'{n}: decided {dec:+.3f} s, all {tot:+.3f}, y<=7200 {small:+.3f}, trimmed {trim:+.3f}, lower in {lower} of {k}')
" >> $L 2>&1
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e150_keep data/cv_preds_e150d_clip3600 2>&1 | tail -1 >> $L || true
echo "[$(date +%H:%M)] done" >> $L
