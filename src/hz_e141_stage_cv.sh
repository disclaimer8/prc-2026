#!/bin/bash
# E141 (journal 98): ctrl + cand_d of src/e134_stage_cv.py, then post_e14. Old ctrl kept for a reproducibility check.
set -euo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6
rm -f data/e141.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e141.failed' ERR
rm -rf data/cv_preds_e134_ctrl_prev && cp -r data/cv_preds_e134_ctrl data/cv_preds_e134_ctrl_prev
echo "[$(date +%H:%M)] E141 stage CV (ctrl + cand_d)"
E134_ONLY=ctrl,cand_d .venv/bin/python src/e134_stage_cv.py > data/e141_stage.log 2>&1
tail -3 data/e141_stage.log
.venv/bin/python - <<'PY'
import glob, numpy as np
d = [float(np.abs(np.load(a)["pred"] - np.load(a.replace("_ctrl_prev", "_ctrl"))["pred"]).max())
     for a in sorted(glob.glob("data/cv_preds_e134_ctrl_prev/fold_*.npz"))]
print("ctrl reproducibility, max |diff| per fold:", [round(x, 4) for x in d])
PY
echo "[$(date +%H:%M)] post_e14 cand_d"
PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_e134_ctrl data/cv_preds_e141 2>&1 | tail -6 || true
echo "[$(date +%H:%M)] done"
