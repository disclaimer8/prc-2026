#!/bin/bash
# Production-config control CV on hetzner (journal 147): cv_preds_e116prod exists only on the Mac, and LightGBM may differ between
# ARM and x86, so hetzner CVs need their own control. Also measures that cross-machine difference against the Mac control.
set -uo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=6
PY=.venv/bin/python
L=data/prodctrl_hz.log
mkdir -p data/cv_preds_e116prod_hz
echo "[$(date +%H:%M)] control CV (hetzner)" >> $L
CV_OUT=data/cv_preds_e116prod_hz FEATURES=data/features_e106.parquet $PY src/cv_fold_preds_nested.py > data/prodctrl_hz_cv.log 2>&1 || echo "cv rc $?" >> $L
$PY - >> $L 2>&1 <<'PY'
import glob, os, numpy as np
for f in sorted(glob.glob("data/cv_preds_e116prod_hz/fold_*.npz")):
    a, b = np.load(f), np.load(f.replace("_hz", ""))
    same = np.array_equal(a["mvt_id"], b["mvt_id"])
    d = a["pred"] - b["pred"]
    print(f"{os.path.basename(f)}: ids equal {same}, identical {np.array_equal(a['pred'], b['pred'])}, RMS hz-mac {np.sqrt((d ** 2).mean()):.3f} s, max |d| {np.abs(d).max():.1f}")
PY
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e116prod_hz 2>&1 | tail -2 >> $L
echo "[$(date +%H:%M)] done" >> $L
