#!/bin/bash
# E158 (journal 169): production nested CV with the regime classifier also seeing the regressors' OOF (stk_*, CLF_ONLY_PREFIX);
# two workers (folds 1-3 and 4-6), then the gates against cv_preds_e116prod and the regressor identity check.
set -uo pipefail
cd ~/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CLF_ONLY_PREFIX=stk_
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/e158.log
mkdir -p data/cv_preds_e158
echo "[$(date +%H:%M)] E158 CV start" >> $L
FOLDS=1,2,3 CV_OUT=data/cv_preds_e158 FEATURES=data/features_e158.parquet $PY src/cv_fold_preds_nested.py > data/e158_cv_a.log 2>&1 &
FOLDS=4,5,6 CV_OUT=data/cv_preds_e158 FEATURES=data/features_e158.parquet $PY src/cv_fold_preds_nested.py > data/e158_cv_b.log 2>&1 &
wait
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e158 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e158 2>&1 | tail -8 >> $L
$PY - >> $L 2>&1 <<'PY'
import glob, os, numpy as np
for p in sorted(glob.glob("data/cv_preds_e158/fold_*.npz")):
    a, b = np.load(p), np.load(p.replace("cv_preds_e158", "cv_preds_e116prod"))
    assert (a["mvt_id"] == b["mvt_id"]).all()
    s = np.nanmax(b["values"][:, :5], 1) - np.nanmin(b["values"][:, :5], 1)
    d = (a["pred"] - a["y"]) ** 2 - (b["pred"] - b["y"]) ** 2
    print(f"{os.path.basename(p)}: reg RMS diff vs control {np.sqrt(np.mean((a['reg'] - b['reg']) ** 2)):.3f} s; "
          f"dSSE spread>3600 {d[s > 3600].sum() / 1e9:+.3f}e9, spread<=3600 {d[s <= 3600].sum() / 1e9:+.3f}e9, y<=1800 {d[a['y'] <= 1800].sum() / 1e9:+.3f}e9")
PY
echo "[$(date +%H:%M)] done" >> $L
