#!/bin/bash
# Night 27->28.09 on hetzner (journal 120-122). Each step is independent: a failure is logged and the next step runs.
set -uo pipefail
cd /root/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
PY=.venv/bin/python
L=data/night_27sep.log
step () { local name=$1; shift; echo "[$(date +%H:%M)] $name" >> $L
  if "$@" > data/night_$name.log 2>&1; then echo "[$(date +%H:%M)] $name ok" >> $L; else echo "[$(date +%H:%M)] $name FAILED" >> $L; fi; }
until grep -qE '\] done|FAILED' data/e147_chain.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] start (E147 finished)" >> $L
# 1. E147b: E64 x5 without vi_n_pts, vi_prev_pts, vi_alt_first vs E64 x5
step e147b env OMP_NUM_THREADS=6 PYTHONPATH=src $PY src/e147b_e64_nodrift_cv.py
# 2. E148: stage from two families, 0.5 LightGBM x5 + 0.5 CatBoost, vs LightGBM x5 (cand_f)
step e148 env OMP_NUM_THREADS=6 E134_ONLY=cand_h $PY src/e134_stage_cv.py
step e148_post env PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e142_hd5 data/cv_preds_e148_twofam
# 3. E147 build on the 10-seed line: v443 -> v454 stage x5 no-drift -> v455 E64 x5 no-drift -> v456 E81 -> v457 LOBT -> v458 E145
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 FEATURES=data/features_e106.parquet PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
step b454 env STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v441.npz $PY src/e141_apply.py 443 454
step b455 env STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=vi_n_pts,vi_prev_pts,vi_alt_first ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v441.npz $PY src/e64_apply.py 454 455
step b456 env BLEND_BASE=455 BLEND_PARTS=44:0.1214,60:0.0270 $PY src/blend_submissions.py 456
step b457 env LOBT_CLIP=1 $PY src/lobt_edge_lift.py 456 457
step b458 $PY src/e145_pru_calib.py apply 457 458
echo "[$(date +%H:%M)] done" >> $L
