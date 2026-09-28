#!/bin/bash
# U4 (journal 137): E151 in the PRODUCTION configuration (E116: PROXY_SCALE=1 and the full DROP_PREFIX). The earlier CVs
# (E127/E132/E137/E138/E151) ran with DROP_PREFIX=day_,tq_,wxr_,month,te_ and without PROXY_SCALE against cv_preds_e106_air,
# which is not the model that is uploaded. Control and candidate in parallel on the Mac, 5 threads each.
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=5
PY=.venv/bin/python
L=data/mac_e151_prodcv.log
echo "[$(date +%H:%M)] start ctrl (features_e106) + cand (features_e151), E116 config" >> $L
CV_OUT=data/cv_preds_e116prod FEATURES=data/features_e106.parquet $PY src/cv_fold_preds_nested.py > data/mac_cv_e116prod.log 2>&1 &
p1=$!
CV_OUT=data/cv_preds_e151prod FEATURES=data/features_e151.parquet $PY src/cv_fold_preds_nested.py > data/mac_cv_e151prod.log 2>&1 &
p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
echo "[$(date +%H:%M)] cv done rc ctrl $r1 cand $r2" >> $L
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e116prod data/cv_preds_e151prod 2>&1 | tail -9 >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e151prod 2>&1 | tail -8 >> $L
scp -q -r data/cv_preds_e116prod data/cv_preds_e151prod hetzner:/root/prc-2026/data/ && echo "[$(date +%H:%M)] copied to hetzner" >> $L
echo "[$(date +%H:%M)] done" >> $L
