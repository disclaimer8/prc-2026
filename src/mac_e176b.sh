#!/bin/bash
# E176b (journal 225): the MLP member re-fitted on the production table and prefix set (features_e106, DROP_PREFIX with te_/ter_ and
# the stand/runway identities), i.e. without the 2025 target encodings of the E100 configuration. OOF arm on MPS -> 6th-member grid
# against the same five (box of E174/E176) and a head-to-head with the E100 member as the 6th.
cd ~/prc-2026
L=data/mac_e176b.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PROXY_SCALE=1 OMP_NUM_THREADS=4
PY=.venv/bin/python
echo "[$(date -u +%H:%MZ)] E176b MLP OOF on E106" >> $L
FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config NN_OUT=data/cv_preds_e176b_nn \
  $PY src/e100_nn_partner.py > data/e176b_oof.log 2>&1 || { echo "[$(date -u +%H:%MZ)] OOF FAILED" >> $L; exit 1; }
mkdir -p data/cv_e130mlpb
$PY -c "
import numpy as np, glob, os
for f in sorted(glob.glob('data/cv_preds_e176b_nn/fold_*.npz')):
    z = np.load(f); np.savez('data/cv_e130mlpb/' + os.path.basename(f), mvt_id=z['mvt_id'], reg2=z['reg'])"
export FEATURES=data/features_e106.parquet
echo "[$(date -u +%H:%MZ)] grid 6th member mlpb" >> $L
MEMBERS=raw,lobt,sched,eobt,aobt,mlpb E172_WMAX=0.95 E172_BOX=0.05-0.15,0.15-0.25,0.1-0.15,0.1-0.2,0.15-0.25,0-0.4 \
  $PY src/e172_members_cv.py > data/e176b_grid.log 2>&1 \
  && $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpb_ctrl data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpb 2>&1 | grep -viE "warn|extension|Polars 2.0" >> data/e176b_grid.log \
  && $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpb 2>&1 | grep SUMMARY >> data/e176b_grid.log \
  || echo "[$(date -u +%H:%MZ)] grid FAILED" >> $L
echo "[$(date -u +%H:%MZ)] done" >> $L
