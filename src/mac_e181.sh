#!/bin/bash
# E181 (journal 244): a better MLP member, tested as a REPLACEMENT of the E176 MLP in the six-member set.
#   a: E100 configuration on the final table features_fin1b (destination ATFM, route, wake ladder, plan residual added)
#   b: E100 configuration (features_e77) with a wider network (NN_WIDTH=1024) and 12 epochs
# Each: OOF arm on MPS -> member dir -> six-member grid (box of the E176 grid with the new MLP in place of mlp) -> gate against
# the six-member set with the E176 MLP (data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp).
cd ~/prc-2026
L=data/mac_e181.log
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PROXY_SCALE=1 OMP_NUM_THREADS=6
PY=.venv/bin/python
run_arm () {  # run_arm <tag> <env...>
  local tag=$1; shift
  echo "[$(date -u +%H:%MZ)] E181$tag OOF" >> $L
  env "$@" NN_OUT=data/cv_preds_e181${tag}_nn $PY src/e100_nn_partner.py > data/e181${tag}_oof.log 2>&1 || { echo "[$(date -u +%H:%MZ)] E181$tag OOF FAILED" >> $L; return 1; }
  mkdir -p data/cv_e130mlp$tag
  $PY -c "
import numpy as np, glob, os
for f in sorted(glob.glob('data/cv_preds_e181${tag}_nn/fold_*.npz')):
    z = np.load(f); np.savez('data/cv_e130mlp$tag/' + os.path.basename(f), mvt_id=z['mvt_id'], reg2=z['reg'])"
  echo "[$(date -u +%H:%MZ)] E181$tag grid" >> $L
  FEATURES=data/features_e106.parquet OMP_NUM_THREADS=2 MEMBERS=raw,lobt,sched,eobt,aobt,mlp$tag E172_WMAX=1.0 \
    E172_BOX=0.05-0.1,0.15-0.2,0.1-0.15,0.1-0.15,0.15-0.2,0.15-0.35 $PY src/e172_members_cv.py > data/e181${tag}_grid.log 2>&1 \
    && FEATURES=data/features_e106.parquet $PY src/gate.py data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlp$tag 2>&1 \
       | grep -viE "warn|extension|Polars 2.0" >> data/e181${tag}_grid.log || echo "[$(date -u +%H:%MZ)] E181$tag grid FAILED" >> $L
}
run_arm a FEATURES=data/features_fin1b.parquet
run_arm b FEATURES=data/features_e77.parquet NN_WIDTH=1024 NN_EPOCHS=12
echo "[$(date -u +%H:%MZ)] done" >> $L
