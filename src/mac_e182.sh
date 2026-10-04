#!/bin/bash
# E182 (journal 248): the five LightGBM members (raw, LOBT, SCHED, EOBT, AOBT) refit on the final table features_fin1b, as E181a did
# for the MLP (decided -0.487, 6/6). Members of the board are on features_e106 while the board base is FIN1b.
# All grids here use the FIN1b production OOF as control (CTRL=data/cv_preds_fin1b_hz, suffix _f1b), the base the board uses,
# so a gain measures what a member adds over FIN1b and not FIN1b information entering through the member.
#   step 0: six with the E176 MLP vs six with the E181a MLP under the FIN1b control (re-check of E181a, minutes)
#   step 1: member OOFs lobt, aobt, eobt on fin1b (Mac, ~2 h each); sched and raw run on hetzner (hz_queue/96_e182_members.sh)
#   step 2: interim grid lobtf+aobtf; final grid with all five f-members once hetzner's arrive; gate vs old members + mlpa
cd ~/prc-2026
L=data/mac_e182.log
PY=.venv/bin/python
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
BOX=0-0.1,0.05-0.2,0.05-0.15,0.05-0.15,0.1-0.2,0.15-0.35
grid () {  # grid <members> <log>
  FEATURES=data/features_e106.parquet OMP_NUM_THREADS=2 CTRL=data/cv_preds_fin1b_hz E172_SUFFIX=_f1b MEMBERS=$1 E172_WMAX=1.0 E172_BOX=$BOX \
    $PY src/e172_members_cv.py > $2 2>&1 || echo "[$(date -u +%H:%MZ)] grid $1 FAILED" >> $L
}
gate () {  # gate <ctrl members> <cand members> <log>
  FEATURES=data/features_e106.parquet $PY src/gate.py data/cv_preds_e172_${1//,/_}_f1b data/cv_preds_e172_${2//,/_}_f1b 2>&1 | grep -viE "warn|extension|Polars 2.0" >> $3
}
echo "[$(date -u +%H:%MZ)] step 0" >> $L
grid raw,lobt,sched,eobt,aobt,mlp data/e182_grid_old_mlp.log
grid raw,lobt,sched,eobt,aobt,mlpa data/e182_grid_old_mlpa.log
gate raw,lobt,sched,eobt,aobt,mlp raw,lobt,sched,eobt,aobt,mlpa data/e182_grid_old_mlpa.log
grep -h "SUMMARY" data/e182_grid_old_mlpa.log | sed 's/^/  E181a under the FIN1b control: /' >> $L
for M in lobt aobt eobt; do
  echo "[$(date -u +%H:%MZ)] E182 member ${M}f" >> $L
  OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 FEATURES=data/features_fin1b.parquet E130_MODE=$M CV_OUT=data/cv_e130${M}f \
    $PY src/e130_raw_reg_cv.py > data/e182_${M}f.log 2>&1 || echo "[$(date -u +%H:%MZ)] ${M}f FAILED" >> $L
done
echo "[$(date -u +%H:%MZ)] interim grid (lobtf, aobtf, eobtf)" >> $L
grid raw,lobtf,sched,eobtf,aobtf,mlpa data/e182_grid_3f.log
gate raw,lobt,sched,eobt,aobt,mlpa raw,lobtf,sched,eobtf,aobtf,mlpa data/e182_grid_3f.log
grep -h "SUMMARY" data/e182_grid_3f.log | sed 's/^/  3f: /' >> $L
echo "[$(date -u +%H:%MZ)] waiting for hetzner schedf, rawf" >> $L
until ssh hetzner "test -s /root/prc-2026/data/e182_hz.ready -o -e /root/prc-2026/data/e182_hz.failed" < /dev/null 2>/dev/null; do sleep 120; done
for M in sched raw; do
  rsync -aq hetzner:/root/prc-2026/data/cv_e130${M}f/ data/cv_e130${M}f/ < /dev/null || echo "[$(date -u +%H:%MZ)] fetch ${M}f FAILED" >> $L
done
echo "[$(date -u +%H:%MZ)] final grid (all five f)" >> $L
grid rawf,lobtf,schedf,eobtf,aobtf,mlpa data/e182_grid_5f.log
gate raw,lobt,sched,eobt,aobt,mlpa rawf,lobtf,schedf,eobtf,aobtf,mlpa data/e182_grid_5f.log
grep -h "SUMMARY" data/e182_grid_5f.log | sed 's/^/  5f: /' >> $L
echo "[$(date -u +%H:%MZ)] done" >> $L
