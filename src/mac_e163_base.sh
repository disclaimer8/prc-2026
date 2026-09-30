#!/bin/bash
# E163 board base on the Mac (journal 202): E116 recipe on features_e163 -> v770 + 5 LightGBM seeds, shipped to hetzner, where
# src/hz_build_e163.sh blends it with the 25 CatBoost seeds and runs the T2 tail. Then the CatBoost E51+E157 seeds resume.
set -uo pipefail
cd ~/prc-2026
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_e163.parquet
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
L=data/mac_e163_base.log
echo "[$(date +%H:%M)] validate" >> $L
$PY src/train.py validate > data/e163_validate.log 2>&1 && cp data/last_validation.json data/validation_v770.json || { echo "[$(date +%H:%M)] validate FAILED" >> $L; exit 1; }
cp data/validation_v770.json data/last_validation.json
echo "[$(date +%H:%M)] submit 770" >> $L
$PY src/train.py submit 770 > data/e163_submit.log 2>&1 || { echo "[$(date +%H:%M)] submit FAILED" >> $L; exit 1; }
REG=$($PY -c "import json;print(int(json.load(open('data/validation_v770.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds (reg rounds $REG)" >> $L
$PY src/seed_avg.py rank 5 $REG > data/e163_lgbseeds.log 2>&1 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v770.npz || { echo "[$(date +%H:%M)] seeds FAILED" >> $L; exit 1; }
for f in submissions/clf_v770*.txt submissions/reg_v770.txt submissions/lgb_reg_seeds_v770.npz submissions/gentle-igloo_v770.parquet data/validation_v770.json; do
  rsync -a --partial --timeout=120 $f hetzner:/root/prc-2026/$f || { echo "[$(date +%H:%M)] ship $f FAILED" >> $L; exit 1; }
done
ssh hetzner "date > /root/prc-2026/data/e163_base.ready"
echo "[$(date +%H:%M)] shipped" >> $L
START_SEED=302 bash src/mac_catseeds_e157.sh
