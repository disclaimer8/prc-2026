#!/bin/bash
# E151 v2 board base on the Mac (journal 136; Denys 28.09 21:35 "про мак не забывай"): runs in parallel with the hetzner CV.
#   waits for CatBoost seed 19 (artifact + process gone), builds features_e151 v2 here with the same code (checksum sent to
#   hetzner, which compares it with its own file before blending), E116 recipe: validate -> v520 -> 5 LightGBM seeds; ships to hetzner.
set -uo pipefail
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10
PY=.venv/bin/python
L=data/mac_e151_base.log
fail () { echo "[$(date +%H:%M)] FAILED: $1" >> $L; exit 1; }
until [[ -s submissions/cat_reg_rank_seed19.npz ]] && ! pgrep -f "[c]atboost_seeds.py 5600 19" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] seed 19 done; features_e151 v2" >> $L
FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e151.parquet $PY src/e151_own_interval_flows.py > data/mac_e151_build.log 2>&1 || fail "features"
$PY - > data/e151_checksum_mac.json <<'PY' || fail "checksum"
import json, polars as pl
f = pl.read_parquet("data/features_e151.parquet")
c = sorted(x for x in f.columns if x.startswith("e151_"))
print(json.dumps({s: {x: [float(g[x].fill_nan(None).sum() or 0), int(g[x].null_count())] for x in c} for s, g in ((s, f.filter(pl.col("split") == s)) for s in ("train", "rank"))}))
PY
export FEATURES=data/features_e151.parquet
echo "[$(date +%H:%M)] validate" >> $L
$PY src/train.py validate > data/mac_e151_validate.log 2>&1 && cp data/last_validation.json data/validation_v520.json || fail "validate"
cp data/validation_v520.json data/last_validation.json
echo "[$(date +%H:%M)] submit 520" >> $L
$PY src/train.py submit 520 > data/mac_e151_submit.log 2>&1 || fail "submit"
REG=$($PY -c "import json;print(int(json.load(open('data/validation_v520.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] 5 LightGBM seeds, rounds $REG" >> $L
$PY src/seed_avg.py rank 5 $REG > data/mac_e151_lgbseeds.log 2>&1 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v520.npz || fail "seed_avg"
scp -q submissions/clf_v520*.txt submissions/reg_v520.txt submissions/gentle-igloo_v520.parquet hetzner:/root/prc-2026/submissions/ || fail "scp models"
scp -q data/validation_v520.json data/e151_checksum_mac.json hetzner:/root/prc-2026/data/ || fail "scp json"
scp -q submissions/lgb_reg_seeds_v520.npz hetzner:/root/prc-2026/submissions/ || fail "scp seeds"   # last: hetzner waits for this file
echo "[$(date +%H:%M)] done, shipped to hetzner" >> $L
