#!/bin/bash
# Night 28 -> 29.09 on the Mac (Denys: "пусть считает всю ночь"). After the production-config E151 CV pair (mac_e151_prodcv.sh):
#   E151 holds (after E14 decided < 0, trimmed < 0, lower >= 4/6) -> CatBoost seeds on features_e151 (seeds 100+, moved to
#   submissions/e151_catseeds/): the E151 line still blends a CatBoost trained on E106 features (weight 0.65).
#   otherwise -> more E106 CatBoost seeds 20+ (copied to hetzner e144_seeds). New seeds start until 07:45.
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=10 OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10
PY=.venv/bin/python
L=data/mac_night_28sep.log
while pgrep -f "[m]ac_e151_prodcv.sh" > /dev/null; do sleep 120; done
res=""
if [[ -s data/cv_preds_e151prod/fold_6_12.npz && -s data/cv_preds_e116prod/fold_6_12.npz ]]; then
  res=$(PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e116prod data/cv_preds_e151prod 2>&1 | grep "AFTER E14" | tail -1)
fi
echo "[$(date +%H:%M)] production CV: ${res:-missing}" >> $L
pass=$($PY -c "
import re, sys
m = re.search(r'decided ([-+0-9.]+), trimmed ([-+0-9.]+), lower (\d)/6', sys.argv[1])
print(1 if m and float(m.group(1)) < 0 and float(m.group(2)) < 0 and int(m.group(3)) >= 4 else 0)" "$res")
if [[ $pass == 1 ]]; then
  export FEATURES=data/features_e151.parquet; s=100; dest=submissions/e151_catseeds
else
  export FEATURES=data/features_e106.parquet; s=20; dest=submissions
fi
mkdir -p $dest
echo "[$(date +%H:%M)] E151 passes: $pass -> CatBoost on $FEATURES from seed $s" >> $L
while [[ $(date +%H%M) < 0745 || $(date +%H%M) > 2000 ]]; do
  echo "[$(date +%H:%M)] seed $s" >> $L
  nice -n 10 $PY src/catboost_seeds.py 5600 $s >> data/mac_night_seeds.log 2>&1 || { echo "[$(date +%H:%M)] seed $s FAILED" >> $L; break; }
  if [[ $pass == 1 ]]; then mv submissions/cat_reg_rank_seed$s.npz $dest/; else scp -q submissions/cat_reg_rank_seed$s.npz hetzner:/root/prc-2026/submissions/e144_seeds/; fi
  echo "[$(date +%H:%M)] seed $s done" >> $L
  s=$((s + 1))
done
echo "[$(date +%H:%M)] done" >> $L
