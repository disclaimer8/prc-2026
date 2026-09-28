#!/bin/bash
# E151 v2 (journal 136): counts fixed for equal times (brute-force checked). Rebuild features, nested CV again against the
# E106 control (as E137 / the first E151 run), gates; then the board base regardless of the gate (the decision to upload is
# taken on the gates): v520 (E116 recipe on features_e151) + 5 LightGBM seeds, v521 = blend with the 20 CatBoost seeds of the
# control line (waits for cat_reg_rank_seeds_v510.npz), tail -> v527.
set -Eeuo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=6 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/e151v2.log
rm -f data/e151v2.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/e151v2.failed' ERR
echo "[$(date +%H:%M)] build features_e151 (v2)" >> $L
POLARS_MAX_THREADS=4 FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e151.parquet $PY src/e151_own_interval_flows.py > data/e151v2_build.log 2>&1
grep -v -i warn data/e151v2_build.log | grep -E "wrote" >> $L
$PY -c "
import polars as pl
f = pl.read_parquet('data/features_e151.parquet')
c = [x for x in f.columns if x.startswith('e151_') and x[-2] in 'DA' and x[-1].isdigit()]
neg = f.select(pl.any_horizontal([pl.col(x) < 0 for x in c]).alias('neg'), 'split').group_by('split').agg(pl.col('neg').sum())
print('rows with a negative count:', dict(neg.iter_rows()))" >> $L 2>&1
out=data/cv_preds_e151v2; mkdir -p "$out"
echo "[$(date +%H:%M)] nested CV" >> $L
CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CV_OUT="$out" FEATURES=data/features_e151.parquet \
  DROP_PREFIX=day_,tq_,wxr_,month,te_ $PY src/cv_fold_preds_nested.py > data/e151v2_cv.log 2>&1
echo "[$(date +%H:%M)] gates" >> $L
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e106_air "$out" 2>&1 | tail -12 >> $L || true
export FEATURES=data/features_e151.parquet
$PY src/train.py validate > data/e151_validate.log 2>&1 && cp data/last_validation.json data/validation_v520.json
cp data/validation_v520.json data/last_validation.json
$PY src/train.py submit 520 > data/e151_submit.log 2>&1
REG=$($PY -c "import json;print(int(json.load(open('data/validation_v520.json'))['best']['reg']*1.1))")
$PY src/seed_avg.py rank 5 $REG > data/e151_lgbseeds.log 2>&1 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v520.npz
echo "[$(date +%H:%M)] base v520 done" >> $L
until [[ -s submissions/gentle-igloo_v511.parquet && -s submissions/cat_reg_rank_seeds_v510.npz ]]; do sleep 120; done
blend () {  # blend <base> <cat seeds npz> <lgb seeds npz> <features> <out>; cat_reg_rank.npz is a fixed path, restored afterwards
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_e151v2.npz
  cp -f "$2" submissions/cat_reg_rank.npz
  FEATURES=$4 LGB_SEEDS=$3 $PY src/blend.py "$1" "$5" 0.65 > data/blend_v$5.log 2>&1 || { cp -p submissions/cat_reg_rank_before_e151v2.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_e151v2.npz submissions/cat_reg_rank.npz
  grep -E "^wrote" data/blend_v$5.log >> $L
}
blend 520 submissions/cat_reg_rank_seeds_v510.npz submissions/lgb_reg_seeds_v520.npz data/features_e151.parquet 521
bash src/hz_tail_on_base.sh 521 522
cat data/tail_v522.log >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((521, 511), (527, 517)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
