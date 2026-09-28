#!/bin/bash
# 30.09 control line on hetzner (journal 124, 136): 20 CatBoost seeds without E151 (split out of hz_e151_build.sh after the
# E151 tie bug). Seed 17 here, 18/19 from the Mac; recipe check on v461; v510 = 20 seeds, v511 = blend 240; tail -> v517.
set -Eeuo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=6 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/seeds20_control.log
rm -f data/seeds20_control.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/seeds20_control.failed' ERR
mkdir -p submissions/e144_seeds
# seed 17 (started by the killed queue): wait for its file and for the process to end
until [[ -s submissions/e144_seeds/cat_reg_rank_seed17.npz ]] || { [[ -s submissions/cat_reg_rank_seed17.npz ]] && ! pgrep -f "[c]atboost_seeds.py 5600 17" > /dev/null; }; do sleep 120; done
[[ -s submissions/cat_reg_rank_seed17.npz ]] && mv submissions/cat_reg_rank_seed17.npz submissions/e144_seeds/
echo "[$(date +%H:%M)] seed 17 in e144_seeds" >> $L
blend () {  # blend <base> <cat seeds npz> <lgb seeds npz> <features> <out>; cat_reg_rank.npz is a fixed path, restored afterwards
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_ctrl20.npz
  cp -f "$2" submissions/cat_reg_rank.npz
  FEATURES=$4 LGB_SEEDS=$3 $PY src/blend.py "$1" "$5" 0.65 > data/blend_v$5.log 2>&1 || { cp -p submissions/cat_reg_rank_before_ctrl20.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_ctrl20.npz submissions/cat_reg_rank.npz
  grep -E "^wrote" data/blend_v$5.log >> $L
}
until [[ -s submissions/e144_seeds/cat_reg_rank_seed18.npz && -s submissions/e144_seeds/cat_reg_rank_seed19.npz ]]; do sleep 120; done
echo "[$(date +%H:%M)] seeds 18/19 from the mac" >> $L
blend () {  # blend <base> <cat seeds npz> <lgb seeds npz> <features> <out>; cat_reg_rank.npz is a fixed path, restored afterwards
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_e151b.npz
  cp -f "$2" submissions/cat_reg_rank.npz
  FEATURES=$4 LGB_SEEDS=$3 $PY src/blend.py "$1" "$5" 0.65 > data/blend_v$5.log 2>&1 || { cp -p submissions/cat_reg_rank_before_e151b.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_e151b.npz submissions/cat_reg_rank.npz
  grep -E "^wrote" data/blend_v$5.log >> $L
}
blend 240 submissions/cat_reg_rank_seeds_v460.npz submissions/lgb_reg_seeds_v240.npz data/features_e106.parquet 519
$PY -c "
import polars as pl, sys
a, b = (pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt') for v in (519, 461))
print('recipe check v519 == v461:', a.equals(b)); sys.exit(0 if a.equals(b) else 1)" >> $L 2>&1
rm -f submissions/gentle-igloo_v519.parquet submissions/parts_v519.npz
$PY - >> $L 2>&1 <<'PY'
import numpy as np
z = np.load("submissions/cat_reg_rank_seeds_v460.npz"); ids, P = z["mvt_id"], z["per_seed"]
assert P.shape[0] == 15, P.shape
extra = [np.load(f"submissions/e144_seeds/cat_reg_rank_seed{s}.npz") for s in range(15, 20)]
assert all(np.array_equal(e["mvt_id"], ids) for e in extra)
P2 = np.vstack([P] + [e["reg"][None, :] for e in extra])
np.savez("submissions/cat_reg_rank_seeds_v510.npz", mvt_id=ids, reg=P2.mean(axis=0), per_seed=P2)
print(f"merged {P2.shape[0]} CatBoost seeds; RMS change of the average vs 15 seeds {np.sqrt(((P2.mean(0) - P.mean(0)) ** 2).mean()):.2f} s")
PY
blend 240 submissions/cat_reg_rank_seeds_v510.npz submissions/lgb_reg_seeds_v240.npz data/features_e106.parquet 511
bash src/hz_tail_on_base.sh 511 512
cat data/tail_v512.log >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((511, 461), (517, 467)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
