#!/bin/bash
# 30.09 bases on hetzner (journal 124, 131, 135). Replaces src/hz_seeds_30sep.sh: the Mac is computing again (28.09 20:11) and
# does CatBoost seeds 18-19; hetzner finishes seed 17 (already running, parent killed), then:
#   E151 base (E116 recipe on data/features_e151.parquet): validate -> v520 submit -> 5 LightGBM seeds (lgb_reg_seeds_v520)
#   waits for seeds 18/19 from the Mac (artifacts in submissions/e144_seeds/), checks the blend recipe on v461 (v519, removed)
#   20 CatBoost seeds -> cat_reg_rank_seeds_v510.npz; v511 = blend 240 (control line), v521 = blend 520 (E151 line)
#   standard tails: hz_tail_on_base.sh 511 512 -> v517 (+ v518 E145), 521 522 -> v527 (+ v528)
set -Eeuo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=6 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/e151_build.log
rm -f data/e151_build.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/e151_build.failed' ERR
mkdir -p submissions/e144_seeds
# seed 17 (started by the killed queue): wait for its file and for the process to end
until [[ -s submissions/e144_seeds/cat_reg_rank_seed17.npz ]] || { [[ -s submissions/cat_reg_rank_seed17.npz ]] && ! pgrep -f "[c]atboost_seeds.py 5600 17" > /dev/null; }; do sleep 120; done
[[ -s submissions/cat_reg_rank_seed17.npz ]] && mv submissions/cat_reg_rank_seed17.npz submissions/e144_seeds/
echo "[$(date +%H:%M)] seed 17 in e144_seeds; E151 base" >> $L

export FEATURES=data/features_e151.parquet
if [[ ! -s submissions/lgb_reg_seeds_v520.npz ]]; then
  $PY src/train.py validate > data/e151_validate.log 2>&1 && cp data/last_validation.json data/validation_v520.json
  cp data/validation_v520.json data/last_validation.json
  $PY src/train.py submit 520 > data/e151_submit.log 2>&1
  REG=$($PY -c "import json;print(int(json.load(open('data/validation_v520.json'))['best']['reg']*1.1))")
  $PY src/seed_avg.py rank 5 $REG > data/e151_lgbseeds.log 2>&1 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v520.npz
fi
echo "[$(date +%H:%M)] E151 base v520: $(tail -c 300 data/last_validation.json | tr -d '\n')" >> $L

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
blend 520 submissions/cat_reg_rank_seeds_v510.npz submissions/lgb_reg_seeds_v520.npz data/features_e151.parquet 521
echo "[$(date +%H:%M)] tails" >> $L
bash src/hz_tail_on_base.sh 511 512
bash src/hz_tail_on_base.sh 521 522
cat data/tail_v512.log data/tail_v522.log >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((511, 461), (517, 467), (521, 511), (527, 517)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
