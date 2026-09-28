#!/bin/bash
# 30.09 line on hetzner (journal 124): the Mac stopped at seed 16 (Denys, 28.09 09:28), so seeds 17-19 run here after the E151 CV.
#   waits for the artifacts of E151 (fold 6+12 written and the chain gone, or its failure marker), never for a log line
#   seeds 17, 18, 19 -> submissions/e144_seeds/; check: the blend recipe on the 15 seeds of v460 gives v461 bit for bit (v519, removed)
#   cat_reg_rank_seeds_v510.npz = v460 per_seed (0-14) + seeds 15-19; v511 = blend.py 240 with it (parts_v511)
set -Eeuo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=6 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/seeds_30sep.log
rm -f data/seeds_30sep.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/seeds_30sep.failed' ERR
[[ -s submissions/cat_reg_rank_seeds_v460.npz ]] || { echo "copy cat_reg_rank_seeds_v460.npz from the mac first" | tee -a $L; exit 1; }
until [[ -e data/e151.failed ]] || { [[ -s data/cv_preds_e151_flows/fold_6_12.npz ]] && ! pgrep -f "[h]z_e151_cv.sh" > /dev/null; }; do sleep 300; done
echo "[$(date +%H:%M)] E151 finished, seeds 17-19" >> $L
mkdir -p submissions/e144_seeds
for s in 17 18 19; do
  [[ -s submissions/e144_seeds/cat_reg_rank_seed$s.npz ]] && continue
  echo "[$(date +%H:%M)] seed $s" >> $L
  $PY src/catboost_seeds.py 5600 $s > data/seed$s.log 2>&1
  mv submissions/cat_reg_rank_seed$s.npz submissions/e144_seeds/
  echo "[$(date +%H:%M)] seed $s done" >> $L
done
blend () {  # blend <cat seeds npz> <out version>; cat_reg_rank.npz is a fixed path, restored afterwards
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_30sep.npz
  cp -f "$1" submissions/cat_reg_rank.npz
  LGB_SEEDS=submissions/lgb_reg_seeds_v240.npz $PY src/blend.py 240 "$2" 0.65 > data/blend_v$2.log 2>&1 || { cp -p submissions/cat_reg_rank_before_30sep.npz submissions/cat_reg_rank.npz; return 1; }
  cp -p submissions/cat_reg_rank_before_30sep.npz submissions/cat_reg_rank.npz
  grep -E "^wrote" data/blend_v$2.log >> $L
}
blend submissions/cat_reg_rank_seeds_v460.npz 519
$PY -c "
import polars as pl, sys
a, b = (pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt') for v in (519, 461))
print('v519 == v461:', a.equals(b)); sys.exit(0 if a.equals(b) else 1)" >> $L 2>&1
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
blend submissions/cat_reg_rank_seeds_v510.npz 511
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
d = r(511) - r(461); print(f'v511 vs v461: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
