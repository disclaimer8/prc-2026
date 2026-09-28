#!/bin/zsh
# 30.09 line, Mac part (journal 124): after CatBoost seeds 15-19 (src/mac_e144_catseeds.sh) merge 20 seeds and blend -> v511.
#   check first: the same recipe on the 15 seeds of v460 must give v461 bit for bit (written as v519, then removed)
#   cat_reg_rank_seeds_v510.npz = v460 per_seed (seeds 0-14) + seeds 15-19; v511 = blend.py 240 with it; v511 + parts_v511 -> hetzner
set -uo pipefail
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
L=data/merge20_30sep.log
until grep -qE "\] (done|FAILED)" data/e144_catseeds.log; do sleep 120; done
grep -q "FAILED" data/e144_catseeds.log && { echo "[$(date +%H:%M)] seed run FAILED, nothing merged" >> $L; exit 1; }
blend () {  # blend <cat seeds npz> <out version>; cat_reg_rank.npz is a fixed path, restored afterwards
  cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_merge20.npz
  cp -f $1 submissions/cat_reg_rank.npz
  LGB_SEEDS=submissions/lgb_reg_seeds_v240.npz .venv/bin/python src/blend.py 240 $2 0.65 2>&1 | grep -E "^wrote" >> $L
  local rc=${pipestatus[1]}
  cp -p submissions/cat_reg_rank_before_merge20.npz submissions/cat_reg_rank.npz
  return $rc
}
echo "[$(date +%H:%M)] check: 15 seeds of v460 -> v519 must equal v461" >> $L
blend submissions/cat_reg_rank_seeds_v460.npz 519 || { echo "[$(date +%H:%M)] blend FAILED" >> $L; exit 1; }
.venv/bin/python -c "
import polars as pl, sys
a, b = (pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt') for v in (519, 461))
print('v519 == v461:', a.equals(b)); sys.exit(0 if a.equals(b) else 1)" >> $L 2>&1 || { echo "[$(date +%H:%M)] recipe does not reproduce v461, stop" >> $L; exit 1; }
rm -f submissions/gentle-igloo_v519.parquet submissions/parts_v519.npz
.venv/bin/python - >> $L 2>&1 <<'PY' || { echo "[$(date +%H:%M)] merge FAILED" >> $L; exit 1; }
import numpy as np
z = np.load("submissions/cat_reg_rank_seeds_v460.npz"); ids, P = z["mvt_id"], z["per_seed"]
assert P.shape[0] == 15, P.shape
extra = [np.load(f"submissions/cat_reg_rank_seed{s}.npz") for s in range(15, 20)]
assert all(np.array_equal(e["mvt_id"], ids) for e in extra)
P2 = np.vstack([P] + [e["reg"][None, :] for e in extra])
np.savez("submissions/cat_reg_rank_seeds_v510.npz", mvt_id=ids, reg=P2.mean(axis=0), per_seed=P2)
print(f"merged {P2.shape[0]} CatBoost seeds; RMS change of the average vs 15 seeds {np.sqrt(((P2.mean(0) - P.mean(0)) ** 2).mean()):.2f} s")
PY
blend submissions/cat_reg_rank_seeds_v510.npz 511 || { echo "[$(date +%H:%M)] blend v511 FAILED" >> $L; exit 1; }
.venv/bin/python -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
d = r(511) - r(461); print(f'v511 vs v461: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')" >> $L 2>&1
scp -q submissions/gentle-igloo_v511.parquet submissions/parts_v511.npz submissions/cat_reg_rank_seeds_v510.npz hetzner:/root/prc-2026/submissions/ \
  && echo "[$(date +%H:%M)] v511 on hetzner" >> $L
echo "[$(date +%H:%M)] done" >> $L
