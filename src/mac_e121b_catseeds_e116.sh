#!/bin/bash
# E121b (25.09): the same CatBoost seed averaging on the E116 configuration (v240 chain) — the hedge for the case
# that E120 (v265) loses to v245 on the board. Runs after E121. Output v291..v295. Seed 0 is taken from
# submissions/cat_reg_rank_v240.npz when present, otherwise fitted here (same params, random_seed=0).
set -euo pipefail
cd /Users/denyskolomiiets/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10 CATBOOST_THREADS=10 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
B=240
rm -f data/e121b.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e121b.failed' ERR
mkdir -p submissions/e121_e120_seeds && (mv submissions/cat_reg_rank_seed[0-9].npz submissions/e121_e120_seeds/ 2>/dev/null || true)
if [ -f submissions/cat_reg_rank_v$B.npz ]; then cp submissions/cat_reg_rank_v$B.npz submissions/cat_reg_rank_seed0.npz; SEEDS="1 2 3 4"; else SEEDS="0 1 2 3 4"; fi
echo "[$(date +%H:%M)] catboost seeds $SEEDS (E116 config)"
.venv/bin/python src/catboost_seeds.py 5600 $SEEDS > data/e121b_catseeds.log 2>&1
.venv/bin/python src/catboost_seeds.py merge 0 1 2 3 4 | tee -a data/e121b_catseeds.log
cp submissions/cat_reg_rank_seeds.npz submissions/cat_reg_rank_seeds_v290.npz
cp submissions/cat_reg_rank_seeds.npz submissions/cat_reg_rank.npz
echo "[$(date +%H:%M)] blend + tail of the chain"
LGB_SEEDS=submissions/lgb_reg_seeds_v$B.npz .venv/bin/python src/blend.py $B 291 0.65 | grep -E "^wrote|seed"
.venv/bin/python src/lirf_tabpfn.py 291 292 | grep -E "^wrote"
.venv/bin/python src/e14_apply.py 292 291 293 | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v291.npz \
  .venv/bin/python src/e70_apply.py 293 294 2>&1 | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v291.npz \
  .venv/bin/python src/e64_apply.py 294 295 2>&1 | grep -E "^wrote"
mkdir -p submissions/e121b_e116_seeds && mv submissions/cat_reg_rank_seed[0-9].npz submissions/e121b_e116_seeds/
cp submissions/cat_reg_rank_v240.npz submissions/cat_reg_rank.npz
.venv/bin/python - <<'PY'
import polars as pl, numpy as np
a=pl.read_parquet("submissions/gentle-igloo_v295.parquet").sort("MVT_ID_mvt")
b=pl.read_parquet("submissions/gentle-igloo_v245.parquet").sort("MVT_ID_mvt")
d=(a["TAXITIME_SEC_mvt"].to_numpy()-b["TAXITIME_SEC_mvt"].to_numpy()).astype(float)
print(f"E121b v295 vs v245: differs {int((d!=0).sum())}, max|diff| {int(np.abs(d).max())} s, RMS {np.sqrt((d**2).mean()):.3f} s")
PY
echo "[$(date +%H:%M)] done"
