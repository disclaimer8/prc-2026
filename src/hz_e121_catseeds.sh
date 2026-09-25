#!/bin/bash
# E121 (25.09): CatBoost seed averaging (the E79 script, never run) on the E120 configuration (v260 chain, hetzner).
# Seeds 1-4 fitted here, seed 0 = submissions/cat_reg_rank_v260.npz; merged average replaces the single CatBoost
# in the blend, then the unchanged tail of the chain: TabPFN -> E14 -> E70 -> E64. Output v281..v285.
set -euo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 CATBOOST_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config,ap_runway TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
B=260
rm -f data/e121.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e121.failed' ERR
echo "[$(date +%H:%M)] catboost seeds 1-4 (5600 it, ~1h45 each)"
.venv/bin/python src/catboost_seeds.py 5600 1 2 3 4 > data/e121_catseeds.log 2>&1
cp submissions/cat_reg_rank_v$B.npz submissions/cat_reg_rank_seed0.npz
.venv/bin/python src/catboost_seeds.py merge 0 1 2 3 4 | tee -a data/e121_catseeds.log
cp submissions/cat_reg_rank_seeds.npz submissions/cat_reg_rank_seeds_v280.npz
cp submissions/cat_reg_rank_seeds.npz submissions/cat_reg_rank.npz   # blend.py reads this fixed path
echo "[$(date +%H:%M)] blend + tail of the chain"
LGB_SEEDS=submissions/lgb_reg_seeds_v$B.npz .venv/bin/python src/blend.py $B 281 0.65 | grep -E "^wrote|seed"
.venv/bin/python src/lirf_tabpfn.py 281 282 | grep -E "^wrote"
.venv/bin/python src/e14_apply.py 282 281 283 | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v281.npz \
  .venv/bin/python src/e70_apply.py 283 284 2>&1 | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v281.npz \
  .venv/bin/python src/e64_apply.py 284 285 2>&1 | grep -E "^wrote"
cp submissions/cat_reg_rank_v$B.npz submissions/cat_reg_rank.npz   # restore the single-seed file for other chains
.venv/bin/python - <<'PY'
import polars as pl, numpy as np
a=pl.read_parquet("submissions/gentle-igloo_v285.parquet").sort("MVT_ID_mvt")
b=pl.read_parquet("submissions/gentle-igloo_v265.parquet").sort("MVT_ID_mvt")
d=(a["TAXITIME_SEC_mvt"].to_numpy()-b["TAXITIME_SEC_mvt"].to_numpy()).astype(float)
print(f"E121 v285 vs v265: differs {int((d!=0).sum())}, max|diff| {int(np.abs(d).max())} s, RMS {np.sqrt((d**2).mean()):.3f} s")
PY
echo "[$(date +%H:%M)] done"
