#!/bin/zsh
# E77: rebuild the accepted chain on the filed-time twins (20.09). Same chain as run_final_mb3.sh minus
# SERVED_ONLY (MB3 was closed by the board at +0.010 s), on data/features_e77.parquet.
#
# The gate result that justifies this build (data/e77_cv.log, both arms recomputed from one feature file
# differing only by DROP_PREFIX): decided -2.73 s, lower in 6 of 6, strict -2.84, y<=7200 -2.38,
# trimmed -3.11. That is nine times the registered centre of -0.30, so the size is a warning, not a
# trophy -- E38 won 10.98 s of CV and lost 3.79 on the board, E50 won 2.79 and lost 5.78.
#
# What separates this from those two: the gain sits where the mechanism says it must. Per airport, on
# decided rows, LTFM -1.251 (the airport whose AOBT_3 is a three-tooth comb), LIRF -0.531, then everything
# else between -0.30 and -0.06, and EHAM -- the best covered airport, the one with nothing wrong with its
# proxy -- at +0.006. A leak would not know which airport is broken. E38 and E50 both added learned
# capacity; this adds an observation the model did not have, which is the class that has transferred.
#
# REGISTERED PREDICTION, WRITTEN BEFORE THE BUILD: v97 lands at 246.0, band 245.3 .. 246.8, against
# v82 = 247.7501 (the same chain on the old features). Not the full -2.73: the CV pair is base model
# against base model, while the chain adds E70 and E64 on top, and those correct the same ADS-B-covered
# rows at LIRF and elsewhere. LTFM cannot be absorbed that way -- its ADS-B coverage is 0.000 -- and LTFM
# is 46% of the measured gain, which is what holds the centre up.
#
#   zsh src/run_final_e77.sh          log: data/final_e77*.log, FINAL submissions/gentle-igloo_v97.parquet
set -euo pipefail
cd "${0:a:h}/.."
export FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
git diff --quiet HEAD -- src/train.py src/features.py src/target_encoding.py src/blend.py src/catboost_reg.py src/seed_avg.py src/lirf_tabpfn.py src/e14_apply.py src/e36_adsb_stand_geometry.py || { echo "ABORT: uncommitted model code"; exit 1; }
V=92
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/final_e77_validate.log 2>&1
grep -E "^best=" data/final_e77_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$V.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/final_e77_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e70_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote|regime probabilities"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e64_apply.py $((V+4)) $((V+5)) 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v$((V+5)).parquet"
