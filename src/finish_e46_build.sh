#!/bin/zsh
# Finish the E46 final on the Mac from what home-pc built (v51 regimes, v52 blend with CatBoost):
# TabPFN tail table -> v53, E14 logit -> v54, E36 stand-geometry stage -> v55 (the version to upload).
# home-pc has no tabpfn package, so only the heavy LightGBM/CatBoost part runs there.
set -euo pipefail
cd "${0:a:h}/.."
for f in submissions/gentle-igloo_v52.parquet submissions/parts_v52.npz; do
  [[ -f $f ]] || { echo "missing $f — copy it from home-pc first"; exit 1; }
done
export FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
uv run python src/lirf_tabpfn.py 52 53 | grep -E "^wrote"
uv run python src/e14_apply.py 53 52 54 | grep -E "^wrote"
PARTS=submissions/parts_v52.npz .venv/bin/python src/e36_adsb_stand_geometry.py apply 54 55 2>&1 | grep -E "^wrote|regime probabilities"
echo "[$(date +%H:%M)] FINAL submissions/gentle-igloo_v55.parquet"
