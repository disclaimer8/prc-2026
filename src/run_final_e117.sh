#!/bin/zsh
# E117 полная цепочка (24.09): E112, но ter_op_ap возвращён (снят только ter_stand_rwy); с нуля 250-255.
set -euo pipefail
cd "${0:a:h}/.."
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_stand_rwy,STAND_mvt,ap_stand TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1
V=250
echo "[$(date +%H:%M)] validate"
uv run python src/train.py validate > data/e117_validate.log 2>&1
grep -E "^best=" data/e117_validate.log
uv run python src/train.py submit $V | grep -E "^wrote"
REG=$(python3 -c "import json;print(int(json.load(open('data/last_validation.json'))['best']['reg']*1.1))")
echo "[$(date +%H:%M)] seeds"
uv run python src/seed_avg.py rank 5 $REG | grep -E "saved"
cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$V.npz
echo "[$(date +%H:%M)] catboost"
uv run python src/catboost_reg.py 5600 > data/e117_catboost.log 2>&1
cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$V.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v$V.npz uv run python src/blend.py $V $((V+1)) 0.65 | grep -E "^wrote"
uv run python src/lirf_tabpfn.py $((V+1)) $((V+2)) | grep -E "^wrote"
uv run python src/e14_apply.py $((V+2)) $((V+1)) $((V+3)) | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e70_apply.py $((V+3)) $((V+4)) 2>&1 | grep -E "^wrote"
ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v$((V+1)).npz \
  .venv/bin/python src/e64_apply.py $((V+4)) $((V+5)) 2>&1 | grep -E "^wrote"
echo "[$(date +%H:%M)] сравнение с v115 (размер изменения)"
.venv/bin/python - <<'PY'
import polars as pl, numpy as np
a=pl.read_parquet("submissions/gentle-igloo_v255.parquet").sort("MVT_ID_mvt")
b=pl.read_parquet("submissions/gentle-igloo_v115.parquet").sort("MVT_ID_mvt")
d=(a["TAXITIME_SEC_mvt"].to_numpy()-b["TAXITIME_SEC_mvt"].to_numpy())
print(f"E117 vs v115: строк {len(d)}, отличается {int((d!=0).sum())}, max|diff| {int(np.abs(d).max())} с, RMS {np.sqrt((d.astype(float)**2).mean()):.3f} с")
PY
