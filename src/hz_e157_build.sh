#!/bin/bash
# E157 board build (journal 161, pre-reg 152): wake-ladder features in the base. Built ahead of the CV verdict (fold 1 -1.47);
# goes to the board only if the CV gate passes. U3 recipe (hz_u3_build.sh) on features_e157: base v620 (E116 recipe + 5 LightGBM
# seeds), v621 = blend with the 25 CatBoost seeds (seeds_v530, E106 features), S1 tail -> v622..v628, compared with S1 v538.
# Waits for the E156 line (hz_e156_line.sh) to free cores 6-11.
set -Eeuo pipefail
cd /root/prc-2026
while pgrep -f "[h]z_e156_line.sh" > /dev/null || pgrep -f "[h]z_e130.sh" > /dev/null; do sleep 120; done
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/e157_build.log
rm -f data/e157_build.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/e157_build.failed' ERR
echo "[$(date +%H:%M)] start" >> $L
export FEATURES=data/features_e157.parquet
$PY src/train.py validate > data/e157_validate.log 2>&1 && cp data/last_validation.json data/validation_v620.json
cp data/validation_v620.json data/last_validation.json
$PY src/train.py submit 620 > data/e157_submit.log 2>&1
REG=$($PY -c "import json;print(int(json.load(open('data/validation_v620.json'))['best']['reg']*1.1))")
$PY src/seed_avg.py rank 5 $REG > data/e157_lgbseeds.log 2>&1 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v620.npz
echo "[$(date +%H:%M)] base v620 done" >> $L
cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_e157.npz
cp -f submissions/cat_reg_rank_seeds_v530.npz submissions/cat_reg_rank.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v620.npz $PY src/blend.py 620 621 0.65 > data/blend_v621.log 2>&1 || { cp -p submissions/cat_reg_rank_before_e157.npz submissions/cat_reg_rank.npz; false; }
cp -p submissions/cat_reg_rank_before_e157.npz submissions/cat_reg_rank.npz
grep -E "^wrote" data/blend_v621.log >> $L
export FEATURES=data/features_e106.parquet
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
B=621 V=622 $PY - >> $L 2>&1 <<'PY'
import os, polars as pl
Y = "TAXITIME_SEC_mvt"; B, V = int(os.environ["B"]), int(os.environ["V"])
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
chg = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt", pl.col(Y).alias("tab"))
s = r(B)
s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns).write_parquet(f"submissions/gentle-igloo_v{V}.parquet")
print(f"wrote v{V}: v{B} + the {chg.height} TabPFN rows of v361")
PY
$PY src/e14_apply.py 622 621 623 2>&1 | grep -E "^wrote" >> $L
STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v621.npz $PY src/e141_apply.py 623 624 2>&1 | grep -E "^wrote" >> $L
E64_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v621.npz $PY src/e64_apply.py 624 625 2>&1 | grep -E "^wrote" >> $L
cp submissions/gentle-igloo_v625.parquet submissions/gentle-igloo_v626.parquet
BLEND_BASE=626 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 627 2>&1 | grep -viE "warn" | tail -1 >> $L
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 627 628 2>&1 | grep -viE "warn" | tail -1 >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((621, 531), (628, 538)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
s = pl.read_parquet('submissions/gentle-igloo_v628.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0; print('row checks ok')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
