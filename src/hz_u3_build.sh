#!/bin/bash
# U3 board build (journal 145): E51 route geometry on the production feature set. Base v590 (E116 recipe on features_e106_e51)
# + 5 LightGBM seeds, v591 = blend with the 25 CatBoost seeds (seeds_v530, E106 features), S1 tail -> v592..v598.
set -Eeuo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/u3_build.log
rm -f data/u3_build.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/u3_build.failed' ERR
echo "[$(date +%H:%M)] features_e106_e51" >> $L
[[ -s data/features_e106_e51.parquet ]] || FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e106_e51.parquet $PY src/e51_add_route_features.py >> $L 2>&1
export FEATURES=data/features_e106_e51.parquet
$PY src/train.py validate > data/u3_validate.log 2>&1 && cp data/last_validation.json data/validation_v590.json
cp data/validation_v590.json data/last_validation.json
$PY src/train.py submit 590 > data/u3_submit.log 2>&1
REG=$($PY -c "import json;print(int(json.load(open('data/validation_v590.json'))['best']['reg']*1.1))")
$PY src/seed_avg.py rank 5 $REG > data/u3_lgbseeds.log 2>&1 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v590.npz
echo "[$(date +%H:%M)] base v590 done" >> $L
cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_u3.npz
cp -f submissions/cat_reg_rank_seeds_v530.npz submissions/cat_reg_rank.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v590.npz $PY src/blend.py 590 591 0.65 > data/blend_v591.log 2>&1 || { cp -p submissions/cat_reg_rank_before_u3.npz submissions/cat_reg_rank.npz; false; }
cp -p submissions/cat_reg_rank_before_u3.npz submissions/cat_reg_rank.npz
grep -E "^wrote" data/blend_v591.log >> $L
export FEATURES=data/features_e106.parquet
D7=n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts
D3=vi_n_pts,vi_prev_pts,vi_alt_first
B=591 V=592 $PY - >> $L 2>&1 <<'PY'
import os, polars as pl
Y = "TAXITIME_SEC_mvt"; B, V = int(os.environ["B"]), int(os.environ["V"])
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
chg = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt", pl.col(Y).alias("tab"))
s = r(B)
s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns).write_parquet(f"submissions/gentle-igloo_v{V}.parquet")
print(f"wrote v{V}: v{B} + the {chg.height} TabPFN rows of v361")
PY
$PY src/e14_apply.py 592 591 593 2>&1 | grep -E "^wrote" >> $L
STAGE_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D7 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v591.npz $PY src/e141_apply.py 593 594 2>&1 | grep -E "^wrote" >> $L
E64_KEEP_ALL=1 STAGE_SEEDS=42,1,2,3,4 DROP_FEATS=$D3 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v591.npz $PY src/e64_apply.py 594 595 2>&1 | grep -E "^wrote" >> $L
cp submissions/gentle-igloo_v595.parquet submissions/gentle-igloo_v596.parquet
BLEND_BASE=596 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 597 2>&1 | grep -viE "warn" | tail -1 >> $L
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 597 598 2>&1 | grep -viE "warn" | tail -1 >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((591, 531), (598, 538)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
s = pl.read_parquet('submissions/gentle-igloo_v598.parquet'); assert s.height == 344841 and s['TAXITIME_SEC_mvt'].null_count() == 0; print('row checks ok')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
