#!/bin/bash
# E151 v2 on hetzner after the base moved to the Mac (src/mac_e151_base.sh, 28.09 21:40). Replaces the rest of hz_e151v2.sh
# (its parent was stopped; the nested CV it started keeps running). Waits for artifacts only.
#   1. CV done (fold 6+12 written, cv_fold_preds_nested gone) -> gates (fold_metrics, post_e14) against cv_preds_e106_air
#   2. Mac base (lgb_reg_seeds_v520.npz shipped last) + control line (seeds_v510, v511); features checksum Mac == hetzner
#   3. v521 = blend 520 with the 20 CatBoost seeds; tail -> v527; comparisons with the control v511 / v517
set -Eeuo pipefail
cd /root/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6
PY=.venv/bin/python
L=data/e151v2.log
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee -a $L data/e151v2.failed' ERR
out=data/cv_preds_e151v2
until [[ -s $out/fold_6_12.npz ]] && ! pgrep -f "[c]v_fold_preds_nested" > /dev/null; do sleep 120; done
echo "[$(date +%H:%M)] gates" >> $L
PYTHONPATH=src $PY src/fold_metrics.py data/cv_preds_e106_air "$out" 2>&1 | tail -9 >> $L || true
PYTHONPATH=src $PY src/post_e14.py data/cv_preds_e106_air "$out" 2>&1 | tail -8 >> $L || true
until [[ -s submissions/lgb_reg_seeds_v520.npz && -s data/e151_checksum_mac.json && -s submissions/cat_reg_rank_seeds_v510.npz && -s submissions/gentle-igloo_v511.parquet ]]; do sleep 120; done
$PY - >> $L 2>&1 <<'PY'
import json, polars as pl
f = pl.read_parquet("data/features_e151.parquet")
c = sorted(x for x in f.columns if x.startswith("e151_"))
here = {s: {x: [float(g[x].fill_nan(None).sum() or 0), int(g[x].null_count())] for x in c} for s, g in ((s, f.filter(pl.col("split") == s)) for s in ("train", "rank"))}
mac = json.load(open("data/e151_checksum_mac.json"))
bad = [(s, x) for s in here for x in here[s] if here[s][x][1] != mac[s][x][1] or abs(here[s][x][0] - mac[s][x][0]) > 1e-6 * max(1.0, abs(here[s][x][0]))]
print("features_e151 checksum mac == hetzner:", not bad, bad[:5])
assert not bad
PY
cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_e151v2.npz
cp -f submissions/cat_reg_rank_seeds_v510.npz submissions/cat_reg_rank.npz
FEATURES=data/features_e151.parquet LGB_SEEDS=submissions/lgb_reg_seeds_v520.npz $PY src/blend.py 520 521 0.65 > data/blend_v521.log 2>&1 || { cp -p submissions/cat_reg_rank_before_e151v2.npz submissions/cat_reg_rank.npz; false; }
cp -p submissions/cat_reg_rank_before_e151v2.npz submissions/cat_reg_rank.npz
grep -E "^wrote" data/blend_v521.log >> $L
bash src/hz_tail_on_base.sh 521 522
cat data/tail_v522.log >> $L
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((521, 511), (527, 517)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s')" >> $L 2>&1
echo "[$(date +%H:%M)] done" >> $L
