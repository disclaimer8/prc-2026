#!/bin/bash
# E144 tail (journal 110) on hetzner: base v441 (blend with 10 CatBoost seeds) -> the v424 line.
#   TabPFN rows transplanted from v361 (TabPFN is not deterministic run to run)
#   v441 -> v442 TabPFN pfx -> v443 E14 -> v444 E141 x5 -> v445 E64 x5 -> v446 E81 -> v447 LOBT (FINAL)
set -Eeuo pipefail
cd /root/prc-2026
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage  # the hz_e135 environment that built v361
export FEATURES=data/features_e106.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
export DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
PY=.venv/bin/python
rm -f data/e144t.failed
trap 'echo "[$(date +%H:%M)] FAILED at line $LINENO" | tee data/e144t.failed' ERR
same () { $PY -c "import sys, polars as pl; a, b = (pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt') for v in sys.argv[1:3]); print(f'v{sys.argv[1]} == v{sys.argv[2]}:', a.equals(b)); sys.exit(0 if a.equals(b) else 1)" "$1" "$2"; }
echo "[$(date +%H:%M)] tail on v441"
# TabPFN is not deterministic run to run on this box (two runs on identical inputs differ on all 42 rows, up to 873 s):
# keep the exact TabPFN draw of the v366 line by transplanting the 42 rows v361 changed against v291.
$PY - <<'PY'
import polars as pl
Y = "TAXITIME_SEC_mvt"
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
b, t, s = r(291), r(361), r(441)
chg = t.join(b, on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt", pl.col(Y).alias("tab"))
out = s.join(chg, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).alias(Y)).select(s.columns)
out.write_parquet("submissions/gentle-igloo_v442.parquet")
print(f"wrote v442: v441 + the {chg.height} TabPFN rows of v361")
PY
$PY src/e14_apply.py 442 441 443 2>&1 | grep -E "^wrote"
STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=submissions/parts_v441.npz $PY src/e141_apply.py 443 444 2>&1 | grep -E "seeds|^wrote"
STAGE_SEEDS=42,1,2,3,4 ADSB_DIR=data/adsblol_kt1 PARTS=submissions/parts_v441.npz $PY src/e64_apply.py 444 445 2>&1 | grep -E "seeds|^wrote"
BLEND_BASE=445 BLEND_PARTS="44:0.1214,60:0.0270" $PY src/blend_submissions.py 446 2>&1 | grep -viE "warn" | tail -1
LOBT_CLIP=1 $PY src/lobt_edge_lift.py 446 447 2>&1 | grep -viE "warn" | tail -1
$PY -c "
import numpy as np, polars as pl
r = lambda v: pl.read_parquet(f'submissions/gentle-igloo_v{v}.parquet').sort('MVT_ID_mvt')['TAXITIME_SEC_mvt'].cast(float).to_numpy()
for a, b in ((447, 424), (447, 366)):
    d = r(a) - r(b); print(f'v{a} vs v{b}: changed {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.2f} s, max |d| {np.abs(d).max():.0f}')
"
echo "[$(date +%H:%M)] done"
