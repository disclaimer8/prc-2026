#!/bin/bash
# Journal 214: same-period test of the rebuilt src/run_final_any_period.sh (T2-members recipe). The base steps are pre-filled
# from the board line so that every one of them is skipped: v901 = v801 (base 240 + three E130 members, journal 207) with
# parts_v801. The tail then runs for real: TabPFN rows of v361 -> (no E14) -> E141 keep-all -> E64 keep-all -> E152 -> E81
# reuse -> LOBT lift -> protected rows = FINAL v919, compared stage by stage with the hetzner build v802..v808
# (src/hz_build_e172.sh). On hetzner the files must be identical; on the Mac the ADS-B stage LightGBM runs on ARM, so small
# differences in v914+ are expected (v912/v913 must still be identical). The Mac lacks the head_* inputs of E141: run on hetzner
# (hz_queue/22_test808.sh, WORK on /srv).
#   MAIN=~/prc-2026 PY=~/prc-2026/.venv/bin/python bash src/test_final_tail_v808.sh
set -euo pipefail
MAIN=${MAIN:-$HOME/prc-2026}
WORK=${WORK:-$HOME/prc-final-test808}
PY=${PY:-$MAIN/.venv/bin/python}
mkdir -p "$WORK"/{logs,submissions,data/prc-2026-datasets}
cd "$WORK"
for f in features_final features_e28 features_e77 features_e106 features_e113; do ln -sf "$MAIN/data/$f.parquet" data/$f.parquet; done
"$PY" -c "
import json; j = json.load(open('$MAIN/data/last_validation.json')); assert 'reg' in j['best']; json.dump(j, open('data/validation_v900.json', 'w'))"
S=$MAIN/submissions
ln -sf "$S/gentle-igloo_v240.parquet" submissions/gentle-igloo_v900.parquet    # skipped 'submit' (only blend reads it; blend is skipped)
ln -sf "$S/lgb_reg_seeds_v240.npz" submissions/lgb_reg_seeds_v900.npz
CATP=$S/cat_reg_rank_seeds_v530.npz; [[ -s $CATP ]] || CATP=$S/cat_reg_rank_seed10.npz   # placeholders: blend is skipped
ln -sf "$CATP" submissions/cat_reg_rank_v900.npz
ln -sf "$CATP" submissions/cat_reg_rank_seeds_v900.npz
for m in raw lobt sched; do ln -sf "$S/e130_${m}_rank.npz" submissions/e130_${m}_rank_v900.npz; done
cp -f "$S/gentle-igloo_v801.parquet" submissions/gentle-igloo_v901.parquet
cp -f "$S/parts_v801.npz" submissions/parts_v901.npz
echo ok > logs/metar.ok                                                         # same period: the METAR step is not needed
MAIN=$MAIN WORK=$WORK RANKING=$MAIN/data/prc-2026-datasets/ranking.parquet SUBMITTING=$MAIN/data/prc-2026-datasets/submitting.parquet \
  TAG=test808 VB=900 CAT_SEEDS=25 MEMBERS=raw:2678:0.13,lobt:2678:0.17,sched:2826:0.13 E167=0 TABPFN_ROWS_FROM=361:291 \
  E81=reuse ADSB=reuse bash "$MAIN/src/run_final_any_period.sh"
"$PY" - <<PY
import numpy as np, polars as pl
Y = "TAXITIME_SEC_mvt"
r = lambda p: pl.read_parquet(p).sort("MVT_ID_mvt")[Y].cast(float).to_numpy()
for new, ref, name in ((912, 802, "TabPFN rows"), (913, 802, "E14 off (copy)"), (914, 803, "E141 keep-all"), (915, 804, "E64 keep-all"),
                       (916, 805, "E152"), (917, 806, "E81"), (918, 807, "LOBT"), (919, 808, "protect = FINAL")):
    d = r(f"submissions/gentle-igloo_v{new}.parquet") - r(f"$S/gentle-igloo_v{ref}.parquet")
    print(f"v{new} vs v{ref} ({name}): differing rows {(d != 0).sum()}, RMS {np.sqrt((d ** 2).mean()):.4f} s, max |d| {np.abs(d).max():.0f}")
PY
