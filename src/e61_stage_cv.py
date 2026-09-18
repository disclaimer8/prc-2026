"""E61: does recovering the discarded stand dwell help the ADS-B stage? (18.09)

    scripts/reextract_standwalk.sh                 # writes data/adsblol_standwalk
    .venv/bin/python src/e61_stage_cv.py           # 6 folds, control = today's features

Control and candidate are the same stage (E36 features, same parameters, same 800 rounds, same coverage >= 2 rule,
same clips, same base and the same folds). The only difference is which extraction the surface features come from:
data/adsblol as it stands, against data/adsblol_standwalk re-derived with STAND_WALK=1.

Why this is not another intermediate-metric experiment (the E55/E57 lesson): the flag does not transform features the
model already has, it restores observations the extractor was discarding. On the control day it changed no existing
estimate and lost none, it only added 173 and promoted 103 flights from coverage 2 to 3. And it lands where the money
is: 38.8% of the long rows (y > 3600) have a reception pause longer than GAP_S, against 0.3% of ordinary rows, and
those long rows hold 44 of the 112 s of decided-row budget.

Recorded before the number (parallel session, 18.09, 262 re-extracted days): the fix is strictly additive at scale -
283456 first-movement estimates become 328781, +45325 or +16.0%, none lost and none shifted - but it does NOT reach the
long tail. Of 3211 rows with y > 3600 the estimate count goes 172 -> 193 and coverage >= 2 does not move at all
(1191 -> 1191). The reason is the rule's own condition: the walk crosses only stationary points within STAND_M, and on a
long row the aircraft left the stand and stopped somewhere else, so there is movement in between that the walk must not
cross. The 38.8% against 0.3% figure says where the break happens, not how much of it this fix can step over.

So: a gain here comes from ordinary rows, where 73.9% of the new estimates land (10-30 min) and another 25.1% below
10 min. If the gate fails, the direction "give the stage back the observations we were discarding" is NOT closed,
because this variant never reached the population that holds 44 of the 112 s.

Gate: SSE over the rows the base model actually decides lower by >= 0.3 s LB-equivalent and lower in >= 4 of 6 folds,
and the same on the rows with y <= 7200. Strict, total and trimmed are reported, not gated.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
_argv, sys.argv = sys.argv, [sys.argv[0], "import"]
import e36_adsb_stand_geometry as E36  # noqa: E402
sys.argv = _argv
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e61_ctrl", "data/cv_preds_e61"
ARMS = {"ctrl": "data/adsblol", "cand": "data/adsblol_standwalk"}

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
base = E36.oof_base()
built = {}
for name, folder in ARMS.items():
    E36.ADSB_DIR = folder
    adsb = E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")
    built[name] = E36.build(raw, base, adsb).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
    cov = built[name]["coverage"]
    print(f"{name:5s} {folder:28s} rows {built[name].height}, coverage>=2 {int((cov >= 2).sum())}, "
          f"coverage==3 {int((cov == 3).sum())}, first movement known {int(built[name]['mvt_first_move'].is_not_null().sum())}", flush=True)

for d in (CTRL_DIR, CAND_DIR):
    os.makedirs(d, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    line = []
    for name, out in (("ctrl", CTRL_DIR), ("cand", CAND_DIR)):
        d = built[name]
        tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
        pred = E36.correct(E36.fit_stage(tr), te)
        y = te[E36.Y].to_numpy()
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E61: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
