"""E59: fit the ADS-B stage against the base it actually corrects (17.09).

    .venv/bin/python src/e59_stage_base.py        6 folds: stage fitted on the E46 base vs stage fitted on the old base

The stage learns a residual y - base and takes `base`, `base_vs_first_move` and `base_vs_first` as features, so it is
tied to the base model it was fitted against. src/e36_adsb_stand_geometry.py fits it on data/cv_preds_nomonth - the
canonical model before E46 - but every submission since v55 applies it on top of the E46 chain, whose predictions sit
38.6 s RMS away (median |difference| 16 s, per fold 35.0-46.2 s). So today's chain corrects the residuals of a model it
no longer is.

Control and candidate differ only in the table the stage is fitted on; features, parameters, rounds, the coverage >= 2
rule and both clips are untouched. Both are applied to the same rows with the same E46 base, so the comparison measures
exactly the mismatch. Type of change (REVIEW_BRIEF.md, "Read first" #2): the stage on a physical observation of the
flight, the second family that has transferred (E25 -15.2, E30 -2.0, E36 -3.7 s on the leaderboard). No new capacity:
same model, same features, correct control.

Gate: SSE over the rows the base model actually decides (src/fold_metrics.py) lower by >= 0.3 s LB-equivalent and lower
in >= 4 of 6 folds; the same on the rows with y <= 7200. Strict, total and trimmed are reported, not gated.

If it passes, the submission needs no rebuild of the model - only the apply step of the existing chain is re-run:
    PARTS=submissions/parts_v52.npz E36_BASE_DIR=data/cv_preds_e46 \
        .venv/bin/python src/e36_adsb_stand_geometry.py apply 54 76
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
# e36_adsb_stand_geometry reads sys.argv[1] at module level and branches on it; "import" matches neither branch.
_argv, sys.argv = sys.argv, [sys.argv[0], "import"]
import e36_adsb_stand_geometry as E36  # noqa: E402
sys.argv = _argv
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e59_ctrl", "data/cv_preds_e59"
OLD_BASE, NEW_BASE = "data/cv_preds_nomonth", "data/cv_preds_e46"

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
adsb = E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")
d_old = E36.build(raw, E36.oof_base(OLD_BASE), adsb).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
d_new = E36.build(raw, E36.oof_base(NEW_BASE), adsb).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
print(f"rows: old base {d_old.height}, new base {d_new.height}", flush=True)

for folder in (CTRL_DIR, CAND_DIR):
    os.makedirs(folder, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr_old = d_old.filter(~pl.col("month").is_in([a, b]))
    tr_new = d_new.filter(~pl.col("month").is_in([a, b]))
    te = d_new.filter(pl.col("month").is_in([a, b]))
    y = te[E36.Y].to_numpy()
    ctrl = E36.correct(E36.fit_stage(tr_old), te)   # stage fitted on the old base, correcting the E46 base: today's chain
    cand = E36.correct(E36.fit_stage(tr_new), te)   # stage fitted on the base it corrects
    mvt = te["MVT_ID_mvt"].to_numpy()
    np.savez(f"{CTRL_DIR}/fold_{a}_{b}.npz", mvt_id=mvt, pred=ctrl, y=y)
    np.savez(f"{CAND_DIR}/fold_{a}_{b}.npz", mvt_id=mvt, pred=cand, y=y)
    print(f"fold {a}+{b}: rows {te.height}, corrected {int((te['coverage'] >= 2).sum())}, "
          f"RMSE {np.sqrt(np.mean((ctrl - y) ** 2)):.2f} -> {np.sqrt(np.mean((cand - y) ** 2)):.2f}", flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E59: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
