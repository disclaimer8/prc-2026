"""E62: tell the stage which stand dwells were reconstructed (18.09).

    STAND_WALK=2 scripts/reextract_standwalk.sh   # after pointing it at data/adsblol_e62
    .venv/bin/python src/e62_stage_cv.py

E61 recovered 68273 first-movement estimates and they helped the rows that got them (-0.063 s), but the run lost
0.25 s overall because the damage landed on the 1.97M rows whose features never changed (+0.261 s). The stage is one
model over all rows, so turning nulls into values and moving 45629 rows from coverage 2 to 3 re-trains the trees, and
that costs more than the new information is worth.

E62 changes the delivery, not the fix. The dwell is still recovered, but:
  * the coverage class is decided on the uninterrupted segment, so nothing is silently relabelled;
  * three features say what was done - sw_recovered (the walk crossed a break), sw_extra_s (how much dwell came
    back), sw_gap_s (how long the break was) - so the trees can separate a reconstructed dwell from an observed one
    instead of blending them.

Control is the production stage on today's features, exactly as E61's control was, so the two experiments are
directly comparable.

Gate: SSE over the rows the base model actually decides lower by >= 0.3 s LB-equivalent and lower in >= 4 of 6 folds,
and the same on the rows with y <= 7200. Strict, total and trimmed reported, not gated.
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

CTRL_DIR, CAND_DIR = "data/cv_preds_e62_ctrl", "data/cv_preds_e62"
raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
base = E36.oof_base()

built, feats = {}, {}
for name, folder, extra in (("ctrl", "data/adsblol", []), ("cand", "data/adsblol_e62", E36.SW)):
    E36.ADSB_DIR = folder
    d = E36.build(raw, base, E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).with_columns(
        month=pl.col("MVT_TIME_UTC_mvt").dt.month())
    for c in extra:
        d = d.with_columns(pl.col(c).fill_null(0.0))
    built[name], feats[name] = d, E36.FEATS + extra
    cov = d["coverage"]
    print(f"{name:5s} {folder:22s} rows {d.height}, coverage==3 {int((cov == 3).sum())}, "
          f"first movement known {int(d['mvt_first_move'].is_not_null().sum())}, features {len(feats[name])}", flush=True)

for x in (CTRL_DIR, CAND_DIR):
    os.makedirs(x, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    line = []
    for name, out in (("ctrl", CTRL_DIR), ("cand", CAND_DIR)):
        d = built[name]
        tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
        m = E36.fit_stage(tr, feats[name])
        pred, y = E36.correct(m, te), te[E36.Y].to_numpy()
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E62: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
