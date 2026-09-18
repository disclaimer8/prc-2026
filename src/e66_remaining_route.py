"""E66: where on the taxi we caught the aircraft, and how much of it is left (18.09).

Population: coverage == 2, the departures picked up mid-taxi - 792 642 decided rows holding 29.94 s AFTER E65,
the second largest pocket left and the largest a stage can reach. None of the geometry the stage already has
(gs_*, sf_*) correlates with what remains even at 0.02, so the stage has exhausted it; the question is whether a
new quantity helps.

Not E51. That measured the stand-to-runway distance along the OSM graph - a property of the PAIR, absorbed by the
categorical ap_stand / ap_runway, +0.01 s. These three are properties of THIS track:
  gs_d_first_rwy   metres from the first observed point to the departure runway - how much is left
  gs_frac_done     d_first_stand / (d_first_stand + d_first_rwy) - how much of the taxi is already behind
  gs_d_stand_rwy   the straight stand-to-runway distance, the denominator

Measured on the control day before registering: the new geometry exists for 3139 of 6120 departures, median
frac_done 0.158 with p10 0.011 and p90 0.689, median distance left 1448 m - so the quantity varies rather than
being a constant per stand.

Pure addition: control is the accepted E65 configuration, candidate adds three columns and nothing else. The
existing offblock5 columns reproduce identically.

Gate: decided metric at least 0.3 s lower, lower in at least 4 of 6 folds, the same on y <= 7200, and the gain
must survive dropping each fold's three best and three worst rows.
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
from adsb_visit import NAMES  # noqa: E402
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e66_ctrl", "data/cv_preds_e66"
raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])
d = E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).join(
    vis, on="MVT_ID_mvt", how="left").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
print(f"rows {d.height}, with visit features {int(d['vi_first_seen'].is_not_null().sum())}, "
      f"with a previous visit {int((d['vi_has_prev'] == 1).sum())}", flush=True)

for x in (CTRL_DIR, CAND_DIR):
    os.makedirs(x, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    y = te[E36.Y].to_numpy()
    line = []
    for name, feats, out in (("ctrl", E36.FEATS + NAMES, CTRL_DIR), ("cand", E36.FEATS + NAMES + NEW, CAND_DIR)):
        pred = E36.correct(E36.fit_stage(tr, feats), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E66: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
