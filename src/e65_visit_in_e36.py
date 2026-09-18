"""E65: give the visit features to the E36 stage as well (18.09).

E64 put the airframe's visit features on the rows with no ground segment and was accepted (-0.64 s CV,
251.42 -> 251.249 on the leaderboard). The footprint measured afterwards says the larger half of the idea was
left on the table: on coverage >= 2 with a previous visit there are 1 026 649 decided rows holding 41.49 s -
against 318 691 rows and 18.93 s for the population E64 corrects - and vi_first_seen correlates +0.151 with the
residual there, higher than the +0.110 that carried E64.

So: the same features, added to the stage that already works on those rows. Pure addition, nothing substituted -
control and candidate differ only by the eight visit columns in the feature list, same base, same folds, same
clips, same rounds. Rows without visit features keep the control's behaviour because the columns are null.

Ceiling before the run: r = 0.151 bounds the linear part at about 2.3% of 41.49 s, so roughly 0.94 s. E64 beat
its own linear ceiling twofold, so this may land anywhere between 0.3 and 1.5 s - or, if the ground track already
carries what the visit says, at nothing.

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
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e65_ctrl", "data/cv_preds_e65"
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
    for name, feats, out in (("ctrl", E36.FEATS, CTRL_DIR), ("cand", E36.FEATS + NAMES, CAND_DIR)):
        pred = E36.correct(E36.fit_stage(tr, feats), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E65: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
