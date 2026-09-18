"""E69: does repairing the visit split pay, on the stage E65 was accepted on? (18.09, proposal by Denys)

Control is the accepted configuration exactly: the E36 stand-geometry stage plus the eight visit columns of
E64/E65. The candidate adds the eleven columns of src/e69_visit_merge.py beside them -- previous-visit geometry
against the assigned stand, the merge flag, and the visit features recomputed once a reception gap has been
folded back into the taxi it split. Nothing is substituted; the old columns stay in both arms, so the two differ
only by the addition (REVIEW_BRIEF.md: change the input by adding, not by substituting).

Why it might pay, stated before the run: the fix is not a flag on bad rows, it is a corrected value for the
feature the stage leans on hardest. E64's own probe put vi_first_seen at +0.110 against the residual and E65's
at +0.151, higher than anything else measured; on the reference flight it reads 147 s where 3771 s elapsed.

Why it might not: the population is 0.54% of rows with a previous visit, and it is nearly unbiased (+15.3 s at
RMSE 313.7), so there is variance to predict rather than a shift to remove.

Priced before the run (src/e69_visit_direction_probe.py, 18 days of 2025): ceiling 0.40 s if those rows were
predicted perfectly, about 0.17 s realistically. Expectation recorded in the brief: -0.05..-0.20 s, most likely
short of the gate.

Gate: decided metric at least 0.3 s lower, lower in at least 4 of 6 folds, the same on y <= 7200, and the gain
must survive dropping each fold's three best and three worst rows.

    .venv/bin/python src/e69_merge_stage.py
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
from e69_visit_merge import NAMES_M  # noqa: E402
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e69_ctrl", "data/cv_preds_e69"

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])
vim = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_m_2025-*.parquet"))])
d = (E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet"))
     .join(vis, on="MVT_ID_mvt", how="left")
     .join(vim, on="MVT_ID_mvt", how="left")
     .with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month()))
print(f"rows {d.height}, with visit features {int(d['vi_first_seen'].is_not_null().sum())}, "
      f"with a previous visit {int((d['vi_has_prev'] == 1).sum())}, "
      f"merged {int((d['vm_merged'] > 0).sum())}", flush=True)

for x in (CTRL_DIR, CAND_DIR):
    os.makedirs(x, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    y = te[E36.Y].to_numpy()
    line = []
    for name, feats, out in (("ctrl", E36.FEATS + NAMES, CTRL_DIR),
                             ("cand", E36.FEATS + NAMES + NAMES_M, CAND_DIR)):
        pred = E36.correct(E36.fit_stage(tr, feats), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E69: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
