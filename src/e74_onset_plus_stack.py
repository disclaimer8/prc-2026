"""E74: the accepted E70 stack on the 1 kt extraction — do the two accepted changes add? (19.09)

E70 (-0.38 s CV, -0.44 s on the board: 248.7737 -> 248.3325) added the sub-threshold family to the stage.
E71 (-0.36 s CV) lowered the movement threshold from 3 kt to 1 kt, which cut the detector's median absolute
error against known block times from 302.6 s to 193.5 s. Both act on the same stage, one by adding columns
and the other by improving the values those columns are computed from, so they may well overlap.

Every source E70 reads has to move together or the arms are confounded: offblock2/3/5, the stand-walk variant
(sw_extra_s is a difference BETWEEN the two extractions and would otherwise straddle thresholds) and the visit
files, which do not depend on the threshold.

Predicted before the run, from the two ratios we have measured for pairs: E67 + E69 reached 88% of their naive
sum and E63 + E66 reached 85%. Naive sum here is -0.74; at the same rate about **-0.63 s**, and I would not be
surprised by less, because E71 improves the inputs of the very columns E70 adds, which is a tighter overlap
than the two pairs above.

Gate: decided metric at least 0.3 s lower than the accepted E70, lower in at least 4 of 6 folds, the same on
y <= 7200, and the gain must survive dropping each fold's three best and three worst rows.

    .venv/bin/python src/e74_onset_plus_stack.py
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

CTRL_DIR, CAND_DIR = "data/cv_preds_e74_ctrl", "data/cv_preds_e74"
SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]
FEATS = E36.FEATS + NAMES + SW + NEW + NAMES_M

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
VISIT_DIR = os.environ.get("VISIT_DIR", "data/adsblol")  # E75: visits re-extracted across midnight live elsewhere
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{VISIT_DIR}/visit_2025-*.parquet"))])
vim = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{VISIT_DIR}/visit_m_2025-*.parquet"))])


def frame(adsb_dir: str, walk_dir: str) -> pl.DataFrame:
    E36.ADSB_DIR = adsb_dir
    d = E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet"))
    d = d.join(vis, on="MVT_ID_mvt", how="left").join(vim, on="MVT_ID_mvt", how="left")
    walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                        for f in sorted(glob.glob(f"{walk_dir}/offblock2_2025-*.parquet"))]).rename(
        {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
    d = d.join(walked, on="MVT_ID_mvt", how="left").with_columns(
        sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
        sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
        sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
    ).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8),
                   month=pl.col("MVT_TIME_UTC_mvt").dt.month())
    missing = [c for c in FEATS if c not in d.columns]
    if missing:
        sys.exit(f"{adsb_dir}: missing {missing}")
    return d


arms = {"ctrl": (frame("data/adsblol", "data/adsblol_standwalk"), CTRL_DIR),
        "cand": (frame("data/adsblol_kt1", "data/adsblol_sw_kt1"), CAND_DIR)}
for name, (d, out) in arms.items():
    os.makedirs(out, exist_ok=True)
    print(f"{name}: rows {d.height}, coverage>=2 {int((d['coverage'] >= 2).sum())}, "
          f"merged {int((d['vm_merged'] > 0).sum())}, walk extended {int(d['sw_recovered'].sum())}", flush=True)

for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    line = []
    for name, (d, out) in arms.items():
        tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
        pred = E36.correct(E36.fit_stage(tr, FEATS), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=te[E36.Y].to_numpy())
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - te[E36.Y].to_numpy()) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR, lb_rmse=248.3325)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E74: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
