"""E70: the whole sub-threshold family at once — E67 (E63 + E66) and E69 together (18.09).

Four results now sit under the gate and every one of them was lower in all six folds: E63 -0.05, E66 -0.15,
E67 -0.17 (the two of them together) and E69 -0.26 (the reception gap folded back into the taxi it split).
E67 answered whether its two halves add up: they do not, -0.05 and -0.15 became -0.17, about 85% of the
naive sum, because the populations overlap.

E69 is a different mechanism — it repairs the value of vi_first_seen rather than adding a view of the
ground segment — so the overlap with E67 should be smaller. Control is the accepted E65 configuration, and
the candidate adds both sets beside it; nothing is substituted.

Predicted before the run: naive sum -0.43, and at E67's own 85% rate about **-0.37 s**, which would just
clear the gate. Written down because today already produced one estimate that was four times too high
(the AOBT_3 abstinence, +25 predicted against +5.99 measured).

Gate: decided metric at least 0.3 s lower, lower in at least 4 of 6 folds, the same on y <= 7200, and the
gain must survive dropping each fold's three best and three worst rows.
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

CTRL_DIR, CAND_DIR = "data/cv_preds_e70_ctrl", "data/cv_preds_e70"
SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])
E36.ADSB_DIR = "data/adsblol"
d = E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).join(
    vis, on="MVT_ID_mvt", how="left").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())

walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                    for f in sorted(glob.glob("data/adsblol_standwalk/offblock2_2025-*.parquet"))]).rename(
    {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
d = d.join(walked, on="MVT_ID_mvt", how="left").with_columns(
    sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
    sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
    sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))

vim = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_m_2025-*.parquet"))])
d = d.join(vim, on="MVT_ID_mvt", how="left")

missing = [c for c in NAMES + NEW + NAMES_M if c not in d.columns]
if missing:
    sys.exit(f"нет колонок: {missing} — сначала переизвлечение с признаками визита и геометрии")
print(f"строк {d.height}; продлён отрезок у {int(d['sw_recovered'].sum())}, "
      f"визит известен у {int(d[NAMES[0]].is_not_null().sum())}, геометрия E66 у {int(d[NEW[0]].is_not_null().sum())}", flush=True)

for p in (CTRL_DIR, CAND_DIR):
    os.makedirs(p, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    y = te[E36.Y].to_numpy()
    line = []
    for name, out, feats in (("ctrl", CTRL_DIR, E36.FEATS + NAMES), ("cand", CAND_DIR, E36.FEATS + NAMES + SW + NEW + NAMES_M)):
        pred = E36.correct(E36.fit_stage(tr, feats), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E70: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
