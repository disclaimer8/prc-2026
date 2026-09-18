"""E71: does lowering the movement threshold from 3 kt to 1 kt pay at the stage? (18.09)

Measured on the raw detector first, against known block times on 18 050 departures over 6 days:

    rule                                coverage   median   |error|   within 60 s
    gs > 3, no filter, no sustain         99.7%     +300.2    302.6      22.9%
    gs > 1, no filter, no sustain         99.9%     +186.7    193.5      34.3%
    median3 + gs > 2 + sustain 30 s       97.3%     +254.0    257.2      27.4%

Below 3 kt sit 13.2% of ground samples: 9.0% parked under 0.5 kt and 4.2% in the 0.5-3 kt band where pushback
happens. Our rule discards that band. The median-filter-with-sustain recipe that traffic and OPDI converge on
loses at every threshold here, because it treats noise our data does not have -- gs is broadcast on 98.02% of
ground samples at 0.1 kt resolution.

The reference is not the problem: only 6.5% of our block times land on a whole minute against 1.67% for a
uniform distribution, and at LSZH, EDDF and EHAM it is 1.7-2.2%, i.e. second resolution. So the 193.5 s is
detector error, not quantisation.

Control is the accepted E65 configuration on data/adsblol; the candidate is the same configuration on
data/adsblol_kt1. Only the extraction differs, so the arms isolate the threshold. The stand-walk columns are
deliberately not involved: they are extracted against the old threshold and mixing the two would confound.

Predicted before the run: -0.2 to -0.8 s. The onset feeds mvt_first_move, aobt_first_move and base_vs_first_move,
which sit near the top of the stage's feature list, and a 109 s cut in their median error is large against the
stage's own residual -- but most of that error is a late bias the stage can already absorb per airport, and
what remains is the row-to-row part.

Gate: decided metric at least 0.3 s lower, lower in at least 4 of 6 folds, the same on y <= 7200, and the
gain must survive dropping each fold's three best and three worst rows.

    .venv/bin/python src/e71_onset_threshold.py
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

CTRL_DIR, CAND_DIR = "data/cv_preds_e71_ctrl", "data/cv_preds_e71"
raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])


def frame(adsb_dir: str) -> pl.DataFrame:
    E36.ADSB_DIR = adsb_dir
    d = E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet"))
    return d.join(vis, on="MVT_ID_mvt", how="left").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())


arms = {"ctrl": (frame("data/adsblol"), CTRL_DIR), "cand": (frame("data/adsblol_kt1"), CAND_DIR)}
for name, (d, out) in arms.items():
    os.makedirs(out, exist_ok=True)
    print(f"{name}: rows {d.height}, coverage>=2 {int((d['coverage'] >= 2).sum())}, "
          f"visit known {int(d['vi_first_seen'].is_not_null().sum())}", flush=True)

for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    line = []
    for name, (d, out) in arms.items():
        tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
        pred = E36.correct(E36.fit_stage(tr, E36.FEATS + NAMES), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=te[E36.Y].to_numpy())
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - te[E36.Y].to_numpy()) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E71: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
