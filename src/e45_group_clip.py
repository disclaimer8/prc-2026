"""E45: clip predictions to the observed range of a narrow homogeneous group (16.09).

    .venv/bin/python src/e45_group_clip.py
Source: gentle_wreath (2024 edition, ranks 8-15) — the only quantified clipping protocol across editions: per-aircraft-type
min/max of the target improved RMSE, a +-1% margin improved it further, +-2% was worse, and clipping by a coarse group
(airline) was catastrophic. Here the group is airport x runway x stand (falling back to airport x runway, then airport), the
bounds come from the fold's training months only, and the margin is a fraction of the bound.
Evaluated out-of-fold on data/cv_preds_nested (leak-free rounds). Gate: total SSE lower in >= 5 of 6 folds and by >= 0.3 s
LB-equivalent for the chosen margin, with the margin fixed before looking at the folds (0, 1%, 2%).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
LB = 6 * 2 * 252.43 * 344841
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", Y]
raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())

def bounds(df, keys, min_n):
    return df.group_by(keys).agg(lo=pl.col(Y).min(), hi=pl.col(Y).max(), n=pl.len()).filter(pl.col("n") >= min_n).drop("n")

for margin in (0.0, 0.01, 0.02):
    tot, lower, touched = 0.0, 0, 0
    for p in sorted(glob.glob("data/cv_preds_nested/fold_*.npz")):
        a, b = [int(x) for x in os.path.basename(p)[5:-4].split("_")]
        z = np.load(p)
        te = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "y": z["y"]}).join(raw, on="MVT_ID_mvt", how="left")
        tr = raw.filter(~pl.col("month").is_in([a, b]))
        j = te.join(bounds(tr, ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"], 30), on=["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"], how="left")
        j = j.join(bounds(tr, ["ADEP_mvt", "RUNWAY_mvt"], 30).rename({"lo": "lo2", "hi": "hi2"}), on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        lo = pl.coalesce(pl.col("lo"), pl.col("lo2")).to_frame().to_series().to_numpy() if False else j.select(pl.coalesce("lo", "lo2"))["lo"].to_numpy()
        hi = j.select(pl.coalesce("hi", "hi2"))["hi"].to_numpy()
        pred, y = j["pred"].to_numpy(), j["y"].to_numpy()
        lo = np.where(np.isnan(lo), -np.inf, lo * (1 - margin))
        hi = np.where(np.isnan(hi), np.inf, hi * (1 + margin))
        new = np.clip(pred, lo, hi)
        d = ((new - y) ** 2 - (pred - y) ** 2).sum()
        tot += d
        lower += d < 0
        touched += int((new != pred).sum())
    print(f"margin {margin:.0%}: SSE {tot:+.3e} = {tot / LB:+.2f} s LB-equivalent, lower in {lower}/6, rows touched {touched}")
