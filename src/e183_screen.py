"""E183 screen (04.10, REPORTED, not decisive): what the probe's correlations are worth in seconds.

Linear cross-fit by our month pairs (k, k+6): per airport, OLS of the (airport, UTC day)-centred residual r = y - s on the
centred neighbour quantities from data/e183_probe_rows.parquet, fitted on the other 10 months' rows without own track
(coverage < 2, 0 < y <= 7200), applied to the held-out months' rows without own track (LIRF rows without NM excluded: the
tail chain owns them); the applied correction is demeaned within (airport, UTC day) as in smart-jigsaw v30.
Delta = SSE change in the gate's LB-equivalent seconds (fold_metrics: / (6 * 2 * 252.43 * 344841)).
Arms: nbq only (ADS-B proxy), nbp only (NM proxy), nbq + nbp, oracle only (ceiling, uses labels).
Also: nbq availability on ranking rows without own track (>= 2 coverage-3 neighbours within +-30 min, same airport).

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e183_screen.py > data/e183_screen.log
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import e183_probe as P  # noqa: E402

R = P.R
LB = 6 * 2 * 252.43 * 344841
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
# E183_POP=no_adsb: only rows no ADS-B stage touches (coverage < 2 and not an E64 row). E183_SPLIT=halves: forward/backward
# in time (fit Jan-Jun -> apply Jul-Dec and the reverse) instead of the month pairs, to see how the signal ages.
POP = os.environ.get("E183_POP", "no_track")
SPLIT = os.environ.get("E183_SPLIT", "pairs")
W = (900, 1800, 3600)
ARMS = {
    "nbq": [f"nbq_{sc}_{w}" for sc in ("apt", "rwy") for w in W],
    "nbp": [f"nbp_{sc}_{w}" for sc in ("apt", "rwy") for w in W],
    "nbq+nbp": [f"{k}_{sc}_{w}" for k in ("nbq", "nbp") for sc in ("apt", "rwy") for w in W],
    "oracle": [f"orc_{sc}_{w}" for sc in ("apt", "rwy") for w in W],
}


def design(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Centred values (NaN -> 0 after centring over the defined rows of the airport-day) plus centred NaN indicators."""
    exprs = []
    for c in cols:
        v = pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c))
        exprs.append((v - v.mean().over("ADEP_mvt", "day")).fill_null(0.0).alias(c))
        ind = pl.col(c).is_nan().cast(pl.Float64)
        exprs.append((ind - ind.mean().over("ADEP_mvt", "day")).alias(f"{c}_na"))
    return df.select(exprs).to_numpy()


def main() -> None:
    d = pl.read_parquet(R + "data/e183_probe_rows.parquet").filter(pl.col("no_adsb") if POP == "no_adsb" else ~pl.col("own_track"))
    print(f"population {POP}, split {SPLIT}", flush=True)
    d = d.with_columns(r=pl.col("y") - pl.col("s"))
    app = d.filter(~((pl.col("ADEP_mvt") == "LIRF") & ~pl.col("has_nm")))
    print(f"rows without own track: {d.height:,}; applied population (minus LIRF without NM): {app.height:,}", flush=True)
    res = []
    for arm, cols in ARMS.items():
        folds = PAIRS if SPLIT == "pairs" else [(list(range(1, 7)), list(range(7, 13))), (list(range(7, 13)), list(range(1, 7)))]
        for f in folds:
            if SPLIT == "pairs":
                a, b = f
                tr = d.filter(~pl.col("month").is_in([a, b]) & pl.col("y").is_between(1, 7200))
                te = app.filter(pl.col("month").is_in([a, b]))
            else:
                a, b = f[1][0], f[1][-1]
                tr = d.filter(pl.col("month").is_in(f[0]) & pl.col("y").is_between(1, 7200))
                te = app.filter(pl.col("month").is_in(f[1]))
            for (ap,), g in tr.group_by(["ADEP_mvt"]):
                h = te.filter(pl.col("ADEP_mvt") == ap)
                if h.height == 0:
                    continue
                X = design(g, cols)
                rc = (g["r"] - g.select(pl.col("r").mean().over("ADEP_mvt", "day")).to_series()).to_numpy()
                keep = np.abs(X).sum(axis=0) > 0
                beta = np.linalg.lstsq(X[:, keep], rc, rcond=None)[0] if keep.any() else np.zeros(0)
                c = design(h, cols)[:, keep] @ beta if keep.any() else np.zeros(h.height)
                c = c - pl.DataFrame({"day": h["day"], "c": c}).select(pl.col("c").mean().over("day")).to_series().to_numpy()
                r = h["r"].to_numpy()
                dsse = (r - c) ** 2 - r ** 2
                small = h["y"].to_numpy() <= 7200
                res.append((arm, f"{a}_{b}", ap, h.height, float(dsse.sum()), float(dsse[small].sum()),
                            float(np.sqrt(np.mean(c ** 2)))))
    t = pl.DataFrame(res, schema=["arm", "fold", "airport", "rows", "dsse", "dsse_small", "rms_c"], orient="row")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=80, float_precision=3, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, tbl_cols=20, tbl_width_chars=300):
        print("\nper arm and fold, LB-equivalent seconds (all applied rows | y <= 7200):")
        print(t.group_by("arm", "fold").agg(s=pl.col("dsse").sum() / LB, s_small=pl.col("dsse_small").sum() / LB).sort("arm", "fold")
              .pivot(on="fold", index="arm", values="s"))
        print(t.group_by("arm").agg(total=pl.col("dsse").sum() / LB, y_le_7200=pl.col("dsse_small").sum() / LB).sort("arm"))
        lower = t.group_by("arm", "fold").agg(pl.col("dsse").sum()).group_by("arm").agg(lower=(pl.col("dsse") < 0).sum()).sort("arm")
        print(lower)
        print("\nper airport (6 folds), LB-equivalent seconds, and RMS of the applied correction:")
        print(t.group_by("arm", "airport").agg(s=pl.col("dsse").sum() / LB, rms_c=(pl.col("rms_c") ** 2 * pl.col("rows")).sum()
                                               / pl.col("rows").sum()).with_columns(pl.col("rms_c").sqrt())
              .pivot(on="arm", index="airport", values=["s", "rms_c"]).sort("airport"))

    if POP != "no_track" or SPLIT != "pairs":
        return
    # nbq availability on ranking rows without own track
    rk = P.load_raw([R + "data/prc-2026-datasets/ranking.parquet"]).join(P.load_adsb(2026), on="MVT_ID_mvt", how="left")
    rk = P.with_track(rk)
    rows = []
    for (ap,), g in rk.group_by(["ADEP_mvt"]):
        g = g.sort("t")
        t_ = g["t"].to_numpy()
        v = np.where((g["coverage"].to_numpy() == 3) & g["adsb_taxi"].is_between(30, 7200).fill_null(False).to_numpy(), 1.0, np.nan)
        m = P.win_mean(t_, v, 1800)
        nt = ~g["own_track"].to_numpy()
        rows.append((ap, int(nt.sum()), float(np.mean(~np.isnan(m[nt]))) if nt.any() else float("nan")))
    d25 = pl.read_parquet(R + "data/e183_probe_rows.parquet", columns=["ADEP_mvt", "own_track", "nbq_apt_1800"]).filter(
        ~pl.col("own_track")).group_by("ADEP_mvt").agg(nbq_avail25=pl.col("nbq_apt_1800").is_not_nan().mean())
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=20, float_precision=3, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        print("\nnbq (apt, +-30 min) defined on rows without own track: 2025 vs ranking")
        print(pl.DataFrame(rows, schema=["ADEP_mvt", "no_track_rows26", "nbq_avail26"], orient="row").join(d25, on="ADEP_mvt").sort("ADEP_mvt"))


if __name__ == "__main__":
    main()
