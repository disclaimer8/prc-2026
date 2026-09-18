"""Does the 'previous visit' of E64/E65 sometimes hold this departure's own early taxi? (18.09)

adsb_visit.py splits an airframe's points at a gap of VISIT_GAP=1800 s and calls everything before the last
gap the previous visit -- the turnaround. But a reception gap in the middle of a long taxi produces exactly the
same shape, so the early part of THIS departure's taxi is then read as a completed arrival.

Verified on EHAM MVT_ID=182026895 (2025-01-15, DAL75, stand E03, taxi-out 3762 s): the segment E65 calls the
previous visit starts 9 s BEFORE block time, 38 m from the assigned stand, and ends 4264 m away -- it is the
first half of this flight's own taxi. The real arrival is the segment before it, which runs the other way
(corr(t, distance to stand) -0.85, ending 63 m from the stand at zero ground speed).

The eight visit features are built from t, alt and gs only; lat/lon are never read, so nothing in them can tell
the two cases apart. This probe measures how large that population is and whether the direction carries signal
against the residual of the accepted E65 configuration.

    .venv/bin/python src/e69_visit_direction_probe.py [n_days]
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

from geometry import load_tables, norm_stand, _haversine

VISIT_GAP = 1800.0
LOOKBACK = 12 * 3600
N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def main() -> None:
    stands, _ = load_tables()
    skey = {(a, k): (la, lo) for a, k, la, lo in stands.iter_rows()}

    feat = pl.read_parquet("data/features_final.parquet",
                           columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "CALLSIGN_flt", "FLIGHT_mvt",
                                    "t_mvt", "split", "month"]).filter(pl.col("split") == "train")
    feat = feat.with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        day=pl.from_epoch(pl.col("t_mvt"), time_unit="s").dt.date(),
        skey=pl.col("STAND_mvt").map_elements(norm_stand, return_dtype=pl.String))

    days = sorted({p.split("surface_")[1][:-8] for p in glob.glob("data/adsblol/surface_2025-*.parquet")})
    days = days[:: max(1, len(days) // N_DAYS)][:N_DAYS]
    print(f"probing {len(days)} days of 2025: {days[0]} .. {days[-1]}", flush=True)

    rows = []
    for day in days:
        pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet",
                              columns=["icao", "airport", "t", "lat", "lon", "gs", "callsign"]).sort("icao", "t")
        cmap: dict[tuple, list] = {}
        for cs, ap, ic in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
            cmap.setdefault((cs, ap), []).append(ic)
        by = {(k[0], k[1]): g.sort("t") for k, g in pts.group_by(["icao", "airport"])}
        dep = feat.filter(pl.col("day") == pl.lit(day).str.to_date())
        for mvt_id, ap, cs, to, sk in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "t_mvt", "skey").iter_rows():
            ic = cmap.get((cs, ap), [])
            if len(ic) != 1 or (ic[0], ap) not in by or (ap, sk) not in skey:
                continue
            g = by[(ic[0], ap)]
            t = g["t"].to_numpy()
            sel = (t >= to - LOOKBACK) & (t <= to + 600)
            if sel.sum() < 2:
                continue
            tt, la, lo = t[sel], g["lat"].to_numpy()[sel], g["lon"].to_numpy()[sel]
            big = np.where(np.diff(tt) > VISIT_GAP)[0]
            if not len(big):
                continue                                   # no previous visit -> nothing to misread
            k = int(big[-1]) + 1
            p0 = 0 if len(big) == 1 else int(big[-2]) + 1   # first index of the previous visit
            p1 = int(big[-1])                               # last index of the previous visit
            if p1 - p0 < 2:
                continue
            slat, slon = skey[(ap, sk)]
            d = _haversine(la, lo, slat, slon)              # metres to the assigned stand
            pv = slice(p0, p1 + 1)
            rows.append((mvt_id, float(d[p0]), float(d[p1]), float(d[pv].min()),
                         float(np.corrcoef(tt[pv], d[pv])[0, 1]) if np.ptp(d[pv]) > 0 else 0.0,
                         float(d[k]), float(to - tt[k]), float(to - tt[p1]), int(p1 - p0 + 1)))
        print(f"  {day}: running total {len(rows)}", flush=True)

    r = pl.DataFrame(rows, orient="row", schema={
        "MVT_ID_mvt": pl.Float64, "pv_d_start": pl.Float64, "pv_d_end": pl.Float64, "pv_d_min": pl.Float64,
        "pv_corr": pl.Float64, "dv_d_start": pl.Float64, "vi_first_seen": pl.Float64,
        "vi_turnaround": pl.Float64, "pv_pts": pl.Int64})
    r.write_parquet("data/e69_visit_direction.parquet")

    oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": np.load(p)["mvt_id"], "pred": np.load(p)["pred"], "y": np.load(p)["y"]})
                     for p in sorted(glob.glob("data/cv_preds_e65/fold_*.npz"))])
    j = r.join(oof, on="MVT_ID_mvt", how="inner").with_columns(
        resid=pl.col("y") - pl.col("pred"),
        away=(pl.col("pv_d_end") - pl.col("pv_d_start")))
    print(f"\nrows with a previous visit and a known stand: {j.height}")

    mid = j.filter((pl.col("pv_corr") > 0.5) & (pl.col("pv_d_end") > 500) & (pl.col("pv_d_min") < 200))
    arr = j.filter(pl.col("pv_corr") < -0.5)
    other = j.filter(~((pl.col("pv_corr") > 0.5) & (pl.col("pv_d_end") > 500) & (pl.col("pv_d_min") < 200))
                     & (pl.col("pv_corr") >= -0.5))
    for name, s in (("previous visit runs AWAY from the stand (mid-taxi gap)", mid),
                    ("previous visit runs TOWARD the stand (a real arrival)", arr),
                    ("neither", other)):
        if s.height == 0:
            continue
        res = s["resid"].to_numpy()
        print(f"\n{name}: n={s.height} ({100*s.height/j.height:.1f}%)")
        print(f"   mean residual {res.mean():+8.1f} s | mean |residual| {np.abs(res).mean():7.1f} s | "
              f"RMSE {np.sqrt((res**2).mean()):7.1f} s | median y {s['y'].median():.0f}")

    res = j["resid"].to_numpy()
    print(f"\nall rows: mean residual {res.mean():+.1f} s | mean |residual| {np.abs(res).mean():.1f} s | RMSE {np.sqrt((res**2).mean()):.1f}")
    for c in ("pv_corr", "away", "pv_d_end", "pv_d_min", "dv_d_start"):
        v = j[c].to_numpy().astype(float)
        m = np.isfinite(v) & np.isfinite(res)
        print(f"   corr(resid, {c:>12}) = {np.corrcoef(v[m], res[m])[0,1]:+.4f}")


if __name__ == "__main__":
    main()
