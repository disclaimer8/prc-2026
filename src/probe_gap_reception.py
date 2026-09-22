"""Probe (22.09, external review point 2): reception AROUND THE START OF THE OWN TRACK, not airport-wide.

E88 closed three airport-level aggregates over [take-off - 30 min, take-off]. The untested version: when a
track starts late, were OTHER aircraft seen near the spot where ours appeared, just before it appeared?
Seen neighbours -> receiver alive there, the late start is the aircraft's (transponder off). No neighbours ->
reception hole, the true off-block may lie well before the first point.

For sampled 2025 days: match rows to airframes exactly as adsb_offblock_v5, take the own first surface point
at t_first (data/adsblol_kt1), count other airframes' surface points within NEAR_M in [t_first - WIN, t_first).
Output: data/probe_gap_reception.parquet; the summary joins the stage OOF (data/cv_preds_e77_stage).
    .venv/bin/python src/probe_gap_reception.py
"""
from __future__ import annotations

import glob
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import surface_mask  # noqa: E402

NEAR_M, WIN = 1500.0, 600.0
DAYS = [f"2025-{m:02d}-{d:02d}" for m in range(1, 13) for d in (5, 15)]


def one_day(day: str) -> pl.DataFrame | None:
    p = f"data/adsblol/surface_{day}.parquet"
    k = f"data/adsblol_kt1/offblock2_{day}.parquet"
    if not (os.path.exists(p) and os.path.exists(k)):
        return None
    pts = pl.read_parquet(p)
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/training_*.parquet")
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    kt = pl.read_parquet(k).select("MVT_ID_mvt", "coverage", "t_first", "t_to")
    dep = dep.join(kt, on="MVT_ID_mvt", how="inner").filter(pl.col("t_first").is_not_null())
    cand: dict = {}
    for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cand.setdefault((cs, ap), []).append(icao)
    surf = pts.filter(pl.col("surf"))
    surf_t = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in surf.group_by("icao", "airport").agg(pl.col("t")).iter_rows()}
    by_ap = {a[0]: g for a, g in surf.group_by(["airport"])}
    rows = []
    for mvt, ap, cs, to, tf, cov in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to", "t_first", "coverage").iter_rows():
        ok = [i for i in cand.get((cs, ap), []) if (ts := surf_t.get((i, ap))) is not None
              and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
        if len(ok) != 1 or ap not in by_ap:
            continue
        g = by_ap[ap]
        own = g.filter((pl.col("icao") == ok[0]) & (pl.col("t") >= tf - 1)).head(1)
        if own.height == 0:
            continue
        la, lo = own["lat"][0], own["lon"][0]
        w = g.filter((pl.col("icao") != ok[0]) & (pl.col("t") >= tf - WIN) & (pl.col("t") < tf))
        dx = (w["lon"].to_numpy() - lo) * 111320 * np.cos(np.radians(la))
        dy = (w["lat"].to_numpy() - la) * 110540
        near = np.hypot(dx, dy) <= NEAR_M
        rows.append((mvt, ap, cov, tf, int(near.sum()), int(w.filter(pl.Series(near))["icao"].n_unique()) if near.any() else 0,
                     int(w.height), int(w["icao"].n_unique())))
    if not rows:
        return None
    return pl.DataFrame(rows, schema={"MVT_ID_mvt": pl.Float64, "ap": pl.String, "coverage": pl.Int64, "t_first": pl.Float64,
                                      "nb_pts": pl.Int64, "nb_ac": pl.Int64, "ap_pts": pl.Int64, "ap_ac": pl.Int64}, orient="row")


def main() -> None:
    with Pool(10) as pool:
        parts = [x for x in pool.map(one_day, DAYS) if x is not None]
    df = pl.concat(parts)
    df.write_parquet("data/probe_gap_reception.parquet")
    print(f"matched rows: {df.height:,} over {len(parts)} days")


if __name__ == "__main__":
    main()
