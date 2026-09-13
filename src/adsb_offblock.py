"""Derive off-block / taxi-start estimates for departures from surface ADS-B points.

For each departure (callsign, airport, take-off time) take the ADS-B track of that callsign
at that airport and the continuous surface segment that ends at take-off. Estimates:
  t_first   first surface point of the segment (transponder switched on / first seen)
  t_move    start of the final movement after the last stop of >= STOP_S seconds
  t_adsb_to last on-ground point before becoming airborne
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

GAP_S = 900  # a gap longer than this splits surface segments
STOP_S = 90  # stationary this long counts as parked / holding at stand
MOVING_KT = 3.0


def estimates_for_track(t: np.ndarray, gs: np.ndarray, ground: np.ndarray, takeoff: float):
    """t sorted. Returns (t_first, t_move, t_adsb_to) or Nones."""
    # airborne transition closest to reported take-off
    idx_air = np.where(~ground & (t >= takeoff - 600) & (t <= takeoff + 600))[0]
    idx_gnd = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx_gnd) == 0:
        return None, None, None
    if len(idx_air):
        last_gnd = idx_gnd[idx_gnd < idx_air[0]]
        if len(last_gnd) == 0:
            return None, None, None
        end = last_gnd[-1]
    else:
        end = idx_gnd[-1]
        if takeoff - t[end] > 600:
            return None, None, None
    start = end
    while start > 0 and ground[start - 1] and t[start] - t[start - 1] <= GAP_S:
        start -= 1
    seg_t, seg_gs = t[start : end + 1], np.nan_to_num(gs[start : end + 1])
    t_move = seg_t[0]
    # walk backwards from take-off to find the last stationary stretch
    moving_since = None
    i = len(seg_t) - 1
    while i >= 0:
        if seg_gs[i] <= MOVING_KT:
            j = i
            while j > 0 and seg_gs[j - 1] <= MOVING_KT:
                j -= 1
            if seg_t[i] - seg_t[j] >= STOP_S:
                t_move = seg_t[i]
                moving_since = True
                break
            i = j - 1
        else:
            i -= 1
    if moving_since is None:
        t_move = seg_t[0]
    return float(seg_t[0]), float(t_move), float(t[end])


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet").drop_nulls("callsign").sort("callsign", "airport", "t")
    y, m, d = map(int, day.split("-"))
    D = "data/prc-2026-datasets/"
    src = pl.read_parquet(D + "training_*.parquet") if y == 2025 else pl.read_parquet(D + "ranking.parquet")
    dep = src.filter(
        (pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))
    ).with_columns(cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""))

    groups = {k: g for k, g in pts.group_by(["callsign", "airport"])}
    out = []
    for r in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("to")).iter_rows():
        mvt_id, ap, cs, to = r
        g = groups.get((cs, ap))
        if g is None:
            out.append((mvt_id, None, None, None, 0))
            continue
        e = estimates_for_track(g["t"].to_numpy(), g["gs"].to_numpy(), g["on_ground"].to_numpy(), float(to))
        out.append((mvt_id, *e, g.height))
    res = pl.DataFrame(out, schema=["MVT_ID_mvt", "t_first", "t_move", "t_adsb_to", "n_pts"], orient="row")
    res.write_parquet(f"data/adsblol/offblock_{day}.parquet")

    if y == 2025:
        j = dep.join(res, on="MVT_ID_mvt").with_columns(
            blk=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
            to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        )
        print("departures", j.height, "matched track", j.filter(pl.col("t_first").is_not_null()).height)
        stats = j.filter(pl.col("t_first").is_not_null()).group_by("ADEP_mvt").agg(
            n=pl.len(),
            cov=pl.len(),
            first_err_med=(pl.col("t_first") - pl.col("blk")).median(),
            first_abs_lt120=((pl.col("t_first") - pl.col("blk")).abs() < 120).mean().round(3),
            move_err_med=(pl.col("t_move") - pl.col("blk")).median(),
            move_abs_lt120=((pl.col("t_move") - pl.col("blk")).abs() < 120).mean().round(3),
            to_err_med=(pl.col("t_adsb_to") - pl.col("to")).median(),
            corr_first=pl.corr(pl.col("to") - pl.col("t_first"), pl.col("TAXITIME_SEC_mvt")).round(3),
            corr_move=pl.corr(pl.col("to") - pl.col("t_move"), pl.col("TAXITIME_SEC_mvt")).round(3),
        ).sort("ADEP_mvt")
        tot = dep.group_by("ADEP_mvt").len().rename({"len": "deps"})
        pl.Config.set_tbl_width_chars(250)
        pl.Config.set_tbl_cols(20)
        print(stats.join(tot, on="ADEP_mvt").with_columns(coverage=(pl.col("n") / pl.col("deps")).round(3)).drop("cov"))


if __name__ == "__main__":
    main()
