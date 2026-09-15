"""E25: off-block / taxi-start estimates from adsb.lol surface points, version 2 (15.09).

Fixes of version 1 (13.09): surface = ground flag OR (ground speed <= 40 kt and altitude <= aerodrome elevation + 800 ft)
— at EDDM only 0.02% of near-aerodrome points carry the ground flag; the track is taken by aircraft (icao24), not by the
last broadcast call sign, so stand points before the first call-sign message are kept.
Matching: an icao24 whose points carry the departure's call sign at that airport and whose last surface point lies within
[MVT - 900 s, MVT + 300 s]; exactly one candidate is required (the inbound flight with the same call sign cannot qualify).
Per departure: coverage 0 no track, 1 track without surface segment before take-off, 2 surface segment starts while
moving (picked up mid-taxi), 3 aircraft seen stationary >= STOP_S before its final movement; t_first, t_move, t_to, n_pts.
    .venv/bin/python src/adsb_offblock_v2.py 2025-07-15        writes data/adsblol/offblock2_<day>.parquet
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12, "LEMD": 1998, "LFPG": 392, "LIRF": 13, "LSZH": 1416, "LTFM": 325}
GAP_S, STOP_S, MOVING_KT, SURFACE_KT = 900, 90, 3.0, 40.0


def surface_mask(df: pl.DataFrame) -> pl.Expr:
    elev = pl.col("airport").replace_strict(ELEV_FT, return_dtype=pl.Float64)
    return pl.col("on_ground") | ((pl.col("gs").fill_null(0) <= SURFACE_KT) & (pl.col("alt").is_null() | (pl.col("alt") <= elev + 800)))


def estimate(t: np.ndarray, gs: np.ndarray, ground: np.ndarray, takeoff: float):
    idx = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx) == 0:
        return 1, None, None, None, None, None, None
    end = idx[-1]
    if takeoff - t[end] > 900:
        return 1, None, None, None, None, None, None
    start = end
    while start > 0 and ground[start - 1] and t[start] - t[start - 1] <= GAP_S:
        start -= 1
    st, sg = t[start:end + 1], np.nan_to_num(gs[start:end + 1])
    stat = sg <= MOVING_KT
    # first movement after an initial stationary stretch (push-back / taxi start from the stand)
    t_first_move, stat_total, n_stops = None, 0.0, 0
    if stat[0]:
        k = 0
        while k < len(st) and stat[k]:
            k += 1
        if k < len(st):
            t_first_move = float(st[k])
    dt = np.diff(st, prepend=st[0])
    stat_total = float(dt[stat].sum())
    n_stops = int(np.sum(stat[1:] & ~stat[:-1]))
    extra = (t_first_move, stat_total, n_stops)
    cov, t_move = 2, st[0]
    i = len(st) - 1
    while i >= 0:  # last stationary stretch of >= STOP_S before take-off
        if sg[i] <= MOVING_KT:
            j = i
            while j > 0 and sg[j - 1] <= MOVING_KT:
                j -= 1
            if st[i] - st[j] >= STOP_S:
                cov, t_move = 3, st[i]
                break
            i = j - 1
        else:
            i -= 1
    return (cov, float(st[0]), float(t_move), float(t[end])) + extra


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    D = "data/prc-2026-datasets/"
    src = pl.read_parquet(D + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""), to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    # candidate aircraft per (call sign, airport): last surface time of that icao at that airport
    cands = pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique()
    surf_times = pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t"))
    by_icao = {k[0]: g for k, g in pts.group_by(["icao"])}
    cand_map: dict[tuple, list] = {}
    for cs, ap, icao in cands.iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    st_map = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in surf_times.iter_rows()}
    out = []
    for mvt_id, ap, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ok = []
        for icao in cand_map.get((cs, ap), []):
            ts = st_map.get((icao, ap))
            if ts is None:
                continue
            before = ts[ts <= to + 300]
            if len(before) and to - before[-1] <= 900:
                ok.append(icao)
        if len(ok) != 1:
            out.append((mvt_id, 0 if not cand_map.get((cs, ap)) else 1, None, None, None, None, None, None, 0, len(ok)))
            continue
        g = by_icao[ok[0]].filter(pl.col("airport") == ap)
        e = estimate(g["t"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), float(to))
        out.append((mvt_id, *e, g.height, 1))
    res = pl.DataFrame(out, schema=["MVT_ID_mvt", "coverage", "t_first", "t_move", "t_to", "t_first_move", "stat_total", "n_stops", "n_pts", "n_cand"], orient="row")
    res.write_parquet(f"data/adsblol/offblock2_{day}.parquet")
    if y == 2025:
        j = dep.join(res, on="MVT_ID_mvt").with_columns(blk=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
        pl.Config.set_tbl_width_chars(250)
        pl.Config.set_tbl_cols(20)
        print(j.group_by("ADEP_mvt").agg(
            deps=pl.len(), cov3=(pl.col("coverage") == 3).mean().round(3), cov2=(pl.col("coverage") == 2).mean().round(3),
            move_err_med=(pl.col("t_move") - pl.col("blk")).median(),
            move_in120=((pl.col("t_move") - pl.col("blk")).abs() <= 120).sum() / pl.col("t_move").is_not_null().sum(),
            move_in120_cov3=(((pl.col("t_move") - pl.col("blk")).abs() <= 120) & (pl.col("coverage") == 3)).sum() / (pl.col("coverage") == 3).sum(),
            corr_move=pl.corr(pl.col("to") - pl.col("t_move"), pl.col("TAXITIME_SEC_mvt")).round(3),
        ).sort("ADEP_mvt"))


if __name__ == "__main__":
    main()
