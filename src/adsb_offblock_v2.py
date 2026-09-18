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
GAP_S, STOP_S, SURFACE_KT = 900, 90, 40.0
# E71 (18.09): the movement threshold, 3 kt by default so the accepted path is unchanged. Measured
# against known block times on 18 050 departures: 3 kt gives 302.6 s median absolute error, 1 kt gives
# 193.5 s at higher coverage, because 4.2% of ground samples sit in the 0.5-3 kt pushback band that a
# 3 kt rule discards. Our gs is broadcast on 98.02% of ground samples at 0.1 kt resolution, so unlike
# OPDI we have no reason to derive it from positions, and a median filter with a sustain rule measured
# worse at every threshold.
MOVING_KT = float(os.environ.get("MOVING_KT", "3.0"))


def surface_mask(df: pl.DataFrame) -> pl.Expr:
    elev = pl.col("airport").replace_strict(ELEV_FT, return_dtype=pl.Float64)
    return pl.col("on_ground") | ((pl.col("gs").fill_null(0) <= SURFACE_KT) & (pl.col("alt").is_null() | (pl.col("alt") <= elev + 800)))


STAND_M = float(os.environ.get("STAND_M", "150"))
OUT_DIR = os.environ.get("ADSB_OUT_DIR", "data/adsblol")


def dist_m(lat1, lon1, lat2, lon2):
    """Great-circle metres; scalars or arrays."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(np.asarray(lat2) - np.asarray(lat1)), np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def segment_start(t, gs, ground, end, lat=None, lon=None, report=False):
    """First index of the surface segment that ends at `end`.

    Default: the last continuous ground run - walk back while the previous point is ground and the gap is at
    most GAP_S. That rule exists so the inbound taxi of the same airframe cannot be swallowed, but it also
    throws away the stand dwell whenever reception drops for more than 15 minutes, which is most of the time:
    on four 2026 days 74.4% of matched departures had earlier ground points discarded, the aircraft having
    moved a median of 62 m across the break, and 2476 of them lost their first-movement estimate entirely.

    STAND_WALK=1 (E61, 18.09) additionally extends the segment backwards through points that are BOTH
    stationary and within STAND_M of the segment start, so the walk stops at the last moment the aircraft was
    in motion - the end of the inbound taxi - and cannot cross it. Walking back on distance alone does swallow
    it: that variant scored RMSE 1927 against the recorded block time, against 444 for the current rule.
    """
    start = end
    while start > 0 and ground[start - 1] and t[start] - t[start - 1] <= GAP_S:
        start -= 1
    plain = start
    if os.environ.get("STAND_WALK") in ("1", "2") and lat is not None:
        while (start > 0 and np.nan_to_num(gs[start - 1]) <= MOVING_KT
               and dist_m(lat[start - 1], lon[start - 1], lat[start], lon[start]) <= STAND_M):
            start -= 1
    return (start, plain) if report else start


def estimate(t: np.ndarray, gs: np.ndarray, ground: np.ndarray, takeoff: float, lat=None, lon=None):
    idx = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx) == 0:
        return 1, None, None, None, None, None, None, 0.0, 0.0, 0.0
    end = idx[-1]
    if takeoff - t[end] > 900:
        return 1, None, None, None, None, None, None, 0.0, 0.0, 0.0
    start, plain = segment_start(t, gs, ground, end, lat, lon, report=True)
    # E62 (18.09): E61 showed the recovered dwell helps its own rows (-0.063 s) but silently changing the feature
    # mix costs more on the untouched ones (+0.261 s). So report the reconstruction instead of hiding it: the
    # coverage class stays the one the uninterrupted segment earns, and the walk is described by its own features.
    sw_recovered = float(plain - start)
    sw_extra_s = float(t[plain] - t[start]) if start < plain else 0.0
    sw_gap_s = float(t[plain] - t[plain - 1]) if start < plain and plain > 0 else 0.0
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
    # STAND_WALK=2: decide the coverage class on the uninterrupted segment, so a reconstructed dwell cannot
    # silently relabel a flight as "seen stationary at the stand" - that relabelling is half of what E61 cost.
    cst = t[plain:end + 1] if os.environ.get("STAND_WALK") == "2" else st
    csg = np.nan_to_num(gs[plain:end + 1]) if os.environ.get("STAND_WALK") == "2" else sg
    cov, t_move = 2, cst[0]
    i = len(cst) - 1
    while i >= 0:  # last stationary stretch of >= STOP_S before take-off
        if csg[i] <= MOVING_KT:
            j = i
            while j > 0 and csg[j - 1] <= MOVING_KT:
                j -= 1
            if cst[i] - cst[j] >= STOP_S:
                cov, t_move = 3, cst[i]
                break
            i = j - 1
        else:
            i -= 1
    return (cov, float(st[0]), float(t_move), float(t[end])) + extra + (sw_recovered, sw_extra_s, sw_gap_s)


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
            out.append((mvt_id, 0 if not cand_map.get((cs, ap)) else 1, None, None, None, None, None, None, 0.0, 0.0, 0.0, 0, len(ok)))
            continue
        g = by_icao[ok[0]].filter(pl.col("airport") == ap)
        e = estimate(g["t"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), float(to),
                     g["lat"].to_numpy(), g["lon"].to_numpy())
        out.append((mvt_id, *e, g.height, 1))
    # explicit dtypes: on a day whose first rows are all nulls polars infers the wrong type and the append fails
    # (2025-06-11 had no offblock2 file for that reason until 16.09)
    schema = {"MVT_ID_mvt": pl.Float64, "coverage": pl.Int64, "t_first": pl.Float64, "t_move": pl.Float64, "t_to": pl.Float64,
              "t_first_move": pl.Float64, "stat_total": pl.Float64, "n_stops": pl.Int64,
              "sw_recovered": pl.Float64, "sw_extra_s": pl.Float64, "sw_gap_s": pl.Float64,
              "n_pts": pl.Int64, "n_cand": pl.Int64}
    res = pl.DataFrame(out, schema=schema, orient="row")
    os.makedirs(OUT_DIR, exist_ok=True)
    res.write_parquet(f"{OUT_DIR}/offblock2_{day}.parquet")
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
