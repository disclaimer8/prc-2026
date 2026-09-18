"""E30: extra surface-track features per departure on top of offblock2 (15.09).

    .venv/bin/python src/adsb_offblock_v3.py 2025-07-15      writes data/adsblol/offblock3_<day>.parquet
Same matching as v2 (one icao24 with the call sign whose last surface point is within [MVT - 900 s, MVT + 300 s]); from the
surface segment that ends at take-off: path length (km), straight distance first point -> last surface point (km), moving
time (gs > 3 kt), longest stop, duration of the last stop before take-off (holding point queue), mean moving speed (kt),
number of points, time between the last surface point and the recorded take-off.
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import GAP_S, MOVING_KT, OUT_DIR, segment_start, surface_mask  # noqa: E402


def segment_features(t, lat, lon, gs, ground, takeoff):
    idx = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx) == 0 or takeoff - t[idx[-1]] > 900:
        return None
    end = idx[-1]
    start = segment_start(t, gs, ground, end, lat, lon)
    st, la, lo, sg = t[start:end + 1], lat[start:end + 1], lon[start:end + 1], np.nan_to_num(gs[start:end + 1])
    k = math.cos(math.radians(float(np.nanmean(la))))
    dy, dx = np.diff(la) * 111.2, np.diff(lo) * 111.2 * k
    path = float(np.sqrt(dx * dx + dy * dy).sum()) if len(st) > 1 else 0.0
    straight = float(math.hypot((la[-1] - la[0]) * 111.2, (lo[-1] - lo[0]) * 111.2 * k))
    dt = np.diff(st, prepend=st[0])
    moving = sg > MOVING_KT
    move_time = float(dt[moving].sum())
    stops, cur = [], 0.0
    for i in range(len(st)):
        if not moving[i]:
            cur += dt[i]
        else:
            if cur > 0:
                stops.append(cur)
            cur = 0.0
    last_stop = cur if cur > 0 else (stops[-1] if stops else 0.0)
    longest = max(stops + [cur]) if (stops or cur) else 0.0
    mean_speed = float(sg[moving].mean()) if moving.any() else 0.0
    return path, straight, move_time, longest, last_stop, mean_speed, int(len(st)), float(takeoff - st[-1])


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    D = "data/prc-2026-datasets/"
    src = pl.read_parquet(D + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""), to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    cands = pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique()
    cand_map: dict[tuple, list] = {}
    for cs, ap, icao in cands.iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    st_map = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t")).iter_rows()}
    by_icao = {k[0]: g for k, g in pts.group_by(["icao"])}
    names = ["sf_path_km", "sf_straight_km", "sf_move_time", "sf_longest_stop", "sf_last_stop", "sf_mean_speed", "sf_n_seg", "sf_to_gap"]
    out = []
    for mvt_id, ap, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ok = [icao for icao in cand_map.get((cs, ap), []) if (ts := st_map.get((icao, ap))) is not None
              and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
        f = None
        if len(ok) == 1:
            g = by_icao[ok[0]].filter(pl.col("airport") == ap)
            f = segment_features(g["t"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), float(to))
        out.append((mvt_id, *(f if f is not None else [None] * len(names))))
    os.makedirs(OUT_DIR, exist_ok=True)
    pl.DataFrame(out, schema=["MVT_ID_mvt", *names], orient="row").write_parquet(f"{OUT_DIR}/offblock3_{day}.parquet")


if __name__ == "__main__":
    main()
