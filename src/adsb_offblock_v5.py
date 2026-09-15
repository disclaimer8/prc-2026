"""E36: stand geometry of the surface track (reviewer proposal 15.09: tell a stop at the stand from a stop in the queue).

    .venv/bin/python src/adsb_offblock_v5.py 2025-07-15      writes data/adsblol/offblock5_<day>.parquet
Aircraft matched as in v2. Own stand coordinates from OpenStreetMap (geometry.load_tables, exact or suffix-stripped ref;
prefix centroids are not used). Features over the surface segment that ends at take-off:
  gs_stand_known       1 if the stand was located
  gs_d_first           distance of the first surface point to the stand (m)
  gs_d_firstmove       distance of the point where the first movement starts (end of the initial stationary stretch)
  gs_d_lastmove        distance to the stand of the point where the final movement starts (end of the last stop)
  gs_d_lastmove_rwy    distance of that point to the departure runway start (m)
  gs_t_leave           time the aircraft is first > 150 m from the stand after having been within 150 m (epoch s)
  gs_dwell_stand       time spent within 150 m of the stand before leaving (s)
  gs_seen_at_stand     1 if any segment point is within 150 m of the stand
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import GAP_S, MOVING_KT, surface_mask  # noqa: E402
from geometry import load_tables, norm_stand, strip_suffix  # noqa: E402

NEAR_M = 150.0


def dist_m(lat, lon, lat0, lon0):
    k = math.cos(math.radians(lat0))
    return np.sqrt(((np.asarray(lat) - lat0) * 111_200) ** 2 + ((np.asarray(lon) - lon0) * 111_200 * k) ** 2)


def features(t, lat, lon, gs, ground, takeoff, stand, rwy):
    idx = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx) == 0 or takeoff - t[idx[-1]] > 900:
        return None
    end = idx[-1]
    start = end
    while start > 0 and ground[start - 1] and t[start] - t[start - 1] <= GAP_S:
        start -= 1
    st, la, lo, sg = t[start:end + 1], lat[start:end + 1], lon[start:end + 1], np.nan_to_num(gs[start:end + 1])
    moving = sg > MOVING_KT
    k0 = int(np.argmax(moving)) if moving.any() else len(st) - 1  # first moving point
    # start of the final movement: after the last stationary point
    stat_idx = np.where(~moving)[0]
    k_last = int(stat_idx[-1] + 1) if len(stat_idx) and stat_idx[-1] + 1 < len(st) else k0
    out = dict(gs_stand_known=0.0, gs_d_first=None, gs_d_firstmove=None, gs_d_lastmove=None, gs_d_lastmove_rwy=None,
               gs_t_leave=None, gs_dwell_stand=None, gs_seen_at_stand=None)
    if rwy is not None:
        out["gs_d_lastmove_rwy"] = float(dist_m(la[k_last], lo[k_last], *rwy))
    if stand is None:
        return out
    d = dist_m(la, lo, *stand)
    near = d <= NEAR_M
    out.update(gs_stand_known=1.0, gs_d_first=float(d[0]), gs_d_firstmove=float(d[k0]), gs_d_lastmove=float(d[k_last]),
               gs_seen_at_stand=float(near.any()))
    if near.any():
        first_near = int(np.argmax(near))
        after = np.where(~near[first_near:])[0]
        if len(after):
            leave = first_near + int(after[0])
            out["gs_t_leave"] = float(st[leave])
            out["gs_dwell_stand"] = float(st[leave] - st[first_near])
    return out


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""), to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    stt, rw = load_tables()
    st_map = {(a, k): (la, lo) for a, k, la, lo in stt.iter_rows()}
    st_nosfx = {}
    for a, k, la, lo in stt.iter_rows():
        st_nosfx.setdefault((a, strip_suffix(k)), (la, lo))
    rw_map = {(a, r): (la, lo) for a, r, la, lo, _ in rw.iter_rows()}
    cands = pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique()
    cand_map: dict[tuple, list] = {}
    for cs, ap, icao in cands.iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    surf_t = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t")).iter_rows()}
    by_icao = {k[0]: g for k, g in pts.group_by(["icao"])}
    names = ["gs_stand_known", "gs_d_first", "gs_d_firstmove", "gs_d_lastmove", "gs_d_lastmove_rwy", "gs_t_leave", "gs_dwell_stand", "gs_seen_at_stand"]
    rows = []
    for mvt_id, ap, cs, to, stand_raw, runway in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to", "STAND_mvt", "RUNWAY_mvt").iter_rows():
        ok = [icao for icao in cand_map.get((cs, ap), []) if (ts := surf_t.get((icao, ap))) is not None
              and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
        f = None
        if len(ok) == 1:
            key = norm_stand(stand_raw) if stand_raw else None
            stand = st_map.get((ap, key)) or st_nosfx.get((ap, strip_suffix(key))) if key else None
            g = by_icao[ok[0]].filter(pl.col("airport") == ap)
            f = features(g["t"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), float(to),
                         stand, rw_map.get((ap, runway)))
        rows.append((mvt_id, *([f[n] for n in names] if f else [None] * len(names))))
    pl.DataFrame(rows, schema={"MVT_ID_mvt": pl.Float64, **{n: pl.Float64 for n in names}}, orient="row").write_parquet(f"data/adsblol/offblock5_{day}.parquet")


if __name__ == "__main__":
    main()
