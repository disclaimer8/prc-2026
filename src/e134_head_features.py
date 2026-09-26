"""E134 (26.09, hypothesis 71): features of the HEAD of the surface segment, where the track starts.

Question from the external review: when the start (off-block) is unseen, does the remaining track carry information
about the missing time? E39 trained the stage on truncated copies with the same hand features and failed; here the
features describe the first part of the segment itself, which E39 did not have.

Per departure, on the surface segment that ends at take-off (same matching and segment rule as offblock v2/v3):
  hd_gs0          ground speed at the first point (kt)
  hd_gs_mean5     mean ground speed over the first 5 points
  hd_moving0      first point already moving (gs > 3 kt)
  hd_accel30      gs at first point +30 s minus hd_gs0
  hd_frac_mov60   share of the first 60 s moving
  hd_t_first_stop seconds from the first point to the first stop of >= 20 s (NaN if none before take-off)
  hd_turn5        absolute heading change over the first 5 points (deg)
  hd_path_frac    path length from the first point to the end / (straight first->end), tortuosity
Writes data/adsblol/head_<day>.parquet for the real departures of the day and, for 2025 days,
data/adsblol/head_trunc_<day>.parquet for the E39 truncated copies (same cut: first movement + U(60, 600) s, seeded by
MVT_ID) together with their recomputed offblock2/offblock3 features.

    python src/e134_head_features.py 2025-07-15
"""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import MOVING_KT, estimate, segment_start, surface_mask  # noqa: E402
from adsb_offblock_v3 import segment_features  # noqa: E402

HEAD = ["hd_gs0", "hd_gs_mean5", "hd_moving0", "hd_accel30", "hd_frac_mov60", "hd_t_first_stop", "hd_turn5", "hd_path_frac"]
SF = ["sf_path_km", "sf_straight_km", "sf_move_time", "sf_longest_stop", "sf_last_stop", "sf_mean_speed", "sf_n_seg", "sf_to_gap"]
OB = ["coverage", "t_first", "t_move", "t_to", "t_first_move", "stat_total", "n_stops"]


def head(t, lat, lon, gs, ground, takeoff):
    idx = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx) == 0 or takeoff - t[idx[-1]] > 900:
        return None
    end = idx[-1]
    start = segment_start(t, gs, ground, end)
    st, la, lo, sg = t[start:end + 1], lat[start:end + 1], lon[start:end + 1], np.nan_to_num(gs[start:end + 1])
    if len(st) < 2:
        return None
    k = math.cos(math.radians(float(np.nanmean(la))))
    dy, dx = np.diff(la) * 111.2, np.diff(lo) * 111.2 * k
    seg = np.sqrt(dx * dx + dy * dy)
    straight = math.hypot((la[-1] - la[0]) * 111.2, (lo[-1] - lo[0]) * 111.2 * k)
    path_frac = float(seg.sum() / straight) if straight > 0.05 else np.nan
    gs0 = float(sg[0]); mean5 = float(sg[:5].mean())
    j30 = np.searchsorted(st, st[0] + 30); accel = float(sg[min(j30, len(sg) - 1)] - gs0)
    m60 = st <= st[0] + 60
    dtt = np.diff(st, prepend=st[0])
    frac60 = float((dtt[m60] * (sg[m60] > MOVING_KT)).sum() / max(dtt[m60].sum(), 1.0))
    t_stop = np.nan
    i = 0
    while i < len(st):
        if sg[i] <= MOVING_KT:
            j = i
            while j + 1 < len(st) and sg[j + 1] <= MOVING_KT:
                j += 1
            if st[j] - st[i] >= 20 and i > 0:
                t_stop = float(st[i] - st[0]); break
            i = j + 1
        else:
            i += 1
    n = min(5, len(dx))
    hd = [math.degrees(math.atan2(dx[q], dy[q])) for q in range(n) if seg[q] > 0.005]
    turn = float(abs(((hd[-1] - hd[0]) + 180) % 360 - 180)) if len(hd) >= 2 else np.nan
    return gs0, mean5, float(gs0 > MOVING_KT), accel, frac60, t_stop, turn, path_frac


def main() -> None:
    day = sys.argv[1]
    y, m, d = map(int, day.split("-"))
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet").with_columns(surf=surface_mask(pl.read_parquet(f"data/adsblol/surface_{day}.parquet"))) if False else None
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    D = "data/prc-2026-datasets/"
    src = pl.read_parquet(D + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""), to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    cand_map: dict[tuple, list] = {}
    for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    st_map = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t")).iter_rows()}
    by_key = {(k[0], k[1]): g for k, g in pts.group_by(["icao", "airport"])}
    real, trunc = [], []
    for mvt_id, ap, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ok = [i for i in cand_map.get((cs, ap), []) if (ts := st_map.get((i, ap))) is not None and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
        if len(ok) != 1:
            continue
        g = by_key[(ok[0], ap)]
        t, gs, surf, lat, lon = g["t"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy()
        h = head(t, lat, lon, gs, surf, float(to))
        if h is not None:
            real.append((mvt_id, *h))
        if y != 2025:
            continue
        e = estimate(t, gs, surf, float(to))
        if e[0] != 3 or e[4] is None:
            continue
        rng = np.random.default_rng(int(mvt_id) % (2**32))
        cut = e[4] + rng.uniform(60, 600)
        keep = t >= cut
        if keep.sum() < 5:
            continue
        e2 = estimate(t[keep], gs[keep], surf[keep], float(to))
        f3 = segment_features(t[keep], lat[keep], lon[keep], gs[keep], surf[keep], float(to))
        h2 = head(t[keep], lat[keep], lon[keep], gs[keep], surf[keep], float(to))
        if e2[0] < 2 or f3 is None or h2 is None:
            continue
        trunc.append((mvt_id, *e2[:7], *f3, *h2))
    pl.DataFrame(real, schema=["MVT_ID_mvt", *HEAD], orient="row").write_parquet(f"data/adsblol/head_{day}.parquet")
    if y == 2025:
        pl.DataFrame(trunc, schema=["MVT_ID_mvt", *OB, *SF, *HEAD], orient="row").write_parquet(f"data/adsblol/head_trunc_{day}.parquet")
    print(day, len(real), len(trunc), flush=True)


if __name__ == "__main__":
    main()
