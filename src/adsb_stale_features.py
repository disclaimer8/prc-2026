"""E44: what part of the surface track is repeated rather than received (16.09).

    .venv/bin/python src/adsb_stale_features.py 2025-07-15      writes data/adsblol/stale_<day>.parquet

readsb marks a trace position with bit 0 of p[6] when nothing has been received for 20 s and the point simply repeats the
last known position (src/adsb_extract.py keeps it as the `stale` column since 16.09). A stand with fresh reception and a
stand held only by repeats are not the same evidence: the off-block estimate of src/adsb_offblock_v2.py treats both as
"the aircraft stood there", so a long repeat stretch can invent a stop that was never observed.

Per departure, over the same surface segment that offblock2 uses (ground points up to take-off + 300 s, walked back while
the gap is <= 900 s):
    stale_share      share of stale points in the segment
    stale_first      the first point of the segment is a repeat
    stale_at_move    the first movement point is a repeat
    stale_run_max    longest uninterrupted repeat stretch, seconds
    stale_before_to  seconds of repeats in the last 600 s before take-off
    fresh_before_move  seconds from the last freshly received point to the first movement (null without a first movement)
Days extracted before 16.09 have no `stale` column and are skipped loudly, so a half-re-extracted archive cannot silently
produce a feature that is zero for half the year.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import GAP_S, MOVING_KT, surface_mask  # noqa: E402

COLS = ["MVT_ID_mvt", "stale_share", "stale_first", "stale_at_move", "stale_run_max", "stale_before_to", "fresh_before_move"]


def segment(t: np.ndarray, ground: np.ndarray, takeoff: float):
    """Indices of the surface segment offblock2 works on, or None."""
    idx = np.where(ground & (t <= takeoff + 300))[0]
    if len(idx) == 0 or takeoff - t[idx[-1]] > 900:
        return None
    end = idx[-1]
    start = end
    while start > 0 and ground[start - 1] and t[start] - t[start - 1] <= GAP_S:
        start -= 1
    return start, end


def stale_row(t: np.ndarray, gs: np.ndarray, ground: np.ndarray, stale: np.ndarray, takeoff: float):
    seg = segment(t, ground, takeoff)
    if seg is None:
        return None
    a, b = seg
    st, sg, sl = t[a:b + 1], np.nan_to_num(gs[a:b + 1]), stale[a:b + 1]
    dt = np.diff(st, prepend=st[0])
    run_max = run = 0.0
    for k in range(len(st)):
        run = run + dt[k] if sl[k] else 0.0
        run_max = max(run_max, run)
    moving = sg > MOVING_KT
    t_move = float(st[np.argmax(moving)]) if moving.any() and not moving[0] else None
    fresh_before_move = None
    if t_move is not None:
        fresh = st[(~sl) & (st <= t_move)]
        fresh_before_move = float(t_move - fresh[-1]) if len(fresh) else None
    last600 = st >= takeoff - 600
    return (float(sl.mean()), bool(sl[0]), bool(sl[np.argmax(moving)]) if t_move is not None else None,
            float(run_max), float(dt[last600 & sl].sum()), fresh_before_move)


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    if "stale" not in pts.columns:
        sys.exit(f"data/adsblol/surface_{day}.parquet has no `stale` column: re-extract the day first (src/reextract_months.py)")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    cand_map: dict[tuple, list] = {}
    for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    st_map = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t")).iter_rows()}
    by_key = {(k[0], k[1]): g for k, g in pts.group_by(["icao", "airport"])}
    out = []
    for mvt_id, ap, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ok = [i for i in cand_map.get((cs, ap), [])
              if (ts := st_map.get((i, ap))) is not None and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
        if len(ok) != 1:
            continue
        g = by_key[(ok[0], ap)]
        r = stale_row(g["t"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(),
                      g["stale"].fill_null(False).to_numpy(), float(to))
        if r is not None:
            out.append((mvt_id, *r))
    pl.DataFrame(out, schema=COLS, orient="row").write_parquet(f"data/adsblol/stale_{day}.parquet")
    print(day, len(out), flush=True)


if __name__ == "__main__":
    main()
