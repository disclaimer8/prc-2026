"""E39: truncated-track augmentation for the ADS-B stage (reviewer 15.09 item 5; also in both deep-research reports).

    python src/adsb_truncate_augment.py 2025-07      writes data/adsblol/offblock_trunc_<day>.parquet for every day of the month
For 2025 departures whose own track (call-sign match as v2) starts stationary (coverage 3), a copy of the track is cut at
t_cut = first movement + U(60 s, 600 s) (seeded per MVT_ID), i.e. the stand and the push-back are removed, and the offblock2
(coverage, t_first, t_move, t_to, t_first_move, stat_total, n_stops, n_pts) and offblock3 (sf_*) features are recomputed on
the cut copy. Stage training (src/e39_adsb_truncation_stage.py) adds these copies to the real rows of the training months;
evaluation uses real rows only; a copy stays in the month (fold) of its flight.
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import estimate, surface_mask  # noqa: E402
from adsb_offblock_v3 import segment_features  # noqa: E402

SF = ["sf_path_km", "sf_straight_km", "sf_move_time", "sf_longest_stop", "sf_last_stop", "sf_mean_speed", "sf_n_seg", "sf_to_gap"]


def day_rows(day: str, deps: pl.DataFrame) -> list:
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    dep = deps.filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))
    cand_map: dict[tuple, list] = {}
    for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    st_map = {(r[0], r[1]): np.sort(np.asarray(r[2])) for r in pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t")).iter_rows()}
    by_key = {(k[0], k[1]): g for k, g in pts.group_by(["icao", "airport"])}
    out = []
    for mvt_id, ap, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ok = [i for i in cand_map.get((cs, ap), []) if (ts := st_map.get((i, ap))) is not None and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
        if len(ok) != 1:
            continue
        g = by_key[(ok[0], ap)]
        t, gs, surf, lat, lon = g["t"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy()
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
        if e2[0] < 2 or f3 is None:
            continue
        out.append((mvt_id, *e2, int(keep.sum()), 1, *f3))
    return out


def main() -> None:
    month = sys.argv[1]
    days = sorted(os.path.basename(f)[8:18] for f in glob.glob(f"data/adsblol/surface_{month}-*.parquet"))
    y, m = map(int, month.split("-"))
    deps = pl.read_parquet(f"data/prc-2026-datasets/training_{month}-01_*.parquet").filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""), to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    cols = ["MVT_ID_mvt", "coverage", "t_first", "t_move", "t_to", "t_first_move", "stat_total", "n_stops", "n_pts", "n_cand", *SF]
    for day in days:
        target = f"data/adsblol/offblock_trunc_{day}.parquet"
        if os.path.exists(target):
            continue
        rows = day_rows(day, deps)
        pl.DataFrame(rows, schema=cols, orient="row").write_parquet(target)
        print(day, len(rows), flush=True)


if __name__ == "__main__":
    main()
