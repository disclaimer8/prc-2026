"""E49: do the other two adsb.lol replicas of a day contain surface points that prod-0 does not? (16.09)

    .venv/bin/python src/e49_replica_probe.py 2026-07-06 2026-01-15

Every day of globe_history exists as up to three releases: `prod-0` (what scripts/get_adsb_day.sh downloads),
`staging-0` (52 of the 62 ranking days) and `mlatonly-0` (all 62 ranking days, 86-140 MB). MLAT positions are computed by
the network from time differences between receivers, so they can exist exactly where an aircraft is not heard on the
ground by a single receiver — LTFM has 0.0% surface coverage in both ranking months, LFPG 21-25%, LIRF 6% in July.
The probe extracts the same day from the other replicas into data/adsblol/probe_<replica>_<day>.parquet and counts,
per airport, points and surface points that the prod-0 file does not already have (a point is the same if the same
aircraft has a point within 1 s of it).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import surface_mask  # noqa: E402


def load(path: str) -> pl.DataFrame | None:
    if not os.path.exists(path):
        return None
    d = pl.read_parquet(path)
    return d.with_columns(surf=surface_mask(d), tsec=pl.col("t").round(0))


def main() -> None:
    for day in sys.argv[1:]:
        base = load(f"data/adsblol/surface_{day}.parquet")
        if base is None:
            print(f"{day}: no prod-0 file")
            continue
        b = base.group_by("airport").agg(pts=pl.len(), surf=pl.col("surf").sum(), icao=pl.col("icao").n_unique())
        print(f"\n{day} prod-0: " + ", ".join(f"{r[0]} {r[1]}/{r[2]} surf" for r in b.sort("airport").iter_rows()))
        known = set(zip(base["icao"].to_list(), base["tsec"].to_list()))
        for rep in ("staging", "mlat"):
            d = load(f"data/adsblol/probe_{rep}_{day}.parquet")
            if d is None:
                print(f"  {rep}: not extracted")
                continue
            new = d.filter(~pl.Series([(i, t) in known for i, t in zip(d["icao"].to_list(), d["tsec"].to_list())]))
            g = new.group_by("airport").agg(pts=pl.len(), surf=pl.col("surf").sum(), icao=pl.col("icao").n_unique()).sort("airport")
            tot_new_surf = int(new["surf"].sum())
            print(f"  {rep}: {d.height} points, new to prod-0 {new.height} ({tot_new_surf} of them surface)")
            for r in g.iter_rows():
                if r[1]:
                    print(f"      {r[0]}: +{r[1]} points, +{r[2]} surface, {r[3]} aircraft")


if __name__ == "__main__":
    main()
