"""E31: off-block features v2 plus a take-off-time fallback match (15.09, probe src/adsb_time_match_probe.py).

    .venv/bin/python src/adsb_offblock_v4.py 2025-07-15      writes data/adsblol/offblock4_<day>.parquet
Rows matched by call sign exactly as in v2 keep match_type = 1. Rows without a unique call-sign match get the aircraft whose
surface segment ends with a take-off (next point airborne within 120 s) within +-90 s of MVT, if exactly one aircraft
qualifies: match_type = 2 (on 15.07.2025 such a match agrees with the call-sign match in 86.6% of rows where both exist).
Columns as offblock2 plus match_type (0 none).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import estimate, surface_mask  # noqa: E402


def main() -> None:
    day = sys.argv[1]
    o2 = pl.read_parquet(f"data/adsblol/offblock2_{day}.parquet")
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).select(
        "MVT_ID_mvt", "ADEP_mvt", to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    takeoffs: dict[str, list] = {}
    by_key = {}
    for (icao, ap), g in pts.group_by(["icao", "airport"]):
        by_key[(icao, ap)] = g
        t, surf = g["t"].to_numpy(), g["surf"].to_numpy()
        for i in np.where(surf[:-1] & ~surf[1:] & (np.diff(t) <= 120))[0]:
            takeoffs.setdefault(ap, []).append((float(t[i]), icao))
    tk = {ap: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for ap, v in takeoffs.items()}
    base = dep.join(o2, on="MVT_ID_mvt", how="left")
    rows = []
    for r in base.iter_rows(named=True):
        if r["coverage"] is not None and r["coverage"] >= 2:
            rows.append({**{k: r[k] for k in o2.columns}, "match_type": 1})
            continue
        out = {k: r.get(k) for k in o2.columns}
        out["match_type"] = 0
        if r["ADEP_mvt"] in tk:
            tt, ii = tk[r["ADEP_mvt"]]
            c = np.unique(ii[np.abs(tt - r["to"]) <= 90])
            if len(c) == 1:
                g = by_key[(c[0], r["ADEP_mvt"])]
                e = estimate(g["t"].to_numpy(), g["gs"].to_numpy(), g["surf"].to_numpy(), r["to"])
                if e[0] >= 2:
                    out.update(dict(zip(["coverage", "t_first", "t_move", "t_to", "t_first_move", "stat_total", "n_stops"], e)))
                    out.update(n_pts=g.height, n_cand=1, match_type=2)
        rows.append(out)
    pl.DataFrame(rows, schema={**o2.schema, "match_type": pl.Int8}).write_parquet(f"data/adsblol/offblock4_{day}.parquet")


if __name__ == "__main__":
    main()
