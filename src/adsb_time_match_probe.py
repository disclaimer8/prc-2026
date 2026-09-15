"""Probe (15.09) for E31: departures without a call-sign match in adsb.lol (coverage 0/1 in offblock2) — can they be matched by
take-off time alone (an aircraft whose last surface point at the airport is within +-90 s of MVT and that is the only such
aircraft)? And how often does the time-only match agree with the call-sign match where both exist?
    .venv/bin/python src/adsb_time_match_probe.py 2025-07-15"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import surface_mask  # noqa: E402

day = sys.argv[1]
pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
pts = pts.with_columns(surf=surface_mask(pts))
last_surf = pts.filter(pl.col("surf")).group_by("icao", "airport").agg(pl.col("t"), cs=pl.col("callsign").drop_nulls().last())
y, m, d = map(int, day.split("-"))
src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
    to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""))
o2 = pl.read_parquet(f"data/adsblol/offblock2_{day}.parquet").select("MVT_ID_mvt", "coverage", "n_cand")
dep = dep.join(o2, on="MVT_ID_mvt", how="left")
# per airport: the end of each surface segment (point followed by a gap > 120 s or the last point)
ends = []
for (icao, ap), g in pts.sort("t").group_by(["icao", "airport"]):
    t, surf = g["t"].to_numpy(), g["surf"].to_numpy()
    alt = g["alt"].to_numpy()
    for i in np.where(surf)[0]:
        # take-off: the next point is airborne (not surface) within 120 s
        if i + 1 < len(t) and not surf[i + 1] and t[i + 1] - t[i] <= 120:
            ends.append((icao, ap, float(t[i])))
E = pl.DataFrame(ends, schema=["icao", "airport", "t_end"], orient="row").sort("t_end")
cs_icao = {}
for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
    cs_icao.setdefault((cs, ap), set()).add(icao)
res = []
for mvt_id, ap, to, cov, cs in dep.select("MVT_ID_mvt", "ADEP_mvt", "to", "coverage", "cs").iter_rows():
    c = E.filter((pl.col("airport") == ap) & (pl.col("t_end") - to).is_between(-90, 90))["icao"].unique().to_list()
    agree = None
    if len(c) == 1 and cov is not None and cov >= 2:
        agree = c[0] in cs_icao.get((cs, ap), set())
    res.append((mvt_id, ap, cov, len(c), agree))
r = pl.DataFrame(res, schema=["MVT_ID_mvt", "ap", "coverage", "n_time_cands", "agree"], orient="row")
print("time-only match agrees with call-sign match:", r.filter(pl.col("agree").is_not_null())["agree"].mean(), "of", r["agree"].is_not_null().sum())
print(r.group_by("ap").agg(deps=pl.len(), cs_matched=(pl.col("coverage") >= 2).sum(),
     unmatched_with_1_time_cand=((pl.col("coverage") < 2) & (pl.col("n_time_cands") == 1)).sum(),
     matched_with_1_time_cand=((pl.col("coverage") >= 2) & (pl.col("n_time_cands") == 1)).sum()).sort("ap"))
