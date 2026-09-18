"""E64: visit features from the AIRBORNE points we already download (18.09).

data/adsblol/surface_<day>.parquet keeps every point within 7 km of the aerodrome and below 2500 ft, airborne
ones included; the offblock extractors use only the ground part and throw the icao24 away after matching. So for
the 824k departures with coverage <= 1 - 61.83 s of the decided-row budget, where no stage reaches - there is in
fact an observation: the airframe is seen on approach and on climb-out, just not on the ground.

Per departure with exactly one icao24 carrying its call sign at that airport, split that airframe's points into
visits separated by a gap of VISIT_GAP. The visit containing the take-off is the departure; the one before it
ended when the aircraft taxied in. Write both, plus what the first point of the departure visit looks like.

Measured before writing this (5 days of 2025, coverage <= 1, n = 4615): the turnaround itself is useless as a
bound - it is a median 6625 s against a median taxi-out of 964 s and binds in 0.2% of rows - and correlates
-0.02 with the residual. What does correlate is how long before take-off the airframe reappears: +0.11, four
times anything else measured on 17-18.09. Mean |residual| by turnaround quartile rises 117 -> 143 -> 155 -> 171 s.

    .venv/bin/python src/adsb_visit.py 2025-07-15     writes data/adsblol/visit_<day>.parquet
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

VISIT_GAP = float(os.environ.get("VISIT_GAP", "1800"))
LOOKBACK = 12 * 3600
OUT_DIR = os.environ.get("ADSB_OUT_DIR", "data/adsblol")
NAMES = ["vi_first_seen", "vi_turnaround", "vi_gap", "vi_n_pts", "vi_prev_pts",
         "vi_alt_first", "vi_gs_first", "vi_has_prev"]


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet").sort("icao", "t")
    y, m, d = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    cmap: dict[tuple, list] = {}
    for cs, ap, ic in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cmap.setdefault((cs, ap), []).append(ic)
    by = {(k[0], k[1]): g.select("t", "alt", "gs").sort("t") for k, g in pts.group_by(["icao", "airport"])}
    out = []
    for mvt_id, ap, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to").iter_rows():
        ic = cmap.get((cs, ap), [])
        if len(ic) != 1 or (ic[0], ap) not in by:
            out.append((mvt_id, *([None] * 7), 0))
            continue
        g = by[(ic[0], ap)]
        t = g["t"].to_numpy()
        sel = (t >= to - LOOKBACK) & (t <= to + 600)
        if sel.sum() < 2:
            out.append((mvt_id, *([None] * 7), 0))
            continue
        tt = t[sel]
        alt = g["alt"].to_numpy()[sel]
        gs = g["gs"].to_numpy()[sel]
        big = np.where(np.diff(tt) > VISIT_GAP)[0]
        k = int(big[-1]) + 1 if len(big) else 0        # first index of the departure visit
        vis = slice(k, len(tt))
        first_seen = float(to - tt[k])
        n_pts = int(len(tt) - k)
        a0 = float(alt[k]) if np.isfinite(alt[k]) else None
        g0 = float(gs[k]) if np.isfinite(gs[k]) else None
        if len(big):
            prev_end = float(tt[int(big[-1])])
            prev_start = 0 if len(big) == 1 else int(big[-2]) + 1
            out.append((mvt_id, first_seen, float(to - prev_end), float(tt[k] - prev_end), n_pts,
                        int(int(big[-1]) - prev_start + 1), a0, g0, 1))
        else:
            out.append((mvt_id, first_seen, None, None, n_pts, None, a0, g0, 0))
    schema = {"MVT_ID_mvt": pl.Float64, **{n: (pl.Int64 if n in ("vi_n_pts", "vi_prev_pts", "vi_has_prev") else pl.Float64) for n in NAMES}}
    os.makedirs(OUT_DIR, exist_ok=True)
    res = pl.DataFrame(out, schema=schema, orient="row")
    res.write_parquet(f"{OUT_DIR}/visit_{day}.parquet")
    print(f"{day}: {res.height} departures, airframe identified for {int(res['vi_has_prev'].is_not_null().sum())}, "
          f"with a previous visit {int((res['vi_has_prev'] == 1).sum())}")


if __name__ == "__main__":
    main()
