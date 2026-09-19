"""E69: merge a reception gap back into the taxi it split (18.09, proposal by Denys).

adsb_visit.py splits an airframe's points at a gap of 1800 s and calls everything before the last gap the
previous visit. A reception gap in the middle of a long taxi has the same shape, so the first half of THIS
departure's taxi is read as a completed arrival. Verified on EHAM MVT_ID=182026895 (2025-01-15, DAL75, stand
E03, taxi-out 3762 s): the segment E65 calls the previous visit starts 9 s BEFORE block time, 38 m from the
assigned stand, and ends 4264 m away, running away from it the whole time (corr(t, distance) +0.97), while the
real arrival is the segment before it, running the other way (-0.85) and ending 63 m from the stand at zero
ground speed. vi_first_seen then reads 147 s where 3773 s elapsed from the first movement to the take-off.

Direction is what tells the two apart, and adsb_visit.py cannot see it: it reads t, alt and gs only, never the
lat/lon sitting in the same file.

This writes a SEPARATE file per day, so data/adsblol/visit_<day>.parquet -- what E64 and E65 were accepted on --
is neither rewritten nor re-extracted. The old columns stay exactly as they are and the new ones are added
beside them (REVIEW_BRIEF.md: change the input by adding, not by substituting).

Merge rule: fold the previous visit into the departure visit while it ends further than MERGE_D_END from the
assigned stand and came closer than MERGE_D_MIN to it at some point -- i.e. it left the stand and never came
back, which an arrival cannot do. Repeats, because a long taxi can be cut more than once.

    .venv/bin/python src/e69_visit_merge.py 2025-01-15      writes data/adsblol/visit_m_<day>.parquet
    .venv/bin/python src/e69_visit_merge.py --all           every day that has a surface file and no output yet
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

from adsb_visit import load_surface  # E75: the 12 h lookback is otherwise cut at midnight

from geometry import _haversine, load_tables, norm_stand, strip_suffix

VISIT_GAP = float(os.environ.get("VISIT_GAP", "1800"))
LOOKBACK = 12 * 3600
MERGE_D_END = float(os.environ.get("MERGE_D_END", "200"))   # left the stand by at least this much
MERGE_D_MIN = float(os.environ.get("MERGE_D_MIN", "200"))   # and was at the stand at some point
OUT_DIR = os.environ.get("ADSB_OUT_DIR", "data/adsblol")

NAMES_M = ["vm_pv_d_start", "vm_pv_d_end", "vm_pv_d_min", "vm_pv_corr", "vm_dv_d_first",
           "vm_merged", "vm_first_seen", "vm_turnaround", "vm_gap", "vm_prev_pts", "vm_n_pts"]
_INT = {"vm_merged", "vm_prev_pts", "vm_n_pts"}
SCHEMA = {"MVT_ID_mvt": pl.Float64, **{n: (pl.Int64 if n in _INT else pl.Float64) for n in NAMES_M}}


def stand_lookup() -> dict:
    """Assigned-stand coordinates with the same three fallbacks add_geometry uses: exact key, key without the
    suffix, then the centroid of the stand prefix."""
    st, _ = load_tables()
    exact = {(a, k): (la, lo) for a, k, la, lo in st.iter_rows()}
    nosfx: dict = {}
    for a, k, la, lo in st.iter_rows():
        nosfx.setdefault((a, strip_suffix(k)), (la, lo))
    pref: dict = {}
    for a, k, la, lo in st.iter_rows():
        p = k[:1] if k[:1].isalpha() else ""
        pref.setdefault((a, p), []).append((la, lo))
    pref = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for k, v in pref.items()}
    return {"exact": exact, "nosfx": nosfx, "pref": pref}


def resolve(lk: dict, ap: str, stand: str | None):
    if stand is None:
        return None
    k = norm_stand(stand)
    return (lk["exact"].get((ap, k)) or lk["nosfx"].get((ap, strip_suffix(k)))
            or lk["pref"].get((ap, k[:1] if k[:1].isalpha() else "")))


def one_day(day: str, lk: dict) -> pl.DataFrame:
    pts, pts_id = load_surface(day, columns=["icao", "airport", "t", "lat", "lon", "callsign"])
    y, m, d_ = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d_))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))

    cmap: dict[tuple, list] = {}
    for cs, ap, ic in pts_id.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cmap.setdefault((cs, ap), []).append(ic)
    by = {(k[0], k[1]): g.sort("t") for k, g in pts.group_by(["icao", "airport"])}

    out = []
    empty = (None, None, None, None, None, 0, None, None, None, None, None)
    for mvt_id, ap, stand, cs, to in dep.select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "cs", "to").iter_rows():
        ic = cmap.get((cs, ap), [])
        sc = resolve(lk, ap, stand)
        if len(ic) != 1 or (ic[0], ap) not in by or sc is None:
            out.append((mvt_id, *empty))
            continue
        g = by[(ic[0], ap)]
        t = g["t"].to_numpy()
        sel = (t >= to - LOOKBACK) & (t <= to + 600)
        if sel.sum() < 2:
            out.append((mvt_id, *empty))
            continue
        tt = t[sel]
        dist = _haversine(g["lat"].to_numpy()[sel], g["lon"].to_numpy()[sel], sc[0], sc[1])  # metres
        big = [int(b) for b in np.where(np.diff(tt) > VISIT_GAP)[0]]
        k = big[-1] + 1 if big else 0
        bounds = [0, *[b + 1 for b in big], len(tt)]          # segment starts, plus the end
        seg = len(bounds) - 2                                  # index of the departure segment

        pv0 = pv1 = None
        if seg > 0:
            pv0, pv1 = bounds[seg - 1], bounds[seg] - 1
        pv_ds = float(dist[pv0]) if pv0 is not None else None
        pv_de = float(dist[pv1]) if pv1 is not None else None
        pv_dm = float(dist[pv0:pv1 + 1].min()) if pv0 is not None else None
        pv_cr = (float(np.corrcoef(tt[pv0:pv1 + 1], dist[pv0:pv1 + 1])[0, 1])
                 if pv0 is not None and pv1 > pv0 and np.ptp(dist[pv0:pv1 + 1]) > 0 else None)

        merged = 0
        while seg > 0:
            a, b = bounds[seg - 1], bounds[seg] - 1
            if not (dist[b] > MERGE_D_END and dist[a:b + 1].min() < MERGE_D_MIN):
                break
            seg -= 1
            merged += 1
        k_m = bounds[seg]
        n_pts_m = int(len(tt) - k_m)
        first_seen_m = float(to - tt[k_m])
        if seg > 0:
            p_end = int(bounds[seg] - 1)
            p_start = bounds[seg - 1]
            turn_m, gap_m, prev_pts_m = float(to - tt[p_end]), float(tt[k_m] - tt[p_end]), int(p_end - p_start + 1)
        else:
            turn_m = gap_m = None
            prev_pts_m = None
        out.append((mvt_id, pv_ds, pv_de, pv_dm, pv_cr, float(dist[k]),
                    merged, first_seen_m, turn_m, gap_m, prev_pts_m, n_pts_m))

    return pl.DataFrame(out, schema=SCHEMA, orient="row")


def main() -> None:
    lk = stand_lookup()
    if sys.argv[1] == "--all":
        days = sorted(p.split("surface_")[1][:-8] for p in glob.glob("data/adsblol/surface_*.parquet"))
        days = [d for d in days if not os.path.exists(f"{OUT_DIR}/visit_m_{d}.parquet")]
    else:
        days = sys.argv[1:]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{len(days)} days to do", flush=True)
    for i, day in enumerate(days, 1):
        res = one_day(day, lk)
        res.write_parquet(f"{OUT_DIR}/visit_m_{day}.parquet")
        print(f"[{i}/{len(days)}] {day}: {res.height} departures, merged {int((res['vm_merged'] > 0).sum())}", flush=True)


if __name__ == "__main__":
    main()
