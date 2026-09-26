"""E139 probe (26.09, journal 88): joint flight-track assignment (Hungarian / Kuhn 1955) vs the E31 per-flight rule.

Truth: departures matched by call sign exactly as offblock v2 does (one candidate icao whose last surface point lies in
[MVT-900, MVT+300]); their take-off event is the icao's surface->air transition nearest MVT (next point airborne <= 120 s).
Hide the call sign of a random HIDE share of truth rows (seeded), then recover them:
  E31:  the unique icao with a take-off event within +-90 s of MVT among ALL events (adsb_offblock_v4 rule).
  HUNG: per airport-day, flights = hidden rows + rows without a call-sign match; events = take-off events not claimed
        by the still-known call-sign pairs; cost |event - MVT| for |dt| <= 90 s, plus one "unknown" option per flight
        with cost UNK (pre-registered 45 s); solved by scipy linear_sum_assignment.
Reports per method: correct / wrong / none on hidden rows, and assignments made on rows that never had a match.
Decision to build the stage (pre-registered before the run): HUNG correct >= 1.5 x E31 correct and HUNG wrong rate
(wrong / (correct + wrong)) <= E31's.

    .venv/bin/python src/e139_assign_probe.py 2025-01-15 2025-02-15 ...
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import surface_mask  # noqa: E402

HIDE, UNK, WIN = float(os.environ.get("HIDE", "0.3")), float(os.environ.get("UNK", "45")), 90.0
BIG = 1e9
train = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"])
                   for f in sorted(__import__("glob").glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
tot = {k: 0 for k in ("hidden", "e31_ok", "e31_bad", "e31_none", "h_ok", "h_bad", "h_none", "never", "e31_never", "h_never")}
for day in sys.argv[1:]:
    y, m, d = map(int, day.split("-"))
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    dep = train.filter(pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d)).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    events: dict[str, list] = {}
    st_map, cand_map = {}, {}
    for (icao, ap), g in pts.group_by(["icao", "airport"]):
        t, surf = g["t"].to_numpy(), g["surf"].to_numpy()
        st_map[(icao, ap)] = np.sort(t[surf])
        for i in np.where(surf[:-1] & ~surf[1:] & (np.diff(t) <= 120))[0]:
            events.setdefault(ap, []).append((float(t[i]), icao))
    for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cand_map.setdefault((cs, ap), []).append(icao)
    rng = np.random.default_rng(int(y * 10000 + m * 100 + d))
    for ap, g in dep.group_by("ADEP_mvt"):
        ap = ap[0]
        ev = events.get(ap, [])
        if not ev:
            continue
        et, ei = np.array([e[0] for e in ev]), np.array([e[1] for e in ev])
        truth, unmatched = {}, []
        for mid, cs, to in g.select("MVT_ID_mvt", "cs", "to").iter_rows():
            ok = [ic for ic in cand_map.get((cs, ap), []) if (ts := st_map.get((ic, ap))) is not None
                  and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
            if len(ok) == 1:
                own = np.where(ei == ok[0])[0]
                if len(own):
                    truth[mid] = (to, ok[0], own[np.argmin(np.abs(et[own] - to))])
                    continue
            unmatched.append((mid, to))
        ids = list(truth)
        hidden = set(np.array(ids)[rng.random(len(ids)) < HIDE].tolist()) if ids else set()
        claimed = {truth[m][2] for m in ids if m not in hidden}
        # E31 rule
        e31 = {}
        for mid, to in [(m, truth[m][0]) for m in hidden] + unmatched:
            c = np.unique(ei[np.abs(et - to) <= WIN])
            e31[mid] = c[0] if len(c) == 1 else None
        # Hungarian
        flights = [(m, truth[m][0]) for m in hidden] + unmatched
        free = np.array([j for j in range(len(et)) if j not in claimed], dtype=int)
        hung = {m: None for m, _ in flights}
        if flights and len(free):
            to_arr = np.array([t for _, t in flights])
            dt = np.abs(to_arr[:, None] - et[free][None, :])
            cost = np.where(dt <= WIN, dt, BIG)
            cost = np.hstack([cost, np.where(np.eye(len(flights)) > 0, UNK, BIG)])
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if j < len(free) and cost[i, j] < BIG:
                    hung[flights[i][0]] = ei[free[j]]
        for m in hidden:
            tot["hidden"] += 1
            for key, res in (("e31", e31), ("h", hung)):
                v = res[m]
                tot[f"{key}_none" if v is None else (f"{key}_ok" if v == truth[m][1] else f"{key}_bad")] += 1
        for m, _ in unmatched:
            tot["never"] += 1
            tot["e31_never"] += e31[m] is not None
            tot["h_never"] += hung[m] is not None
    print(day, dict(tot), flush=True)

rate = lambda ok, bad: bad / max(ok + bad, 1)
print(f"E31: correct {tot['e31_ok']}, wrong {tot['e31_bad']} ({rate(tot['e31_ok'], tot['e31_bad']):.1%}), none {tot['e31_none']}; "
      f"assigned on never-matched {tot['e31_never']}/{tot['never']}")
print(f"HUNG: correct {tot['h_ok']}, wrong {tot['h_bad']} ({rate(tot['h_ok'], tot['h_bad']):.1%}), none {tot['h_none']}; "
      f"assigned on never-matched {tot['h_never']}/{tot['never']}")
ok = tot["h_ok"] >= 1.5 * tot["e31_ok"] and rate(tot["h_ok"], tot["h_bad"]) <= rate(tot["e31_ok"], tot["e31_bad"])
print(f"DECISION E139 probe: {'BUILD STAGE' if ok else 'CLOSE'} (hidden {tot['hidden']}, HIDE {HIDE}, UNK {UNK})")
