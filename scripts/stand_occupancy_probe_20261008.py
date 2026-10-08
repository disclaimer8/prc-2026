"""Stand-occupancy identification probe (win_program №8, pre-registered 08.10.2026, go/no-go only, no CV).

Question: for departures WITHOUT an own surface segment (kt1/offblock2 coverage < 2 — the rows E141 cannot serve),
can the aircraft be found by its STAND instead of its call sign: any icao24 that sat stationary within R m of the
departure's assigned stand during [T − 90 min, T − 2 min] (T = take-off, MVT_TIME_UTC) and then started moving?

Protocol (all fixed before the run):
  days 2026    2026-02-05, 2026-06-10, 2026-07-02 (Thu/Wed/Thu; 06-11 is truncated in the archive, avoided)
  days 2025    2025-02-06, 2025-06-11, 2025-07-03 (same months, same weekdays; labels exist)
  airports     EGLL, LEMD, LEBL, LSZH, EDDM (ground reception present, unmatched rows exist in 2026)
  pool         DEP rows of the day at those airports with kt1/offblock2 coverage < 2
               (2026: data/final_adsb_derivatives_20261008/data/adsblol_kt1, 2025: data/adsblol_kt1)
  stand        OSM stand coordinates as src/geometry.py resolves them, exact or suffix-stripped key only
               (geo_match 2 or 1); the prefix-centroid fallback is meaningless at 40 m and counts as "no stand"
  surface      src/adsb_offblock_v2.surface_mask (ground flag OR gs <= 40 kt and alt <= elev + 800 ft)
  dwell        stationary = gs <= 1 kt (MOVING_KT of the production kt1 products); a candidate icao24 needs >= 2
               stationary surface points within R = 40 m of the stand inside the window, spanning >= 90 s (STOP_S)
  movement     t_mv = first point of that icao24 after its last dwell point with gs > 1 kt and t <= T + 300 s;
               it must come within 900 s of the last dwell point (GAP_S), otherwise the movement time is unknown
  taken        an icao24 whose points carry the call sign of ANOTHER departure of the same airport-day is not a
               candidate; one carrying THIS departure's call sign is flagged same_cs (identity check)
  base         implied off-block = T − base taxi prediction.
               2025: E195 joint OOF (data/e195_pre_e81_20261007_v2/predictions.parquet, the chain before E81);
               2026 Jul: v12090 canary (submissions/gentle-igloo_v12090.parquet, Jan/Jul rows);
               2026 Feb/Jun: NM proxy T − AOBT_3 (no chain prediction exists for those months yet).
               NM proxy on 2025 is reported as a sensitivity so 2026 Feb/Jun numbers can be read against it.
  gate         |t_mv + LAG − implied off-block| <= 300 s, LAG = 0 s (1-kt first movement ≈ push-back start)
  accepted     exactly one icao24 passes the gate (unique); ">= 1" is reported alongside
  implied taxi T − t_mv + LAG  (compared with TAXITIME_SEC_mvt on 2025 rows)
Go (from the program): >= 20 % of the pool gets an accepted candidate AND on 2025 days the accepted candidates'
implied taxi-out has lower SSE than the chain (joint) on the same rows.

    .venv/bin/python scripts/stand_occupancy_probe_20261008.py --year 2026 [--surface-dir DIR] [--out DIR]
    .venv/bin/python scripts/stand_occupancy_probe_20261008.py --year 2025 --surface-dir <scratch with 2025 files>
    .venv/bin/python scripts/stand_occupancy_probe_20261008.py report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adsb_offblock_v2 import surface_mask  # noqa: E402
from geometry import load_tables, norm_stand, strip_suffix  # noqa: E402

PROTOCOL = "stand-occupancy-probe-R40-win5400-120-dwell90-gate300-lag0-v1"
DAYS = {2026: ["2026-02-05", "2026-06-10", "2026-07-02"], 2025: ["2025-02-06", "2025-06-11", "2025-07-03"]}
AIRPORTS = ["EGLL", "LEMD", "LEBL", "LSZH", "EDDM"]
R_M, WIN_BEFORE, WIN_END, STOP_S, GAP_S, MOVING_KT, GATE_S, LAG_S, AFTER_TO = 40.0, 5400.0, 120.0, 90.0, 900.0, 1.0, 300.0, 0.0, 300.0
MIN_DWELL_PTS = 2
D = "data/prc-2026-datasets/"
FINAL_RANKING = D + "final_ranking.parquet"
KT1 = {2026: "data/final_adsb_derivatives_20261008/data/adsblol_kt1", 2025: "data/adsblol_kt1"}
JOINT = "data/e195_pre_e81_20261007_v2/predictions.parquet"
V12090 = "submissions/gentle-igloo_v12090.parquet"
DEFAULT_OUT = "analysis/stand_occupancy_probe_20261008"


def dist_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(np.asarray(lat2) - np.asarray(lat1)), np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def stand_coords(dep: pl.DataFrame) -> pl.DataFrame:
    """Exact or suffix-stripped OSM stand match (geo_match 2/1 of src/geometry.add_geometry); no prefix fallback."""
    st, _ = load_tables()
    st_nosfx = (st.with_columns(stand_key=pl.col("stand_key").map_elements(strip_suffix, return_dtype=pl.String))
                .unique(["ADEP_mvt", "stand_key"], keep="first").rename({"stand_lat": "lat2", "stand_lon": "lon2", "stand_key": "key_nosfx"}))
    keys = dep.select("MVT_ID_mvt", "ADEP_mvt", stand_key=pl.col("STAND_mvt").map_elements(norm_stand, return_dtype=pl.String))
    keys = keys.with_columns(key_nosfx=pl.col("stand_key").map_elements(strip_suffix, return_dtype=pl.String))
    g = (keys.join(st, on=["ADEP_mvt", "stand_key"], how="left").join(st_nosfx, on=["ADEP_mvt", "key_nosfx"], how="left")
         .with_columns(geo_match=pl.when(pl.col("stand_lat").is_not_null()).then(2).when(pl.col("lat2").is_not_null()).then(1).otherwise(None),
                       slat=pl.coalesce("stand_lat", "lat2"), slon=pl.coalesce("stand_lon", "lon2")))
    return g.select("MVT_ID_mvt", "geo_match", "slat", "slon")


def departures(day: str) -> pl.DataFrame:
    y, m, d = map(int, day.split("-"))
    if y == 2026:
        src = pl.scan_parquet(FINAL_RANKING)
    else:
        nxt = (y, m + 1) if m < 12 else (y + 1, 1)
        src = pl.scan_parquet(D + f"training_{y}-{m:02d}-01_{nxt[0]}-{nxt[1]:02d}-01.parquet")
    dep = (src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d)))
           .select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt")
           .collect())
    return dep.with_columns(cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
                            T=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
                            nm_taxi=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))


def base_taxi(dep: pl.DataFrame, year: int) -> pl.DataFrame:
    """base_taxi + base_src per row; joint (2025) / v12090 (2026 Jan-Jul) / NM proxy (2026 Feb-Jun, or missing)."""
    if year == 2025:
        j = pl.read_parquet(JOINT, columns=["MVT_ID_mvt", "joint"]).with_columns(pl.col("joint").cast(pl.Float64))
        dep = dep.join(j, on="MVT_ID_mvt", how="left").with_columns(
            base_taxi=pl.col("joint"), base_src=pl.when(pl.col("joint").is_not_null()).then(pl.lit("joint")).otherwise(pl.lit("none")))
    else:
        v = pl.read_parquet(V12090).rename({"TAXITIME_SEC_mvt": "v12090"}).with_columns(pl.col("v12090").cast(pl.Float64))
        dep = dep.join(v, on="MVT_ID_mvt", how="left").with_columns(
            base_taxi=pl.coalesce("v12090", "nm_taxi"),
            base_src=pl.when(pl.col("v12090").is_not_null()).then(pl.lit("v12090")).when(pl.col("nm_taxi").is_not_null()).then(pl.lit("nm_proxy")).otherwise(pl.lit("none")))
    return dep


def run_day(day: str, year: int, surface_dir: str) -> tuple[pl.DataFrame, dict]:
    t0 = time.time()
    dep = departures(day).filter(pl.col("ADEP_mvt").is_in(AIRPORTS))
    kt1 = pl.read_parquet(f"{KT1[year]}/offblock2_{day}.parquet", columns=["MVT_ID_mvt", "coverage"])
    dep = dep.join(kt1, on="MVT_ID_mvt", how="left")
    if dep["coverage"].null_count():
        raise ValueError(f"{day}: {dep['coverage'].null_count()} departures without a kt1 product row")
    dep = base_taxi(dep.join(stand_coords(dep), on="MVT_ID_mvt", how="left"), year)
    pool = dep.filter(pl.col("coverage") < 2)
    # call signs of every departure of the airport-day (for the 'taken' rule), by airport
    pts = pl.read_parquet(f"{surface_dir}/surface_{day}.parquet").filter(pl.col("airport").is_in(AIRPORTS))
    pts = pts.with_columns(surf=surface_mask(pts)).sort("icao", "t")
    recs = []
    for ap in AIRPORTS:
        p = pts.filter(pl.col("airport") == ap)
        dep_ap = dep.filter(pl.col("ADEP_mvt") == ap)
        pool_ap = pool.filter(pl.col("ADEP_mvt") == ap)
        # call sign -> departure ids (airport-day)
        cs_map: dict[str, set] = {}
        for cs, mid in dep_ap.select("cs", "MVT_ID_mvt").iter_rows():
            if cs:
                cs_map.setdefault(cs, set()).add(mid)
        # per icao: arrays + call signs seen
        tracks: dict[str, dict] = {}
        for (icao,), g in p.group_by(["icao"]):
            tracks[icao] = {"t": g["t"].to_numpy(), "gs": np.nan_to_num(g["gs"].to_numpy().astype(float)),
                            "cs": {c.replace(" ", "") for c in g["callsign"].drop_nulls().unique().to_list()}}
        # dwell candidates: stationary surface points
        s = p.filter(pl.col("surf") & (pl.col("gs").fill_null(0) <= MOVING_KT)).sort("t")
        st, slat, slon, sic = s["t"].to_numpy(), s["lat"].to_numpy(), s["lon"].to_numpy(), s["icao"].to_numpy()
        for row in pool_ap.iter_rows(named=True):
            T = row["T"]
            rec = {"day": day, "year": year, "airport": ap, "MVT_ID_mvt": row["MVT_ID_mvt"], "T": T, "coverage": row["coverage"],
                   "geo_match": row["geo_match"], "base_src": row["base_src"], "base_taxi": row["base_taxi"], "nm_taxi": row["nm_taxi"],
                   "label": row["TAXITIME_SEC_mvt"], "n_dwell": 0, "n_mv": 0, "n_taken": 0, "n_gate": 0, "icao": None, "t_mv": None,
                   "implied_taxi": None, "same_cs": None, "dwell_span": None, "n_gate_nm": 0, "icao_nm": None, "t_mv_nm": None}
            if row["geo_match"] is None:
                recs.append(rec)
                continue
            lo, hi = np.searchsorted(st, T - WIN_BEFORE), np.searchsorted(st, T - WIN_END, side="right")
            if hi <= lo:
                recs.append(rec)
                continue
            d = dist_m(slat[lo:hi], slon[lo:hi], row["slat"], row["slon"])
            near = d <= R_M
            if not near.any():
                recs.append(rec)
                continue
            cands = []
            for icao in np.unique(sic[lo:hi][near]):
                sel = near & (sic[lo:hi] == icao)
                tt = st[lo:hi][sel]
                if len(tt) < MIN_DWELL_PTS or tt[-1] - tt[0] < STOP_S:
                    continue
                rec["n_dwell"] += 1
                tr = tracks[icao]
                t_last = tt[-1]
                k = np.searchsorted(tr["t"], t_last, side="right")
                mv = np.where(tr["gs"][k:] > MOVING_KT)[0]
                if len(mv) == 0:
                    continue
                t_mv = float(tr["t"][k + mv[0]])
                if t_mv > T + AFTER_TO or t_mv - t_last > GAP_S:
                    continue
                rec["n_mv"] += 1
                others = any(cs_map.get(c, set()) - {row["MVT_ID_mvt"]} for c in tr["cs"])
                same = row["cs"] in tr["cs"]
                if others and not same:
                    rec["n_taken"] += 1
                    continue
                cands.append((icao, t_mv, same, float(tt[-1] - tt[0])))
            if row["base_taxi"] is not None:
                ob = T - row["base_taxi"]
                ok = [c for c in cands if abs(c[1] + LAG_S - ob) <= GATE_S]
                rec["n_gate"] = len(ok)
                if len(ok) == 1:
                    icao, t_mv, same, span = ok[0]
                    rec.update(icao=icao, t_mv=t_mv, implied_taxi=T - t_mv + LAG_S, same_cs=same, dwell_span=span)
            if row["nm_taxi"] is not None:  # sensitivity: NM-proxy gate
                ob = T - row["nm_taxi"]
                ok = [c for c in cands if abs(c[1] + LAG_S - ob) <= GATE_S]
                rec["n_gate_nm"] = len(ok)
                if len(ok) == 1:
                    rec.update(icao_nm=ok[0][0], t_mv_nm=ok[0][1])
            recs.append(rec)
    out = pl.DataFrame(recs, schema={"day": pl.String, "year": pl.Int32, "airport": pl.String, "MVT_ID_mvt": pl.Float64, "T": pl.Float64,
                                     "coverage": pl.Int64, "geo_match": pl.Int64, "base_src": pl.String, "base_taxi": pl.Float64, "nm_taxi": pl.Float64,
                                     "label": pl.Int32, "n_dwell": pl.Int32, "n_mv": pl.Int32, "n_taken": pl.Int32, "n_gate": pl.Int32,
                                     "icao": pl.String, "t_mv": pl.Float64, "implied_taxi": pl.Float64, "same_cs": pl.Boolean, "dwell_span": pl.Float64,
                                     "n_gate_nm": pl.Int32, "icao_nm": pl.String, "t_mv_nm": pl.Float64})
    receipt = {"day": day, "departures_at_airports": dep.height, "pool": pool.height, "seconds": round(time.time() - t0, 1),
               "surface_points": pts.height, "per_airport_dep": dep.group_by("ADEP_mvt").len().sort("ADEP_mvt").to_dict(as_series=False)}
    return out, receipt


def report(out_dir: Path) -> str:
    files = sorted(out_dir.glob("rows_*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = df.with_columns(month=pl.col("day").str.slice(0, 7), acc=pl.col("icao").is_not_null(), any_c=pl.col("n_gate") >= 1,
                         acc_nm=pl.col("icao_nm").is_not_null())
    lines = [f"# Stand-occupancy probe — {PROTOCOL}", "", f"Rows files: {[f.name for f in files]}", ""]
    agg = (df.group_by("month", "airport").agg(pool=pl.len(), with_stand=pl.col("geo_match").is_not_null().sum(),
                                                 with_base=pl.col("base_taxi").is_not_null().sum(), dwell=(pl.col("n_dwell") >= 1).sum(),
                                                 moved=(pl.col("n_mv") >= 1).sum(), any_cand=pl.col("any_c").sum(), unique=pl.col("acc").sum(),
                                                 unique_nm=pl.col("acc_nm").sum(), same_cs=pl.col("same_cs").fill_null(False).sum())
           .sort("month", "airport"))
    lines += ["## Pool and candidates per airport-month (one day each)", "",
              "| month | airport | pool (cov<2) | stand coords | base | dwell≥1 | moved≥1 | ≥1 gated | unique (accepted) | unique % of pool | unique (NM-proxy gate) | same call sign |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg.iter_rows(named=True):
        lines.append(f"| {r['month']} | {r['airport']} | {r['pool']} | {r['with_stand']} | {r['with_base']} | {r['dwell']} | {r['moved']} | {r['any_cand']} | {r['unique']} | {100*r['unique']/max(r['pool'],1):.1f} | {r['unique_nm']} | {r['same_cs']} |")
    tot = df.group_by("year").agg(pool=pl.len(), with_stand=pl.col("geo_match").is_not_null().sum(), any_cand=pl.col("any_c").sum(),
                                  unique=pl.col("acc").sum(), unique_nm=pl.col("acc_nm").sum(), same_cs=pl.col("same_cs").fill_null(False).sum()).sort("year")
    lines += ["", "## Totals per year", "", "| year | pool | stand coords | ≥1 gated | unique | unique % | unique (NM gate) | same call sign |", "|---|---|---|---|---|---|---|---|"]
    for r in tot.iter_rows(named=True):
        lines.append(f"| {r['year']} | {r['pool']} | {r['with_stand']} | {r['any_cand']} | {r['unique']} | {100*r['unique']/max(r['pool'],1):.1f} | {r['unique_nm']} | {r['same_cs']} |")
    summary = {"totals": tot.to_dicts(), "per_airport_month": agg.to_dicts()}
    a = df.filter(pl.col("acc") & (pl.col("year") == 2025) & pl.col("label").is_not_null())
    if a.height:
        e_c = (a["implied_taxi"] - a["label"]).to_numpy()
        e_b = (a["base_taxi"] - a["label"]).to_numpy()
        half = 0.5 * (a["implied_taxi"] + a["base_taxi"]).to_numpy() - a["label"].to_numpy()
        acc25 = {"n": int(a.height), "cand_mae_median": float(np.median(np.abs(e_c))), "cand_within_60s": float(np.mean(np.abs(e_c) <= 60)),
                 "cand_bias_median": float(np.median(e_c)), "cand_rmse": float(np.sqrt(np.mean(e_c ** 2))), "chain_rmse": float(np.sqrt(np.mean(e_b ** 2))),
                 "chain_mae_median": float(np.median(np.abs(e_b))), "chain_within_60s": float(np.mean(np.abs(e_b) <= 60)),
                 "cand_sse": float(np.sum(e_c ** 2)), "chain_sse": float(np.sum(e_b ** 2)), "half_blend_sse": float(np.sum(half ** 2)),
                 "cand_beats_chain_rows": float(np.mean(np.abs(e_c) < np.abs(e_b))), "same_cs_n": int(a["same_cs"].fill_null(False).sum())}
        per_ap = []
        for (ap,), g in a.group_by(["airport"]):
            ec, eb = (g["implied_taxi"] - g["label"]).to_numpy(), (g["base_taxi"] - g["label"]).to_numpy()
            per_ap.append({"airport": ap, "n": g.height, "cand_mae_median": float(np.median(np.abs(ec))), "chain_mae_median": float(np.median(np.abs(eb))),
                           "cand_sse": float(np.sum(ec ** 2)), "chain_sse": float(np.sum(eb ** 2)), "cand_within_60s": float(np.mean(np.abs(ec) <= 60))})
        acc25["per_airport"] = sorted(per_ap, key=lambda r: r["airport"])
        s = a.filter(pl.col("same_cs"))
        if s.height:
            ec = (s["implied_taxi"] - s["label"]).to_numpy(); eb = (s["base_taxi"] - s["label"]).to_numpy()
            acc25["same_cs_subset"] = {"n": s.height, "cand_mae_median": float(np.median(np.abs(ec))), "cand_sse": float(np.sum(ec ** 2)), "chain_sse": float(np.sum(eb ** 2))}
        summary["accuracy_2025"] = acc25
        lines += ["", "## 2025 accepted candidates against the label", "",
                  f"n = {acc25['n']}; candidate implied taxi: median |err| {acc25['cand_mae_median']:.0f} s, within 60 s {100*acc25['cand_within_60s']:.1f} %, "
                  f"median signed {acc25['cand_bias_median']:+.0f} s, RMSE {acc25['cand_rmse']:.1f}; chain (joint) on the same rows: median |err| {acc25['chain_mae_median']:.0f} s, "
                  f"within 60 s {100*acc25['chain_within_60s']:.1f} %, RMSE {acc25['chain_rmse']:.1f}. SSE candidate {acc25['cand_sse']:.3e} vs chain {acc25['chain_sse']:.3e} "
                  f"(½-blend {acc25['half_blend_sse']:.3e}); candidate closer on {100*acc25['cand_beats_chain_rows']:.1f} % of rows; same call sign on {acc25['same_cs_n']} rows.", "",
                  "| airport | n | cand median abs err | chain median abs err | cand within 60 s | cand SSE | chain SSE |", "|---|---|---|---|---|---|---|"]
        for r in acc25["per_airport"]:
            lines.append(f"| {r['airport']} | {r['n']} | {r['cand_mae_median']:.0f} | {r['chain_mae_median']:.0f} | {100*r['cand_within_60s']:.1f} % | {r['cand_sse']:.3e} | {r['chain_sse']:.3e} |")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("--year", type=int, choices=[2025, 2026])
    ap.add_argument("--surface-dir", default="data/adsblol")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "report":
        print(report(out))
        return
    if a.year is None:
        ap.error("--year required for run")
    for day in DAYS[a.year]:
        rows, rc = run_day(day, a.year, a.surface_dir)
        rows.write_parquet(out / f"rows_{day}.parquet")
        (out / f"receipt_{day}.json").write_text(json.dumps(rc, indent=1) + "\n")
        print(json.dumps(rc), flush=True)


if __name__ == "__main__":
    main()
