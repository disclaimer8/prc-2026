#!/usr/bin/env python
"""Label-free audit of the final-month inputs against 2025 training coverage (journal 437).

Reads the sealed FIN1b feature workspace (split ``train`` = 2025, ``rank`` = the four final
months) and the final ranking file, and answers, per airport x final month:

1. runway usage: share of final departures whose (airport, runway) or (airport, runway
   configuration) has fewer than 100 departures in 2025 (all year / same calendar month),
   or never appeared;
2. context: share of final rows whose traffic, destination-ATFM and METAR features fall
   outside the 2025 1st-99th percentile band of the same airport (per feature and any-of),
   plus METAR flag rates against 2025;
3. NM coverage: share of departures without an NM record (LOBT null);
4. named episodes: LTFM 19-23 Feb (the journal's snow episode is 19-23 Feb **2025**, i.e. in
   training; the same 2026 window and the actual 2026 snow days are both shown), EHAM taxiway
   works (June 2026 up to 26 June), EDDF Feb 2026 (ADS-B collapse; not visible in these features).

No model, no fitting, no upload. Rewrites the generated part (below the ``<!-- GENERATED
TABLES -->`` marker) of ``analysis/final_months_input_audit_20261009.md``; the summary above
the marker is written by hand. ``--json PATH`` additionally dumps the numbers.

Run on the Mac:
    POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage POLARS_MAX_THREADS=4 \
      .venv/bin/python scripts/final_months_input_audit_20261009.py [--json PATH]
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIN1B = ROOT / "data/final_feature_work_mac_20261008/data/features_fin1b.parquet"
RANKING = ROOT / "data/prc-2026-datasets/final_ranking.parquet"
OUT_MD = ROOT / "analysis/final_months_input_audit_20261009.md"
OUT_JSON = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None

FINAL_MONTHS = [1, 2, 6, 7]
MIN_TRAIN = 100

# Context features tested against the 2025 P1-P99 band of the same airport.
TRAFFIC = [
    "dep_before_60", "dep_after_60", "arr_before_60", "offblocks_around_ob_60",
    "taxiing_at_ob", "takeoffs_between_ob_mvt", "day_movements",
]
DELAY = ["dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr"]
WX_NUM = ["wx_temp", "wx_wind", "wx_gust", "wx_vis_mi", "wx_ceiling_ft", "wxr_headwind", "wxr_crosswind"]
WX_FLAGS = ["wx_snow", "wx_freezing", "wx_rain", "wx_thunder", "wx_fog", "wx_icing", "wx_snow_recent", "wx_icing_recent"]
NUMERIC = TRAFFIC + DELAY + WX_NUM

KEEP = [
    "MVT_ID_mvt", "split", "airport", "month", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "ap_runway",
    "rwy_config", "LOBT_flt", "dst_known", "metar_age_min",
] + NUMERIC + WX_FLAGS


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---:" if i else "---" for i, _ in enumerate(cols)) + "|"
    body = ["| " + " | ".join(str(r[c[0]]) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main() -> int:
    t0 = dt.datetime.now(dt.timezone.utc)
    df = pl.scan_parquet(FIN1B).select(KEEP).collect()
    df = df.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("year"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
        pl.col("LOBT_flt").is_null().alias("no_nm"),
        pl.col("RUNWAY_mvt").fill_null("<null>").alias("rwy"),
        pl.col("rwy_config").fill_null("<null>").alias("cfg"),
    )
    train = df.filter(pl.col("split") == "train")
    rank = df.filter(pl.col("split") == "rank")
    assert train["year"].unique().to_list() == [2025], train["year"].unique().to_list()
    assert rank["year"].unique().to_list() == [2026]
    assert sorted(rank["month"].unique().to_list()) == FINAL_MONTHS

    # Cross-check against the ranking file: same ids, same NM-null share per airport-month.
    rk_all = pl.read_parquet(RANKING, columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"])
    n_all_mvt = rk_all.height  # departures and arrivals
    rk = rk_all.filter(pl.col("PHASE_mvt") == "DEP")
    assert rk.height == rank.height == 670_790, (n_all_mvt, rk.height, rank.height)
    # Reconciliation of the EHAM June count quoted in docs (23,812 / 20,753): ADEP == EHAM over BOTH phases.
    eham_jun_any = rk_all.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("MVT_TIME_UTC_mvt").dt.month() == 6))
    eham_recon = {
        "dep_rows_june": rk.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("MVT_TIME_UTC_mvt").dt.month() == 6)).height,
        "dep_rows_1_26_june": rk.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("MVT_TIME_UTC_mvt").dt.date().is_between(dt.date(2026, 6, 1), dt.date(2026, 6, 26)))).height,
        "adep_eham_any_phase_june": eham_jun_any.height,
        "adep_eham_any_phase_1_26_june": eham_jun_any.filter(pl.col("MVT_TIME_UTC_mvt").dt.date() <= dt.date(2026, 6, 26)).height,
        "adep_eham_arr_phase_june": eham_jun_any.filter(pl.col("PHASE_mvt") == "ARR").height,
    }
    assert set(rk["MVT_ID_mvt"].to_list()) == set(rank["MVT_ID_mvt"].to_list())
    rk_nonm = (
        rk.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"), pl.col("LOBT_flt").is_null().alias("no_nm"))
        .group_by("ADEP_mvt", "month").agg(pl.col("no_nm").mean()).sort("ADEP_mvt", "month")
    )
    fb_nonm = rank.group_by("airport", "month").agg(pl.col("no_nm").mean()).sort("airport", "month")
    assert (rk_nonm["no_nm"] - fb_nonm["no_nm"]).abs().max() < 1e-12

    out: dict = {"generated_utc": t0.isoformat(), "rows_train": train.height, "rows_rank": rank.height, "eham_june_reconciliation": eham_recon}
    md: list[str] = []

    # ---------------------------------------------------------------- 1. runways
    def unseen_share(key: str) -> pl.DataFrame:
        all25 = train.group_by("airport", key).len().rename({"len": "n25_all"})
        mon25 = train.group_by("airport", "month", key).len().rename({"len": "n25_mon"})
        r = (
            rank.join(all25, on=["airport", key], how="left")
            .join(mon25, on=["airport", "month", key], how="left")
            .with_columns(pl.col("n25_all").fill_null(0), pl.col("n25_mon").fill_null(0))
        )
        return r.group_by("airport", "month").agg(
            pl.len().alias("n"),
            (pl.col("n25_all") == 0).mean().alias("never_all"),
            (pl.col("n25_all") < MIN_TRAIN).mean().alias("lt100_all"),
            (pl.col("n25_mon") == 0).mean().alias("never_mon"),
            (pl.col("n25_mon") < MIN_TRAIN).mean().alias("lt100_mon"),
            pl.col(key).n_unique().alias("n_cat"),
        ).sort("airport", "month")

    rw = unseen_share("rwy")
    cf = unseen_share("cfg")
    # Which new runway labels appear (never in 2025, any count) - for the text.
    new_rwy = (
        rank.join(train.select("airport", "rwy").unique().with_columns(pl.lit(True).alias("seen")), on=["airport", "rwy"], how="left")
        .filter(pl.col("seen").is_null()).group_by("airport", "month", "rwy").len().sort("airport", "month", "rwy")
    )
    out["runway_unseen"] = rw.to_dicts()
    out["config_unseen"] = cf.to_dicts()
    out["new_runway_labels"] = new_rwy.to_dicts()

    md.append("## 1. Runway and runway-configuration usage against 2025\n")
    md.append(
        f"Share of final departures whose (airport, `RUNWAY_mvt`) or (airport, `rwy_config`) combination has "
        f"fewer than {MIN_TRAIN} departures in 2025 (`<100`) or none at all (`never`); `all` = whole of 2025, "
        f"`same mo` = same calendar month of 2025. `cat` = distinct values used in the final month.\n"
    )
    rows = []
    for a, b in zip(rw.to_dicts(), cf.to_dicts()):
        assert (a["airport"], a["month"]) == (b["airport"], b["month"])
        rows.append({
            "ap": a["airport"], "mo": f"2026-{a['month']:02d}", "n": f"{a['n']:,}",
            "r_cat": a["n_cat"], "r_never": pct(a["never_all"]), "r_lt_all": pct(a["lt100_all"]), "r_lt_mon": pct(a["lt100_mon"]),
            "c_cat": b["n_cat"], "c_never": pct(b["never_all"]), "c_lt_all": pct(b["lt100_all"]), "c_lt_mon": pct(b["lt100_mon"]),
        })
    md.append(md_table(rows, [
        ("ap", "Airport"), ("mo", "Month"), ("n", "Departures"),
        ("r_cat", "Rwy cat"), ("r_never", "Rwy never"), ("r_lt_all", "Rwy <100 all"), ("r_lt_mon", "Rwy <100 same mo"),
        ("c_cat", "Cfg cat"), ("c_never", "Cfg never"), ("c_lt_all", "Cfg <100 all"), ("c_lt_mon", "Cfg <100 same mo"),
    ]))
    if new_rwy.height:
        md.append("\nRunway labels never used in 2025 at that airport:\n")
        md.append(md_table(
            [{"ap": r["airport"], "mo": f"2026-{r['month']:02d}", "rwy": f"`{r['rwy']}`", "n": r["len"]} for r in new_rwy.to_dicts()],
            [("ap", "Airport"), ("mo", "Month"), ("rwy", "Runway"), ("n", "Departures")],
        ))
    else:
        md.append("\nNo runway label in the final months is absent from 2025 at the same airport.")
    md.append("")

    # ---------------------------------------------------------------- 2. context
    md.append("## 2. Traffic, destination-delay and METAR context against the 2025 P1-P99 band\n")
    md.append(
        "For each airport and feature the 1st and 99th percentiles are computed on 2025 (all year, and separately "
        "on the same calendar month). `outside` = share of final rows strictly below P1 or above P99. The 2025 "
        "self-rate (share of 2025 rows outside their own band; at most 2 %, less for discrete counts because of "
        "ties) is the reference. `any-of` = outside on at least one of the "
        f"{len(NUMERIC)} numeric features. Nulls are not counted as outside.\n"
    )
    md.append("Features: traffic " + ", ".join(f"`{c}`" for c in TRAFFIC) + "; destination ATFM "
              + ", ".join(f"`{c}`" for c in DELAY) + "; METAR " + ", ".join(f"`{c}`" for c in WX_NUM) + ".\n")

    q_all = train.group_by("airport").agg(
        [pl.col(c).quantile(0.01, "linear").alias(f"{c}__lo") for c in NUMERIC]
        + [pl.col(c).quantile(0.99, "linear").alias(f"{c}__hi") for c in NUMERIC]
    )
    q_mon = train.group_by("airport", "month").agg(
        [pl.col(c).quantile(0.01, "linear").alias(f"{c}__lo_m") for c in NUMERIC]
        + [pl.col(c).quantile(0.99, "linear").alias(f"{c}__hi_m") for c in NUMERIC]
    )

    def flag_outside(d: pl.DataFrame, suffix: str) -> pl.DataFrame:
        exprs = []
        for c in NUMERIC:
            lo, hi = f"{c}__lo{suffix}", f"{c}__hi{suffix}"
            exprs.append(((pl.col(c) < pl.col(lo)) | (pl.col(c) > pl.col(hi))).fill_null(False).alias(f"out__{c}"))
        d = d.with_columns(exprs)
        return d.with_columns(pl.any_horizontal([f"out__{c}" for c in NUMERIC]).alias("out__any"))

    r_all = flag_outside(rank.join(q_all, on="airport", how="left"), "")
    r_mon = flag_outside(rank.join(q_mon, on=["airport", "month"], how="left"), "_m")
    t_all = flag_outside(train.join(q_all, on="airport", how="left"), "")
    t_mon = flag_outside(train.join(q_mon, on=["airport", "month"], how="left"), "_m")

    def agg_out(d: pl.DataFrame, by: list[str]) -> pl.DataFrame:
        return d.group_by(by).agg([pl.len().alias("n")] + [pl.col(f"out__{c}").mean() for c in NUMERIC] + [pl.col("out__any").mean()]).sort(by)

    o_r_all = agg_out(r_all, ["airport", "month"])
    o_r_mon = agg_out(r_mon, ["airport", "month"])
    o_t_all = agg_out(t_all, ["airport"])
    o_t_mon = agg_out(t_mon.filter(pl.col("month").is_in(FINAL_MONTHS)), ["airport", "month"])
    out["context_outside_all2025"] = o_r_all.to_dicts()
    out["context_outside_samemonth"] = o_r_mon.to_dicts()
    out["context_selfrate_all2025"] = o_t_all.to_dicts()
    out["context_selfrate_samemonth"] = o_t_mon.to_dicts()

    # Summary table: any-of and the three groups' max per airport-month.
    def grp_max(row: dict, cols: list[str]) -> tuple[str, float]:
        best = max(cols, key=lambda c: row[f"out__{c}"])
        return best, row[f"out__{best}"]

    md.append("### 2a. Any-of share per airport x month\n")
    rows = []
    self_all = {r["airport"]: r for r in o_t_all.to_dicts()}
    self_mon = {(r["airport"], r["month"]): r for r in o_t_mon.to_dicts()}
    for a, b in zip(o_r_all.to_dicts(), o_r_mon.to_dicts()):
        sa, sm = self_all[a["airport"]], self_mon[(a["airport"], a["month"])]
        bt, bv = grp_max(a, TRAFFIC); bd, dv = grp_max(a, DELAY); bw, wv = grp_max(a, WX_NUM)
        rows.append({
            "ap": a["airport"], "mo": f"2026-{a['month']:02d}", "n": f"{a['n']:,}",
            "any_all": pct(a["out__any"]), "self_all": pct(sa["out__any"]),
            "any_mon": pct(b["out__any"]), "self_mon": pct(sm["out__any"]),
            "traf": f"{pct(bv)} `{bt}`", "dly": f"{pct(dv)} `{bd}`", "wx": f"{pct(wv)} `{bw}`",
        })
    md.append(md_table(rows, [
        ("ap", "Airport"), ("mo", "Month"), ("n", "Rows"),
        ("any_all", "any-of vs all 2025"), ("self_all", "2025 self-rate"),
        ("any_mon", "any-of vs same month 2025"), ("self_mon", "2025 same-month self-rate"),
        ("traf", "worst traffic feature (vs all 2025)"), ("dly", "worst delay feature"), ("wx", "worst METAR feature"),
    ]))
    md.append("")

    # Per-feature table: features whose outside share is at least 5 % and more than twice the self-rate.
    md.append("### 2b. Per-feature outside share against all of 2025 (airport-months where share >= 5 % and > 2 x self-rate)\n")
    rows = []
    for a in o_r_all.to_dicts():
        sa = self_all[a["airport"]]
        for c in NUMERIC:
            v, s = a[f"out__{c}"], sa[f"out__{c}"]
            if v >= 0.05 and v > 2 * s:
                side = r_all.filter((pl.col("airport") == a["airport"]) & (pl.col("month") == a["month"]) & pl.col(f"out__{c}"))
                lo = side.filter(pl.col(c) < pl.col(f"{c}__lo")).height
                hi = side.height - lo
                rows.append({"ap": a["airport"], "mo": f"2026-{a['month']:02d}", "f": f"`{c}`", "v": pct(v), "s": pct(s), "lo": lo, "hi": hi})
    rows.sort(key=lambda r: -float(r["v"].split()[0]))
    md.append(md_table(rows, [("ap", "Airport"), ("mo", "Month"), ("f", "Feature"), ("v", "outside"), ("s", "2025 self-rate"), ("lo", "rows below P1"), ("hi", "rows above P99")]) if rows else "None.")
    md.append("")
    out["context_per_feature_exceed"] = rows

    # Drivers of the two dominant out-of-range features: daily movement count and temperature.
    md.append("### 2b'. What drives the out-of-range rows: daily movements and temperature, 2026 against the same month of 2025\n")
    md.append("`day_movements` = departures + arrivals of the airport on the local day (`src/features.py`); the 2025 P99 of the airport "
              "is a few per cent above the 2025 median, so a traffic change of that size moves most of a month outside the band. "
              "`dep/day` = final departure rows per calendar day. Temperature in degrees C.\n")
    drv = df.filter(pl.col("month").is_in(FINAL_MONTHS)).group_by("split", "airport", "month").agg(
        pl.col("day_movements").median().alias("dm_med"), pl.col("day_movements").max().alias("dm_max"),
        (pl.len() / pl.col("day").n_unique()).alias("dep_day"),
        pl.col("wx_temp").median().alias("t_med"), pl.col("wx_temp").quantile(0.99, "linear").alias("t_p99"),
    )
    dm_all = train.group_by("airport").agg(pl.col("day_movements").quantile(0.99, "linear").alias("dm_p99"), pl.col("day_movements").max().alias("dm_max25"),
                                           pl.col("wx_temp").quantile(0.99, "linear").alias("t_p99_25"), pl.col("wx_temp").quantile(0.01, "linear").alias("t_p01_25"))
    dm_all_d = {r["airport"]: r for r in dm_all.to_dicts()}
    drv_d = {(r["split"], r["airport"], r["month"]): r for r in drv.to_dicts()}
    rows = []
    for ap in sorted({r["airport"] for r in drv.to_dicts()}):
        for mo in FINAL_MONTHS:
            r26, r25, ra = drv_d[("rank", ap, mo)], drv_d[("train", ap, mo)], dm_all_d[ap]
            rows.append({
                "ap": ap, "mo": f"2026-{mo:02d}",
                "dm": f"{r26['dm_med']:.0f} / {r25['dm_med']:.0f} ({100 * (r26['dm_med'] / r25['dm_med'] - 1):+.1f} %)",
                "dmx": f"{r26['dm_max']:.0f} / {ra['dm_p99']:.0f} / {ra['dm_max25']:.0f}",
                "dd": f"{r26['dep_day']:.0f} / {r25['dep_day']:.0f} ({100 * (r26['dep_day'] / r25['dep_day'] - 1):+.1f} %)",
                "t": f"{r26['t_med']:.0f} / {r25['t_med']:.0f}",
                "tp": f"{r26['t_p99']:.0f} / {r25['t_p99']:.0f} / {ra['t_p99_25']:.0f}",
            })
    md.append(md_table(rows, [("ap", "Airport"), ("mo", "Month"), ("dm", "day_movements median 2026 / 2025"), ("dmx", "max 2026 / P99 all 2025 / max 2025"),
                              ("dd", "dep/day 2026 / 2025"), ("t", "temp median 2026 / 2025"), ("tp", "temp P99 2026 / same month 2025 / all 2025")]))
    md.append("")
    out["drivers"] = rows

    # METAR flags: rate in final vs 2025 same month vs all 2025, and count of 2025 rows with flag=1 at the airport.
    md.append("### 2c. METAR flags: share of rows with the flag set (final month / same month 2025 / all 2025) and 2025 support\n")
    fr = rank.group_by("airport", "month").agg([pl.len().alias("n")] + [pl.col(c).cast(pl.Float64).mean().alias(c) for c in WX_FLAGS]).sort("airport", "month")
    fm = train.group_by("airport", "month").agg([pl.col(c).cast(pl.Float64).mean().alias(c) for c in WX_FLAGS])
    fa = train.group_by("airport").agg([pl.col(c).cast(pl.Float64).mean().alias(c) for c in WX_FLAGS] + [pl.col(c).cast(pl.Int64).sum().alias(f"{c}__n") for c in WX_FLAGS])
    fm_d = {(r["airport"], r["month"]): r for r in fm.to_dicts()}
    fa_d = {r["airport"]: r for r in fa.to_dicts()}
    rows = []
    for r in fr.to_dicts():
        m, a = fm_d[(r["airport"], r["month"])], fa_d[r["airport"]]
        row = {"ap": r["airport"], "mo": f"2026-{r['month']:02d}"}
        for c in WX_FLAGS:
            row[c] = f"{100*r[c]:.1f} / {100*m[c]:.1f} / {100*a[c]:.1f} (n25={a[f'{c}__n']:,})"
        rows.append(row)
    md.append(md_table(rows, [("ap", "Airport"), ("mo", "Month")] + [(c, f"`{c}`") for c in WX_FLAGS]))
    md.append("")
    out["metar_flags"] = {"rank": fr.to_dicts(), "train_month": fm.to_dicts(), "train_all": fa.to_dicts()}

    # Flag combinations unsupported in 2025 at the airport (< 100 rows with that exact flag vector).
    flag_vec = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in WX_FLAGS], separator="").alias("wxvec")
    tv = train.with_columns(flag_vec).group_by("airport", "wxvec").len().rename({"len": "n25"})
    rv = rank.with_columns(flag_vec).join(tv, on=["airport", "wxvec"], how="left").with_columns(pl.col("n25").fill_null(0))
    rvs = rv.group_by("airport", "month").agg(pl.len().alias("n"), (pl.col("n25") == 0).mean().alias("never"), (pl.col("n25") < MIN_TRAIN).mean().alias("lt100")).sort("airport", "month")
    out["metar_flag_vector_unseen"] = rvs.to_dicts()
    md.append(f"Share of final rows whose exact METAR flag vector (8 flags) has < {MIN_TRAIN} / 0 rows in 2025 at the airport:\n")
    md.append(md_table([{"ap": r["airport"], "mo": f"2026-{r['month']:02d}", "lt": pct(r["lt100"]), "nv": pct(r["never"])} for r in rvs.to_dicts()],
                       [("ap", "Airport"), ("mo", "Month"), ("lt", "flag vector <100 in 2025"), ("nv", "never in 2025")]))
    md.append("")

    # ---------------------------------------------------------------- 3. NM coverage
    md.append("## 3. Departures without an NM record (`LOBT_flt` null)\n")
    nm_r = rank.group_by("airport", "month").agg(pl.len().alias("n"), pl.col("no_nm").sum().alias("k"), pl.col("no_nm").mean().alias("p")).sort("airport", "month")
    nm_m = train.group_by("airport", "month").agg(pl.col("no_nm").mean().alias("p25m"), pl.col("no_nm").sum().alias("k25m"))
    nm_a = train.group_by("airport").agg(pl.col("no_nm").mean().alias("p25"), pl.col("no_nm").max())
    nm_m_d = {(r["airport"], r["month"]): r for r in nm_m.to_dicts()}
    nm_a_d = {r["airport"]: r for r in nm_a.to_dicts()}
    # Also the worst 2025 airport-month for the airport.
    nm_worst = train.group_by("airport", "month").agg(pl.col("no_nm").mean().alias("p")).group_by("airport").agg(pl.col("p").max().alias("pmax"))
    nm_w_d = {r["airport"]: r["pmax"] for r in nm_worst.to_dicts()}
    rows = []
    for r in nm_r.to_dicts():
        m = nm_m_d[(r["airport"], r["month"])]
        rows.append({"ap": r["airport"], "mo": f"2026-{r['month']:02d}", "n": f"{r['n']:,}", "k": f"{r['k']:,}", "p": pct(r["p"]),
                     "p25m": pct(m["p25m"]), "p25": pct(nm_a_d[r["airport"]]["p25"]), "pmax": pct(nm_w_d[r["airport"]]),
                     "flag": "above every 2025 month" if r["p"] > nm_w_d[r["airport"]] else ""})
    md.append(md_table(rows, [("ap", "Airport"), ("mo", "Month"), ("n", "Departures"), ("k", "no NM"), ("p", "share 2026"),
                              ("p25m", "same month 2025"), ("p25", "all 2025"), ("pmax", "worst 2025 month"), ("flag", "")]))
    tot = rank.group_by("month").agg(pl.len().alias("n"), pl.col("no_nm").mean().alias("p")).sort("month")
    tot25 = train.group_by("month").agg(pl.col("no_nm").mean().alias("p25")).sort("month")
    t25d = {r["month"]: r["p25"] for r in tot25.to_dicts()}
    md.append("\nPer month, all airports: " + "; ".join(f"2026-{r['month']:02d} {pct(r['p'])} (2025 same month {pct(t25d[r['month']])})" for r in tot.to_dicts())
              + f"; all 2025 {pct(train['no_nm'].mean())}.")
    md.append("")
    out["no_nm"] = {"rank": nm_r.to_dicts(), "train_month": nm_m.to_dicts(), "train_all": nm_a.to_dicts(), "rank_month": tot.to_dicts(), "train_month_all": tot25.to_dicts()}

    # ---------------------------------------------------------------- 4. episodes
    md.append("## 4. Named episodes\n")

    def episode(name: str, mask_rank: pl.Expr, ap: str, mo: int, ctrl_label: str, ctrl_mask: pl.Expr | None) -> dict:
        ep = r_all.filter(mask_rank)
        ep_m = r_mon.filter(mask_rank)
        base = r_all.filter((pl.col("airport") == ap) & (pl.col("month") == mo))
        rest = base.filter(~mask_rank)
        # runway/config unseen
        rwk = rank.filter(mask_rank).join(train.group_by("airport", "rwy").len().rename({"len": "n25"}), on=["airport", "rwy"], how="left").with_columns(pl.col("n25").fill_null(0))
        cfk = rank.filter(mask_rank).join(train.group_by("airport", "cfg").len().rename({"len": "n25"}), on=["airport", "cfg"], how="left").with_columns(pl.col("n25").fill_null(0))
        d = {
            "name": name, "rows": ep.height, "rows_rest_of_month": rest.height, "rows_month": base.height,
            "rwy_lt100": (rwk["n25"] < MIN_TRAIN).mean(), "cfg_lt100": (cfk["n25"] < MIN_TRAIN).mean(),
            "cfg_never": (cfk["n25"] == 0).mean(),
            "ctx_any_all": ep["out__any"].mean(), "ctx_any_mon": ep_m["out__any"].mean(),
            "ctx_any_rest": rest["out__any"].mean() if rest.height else None,
            "ctx_self_all": self_all[ap]["out__any"],
            "no_nm": rank.filter(mask_rank)["no_nm"].mean(), "no_nm_rest": rest["no_nm"].mean() if rest.height else None,
            "no_nm_25m": nm_m_d[(ap, mo)]["p25m"],
            "wx": {c: float(rank.filter(mask_rank)[c].cast(pl.Float64).mean()) for c in WX_FLAGS},
            "wx_rest": {c: float(rest[c].cast(pl.Float64).mean()) for c in WX_FLAGS} if rest.height else None,
            "per_feature": {c: float(ep[f"out__{c}"].mean()) for c in NUMERIC},
        }
        if ctrl_mask is not None:
            ctrl = t_all.filter(ctrl_mask)
            d["ctrl_label"] = ctrl_label
            d["ctrl_rows"] = ctrl.height
            d["ctrl_wx"] = {c: float(ctrl[c].cast(pl.Float64).mean()) for c in WX_FLAGS}
            d["ctrl_ctx_any"] = ctrl["out__any"].mean()
        return d

    eps = []
    # LTFM 19-23 Feb: the journal's snow episode (review3, 2026-09-13) is 19-23 Feb 2025 and sits in training.
    # The same calendar window of 2026 is audited as requested, with the 2025 window as control.
    m_ltfm = (pl.col("airport") == "LTFM") & (pl.col("day") >= dt.date(2026, 2, 19)) & (pl.col("day") <= dt.date(2026, 2, 23))
    eps.append(episode("LTFM 19-23 Feb 2026 (same calendar window as the 2025 snow episode)", m_ltfm, "LTFM", 2,
                       "LTFM 19-23 Feb 2025 (the snow episode of the journal, in training)",
                       (pl.col("airport") == "LTFM") & (pl.col("day") >= dt.date(2025, 2, 19)) & (pl.col("day") <= dt.date(2025, 2, 23))))
    # The days of February 2026 on which LTFM METARs actually report snow.
    ltfm_snow_days_26 = sorted(rank.filter((pl.col("airport") == "LTFM") & (pl.col("month") == 2) & (pl.col("wx_snow") == 1))["day"].unique().to_list())
    m_ltfm_snow = (pl.col("airport") == "LTFM") & pl.col("day").is_in(ltfm_snow_days_26)
    eps.append(episode("LTFM Feb 2026, days with any snow METAR (" + ", ".join(d.strftime("%d") for d in ltfm_snow_days_26) + " Feb)", m_ltfm_snow, "LTFM", 2,
                       "LTFM 2025, rows with wx_snow=1", (pl.col("airport") == "LTFM") & (pl.col("wx_snow") == 1)))
    out["ltfm_snow_days_2026"] = [d.isoformat() for d in ltfm_snow_days_26]
    # EHAM taxiway A/B works, June 2026 (window 4 May - 26 June; only June is in the final)
    m_eham = (pl.col("airport") == "EHAM") & (pl.col("day") >= dt.date(2026, 6, 1)) & (pl.col("day") <= dt.date(2026, 6, 26))
    eps.append(episode("EHAM 1-26 Jun 2026 (taxiway A/B works)", m_eham, "EHAM", 6, "EHAM June 2025", (pl.col("airport") == "EHAM") & (pl.col("month") == 6)))
    # EDDF Feb 2026 (ADS-B collapse)
    m_eddf = (pl.col("airport") == "EDDF") & (pl.col("month") == 2)
    eps.append(episode("EDDF Feb 2026 (ADS-B ground-reception collapse)", m_eddf, "EDDF", 2, "EDDF Feb 2025", (pl.col("airport") == "EDDF") & (pl.col("month") == 2)))
    out["episodes"] = eps

    # LTFM daily detail for Feb 2026 around the episode.
    ltfm_days = (
        r_all.filter((pl.col("airport") == "LTFM") & (pl.col("month") == 2) & (pl.col("day") >= dt.date(2026, 2, 16)) & (pl.col("day") <= dt.date(2026, 2, 26)))
        .group_by("day").agg(pl.len().alias("n"), pl.col("wx_snow").cast(pl.Float64).mean().alias("snow"), pl.col("wx_snow_recent").cast(pl.Float64).mean().alias("snow_recent"),
                             pl.col("wx_freezing").cast(pl.Float64).mean().alias("freezing"), pl.col("wx_vis_mi").median().alias("vis_med"), pl.col("wx_temp").median().alias("temp_med"),
                             pl.col("wx_wind").median().alias("wind_med"), pl.col("out__any").mean().alias("ctx_any"), pl.col("no_nm").mean().alias("no_nm"),
                             pl.col("dep_before_60").median().alias("dep60_med"))
        .sort("day")
    )
    out["ltfm_feb_days"] = [{**r, "day": r["day"].isoformat()} for r in ltfm_days.to_dicts()]
    ltfm_snow_days_25 = train.filter((pl.col("airport") == "LTFM") & (pl.col("wx_snow") == 1)).group_by("day").len().sort("day")
    out["ltfm_snow_days_2025"] = [{"day": r["day"].isoformat(), "n": r["len"]} for r in ltfm_snow_days_25.to_dicts()]
    ltfm_snow_rows_25 = int(ltfm_snow_days_25["len"].sum())

    # EHAM June: works window vs rest, runway/config mix.
    eham_cfg = (
        rank.filter((pl.col("airport") == "EHAM") & (pl.col("month") == 6)).with_columns(pl.when(m_eham).then(pl.lit("1-26 Jun")).otherwise(pl.lit("27-30 Jun")).alias("part"))
        .group_by("part", "cfg").len().sort("part", "len", descending=[False, True])
    )
    eham_cfg25 = train.filter((pl.col("airport") == "EHAM")).group_by("cfg").len().rename({"len": "n25"})
    eham_cfg = eham_cfg.join(eham_cfg25, on="cfg", how="left").with_columns(pl.col("n25").fill_null(0))
    out["eham_june_configs"] = eham_cfg.to_dicts()
    eham_rwy = rank.filter((pl.col("airport") == "EHAM") & (pl.col("month") == 6)).group_by("rwy").len().join(train.filter(pl.col("airport") == "EHAM").group_by("rwy").len().rename({"len": "n25"}), on="rwy", how="left").with_columns(pl.col("n25").fill_null(0)).sort("len", descending=True)
    eham_rwy_jun25 = train.filter((pl.col("airport") == "EHAM") & (pl.col("month") == 6)).group_by("rwy").len().rename({"len": "n25_jun"})
    eham_rwy = eham_rwy.join(eham_rwy_jun25, on="rwy", how="left").with_columns(pl.col("n25_jun").fill_null(0))
    out["eham_june_runways"] = eham_rwy.to_dicts()

    for e in eps:
        md.append(f"### {e['name']}\n")
        md.append(f"- Rows: **{e['rows']:,}** of {e['rows_month']:,} in the airport-month (rest of month {e['rows_rest_of_month']:,}).")
        md.append(f"- (1) Runway with < {MIN_TRAIN} 2025 departures: {pct(e['rwy_lt100'])}; configuration < {MIN_TRAIN}: {pct(e['cfg_lt100'])} (never seen: {pct(e['cfg_never'])}).")
        rest_txt = f"; rest of the month {pct(e['ctx_any_rest'])}" if e["ctx_any_rest"] is not None else ""
        md.append(f"- (2) Context any-of outside the 2025 band: {pct(e['ctx_any_all'])} vs all 2025 ({pct(e['ctx_any_mon'])} vs same month 2025; airport self-rate {pct(e['ctx_self_all'])}{rest_txt}).")
        top = sorted(e["per_feature"].items(), key=lambda kv: -kv[1])[:4]
        md.append("  Largest per-feature shares: " + ", ".join(f"`{k}` {pct(v)}" for k, v in top) + ".")
        wx = ", ".join(f"`{k}` {pct(v)}" for k, v in e["wx"].items() if v > 0)
        md.append(f"- METAR flags set in the episode: {wx or 'none'}.")
        if e.get("ctrl_wx") is not None:
            cwx = ", ".join(f"`{k}` {pct(v)}" for k, v in e["ctrl_wx"].items() if v > 0)
            md.append(f"  Control {e['ctrl_label']} ({e['ctrl_rows']:,} rows): {cwx or 'none'}; any-of {pct(e['ctrl_ctx_any'])}.")
        rest_nm = f"; rest of the month {pct(e['no_nm_rest'])}" if e["no_nm_rest"] is not None else ""
        md.append(f"- (3) No NM record: {pct(e['no_nm'])} (same month 2025 {pct(e['no_nm_25m'])}{rest_nm}).")
        md.append("")

    md.append("LTFM February 2026, per day around the episode (`snow` = share of rows with `wx_snow`, medians of visibility (mi), temperature (C), wind (kt), departures in the previous hour):\n")
    md.append(md_table([{**r, "day": r["day"].isoformat(), "snow": pct(r["snow"]), "snow_recent": pct(r["snow_recent"]), "freezing": pct(r["freezing"]), "ctx_any": pct(r["ctx_any"]), "no_nm": pct(r["no_nm"]),
                         "vis_med": f"{r['vis_med']:.1f}", "temp_med": f"{r['temp_med']:.0f}", "wind_med": f"{r['wind_med']:.0f}", "dep60_med": f"{r['dep60_med']:.0f}"} for r in ltfm_days.to_dicts()],
                       [("day", "Day"), ("n", "Rows"), ("snow", "snow"), ("snow_recent", "snow recent"), ("freezing", "freezing"), ("vis_med", "vis med"), ("temp_med", "temp med"), ("wind_med", "wind med"), ("dep60_med", "dep 60 med"), ("ctx_any", "ctx any-of"), ("no_nm", "no NM")]))
    md.append(f"\nLTFM 2025 has {ltfm_snow_rows_25:,} departures with `wx_snow` = 1 on {ltfm_snow_days_25.height} days ("
              + ", ".join(f"{r['day'].isoformat()} ({r['len']})" for r in ltfm_snow_days_25.to_dicts()[:12]) + (", ..." if ltfm_snow_days_25.height > 12 else "") + ").")
    ltfm25 = (
        t_all.filter((pl.col("airport") == "LTFM") & (pl.col("day") >= dt.date(2025, 2, 19)) & (pl.col("day") <= dt.date(2025, 2, 23)))
        .group_by("day").agg(pl.len().alias("n"), pl.col("wx_snow").cast(pl.Float64).mean().alias("snow"), pl.col("wx_vis_mi").median().alias("vis_med"),
                             pl.col("wx_temp").median().alias("temp_med"), pl.col("day_movements").median().alias("dm_med"), pl.col("out__any").mean().alias("ctx_any"), pl.col("no_nm").mean().alias("no_nm"))
        .sort("day")
    )
    md.append("\nThe 2025 episode itself (training rows), per day:\n")
    md.append(md_table([{"day": r["day"].isoformat(), "n": r["n"], "snow": pct(r["snow"]), "vis": f"{r['vis_med']:.1f}", "t": f"{r['temp_med']:.0f}", "dm": f"{r['dm_med']:.0f}", "ctx": pct(r["ctx_any"]), "nm": pct(r["no_nm"])} for r in ltfm25.to_dicts()],
                       [("day", "Day"), ("n", "Rows"), ("snow", "snow"), ("vis", "vis med"), ("t", "temp med"), ("dm", "day_movements med"), ("ctx", "ctx any-of"), ("nm", "no NM")]))
    out["ltfm_feb_2025_episode_days"] = [{**r, "day": r["day"].isoformat()} for r in ltfm25.to_dicts()]
    md.append("")
    md.append(
        f"EHAM June 2026 count reconciliation (final ranking file): departure rows with `ADEP_mvt = EHAM` in June = **{eham_recon['dep_rows_june']:,}**, "
        f"of which {eham_recon['dep_rows_1_26_june']:,} on 1-26 June ({100 * eham_recon['dep_rows_1_26_june'] / eham_recon['dep_rows_june']:.1f} %). "
        f"Rows with `ADEP_mvt = EHAM` over both phases = {eham_recon['adep_eham_any_phase_june']:,} ({eham_recon['adep_eham_arr_phase_june']:,} of them are arrival-phase "
        f"records of flights that left Schiphol, i.e. not EHAM departure rows), {eham_recon['adep_eham_any_phase_1_26_june']:,} of them on 1-26 June. "
        f"The figures 23,812 / 20,753 quoted in `docs/final-submission.md` section 5 and `docs/model-card.md` are these two-phase counts; "
        f"the departure-row counts are {eham_recon['dep_rows_june']:,} / {eham_recon['dep_rows_1_26_june']:,}.")
    md.append("")
    md.append("EHAM June 2026 runway configurations, works window against the rest of the month (`n25` = 2025 departures with the same configuration):\n")
    md.append(md_table([{"part": r["part"], "cfg": f"`{r['cfg']}`", "n": r["len"], "n25": r["n25"]} for r in eham_cfg.to_dicts() if r["len"] >= 200],
                       [("part", "Part"), ("cfg", "Configuration"), ("n", "Departures"), ("n25", "2025 all")]))
    md.append("\nEHAM June 2026 departure runways (`n25` = 2025 all, `n25_jun` = June 2025):\n")
    md.append(md_table([{"rwy": f"`{r['rwy']}`", "n": r["len"], "n25": r["n25"], "n25j": r["n25_jun"]} for r in eham_rwy.to_dicts()],
                       [("rwy", "Runway"), ("n", "Jun 2026"), ("n25", "2025 all"), ("n25j", "Jun 2025")]))
    md.append("")

    # ---------------------------------------------------------------- write
    elapsed = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
    header = [
        "<!-- TABLES BELOW ARE GENERATED by scripts/final_months_input_audit_20261009.py; the summary above them is written by hand. -->",
        f"<!-- generated {t0.isoformat(timespec='seconds')} in {elapsed:.0f} s; FIN1b rows train {train.height:,}, rank {rank.height:,} -->",
        "",
    ]
    tables = "\n".join(header + md)
    marker = "<!-- GENERATED TABLES -->"
    if OUT_MD.exists():
        text = OUT_MD.read_text()
        if marker in text:
            text = text.split(marker)[0] + marker + "\n" + tables
        else:
            text = text + "\n" + marker + "\n" + tables
    else:
        text = marker + "\n" + tables
    OUT_MD.write_text(text)
    if OUT_JSON is not None:
        OUT_JSON.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {OUT_MD}" + (f" and {OUT_JSON}" if OUT_JSON else "") + f" in {elapsed:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
