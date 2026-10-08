"""Generalisation of the 2025 out-of-fold predictions across calendar months (docs/generalisation.md).

    .venv/bin/python scripts/generalisation_report.py --data-root data

Run from the repository root. No model is trained; nothing is written outside --tables / --figures.
Every number used in the document is printed with a `[G..]` prefix or written to a CSV table.

Inputs (paths under --data-root):
  e195_pre_e81_20261007_v2/predictions.parquet   six-fold out-of-fold predictions of the current chain
                                                  before E81 (column `joint`), with month, airport, has_nm,
                                                  evaluation_mask (see docs/validation.md)
  features_fin1b.parquet                          production feature table: departure time (day of the
                                                  bootstrap), MVT minus AOBT_3 (the proxy baseline) and the
                                                  planning-time differences used for the regime labels
  codex_chain_diagnostic_20261004/residual_map.parquet   per-row ADS-B coverage flags derived from the
                                                  production stage inputs (own_track, coverage_prodcov);
                                                  optional, the coverage columns are skipped if missing

Folds hold out month pairs 1+7, 2+8, 3+9, 4+10, 5+11 and 6+12, so every 2025 month was predicted by a
model that did not see it. Baselines are computed fold-honestly: statistics for a held-out month pair
come from the other ten months only. Bootstrap confidence intervals resample calendar days with
replacement inside each group, so rows of one day move together.

Regime labels come from src/train.py (`regime_label`) with the registered base settings pinned below.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
REGIME_ENV = {"REGIME_TOL": "6", "REGIME_EXCLUDE": "EDDF,EHAM,LSZH", "SCHED_TOL_LONG": "0"}

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MONTHS = list(range(1, 13))
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FOLD_OF_MONTH = {m: f"{((m - 1) % 6) + 1}_{((m - 1) % 6) + 7}" for m in MONTHS}
GROUPS = {"Jan+Jul (leaderboard months)": [1, 7], "Feb+Jun (new final months)": [2, 6],
          "other eight months": [3, 4, 5, 8, 9, 10, 11, 12]}
BODY = 7200  # "body" subset: labels up to two hours, as in docs/validation.md

# Colours: validated reference palette of the dataviz method (same constants as scripts/make_doc_figures.py).
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = {"model": "#2a78d6", "proxy": "#eb6834", "median": "#1baf7a"}  # categorical slots 1-3
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue
DPI = 180


# ----------------------------------------------------------------------------- helpers

def rmse_of(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def boot_rmse_ci(day_sse: np.ndarray, day_n: np.ndarray, rng: np.random.Generator, B: int) -> tuple[float, float]:
    """Percentile 95 % interval of the RMSE when calendar days are resampled with replacement."""
    D = len(day_sse)
    if D < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, D, size=(B, D))
    sse = day_sse[idx].sum(axis=1)
    n = day_n[idx].sum(axis=1)
    r = np.sqrt(sse / np.maximum(n, 1))
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_diff_ci(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], rng, B) -> tuple[float, float]:
    """95 % interval of RMSE(a) - RMSE(b), days resampled independently in the two disjoint groups."""
    ia = rng.integers(0, len(a[0]), size=(B, len(a[0])))
    ib = rng.integers(0, len(b[0]), size=(B, len(b[0])))
    ra = np.sqrt(a[0][ia].sum(1) / a[1][ia].sum(1))
    rb = np.sqrt(b[0][ib].sum(1) / b[1][ib].sum(1))
    d = ra - rb
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def daily(df: pl.DataFrame, se_col: str, mask: pl.Expr | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = df if mask is None else df.filter(mask)
    g = x.group_by("day").agg(sse=pl.col(se_col).sum(), n=pl.len()).sort("day")
    return g["sse"].to_numpy().astype(float), g["n"].to_numpy().astype(float)


def fold_honest_airport_stat(df: pl.DataFrame, value: pl.Expr, name: str, stat: str = "median") -> pl.DataFrame:
    """Per-airport statistic of `value` computed on the five other folds, joined back by (fold, airport)."""
    rows = []
    for f in sorted(df["fold"].unique().to_list()):
        other = df.filter(pl.col("fold") != f)
        agg = other.group_by("ADEP_mvt").agg(getattr(value, stat)().alias(name)).with_columns(fold=pl.lit(f))
        rows.append(agg)
    return pl.concat(rows)


# ----------------------------------------------------------------------------- load

def load(root: Path) -> pl.DataFrame:
    pred_path = root / "e195_pre_e81_20261007_v2" / "predictions.parquet"
    feat_path = root / "features_fin1b.parquet"
    rmap_path = root / "codex_chain_diagnostic_20261004" / "residual_map.parquet"
    for pth in (pred_path, feat_path):
        if not pth.exists():
            sys.exit(f"missing input: {pth}")
    p = pl.read_parquet(pred_path, columns=[ID, Y, "month", "ADEP_mvt", "has_nm", "evaluation_mask", "joint"])
    assert p.height == 2_085_047 and p[ID].n_unique() == p.height, "unexpected OOF inventory"
    f = pl.read_parquet(feat_path, columns=[ID, "split", "MVT_TIME_UTC_mvt", "d_mvt_aobt", "d_mvt_sched",
                                            "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"])
    f = f.filter(pl.col("split") == "train").drop("split")
    x = p.join(f, on=ID, how="left", validate="1:1")
    assert x["MVT_TIME_UTC_mvt"].null_count() == 0, "feature table does not cover every OOF row"
    if rmap_path.exists():
        m = pl.read_parquet(rmap_path, columns=[ID, "own_track", "previous_visit", "coverage_prodcov"])
        x = x.join(m, on=ID, how="left", validate="1:1")
        print(f"[G0] ADS-B coverage flags joined from {rmap_path.relative_to(root)}")
    else:
        x = x.with_columns(own_track=pl.lit(None, pl.Boolean), previous_visit=pl.lit(None, pl.Boolean),
                           coverage_prodcov=pl.lit(None, pl.Int64))
        print("[G0] residual_map.parquet missing: ADS-B coverage columns left empty")

    # regime labels exactly as the model was trained (src/train.py)
    os.environ.update(REGIME_ENV)
    sys.path.insert(0, str(REPO / "src"))
    import train as T  # noqa: E402

    lab = T.regime_label(x.select(Y, "ADEP_mvt", "d_mvt_sched", "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"))
    classes = list(T.CLASSES)
    x = x.with_columns(
        regime=pl.Series(np.array(classes, dtype=object)[lab].astype(str)),
        day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        fold=pl.col("month").cast(pl.Int64).replace_strict({m: FOLD_OF_MONTH[m] for m in MONTHS}, return_dtype=pl.String),
        y=pl.col(Y).cast(pl.Float64),
        pred=pl.col("joint").cast(pl.Float64),
    )
    return x


# ----------------------------------------------------------------------------- analysis

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--tables", default=REPO / "docs" / "tables", type=Path)
    ap.add_argument("--figures", default=REPO / "docs" / "figures", type=Path)
    ap.add_argument("--boot", default=1000, type=int, help="bootstrap replicates (days resampled)")
    ap.add_argument("--seed", default=20261008, type=int)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    a.tables.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    x = load(a.data_root)
    n_all = x.height
    ev = x.filter(pl.col("evaluation_mask"))
    print(f"[G1] {n_all:,} OOF rows of 2025; evaluation_mask keeps {ev.height:,} "
          f"(excludes {n_all - ev.height} LIRF rows of the fixed TabPFN tail segment)")
    print(f"[G1] RMSE joint: all {n_all:,} rows {np.sqrt(((x['pred'] - x['y']) ** 2).mean()):.2f} s; "
          f"evaluated rows {np.sqrt(((ev['pred'] - ev['y']) ** 2).mean()):.2f} s")
    x = ev  # everything below uses the evaluated rows, as docs/validation.md does

    # --- fold-honest baselines -------------------------------------------------------------------------
    med = fold_honest_airport_stat(x, pl.col("y"), "ap_median")
    # proxy: take-off minus AOBT_3 (NM rows); the per-airport offset is the median of y - proxy on the other folds
    off = fold_honest_airport_stat(x.filter(pl.col("d_mvt_aobt").is_not_null()),
                                   pl.col("y") - pl.col("d_mvt_aobt"), "proxy_offset")
    x = x.join(med, on=["fold", "ADEP_mvt"], how="left").join(off, on=["fold", "ADEP_mvt"], how="left")
    x = x.with_columns(
        base_median=pl.col("ap_median"),
        base_proxy=pl.coalesce(pl.col("d_mvt_aobt"), pl.col("ap_median")),
        base_proxy_adj=pl.coalesce(pl.col("d_mvt_aobt") + pl.col("proxy_offset"), pl.col("ap_median")),
    )
    for c in ("pred", "base_median", "base_proxy", "base_proxy_adj"):
        x = x.with_columns(((pl.col(c) - pl.col("y")) ** 2).alias(f"se_{c}"))
    x = x.with_columns(body=pl.col("y") <= BODY, cov2=(pl.col("coverage_prodcov") >= 2))
    print(f"[G2] proxy available (AOBT_3 present) on {x['d_mvt_aobt'].is_not_null().mean() * 100:.2f}% of rows; "
          f"no-NM rows fall back to the fold-honest airport median")
    offs = off.group_by("ADEP_mvt").agg(pl.col("proxy_offset").mean()).sort("ADEP_mvt")
    print("[G2] mean fold-honest proxy offset (median of y - (MVT - AOBT_3)) by airport, s: "
          + ", ".join(f"{r['ADEP_mvt']} {r['proxy_offset']:.0f}" for r in offs.to_dicts()))

    # --- per month -----------------------------------------------------------------------------------
    rows = []
    for m in MONTHS:
        g = x.filter(pl.col("month") == m)
        gb = g.filter(pl.col("body"))
        r = {"month": m, "month_name": MONTH_ABBR[m - 1], "fold": FOLD_OF_MONTH[m],
             "group": "Jan+Jul" if m in (1, 7) else "Feb+Jun" if m in (2, 6) else "other",
             "rows": g.height, "rows_body": gb.height, "days": g["day"].n_unique()}
        for c in ("pred", "base_median", "base_proxy", "base_proxy_adj"):
            r[f"rmse_all_{c}"] = rmse_of(g[f"se_{c}"].sum(), g.height)
            r[f"rmse_body_{c}"] = rmse_of(gb[f"se_{c}"].sum(), gb.height)
        lo, hi = boot_rmse_ci(*daily(g, "se_pred"), rng, a.boot)
        r["ci_all_lo"], r["ci_all_hi"] = lo, hi
        lo, hi = boot_rmse_ci(*daily(gb, "se_pred"), rng, a.boot)
        r["ci_body_lo"], r["ci_body_hi"] = lo, hi
        r["gain_all_vs_proxy"] = r["rmse_all_base_proxy"] - r["rmse_all_pred"]
        r["gain_body_vs_proxy"] = r["rmse_body_base_proxy"] - r["rmse_body_pred"]
        r["gain_body_vs_proxy_adj"] = r["rmse_body_base_proxy_adj"] - r["rmse_body_pred"]
        r["gain_body_vs_median"] = r["rmse_body_base_median"] - r["rmse_body_pred"]
        rows.append(r)
    by_month = pl.DataFrame(rows)
    by_month.write_csv(a.tables / "generalisation_by_month.csv", float_precision=3)
    print("[G3] month  rows      RMSE all [95% CI]        RMSE body [95% CI]     proxy all/body  proxy+off body  median body")
    for r in rows:
        print(f"[G3] {r['month_name']:>4} {r['rows']:>8,}  {r['rmse_all_pred']:6.1f} [{r['ci_all_lo']:6.1f},{r['ci_all_hi']:6.1f}]  "
              f"{r['rmse_body_pred']:6.1f} [{r['ci_body_lo']:6.1f},{r['ci_body_hi']:6.1f}]   "
              f"{r['rmse_all_base_proxy']:6.1f}/{r['rmse_body_base_proxy']:6.1f}   {r['rmse_body_base_proxy_adj']:6.1f}        "
              f"{r['rmse_body_base_median']:6.1f}")

    # --- month groups --------------------------------------------------------------------------------
    grows = []
    gdaily = {}
    for name, months in GROUPS.items():
        g = x.filter(pl.col("month").is_in(months))
        gb = g.filter(pl.col("body"))
        r = {"group": name, "months": "+".join(MONTH_ABBR[m - 1] for m in months), "rows": g.height,
             "rows_body": gb.height, "days": g["day"].n_unique()}
        for c in ("pred", "base_median", "base_proxy", "base_proxy_adj"):
            r[f"rmse_all_{c}"] = rmse_of(g[f"se_{c}"].sum(), g.height)
            r[f"rmse_body_{c}"] = rmse_of(gb[f"se_{c}"].sum(), gb.height)
        da, db = daily(g, "se_pred"), daily(gb, "se_pred")
        gdaily[name] = (da, db)
        r["ci_all_lo"], r["ci_all_hi"] = boot_rmse_ci(*da, rng, a.boot)
        r["ci_body_lo"], r["ci_body_hi"] = boot_rmse_ci(*db, rng, a.boot)
        r["labels_gt_4h"] = int((g["y"] > 14400).sum())
        top = np.sort(g["se_pred"].to_numpy())[::-1]
        r["sse_share_top10_rows"] = float(top[:10].sum() / top.sum())
        grows.append(r)
    groups = pl.DataFrame(grows)
    k1, k2 = list(GROUPS)[0], list(GROUPS)[1]
    d_all = boot_diff_ci(gdaily[k2][0], gdaily[k1][0], rng, a.boot)
    d_body = boot_diff_ci(gdaily[k2][1], gdaily[k1][1], rng, a.boot)
    groups = groups.with_columns(
        diff_vs_janjul_all=pl.col("rmse_all_pred") - grows[0]["rmse_all_pred"],
        diff_vs_janjul_body=pl.col("rmse_body_pred") - grows[0]["rmse_body_pred"])
    groups.write_csv(a.tables / "generalisation_month_groups.csv", float_precision=3)
    for r in grows:
        print(f"[G4] {r['group']:<30} rows {r['rows']:>9,}  RMSE all {r['rmse_all_pred']:.2f} [{r['ci_all_lo']:.1f}, {r['ci_all_hi']:.1f}]  "
              f"body {r['rmse_body_pred']:.2f} [{r['ci_body_lo']:.1f}, {r['ci_body_hi']:.1f}]  labels>4h {r['labels_gt_4h']}  "
              f"top-10 rows hold {100 * r['sse_share_top10_rows']:.1f}% of SSE")
    print(f"[G4] Feb+Jun minus Jan+Jul: all rows {grows[1]['rmse_all_pred'] - grows[0]['rmse_all_pred']:+.2f} s "
          f"[{d_all[0]:+.1f}, {d_all[1]:+.1f}]; body {grows[1]['rmse_body_pred'] - grows[0]['rmse_body_pred']:+.2f} s "
          f"[{d_body[0]:+.1f}, {d_body[1]:+.1f}] (95% bootstrap by day, groups independent)")
    for r in grows:
        print(f"[G4] {r['group']:<30} gain over proxy: all {r['rmse_all_base_proxy'] - r['rmse_all_pred']:.1f} s, "
              f"body {r['rmse_body_base_proxy'] - r['rmse_body_pred']:.1f} s; over proxy+offset body "
              f"{r['rmse_body_base_proxy_adj'] - r['rmse_body_pred']:.1f} s; over airport median body "
              f"{r['rmse_body_base_median'] - r['rmse_body_pred']:.1f} s")

    # --- airport x month -----------------------------------------------------------------------------
    am = (x.group_by("ADEP_mvt", "month").agg(
        rows=pl.len(), rows_body=pl.col("body").sum(),
        sse=pl.col("se_pred").sum(), sse_body=pl.col("se_pred").filter(pl.col("body")).sum(),
        sse_proxy_body=pl.col("se_base_proxy").filter(pl.col("body")).sum(),
        sse_median_body=pl.col("se_base_median").filter(pl.col("body")).sum(),
        bias_body=(pl.col("pred") - pl.col("y")).filter(pl.col("body")).mean(),
        own_track_share=pl.col("own_track").cast(pl.Float64).mean(),
        no_nm_share=(~pl.col("has_nm")).cast(pl.Float64).mean(),
        labels_gt_2h=(~pl.col("body")).sum())
        .with_columns(rmse_all=(pl.col("sse") / pl.col("rows")).sqrt(),
                      rmse_body=(pl.col("sse_body") / pl.col("rows_body")).sqrt(),
                      rmse_body_proxy=(pl.col("sse_proxy_body") / pl.col("rows_body")).sqrt(),
                      rmse_body_median=(pl.col("sse_median_body") / pl.col("rows_body")).sqrt())
        .with_columns(gain_body_vs_proxy=pl.col("rmse_body_proxy") - pl.col("rmse_body"))
        .drop("sse", "sse_body", "sse_proxy_body", "sse_median_body")
        .sort("ADEP_mvt", "month"))
    am.write_csv(a.tables / "generalisation_by_airport_month.csv", float_precision=3)
    print("[G5] body RMSE by airport and month (s); columns Jan..Dec")
    for ap_ in AIRPORTS:
        s = am.filter(pl.col("ADEP_mvt") == ap_).sort("month")
        print(f"[G5] {ap_} " + " ".join(f"{v:6.1f}" for v in s["rmse_body"].to_list()))
    print("[G5] all-rows RMSE by airport and month (s)")
    for ap_ in AIRPORTS:
        s = am.filter(pl.col("ADEP_mvt") == ap_).sort("month")
        print(f"[G5] {ap_} " + " ".join(f"{v:6.1f}" for v in s["rmse_all"].to_list()))
    worse = am.filter(pl.col("gain_body_vs_proxy") < 0)
    print(f"[G5] airport-months where the model body RMSE is worse than the raw proxy: {worse.height} of {am.height}"
          + ("" if worse.height == 0 else "; " + ", ".join(f"{r['ADEP_mvt']} {MONTH_ABBR[r['month'] - 1]} ({r['gain_body_vs_proxy']:+.1f})" for r in worse.to_dicts())))
    # dispersion of body RMSE across months per airport (coefficient of variation)
    disp = am.group_by("ADEP_mvt").agg(mean=pl.col("rmse_body").mean(), std=pl.col("rmse_body").std(),
                                        lo=pl.col("rmse_body").min(), hi=pl.col("rmse_body").max()).sort("ADEP_mvt")
    print("[G5] body RMSE across the 12 months per airport: mean, min-max, CV")
    for r in disp.to_dicts():
        print(f"[G5] {r['ADEP_mvt']} {r['mean']:.1f}  {r['lo']:.1f}-{r['hi']:.1f}  CV {r['std'] / r['mean']:.2f}")

    # --- what differs between months -----------------------------------------------------------------
    tot = x["se_pred"].sum()
    drv = (x.group_by("month").agg(
        rows=pl.len(),
        label_sd=pl.col("y").std(),
        label_mean=pl.col("y").mean(),
        share_no_nm=(~pl.col("has_nm")).cast(pl.Float64).mean(),
        sse_share_no_nm=pl.col("se_pred").filter(~pl.col("has_nm")).sum() / pl.col("se_pred").sum(),
        sse_share_lirf=pl.col("se_pred").filter(pl.col("ADEP_mvt") == "LIRF").sum() / pl.col("se_pred").sum(),
        sse_share_lirf_no_nm=pl.col("se_pred").filter((pl.col("ADEP_mvt") == "LIRF") & ~pl.col("has_nm")).sum() / pl.col("se_pred").sum(),
        sse_share_labels_gt_2h=pl.col("se_pred").filter(~pl.col("body")).sum() / pl.col("se_pred").sum(),
        labels_gt_2h=(~pl.col("body")).sum(),
        labels_gt_4h=(pl.col("y") > 14400).sum(),
        share_regime_sched=(pl.col("regime") == "sched").cast(pl.Float64).mean(),
        share_regime_planned=(pl.col("regime") != "regular").cast(pl.Float64).mean(),
        share_own_track=pl.col("own_track").cast(pl.Float64).mean(),
        share_coverage_ge2=pl.col("cov2").cast(pl.Float64).mean(),
        share_lirf_own_track=pl.col("own_track").filter(pl.col("ADEP_mvt") == "LIRF").cast(pl.Float64).mean(),
        share_lemd_own_track=pl.col("own_track").filter(pl.col("ADEP_mvt") == "LEMD").cast(pl.Float64).mean(),
        rmse_nm=(pl.col("se_pred").filter(pl.col("has_nm")).sum() / pl.col("has_nm").sum()).sqrt(),
        rmse_no_nm=(pl.col("se_pred").filter(~pl.col("has_nm")).sum() / (~pl.col("has_nm")).sum()).sqrt(),
        rmse_body_regular=(pl.col("se_pred").filter(pl.col("body") & (pl.col("regime") == "regular")).sum()
                           / (pl.col("body") & (pl.col("regime") == "regular")).sum()).sqrt(),
        month_sse_share=pl.col("se_pred").sum() / tot,
    ).sort("month").with_columns(month_name=pl.col("month").cast(pl.Int64).replace_strict(
        {m: MONTH_ABBR[m - 1] for m in MONTHS}, return_dtype=pl.String)))
    drv.write_csv(a.tables / "generalisation_drivers_by_month.csv", float_precision=4)
    print("[G6] month  label sd  no-NM%  SSE% no-NM  SSE% LIRF  SSE% y>2h  n>2h  n>4h  sched%  own-track%  LIRF own%  LEMD own%  RMSE NM / no-NM  body-regular RMSE")
    for r in drv.to_dicts():
        ot = f"{100 * r['share_own_track']:5.1f}" if r["share_own_track"] is not None else "  n/a"
        lo = f"{100 * r['share_lirf_own_track']:5.1f}" if r["share_lirf_own_track"] is not None else "  n/a"
        le = f"{100 * r['share_lemd_own_track']:5.1f}" if r["share_lemd_own_track"] is not None else "  n/a"
        print(f"[G6] {r['month_name']:>4}  {r['label_sd']:7.0f}  {100 * r['share_no_nm']:5.2f}  {100 * r['sse_share_no_nm']:9.1f}  "
              f"{100 * r['sse_share_lirf']:8.1f}  {100 * r['sse_share_labels_gt_2h']:8.1f}  {r['labels_gt_2h']:4d}  {r['labels_gt_4h']:4d}  "
              f"{100 * r['share_regime_sched']:5.2f}  {ot}      {lo}      {le}      {r['rmse_nm']:6.1f} / {r['rmse_no_nm']:6.1f}   {r['rmse_body_regular']:6.1f}")
    # correlation between monthly body RMSE and candidate drivers (12 points; descriptive only)
    bm = by_month.join(drv, on="month")
    for c in ("label_sd", "share_no_nm", "share_own_track", "share_regime_sched", "labels_gt_2h"):
        v = bm[c].to_numpy().astype(float)
        if np.isfinite(v).all():
            print(f"[G6] corr(monthly all-rows RMSE, {c}) = {np.corrcoef(bm['rmse_all_pred'].to_numpy(), v)[0, 1]:+.2f}; "
                  f"corr(monthly body RMSE, {c}) = {np.corrcoef(bm['rmse_body_pred'].to_numpy(), v)[0, 1]:+.2f}")

    # --- figures -------------------------------------------------------------------------------------
    if not a.no_figures:
        fig_months(by_month, a.figures / "fig_generalisation_months.png")
        fig_airport_month(am, a.figures / "fig_generalisation_airport_month.png")
    print("[done] tables in", a.tables, "figures in", a.figures)


# ----------------------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.titlepad": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": DPI,
        "lines.solid_capstyle": "round",
    })
    return plt


def fig_months(bm: pl.DataFrame, out: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    m = bm["month"].to_numpy()
    for ax, kind, title in ((axes[0], "all", "All evaluated rows"), (axes[1], "body", "Labels up to 7,200 s")):
        ax.set_title(f"{title}: out-of-fold RMSE by month, 2025", loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for mm in (1, 7):
            ax.axvspan(mm - 0.5, mm + 0.5, color="#f0efec", zorder=0)
        for mm in (2, 6):
            ax.axvspan(mm - 0.5, mm + 0.5, color=RAMP[0], alpha=0.55, zorder=0)
        ax.plot(m, bm[f"rmse_{kind}_base_median"], color=SERIES["median"], lw=2, marker="o", ms=4.5, label="airport median (fold-honest)")
        ax.plot(m, bm[f"rmse_{kind}_base_proxy"], color=SERIES["proxy"], lw=2, marker="o", ms=4.5, label="take-off − AOBT$_3$ proxy")
        lo, hi = bm[f"ci_{kind}_lo"].to_numpy(), bm[f"ci_{kind}_hi"].to_numpy()
        ax.errorbar(m, bm[f"rmse_{kind}_pred"], yerr=[bm[f"rmse_{kind}_pred"].to_numpy() - lo, hi - bm[f"rmse_{kind}_pred"].to_numpy()],
                    color=SERIES["model"], lw=2, marker="o", ms=5, capsize=2.5, elinewidth=1, label="model (joint, 95% CI by day)", zorder=5)
        ax.set_xticks(m)
        ax.set_xticklabels(MONTH_ABBR)
        ax.set_ylabel("RMSE, seconds")
        ax.set_ylim(0, None)
        for mm in (1, 7):
            ax.text(mm, ax.get_ylim()[1] * 0.985, "board", ha="center", va="top", fontsize=8, color=INK2)
        for mm in (2, 6):
            ax.text(mm, ax.get_ylim()[1] * 0.985, "final", ha="center", va="top", fontsize=8, color=INK2)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.text(0.01, 0.01, "Six folds hold out month pairs 1+7 … 6+12; baselines use the other ten months. "
             "Shaded: Jan/Jul = leaderboard months, Feb/Jun = months added in the final. 2025 out-of-fold, not 2026.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


def fig_airport_month(am: pl.DataFrame, out: Path) -> None:
    plt = _style()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    M = np.full((len(AIRPORTS), 12), np.nan)
    G = np.full((len(AIRPORTS), 12), np.nan)
    for r in am.to_dicts():
        i, j = AIRPORTS.index(r["ADEP_mvt"]), r["month"] - 1
        M[i, j] = r["rmse_body"]
        G[i, j] = r["gain_body_vs_proxy"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1, 1]})
    bounds = [60, 90, 120, 150, 200, 250, 300, 400]
    cmap = ListedColormap(RAMP)
    norm = BoundaryNorm(bounds, cmap.N)
    ax = axes[0]
    ax.set_title("Out-of-fold RMSE on labels ≤ 7,200 s, by airport and month (s)", loc="left")
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(AIRPORTS)):
        for j in range(12):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.6, color="white" if v >= 200 else INK)
    ax = axes[1]
    ax.set_title("Model gain over the take-off − AOBT$_3$ proxy, same cells (s)", loc="left")
    gb = [0, 120, 150, 180, 210, 250, 300, 500]
    ax.imshow(np.clip(G, 0, None), cmap=cmap, norm=BoundaryNorm(gb, cmap.N), aspect="auto")
    for i in range(len(AIRPORTS)):
        for j in range(12):
            v = G[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7.6, color="white" if v >= 210 else INK)
    for ax in axes:
        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_ABBR)
        ax.set_yticks(range(len(AIRPORTS)))
        ax.set_yticklabels(AIRPORTS)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for j in (0, 1, 5, 6):
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(AIRPORTS), fill=False, lw=1.4,
                                       edgecolor=INK if j in (0, 6) else "#eb6834"))
    fig.text(0.01, 0.01, "Black frames: Jan/Jul (leaderboard months). Orange frames: Feb/Jun (months added in the final). "
             "Cells are 2025 out-of-fold rows; colour steps are a single sequential ramp.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
