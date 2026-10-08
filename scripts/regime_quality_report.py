#!/usr/bin/env python
"""Recording-regime detector as a data-quality tool, and taxi-out predictions in PRU terms.

Two reviewer-facing analyses built from existing outputs (no model is trained here):

1. The regime classifier of the production model (``src/train.py``) estimates, per departure, the
   probability that the recorded block time is not a physical off-block but a copy of a planning
   timestamp (SCHED, IOBT, LOBT, EOBT_1) or a one-day shift.  This script measures that detector
   against the +-6 s label rule on 2025 out-of-fold predictions (calibration, precision/recall),
   and quantifies what such records do to monthly taxi-out statistics and to a PRU-style
   "additional taxi-out time" indicator.
2. The out-of-fold prediction is decomposed into a PRU-style unimpeded reference (10th percentile of
   taxi-out per stand x runway group) plus predicted additional time, and compared with the labels.

Inputs (under ``--data-root``, default ``data``):
    cv_preds_fin1b_hz/fold_*.npz          production out-of-fold predictions: mvt_id, pred, reg, y,
                                          proba (n x 6, class order train.CLASSES), values (n x 6)
    prc-2026-datasets/training_*.parquet  challenge training months (departure records are used)
    taxi_out_additional_time_2025.csv     optional: PRU's published monthly indicator (ansperformance.eu),
                                          used only as a cross-check of the emulation

Outputs (under the repository, ``docs/tables`` and ``docs/figures``; override with ``--out-root``):
    regime_detector_summary.csv, regime_detector_calibration.csv, regime_detector_thresholds.csv,
    regime_detector_class_agreement.csv, regime_effect_by_airport_month.csv, regime_effect_by_airport.csv,
    pru_reference_groups.csv, pru_view_additional_by_airport.csv, pru_view_monthly_kpi.csv,
    pru_view_monthly_kpi_summary.csv, regime_probabilities_2025.parquet, regime_quality_run.json,
    fig6_regime_detector_calibration.png, fig7_regime_effect_on_kpi.png

Usage:
    POLARS_MAX_THREADS=2 nice -n 19 .venv/bin/python scripts/regime_quality_report.py --data-root data
    # per-flight table for a new period from a saved parts file (mvt_id + proba) and the matching MVT parquet:
    .venv/bin/python scripts/regime_quality_report.py export --parts submissions/parts_vN.npz \
        --mvt data/prc-2026-datasets/final_ranking.parquet --out docs/tables/regime_probabilities_2026.parquet
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "2")

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train as T  # noqa: E402  regime_label, CLASSES, TOL, PLANNED_REGIME_EXCLUDED, CANDIDATES

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
LABELLED = [a for a in AIRPORTS if a not in T.PLANNED_REGIME_EXCLUDED]
THRESHOLDS = (0.5, 0.8, 0.95)
CAL_BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
# PRU (ATXOT indicator documentation, ed. 01.00, March 2023): taxi-out = ATOT - AOBT; reference = 10th percentile of
# taxi-out per stand x departure-runway group over 12 months; a reference is valid when >= 10 flights of the group lie
# at or below it; flights without stand, runway or AOBT, taxi-out above 120 min, helicopters and de-icing after AOBT are
# excluded; the airport indicator is the mean additional time over flights with a valid reference.
PRU_MAX_TAXI = 7200
PRU_MIN_AT_OR_BELOW = 10
PRU_PCT = 0.10
# The challenge data carry no helicopter flag; this list of common ICAO helicopter designators is our approximation.
HELICOPTER_TYPES = {
    "EC20", "EC25", "EC30", "EC35", "EC45", "EC55", "EC75", "H135", "H145", "H160", "H175", "A109", "A119",
    "A139", "A149", "A169", "A189", "AS32", "AS50", "AS55", "AS65", "B06", "B105", "B212", "B412", "B429",
    "B407", "B505", "S61", "S76", "S92", "R22", "R44", "R66", "H500", "H60", "H47", "NH90", "AW09", "LYNX", "PUMA",
}
PROB_COLS = ["p_regular", "p_sched", "p_iobt", "p_lobt", "p_eobt", "p_dayshift"]
assert [c[2:] for c in PROB_COLS] == T.CLASSES, T.CLASSES

# dataviz palette (validated: node validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light -> ALL CHECKS PASS)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_TEXT, C_TEXT2, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------- load


def load_oof(oof_dir: str) -> tuple[pl.DataFrame, list[dict]]:
    files = sorted(glob.glob(os.path.join(oof_dir, "fold_*.npz")))
    if not files:
        sys.exit(f"no fold_*.npz under {oof_dir}")
    parts, meta = [], []
    for f in files:
        z = np.load(f)
        proba = z["proba"].astype(np.float64)
        assert proba.shape[1] == len(T.CLASSES), proba.shape
        ids = z["mvt_id"]
        assert np.all(ids == np.round(ids)), "MVT ids are not integral"
        fold = os.path.basename(f)[len("fold_"):-len(".npz")]
        d = {"MVT_ID_mvt": ids.astype(np.int64), "fold": np.full(len(ids), fold), "pred": z["pred"].astype(np.float64),
             "reg": z["reg"].astype(np.float64), "y_oof": z["y"].astype(np.float64)}
        for i, c in enumerate(PROB_COLS):
            d[c] = proba[:, i]
        parts.append(pl.DataFrame(d))
        meta.append({"file": f, "rows": int(len(ids)), "md5": md5(f)})
    df = pl.concat(parts)
    assert df["MVT_ID_mvt"].n_unique() == df.height, "duplicate MVT ids across folds"
    return df, meta


def load_departures(data_root: str) -> pl.DataFrame:
    files = sorted(glob.glob(os.path.join(data_root, "prc-2026-datasets", "training_*.parquet")))
    if not files:
        sys.exit("no training_*.parquet")
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
            "EOBT_1_flt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt"]

    def secs(a, b):
        return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)

    lf = pl.concat([pl.scan_parquet(f).select(cols) for f in files]).filter(pl.col("PHASE_mvt") == "DEP")
    lf = lf.select(
        pl.col("MVT_ID_mvt").cast(pl.Int64),
        "ADEP_mvt",
        pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32).alias("month"),
        "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
        pl.col(T.TARGET).cast(pl.Float64),
        d_mvt_sched=secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        d_mvt_iobt=secs("MVT_TIME_UTC_mvt", "IOBT_flt"),
        d_mvt_lobt=secs("MVT_TIME_UTC_mvt", "LOBT_flt"),
        d_mvt_eobt=secs("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
    )
    return lf.collect()


def weighted_quantile_by_group(df: pl.DataFrame, group: list[str], w: str, q: float, out: str) -> pl.DataFrame:
    """Inverted empirical (weighted) CDF: the smallest y whose cumulative weight reaches q of the group's weight.
    Returns one row per group with `out` (the quantile), `n_w` (total weight) and `n_at_or_below` (weight at or below it)."""
    d = df.filter(pl.col(w) > 0).sort(group + ["y"])
    d = d.with_columns(cw=pl.col(w).cum_sum().over(group), W=pl.col(w).sum().over(group))
    qrow = d.filter(pl.col("cw") >= q * pl.col("W")).group_by(group).agg(pl.col("y").first().alias(out), pl.col("W").first().alias("n_w"))
    d = d.join(qrow.select(group + [out]), on=group, how="left")
    below = d.filter(pl.col("y") <= pl.col(out)).group_by(group).agg(pl.col(w).sum().alias("n_at_or_below"))
    return qrow.join(below, on=group, how="left")


def weighted_median(y: np.ndarray, w: np.ndarray) -> float:
    m = w > 0
    y, w = y[m], w[m]
    if len(y) == 0:
        return float("nan")
    o = np.argsort(y, kind="stable")
    cw = np.cumsum(w[o])
    return float(y[o][np.searchsorted(cw, 0.5 * cw[-1])])


# ------------------------------------------------------------------------------------------------------------ detector


def detector_tables(df: pl.DataFrame, out_dir: str) -> dict:
    from sklearn.metrics import roc_auc_score

    p = df["p_planning_copy"].to_numpy()
    lab = df["label_class"].to_numpy()
    pos = lab > 0
    ap = df["ADEP_mvt"].to_numpy()

    summary, cal, thr = [], [], []
    for scope in ["ALL", "LABELLED"] + AIRPORTS:
        if scope == "ALL":
            m = np.ones(len(p), bool)
        elif scope == "LABELLED":
            m = np.isin(ap, LABELLED)
        else:
            m = ap == scope
        ps, ys = p[m], pos[m]
        n = int(m.sum())
        row = {"scope": scope, "n": n, "label_share": float(ys.mean()), "mean_p_planning_copy": float(ps.mean()),
               "expected_count": float(ps.sum()), "label_count": int(ys.sum()),
               "brier": float(np.mean((ps - ys) ** 2)),
               "auc": float(roc_auc_score(ys, ps)) if 0 < ys.sum() < n else float("nan"),
               "share_p_ge_0.5": float((ps >= 0.5).mean())}
        summary.append(row)
        for t in THRESHOLDS:
            f = ps >= t
            tp, fp, fn = int((f & ys).sum()), int((f & ~ys).sum()), int((~f & ys).sum())
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            thr.append({"scope": scope, "threshold": t, "n_flagged": int(f.sum()), "tp": tp, "fp": fp, "fn": fn,
                        "precision": prec, "recall": rec,
                        "f1": 2 * prec * rec / (prec + rec) if tp else float("nan")})
        idx = np.digitize(ps, CAL_BINS) - 1
        for b in range(len(CAL_BINS) - 1):
            mb = idx == b
            if mb.sum() == 0:
                continue
            cal.append({"scope": scope, "bin_lo": CAL_BINS[b], "bin_hi": min(CAL_BINS[b + 1], 1.0), "n": int(mb.sum()),
                        "mean_p": float(ps[mb].mean()), "label_rate": float(ys[mb].mean())})
    pl.DataFrame(summary).write_csv(os.path.join(out_dir, "regime_detector_summary.csv"), float_precision=6)
    pl.DataFrame(cal).write_csv(os.path.join(out_dir, "regime_detector_calibration.csv"), float_precision=6)
    pl.DataFrame(thr).write_csv(os.path.join(out_dir, "regime_detector_thresholds.csv"), float_precision=6)

    # which planning class: among rows the detector flags at 0.5, does its most probable non-regular class match the label?
    P = df.select(PROB_COLS).to_numpy()
    top = 1 + np.argmax(P[:, 1:], axis=1)
    agree = []
    for t in THRESHOLDS:
        f = p >= t
        for k in range(1, len(T.CLASSES)):
            mk = f & (lab == k)
            agree.append({"threshold": t, "label_class": T.CLASSES[k], "n_label_flagged": int(mk.sum()),
                          "top_class_matches": int((top[mk] == k).sum()),
                          "match_rate": float((top[mk] == k).mean()) if mk.sum() else float("nan")})
    pl.DataFrame(agree).write_csv(os.path.join(out_dir, "regime_detector_class_agreement.csv"), float_precision=6)
    return {"summary": summary, "thresholds": thr}


# ------------------------------------------------------------------------------------------------------- PRU emulation


def pru_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Apply PRU filters, add the three weight columns and the three stand x runway references."""
    base = df.with_columns(
        heli=pl.col("AIRCRAFT_TYPE_mvt").is_in(list(HELICOPTER_TYPES)),
        y=pl.col(T.TARGET),
    ).with_columns(
        pru_ok=(pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null() & ~pl.col("heli")
                & (pl.col("y") >= 0) & (pl.col("y") <= PRU_MAX_TAXI)),
        w_all=pl.lit(1.0),
        w_rule=(pl.col("label_class") == 0).cast(pl.Float64),
        w_prob=pl.col("p_regular"),
    )
    grp = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
    ok = base.filter(pl.col("pru_ok"))
    for v in ["all", "rule", "prob"]:
        q = weighted_quantile_by_group(ok, grp, f"w_{v}", PRU_PCT, f"ref_{v}")
        q = q.with_columns((pl.col("n_at_or_below") >= PRU_MIN_AT_OR_BELOW).alias(f"valid_{v}")).rename(
            {"n_w": f"n_{v}", "n_at_or_below": f"n_le_{v}"})
        base = base.join(q, on=grp, how="left")
    return base.with_columns([pl.col(f"valid_{v}").fill_null(False) for v in ["all", "rule", "prob"]])


def wmean(y: pl.Expr, w: pl.Expr) -> pl.Expr:
    return (y * w).sum() / w.sum()


def effect_tables(pf: pl.DataFrame, pru_csv: str | None, out_dir: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    key = ["ADEP_mvt", "month"]
    ok = pf.filter(pl.col("pru_ok"))
    # reference groups per airport (for the doc): groups, valid groups, share of flights with a valid reference
    groups = (ok.group_by("ADEP_mvt").agg(
        n_flights=pl.len(),
        n_groups=pl.struct("STAND_mvt", "RUNWAY_mvt").n_unique(),
        n_groups_valid=pl.struct("STAND_mvt", "RUNWAY_mvt").filter(pl.col("valid_all")).n_unique(),
        share_flights_valid_ref=pl.col("valid_all").mean(),
        share_flights_valid_ref_rule=pl.col("valid_rule").mean(),
        mean_ref_all=pl.col("ref_all").filter(pl.col("valid_all")).mean(),
        mean_ref_rule=pl.col("ref_rule").filter(pl.col("valid_rule")).mean(),
        mean_ref_prob=pl.col("ref_prob").filter(pl.col("valid_prob")).mean(),
        mean_abs_ref_all_minus_rule=(pl.col("ref_all") - pl.col("ref_rule")).abs().filter(pl.col("valid_all") & pl.col("valid_rule")).mean(),
    ).sort("ADEP_mvt"))
    groups.write_csv(os.path.join(out_dir, "pru_reference_groups.csv"), float_precision=4)

    raw = pf.group_by(key).agg(
        n_dep=pl.len(),
        label_share=(pl.col("label_class") > 0).mean(),
        prob_share=pl.col("p_planning_copy").mean(),
        label_count=(pl.col("label_class") > 0).sum(),
        expected_count=pl.col("p_planning_copy").sum(),
        # copies that would inflate a mean: label rows (resp. probability mass) with a recorded taxi-out above 30 min
        label_count_y_gt_1800=((pl.col("label_class") > 0) & (pl.col("y") > 1800)).sum(),
        expected_count_y_gt_1800=pl.col("p_planning_copy").filter(pl.col("y") > 1800).sum(),
        mean_taxi_raw_all=pl.col("y").mean(),
        mean_taxi_raw_rule=wmean(pl.col("y"), pl.col("w_rule")),
        mean_taxi_raw_prob=wmean(pl.col("y"), pl.col("w_prob")),
    )
    filt = ok.group_by(key).agg(
        n_pru_ok=pl.len(),
        label_share_pru_ok=(pl.col("label_class") > 0).mean(),
        prob_share_pru_ok=pl.col("p_planning_copy").mean(),
        mean_taxi_all=pl.col("y").mean(),
        mean_taxi_rule=wmean(pl.col("y"), pl.col("w_rule")),
        mean_taxi_prob=wmean(pl.col("y"), pl.col("w_prob")),
        median_taxi_all=pl.col("y").median(),
        n_valid_all=pl.col("valid_all").sum(),
        kpi_add_all=wmean(pl.col("y") - pl.col("ref_all"), pl.col("w_all") * pl.col("valid_all")),
        kpi_add_rule=wmean(pl.col("y") - pl.col("ref_rule"), pl.col("w_rule") * pl.col("valid_rule")),
        kpi_add_prob=wmean(pl.col("y") - pl.col("ref_prob"), pl.col("w_prob") * pl.col("valid_prob")),
        # same reference (as-recorded) but copies removed: isolates the effect of the flights from that of the reference
        kpi_add_rule_sameref=wmean(pl.col("y") - pl.col("ref_all"), pl.col("w_rule") * pl.col("valid_all")),
        kpi_pred_all=wmean(pl.col("pred") - pl.col("ref_all"), pl.col("w_all") * pl.col("valid_all")),
        kpi_pred_rule=wmean(pl.col("pred") - pl.col("ref_rule"), pl.col("w_rule") * pl.col("valid_rule")),
        kpi_reg_rule=wmean(pl.col("reg") - pl.col("ref_rule"), pl.col("w_rule") * pl.col("valid_rule")),
    )
    # weighted medians (numpy; 120 cells)
    med = []
    for (ap, mo), g in ok.select(key + ["y", "w_rule", "w_prob"]).group_by(key, maintain_order=True):
        med.append({"ADEP_mvt": ap, "month": mo,
                    "median_taxi_rule": weighted_median(g["y"].to_numpy(), g["w_rule"].to_numpy()),
                    "median_taxi_prob": weighted_median(g["y"].to_numpy(), g["w_prob"].to_numpy())})
    am = raw.join(filt, on=key, how="left").join(pl.DataFrame(med), on=key, how="left")
    am = am.with_columns(
        d_mean_raw_rule=pl.col("mean_taxi_raw_all") - pl.col("mean_taxi_raw_rule"),
        d_mean_raw_prob=pl.col("mean_taxi_raw_all") - pl.col("mean_taxi_raw_prob"),
        d_mean_rule=pl.col("mean_taxi_all") - pl.col("mean_taxi_rule"),
        d_mean_prob=pl.col("mean_taxi_all") - pl.col("mean_taxi_prob"),
        d_median_rule=pl.col("median_taxi_all") - pl.col("median_taxi_rule"),
        d_median_prob=pl.col("median_taxi_all") - pl.col("median_taxi_prob"),
        d_kpi_rule=pl.col("kpi_add_all") - pl.col("kpi_add_rule"),
        d_kpi_prob=pl.col("kpi_add_all") - pl.col("kpi_add_prob"),
        d_kpi_rule_sameref=pl.col("kpi_add_all") - pl.col("kpi_add_rule_sameref"),
    )
    if pru_csv and os.path.exists(pru_csv):
        pru = pl.read_csv(pru_csv, schema_overrides={"MONTH_NUM": pl.Int32, "COMMENT": pl.Utf8}).filter(pl.col("APT_ICAO").is_in(AIRPORTS))
        pru = pru.select(
            pl.col("APT_ICAO").alias("ADEP_mvt"), pl.col("MONTH_NUM").alias("month"),
            pru_total_flights=pl.col("TF"), pru_valid_flights=pl.col("VALID_FL"), pru_flights_with_ref=pl.col("TOTAL_REF_NB_FL"),
            pru_add_s=pl.col("TOTAL_ADD_TIME_MIN") * 60 / pl.col("TOTAL_REF_NB_FL"),
            pru_ref_s=pl.col("TOTAL_REF_TIME_MIN") * 60 / pl.col("TOTAL_REF_NB_FL"),
        )
        am = am.join(pru, on=key, how="left").with_columns(
            kpi_add_all_minus_pru=pl.col("kpi_add_all") - pl.col("pru_add_s"),
            n_dep_minus_pru_tf=pl.col("n_dep") - pl.col("pru_total_flights"))
    am = am.sort(key)
    am.write_csv(os.path.join(out_dir, "regime_effect_by_airport_month.csv"), float_precision=4)

    # airport x year
    ok_valid = ok
    ay = pf.group_by("ADEP_mvt").agg(
        n_dep=pl.len(), label_share=(pl.col("label_class") > 0).mean(), prob_share=pl.col("p_planning_copy").mean(),
        label_count=(pl.col("label_class") > 0).sum(), expected_count=pl.col("p_planning_copy").sum(),
        d_mean_raw_rule=pl.col("y").mean() - wmean(pl.col("y"), pl.col("w_rule")),
        d_mean_raw_prob=pl.col("y").mean() - wmean(pl.col("y"), pl.col("w_prob")),
    ).join(ok_valid.group_by("ADEP_mvt").agg(
        n_pru_ok=pl.len(),
        share_dropped_by_pru_filters=pl.lit(0.0),
        label_share_pru_ok=(pl.col("label_class") > 0).mean(),
        prob_share_pru_ok=pl.col("p_planning_copy").mean(),
        mean_taxi_all=pl.col("y").mean(),
        d_mean_rule=pl.col("y").mean() - wmean(pl.col("y"), pl.col("w_rule")),
        d_mean_prob=pl.col("y").mean() - wmean(pl.col("y"), pl.col("w_prob")),
        kpi_add_all=wmean(pl.col("y") - pl.col("ref_all"), pl.col("w_all") * pl.col("valid_all")),
        kpi_add_rule=wmean(pl.col("y") - pl.col("ref_rule"), pl.col("w_rule") * pl.col("valid_rule")),
        kpi_add_prob=wmean(pl.col("y") - pl.col("ref_prob"), pl.col("w_prob") * pl.col("valid_prob")),
        kpi_add_rule_sameref=wmean(pl.col("y") - pl.col("ref_all"), pl.col("w_rule") * pl.col("valid_all")),
    ), on="ADEP_mvt", how="left").with_columns(
        share_dropped_by_pru_filters=1 - pl.col("n_pru_ok") / pl.col("n_dep"),
        d_kpi_rule=pl.col("kpi_add_all") - pl.col("kpi_add_rule"),
        d_kpi_prob=pl.col("kpi_add_all") - pl.col("kpi_add_prob"),
        d_kpi_rule_sameref=pl.col("kpi_add_all") - pl.col("kpi_add_rule_sameref"),
    ).join(am.group_by("ADEP_mvt").agg(
        max_month_d_kpi_rule=pl.col("d_kpi_rule").max(), max_month_d_kpi_prob=pl.col("d_kpi_prob").max(),
        max_month_label_share=pl.col("label_share").max(), max_month_prob_share=pl.col("prob_share").max(),
    ), on="ADEP_mvt", how="left").sort("ADEP_mvt")
    ay.write_csv(os.path.join(out_dir, "regime_effect_by_airport.csv"), float_precision=4)
    return am, ay


# --------------------------------------------------------------------------------------------- contribution 2: PRU view


def pru_view_tables(pf: pl.DataFrame, am: pl.DataFrame, out_dir: str) -> tuple[pl.DataFrame, dict]:
    # regular regime by the label rule, PRU filters, valid rule-cleaned reference
    r = pf.filter(pl.col("pru_ok") & (pl.col("label_class") == 0) & pl.col("valid_rule")).with_columns(
        add_actual=pl.col("y") - pl.col("ref_rule"),
        add_pred=pl.col("pred") - pl.col("ref_rule"),
        add_reg=pl.col("reg") - pl.col("ref_rule"),
    )
    r = r.with_columns(month_mean_add=pl.col("add_actual").mean().over("ADEP_mvt", "month"))

    def stats(g: pl.DataFrame, scope: str) -> dict:
        a, p, q, mm = (g[c].to_numpy() for c in ["add_actual", "add_pred", "add_reg", "month_mean_add"])
        return {"scope": scope, "n": g.height, "mean_ref_s": float(g["ref_rule"].mean()),
                "mean_additional_s": float(a.mean()), "sd_additional_s": float(a.std()),
                "rmse_pred_s": float(np.sqrt(np.mean((p - a) ** 2))), "bias_pred_s": float(np.mean(p - a)),
                "rmse_reg_s": float(np.sqrt(np.mean((q - a) ** 2))), "bias_reg_s": float(np.mean(q - a)),
                "rmse_airport_month_mean_s": float(np.sqrt(np.mean((mm - a) ** 2))),
                "skill_pred_vs_sd": float(1 - np.mean((p - a) ** 2) / np.var(a)),
                "skill_pred_vs_month_mean": float(1 - np.mean((p - a) ** 2) / np.mean((mm - a) ** 2)),
                "corr_pred_actual": float(np.corrcoef(p, a)[0, 1])}

    rows = [stats(r, "ALL")] + [stats(g, ap) for (ap,), g in r.group_by("ADEP_mvt", maintain_order=True)]
    rows.sort(key=lambda d: (d["scope"] != "ALL", d["scope"]))
    pl.DataFrame(rows).write_csv(os.path.join(out_dir, "pru_view_additional_by_airport.csv"), float_precision=4)

    # monthly KPI, predictions vs labels
    k = am.select("ADEP_mvt", "month", "n_valid_all", "kpi_add_all", "kpi_pred_all", "kpi_add_rule", "kpi_pred_rule", "kpi_reg_rule",
                  *[c for c in ["pru_add_s"] if c in am.columns]).with_columns(
        d_pred_all=pl.col("kpi_pred_all") - pl.col("kpi_add_all"),
        d_pred_rule=pl.col("kpi_pred_rule") - pl.col("kpi_add_rule"),
        d_reg_rule=pl.col("kpi_reg_rule") - pl.col("kpi_add_rule"),
    )
    k.write_csv(os.path.join(out_dir, "pru_view_monthly_kpi.csv"), float_precision=4)
    summ = []
    for lab_col, pred_col, name in [("kpi_add_all", "kpi_pred_all", "as_recorded"), ("kpi_add_rule", "kpi_pred_rule", "regular_mixture_pred"),
                                    ("kpi_add_rule", "kpi_reg_rule", "regular_regressor_reg")]:
        a, b = k[lab_col].to_numpy(), k[pred_col].to_numpy()
        summ.append({"variant": name, "cells": len(a), "mae_s": float(np.mean(np.abs(b - a))), "bias_s": float(np.mean(b - a)),
                     "max_abs_s": float(np.max(np.abs(b - a))), "corr": float(np.corrcoef(a, b)[0, 1]),
                     "sd_label_kpi_s": float(a.std())})
    if "pru_add_s" in k.columns:
        kk = k.drop_nulls("pru_add_s")
        a, b = kk["pru_add_s"].to_numpy(), kk["kpi_add_all"].to_numpy()
        summ.append({"variant": "our_emulation_vs_pru_published", "cells": len(a), "mae_s": float(np.mean(np.abs(b - a))),
                     "bias_s": float(np.mean(b - a)), "max_abs_s": float(np.max(np.abs(b - a))), "corr": float(np.corrcoef(a, b)[0, 1]),
                     "sd_label_kpi_s": float(a.std())})
    pl.DataFrame(summ).write_csv(os.path.join(out_dir, "pru_view_monthly_kpi_summary.csv"), float_precision=4)
    return pl.DataFrame(rows), {"additional": rows, "kpi": summ}


# -------------------------------------------------------------------------------------------------------------- figures


def _style(ax, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=9, length=0)
    ax.grid(True, axis="y", color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_facecolor(C_SURFACE)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=C_TEXT, fontweight="bold")


def fig_calibration(out_dir_tables: str, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = pl.read_csv(os.path.join(out_dir_tables, "regime_detector_calibration.csv"))
    summ = pl.read_csv(os.path.join(out_dir_tables, "regime_detector_summary.csv"))
    thr = pl.read_csv(os.path.join(out_dir_tables, "regime_detector_thresholds.csv"))
    scopes = ["LABELLED"] + LABELLED
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6), facecolor=C_SURFACE)
    for ax, sc in zip(axes.ravel(), scopes):
        c = cal.filter((pl.col("scope") == sc) & (pl.col("n") >= 30))  # bins with fewer rows stay in the CSV only
        s = summ.filter(pl.col("scope") == sc).row(0, named=True)
        _style(ax, ("Seven labelled airports" if sc == "LABELLED" else sc))
        ax.plot([0, 1], [0, 1], color=C_GRID, linewidth=1)
        ax.plot(c["mean_p"], c["label_rate"], color=C_BLUE, linewidth=2, solid_joinstyle="round")
        sizes = 16 + 60 * np.log10(np.maximum(c["n"].to_numpy(), 1)) / 6
        ax.scatter(c["mean_p"], c["label_rate"], s=sizes, color=C_BLUE, edgecolor=C_SURFACE, linewidth=1.5, zorder=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        t = thr.filter((pl.col("scope") == sc) & (pl.col("threshold") == 0.5)).row(0, named=True)
        flag = (f"P >= 0.5: {t['n_flagged']:,} flagged, precision {100 * t['precision']:.0f} %, recall {100 * t['recall']:.1f} %"
                if t["n_flagged"] >= 30 else f"P >= 0.5: {t['n_flagged']} rows flagged")
        txt = (f"n = {s['n']:,}\nlabel share {100 * s['label_share']:.1f} %, mean P {100 * s['mean_p_planning_copy']:.1f} %\n"
               f"AUC {s['auc']:.3f}, Brier {s['brier']:.4f}\n{flag}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.2, color=C_TEXT2, linespacing=1.35)
        if sc in ("LABELLED", "LEBL"):
            ax.set_ylabel("share of rows whose label is a planning copy", color=C_TEXT2, fontsize=9)
        if ax in axes[1]:
            ax.set_xlabel("mean predicted P(planning copy) in bin", color=C_TEXT2, fontsize=9)
    fig.suptitle("Reliability of the recording-regime detector, 2025 out-of-fold", x=0.01, ha="left", fontsize=14,
                 color=C_TEXT, fontweight="bold")
    fig.text(0.01, 0.925, "Each dot is a probability bin with at least 30 rows (marker size grows with the number of rows); "
             "the grey diagonal is perfect calibration.\nLabel: block time within +-6 s of a planning time, or a one-day shift "
             "(regime_label, src/train.py). EDDF, EHAM and LSZH are not labelled and are not shown.",
             fontsize=9.2, color=C_TEXT2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(fig_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def fig_effect(out_dir_tables: str, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    am = pl.read_csv(os.path.join(out_dir_tables, "regime_effect_by_airport_month.csv"))
    fig, axes = plt.subplots(2, len(LABELLED), figsize=(15, 6.8), facecolor=C_SURFACE, sharex=True)
    months = list(range(1, 13))
    for j, ap in enumerate(LABELLED):
        a = am.filter(pl.col("ADEP_mvt") == ap).sort("month")
        top, bot = axes[0, j], axes[1, j]
        _style(top, ap)
        _style(bot)
        top.plot(a["month"], 100 * a["label_share"], color=C_BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=C_SURFACE, markeredgewidth=1.2)
        top.plot(a["month"], 100 * a["prob_share"], color=C_ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=C_SURFACE, markeredgewidth=1.2)
        bot.plot(a["month"], a["d_kpi_rule"], color=C_BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=C_SURFACE, markeredgewidth=1.2)
        bot.plot(a["month"], a["d_kpi_prob"], color=C_ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=C_SURFACE, markeredgewidth=1.2)
        bot.axhline(0, color=C_GRID, linewidth=1)
        bot.set_xticks([1, 4, 7, 10])
        bot.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        if j == 0:
            top.set_ylabel("planning-copy records, % of departures", color=C_TEXT2, fontsize=9)
            bot.set_ylabel("effect on monthly additional\ntaxi-out time, s per flight", color=C_TEXT2, fontsize=9)
    top_max = 100 * max(am["label_share"].max(), am["prob_share"].max()) * 1.08
    for j, ap in enumerate(LABELLED):
        a = am.filter(pl.col("ADEP_mvt") == ap)
        axes[0, j].set_ylim(0, top_max)
        lo = min(0, a["d_kpi_rule"].min(), a["d_kpi_prob"].min())
        hi = max(0, a["d_kpi_rule"].max(), a["d_kpi_prob"].max())
        pad = 0.08 * (hi - lo)
        axes[1, j].set_ylim(lo - pad, hi + pad)
    handles = [Line2D([], [], color=C_BLUE, linewidth=2, marker="o", markersize=5, label="+-6 s label rule"),
               Line2D([], [], color=C_ORANGE, linewidth=2, marker="o", markersize=5, label="detector probability (expected count / P-weighted)")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.955), ncol=2, frameon=False, fontsize=9.5, labelcolor=C_TEXT)
    fig.suptitle("Planning-copy records by airport and month, 2025, and their effect on a PRU-style indicator", x=0.01,
                 ha="left", fontsize=14, color=C_TEXT, fontweight="bold")
    fig.text(0.01, 0.925, "Top: share of departures whose recorded block time copies a planning time. Bottom: mean additional "
             "taxi-out time (taxi-out minus the 10th-percentile reference per stand x runway)\ncomputed from all records minus "
             "the same indicator with the copies removed (rule) or down-weighted by P(regular) (probability). PRU filters applied; "
             "references from 2025 as a whole.\nThe top row shares one scale; the bottom row has its own scale per airport.",
             fontsize=9.2, color=C_TEXT2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(fig_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------- export


def write_probability_table(df: pl.DataFrame, path: str, with_fold: bool) -> int:
    cols = [pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").alias("airport")]
    if with_fold:
        cols.append(pl.col("fold"))
    cols += [pl.col("p_planning_copy").round(4).cast(pl.Float32)] + [pl.col(c).round(4).cast(pl.Float32) for c in PROB_COLS]
    df.select(cols).sort("MVT_ID_mvt").write_parquet(path, compression="zstd", compression_level=12, statistics=True)
    return os.path.getsize(path)


def export_mode(args) -> None:
    z = np.load(args.parts)
    proba = z["proba"].astype(np.float64)
    assert proba.shape[1] == len(T.CLASSES), proba.shape
    ids = z["mvt_id"]
    assert np.all(ids == np.round(ids))
    d = {"MVT_ID_mvt": ids.astype(np.int64)}
    for i, c in enumerate(PROB_COLS):
        d[c] = proba[:, i]
    df = pl.DataFrame(d).with_columns(p_planning_copy=(1 - pl.col("p_regular")).clip(0, 1))
    mvt = pl.scan_parquet(args.mvt).filter(pl.col("PHASE_mvt") == "DEP").select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt").collect()
    df = df.join(mvt, on="MVT_ID_mvt", how="left")
    missing = df["ADEP_mvt"].null_count()
    size = write_probability_table(df, args.out, with_fold=False)
    print(json.dumps({"rows": df.height, "rows_without_airport": missing, "bytes": size,
                      "mean_p_planning_copy": float(df["p_planning_copy"].mean()), "out": args.out}, indent=1))


# ----------------------------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    rep = sub.add_parser("report", help="2025 out-of-fold analysis (default)")
    rep.add_argument("--data-root", default="data")
    rep.add_argument("--oof", default="cv_preds_fin1b_hz", help="OOF directory under data-root (production control)")
    rep.add_argument("--pru-csv", default="taxi_out_additional_time_2025.csv", help="PRU published indicator under data-root; optional")
    rep.add_argument("--out-root", default=ROOT, help="repository root holding docs/tables and docs/figures")
    rep.add_argument("--no-figures", action="store_true")
    exp = sub.add_parser("export", help="per-flight probability table for a new period from a parts npz")
    exp.add_argument("--parts", required=True, help="npz with mvt_id and proba (n x 6) from the final classifier fit")
    exp.add_argument("--mvt", required=True, help="challenge MVT parquet of that period (airport per MVT_ID)")
    exp.add_argument("--out", required=True)
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["report"] + argv
    args = ap.parse_args(argv)
    if args.cmd == "export":
        export_mode(args)
        return

    t0 = time.time()
    tables = os.path.join(args.out_root, "docs", "tables")
    figures = os.path.join(args.out_root, "docs", "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    oof, oof_meta = load_oof(os.path.join(args.data_root, args.oof))
    dep = load_departures(args.data_root)
    df = oof.join(dep, on="MVT_ID_mvt", how="inner")
    assert df.height == oof.height, (df.height, oof.height)
    assert float((df["y_oof"] - df[T.TARGET]).abs().max()) == 0.0, "OOF y differs from the challenge label"
    df = df.with_columns(label_class=pl.Series(T.regime_label(df)).cast(pl.Int32),
                         p_planning_copy=(1 - pl.col("p_regular")).clip(0, 1))
    print(f"rows {df.height:,} | labelled planning/dayshift {int((df['label_class'] > 0).sum()):,} | {time.time() - t0:.0f}s", flush=True)

    det = detector_tables(df, tables)
    pf = pru_frame(df)
    am, ay = effect_tables(pf, os.path.join(args.data_root, args.pru_csv), tables)
    view, view_meta = pru_view_tables(pf, am, tables)
    size = write_probability_table(df, os.path.join(tables, "regime_probabilities_2025.parquet"), with_fold=True)
    if not args.no_figures:
        fig_calibration(tables, os.path.join(figures, "fig6_regime_detector_calibration.png"))
        fig_effect(tables, os.path.join(figures, "fig7_regime_effect_on_kpi.png"))

    heli_n = int(pf["heli"].sum())
    run = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": os.path.relpath(__file__, args.out_root),
        "inputs": {"oof": oof_meta, "training_parquets": len(glob.glob(os.path.join(args.data_root, "prc-2026-datasets", "training_*.parquet"))),
                   "pru_csv": os.path.exists(os.path.join(args.data_root, args.pru_csv))},
        "label_rule": {"tolerance_s": T.TOL, "planned_regimes_excluded_at": sorted(T.PLANNED_REGIME_EXCLUDED), "classes": T.CLASSES,
                       "sched_tol_long": os.environ.get("SCHED_TOL_LONG", "0")},
        "pru_emulation": {"group": "ADEP x STAND x RUNWAY", "percentile": PRU_PCT, "min_flights_at_or_below_ref": PRU_MIN_AT_OR_BELOW,
                          "max_taxi_s": PRU_MAX_TAXI, "window": "calendar year 2025 (PRU: rolling 12 months ending with the month)",
                          "month_basis": "UTC month of take-off (PRU: local time)", "helicopter_rows_excluded": heli_n,
                          "deicing_filter": "not available in the challenge data"},
        "rows": df.height, "label_positive": int((df["label_class"] > 0).sum()), "expected_positive": float(df["p_planning_copy"].sum()),
        "pru_ok_rows": int(pf["pru_ok"].sum()), "rows_with_valid_reference": int((pf["pru_ok"] & pf["valid_all"]).sum()),
        "probability_table_bytes": size,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(tables, "regime_quality_run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print(json.dumps({k: run[k] for k in ["rows", "label_positive", "expected_positive", "pru_ok_rows", "rows_with_valid_reference",
                                          "probability_table_bytes", "elapsed_s"]}, indent=1))
    print(pl.DataFrame(det["summary"]).filter(pl.col("scope").is_in(["ALL", "LABELLED", "LIRF"])))
    print(ay.select("ADEP_mvt", "label_share", "prob_share", "d_mean_raw_rule", "d_mean_rule", "d_kpi_rule", "d_kpi_prob"))
    print(view.select("scope", "n", "mean_additional_s", "sd_additional_s", "rmse_pred_s", "bias_pred_s", "skill_pred_vs_month_mean"))
    print(pl.DataFrame(view_meta["kpi"]))


if __name__ == "__main__":
    main()
