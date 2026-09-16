"""Smoothed target encodings of historical taxi-out time.

Rows being encoded never see their own target: training rows get out-of-fold
values (folds = calendar months), prediction rows get values from all training rows.
"""
from __future__ import annotations

import os

import polars as pl

TARGET = "TAXITIME_SEC_mvt"
CLIP = 7200
SMOOTH = 20.0

KEYS = {
    "te_stand_rwy": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "te_rwy_hour": ["ADEP_mvt", "RUNWAY_mvt", "local_hour_int"],
    "te_stand": ["ADEP_mvt", "STAND_mvt"],
    "te_op_ap": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
    "te_type_ap": ["ADEP_mvt", "AIRCRAFT_TYPE_mvt"],
    "te_ades_ap": ["ADEP_mvt", "ADES_mvt"],
}
# encodings of the residual against the NM off-block proxy
RESID_KEYS = {
    "ter_stand_rwy": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "ter_op_ap": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
}


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        local_hour_int=pl.col("local_hour").floor().cast(pl.Int8),
        _y=pl.col(TARGET).clip(0, CLIP).cast(pl.Float64),
        _r=(pl.col(TARGET).clip(0, CLIP) - pl.col("d_mvt_aobt")).cast(pl.Float64),
    )


def _encode(stats_src: pl.DataFrame, target_rows: pl.DataFrame, name: str, keys: list[str], col: str) -> pl.DataFrame:
    prior = stats_src.group_by("ADEP_mvt").agg(_prior=pl.col(col).mean())
    stats = stats_src.drop_nulls(col).group_by(keys).agg(_s=pl.col(col).sum(), _n=pl.len())
    out = (
        target_rows.select("MVT_ID_mvt", *keys)
        .join(stats, on=keys, how="left", nulls_equal=True)
        .join(prior, on="ADEP_mvt", how="left")
        .with_columns(
            ((pl.col("_s").fill_null(0) + SMOOTH * pl.col("_prior")) / (pl.col("_n").fill_null(0) + SMOOTH)).alias(name),
            pl.col("_n").fill_null(0).alias(f"{name}_n"),
        )
        .select("MVT_ID_mvt", name, f"{name}_n")
    )
    return out


def _quantile(stats_src: pl.DataFrame, target_rows: pl.DataFrame, name: str, keys: list[str], parent: list[str], q: float) -> pl.DataFrame:
    """Unimpeded-taxi reference in the spirit of the EUROCONTROL PRU indicator: the q-quantile of taxi-out per group,
    falling back to the parent group when the group has fewer than 10 rows."""
    child = stats_src.group_by(keys).agg(_q=pl.col("_y").quantile(q), _n=pl.len())
    par = stats_src.group_by(parent).agg(_qp=pl.col("_y").quantile(q))
    return (
        target_rows.select("MVT_ID_mvt", *keys)
        .join(child, on=keys, how="left", nulls_equal=True)
        .join(par, on=parent, how="left", nulls_equal=True)
        .select("MVT_ID_mvt", pl.when(pl.col("_n") >= 10).then(pl.col("_q")).otherwise(pl.col("_qp")).alias(name))
    )


QUANTILE_SPECS = [
    ("tq_p10_stand_rwy", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], 0.10),
    ("tq_p50_stand_rwy", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], 0.50),
]


# E50 (REGIME_RATES=1): how often a group records the block time from a planned time at all. The classifier currently has
# to learn that from raw categories; the ceiling of perfect regime detection is -16.2 s on the rows the base model decides
# (measured 16.09 on the fold predictions), so the prior is worth giving it directly. Shares are out-of-fold by month,
# smoothed towards the airport share, exactly like the taxi encodings above.
RATE_KEYS = {
    "rr_ap": ["ADEP_mvt"],
    "rr_op_ap": ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
    "rr_stand": ["ADEP_mvt", "STAND_mvt"],
    "rr_hour": ["ADEP_mvt", "local_hour_int"],
    # by flight number: measured 16.09 on LIRF departures that do have an NM record — of the flights whose other months
    # are planned in >= 95% of rows, 99.0% of the current month's rows are planned too (2122 rows), against 9.1% for
    # flights below 5%. Taking the candidate outright in that band is worth -0.53 s, while replacing the model's
    # probability by this rate everywhere is worth +19.5 s: the model must keep conditioning on the row itself,
    # so the rate goes in as a feature, not as a rule.
    "rr_flight": ["ADEP_mvt", "FLIGHT_mvt"],
}


def _rates(stats_src: pl.DataFrame, target_rows: pl.DataFrame, name: str, keys: list[str]) -> pl.DataFrame:
    """Share of each planned regime (and of "any planned") in the group, smoothed towards the airport share."""
    cols = [c for c in stats_src.columns if c.startswith("_lab")]
    prior = stats_src.group_by("ADEP_mvt").agg([pl.col(c).mean().alias(f"_p{c}") for c in cols])
    stats = stats_src.group_by(keys).agg([pl.col(c).sum().alias(f"_s{c}") for c in cols] + [pl.len().alias("_n")])
    out = (target_rows.select("MVT_ID_mvt", *keys)
           .join(stats, on=keys, how="left", nulls_equal=True)
           .join(prior, on="ADEP_mvt", how="left")
           .with_columns([((pl.col(f"_s{c}").fill_null(0) + SMOOTH * pl.col(f"_p{c}")) / (pl.col("_n").fill_null(0) + SMOOTH)
                           ).alias(f"{name}{c[4:]}") for c in cols]))
    return out.select("MVT_ID_mvt", *[f"{name}{c[4:]}" for c in cols])


def add_encodings(fit: pl.DataFrame, pred: pl.DataFrame, fit_labels_regular, fit_labels=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (fit, pred) with encoding columns. Only regular-regime fit rows feed the statistics."""
    fit = _prep(fit).with_columns(_regular=pl.Series(fit_labels_regular))
    pred = _prep(pred)
    src_all = fit.filter(pl.col("_regular"))
    specs = [(n, k, "_y") for n, k in KEYS.items()] + [(n, k, "_r") for n, k in RESID_KEYS.items()]

    months = fit["month"].unique().sort().to_list()
    fit_parts = []
    for m in months:
        rows = fit.filter(pl.col("month") == m)
        src = src_all.filter(pl.col("month") != m)
        enc = rows.select("MVT_ID_mvt")
        for name, keys, col in specs:
            enc = enc.join(_encode(src, rows, name, keys, col), on="MVT_ID_mvt")
        for name, keys, parent, q in QUANTILE_SPECS:
            enc = enc.join(_quantile(src, rows, name, keys, parent, q), on="MVT_ID_mvt")
        fit_parts.append(enc)
    fit_enc = pl.concat(fit_parts)

    pred_enc = pred.select("MVT_ID_mvt")
    for name, keys, col in specs:
        pred_enc = pred_enc.join(_encode(src_all, pred, name, keys, col), on="MVT_ID_mvt")
    for name, keys, parent, q in QUANTILE_SPECS:
        pred_enc = pred_enc.join(_quantile(src_all, pred, name, keys, parent, q), on="MVT_ID_mvt")

    if os.environ.get("REGIME_RATES") == "1" and fit_labels is not None:
        lab = pl.Series(fit_labels)
        fit = fit.with_columns([(lab == i).alias(f"_lab_{n}") for i, n in enumerate(("sched", "iobt", "lobt", "eobt"), start=1)]
                               + [(lab > 0).alias("_lab_any")])
        rate_parts = []
        for m in months:
            rows = fit.filter(pl.col("month") == m)
            src = fit.filter(pl.col("month") != m)
            enc = rows.select("MVT_ID_mvt")
            for name, keys in RATE_KEYS.items():
                enc = enc.join(_rates(src, rows, name, keys), on="MVT_ID_mvt")
            rate_parts.append(enc)
        fit_enc = fit_enc.join(pl.concat(rate_parts), on="MVT_ID_mvt", how="left")
        pr = pred.select("MVT_ID_mvt")
        for name, keys in RATE_KEYS.items():
            pr = pr.join(_rates(fit, pred, name, keys), on="MVT_ID_mvt")
        pred_enc = pred_enc.join(pr, on="MVT_ID_mvt", how="left")
        fit = fit.drop([c for c in fit.columns if c.startswith("_lab")])

    drop = ["local_hour_int", "_y", "_r"]
    fit = fit.drop(drop + ["_regular"]).join(fit_enc, on="MVT_ID_mvt", how="left")
    pred = pred.drop(drop).join(pred_enc, on="MVT_ID_mvt", how="left")
    return fit, pred
