"""Smoothed target encodings of historical taxi-out time.

Rows being encoded never see their own target: training rows get out-of-fold
values (folds = calendar months), prediction rows get values from all training rows.
"""
from __future__ import annotations

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


def add_encodings(fit: pl.DataFrame, pred: pl.DataFrame, fit_labels_regular) -> tuple[pl.DataFrame, pl.DataFrame]:
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
        fit_parts.append(enc)
    fit_enc = pl.concat(fit_parts)

    pred_enc = pred.select("MVT_ID_mvt")
    for name, keys, col in specs:
        pred_enc = pred_enc.join(_encode(src_all, pred, name, keys, col), on="MVT_ID_mvt")

    drop = ["local_hour_int", "_y", "_r"]
    fit = fit.drop(drop + ["_regular"]).join(fit_enc, on="MVT_ID_mvt", how="left")
    pred = pred.drop(drop).join(pred_enc, on="MVT_ID_mvt", how="left")
    return fit, pred
