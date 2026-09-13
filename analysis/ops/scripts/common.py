import polars as pl, glob, os
ROOT = os.path.expanduser("~/prc-2026")
FILES = sorted(glob.glob(f"{ROOT}/data/prc-2026-datasets/training_2025-*.parquet"))
CHARTS = f"{ROOT}/analysis/ops/charts"
def airport_expr():
    return pl.when(pl.col("PHASE_mvt")=="DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("APT")
def load(cols=None):
    lf = pl.scan_parquet(FILES).with_columns(airport_expr())
    if cols: lf = lf.select(cols + ["APT"] if "APT" not in cols else cols)
    return lf
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)

APTS = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
TZ = {"EDDF":"Europe/Berlin","EDDM":"Europe/Berlin","EGLL":"Europe/London","EHAM":"Europe/Amsterdam","LEBL":"Europe/Madrid","LEMD":"Europe/Madrid","LFPG":"Europe/Paris","LIRF":"Europe/Rome","LSZH":"Europe/Zurich","LTFM":"Europe/Istanbul"}
TAXI_MIN, TAXI_MAX = 60, 14400  # "clean" window: 1 min .. 4 h

def base(extra=None):
    cols = ["MVT_ID_mvt","PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","RUNWAY_mvt","STAND_mvt","TAXITIME_SEC_mvt","MARKET_SEGMENT_flt","WK_TBL_CAT_flt","AIRCRAFT_TYPE_mvt"]
    if extra: cols += extra
    df = pl.scan_parquet(FILES).select(cols).with_columns(airport_expr()).collect()
    # local block time per airport (DST-aware)
    parts = []
    for a, tz in TZ.items():
        p = df.filter(pl.col("APT")==a).with_columns(pl.col("BLOCK_TIME_UTC_mvt").dt.convert_time_zone(tz).dt.replace_time_zone(None).alias("BLOCK_LOCAL"))
        parts.append(p)
    df = pl.concat(parts)
    df = df.with_columns(
        pl.col("BLOCK_LOCAL").dt.hour().alias("hour_local"),
        pl.col("BLOCK_LOCAL").dt.weekday().alias("dow"),   # 1=Mon
        pl.col("BLOCK_LOCAL").dt.month().alias("month"),
        pl.col("BLOCK_LOCAL").dt.date().alias("date_local"),
        (pl.col("TAXITIME_SEC_mvt")/60).alias("taxi_min"),
        pl.col("TAXITIME_SEC_mvt").is_between(TAXI_MIN, TAXI_MAX).alias("clean"),
    )
    return df

def stats(col="taxi_min"):
    return [pl.len().alias("N"), pl.col(col).mean().round(2).alias("mean"), pl.col(col).median().round(2).alias("median"), pl.col(col).quantile(0.9).round(2).alias("p90")]
