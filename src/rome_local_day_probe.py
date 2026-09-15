"""Probe (15.09, competitor idea "Rome local day"): for LIRF departures without NM at D = 4-30 h in 2025, regime versus
whether take-off and schedule fall on different local (Europe/Rome) calendar days, and the same by UTC dates.
    .venv/bin/python src/rome_local_day_probe.py"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

Y = "TAXITIME_SEC_mvt"
tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))], how="diagonal_relaxed")
x = tr.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
    D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
    local_change=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date() != pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date(),
    utc_change=pl.col("MVT_TIME_UTC_mvt").dt.date() != pl.col("SCHED_TIME_UTC_mvt").dt.date(),
).with_columns(cls=pl.when((pl.col(Y) - pl.col("D")).abs() <= 6).then(pl.lit("sched")).when(pl.col(Y).is_between(86400, 93600)).then(pl.lit("day")).otherwise(pl.lit("regular")),
               Dbin=pl.col("D").cut([4 * 3600, 8 * 3600, 12 * 3600, 30 * 3600], labels=["<4h", "4-8h", "8-12h", "12-30h", ">30h"]))
for key in ("local_change", "utc_change"):
    print(key)
    print(x.filter(pl.col("D") >= 4 * 3600).group_by("Dbin", key).agg(n=pl.len(), sched=(pl.col("cls") == "sched").sum(), day=(pl.col("cls") == "day").sum(),
          regular=(pl.col("cls") == "regular").sum()).sort("Dbin", key).to_pandas().to_string(index=False))
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet").filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
    D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
    local_change=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date() != pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date())
print("ranking LIRF no-NM D>=4h by local_change:", rk.filter(pl.col("D") >= 14400).group_by("local_change").len().to_dicts())
