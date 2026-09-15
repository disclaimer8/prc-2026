"""Probe (15.09): 'postponed flight' pattern behind the two LFPG outliers. A departure without NM whose flight number has no
departure at the same airport on the previous calendar day, while the number flies on >= N other days of the month.
How often does it occur in 2025, how often is the target huge (BLOCK on the previous day), and how many such rows are in ranking?
    .venv/bin/python src/postponed_probe.py"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

Y = "TAXITIME_SEC_mvt"


def pattern(df: pl.DataFrame) -> pl.DataFrame:
    dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(date=pl.col("MVT_TIME_UTC_mvt").dt.date(), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"))
    days = dep.select("ADEP_mvt", "FLIGHT_mvt", "date").unique()
    n_days = dep.group_by("ADEP_mvt", "FLIGHT_mvt", "ym").agg(n_days=pl.col("date").n_unique())
    prev = days.with_columns(date=pl.col("date") + pl.duration(days=1), prev_exists=pl.lit(True))
    x = dep.filter(pl.col("FLIGHT_ID_mvt").is_null() & pl.col("FLIGHT_mvt").is_not_null()).join(n_days, on=["ADEP_mvt", "FLIGHT_mvt", "ym"], how="left") \
        .join(prev, on=["ADEP_mvt", "FLIGHT_mvt", "date"], how="left").with_columns(pl.col("prev_exists").fill_null(False))
    return x.with_columns(D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())


tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))], how="diagonal_relaxed")
x = pattern(tr)
for N in (5, 10, 15):
    hit = x.filter((pl.col("n_days") >= N) & ~pl.col("prev_exists"))
    print(f"2025, regular >= {N} days, no departure the previous day, no NM: rows {hit.height}; y > 20000: {hit.filter(pl.col(Y) > 20000).height}; "
          f"y > 3600: {hit.filter(pl.col(Y) > 3600).height}; D < 4 h and y > 20000: {hit.filter((pl.col('D') < 14400) & (pl.col(Y) > 20000)).height}")
    print("   by airport:", hit.group_by("ADEP_mvt").agg(n=pl.len(), big=(pl.col(Y) > 20000).sum()).sort("n", descending=True).to_dicts())
big = x.filter((pl.col(Y) > 20000) & (pl.col("D") < 14400))
print("\n2025 no-NM rows with y > 20000 and D < 4 h (not explained by schedule delay):", big.height)
print(big.select("ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", Y, "D", "n_days", "prev_exists").sort(Y, descending=True).head(20))
rk = pattern(pl.read_parquet("data/prc-2026-datasets/ranking.parquet"))
for N in (5, 10, 15):
    h = rk.filter((pl.col("n_days") >= N) & ~pl.col("prev_exists"))
    print(f"ranking, regular >= {N}: rows {h.height}, D < 4 h: {h.filter(pl.col('D') < 14400).height}", h.group_by("ADEP_mvt").len().sort("len", descending=True).head(5).to_dicts())
