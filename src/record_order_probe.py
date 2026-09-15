"""Probe (15.09, competitor idea "record ordering"): deviation of MVT_ID from the rolling median of movements at the same
airport in the same monthly file, sorted by MVT_TIME (window +-250 rows), scaled by the rolling p90-p10. Is it related to the
regime of departures without NM (2025 labels)? Computed per (airport, phase): departure and arrival IDs come from different ranges.
    .venv/bin/python src/record_order_probe.py"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

Y = "TAXITIME_SEC_mvt"
W = 501


def record_order(d: pl.DataFrame) -> pl.DataFrame:
    d = d.with_columns(ap=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt"))).sort("ap", "MVT_TIME_UTC_mvt")
    return d.with_columns(
        med=pl.col("MVT_ID_mvt").rolling_quantile(0.5, window_size=W, center=True, min_samples=50).over("ap", "PHASE_mvt"),
        spread=(pl.col("MVT_ID_mvt").rolling_quantile(0.9, window_size=W, center=True, min_samples=50)
                - pl.col("MVT_ID_mvt").rolling_quantile(0.1, window_size=W, center=True, min_samples=50)).over("ap", "PHASE_mvt"),
    ).with_columns(id_dev=(pl.col("MVT_ID_mvt") - pl.col("med")) / pl.col("spread").clip(1, None))


cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", Y]
parts = [record_order(pl.read_parquet(f, columns=cols)) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]
x = pl.concat(parts).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(), nonm=pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
    sched=((pl.col(Y) - pl.col("D")).abs() <= 6), big=pl.col(Y) > 3600,
    dev_bin=pl.col("id_dev").cut([-2, -0.5, -0.1, 0.1, 0.5, 2], labels=["<-2", "-2..-0.5", "-0.5..-0.1", "±0.1", "0.1..0.5", "0.5..2", ">2"]))
pl.Config.set_tbl_rows(40)
for name, sub in (("LIRF no NM", x.filter(pl.col("nonm") & (pl.col("ADEP_mvt") == "LIRF"))), ("no NM outside LIRF", x.filter(pl.col("nonm") & (pl.col("ADEP_mvt") != "LIRF"))),
                  ("with NM", x.filter(~pl.col("nonm")))):
    print(name)
    print(sub.group_by("dev_bin").agg(n=pl.len(), sched=pl.col("sched").mean().round(3), y_gt3600=pl.col("big").mean().round(3), y_mean=pl.col(Y).mean().round(0)).sort("dev_bin"))
