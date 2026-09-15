"""E16 design probe (15.09): regime of LIRF departures without NM at D = 4-30 h in 2025 versus candidate mechanism signals —
day share of LIRF arrivals whose block equals schedule (+-6 s) with taxi-in > 30 min, flight identity in the same month
(never matched to NM / matched on >= 10 days), garbled call sign. Exploration only; the decision is taken by the gate of E16.
    .venv/bin/python src/e16_signal_probe.py"""
import glob
import os
import re

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

Y = "TAXITIME_SEC_mvt"
tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))], how="diagonal_relaxed")
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet")


def arrival_sub(df):
    a = df.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")).with_columns(date=pl.col("MVT_TIME_UTC_mvt").dt.date())
    return a.group_by("date").agg(n_arr=pl.len(), arr_sub=(((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6)
                                                          & (pl.col(Y) > 1800)).mean())


def tail(df):
    dep = df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")).with_columns(
        date=pl.col("MVT_TIME_UTC_mvt").dt.date(), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), nm=pl.col("FLIGHT_ID_mvt").is_not_null())
    cnt = dep.group_by("FLIGHT_mvt", "ym").agg(n_rk=pl.len(), n_nm=pl.col("nm").sum())
    t = dep.filter(~pl.col("nm")).join(cnt, on=["FLIGHT_mvt", "ym"], how="left").with_columns(
        D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).filter(pl.col("D").is_between(4 * 3600, 30 * 3600))
    t = t.join(arrival_sub(df), on="date", how="left").with_columns(
        garbled=pl.col("FLIGHT_mvt").map_elements(lambda s: bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or "")), return_dtype=pl.Boolean),
        ident=pl.when(pl.col("n_nm").fill_null(0) == 0).then(pl.lit("orphan")).when(pl.col("n_nm") >= 10).then(pl.lit("regular")).otherwise(pl.lit("mid")),
        Dbin=pl.col("D").cut([8 * 3600, 12 * 3600], labels=["4-8h", "8-12h", "12-30h"]))
    return t


t = tail(tr).with_columns(arr_sub_pos=pl.col("arr_sub").fill_null(0) > 0).with_columns(cls=pl.when((pl.col(Y) - pl.col("D")).abs() <= 6).then(pl.lit("sched")).when(pl.col(Y).is_between(86400, 93600)).then(pl.lit("day")).otherwise(pl.lit("regular")))
print("2025 rows", t.height)
for g in [["Dbin"], ["Dbin", "ident"], ["Dbin", "garbled"], ["Dbin", "arr_sub_pos"]]:
    print(t.group_by(g).agg(n=pl.len(), sched=(pl.col("cls") == "sched").sum(), day=(pl.col("cls") == "day").sum(), regular=(pl.col("cls") == "regular").sum()).sort(g).to_pandas().to_string(index=False), "\n")
print("arr_sub day share: 2025 LIRF mean", arrival_sub(tr)["arr_sub"].mean(), "ranking mean", arrival_sub(rk)["arr_sub"].mean())
r = tail(rk)
print("ranking tail rows", r.height, r.group_by("Dbin", "ident").len().sort("Dbin", "ident").to_dicts())
print("ranking tail arr_sub>0:", r.filter(pl.col("arr_sub") > 0).height)
