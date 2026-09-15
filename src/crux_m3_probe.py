"""Review 'top-3 levers' (14.09), M3/E15 crux: for LIRF departures without NM, does the regime depend on how often the same
FLIGHT_mvt matches NM on other days of the same month? Uses 2025 labels.   .venv/bin/python src/crux_m3_probe.py"""
import os, glob; os.environ['POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR']='load_as_storage'
import polars as pl
Y="TAXITIME_SEC_mvt"
tr=pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))],how="diagonal_relaxed")
dep=tr.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADEP_mvt")=="LIRF")).with_columns(
    month=pl.col("MVT_TIME_UTC_mvt").dt.month(), nm=pl.col("FLIGHT_ID_mvt").is_not_null())
cnt=dep.group_by("FLIGHT_mvt","month").agg(n_rk=pl.len(), n_nm=pl.col("nm").sum())
x=dep.filter(~pl.col("nm")).join(cnt,on=["FLIGHT_mvt","month"]).with_columns(
    D=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
x=x.with_columns(sched=((pl.col(Y)-pl.col("D")).abs()<=6), day=pl.col(Y).is_between(86400, 86400 + 7200),
    grp=pl.when(pl.col("n_nm")==0).then(pl.lit("A n_nm=0")).when(pl.col("n_nm")>=10).then(pl.lit("B n_nm>=10")).otherwise(pl.lit("mid 1-9")),
    Dbin=pl.col("D").cut([0,4*3600,8*3600,30*3600],labels=["<0","0-4h","4-8h","8-30h",">30h"]))
print(x.group_by("Dbin","grp").agg(n=pl.len(),sched=pl.col("sched").mean().round(2),day=pl.col("day").sum(),
      regular=(~pl.col("sched")&~pl.col("day")).mean().round(2)).sort("Dbin","grp").to_pandas().to_string())
