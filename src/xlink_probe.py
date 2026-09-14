"""Review 6 probe (14.09): link departures without NM to a unique NM arrival of the same flight at the destination hub.

    .venv/bin/python src/xlink_probe.py
Basis for E15 (REVIEW_BRIEF.md, sections 7 and 16)."""
import os, glob; os.environ['POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR']='load_as_storage'
import polars as pl
H=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LTFM","LSZH"]
def link(df, name):
    print(df.select(pl.col("PHASE_mvt").unique()).to_series().to_list()) if name=="rank" else None
    dep=df.filter((pl.col("ADEP_mvt").is_in(H)) & pl.col("FLIGHT_ID_mvt").is_null() & pl.col("PHASE_mvt").str.to_uppercase().str.starts_with("DEP"))
    arr=df.filter(pl.col("ADES_mvt").is_in(H) & pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("PHASE_mvt").str.to_uppercase().str.starts_with("ARR"))
    arr=arr.select("FLIGHT_mvt","ADEP_mvt","ADES_mvt",a_mvt=pl.col("MVT_TIME_UTC_mvt"),a_lobt=pl.col("LOBT_flt"),a_aobt=pl.col("AOBT_3_flt"),a_eobt=pl.col("EOBT_1_flt"))
    j=dep.join(arr,on=["FLIGHT_mvt","ADEP_mvt","ADES_mvt"],how="inner").filter(
        ((pl.col("a_mvt")-pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().is_between(0, 16*3600)))
    j=j.with_columns(n=pl.len().over("MVT_ID_mvt")).filter(pl.col("n")==1)
    print(name,"noNM deps to hubs",dep.filter(pl.col("ADES_mvt").is_in(H)).height,"linked uniquely",j.height)
    print(j.group_by("ADEP_mvt").len().sort("len",descending=True).head(10).to_dicts())
    return j
tr=pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))],how="diagonal_relaxed")
j=link(tr,"train")
Y="TAXITIME_SEC_mvt"
j=j.with_columns(
  D=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
  sched=((pl.col(Y)-(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).abs()<=6),
  e_aobt=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("a_aobt")).dt.total_seconds(),
  gap=(pl.col("SCHED_TIME_UTC_mvt")-pl.col("a_lobt")).dt.total_seconds().abs(),
  proxy=(pl.col("MVT_TIME_UTC_mvt")-pl.col("a_aobt")).dt.total_seconds())
print("target>3600:",j.filter(pl.col(Y)>3600).height, "sched:",j["sched"].sum(), "SSE share of y^2>3600^2 rows")
print(j.with_columns(far=pl.col("gap")>3606).group_by("far").agg(n=pl.len(),sched=pl.col("sched").mean(),ymed=pl.col(Y).median(),ymax=pl.col(Y).max(),
      abs_e_aobt_med=pl.col("e_aobt").abs().median()).to_dicts())
nonsched=j.filter(~pl.col("sched"))
print("non-sched rows: |y - (MVT-AOBT_arr)| median", (nonsched[Y]-nonsched["proxy"]).abs().median(), "rmse", (((nonsched[Y]-nonsched["proxy"])**2).mean())**0.5)
rk=pl.read_parquet("data/prc-2026-datasets/ranking.parquet")
link(rk,"rank")
print("---- beyond D")
k=j.with_columns(far=pl.col("gap")>3606, Dbig=pl.col("D").abs()>3606)
print(k.group_by("Dbig","far").agg(n=pl.len(),sched=pl.col("sched").mean(),y_rmse_like=(pl.col(Y).cast(pl.Float64)**2).mean().sqrt()).sort("Dbig","far").to_dicts())
print("big rows:",k.filter(pl.col(Y)>20000).select("ADEP_mvt","FLIGHT_mvt","MVT_TIME_UTC_mvt",Y,"D","gap","sched").to_dicts())
