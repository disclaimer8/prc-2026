from common import *
EXT = f"{ROOT}/analysis/ops/data_ext"
df = base()
c = df.filter("clean")
# 1) frozen-wx: how many distinct local days carry the effect
mets=[]
for a in APTS:
    m = pl.read_csv(f"{EXT}/metar_{a}.csv", null_values="M", infer_schema_length=0).select(pl.col("station").alias("APT"), pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M").dt.replace_time_zone("UTC").alias("mt"), pl.col("tmpc").cast(pl.Float64), pl.col("wxcodes").fill_null(""))
    mets.append(m)
met = pl.concat(mets).unique(["APT","mt"]).sort("mt")
met = met.with_columns(pl.col("wxcodes").str.contains(r"(SN|SG|PL|GS|FZRA|FZDZ|FZFG|IC)").alias("frozen"))
d = c.filter(pl.col("PHASE_mvt")=="DEP").select("APT","BLOCK_TIME_UTC_mvt","date_local","taxi_min").sort("BLOCK_TIME_UTC_mvt").join_asof(met, left_on="BLOCK_TIME_UTC_mvt", right_on="mt", by="APT", strategy="backward", tolerance="90m")
fz = d.filter(pl.col("frozen")==True)
print("=== frozen-wx departures: distinct local days and share of departures from top-3 days ===")
days = fz.group_by("APT","date_local").agg(pl.len().alias("n"), pl.col("taxi_min").median().alias("med"))
print(days.group_by("APT").agg(pl.len().alias("days"), pl.col("n").sum().alias("deps"), (pl.col("n").sort(descending=True).head(3).sum()/pl.col("n").sum()*100).round(1).alias("pct_top3_days"), pl.col("med").median().round(2).alias("median_of_daily_medians")).sort("APT"))
# robustness: day-level comparison — median over days of (daily median in frozen hours) vs same airport dry>3C hours median
# 2) timestamp resolution: seconds modulo 5
print("=== share of BLOCK timestamps whose seconds are a multiple of 5 (1-s data expect ~20%) ===")
print(df.group_by("APT").agg((pl.col("BLOCK_TIME_UTC_mvt").dt.second()%5==0).mean().round(3).alias("block_sec_mod5_eq0"), (pl.col("MVT_TIME_UTC_mvt").dt.second()%5==0).mean().round(3).alias("mvt_sec_mod5_eq0")).sort("APT"))
# 3) wake category
print("=== taxi-out by wake category (DEP, clean) pooled and per airport median ===")
print(c.filter(pl.col("PHASE_mvt")=="DEP").group_by("WK_TBL_CAT_flt").agg(stats()).sort("N", descending=True))
print(c.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("WK_TBL_CAT_flt").is_in(["M","H","J"])).group_by("APT","WK_TBL_CAT_flt").agg(pl.col("taxi_min").median().round(2)).pivot(on="WK_TBL_CAT_flt", index="APT", values="taxi_min").sort("APT"))
# 4) ranking set: 2026 vs 2025 same months
r = pl.scan_parquet(f"{ROOT}/data/prc-2026-datasets/ranking.parquet").with_columns(airport_expr()).collect()
print("ranking rows:", r.height, " DEP:", r.filter(pl.col("PHASE_mvt")=="DEP").height, " DEP with BLOCK null:", r.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("BLOCK_TIME_UTC_mvt").is_null()).height, " DEP taxi null:", r.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("TAXITIME_SEC_mvt").is_null()).height)
print("ranking months:", r.group_by(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym")).len().sort("ym"))
r = r.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"), (pl.col("TAXITIME_SEC_mvt")/60).alias("taxi_min"))
t25 = df.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m")).filter(pl.col("m").is_in([1,7]) & (pl.col("MVT_TIME_UTC_mvt").dt.year()==2025))
cnt = t25.group_by("APT","m").len().rename({"len":"n2025"}).join(r.filter(pl.col("m").is_in([1,7])).group_by("APT","m").len().rename({"len":"n2026"}), on=["APT","m"]).with_columns(((pl.col("n2026")/pl.col("n2025")-1)*100).round(1).alias("chg_pct")).sort("m","APT")
print("=== movements (DEP+ARR, by UTC month of MVT time): Jan/Jul 2026 (ranking) vs Jan/Jul 2025 (training) ==="); print(cnt)
print(cnt.group_by("m").agg(pl.col("n2025").sum(), pl.col("n2026").sum()).with_columns(((pl.col("n2026")/pl.col("n2025")-1)*100).round(2).alias("chg_pct")))
ti = t25.filter((pl.col("PHASE_mvt")=="ARR") & pl.col("clean")).group_by("APT","m").agg(pl.col("taxi_min").median().alias("med2025")).join(
     r.filter((pl.col("PHASE_mvt")=="ARR") & pl.col("TAXITIME_SEC_mvt").is_between(60,14400)).group_by("APT","m").agg(pl.col("taxi_min").median().alias("med2026"), pl.len().alias("N2026")), on=["APT","m"]).with_columns((pl.col("med2026")-pl.col("med2025")).round(2).alias("d")).sort("m","APT")
print("=== ARR taxi-in median, Jan/Jul 2026 vs 2025 ==="); print(ti)
