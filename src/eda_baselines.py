"""Quick EDA: target distribution and RMSE of naive baselines on departures."""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

D = "data/prc-2026-datasets/"
train = pl.read_parquet(D + "training_*.parquet").filter(pl.col("PHASE_mvt") == "DEP")
rank = pl.read_parquet(D + "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
y = "TAXITIME_SEC_mvt"

print("train deps", train.height, "ranking deps", rank.height)
print("ranking months", rank.select(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").unique().sort()).to_series().to_list())
print(train[y].describe())
print("quantiles", [train[y].quantile(q) for q in (0.001, 0.01, 0.5, 0.99, 0.999)])

# sanity: target == MVT - BLOCK ?
chk = train.with_columns(((pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds()).alias("d"))
print("target==MVT-BLOCK share", (chk["d"] == chk[y]).mean())

per_ap = train.group_by("ADEP_mvt").agg(
    n=pl.len(), mean=pl.col(y).mean(), sd=pl.col(y).std(), med=pl.col(y).median()
).sort("ADEP_mvt")
print(per_ap)
print("ranking deps per airport", rank.group_by("ADEP_mvt").len().sort("ADEP_mvt").to_dicts())

def rmse(df, pred):
    d = df.select(((pl.col(y) - pred) ** 2).mean().sqrt()).item()
    return round(d, 1)

jul = train.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
print("\nbaselines on Jan+Jul 2025 train deps (in-sample means):")
print(" global mean", rmse(jul, pl.lit(train[y].mean())))
ap_mean = train.group_by("ADEP_mvt").agg(m=pl.col(y).mean())
print(" airport mean", rmse(jul.join(ap_mean, on="ADEP_mvt"), pl.col("m")))
proxy = (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds()
j = jul.filter(pl.col("AOBT_3_flt").is_not_null())
print(" MVT - AOBT_3 raw", rmse(j, proxy), "n", j.height)
print(" corr(proxy,target)", j.select(pl.corr(proxy, pl.col(y))).item())
for col in ["EOBT_1_flt", "LOBT_flt", "IOBT_flt", "SCHED_TIME_UTC_mvt"]:
    p = (pl.col("MVT_TIME_UTC_mvt") - pl.col(col)).dt.total_seconds()
    jj = jul.filter(pl.col(col).is_not_null())
    print(f" corr(MVT-{col}, target)", round(jj.select(pl.corr(p, pl.col(y))).item(), 3))
print(" AOBT_3 seconds component nonzero share", j.select((pl.col("AOBT_3_flt").dt.second() != 0).mean()).item())
