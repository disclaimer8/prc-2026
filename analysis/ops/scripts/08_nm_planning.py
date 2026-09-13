import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
df = base(extra=["EOBT_1_flt","AOBT_3_flt","IOBT_flt","LOBT_flt","ARVT_1_flt","ARVT_3_flt","ADEP_flt","ADES_flt","ADES_FILED_flt"])
dep = df.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("EOBT_1_flt").is_not_null())
print("DEP rows with NM data:", dep.height, "of", df.filter(pl.col("PHASE_mvt")=="DEP").height)
m = lambda a,b: ((pl.col(a)-pl.col(b)).dt.total_seconds()/60)
dep = dep.with_columns(m("EOBT_1_flt","BLOCK_TIME_UTC_mvt").alias("eobt1_err"), m("AOBT_3_flt","BLOCK_TIME_UTC_mvt").alias("aobt3_err"),
                       m("IOBT_flt","SCHED_TIME_UTC_mvt").alias("iobt_vs_sched"), m("LOBT_flt","EOBT_1_flt").alias("lobt_vs_eobt1"), m("EOBT_1_flt","SCHED_TIME_UTC_mvt").alias("eobt1_vs_sched"))
print("equalities: IOBT==SCHED:", round((dep["iobt_vs_sched"]==0).mean()*100,2), "%;  LOBT==EOBT_1:", round((dep["lobt_vs_eobt1"]==0).mean()*100,2), "%;  IOBT==EOBT_1:", round((dep["IOBT_flt"]==dep["EOBT_1_flt"]).mean()*100,2), "%;  EOBT_1==SCHED:", round((dep["eobt1_vs_sched"]==0).mean()*100,2),"%")
print("AOBT_3 seconds!=0 share:", round((dep["AOBT_3_flt"].dt.second()!=0).mean()*100,2), "%;  EOBT_1 seconds!=0:", round((dep["EOBT_1_flt"].dt.second()!=0).mean()*100,2), "%;  BLOCK seconds!=0:", round((dep["BLOCK_TIME_UTC_mvt"].dt.second()!=0).mean()*100,2), "%")
W = 360
def es(col):
    x = pl.col(col)
    return [pl.len().alias("N_all"), (x.abs()<=W).sum().alias("N_win"), x.filter(x.abs()<=W).mean().round(2).alias("mean"), x.filter(x.abs()<=W).median().round(2).alias("median"),
            x.filter(x.abs()<=W).abs().median().round(2).alias("MAE_med"), x.filter(x.abs()<=W).abs().mean().round(2).alias("MAE_mean"), x.filter(x.abs()<=W).abs().quantile(0.9).round(1).alias("absp90"),
            ((x.abs()<=3).mean()*100).round(1).alias("pct_within3"), ((x.abs()<=15).mean()*100).round(1).alias("pct_within15"), ((x < -15).mean()*100).round(1).alias("pct_block_later_gt15")]
for col, desc in [("eobt1_err","EOBT_1 (filed plan) minus airport BLOCK time"),("aobt3_err","AOBT_3 (NM actual) minus airport BLOCK time"),("iobt_vs_sched","IOBT minus airport SCHED")]:
    print(f"=== {desc}; stats on |err|<=6h, pct on all ===")
    print(dep.select(es(col)))
    print(dep.group_by("APT").agg(es(col)).sort("MAE_mean"))
# EOBT_1 error vs delay; segments
print("=== EOBT_1 error by market segment ==="); print(dep.group_by("MARKET_SEGMENT_flt").agg(es("eobt1_err")).sort("N_all", descending=True))
# how much of airport off-block delay was already in EOBT_1 (i.e., known at filing)?
dd = dep.with_columns(m("BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt").alias("delay")).filter(pl.col("delay").is_between(15, 360) & pl.col("eobt1_vs_sched").abs().le(360))
print("=== departures with off-block delay 15..360 min: how much of the delay was already reflected in EOBT_1 (EOBT_1 - SCHED) ===")
print(dd.select(pl.len().alias("N"), pl.col("delay").median().alias("med_delay"), pl.col("eobt1_vs_sched").median().alias("med_eobt1_minus_sched"), ((pl.col("eobt1_vs_sched")>=15).mean()*100).round(1).alias("pct_eobt1_shows_ge15_delay"), ((pl.col("eobt1_vs_sched")/pl.col("delay")).clip(-1,2).median()).round(3).alias("median_share_known")))
# ARR: ARVT_3 vs landing time
arr = df.filter((pl.col("PHASE_mvt")=="ARR") & pl.col("ARVT_3_flt").is_not_null()).with_columns(m("ARVT_3_flt","MVT_TIME_UTC_mvt").alias("arvt3_err"), m("ARVT_1_flt","MVT_TIME_UTC_mvt").alias("arvt1_err"))
print("=== ARR: ARVT_3 (NM flown) minus airport landing; ARVT_1 (FPL) minus landing ===")
print(arr.select(es("arvt3_err"))); print(arr.select(es("arvt1_err")))
print(arr.group_by("APT").agg(es("arvt1_err")).sort("MAE_mean"))
# chart: EOBT_1 error distribution by airport (median + IQR + p10/p90)
q = dep.filter(pl.col("eobt1_err").abs()<=W).group_by("APT").agg([pl.col("eobt1_err").quantile(p).alias(f"q{int(p*100)}") for p in (0.1,0.25,0.5,0.75,0.9)]).sort("q50")
fig, ax = plt.subplots(figsize=(9,5.5))
for i,r in enumerate(q.iter_rows(named=True)):
    ax.plot([r["q10"],r["q90"]],[i,i], color="#9ac1e8", lw=2); ax.plot([r["q25"],r["q75"]],[i,i], color="#2b6cb0", lw=7); ax.scatter([r["q50"]],[i], color="white", edgecolor="#1a365d", zorder=3, s=30)
ax.axvline(0,color="k",lw=.6); ax.set_yticks(range(q.height)); ax.set_yticklabels(q["APT"])
ax.set_xlabel("EOBT_1 (filed flight plan) minus actual off-block at the airport, minutes\nnegative = aircraft left the stand later than the flight plan said")
ax.set_title("How well the filed off-block time predicts the real one, 2025\n(bar = IQR, whiskers = p10–p90, dot = median; |error| ≤ 6 h)")
ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL NM + airport data). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/08_eobt1_error.png", dpi=150)
print(q)
