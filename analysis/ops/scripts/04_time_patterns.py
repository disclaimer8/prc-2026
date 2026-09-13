import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
df = base()
c = df.filter("clean")
dep = c.filter(pl.col("PHASE_mvt")=="DEP"); arr = c.filter(pl.col("PHASE_mvt")=="ARR")
print("=== DEP taxi-out median by month x airport (local month of off-block) ===")
m = dep.group_by("APT","month").agg(pl.col("taxi_min").median().round(2).alias("med")).pivot(on="month", index="APT", values="med", sort_columns=True).sort("APT")
print(m)
print("=== ARR taxi-in median by month x airport ===")
ma = arr.group_by("APT","month").agg(pl.col("taxi_min").median().round(2).alias("med")).pivot(on="month", index="APT", values="med", sort_columns=True).sort("APT")
print(ma)
# winter effect: Jan+Feb vs Jun+Jul+Aug? use Jan-Feb vs May-Sep? choose Jan+Feb vs Jul+Aug
def season(x): return x.with_columns(pl.when(pl.col("month").is_in([1,2])).then(pl.lit("JanFeb")).when(pl.col("month").is_in([7,8])).then(pl.lit("JulAug")).otherwise(None).alias("season")).filter(pl.col("season").is_not_null())
w = season(c).group_by("APT","PHASE_mvt","season").agg(pl.len().alias("N"), pl.col("taxi_min").median().alias("med"), pl.col("taxi_min").mean().alias("mean"), pl.col("taxi_min").quantile(0.9).alias("p90"))
wp = w.pivot(on=["season"], index=["APT","PHASE_mvt"], values=["N","med","mean","p90"])
wp = wp.with_columns((pl.col("med_JanFeb")-pl.col("med_JulAug")).round(2).alias("d_med"), (pl.col("mean_JanFeb")-pl.col("mean_JulAug")).round(2).alias("d_mean"), (pl.col("p90_JanFeb")-pl.col("p90_JulAug")).round(2).alias("d_p90"))
print("=== Winter (Jan+Feb) minus summer (Jul+Aug), minutes ===")
print(wp.select("APT","PHASE_mvt","N_JanFeb","N_JulAug","med_JanFeb","med_JulAug","d_med","d_mean","d_p90").sort("PHASE_mvt","d_mean", descending=[True,True]))
dd = wp.filter(pl.col("PHASE_mvt")=="DEP").select("APT", pl.col("d_mean").alias("dep_d_mean"), pl.col("d_p90").alias("dep_d_p90")).join(wp.filter(pl.col("PHASE_mvt")=="ARR").select("APT", pl.col("d_mean").alias("arr_d_mean"), pl.col("d_p90").alias("arr_d_p90")), on="APT").with_columns((pl.col("dep_d_mean")-pl.col("arr_d_mean")).round(2).alias("dep_minus_arr_d_mean")).sort("dep_minus_arr_d_mean", descending=True)
print("=== difference-in-differences: winter excess taxi-out minus winter excess taxi-in (mean, minutes) ==="); print(dd)

# hour of day
h = dep.group_by("hour_local").agg(stats()).sort("hour_local"); print("=== DEP by local hour (all airports pooled) ==="); print(h)
hh = dep.group_by("APT","hour_local").agg(pl.len().alias("N"), pl.col("taxi_min").median().alias("med"))
peak = hh.filter(pl.col("N")>=3000).group_by("APT").agg(pl.col("hour_local").sort_by("med").last().alias("worst_hour"), pl.col("med").max().round(2).alias("worst_med"), pl.col("hour_local").sort_by("med").first().alias("best_hour"), pl.col("med").min().round(2).alias("best_med")).with_columns((pl.col("worst_med")-pl.col("best_med")).round(2).alias("gap")).sort("gap", descending=True)
print("=== DEP: best/worst local hour per airport (hours with N>=3000) ==="); print(peak)
# hourly departures volume vs taxi-out median correlation per airport
cor = hh.filter(pl.col("N")>=3000).group_by("APT").agg(pl.corr("N","med", method="spearman").round(2).alias("spearman_volume_vs_median"), pl.len().alias("hours")).sort("APT")
print("=== Spearman corr, departures per local hour-of-day (year total) vs median taxi-out ==="); print(cor)
# day of week
d = dep.group_by("dow").agg(stats()).sort("dow"); print("=== DEP by weekday (1=Mon) ==="); print(d)
dw = dep.group_by("APT","dow").agg(pl.col("taxi_min").median().round(2)).pivot(on="dow", index="APT", values="taxi_min", sort_columns=True).sort("APT"); print(dw)

# daily worst days
daily = dep.group_by("APT","date_local").agg(pl.len().alias("N"), pl.col("taxi_min").median().alias("med"), pl.col("taxi_min").quantile(0.9).alias("p90"))
base_med = dep.group_by("APT").agg(pl.col("taxi_min").median().alias("apt_med"))
daily = daily.join(base_med, on="APT").with_columns((pl.col("med")-pl.col("apt_med")).round(2).alias("excess_med"))
print("=== top 20 airport-days by daily median taxi-out excess over annual airport median (days with N>=100) ===")
print(daily.filter(pl.col("N")>=100).sort("excess_med", descending=True).head(20))
print("month distribution of the top-50 excess days:"); print(daily.filter(pl.col("N")>=100).sort("excess_med", descending=True).head(50).with_columns(pl.col("date_local").dt.month().alias("m")).group_by("m").len().sort("m"))
daily.write_parquet("/private/tmp/claude-501/-Users-denyskolomiiets/f3f3ada5-5c4f-4911-a523-fe524ce3746a/scratchpad/daily_dep.parquet")

# chart: monthly medians heat
import numpy as np
fig, axes = plt.subplots(1,2, figsize=(12,5))
for ax, tab, ttl in [(axes[0], m, "Taxi-out"), (axes[1], ma, "Taxi-in")]:
    apts = tab["APT"].to_list(); vals = tab.drop("APT").to_numpy()
    rel = vals - np.nanmedian(vals, axis=1, keepdims=True)
    im = ax.imshow(rel, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(apts))); ax.set_yticklabels(apts); ax.set_xticks(range(12)); ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]): ax.text(j,i,f"{vals[i,j]:.0f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"{ttl}: monthly median, min", fontsize=9)
fig.colorbar(im, ax=axes, shrink=.8, label="minutes vs airport's typical month")
fig.suptitle("Seasonality of taxi times, 2025 (1 min ≤ taxi ≤ 4 h)")
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.savefig(f"{CHARTS}/03_monthly_heatmap.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9,5))
x = dd.sort("dep_minus_arr_d_mean")
yy = range(len(x))
ax.barh([i+0.2 for i in yy], x["dep_d_mean"], height=0.4, color="#2b6cb0", label="Taxi-out: Jan–Feb minus Jul–Aug")
ax.barh([i-0.2 for i in yy], x["arr_d_mean"], height=0.4, color="#9ac1e8", label="Taxi-in: Jan–Feb minus Jul–Aug")
ax.axvline(0, color="k", lw=.6); ax.set_yticks(list(yy)); ax.set_yticklabels(x["APT"]); ax.set_xlabel("difference in mean taxi time, minutes")
ax.set_title("Winter penalty on the ground, 2025: departures vs arrivals\n(mean, 1 min ≤ taxi ≤ 4 h)"); ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/04_winter_penalty.png", dpi=150)
