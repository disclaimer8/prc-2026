import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from common import *
df = base()
c = df.filter("clean")
dep = c.filter(pl.col("PHASE_mvt")=="DEP")
# A) variance explained by stand vs runway (same as 03, re-computed here for the chart)
d = dep.select("APT","STAND_mvt","RUNWAY_mvt","taxi_min")
def r2(keys, name):
    g = d.with_columns(pl.col("taxi_min").mean().over(keys).alias("m"), pl.col("taxi_min").mean().over("APT").alias("ma"))
    return g.group_by("APT").agg((1 - ((pl.col("taxi_min")-pl.col("m"))**2).sum()/((pl.col("taxi_min")-pl.col("ma"))**2).sum()).round(3).alias(name))
R = r2(["APT","RUNWAY_mvt"],"runway").join(r2(["APT","STAND_mvt"],"stand"), on="APT").join(r2(["APT","STAND_mvt","RUNWAY_mvt"],"stand_x_runway"), on="APT").sort("stand_x_runway")
print("R2 (in-sample) of DEP taxi-out explained by group means:"); print(R)
fig, ax = plt.subplots(figsize=(9,5)); y=np.arange(R.height); h=0.27
ax.barh(y+h, R["runway"], h, color="#9ac1e8", label="runway only"); ax.barh(y, R["stand"], h, color="#4a7fb5", label="stand only"); ax.barh(y-h, R["stand_x_runway"], h, color="#1a365d", label="stand × runway")
ax.set_yticks(y); ax.set_yticklabels(R["APT"]); ax.set_xlabel("share of within-airport taxi-out variance explained (in-sample R²)")
ax.set_title("Where the aircraft is parked matters more than which runway it uses\n(departures 2025, 1 min ≤ taxi ≤ 4 h)"); ax.legend(fontsize=8, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/12_variance_stand_vs_runway.png", dpi=150)
# B) punctuality by segment
P = df.with_columns(((pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()/60).alias("delay")).filter((pl.col("PHASE_mvt")=="DEP") & pl.col("delay").is_between(-360,1440))
S = P.filter(pl.col("MARKET_SEGMENT_flt").is_in(["Mainline","Lowcost","Regional","Cargo","Business"])).group_by("MARKET_SEGMENT_flt").agg(pl.len().alias("N"),
    ((pl.col("delay")<=15).mean()*100).round(1).alias("le15"), ((pl.col("delay").is_between(15,60,closed="right")).mean()*100).round(1).alias("d15_60"), ((pl.col("delay")>60).mean()*100).round(1).alias("gt60"), ((pl.col("delay")>180).mean()*100).round(2).alias("gt180")).sort("gt60")
print("DEP delay bands by segment (window -6h..+24h):"); print(S)
fig, ax = plt.subplots(figsize=(9,4.2)); y=np.arange(S.height)
ax.barh(y, S["le15"], color="#9ac1e8", label="≤ 15 min"); ax.barh(y, S["d15_60"], left=S["le15"], color="#4a7fb5", label="15–60 min"); ax.barh(y, S["gt60"], left=S["le15"]+S["d15_60"], color="#c05621", label="> 60 min")
for i,(g,n) in enumerate(zip(S["gt60"], S["N"])): ax.text(101, i, f">60 min: {g:.1f}%  (N={n:,})", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(S["MARKET_SEGMENT_flt"]); ax.set_xlim(0,135); ax.set_xlabel("% of departures by off-block delay vs schedule")
ax.set_title("Same on-time share, different tails: departure delay by market segment, 2025"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4,-0.2), ncol=3, frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/13_delay_bands_by_segment.png", dpi=150)
# C) hour of day taxi-out, selected airports
H = dep.group_by("APT","hour_local").agg(pl.len().alias("N"), pl.col("taxi_min").median().alias("med")).filter(pl.col("N")>=3000).sort("APT","hour_local")
fig, ax = plt.subplots(figsize=(9,5))
for a, col in [("EGLL","#1a365d"),("LIRF","#c05621"),("LFPG","#2b6cb0"),("EHAM","#68a063"),("LSZH","#999999")]:
    s = H.filter(pl.col("APT")==a); ax.plot(s["hour_local"], s["med"], marker="o", ms=3, color=col, label=a)
ax.set_xlabel("local hour of off-block"); ax.set_ylabel("median taxi-out, min"); ax.set_xticks(range(0,24,2))
ax.set_title("Taxi-out through the day, 2025 (hours with ≥3,000 departures)"); ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/14_taxiout_by_hour.png", dpi=150)
print(H.filter(pl.col("APT").is_in(["EGLL","EHAM"])))
