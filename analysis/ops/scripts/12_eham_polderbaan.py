import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
df = base()
c = df.filter(pl.col("clean") & (pl.col("APT")=="EHAM") & (pl.col("PHASE_mvt")=="ARR"))
m = c.group_by("month").agg(pl.len().alias("N"), (pl.col("RUNWAY_mvt")=="18R").mean().mul(100).round(1).alias("pct_18R"), pl.col("taxi_min").median().round(2).alias("median_taxi_in"),
                            pl.col("taxi_min").filter(pl.col("RUNWAY_mvt")!="18R").median().round(2).alias("median_not18R"), pl.col("taxi_min").filter(pl.col("RUNWAY_mvt")=="18R").median().round(2).alias("median_18R")).sort("month")
print("EHAM arrivals, clean (1 min..4 h), by local month"); print(m)
print("Pearson r (monthly pct_18R vs monthly median taxi-in):", round(m.select(pl.corr("pct_18R","median_taxi_in")).item(),3), " Spearman:", round(m.select(pl.corr("pct_18R","median_taxi_in", method="spearman")).item(),3))
d = c.group_by("date_local").agg(pl.len().alias("N"), (pl.col("RUNWAY_mvt")=="18R").mean().mul(100).alias("pct_18R"), pl.col("taxi_min").median().alias("med")).filter(pl.col("N")>=200)
print("daily (N>=200 arrivals/day, days =", d.height, ") Spearman pct_18R vs daily median taxi-in:", round(d.select(pl.corr("pct_18R","med", method="spearman")).item(),3))
fig, ax = plt.subplots(figsize=(7,4.8))
ax.scatter(d["pct_18R"], d["med"], s=10, alpha=.5, color="#2b6cb0")
ax.set_xlabel("share of the day's arrivals landing on 18R (Polderbaan), %"); ax.set_ylabel("daily median taxi-in, min")
ax.set_title(f"Schiphol: taxi-in depends on how many land on the Polderbaan\n2025, one dot = one day (N = {d.height} days, ≥200 arrivals each)", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/11_eham_polderbaan_taxi_in.png", dpi=150)
