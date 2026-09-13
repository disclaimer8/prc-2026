"""Diurnal and weekly profiles in LOCAL time, night share, peak hours. 2025 full year (365 days, all hubs)."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, HUBS, CITY, OUT

df = load_all().select("hub", "src", "month", "PHASE_mvt", "t_local", "date_local", "MARKET_SEGMENT_flt").collect()
df = df.with_columns(hr=pl.col("t_local").dt.hour(), wd=pl.col("date_local").dt.weekday())
y = df.filter(pl.col("src") == "training")
nd = y.group_by("hub").agg(pl.col("date_local").n_unique().alias("local_days"))
print("local dates per hub in 2025 file set (edge days at year boundary can appear):"); print(nd.sort("hub"))

night = y.group_by("hub").agg(
    mov=pl.len(),
    night_23_06_pct=((pl.col("hr") >= 23) | (pl.col("hr") < 6)).mean() * 100,
    late_22_24_pct=(pl.col("hr") >= 22).mean() * 100,
    early_05_07_pct=((pl.col("hr") >= 5) & (pl.col("hr") < 7)).mean() * 100,
    night_dep_pct=(((pl.col("hr") >= 23) | (pl.col("hr") < 6)) & (pl.col("PHASE_mvt") == "DEP")).sum() / (pl.col("PHASE_mvt") == "DEP").sum() * 100,
    night_arr_pct=(((pl.col("hr") >= 23) | (pl.col("hr") < 6)) & (pl.col("PHASE_mvt") == "ARR")).sum() / (pl.col("PHASE_mvt") == "ARR").sum() * 100,
).sort("night_23_06_pct", descending=True)
print("\n=== 2025: share of movements 23:00-05:59 local time ===")
with pl.Config(float_precision=1, tbl_cols=-1):
    print(night)

hourly = y.group_by("hub", "hr").len().with_columns(avg_per_day=pl.col("len") / 365, pct=pl.col("len") / pl.col("len").sum().over("hub") * 100)
peak = hourly.sort("len", descending=True).group_by("hub").first().select("hub", peak_local_hour="hr", avg_mov_in_peak_hour="avg_per_day", pct_in_peak_hour="pct").sort("hub")
print("\n=== 2025: busiest local clock hour (average over 365 days) ===")
with pl.Config(float_precision=1):
    print(peak)
# max single clock hour in 2025
hh = y.with_columns(h=pl.col("t_local").dt.truncate("1h")).group_by("hub", "h").len()
mx = hh.sort("len", descending=True).group_by("hub").first().sort("hub")
print("\n=== 2025: single busiest local clock hour of the year ===")
print(mx)
# peak-to-mean ratio: p95 of hourly counts (06-22 local) vs mean
pr = hh.with_columns(hr=pl.col("h").dt.hour()).filter((pl.col("hr") >= 6) & (pl.col("hr") < 23)).group_by("hub").agg(
    mean_06_23=pl.col("len").mean(), p95=pl.col("len").quantile(0.95), cv=pl.col("len").std() / pl.col("len").mean())
with pl.Config(float_precision=2):
    print("\nHourly counts 06-23 local: mean, p95, CV (hours with >=1 movement only)"); print(pr.sort("cv"))

daily = y.group_by("hub", "date_local", "wd").len()
wk = daily.group_by("hub", "wd").agg(pl.col("len").mean()).with_columns(idx=pl.col("len") / pl.col("len").mean().over("hub") * 100)
print("\n=== 2025 weekday index (mean daily movements by weekday / mean over weekdays x100), 1=Mon ===")
with pl.Config(float_precision=1, tbl_cols=-1):
    wkp = wk.pivot(on="wd", index="hub", values="idx").select(["hub"] + [str(i) for i in range(1, 8)]).sort("hub")
    print(wkp)
    print("spread max-min weekday index:", wkp.with_columns(spread=pl.max_horizontal([str(i) for i in range(1, 8)]) - pl.min_horizontal([str(i) for i in range(1, 8)])).select("hub", "spread").sort("spread").to_dicts())

# night share YoY
ny = df.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).group_by("hub", "month").agg(
    night_pct=((pl.col("hr") >= 23) | (pl.col("hr") < 6)).mean() * 100)
print("\n=== Night share (23-06 local) Jan/Jul 2025 vs 2026 ===")
with pl.Config(float_precision=1, tbl_cols=-1):
    print(ny.pivot(on="month", index="hub", values="night_pct").sort("hub").select("hub", "2025-01", "2026-01", "2025-07", "2026-07"))

# low-cost by hour: share of lowcost in first wave 06-08 vs overall
lc = y.filter(pl.col("MARKET_SEGMENT_flt").is_not_null()).group_by("hub").agg(
    lc_all=(pl.col("MARKET_SEGMENT_flt") == "Lowcost").mean() * 100,
    lc_night=((pl.col("MARKET_SEGMENT_flt") == "Lowcost") & ((pl.col("hr") >= 23) | (pl.col("hr") < 6))).sum() / ((pl.col("hr") >= 23) | (pl.col("hr") < 6)).sum() * 100,
    cargo_night=((pl.col("MARKET_SEGMENT_flt") == "Cargo") & ((pl.col("hr") >= 23) | (pl.col("hr") < 6))).sum() / ((pl.col("hr") >= 23) | (pl.col("hr") < 6)).sum() * 100,
    cargo_all=(pl.col("MARKET_SEGMENT_flt") == "Cargo").mean() * 100)
print("\n=== 2025 segment share overall vs at night ==="); 
with pl.Config(float_precision=1):
    print(lc.sort("hub"))

# heatmap
M = np.zeros((len(HUBS), 24))
order = night["hub"].to_list()[::-1]
for i, h in enumerate(order):
    for r in hourly.filter(pl.col("hub") == h).iter_rows(named=True):
        M[i, r["hr"]] = r["avg_per_day"]
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(M, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(len(order)), [f"{CITY[h]} ({h})" for h in order])
ax.set_xticks(range(24), [f"{i:02d}" for i in range(24)], fontsize=8)
ax.set_xlabel("Local hour"); ax.set_title("Average movements per local clock hour, 2025 (sorted by night share)")
for i in range(len(order)):
    for j in range(24):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color="black" if M[i, j] < M.max() * 0.6 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.025); cb.set_label("movements / hour")
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Arrivals + departures, 365 days of 2025.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "/charts/05_hourly_heatmap.png", dpi=150)
print("chart saved")
