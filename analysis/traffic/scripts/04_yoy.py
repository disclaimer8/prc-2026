"""Jan/Jul 2026 vs Jan/Jul 2025: movements per hub, per segment. Day-coverage, weekday-mix and
disruption-day sensitivity checks."""
import polars as pl, datetime as dt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, HUBS, CITY, OUT

lf = load_all().filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"]))
df = lf.select("hub", "PHASE_mvt", "date_utc", "month", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt").collect()
df = df.with_columns(mon=pl.col("month").str.slice(5, 2), yr=pl.col("month").str.slice(0, 4),
                     wd=pl.col("date_utc").dt.weekday())
DISRUPT = [("EHAM", dt.date(2026, 1, d)) for d in (4, 5, 6, 7)]

daily = df.group_by("hub", "yr", "mon", "date_utc", "wd").len()
cov = daily.group_by("hub", "yr", "mon").agg(ndays=pl.len(), mov=pl.col("len").sum(),
                                             med=pl.col("len").median())
print("Coverage (days with data) per hub/period:")
print(cov.group_by("yr", "mon").agg(pl.col("ndays").min().alias("min_days"), pl.col("ndays").max().alias("max_days")).sort("yr", "mon"))

def yoy(d, label):
    t = d.group_by("hub", "yr", "mon").agg(mov=pl.col("len").sum(), ndays=pl.len(),
                                            med=pl.col("len").median(),
                                            # weekday-balanced mean: mean over weekdays of mean daily count
                                            )
    wdb = d.group_by("hub", "yr", "mon", "wd").agg(pl.col("len").mean()).group_by("hub", "yr", "mon").agg(wdbal=pl.col("len").mean())
    t = t.join(wdb, on=["hub", "yr", "mon"])
    a = t.filter(pl.col("yr") == "2025").drop("yr"); b = t.filter(pl.col("yr") == "2026").drop("yr")
    j = a.join(b, on=["hub", "mon"], suffix="_26").with_columns(
        per_day_25=pl.col("mov") / pl.col("ndays"), per_day_26=pl.col("mov_26") / pl.col("ndays_26"))
    j = j.with_columns(
        yoy_perday_pct=(pl.col("per_day_26") / pl.col("per_day_25") - 1) * 100,
        yoy_median_pct=(pl.col("med_26") / pl.col("med") - 1) * 100,
        yoy_wdbal_pct=(pl.col("wdbal_26") / pl.col("wdbal") - 1) * 100,
    ).sort("mon", "hub")
    print(f"\n=== YoY by hub ({label}) ===")
    with pl.Config(tbl_rows=40, tbl_cols=-1, float_precision=1, tbl_width_chars=250):
        print(j.select("mon", "hub", "ndays", "ndays_26", "mov", "mov_26", "per_day_25", "per_day_26", "yoy_perday_pct", "yoy_median_pct", "yoy_wdbal_pct"))
    return j

j_all = yoy(daily, "all days")
daily_x = daily.filter(~pl.struct("hub", "date_utc").is_in([{"hub": h, "date_utc": d} for h, d in DISRUPT]))
j_x = yoy(daily_x, "excluding EHAM 4-7 Jan 2026 disruption days")

tot = daily.group_by("yr", "mon").agg(mov=pl.col("len").sum(), nd=pl.col("date_utc").n_unique()).sort("mon", "yr")
print("\n=== All 10 hubs total ===")
print(tot)
for mon in ["01", "07"]:
    a = tot.filter((pl.col("yr") == "2025") & (pl.col("mon") == mon))["mov"][0]
    b = tot.filter((pl.col("yr") == "2026") & (pl.col("mon") == mon))["mov"][0]
    print(f"month {mon}: 2025={a} 2026={b} yoy={100*(b/a-1):+.2f}% (31 vs 31 days)")
wd = daily.group_by("yr", "mon", "wd").agg(n_days=pl.col("date_utc").n_unique()).sort("mon", "yr", "wd")
print("\nWeekday mix (1=Mon..7=Sun), number of calendar days:")
print(wd.pivot(on="wd", index=["yr", "mon"], values="n_days"))

# segments, all hubs
seg = df.with_columns(seg=pl.col("MARKET_SEGMENT_flt").fill_null("(unmatched)")).group_by("seg", "yr", "mon").len()
s = seg.pivot(on="yr", index=["mon", "seg"], values="len").fill_null(0).with_columns(
    yoy_pct=(pl.col("2026") / pl.col("2025") - 1) * 100).sort("mon", "2025", descending=[False, True])
print("\n=== YoY by market segment, all 10 hubs (31 vs 31 days) ===")
with pl.Config(tbl_rows=40, float_precision=1):
    print(s)
# segment x hub
sh = df.filter(pl.col("MARKET_SEGMENT_flt").is_in(["Mainline", "Lowcost", "Regional", "Cargo", "Business"])).group_by("hub", "MARKET_SEGMENT_flt", "yr", "mon").len()
sh = sh.pivot(on="yr", index=["mon", "hub", "MARKET_SEGMENT_flt"], values="len").fill_null(0).with_columns(
    yoy_pct=(pl.col("2026") / pl.col("2025") - 1) * 100)
print("\n=== YoY % by hub x segment (31 vs 31 days) ===")
with pl.Config(tbl_rows=40, tbl_cols=-1, float_precision=1):
    print(sh.pivot(on="MARKET_SEGMENT_flt", index=["mon", "hub"], values="yoy_pct").sort("mon", "hub"))
    print(sh.pivot(on="MARKET_SEGMENT_flt", index=["mon", "hub"], values="2026").sort("mon", "hub"))
j_all.write_csv(OUT + "/scripts/04_yoy_hub.csv")

# chart
fig, ax = plt.subplots(figsize=(9, 5))
hubs = sorted(HUBS, key=lambda h: j_all.filter((pl.col("hub") == h) & (pl.col("mon") == "07"))["yoy_perday_pct"][0])
import numpy as np
y = np.arange(len(hubs))
for k, (mon, c, lab) in enumerate([("01", "#6b8fb3", "January"), ("07", "#d9822b", "July")]):
    v = [j_all.filter((pl.col("hub") == h) & (pl.col("mon") == mon))["yoy_perday_pct"][0] for h in hubs]
    ax.barh(y + (k - 0.5) * 0.38, v, height=0.36, color=c, label=f"{lab} 2026 vs {lab} 2025")
    for yy, vv in zip(y + (k - 0.5) * 0.38, v):
        ax.text(vv + (0.15 if vv >= 0 else -0.15), yy, f"{vv:+.1f}%", va="center", ha="left" if vv >= 0 else "right", fontsize=8)
ax.set_yticks(y, [f"{CITY[h]} ({h})" for h in hubs])
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("Change in airport movements (arrivals + departures), %, 31 vs 31 days")
ax.set_title("Traffic change year on year at 10 European hubs")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False); ax.set_xlim(-15, 9)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Amsterdam Jan 2026 includes 4 disrupted days (4-7 Jan).", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT + "/charts/01_yoy_by_hub.png", dpi=150)
print("chart saved")
