"""Small multiples of daily movements per hub (UTC days), 2025 + Jan/Jul 2026, with low days (<70% of hub-month median) marked."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from common import load_all, HUBS, CITY, OUT
d = load_all().group_by("hub", "date_utc").len().collect().sort("hub", "date_utc")
d = d.with_columns(month=pl.col("date_utc").dt.strftime("%Y-%m")).with_columns(med=pl.col("len").median().over("hub", "month")).with_columns(low=pl.col("len") < 0.7 * pl.col("med"))
print("low days:", d.filter(pl.col("low")).select("hub", "date_utc", "len", "med").to_dicts())
fig, axes = plt.subplots(5, 2, figsize=(12, 11), sharex=True)
for ax, h in zip(axes.flat, HUBS):
    x = d.filter(pl.col("hub") == h)
    for part, sel in [("2025", x.filter(pl.col("date_utc").dt.year() == 2025)), ("2026-01", x.filter(pl.col("month") == "2026-01")), ("2026-07", x.filter(pl.col("month") == "2026-07"))]:
        ax.plot(sel["date_utc"].to_list(), sel["len"].to_list(), lw=0.8, color="#5b85aa" if part == "2025" else "#d9822b")
    lo = x.filter(pl.col("low"))
    ax.scatter(lo["date_utc"].to_list(), lo["len"].to_list(), color="#b2182b", s=14, zorder=3)
    for r in lo.iter_rows(named=True):
        ax.annotate(r["date_utc"].strftime("%d %b %y"), (r["date_utc"], r["len"]), fontsize=6, xytext=(3, -2), textcoords="offset points", color="#b2182b")
    ax.set_title(f"{CITY[h]} ({h})", fontsize=9, loc="left"); ax.set_ylim(0, None)
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=7)
axes.flat[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
fig.suptitle("Daily airport movements, 2025 (blue) and Jan/Jul 2026 (orange); red = day below 70% of that month's median", fontsize=10)
fig.text(0.01, 0.005, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. UTC days. Military, head-of-state and sensitive flights removed by the source.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(OUT + "/charts/09_daily_small_multiples.png", dpi=150)
print("chart saved")
