"""Low-cost share of movements by hub: monthly 2025 + Jan/Jul 2026. Denominator = movements with a market segment."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, HUBS, CITY, OUT
df = load_all().select("hub", "month", "MARKET_SEGMENT_flt").collect().filter(pl.col("MARKET_SEGMENT_flt").is_not_null())
m = df.group_by("hub", "month").agg(n=pl.len(), lc=(pl.col("MARKET_SEGMENT_flt") == "Lowcost").sum()).with_columns(pct=pl.col("lc") / pl.col("n") * 100)
with pl.Config(tbl_rows=20, tbl_cols=-1, float_precision=1, tbl_width_chars=250):
    print("=== Low-cost % of movements (segment known), by month ===")
    print(m.pivot(on="hub", index="month", values="pct").sort("month").select(["month"] + HUBS))
y = df.filter(pl.col("month").str.starts_with("2025")).group_by("hub").agg(n=pl.len(), lc=(pl.col("MARKET_SEGMENT_flt") == "Lowcost").sum()).with_columns(pct=pl.col("lc") / pl.col("n") * 100).sort("pct", descending=True)
print("\n=== 2025 full year ==="); print(y)
ya = df.filter(pl.col("month").str.starts_with("2025"))
print("all hubs 2025 lowcost %:", round((ya["MARKET_SEGMENT_flt"] == "Lowcost").mean() * 100, 2))
p = m.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).pivot(on="month", index="hub", values="pct").with_columns(
    d_jan_pp=pl.col("2026-01") - pl.col("2025-01"), d_jul_pp=pl.col("2026-07") - pl.col("2025-07")).sort("d_jul_pp", descending=True)
print("\n=== Low-cost share change, percentage points (31 vs 31 days) ===")
with pl.Config(float_precision=1, tbl_cols=-1):
    print(p.select("hub", "2025-01", "2026-01", "d_jan_pp", "2025-07", "2026-07", "d_jul_pp"))
# seasonal amplitude 2025: Jul share - Jan share
# chart: 2025 annual share + Jul 2025/Jul 2026 markers
order = y.sort("pct")["hub"].to_list()
fig, ax = plt.subplots(figsize=(9, 5))
yy = np.arange(len(order))
ax.barh(yy, [y.filter(pl.col("hub") == h)["pct"][0] for h in order], color="#cfd9e3", label="2025 full year")
ax.scatter([p.filter(pl.col("hub") == h)["2025-07"][0] for h in order], yy, color="#5b85aa", zorder=3, label="July 2025")
ax.scatter([p.filter(pl.col("hub") == h)["2026-07"][0] for h in order], yy, color="#d9822b", marker="D", zorder=3, label="July 2026")
for i, h in enumerate(order):
    v = y.filter(pl.col("hub") == h)["pct"][0]; ax.text(max(v, p.filter(pl.col("hub") == h)["2026-07"][0]) + 1.5, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_yticks(yy, [f"{CITY[h]} ({h})" for h in order]); ax.set_xlabel("Low-cost share of airport movements, %")
ax.set_title("Low-cost carriers' share at 10 European hubs")
ax.legend(frameon=False, loc="lower right")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Market segment as classified by EUROCONTROL NM.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "/charts/06_lowcost_share.png", dpi=150)
print("chart saved")
