"""Diversions: ADES_flt != ADES_FILED_flt. Two views seen from the hubs:
 (A) flights that DIVERTED INTO a hub (ARR row at hub, filed destination elsewhere);
 (B) flights that DEPARTED a hub and ended elsewhere than filed (DEP row), incl. return-to-origin.
Dedup by FLIGHT_ID where available (a hub-to-hub flight appears twice)."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, HUBS, CITY, OUT, airports
df = load_all().select("hub", "month", "PHASE_mvt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "date_utc", "MVT_ID_mvt").collect()
ap = airports().select("icao", "municipality", "iso_country")
matched = df.filter(pl.col("ADES_flt").is_not_null())
div = matched.filter(pl.col("ADES_flt") != pl.col("ADES_FILED_flt"))
print("movements with NM match:", matched.height, " with ADES_flt != ADES_FILED_flt:", div.height, f"({div.height/matched.height*100:.3f}%)")
A = div.filter(pl.col("PHASE_mvt") == "ARR")
B = div.filter(pl.col("PHASE_mvt") == "DEP")
print("\n(A) diverted INTO hub, ARR rows:", A.height, "| filed destination is another of the 10 hubs:", A.filter(pl.col("ADES_FILED_flt").is_in(HUBS)).height)
print("(B) departed hub, diverted:", B.height, "| of which returned to the hub (ADES_flt == hub):", B.filter(pl.col("ADES_flt") == pl.col("hub")).height)

y = matched.filter(pl.col("month").str.starts_with("2025"))
yA = A.filter(pl.col("month").str.starts_with("2025")); yB = B.filter(pl.col("month").str.starts_with("2025"))
print("\n=== 2025 per hub: diversions IN (per 10,000 arrivals) and OUT (per 10,000 departures) ===")
arr = y.filter(pl.col("PHASE_mvt") == "ARR").group_by("hub").len().rename({"len": "arr"})
dep = y.filter(pl.col("PHASE_mvt") == "DEP").group_by("hub").len().rename({"len": "dep"})
t = arr.join(dep, on="hub").join(yA.group_by("hub").agg(div_in=pl.len()), on="hub", how="left").join(
    yB.group_by("hub").agg(div_out=pl.len(), rto=(pl.col("ADES_flt") == pl.col("hub")).sum()), on="hub", how="left").fill_null(0)
t = t.with_columns(in_per10k=pl.col("div_in") / pl.col("arr") * 1e4, out_per10k=pl.col("div_out") / pl.col("dep") * 1e4, rto_per10k=pl.col("rto") / pl.col("dep") * 1e4).sort("in_per10k", descending=True)
with pl.Config(float_precision=1, tbl_cols=-1):
    print(t)
print("\n=== (A) 2025: where were flights diverting INTO each hub originally going? top-3 filed destinations ===")
fa = yA.group_by("hub", "ADES_FILED_flt").len().join(ap, left_on="ADES_FILED_flt", right_on="icao", how="left").sort("len", descending=True)
for h in HUBS:
    x = fa.filter(pl.col("hub") == h).head(3)
    print(h, "; ".join(f"{r['ADES_FILED_flt']} {r['municipality']} ({r['iso_country']}) {r['len']}" for r in x.iter_rows(named=True)))
print("\n=== (B) 2025: departures from hub that diverted (excl. return to hub): top-3 actual destinations with filed dest ===")
fb = yB.filter(pl.col("ADES_flt") != pl.col("hub")).group_by("hub", "ADES_FILED_flt", "ADES_flt").len().sort("len", descending=True)
for h in HUBS:
    x = fb.filter(pl.col("hub") == h).head(3)
    print(h, "; ".join(f"filed {r['ADES_FILED_flt']} -> landed {r['ADES_flt']}: {r['len']}" for r in x.iter_rows(named=True)))
# seasonality: all hubs, monthly, per 10k movements
mon = matched.group_by("month").len().join(div.group_by("month").agg(div=pl.len(), div_in=(pl.col("PHASE_mvt") == "ARR").sum()), on="month").with_columns(
    per10k=pl.col("div") / pl.col("len") * 1e4).sort("month")
print("\n=== Monthly diversion rows per 10,000 matched movements, all hubs ===")
with pl.Config(tbl_rows=20, float_precision=2):
    print(mon)
# top days (diversion clusters)
dd = A.group_by("hub", "date_utc").len().sort("len", descending=True).head(12)
print("\n=== Top hub-days by diversions INTO hub ==="); print(dd)
dd2 = B.group_by("hub", "date_utc").len().sort("len", descending=True).head(8)
print("\n=== Top hub-days by diverted departures ==="); print(dd2)
fig, ax = plt.subplots(figsize=(9, 4))
mm = mon["month"].to_list(); x = np.arange(len(mm))
ax.bar(x, mon["per10k"], color=["#5b85aa"] * 12 + ["#d9822b"] * (len(mm) - 12))
for xi, v in zip(x, mon["per10k"]): ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x, [s[2:].replace("-", "/") for s in mm], fontsize=8)
ax.set_ylabel("per 10,000 movements"); ax.set_title("Diverted flights touching 10 European hubs, by month")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Diversion = NM actual destination differs from filed destination.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT + "/charts/07_diversions_monthly.png", dpi=150)
print("chart saved")
