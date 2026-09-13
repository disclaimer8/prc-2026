"""Fleet mix per hub: type families, neo/MAX penetration, Heavy/Super share; 2025 and Jan/Jul YoY."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, HUBS, CITY, OUT, TYPE2FAM

df = load_all().select("hub", "month", "src", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt",
                       typ=pl.coalesce("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt")).collect()
df = df.with_columns(fam=pl.col("typ").replace_strict(TYPE2FAM, default="Other"))
print("rows:", df.height, " type null:", df["typ"].null_count())
print("types in 'Other' (top 25 by movements):")
print(df.filter(pl.col("fam") == "Other").group_by("typ").len().sort("len", descending=True).head(25).to_dicts())
y = df.filter(pl.col("src") == "training")
fams = ["A320neo family", "A320ceo family", "737 MAX", "737 NG/classic", "A220", "E-Jet", "CRJ", "Turboprop", "Widebody", "Other"]
share = y.group_by("hub", "fam").len().with_columns(pct=pl.col("len") / pl.col("len").sum().over("hub") * 100)
w = share.pivot(on="fam", index="hub", values="pct").fill_null(0).select(["hub"] + fams).sort("hub")
print("\n=== 2025 full year: % of movements by type family ===")
with pl.Config(tbl_cols=-1, float_precision=1, tbl_width_chars=250):
    print(w)
w.write_csv(OUT + "/scripts/08_fleet_family_2025.csv")

def pen(d, by):
    return d.group_by(by).agg(
        mov=pl.len(),
        neo_in_a320fam=(pl.col("fam") == "A320neo family").sum() / pl.col("fam").is_in(["A320neo family", "A320ceo family"]).sum() * 100,
        max_in_737=(pl.col("fam") == "737 MAX").sum() / pl.col("fam").is_in(["737 MAX", "737 NG/classic"]).sum() * 100,
        widebody_pct=(pl.col("fam") == "Widebody").mean() * 100,
        heavy_super_pct=pl.col("WK_TBL_CAT_flt").is_in(["H", "J"]).sum() / pl.col("WK_TBL_CAT_flt").is_not_null().sum() * 100,
        super_A388_mov=(pl.col("typ") == "A388").sum(),
    )
print("\n=== 2025 full year penetration per hub ===")
with pl.Config(tbl_cols=-1, float_precision=1, tbl_width_chars=250):
    print(pen(y, "hub").sort("neo_in_a320fam", descending=True))
    print("all hubs 2025:", pen(y.with_columns(k=pl.lit(1)), "k").to_dicts())
    p = pen(df.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])), ["hub", "month"]).sort("hub", "month")
    print("\n=== Jan/Jul 2025 vs 2026 (31 vs 31 days) ===")
    print(p.pivot(on="month", index="hub", values=["neo_in_a320fam", "max_in_737", "widebody_pct", "heavy_super_pct"]).select(
        "hub", "neo_in_a320fam_2025-01", "neo_in_a320fam_2026-01", "neo_in_a320fam_2025-07", "neo_in_a320fam_2026-07",
        "max_in_737_2025-07", "max_in_737_2026-07", "widebody_pct_2025-07", "widebody_pct_2026-07", "heavy_super_pct_2025-07", "heavy_super_pct_2026-07"))
    allp = pen(df.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])), ["month"]).sort("month")
    print(allp)
    print("\nA380 movements per hub per period:")
    print(p.pivot(on="month", index="hub", values="super_A388_mov").sort("hub"))
# monthly neo share all hubs, to show trend
m = pen(df, "month").sort("month")
print("\n=== Monthly neo share in A320 family & MAX share in 737, all 10 hubs ===")
with pl.Config(tbl_rows=20, float_precision=1):
    print(m.select("month", "mov", "neo_in_a320fam", "max_in_737", "widebody_pct", "heavy_super_pct"))
print("\n=== Top 10 types 2025 all hubs ===")
t = y.group_by("typ").len().with_columns(pct=pl.col("len") / y.height * 100).sort("len", descending=True).head(10)
print(t)
print("\n=== Widebody type mix 2025 (top 10 widebody types) ===")
print(y.filter(pl.col("fam") == "Widebody").group_by("typ").len().sort("len", descending=True).head(10))

# chart: monthly neo & MAX penetration, all hubs
fig, ax = plt.subplots(figsize=(9, 4.5))
months = m["month"].to_list(); x = np.arange(len(months))
ax.plot(x[:12], m["neo_in_a320fam"][:12], "-o", color="#5b85aa", label="A320neo share of A320-family movements")
ax.plot(x[12:], m["neo_in_a320fam"][12:], "o", color="#5b85aa")
ax.plot(x[:12], m["max_in_737"][:12], "-o", color="#d9822b", label="737 MAX share of 737 movements")
ax.plot(x[12:], m["max_in_737"][12:], "o", color="#d9822b")
for xi, v in zip(x, m["neo_in_a320fam"]):
    if xi in (0, 6, 12, 13): ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
for xi, v in zip(x, m["max_in_737"]):
    if xi in (0, 6, 12, 13): ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)
ax.set_xticks(x, [mm[2:].replace("-", "/") for mm in months], fontsize=8)
ax.axvline(11.5, color="#bbb", ls="--", lw=0.8); ax.text(11.6, ax.get_ylim()[0] + 1, "2026 (Jan, Jul only)", fontsize=8, color="#777")
ax.set_ylabel("%"); ax.set_title("New-generation narrowbodies at 10 European hubs, by month")
ax.legend(frameon=False, loc="upper left", fontsize=8)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Airport movements (arr+dep), ICAO type codes.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "/charts/03_neo_max_share.png", dpi=150)

# chart: family mix per hub
order = w.sort("Widebody")["hub"].to_list()
cols = {"A320neo family": "#2f6f9f", "A320ceo family": "#9fc3de", "737 MAX": "#c0602a", "737 NG/classic": "#efb58a", "A220": "#6aa56a",
        "E-Jet": "#9dcf9d", "CRJ": "#c8e3c8", "Turboprop": "#b9a3cf", "Widebody": "#333333", "Other": "#cccccc"}
fig, ax = plt.subplots(figsize=(10, 5.5)); left = np.zeros(len(order))
for f in fams:
    v = np.array([w.filter(pl.col("hub") == h)[f][0] for h in order])
    ax.barh(range(len(order)), v, left=left, color=cols[f], label=f)
    for i, (l, vv) in enumerate(zip(left, v)):
        if vv >= 7: ax.text(l + vv / 2, i, f"{vv:.0f}", ha="center", va="center", fontsize=7, color="white" if f in ("A320neo family", "737 MAX", "Widebody") else "#222")
    left += v
ax.set_yticks(range(len(order)), [f"{CITY[h]} ({h})" for h in order]); ax.set_xlim(0, 100)
ax.set_xlabel("% of airport movements, 2025"); ax.set_title("Fleet mix at 10 European hubs, 2025")
ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Grouping of ICAO type codes: see REPORT.md.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "/charts/04_fleet_mix.png", dpi=150)
print("charts saved")
