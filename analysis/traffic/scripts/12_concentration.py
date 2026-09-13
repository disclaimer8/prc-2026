"""Hub dominance using the anonymised operator hash only: top-1 / top-3 share, HHI. No operator is identified."""
import polars as pl, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import DATA, HUBS, CITY, OUT
cols = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"]
tr = pl.read_parquet(sorted(glob.glob(f"{DATA}/training_2025-*.parquet")), columns=cols)
rk = pl.read_parquet(f"{DATA}/ranking.parquet", columns=cols)
d = pl.concat([tr, rk]).select(
    hub=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
    month=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"), op=pl.col("AIRCRAFT_OPERATOR_flt").cast(pl.Utf8), seg=pl.col("MARKET_SEGMENT_flt"))
print("rows:", d.height, " operator null:", d["op"].null_count())
d = d.filter(pl.col("op").is_not_null())

def conc(x, by):
    s = x.group_by(by + ["op"]).len().with_columns(sh=pl.col("len") / pl.col("len").sum().over(by))
    s = s.sort("sh", descending=True)
    return s.group_by(by).agg(
        n_ops=pl.len(), top1_pct=pl.col("sh").max() * 100, top3_pct=pl.col("sh").head(3).sum() * 100,
        hhi=(pl.col("sh") ** 2).sum() * 10000,
        n_ops_ge_1pct=(pl.col("sh") >= 0.01).sum())
y = d.filter(pl.col("month").str.starts_with("2025"))
c = conc(y, ["hub"]).sort("top1_pct", descending=True)
print("\n=== 2025: operator concentration per hub (movements with operator known) ===")
with pl.Config(float_precision=1, tbl_cols=-1):
    print(c)
p = conc(d.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])), ["hub", "month"])
print("\n=== top-1 operator share %, Jan/Jul 2025 vs 2026 ===")
with pl.Config(float_precision=1, tbl_cols=-1):
    print(p.pivot(on="month", index="hub", values="top1_pct").sort("hub").select("hub", "2025-01", "2026-01", "2025-07", "2026-07"))
    print(p.pivot(on="month", index="hub", values="hhi").sort("hub").select("hub", "2025-01", "2026-01", "2025-07", "2026-07"))
c.write_csv(OUT + "/scripts/12_concentration_2025.csv")
fig, ax = plt.subplots(figsize=(9, 5))
o = c.sort("top1_pct")
ax.barh(range(o.height), o["top3_pct"], color="#cfd9e3", label="Top-3 operators")
ax.barh(range(o.height), o["top1_pct"], color="#2f6f9f", label="Largest operator")
for i, r in enumerate(o.iter_rows(named=True)):
    ax.text(r["top1_pct"] - 1, i, f"{r['top1_pct']:.0f}%", ha="right", va="center", color="white", fontsize=8)
    ax.text(r["top3_pct"] + 1, i, f"{r['top3_pct']:.0f}%", va="center", fontsize=8)
ax.set_yticks(range(o.height), [f"{CITY[h]} ({h})" for h in o["hub"]]); ax.set_xlim(0, 100)
ax.set_xlabel("Share of airport movements, 2025, %"); ax.set_title("How dominated is each hub by its largest operators?")
ax.legend(frameon=False, loc="lower right")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network. Operators are anonymised in the source; none identified here.", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "/charts/08_operator_concentration.png", dpi=150)
print("chart saved")
