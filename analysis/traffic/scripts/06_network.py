"""Network: destinations, distance bands, countries, new/lost routes. Passenger scheduled departures."""
import polars as pl, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_all, HUBS, CITY, OUT, airports, haversine_km

PAX = ["Mainline", "Lowcost", "Regional"]
lf = load_all().filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MARKET_SEGMENT_flt").is_in(PAX))
d = lf.select("hub", pl.col("other_nm").alias("other"), "month", "src").collect()
ap = airports()
d = d.join(ap.select(pl.col("icao").alias("hub"), hlat="lat", hlon="lon", hcountry="iso_country"), on="hub", how="left") \
     .join(ap.select(pl.col("icao").alias("other"), "name", "municipality", "iso_country", "continent", "lat", "lon"), on="other", how="left")
print("pax departures (all periods):", d.height, " destination not found in OurAirports:", d.filter(pl.col("lat").is_null()).height)
print("unmatched dest codes (top):", d.filter(pl.col("lat").is_null()).group_by("other").len().sort("len", descending=True).head(10).to_dicts())
d = d.with_columns(km=pl.Series(haversine_km(d["hlat"].to_numpy(), d["hlon"].to_numpy(), d["lat"].to_numpy(), d["lon"].to_numpy()))).with_columns(pl.col("km").fill_nan(None))
d = d.with_columns(band=pl.when(pl.col("km").is_null()).then(pl.lit("unknown"))
                   .when(pl.col("km") < 1500).then(pl.lit("<1500 km"))
                   .when(pl.col("km") < 4000).then(pl.lit("1500-4000 km"))
                   .otherwise(pl.lit(">=4000 km")),
                   domestic=(pl.col("iso_country") == pl.col("hcountry")))
y25 = d.filter(pl.col("src") == "training")

print("\n=== 2025 full year, passenger departures per hub: distance bands, domestic share, unique destinations ===")
g = y25.group_by("hub").agg(
    deps=pl.len(),
    share_lt1500=(pl.col("band") == "<1500 km").mean() * 100,
    share_1500_4000=(pl.col("band") == "1500-4000 km").mean() * 100,
    share_longhaul_ge4000=(pl.col("band") == ">=4000 km").mean() * 100,
    domestic_pct=pl.col("domestic").mean() * 100,
    mean_km=pl.col("km").mean(),
    n_dest_any=pl.col("other").n_unique(),
)
wk = y25.group_by("hub", "other").len().filter(pl.col("len") >= 52).group_by("hub").agg(n_dest_ge52_deps=pl.len())
cty = y25.group_by("hub", "iso_country").len().filter(pl.col("len") >= 52).group_by("hub").agg(n_countries_ge52=pl.len())
g = g.join(wk, on="hub").join(cty, on="hub").sort("share_longhaul_ge4000", descending=True)
with pl.Config(tbl_cols=-1, float_precision=1, tbl_width_chars=250):
    print(g)
g.write_csv(OUT + "/scripts/06_network_hub.csv")

print("\n=== 2025 top-5 destinations per hub (passenger departures) ===")
top = y25.group_by("hub", "other", "municipality", "iso_country").len().sort("len", descending=True)
tot = y25.group_by("hub").len().rename({"len": "tot"})
for h in HUBS:
    t = top.filter(pl.col("hub") == h).head(5).join(tot, on="hub").with_columns(pct=pl.col("len") / pl.col("tot") * 100)
    print(h, "; ".join(f"{r['other']} {r['municipality']} ({r['iso_country']}) {r['len']} = {r['pct']:.1f}%" for r in t.iter_rows(named=True)))

print("\n=== 2025 top long-haul (>=4000 km) destinations per hub ===")
for h in HUBS:
    t = y25.filter((pl.col("hub") == h) & (pl.col("band") == ">=4000 km")).group_by("other", "municipality", "iso_country").len().sort("len", descending=True).head(5)
    print(h, "; ".join(f"{r['other']} {r['municipality']} ({r['iso_country']}) {r['len']}" for r in t.iter_rows(named=True)))

print("\n=== 2025 long-haul departures by destination continent, per hub (% of hub's long-haul) ===")
lh = y25.filter(pl.col("band") == ">=4000 km").group_by("hub", "continent").len()
lh = lh.with_columns(pct=pl.col("len") / pl.col("len").sum().over("hub") * 100)
with pl.Config(tbl_cols=-1, float_precision=1):
    print(lh.pivot(on="continent", index="hub", values="pct").fill_null(0).sort("hub"))

print("\n=== Destination countries 2025: top-5 foreign countries per hub (% of pax departures) ===")
cc = y25.filter(~pl.col("domestic")).group_by("hub", "iso_country").len().join(tot, on="hub").with_columns(pct=pl.col("len") / pl.col("tot") * 100).sort("len", descending=True)
for h in HUBS:
    t = cc.filter(pl.col("hub") == h).head(5)
    print(h, "; ".join(f"{r['iso_country']} {r['pct']:.1f}%" for r in t.iter_rows(named=True)))

print("\n=== Long-haul share YoY (Jan, Jul): % of pax departures >=4000 km; N = 31 vs 31 days ===")
yy = d.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).group_by("hub", "month").agg(
    deps=pl.len(), lh=(pl.col("band") == ">=4000 km").sum()).with_columns(pct=pl.col("lh") / pl.col("deps") * 100)
with pl.Config(tbl_cols=-1, float_precision=1, tbl_rows=50):
    pv = yy.pivot(on="month", index="hub", values=["lh", "pct"]).sort("hub"); print(pv.select(["hub"] + [f"{v}_{m}" for v in ["lh", "pct"] for m in ["2025-01", "2026-01", "2025-07", "2026-07"]]))

# New / dropped routes
print("\n=== Routes new or dropped: >=8 pax departures in month of one year, 0 in same month of other year ===")
rt = d.filter(pl.col("month").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).group_by("hub", "other", "month").len()
names = d.select("hub", "other", "municipality", "iso_country", "km").unique(subset=["hub", "other"])
res = []
for mon in ["01", "07"]:
    a = rt.filter(pl.col("month") == f"2025-{mon}").select("hub", "other", n25="len")
    b = rt.filter(pl.col("month") == f"2026-{mon}").select("hub", "other", n26="len")
    j = a.join(b, on=["hub", "other"], how="full", coalesce=True).with_columns(pl.col("n25").fill_null(0), pl.col("n26").fill_null(0)).join(names, on=["hub", "other"], how="left")
    new = j.filter((pl.col("n26") >= 8) & (pl.col("n25") == 0)).sort("hub", "n26", descending=[False, True])
    lost = j.filter((pl.col("n25") >= 8) & (pl.col("n26") == 0)).sort("hub", "n25", descending=[False, True])
    # pair new vs dropped at same hub whose destinations are <60 km apart: ICAO code change or airport relocation/swap
    ll = ap.select(pl.col("icao").alias("other"), "lat", "lon")
    nn = new.join(ll, on="other", how="left"); dd = lost.join(ll, on="other", how="left")
    pairs = nn.join(dd, on="hub", suffix="_old").filter(pl.col("lat").is_not_null() & pl.col("lat_old").is_not_null())
    pairs = pairs.with_columns(sep_km=pl.Series(haversine_km(pairs["lat"].to_numpy(), pairs["lon"].to_numpy(), pairs["lat_old"].to_numpy(), pairs["lon_old"].to_numpy()))).filter(pl.col("sep_km") < 60)
    print(f"\n-- month {mon}: new/dropped pairs at same hub <60 km apart (code change or airport swap):")
    with pl.Config(tbl_rows=50, float_precision=0):
        print(pairs.select("hub", "other", "other_old", "municipality", "n25_old", "n26", "sep_km"))
    new = new.join(pairs.select("hub", "other"), on=["hub", "other"], how="anti")
    lost = lost.join(pairs.select("hub", pl.col("other_old").alias("other")), on=["hub", "other"], how="anti")
    print(f"\n-- month {mon} (after removing code-change pairs): NEW routes total {new.height} (long-haul >=4000 km: {new.filter(pl.col('km')>=4000).height}), DROPPED routes total {lost.height} (long-haul: {lost.filter(pl.col('km')>=4000).height})")
    summ = new.group_by("hub").agg(new=pl.len()).join(lost.group_by("hub").agg(dropped=pl.len()), on="hub", how="full", coalesce=True).fill_null(0).sort("hub")
    print(summ)
    with pl.Config(tbl_rows=200, tbl_width_chars=200, float_precision=0):
        print("NEW:"); print(new)
        print("DROPPED:"); print(lost)
    new.with_columns(mon=pl.lit(mon), kind=pl.lit("new")).vstack(lost.with_columns(mon=pl.lit(mon), kind=pl.lit("dropped"))).write_csv(OUT + f"/scripts/06_routes_newdropped_{mon}.csv")

# chart: distance band shares per hub 2025
g2 = g.sort("share_longhaul_ge4000")
fig, ax = plt.subplots(figsize=(9, 5))
left = np.zeros(g2.height)
for col, c, lab in [("share_lt1500", "#9fb8cf", "< 1,500 km"), ("share_1500_4000", "#5b85aa", "1,500–4,000 km"), ("share_longhaul_ge4000", "#d9822b", "≥ 4,000 km")]:
    v = g2[col].to_numpy(); ax.barh(range(g2.height), v, left=left, color=c, label=lab)
    for i, (l, vv) in enumerate(zip(left, v)):
        if vv > 6: ax.text(l + vv / 2, i, f"{vv:.0f}%", ha="center", va="center", fontsize=8, color="white" if col != "share_lt1500" else "#222")
    left += v
ax.set_yticks(range(g2.height), [f"{CITY[h]} ({h})" for h in g2["hub"]])
ax.set_xlim(0, 100); ax.set_xlabel("Share of passenger departures by great-circle distance, 2025, %")
ax.set_title("How long-haul is each hub? Passenger departures 2025")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.text(0.01, 0.01, "Data: EUROCONTROL PRC Data Challenge 2026 / OpenSky Network; airport coordinates: OurAirports (public domain).", fontsize=7, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "/charts/02_distance_bands.png", dpi=150)
print("chart saved")
