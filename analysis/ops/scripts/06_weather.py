import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
EXT = f"{ROOT}/analysis/ops/data_ext"
mets = []
for a in APTS:
    m = pl.read_csv(f"{EXT}/metar_{a}.csv", null_values="M", infer_schema_length=0)
    m = m.select(pl.col("station").alias("APT"), pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M").dt.replace_time_zone("UTC").alias("mt"), pl.col("tmpc").cast(pl.Float64), pl.col("wxcodes").fill_null(""))
    mets.append(m)
met = pl.concat(mets).unique(["APT","mt"]).sort("APT","mt")
print("METAR rows:", met.height, met.group_by("APT").len().sort("APT"))
FROZEN = r"(SN|SG|PL|GS|FZRA|FZDZ|FZFG|IC)"
PRECIP = r"(RA|DZ|SH|TS|GR|UP)"
met = met.with_columns(
    pl.when(pl.col("wxcodes").str.contains(FROZEN)).then(pl.lit("1 frozen precip / freezing fog"))
    .when(pl.col("tmpc")<=3).then(pl.lit("2 cold <=3C, no frozen wx"))
    .when(pl.col("wxcodes").str.contains("TS")).then(pl.lit("4 thunderstorm >3C"))
    .when(pl.col("wxcodes").str.contains(PRECIP)).then(pl.lit("5 rain >3C"))
    .when(pl.col("tmpc")>3).then(pl.lit("3 dry >3C"))
    .otherwise(pl.lit("9 unknown")).alias("wx"))
df = base()
c = df.filter("clean").select("APT","PHASE_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","taxi_min","date_local").with_columns(
    pl.when(pl.col("PHASE_mvt")=="DEP").then(pl.col("BLOCK_TIME_UTC_mvt")).otherwise(pl.col("MVT_TIME_UTC_mvt")).alias("t")).sort("t")
j = c.join_asof(met.sort("mt"), left_on="t", right_on="mt", by="APT", strategy="backward", tolerance="90m")
print("rows without METAR within 90 min:", j.filter(pl.col("wx").is_null()).height, "of", j.height)
j = j.with_columns(pl.col("wx").fill_null("9 unknown"))
res = j.group_by("APT","PHASE_mvt","wx").agg(stats()).sort("PHASE_mvt","APT","wx")
print("=== taxi time by METAR class at off-block (DEP) / landing (ARR) hour ==="); print(res)
piv = res.filter(pl.col("wx").is_in(["1 frozen precip / freezing fog","3 dry >3C"])).with_columns(pl.col("wx").str.slice(0,1).alias("k")).pivot(on="k", index=["APT","PHASE_mvt"], values=["N","median","mean","p90"])
piv = piv.with_columns((pl.col("median_1")-pl.col("median_3")).round(2).alias("d_median"), (pl.col("mean_1")-pl.col("mean_3")).round(2).alias("d_mean"), (pl.col("p90_1")-pl.col("p90_3")).round(2).alias("d_p90"))
print("=== frozen-wx minus dry>3C (minutes) ==="); print(piv.sort("PHASE_mvt","d_median", descending=[True,True]))
pooled = j.filter(pl.col("wx").is_in(["1 frozen precip / freezing fog","2 cold <=3C, no frozen wx","3 dry >3C"])).group_by("PHASE_mvt","wx").agg(stats()).sort("PHASE_mvt","wx")
print("=== pooled 10 airports ==="); print(pooled)
# weather on top days
for a, d in [("LTFM","2025-02-23"),("LTFM","2025-02-19"),("EDDM","2025-12-24"),("LIRF","2025-07-13"),("LFPG","2025-01-17"),("EDDM","2025-01-15"),("LIRF","2025-07-28"),("EGLL","2025-09-01")]:
    dd = met.filter((pl.col("APT")==a) & (pl.col("mt").dt.convert_time_zone(TZ[a]).dt.date()==pl.lit(d).str.to_date()))
    codes = dd.filter(pl.col("wxcodes")!="")["wxcodes"].value_counts().sort("count", descending=True).head(6)
    print(a, d, "METARs:", dd.height, "min T:", dd["tmpc"].min(), "max T:", dd["tmpc"].max(), "wx:", dict(zip(codes["wxcodes"], codes["count"])))
# chart
p = piv.filter((pl.col("PHASE_mvt")=="DEP") & (pl.col("N_1")>=150)).sort("d_median")
print("chart: airports with >=150 departures in frozen-wx class:", p["APT"].to_list())
pa = p.select("APT").join(piv.filter(pl.col("PHASE_mvt")=="ARR"), on="APT", how="left")
fig, ax = plt.subplots(figsize=(9,5.5)); y=range(p.height)
ax.barh([i+0.2 for i in y], p["d_median"], height=.4, color="#2b6cb0", label="Taxi-out")
ax.barh([i-0.2 for i in y], pa["d_median"], height=.4, color="#9ac1e8", label="Taxi-in (control)")
for i,(v,n) in enumerate(zip(p["d_median"], p["N_1"])): ax.text(max(v,0)+0.3, i+0.2, f"+{v:.1f} min (N={n:,})" if v>=0 else f"{v:.1f} (N={n:,})", va="center", fontsize=7)
ax.axvline(0,color="k",lw=.6); ax.set_yticks(list(y)); ax.set_yticklabels(p["APT"])
ax.set_xlabel("median taxi time: frozen precipitation / freezing fog in METAR minus dry & >3 °C, minutes")
ax.set_title("Snow and freezing weather cost minutes on the way out, not on the way in\n2025, METAR at hour of off-block (DEP) or landing (ARR); N = movements in frozen-wx class", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Sources: PRC Data Challenge 2026 (EUROCONTROL); METAR via Iowa Environmental Mesonet. Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/05_frozen_weather_taxi.png", dpi=150)
