import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
df = base()
c = df.filter("clean")
MINN = 2000
for ph in ["DEP","ARR"]:
    r = c.filter(pl.col("PHASE_mvt")==ph).group_by("APT","RUNWAY_mvt").agg(stats()).with_columns((pl.col("N")/pl.col("N").sum().over("APT")*100).round(1).alias("share_pct")).sort("APT","median")
    print(f"=== {ph} by runway (all runways) ==="); print(r)
    rr = r.filter(pl.col("N")>=MINN)
    spread = rr.group_by("APT").agg(pl.len().alias("rwys_N>=2000"), pl.col("median").min().alias("min_med"), pl.col("median").max().alias("max_med"),
        pl.col("RUNWAY_mvt").sort_by("median").first().alias("fastest"), pl.col("RUNWAY_mvt").sort_by("median").last().alias("slowest")).with_columns((pl.col("max_med")-pl.col("min_med")).round(2).alias("spread_min"), (pl.col("max_med")/pl.col("min_med")).round(2).alias("ratio")).sort("spread_min", descending=True)
    print(f"=== {ph} runway spread (runways with N>={MINN}) ==="); print(spread)
    if ph=="DEP": dep_rwy = rr
# airline-only (Mainline/Lowcost/Regional) runway medians: removes business-aviation/GA runways such as EHAM 22/04
AL = c.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("MARKET_SEGMENT_flt").is_in(["Mainline","Lowcost","Regional"]))
dep_rwy = AL.group_by("APT","RUNWAY_mvt").agg(stats()).filter(pl.col("N")>=MINN).sort("APT","median")
print("=== DEP by runway, airline segments only (Mainline/Lowcost/Regional), N>=2000 ==="); print(dep_rwy)
print(dep_rwy.group_by("APT").agg(pl.len().alias("rwys"), pl.col("median").min().alias("min_med"), pl.col("median").max().alias("max_med"), pl.col("RUNWAY_mvt").sort_by("median").first().alias("fastest"), pl.col("RUNWAY_mvt").sort_by("median").last().alias("slowest")).with_columns((pl.col("max_med")-pl.col("min_med")).round(2).alias("spread_min")).sort("spread_min", descending=True))
# chart: DEP runway medians for airports (dot plot)
fig, ax = plt.subplots(figsize=(10,6))
order = dep_rwy.group_by("APT").agg(pl.col("median").median()).sort("median")["APT"].to_list()
for i,a in enumerate(order):
    s = dep_rwy.filter(pl.col("APT")==a)
    ax.plot([s["median"].min(), s["median"].max()],[i,i], color="#cbd5e0", lw=6, solid_capstyle="round", zorder=1)
    ax.scatter(s["median"], [i]*len(s), s=(s["N"]/800).clip(15,300), color="#2b6cb0", zorder=2, alpha=.85)
    for k,(m, rw) in enumerate(zip(s["median"], s["RUNWAY_mvt"])):
        ax.annotate(rw, (m, i), textcoords="offset points", xytext=(0, 9 if k%2==0 else -14), ha="center", fontsize=7)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlabel("median taxi-out, minutes")
ax.set_title(f"Same airport, different runway: median taxi-out by departure runway, 2025\n(Mainline/Low-cost/Regional only; runways with ≥{MINN} such departures; dot size ∝ departures; 1 min ≤ taxi ≤ 4 h)")
ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/02_taxiout_by_runway.png", dpi=150)

# stands: DEP
s = c.filter(pl.col("PHASE_mvt")=="DEP").group_by("APT","STAND_mvt").agg(stats()).filter(pl.col("N")>=500)
print("=== DEP stand-level medians (stands with N>=500): distribution of stand medians per airport ===")
print(s.group_by("APT").agg(pl.len().alias("stands"), pl.col("N").sum().alias("deps_covered"), pl.col("median").quantile(0.1).alias("p10_of_stand_medians"), pl.col("median").median().alias("median_of_stand_medians"), pl.col("median").quantile(0.9).alias("p90_of_stand_medians"), pl.col("median").min().alias("min"), pl.col("median").max().alias("max")).sort("APT"))
print("=== top 15 slowest stands (DEP, N>=500) ==="); print(s.sort("median", descending=True).head(15))
# stand x runway interaction share of variance (simple): within-airport variance explained by stand, runway, stand+runway group medians
d = c.filter(pl.col("PHASE_mvt")=="DEP").select("APT","STAND_mvt","RUNWAY_mvt","taxi_min")
def r2(keys):
    g = d.with_columns(pl.col("taxi_min").mean().over(keys).alias("m"), pl.col("taxi_min").mean().over("APT").alias("ma"))
    out = g.group_by("APT").agg((1 - ((pl.col("taxi_min")-pl.col("m"))**2).sum()/((pl.col("taxi_min")-pl.col("ma"))**2).sum()).round(3).alias("R2_"+"_".join(k.split("_")[0] for k in keys if k!="APT")))
    return out
res = r2(["APT","RUNWAY_mvt"]).join(r2(["APT","STAND_mvt"]), on="APT").join(r2(["APT","STAND_mvt","RUNWAY_mvt"]), on="APT").sort("APT")
print("=== in-sample share of within-airport taxi-out variance explained by group means (DEP, clean; R2, optimistic for small groups) ==="); print(res)
