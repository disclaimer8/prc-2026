import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
df = base(extra=["EOBT_1_flt","AOBT_3_flt","IOBT_flt","LOBT_flt","ARVT_3_flt","FLIGHT_TYPE_flt"])
df = df.with_columns(((pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()/60).alias("delay_min"))
print("delay = BLOCK_TIME - SCHED_TIME (min). DEP: off-block vs scheduled departure; ARR: in-block vs scheduled arrival.")
for ph in ["DEP","ARR"]:
    x = df.filter(pl.col("PHASE_mvt")==ph)
    print(ph, "N", x.height, "nulls", x["delay_min"].null_count(), "|delay|>24h:", (x["delay_min"].abs()>1440).sum(), " <-6h:", (x["delay_min"]<-360).sum(), " >12h:", (x["delay_min"]>720).sum())
# keep plausible window
P = df.filter(pl.col("delay_min").is_between(-360, 1440))
print("window for punctuality stats: -6 h <= delay <= +24 h; kept", P.height, "of", df.height)
def pstats():
    return [pl.len().alias("N"), pl.col("delay_min").mean().round(2).alias("mean"), pl.col("delay_min").median().round(2).alias("median"), pl.col("delay_min").quantile(0.9).round(1).alias("p90"),
            ((pl.col("delay_min")<=15).mean()*100).round(1).alias("pct_le15"), ((pl.col("delay_min")<=5).mean()*100).round(1).alias("pct_le5"), ((pl.col("delay_min")<-5).mean()*100).round(1).alias("pct_early_gt5"), ((pl.col("delay_min")>60).mean()*100).round(2).alias("pct_gt60")]
a = P.group_by("APT","PHASE_mvt").agg(pstats()).sort("PHASE_mvt","pct_le15", descending=[True,True])
print("=== punctuality by airport (pct_le15 = on-time within 15 min, incl. early) ==="); print(a)
print(P.group_by("PHASE_mvt").agg(pstats()))
seg = P.filter(pl.col("PHASE_mvt")=="DEP").group_by("MARKET_SEGMENT_flt").agg(pstats()).sort("N", descending=True)
print("=== DEP punctuality by market segment ==="); print(seg)
segapt = P.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("MARKET_SEGMENT_flt").is_in(["Mainline","Lowcost","Regional","Cargo","Business"])).group_by("APT","MARKET_SEGMENT_flt").agg(pl.len().alias("N"), ((pl.col("delay_min")<=15).mean()*100).round(1).alias("otp15"))
print(segapt.pivot(on="MARKET_SEGMENT_flt", index="APT", values="otp15").sort("APT"))
print(segapt.pivot(on="MARKET_SEGMENT_flt", index="APT", values="N").sort("APT"))
# sched time granularity
print("SCHED seconds!=0 share:", (df["SCHED_TIME_UTC_mvt"].dt.second()!=0).mean(), " SCHED minute%5!=0 share:", (df["SCHED_TIME_UTC_mvt"].dt.minute()%5!=0).mean())
# delay by month & hour (DEP)
dm = P.filter(pl.col("PHASE_mvt")=="DEP").group_by("month").agg(pstats()).sort("month"); print("=== DEP by month ==="); print(dm)
dh = P.filter(pl.col("PHASE_mvt")=="DEP").group_by("hour_local").agg(pstats()).sort("hour_local"); print("=== DEP by local hour (of actual off-block) ==="); print(dh)
# does off-block delay relate to taxi-out? (clean taxi)
q = P.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("clean")).with_columns(pl.col("delay_min").cut([-5,0,15,30,60,120], labels=["<-5","-5..0","0..15","15..30","30..60","60..120",">120"]).alias("dbin"))
print("=== DEP taxi-out by off-block delay bin (pooled) ==="); print(q.group_by("dbin").agg(stats()).sort("dbin"))
# chart: distribution
fig, ax = plt.subplots(figsize=(9,5))
d = P.filter(pl.col("PHASE_mvt")=="DEP")["delay_min"].clip(-30,120).to_numpy()
import numpy as np
ax.hist(d, bins=np.arange(-30,121,1), color="#2b6cb0")
ax.axvline(15, color="#c53030", ls="--", lw=1); ax.text(16, ax.get_ylim()[1]*.9, "15 min", color="#c53030", fontsize=8)
ax.set_xlabel("actual off-block minus scheduled departure, minutes (clipped at −30 / +120)"); ax.set_ylabel("departures per 1-min bin")
ax.set_title(f"Departure off-block delay, 10 airports, 2025 (N = {len(d):,})"); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/06_offblock_delay_hist.png", dpi=150)
fig, ax = plt.subplots(figsize=(9,5))
dd = a.filter(pl.col("PHASE_mvt")=="DEP").sort("pct_le15")
ax.barh(dd["APT"], dd["pct_le15"], color="#2b6cb0")
for i,(v,m) in enumerate(zip(dd["pct_le15"], dd["median"])): ax.text(v+0.5, i, f"{v:.1f}%  (median {m:+.0f} min)", va="center", fontsize=8)
ax.set_xlim(0,100); ax.set_xlabel("% departures off-block within 15 min of schedule (early included)")
ax.set_title("Departure punctuality at the stand, 2025"); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Window −6 h..+24 h. Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/07_dep_otp15_by_airport.png", dpi=150)
