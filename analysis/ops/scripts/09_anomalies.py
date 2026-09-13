import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
df = base(extra=["FLIGHT_ID_mvt","EOBT_1_flt","AOBT_3_flt","FLIGHT_TYPE_flt","FLIGHT_RULE_mvt","CALLSIGN_flt"])
N = df.height
t = pl.col("TAXITIME_SEC_mvt")
df = df.with_columns(pl.when(t.is_null()).then(pl.lit("null"))
    .when(t<0).then(pl.lit("a <0"))
    .when(t<60).then(pl.lit("b 0..59s"))
    .when(t<=3600).then(pl.lit("c 1min..1h"))
    .when(t<=14400).then(pl.lit("d 1h..4h"))
    .when(t<=86400).then(pl.lit("e 4h..24h"))
    .otherwise(pl.lit("f >24h")).alias("tbin"))
b = df.group_by("PHASE_mvt","tbin").agg(pl.len().alias("N")).with_columns((pl.col("N")/pl.col("N").sum().over("PHASE_mvt")*100).round(4).alias("pct")).sort("PHASE_mvt","tbin")
print("=== taxi-time bins ==="); print(b)
print("=== anomalies by airport (count) ===")
print(df.filter(~pl.col("tbin").is_in(["c 1min..1h","d 1h..4h"])).group_by("APT","PHASE_mvt","tbin").len().pivot(on="tbin", index=["APT","PHASE_mvt"], values="len", sort_columns=True).sort("PHASE_mvt","APT"))
df = df.with_columns(((pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()/60).alias("delay_min"), ((pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()/60).alias("mvt_vs_sched_min"),
    ((pl.col("AOBT_3_flt")-pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds()/60).alias("aobt3_minus_block"))
for lab in ["a <0","e 4h..24h","f >24h"]:
    x = df.filter(pl.col("tbin")==lab)
    print(f"--- {lab}: N={x.height}")
    print(x.select(pl.col("taxi_min").median().alias("med_taxi_min"), pl.col("delay_min").median().alias("med_block_minus_sched"), pl.col("mvt_vs_sched_min").median().alias("med_mvt_minus_sched"),
                   pl.col("aobt3_minus_block").median().alias("med_AOBT3_minus_block"), pl.col("FLIGHT_ID_mvt").is_null().mean().alias("share_no_NM_match"),
                   (pl.col("aobt3_minus_block").abs()>60).mean().alias("share_AOBT3_differs_gt60min")))
    print(x.group_by("PHASE_mvt","MARKET_SEGMENT_flt").len().sort("len", descending=True).head(8))
# >24h: is it an exact-day offset?
x = df.filter(pl.col("tbin")=="f >24h").with_columns((pl.col("TAXITIME_SEC_mvt")/86400).round(3).alias("days"))
print(">24h taxi values in days:", x.sort("days")["days"].to_list())
# DEP >4h: does AOBT_3 agree better with MVT time (i.e. block time wrong)?
x = df.filter((pl.col("PHASE_mvt")=="DEP") & (t>14400) & pl.col("AOBT_3_flt").is_not_null()).with_columns(((pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds()/60).alias("taxi_from_aobt3"))
print("DEP >4h with NM match:", x.height, " taxi computed from AOBT_3 instead: median", x["taxi_from_aobt3"].median(), "min; share within 1..60 min:", round(x["taxi_from_aobt3"].is_between(1,60).mean()*100,1), "%")
x = df.filter((pl.col("PHASE_mvt")=="DEP") & (t<0) & pl.col("AOBT_3_flt").is_not_null()).with_columns(((pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds()/60).alias("taxi_from_aobt3"))
print("DEP <0 with NM match:", x.height, " taxi from AOBT_3: median", x["taxi_from_aobt3"].median(), "min; share within 1..60 min:", round(x["taxi_from_aobt3"].is_between(1,60).mean()*100,1), "%")
# resolution of timestamps per airport
print("=== share of timestamps with seconds != 0, by airport ===")
print(df.group_by("APT","PHASE_mvt").agg((pl.col("BLOCK_TIME_UTC_mvt").dt.second()!=0).mean().round(3).alias("block_sec"), (pl.col("MVT_TIME_UTC_mvt").dt.second()!=0).mean().round(3).alias("mvt_sec"), (t%60==0).mean().round(3).alias("taxi_multiple_of_60s")).sort("PHASE_mvt","APT"))
# AOBT3-block offset by airport (definition differences)
print("=== median AOBT_3 - airport BLOCK (DEP, minutes) by airport and month (definition drift check) ===")
print(df.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("aobt3_minus_block").abs().le(60)).group_by("APT","month").agg(pl.col("aobt3_minus_block").median().round(2)).pivot(on="month", index="APT", values="aobt3_minus_block", sort_columns=True).sort("APT"))
# EHAM runway 22/04 departures
e = df.filter((pl.col("APT")=="EHAM") & (pl.col("PHASE_mvt")=="DEP") & pl.col("RUNWAY_mvt").is_in(["22","04","HEL"]))
print("=== EHAM DEP on 22/04/HEL: segment & wake mix ==="); print(e.group_by("RUNWAY_mvt","MARKET_SEGMENT_flt").len().sort("RUNWAY_mvt","len", descending=[False,True]))
print(e.group_by("RUNWAY_mvt","WK_TBL_CAT_flt").len().sort("RUNWAY_mvt","len", descending=[False,True]))
print(e.group_by("RUNWAY_mvt").agg(pl.col("FLIGHT_RULE_mvt").value_counts()).explode("FLIGHT_RULE_mvt"))
# unmatched movements
u = df.filter(pl.col("FLIGHT_ID_mvt").is_null())
print("=== rows without NM FLIGHT_ID:", u.height, "of", N, "by airport/phase ==="); print(u.group_by("APT","PHASE_mvt").len().sort("len", descending=True).head(12))
seg_null = df.filter(pl.col("MARKET_SEGMENT_flt").is_null())
print("rows with FLIGHT_ID but null segment:", seg_null.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).height, "; null segment total:", seg_null.height)
print("null-segment rows: median |BLOCK-SCHED| min:", seg_null["delay_min"].abs().median(), "vs all:", df["delay_min"].abs().median())
print("null-segment: FLIGHT_RULE_mvt:", seg_null["FLIGHT_RULE_mvt"].value_counts().to_dicts())
# movements between two airports in the set (same flight appears twice)
both = df.filter(pl.col("ADEP_mvt").is_in(APTS) & pl.col("ADES_mvt").is_in(APTS))
print("movements with both ADEP and ADES in the 10-airport set:", both.height, f"({both.height/N*100:.1f}%)")
fid = df.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).group_by("FLIGHT_ID_mvt").agg(pl.len().alias("n"), pl.col("PHASE_mvt").n_unique().alias("ph"))
print("FLIGHT_ID appearing twice (DEP at one airport + ARR at another):", fid.filter((pl.col("n")==2)&(pl.col("ph")==2)).height, "; FLIGHT_ID with >2 rows:", fid.filter(pl.col("n")>2).height, "; same phase duplicates:", fid.filter((pl.col("n")>=2)&(pl.col("ph")==1)).height)
# MVT before SCHED by a lot? DST sanity: median delay by month per airport near DST switch weeks
# chart: anomaly shares
fig, ax = plt.subplots(figsize=(9,4.5))
lab = ["a <0","b 0..59s","d 1h..4h","e 4h..24h","f >24h"]
dd = b.filter(pl.col("tbin").is_in(lab))
import numpy as np
w=0.38; xs=np.arange(len(lab))
for k,(ph,col) in enumerate([("DEP","#2b6cb0"),("ARR","#9ac1e8")]):
    v=[dd.filter((pl.col("PHASE_mvt")==ph)&(pl.col("tbin")==l))["N"].sum() for l in lab]
    ax.bar(xs+(k-0.5)*w, v, w, color=col, label=ph)
    for xi,vi in zip(xs,v): ax.text(xi+(k-0.5)*w, vi, f"{vi:,}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(["negative","0–59 s","1–4 h","4–24 h","> 24 h"]); ax.set_ylabel("movements"); ax.set_yscale("log")
ax.set_title(f"Implausible taxi times are rare: out-of-range values in {N:,} movements, 2025"); ax.legend(); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Source: PRC Data Challenge 2026 (EUROCONTROL). Aggregates only.", fontsize=7, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/09_anomalies.png", dpi=150)
