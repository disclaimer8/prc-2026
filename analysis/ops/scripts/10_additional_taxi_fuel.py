"""Additional taxi-out time (EUROCONTROL ATXOT-style) and an explicit fuel/CO2 translation.
Methodology references (files in ../data_ext):
 - EUROCONTROL, Additional taxi-out time performance indicator document, Ed. 01.00 (Mar 2023), ansperformance.eu:
   reference = P10 of taxi-out per APT x STAND x DEP RUNWAY over 12 months; min 10 flights <= P10; exclude taxi > 120 min, helicopters,
   de-icing after off-block (NOT available here -> not excluded).
 - ICAO Aircraft Engine Emissions Databank, issue 03/2026 (EASA-hosted): 'Fuel Flow Idle (kg/sec)' per engine, non-superseded rows.
 - Regulation (EU) 2018/2066 (consolidated 2024-01-01), Annex III: Jet kerosene 3.16 t CO2 / t fuel.
Simplifications (all explicit): all engines running during taxi (upper bound, no single-engine taxi); idle fuel flow of the ICAO LTO idle
mode (sea-level static, no bleed) used as taxi fuel flow; one engine family per aircraft type, fuel flow = median over matching EEDB rows,
range = min..max over those rows.
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd, re
from common import *
EXT = f"{ROOT}/analysis/ops/data_ext"
CO2_PER_KG_FUEL = 3.16
ENG = {  # type: (engines, [EEDB 'Engine Identification' substrings])
 "A318":(2,["CFM56-5B","PW6122"]), "A319":(2,["CFM56-5B","V2522","V2524","V2527M"]), "A320":(2,["CFM56-5B","V2527","V2500-A1"]), "A321":(2,["CFM56-5B","V2530","V2533"]),
 "A20N":(2,["LEAP-1A","PW112"]), "A21N":(2,["LEAP-1A","PW113"]), "B737":(2,["CFM56-7B"]), "B738":(2,["CFM56-7B"]), "B739":(2,["CFM56-7B"]),
 "B38M":(2,["LEAP-1B"]), "B39M":(2,["LEAP-1B"]), "BCS1":(2,["PW15"]), "BCS3":(2,["PW15"]), "E170":(2,["CF34-8E"]), "E75L":(2,["CF34-8E"]),
 "E190":(2,["CF34-10E"]), "E195":(2,["CF34-10E"]), "E290":(2,["PW19"]), "E295":(2,["PW19"]), "CRJ9":(2,["CF34-8C"]), "CRJX":(2,["CF34-8C"]),
 "A332":(2,["Trent 7","CF6-80E","PW4168"]), "A333":(2,["Trent 7","CF6-80E","PW4168"]), "A339":(2,["Trent7000"]), "A359":(2,["Trent XWB-84"]), "A35K":(2,["Trent XWB-97"]),
 "A343":(4,["CFM56-5C"]), "A388":(4,["Trent 9","GP72"]), "B744":(4,["CF6-80C2","PW4056","RB211-524G","RB211-524H"]), "B748":(4,["GEnx-2B"]),
 "B763":(2,["CF6-80C2","PW4060","RB211-524H"]), "B764":(2,["CF6-80C2"]), "B772":(2,["GE90-9","Trent 8","PW4090","PW4077"]), "B77L":(2,["GE90-11"]), "B77W":(2,["GE90-115"]),
 "B788":(2,["GEnx-1B","Trent 10"]), "B789":(2,["GEnx-1B","Trent 10"]), "B78X":(2,["GEnx-1B","Trent 10"]), "A306":(2,["CF6-80C2","PW4158"]), "B752":(2,["RB211-535","PW2037","PW2040"]),
}
e = pd.read_excel(f"{EXT}/icao_eedb_2026-03.xlsx", "Gaseous Emissions and Smoke")
e = e[e["Data Superseded"].astype(str).str.strip().str.lower()!="yes"]
rows=[]
for t,(n,fams) in ENG.items():
    m = e[e["Engine Identification"].astype(str).apply(lambda s: any(f in s for f in fams))]
    ff = m["Fuel Flow Idle (kg/sec)"].dropna()
    rows.append(dict(type=t, n_eng=n, eedb_rows=len(ff), ff_med=float(ff.median()) if len(ff) else None, ff_min=float(ff.min()) if len(ff) else None, ff_max=float(ff.max()) if len(ff) else None, families=";".join(fams)))
ff = pl.DataFrame(rows)
print("=== idle fuel flow per engine (kg/s) from ICAO EEDB 03/2026, non-superseded ==="); print(ff)
ff = ff.filter(pl.col("eedb_rows")>0)

df = base()
dep = df.filter((pl.col("PHASE_mvt")=="DEP") & pl.col("TAXITIME_SEC_mvt").is_between(60, 7200) & ~pl.col("RUNWAY_mvt").is_in(["HEL","H","NA"]) & pl.col("STAND_mvt").is_not_null())
print("DEP after EUROCONTROL-like filters (1 min..120 min, no helicopter pseudo-runways, stand+runway present):", dep.height)
ref = dep.group_by("APT","STAND_mvt","RUNWAY_mvt").agg(pl.len().alias("n_combo"), pl.col("taxi_min").quantile(0.10, "lower").alias("ref_p10"))
dep = dep.join(ref, on=["APT","STAND_mvt","RUNWAY_mvt"])
dep = dep.with_columns((pl.col("taxi_min") <= pl.col("ref_p10")).sum().over(["APT","STAND_mvt","RUNWAY_mvt"]).alias("n_le_p10"))
valid = dep.filter(pl.col("n_le_p10")>=10).with_columns((pl.col("taxi_min")-pl.col("ref_p10")).alias("add_min"))
print("DEP with a valid reference (>=10 flights at or below combo P10):", valid.height, f"({valid.height/dep.height*100:.1f}%)")
A = valid.group_by("APT").agg(pl.len().alias("N"), pl.col("taxi_min").mean().round(2).alias("taxi_mean"), pl.col("ref_p10").mean().round(2).alias("ref_mean"),
    pl.col("add_min").mean().round(2).alias("add_mean"), pl.col("add_min").median().round(2).alias("add_median"), pl.col("add_min").quantile(0.9).round(2).alias("add_p90"),
    (pl.col("add_min").sum()/60).round(0).alias("add_hours_total")).sort("add_mean", descending=True)
print("=== additional taxi-out time per departure, 2025 (single annual reference, not rolling) ==="); print(A)
print("ALL:", valid.select(pl.len().alias("N"), pl.col("add_min").mean().round(2).alias("add_mean"), pl.col("add_min").median().round(2).alias("add_median"), (pl.col("add_min").sum()/60).round(0).alias("hours"), (pl.col("add_min")<0).mean().round(3).alias("share_negative")))
# by month (winter check) and hour
print(valid.group_by("month").agg(pl.col("add_min").mean().round(2).alias("add_mean"), pl.col("add_min").median().round(2).alias("add_median"), pl.len()).sort("month"))

# fuel
v = valid.join(ff.select(pl.col("type").alias("AIRCRAFT_TYPE_mvt"),"n_eng","ff_med","ff_min","ff_max"), on="AIRCRAFT_TYPE_mvt", how="left")
cov = v.filter(pl.col("ff_med").is_not_null())
print("fuel coverage: departures with mapped type:", cov.height, "of", v.height, f"({cov.height/v.height*100:.1f}%)")
F = cov.with_columns([(pl.col("add_min")*60*pl.col("n_eng")*pl.col(c)).alias("addfuel_"+c[3:]) for c in ["ff_med","ff_min","ff_max"]] +
                     [(pl.col("taxi_min")*60*pl.col("n_eng")*pl.col("ff_med")).alias("taxifuel_med")])
agg = F.group_by("APT").agg(pl.len().alias("N_covered"), (pl.col("taxifuel_med").sum()/1000).round(0).alias("taxi_out_fuel_t"),
      (pl.col("addfuel_med").sum()/1000).round(0).alias("add_fuel_t"), (pl.col("addfuel_min").sum()/1000).round(0).alias("add_fuel_t_low"), (pl.col("addfuel_max").sum()/1000).round(0).alias("add_fuel_t_high"),
      (pl.col("addfuel_med").mean()).round(1).alias("add_fuel_kg_per_dep")).with_columns((pl.col("add_fuel_t")*CO2_PER_KG_FUEL).round(0).alias("add_CO2_t"), (pl.col("taxi_out_fuel_t")*CO2_PER_KG_FUEL).round(0).alias("taxi_out_CO2_t"), (pl.col("add_fuel_kg_per_dep")*CO2_PER_KG_FUEL).round(1).alias("add_CO2_kg_per_dep")).sort("add_fuel_t", descending=True)
print("=== fuel (t) and CO2 (t): all-engine idle, ICAO EEDB idle fuel flow; covered types only ==="); print(agg)
tot = F.select(pl.len().alias("N"), (pl.col("taxifuel_med").sum()/1000).round(0).alias("taxi_out_fuel_t"), (pl.col("addfuel_med").sum()/1000).round(0).alias("add_fuel_t"), (pl.col("addfuel_min").sum()/1000).round(0).alias("low"), (pl.col("addfuel_max").sum()/1000).round(0).alias("high"))
print("TOTAL:", tot, " CO2 add (t):", round(tot["add_fuel_t"][0]*CO2_PER_KG_FUEL), " range:", round(tot["low"][0]*CO2_PER_KG_FUEL), "-", round(tot["high"][0]*CO2_PER_KG_FUEL))
print("sanity: mean all-engine idle fuel flow per covered departure, kg/min:", F.select((pl.col("n_eng")*pl.col("ff_med")*60).mean().round(1)))

fig, ax = plt.subplots(figsize=(9,5.5))
a = A.sort("add_mean"); g = agg.select("APT","add_CO2_t")
a = a.join(g, on="APT", how="left")
ax.barh(a["APT"], a["add_mean"], color="#2b6cb0")
for i,(v_,n,c) in enumerate(zip(a["add_mean"], a["N"], a["add_CO2_t"])): ax.text(v_+0.05, i, f"{v_:.1f} min/dep · ≈{c:,.0f} t CO₂", va="center", fontsize=8)
ax.set_xlabel("mean additional taxi-out time per departure, minutes (actual − P10 reference of stand×runway)")
ax.set_title("Time spent queuing on the way to the runway, 2025\nEUROCONTROL-style reference; CO₂ = all-engine ICAO idle fuel flow × 3.16 (upper-bound style)", fontsize=10)
ax.set_xlim(0, a["add_mean"].max()*1.6); ax.spines[["top","right"]].set_visible(False)
fig.text(0.01,0.01,"Sources: PRC Data Challenge 2026 (EUROCONTROL); ICAO Engine Emissions Databank 03/2026; Reg. (EU) 2018/2066. Aggregates only.", fontsize=6.5, color="gray")
fig.tight_layout(); fig.savefig(f"{CHARTS}/10_additional_taxi_co2.png", dpi=150)
