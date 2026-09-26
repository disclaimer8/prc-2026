import os, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
LB=236.3209
cols=["MVT_ID_mvt","split","ADEP_mvt","FLIGHT_ID_mvt","d_mvt_sched","d_mvt_aobt","d_mvt_lobt","d_mvt_iobt","d_mvt_eobt","TAXITIME_SEC_mvt"]
f=pl.read_parquet("data/features_e106.parquet",columns=cols).filter(pl.col("split")=="train")
ids=[];pr=[];ys=[]
for p in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz")):
    d=np.load(p); ids.append(d["mvt_id"]); pr.append(d["pred"]); ys.append(d["y"])
o=pl.DataFrame({"MVT_ID_mvt":np.concatenate(ids).astype(np.int64),"pred":np.concatenate(pr),"y":np.concatenate(ys).astype(float)}).with_columns(pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"]))
j=o.join(f,on="MVT_ID_mvt")
def lbeq(dsse): return np.sqrt(max(LB**2+dsse/6/344841,0))-LB
def report(name, m, alt):
    x=j.filter(m)
    if x.height==0: print(f"{name}: 0 rows"); return
    a=x.select(alt).to_series().to_numpy().astype(float); p=x["pred"].to_numpy(); y=x["y"].to_numpy()
    s0=((p-y)**2).sum(); s1=((a-y)**2).sum()
    print(f"{name}: rows {x.height}, OOF SSE {s0:.3e}, SSE if alt {s1:.3e}, oracle LB-eq {lbeq(s1-s0):+.3f} s")
y=pl.col("y")
print("== A. planned regimes disabled at EDDF/EHAM/LSZH (REGIME_EXCLUDE)")
for c,name in (("d_mvt_sched","sched"),("d_mvt_iobt","iobt"),("d_mvt_lobt","lobt"),("d_mvt_eobt","eobt")):
    for lo in (0,1800,3600):
        report(f"  {name} copy |y-{name}|<=6, {name}>{lo}, excluded airports", pl.col("ADEP_mvt").is_in(["EDDF","EHAM","LSZH"]) & ((y-pl.col(c)).abs()<=6) & (pl.col(c)>lo), c)
print("== B. dayshift window [86400, 93600]")
report("  y in (93600, 108000], near D+86400? (alt = y itself oracle)", (y>93600)&(y<=108000), "y")
report("  y in [79200, 86400)", (y>=79200)&(y<86400), "y")
print("== C. proxies used only in [0, 7200]")
report("  MVT-AOBT_3 > 7200 and |y - (MVT-AOBT_3)| <= 300 (alt = MVT-AOBT_3)", (pl.col("d_mvt_aobt")>7200)&((y-pl.col("d_mvt_aobt")).abs()<=300), "d_mvt_aobt")
report("  MVT-AOBT_3 > 7200, all such rows (alt = MVT-AOBT_3)", pl.col("d_mvt_aobt")>7200, "d_mvt_aobt")
print("== D. schedule copies at non-LIRF airports with D > 3600, not excluded")
for ap in ("EDDM","EGLL","LEBL","LEMD","LFPG","LTFM"):
    report(f"  {ap} |y-D|<=65 & D>3600", (pl.col("ADEP_mvt")==ap)&((y-pl.col("d_mvt_sched")).abs()<=65)&(pl.col("d_mvt_sched")>3600), "d_mvt_sched")
print("== E. LIRF no-NM, E14 band 0<D<4h (history<3 not served by E14): copies")
report("  LIRF no-NM 0<D<14400 & |y-D|<=65", (pl.col("ADEP_mvt")=="LIRF")&pl.col("FLIGHT_ID_mvt").is_null()&(pl.col("d_mvt_sched")>0)&(pl.col("d_mvt_sched")<14400)&((y-pl.col("d_mvt_sched")).abs()<=65), "d_mvt_sched")
print("== F. negative labels (prediction floored at 0)")
report("  y < 0", y<0, "y")
