"""Are YoY segment changes real or relabelling? Uses anonymised operator hash only (no names printed)."""
import polars as pl
from common import load_all
import glob
from common import DATA
tr = pl.read_parquet(sorted(glob.glob(f"{DATA}/training_2025-0[17]-*.parquet")), columns=["PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt"]).lazy()
rk = pl.read_parquet(f"{DATA}/ranking.parquet", columns=["PHASE_mvt","ADEP_mvt","ADES_mvt","MVT_TIME_UTC_mvt","MARKET_SEGMENT_flt","AIRCRAFT_OPERATOR_flt"]).lazy()
def base(lf):
    return lf.select(
        hub=pl.when(pl.col("PHASE_mvt")=="DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
        yr=pl.col("MVT_TIME_UTC_mvt").dt.year().cast(pl.Utf8), mon=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        seg=pl.col("MARKET_SEGMENT_flt"), op=pl.col("AIRCRAFT_OPERATOR_flt"))
d = pl.concat([base(tr), base(rk)]).filter(pl.col("op").is_not_null()).collect()
print("rows with operator:", d.height)
ops25 = set(d.filter(pl.col("yr")=="2025")["op"].unique()); ops26 = set(d.filter(pl.col("yr")=="2026")["op"].unique())
print(f"operator hashes: 2025(Jan+Jul)={len(ops25)} 2026(Jan+Jul)={len(ops26)} in both={len(ops25&ops26)}")
m26 = d.filter(pl.col("yr")=="2026")
print("share of 2026 movements by operators also seen in 2025:", round(m26.filter(pl.col("op").is_in(list(ops25))).height / m26.height, 4))
# segments per operator per year
nseg = d.group_by("op","yr").agg(pl.col("seg").n_unique().alias("nseg"))
print("operator-years with >1 segment label:", nseg.filter(pl.col("nseg")>1).height, "of", nseg.height)
modal = d.group_by("op","yr","seg").len().sort("len", descending=True).group_by("op","yr").first()
w = modal.pivot(on="yr", index="op", values="seg")
ch = w.filter(pl.col("2025").is_not_null() & pl.col("2026").is_not_null() & (pl.col("2025") != pl.col("2026")))
mv = d.join(ch.select("op"), on="op").group_by("yr").len()
print("operators whose modal segment changed 2025->2026:", ch.height)
print("movements of those operators:"); print(mv)
print("transitions (count of operators, movements in 2026 Jan+Jul, movements 2025 Jan+Jul):")
tx = ch.join(d.group_by("op","yr").len().pivot(on="yr", index="op", values="len"), on="op", suffix="_n")
with pl.Config(tbl_rows=30):
    print(tx.group_by(pl.col("2025").alias("seg_2025"), pl.col("2026").alias("seg_2026")).agg(n_ops=pl.len(), mov_2025=pl.col("2025_n").sum(), mov_2026=pl.col("2026_n").sum()).sort("mov_2026", descending=True))
# Recompute segment YoY with operator segment fixed to its 2025 modal label
lab = modal.filter(pl.col("yr")=="2025").select("op", seg_fixed=pl.col("seg"))
d2 = d.join(lab, on="op", how="left")
t = d2.filter(pl.col("seg_fixed").is_not_null()).group_by("mon","seg_fixed","yr").len().pivot(on="yr", index=["mon","seg_fixed"], values="len").with_columns(yoy_pct=((pl.col("2026")/pl.col("2025"))-1)*100).sort("mon","2025", descending=[False,True])
print("\nSegment YoY with each operator frozen at its 2025 modal segment (only operators present in 2025):")
with pl.Config(tbl_rows=30, float_precision=1):
    print(t)
th = d2.filter(pl.col("seg_fixed").is_in(["Lowcost","Regional","Mainline"])).group_by("mon","hub","seg_fixed","yr").len().pivot(on="yr", index=["mon","hub","seg_fixed"], values="len").with_columns(yoy_pct=((pl.col("2026")/pl.col("2025"))-1)*100)
with pl.Config(tbl_rows=40, tbl_cols=-1, float_precision=1):
    print(th.pivot(on="seg_fixed", index=["mon","hub"], values="yoy_pct").sort("mon","hub"))
