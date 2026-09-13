from common import *
d = pl.scan_parquet(FILES).filter(pl.col("PHASE_mvt")=="DEP").group_by("AIRCRAFT_TYPE_mvt").agg(pl.len().alias("N"), pl.col("WK_TBL_CAT_flt").mode().first().alias("wtc")).sort("N", descending=True).collect()
d = d.with_columns((pl.col("N").cum_sum()/pl.col("N").sum()*100).round(1).alias("cum_pct"))
print(d.head(45))
