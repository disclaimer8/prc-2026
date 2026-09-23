# E108: fill NM-derived fields of rows without an NM record from lookups built on NM rows (no target used):
# operator / market segment / flight type by airline prefix (mode), wake category by aircraft type (mode).
import polars as pl
f = pl.read_parquet("data/features_e106.parquet")
nm = f.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
def mode_map(key, col):
    return (nm.drop_nulls([key, col]).group_by(key, col).len().sort("len", descending=True)
              .group_by(key).agg(pl.col(col).first().alias(col + "_imp")))
no = pl.col("FLIGHT_ID_mvt").is_null()
for key, col in [("airline_prefix", "AIRCRAFT_OPERATOR_flt"), ("airline_prefix", "MARKET_SEGMENT_flt"),
                 ("airline_prefix", "FLIGHT_TYPE_flt"), ("AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt")]:
    before = f.filter(no)[col].null_count()
    f = f.join(mode_map(key, col), on=key, how="left", maintain_order="left")
    f = f.with_columns(pl.when(no & pl.col(col).is_null()).then(pl.col(col + "_imp")).otherwise(pl.col(col)).alias(col)).drop(col + "_imp")
    print(col, "no-NM null", before, "->", f.filter(no)[col].null_count())
f.write_parquet("data/features_e108.parquet")
print("rows", f.height)
