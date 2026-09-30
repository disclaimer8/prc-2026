"""E170 (journal 203): features_e163 + the en-route ATFM features of H24 (data/h24_ert_features.parquet, src/h24_enroute_atfm_probe.py)
-> data/features_e170.parquet. ert_net (network level, one value per day) stays out: a day-level column, as day_* in DROP_PREFIX.
"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

ert = pl.read_parquet("data/h24_ert_features.parquet").drop("ert_net")
f = pl.read_parquet("data/features_e163.parquet").join(ert, on="MVT_ID_mvt", how="left")
for s in ("train", "rank"):
    x = f.filter(pl.col("split") == s)
    print(s, x.height, {c: round(float(x[c].is_not_null().mean()), 3) for c in ert.columns[1:]})
f.write_parquet("data/features_e170.parquet")
print("wrote data/features_e170.parquet", f.shape)
