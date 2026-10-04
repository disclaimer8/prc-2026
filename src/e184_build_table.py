"""E184 CV table: data/features_fin1b_e184.parquet = data/features_fin1b.parquet + the three `ret_*` columns of
data/features_e184.parquet, left join on MVT_ID_mvt — the same result as

    .venv/bin/python src/build_final_table.py data/features_fin1b_e184.parquet data/features_fin1b.parquet data/features_e184.parquet

but streamed (scan -> join -> sink) with the row order of FIN1b kept, so the Mac does not hold two copies of the
2.43 M x 210 table in memory while other jobs run. Row order matters only for bit-level comparability of the LightGBM
runs with the control (data/cv_preds_fin1b_hz was trained on the FIN1b row order); the check below asserts it.

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e184_build_table.py
"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

BASE, ADD, OUT = "data/features_fin1b.parquet", "data/features_e184.parquet", "data/features_fin1b_e184.parquet"
NEW = ["ret_overdue_apt", "ret_overdue_rwy", "ret_atraso_15m"]

base_schema = pl.read_parquet_schema(BASE)
add = pl.read_parquet(ADD)
assert set(add.columns) == {"MVT_ID_mvt", *NEW} and not set(NEW) & set(base_schema)
assert add["MVT_ID_mvt"].is_duplicated().sum() == 0
add = add.with_columns(pl.col("MVT_ID_mvt").cast(base_schema["MVT_ID_mvt"]))
(pl.scan_parquet(BASE).join(add.lazy(), on="MVT_ID_mvt", how="left", maintain_order="left")
 .sink_parquet(OUT, compression="zstd"))

# checks: same rows in the same order, base columns untouched, new columns equal to the add table
out_schema = pl.read_parquet_schema(OUT)
assert list(out_schema) == [*base_schema, *NEW], "column order / set"
assert all(out_schema[c] == base_schema[c] for c in base_schema), "base dtypes"
ids_b = pl.read_parquet(BASE, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
o = pl.read_parquet(OUT, columns=["MVT_ID_mvt", "split", "t_aobt", "taxiing_at_ob", *NEW])
assert np.array_equal(ids_b, o["MVT_ID_mvt"].to_numpy()), "row order"
chk = o.select("MVT_ID_mvt", *NEW).join(add, on="MVT_ID_mvt", how="left", suffix="_a")
for c in NEW:
    a, b = chk[c].to_numpy(), chk[f"{c}_a"].to_numpy()
    assert np.array_equal(np.isnan(a.astype(float)), np.isnan(b.astype(float))) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]), c
b2 = pl.read_parquet(BASE, columns=["t_aobt", "taxiing_at_ob"])
assert o["t_aobt"].equals(b2["t_aobt"]) and o["taxiing_at_ob"].equals(b2["taxiing_at_ob"]), "base values"
print(OUT, (o.height, len(out_schema)), "| null share:",
      o.group_by("split").agg(**{c: pl.col(c).is_null().mean().round(4) for c in NEW}).sort("split").to_dicts())
