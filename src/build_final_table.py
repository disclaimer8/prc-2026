"""Final-table candidate (journal 209): one table with every base-feature addition that passed its own CV, to check that they add.

    .venv/bin/python src/build_final_table.py <out> <base table> <add table> [<add table> ...]

Each add table contributes the columns it has beyond data/features_e106.parquet (the common ancestor), joined on MVT_ID_mvt.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

out, base, adds = sys.argv[1], sys.argv[2], sys.argv[3:]
anc = set(pl.read_parquet_schema("data/features_e106.parquet"))
f = pl.read_parquet(base)
for a in adds:
    new = [c for c in pl.read_parquet_schema(a) if c not in anc and c not in f.columns]
    f = f.join(pl.read_parquet(a, columns=["MVT_ID_mvt", *new]), on="MVT_ID_mvt", how="left")
    print(f"{a}: +{len(new)} columns {new}")
print(f"{out}: {f.shape}, columns beyond e106: {[c for c in f.columns if c not in anc]}")
f.write_parquet(out)
