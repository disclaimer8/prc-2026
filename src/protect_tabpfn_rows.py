"""Fable review 3.3 (journal 192-193): keep the 42 TabPFN rows of the LIRF tail at their TabPFN values after the ADS-B stages and E81.

    .venv/bin/python src/protect_tabpfn_rows.py <final version> <TabPFN-rows version (the tail's first file)> <new version>

Rows = v361 vs v291 (the 42 rows the TabPFN step writes, as in hz_builds_*); value = the TabPFN-rows file's value.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

fin_v, tab_v, new_v = (int(x) for x in sys.argv[1:4])
Y = "TAXITIME_SEC_mvt"
r = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet")
ids = r(361).join(r(291), on="MVT_ID_mvt", suffix="_b").filter(pl.col(Y) != pl.col(f"{Y}_b")).select("MVT_ID_mvt")
assert ids.height == 42
tab = r(tab_v).join(ids, on="MVT_ID_mvt").rename({Y: "tab"})
fin = r(fin_v)
out = fin.join(tab, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("tab", Y).cast(fin.schema[Y]).alias(Y)).select(fin.columns)
d = (out.sort("MVT_ID_mvt")[Y].cast(pl.Float64) - fin.sort("MVT_ID_mvt")[Y].cast(pl.Float64)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v} = v{fin_v} with the 42 TabPFN rows of v{tab_v}: changed {(d != 0).sum()}, max |d| {np.abs(d).max():.0f} s, RMS {np.sqrt(np.mean(d ** 2)):.3f} s")
