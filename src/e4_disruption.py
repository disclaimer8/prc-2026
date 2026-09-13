"""E4: longer taxi for departures without an NM record on January 2026 matching-outage days.

    uv run python src/e4_disruption.py <base_version> <new_version> <factor>

Rows: ranking departures without FLIGHT_ID at EDDF/EDDM/EHAM/LFPG/LSZH in January 2026 on airport-days whose share of
movements without NM is at least 3x the airport's typical day (day_nonm_ratio >= 3). The regular-regime part of the
prediction is multiplied by the factor; the other regime components are unchanged. On 11 comparable 2025 days
no-NM taxi was 1.2-2.4x that of NM neighbours (diagnostic review); 1.3 is the conservative end.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

from features import TARGET

base, new, factor = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
f = pl.read_parquet("data/features_v8.parquet", columns=["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "day_nonm_ratio"])
rows = f.filter(
    (pl.col("split") == "rank")
    & pl.col("FLIGHT_ID_mvt").is_null()
    & pl.col("ADEP_mvt").is_in(["EDDF", "EDDM", "EHAM", "LFPG", "LSZH"])
    & (pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)
    & (pl.col("day_nonm_ratio") >= 3)
)
print("rows:", rows.height, rows.group_by("ADEP_mvt").len().sort("ADEP_mvt").to_dicts())

parts = np.load("submissions/parts_v7.npz")  # regime probabilities and regular prediction of the v7 model
p_reg = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "p_regular": parts["proba"][:, 0], "reg": parts["reg"]})
sub = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
adj = rows.select("MVT_ID_mvt").join(p_reg, on="MVT_ID_mvt").join(sub, on="MVT_ID_mvt").with_columns(
    new=pl.col(TARGET) + pl.col("p_regular") * pl.col("reg") * (factor - 1)
)
out = sub.join(adj.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left").with_columns(
    pl.coalesce(pl.col("new").round(0).cast(sub.schema[TARGET]), pl.col(TARGET)).alias(TARGET)
).select(sub.columns)
assert out.height == sub.height and out[TARGET].null_count() == 0
out.write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
d = adj.select((pl.col("new") - pl.col(TARGET)).alias("d"))["d"]
print(f"wrote v{new}: {adj.height} rows changed, mean +{d.mean():.0f}s, median +{d.median():.0f}s")
