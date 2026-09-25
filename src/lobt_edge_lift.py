"""Hypothesis 36 (25.09): lift predictions pinned to the lower edge of the LOBT matching window.

The NM matching window is |BLOCK - LOBT| <= 3606 s for 100 % of the 2 062 577 labelled 2025 rows, so a
prediction sitting exactly on the lower edge (taxi = d_mvt_lobt - 3606) is a clipped candidate. In 2025 the
mean overshoot above the edge is +1074 s for rows exactly on it and +165 s for rows within 60 s of it.
CV (decided rows): -0.139 s, lower in 6/6 folds. Board: v227 = v226 - 0.008, v247 = v246 - 0.02.

Usage: python src/lobt_edge_lift.py <base_version> <new_version>
"""
import os
import sys

import numpy as np
import polars as pl

base, new = int(sys.argv[1]), int(sys.argv[2])
features = os.environ.get("FEATURES", "data/features_e106.parquet")
s = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
r = pl.read_parquet(features, columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "rank")
j = s.join(r.select("MVT_ID_mvt", "d_mvt_lobt"), on="MVT_ID_mvt", how="left", maintain_order="left")
lo = j["d_mvt_lobt"].to_numpy() - 3606
p = j["TAXITIME_SEC_mvt"].to_numpy().astype(float)
g1 = np.isfinite(lo) & (np.abs(p - lo) <= 2)
g2 = np.isfinite(lo) & (p - lo > 2) & (p - lo <= 60)
out = p.copy()
out[g1] = lo[g1] + 1074
out[g2] = np.maximum(p[g2], lo[g2] + 165)
print(f"lifted exact {int(g1.sum())} near {int(g2.sum())} RMS change {np.sqrt(((out - p) ** 2).mean()):.2f} s")
s.with_columns(pl.Series("TAXITIME_SEC_mvt", np.round(out)).cast(s.schema["TAXITIME_SEC_mvt"])).write_parquet(
    f"submissions/gentle-igloo_v{new}.parquet")
print(f"wrote v{new} = v{base} + LOBT edge lift")
