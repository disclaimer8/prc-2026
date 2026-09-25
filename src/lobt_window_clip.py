"""Hypothesis 54 (25.09, external review): re-impose the LOBT matching window on the final file.

|BLOCK - LOBT| <= 3606 s held for 100 % of the 2 062 577 labelled 2025 rows (REVIEW_BRIEF, 13.09), so taxi must lie in
[d_mvt_lobt - 3606, d_mvt_lobt + 3606]. blend.py clips the base (train.lobt_clip), but the ADS-B stages (E36/E64/E70)
add their correction afterwards and do not re-clip: v247 had 31 rows outside, up to 403 s. Ceiling for v247 ~0.24 s,
CV fold estimate ~0.02 s. Pure hygiene: it can only move a prediction towards a feasible value.

Usage: python src/lobt_window_clip.py <base_version> <new_version>
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
d = j["d_mvt_lobt"].to_numpy()
p = j["TAXITIME_SEC_mvt"].to_numpy().astype(float)
ok = np.isfinite(d)
out = p.copy()
out[ok] = np.clip(p[ok], d[ok] - 3606, d[ok] + 3606)
out = np.maximum(out, 0)
ch = out != p
print(f"clipped {int(ch.sum())} rows (below {int((p < d - 3606)[ok].sum())}, above {int((p > d + 3606)[ok].sum())}), max move {np.abs(out - p).max():.0f} s, RMS move {np.sqrt(((out - p) ** 2).mean()):.3f} s")
s.with_columns(pl.Series("TAXITIME_SEC_mvt", np.round(out)).cast(s.schema["TAXITIME_SEC_mvt"])).write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
print(f"wrote v{new} = v{base} + LOBT window clip")
