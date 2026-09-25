"""E127 (25.09, hypothesis 64, idea from team likable-eagle's `dis_state_30m`): the push-back delay of the neighbours.

Adds to a features file the statistics of other departures' NM push-back delay (AOBT_3 - EOBT_1, clipped to
[-3600, 7200], only rows with a feasible AOBT) around the row's take-off: airport (-30, 0) and (-30, +30) min,
runway (-60, 0) min — count, mean, std, own minus mean. The row itself is excluded. Everything is observed in the
ranking file (2026 data), i.e. the class of features that has transferred to the board.

    FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e127.parquet python src/e127_neighbor_pushdelay.py
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import numpy as np
import polars as pl

from features import _window_stats

src = os.environ.get("FEATURES", "data/features_e106.parquet")
dst = os.environ.get("FEATURES_OUT", "data/features_e127.parquet")
feat = pl.read_parquet(src)
dep = feat.select(
    "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t_mvt",
    pd_=pl.when(pl.col("d_aobt_eobt").is_between(-3600, 7200) & (pl.col("aobt_infeasible") != 1)).then(pl.col("d_aobt_eobt")),
)
groups = {"ap": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"]}
windows = {"ap": [(-1800, 0), (-1800, 1800)], "rwy": [(-3600, 0)]}
out = [dep.select("MVT_ID_mvt")]
for gname, keys in groups.items():
    parts = []
    for _, g in dep.drop_nulls(keys).group_by(keys):
        obs = g.drop_nulls("pd_").sort("t_mvt")
        times, vals = obs["t_mvt"].to_numpy().astype(float), obs["pd_"].to_numpy().astype(float)
        q = g["t_mvt"].to_numpy().astype(float)
        own = g["pd_"].to_numpy().astype(float)
        cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        for lo, hi in windows[gname]:
            self_in = np.where((lo <= 0) & (hi >= 0), own, np.nan)
            n, mean, std = _window_stats(times, vals, q, lo, hi, self_in) if len(times) else (np.zeros(len(q)), np.full(len(q), np.nan), np.full(len(q), np.nan))
            tag = f"nbp_{gname}_{-lo // 60}_{hi // 60}"
            cols[f"{tag}_n"], cols[f"{tag}_mean"], cols[f"{tag}_std"] = n, mean, std
            cols[f"{tag}_own_minus"] = own - mean
        parts.append(pl.DataFrame(cols))
    out.append(pl.concat(parts, how="diagonal_relaxed"))
nb = out[0]
for p_ in out[1:]:
    nb = nb.join(p_, on="MVT_ID_mvt", how="left")
res = feat.join(nb, on="MVT_ID_mvt", how="left")
new = [c for c in res.columns if c.startswith("nbp_")]
print("added", new)
print(res.select([pl.col(c).is_not_null().mean().round(3).alias(c) for c in new if c.endswith("_mean")]).to_dicts())
res.write_parquet(dst)
print("wrote", dst, res.shape)
