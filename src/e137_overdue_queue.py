"""E137 (26.09, hypothesis 79, team unique-umbrella: board -2.06 from `overdue_runway_queue`): state counts at the off-block moment.

At the row's off-block estimate t (AOBT_3, else take-off - 900 s), counted over OTHER departures of the same runway / airport:
  ovq_rwy / ovq_ap : flights "overdue at the stand" — EOBT_1 <= t < AOBT_3, intervals not longer than 2 h
  rq_rwy           : flights "in the runway queue" — AOBT_3 <= t < MVT, intervals not longer than 2 h
Intervals are counted with sorted starts/ends (#starts <= t minus #ends <= t), the row itself removed. All inputs are observed
in the ranking file (2026 data).

    FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e137.parquet python src/e137_overdue_queue.py
"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

src = os.environ.get("FEATURES", "data/features_e106.parquet")
dst = os.environ.get("FEATURES_OUT", "data/features_e137.parquet")
feat = pl.read_parquet(src)
d = feat.select("MVT_ID_mvt", "split", "ADEP_mvt", "RUNWAY_mvt", "t_mvt", "t_aobt", "d_aobt_eobt").with_columns(
    t_ob=pl.col("t_aobt").fill_null(pl.col("t_mvt") - 900).cast(pl.Float64),
    t_eobt=(pl.col("t_aobt") - pl.col("d_aobt_eobt")).cast(pl.Float64),
)


def count_at(q: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    ok = np.isfinite(start) & np.isfinite(end) & (end > start) & (end - start <= 7200)
    s, e = np.sort(start[ok]), np.sort(end[ok])
    return (np.searchsorted(s, q, side="right") - np.searchsorted(e, q, side="right")).astype(float)


def self_in(q, start, end):
    ok = np.isfinite(start) & np.isfinite(end) & (end > start) & (end - start <= 7200)
    return (ok & (start <= q) & (q < end)).astype(float)


parts = []
for keys, tag in ((["split", "ADEP_mvt", "RUNWAY_mvt"], "rwy"), (["split", "ADEP_mvt"], "ap")):
    for _, g in d.drop_nulls(keys).group_by(keys):
        q = g["t_ob"].to_numpy(); eob = g["t_eobt"].to_numpy(); aob = g["t_aobt"].to_numpy().astype(float); mvt = g["t_mvt"].to_numpy().astype(float)
        cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
        cols[f"ovq_{tag}"] = count_at(q, eob, aob) - self_in(q, eob, aob)
        if tag == "rwy":
            cols["rq_rwy"] = count_at(q, aob, mvt) - self_in(q, aob, mvt)
        parts.append((tag, pl.DataFrame(cols)))
res = feat
for tag in ("rwy", "ap"):
    res = res.join(pl.concat([p for t, p in parts if t == tag]), on="MVT_ID_mvt", how="left")
new = ["ovq_rwy", "rq_rwy", "ovq_ap"]
print(res.filter(pl.col("split") == "train").select([pl.col(c).mean().round(2).alias(c + "_mean") for c in new] + [pl.col(c).is_not_null().mean().round(3).alias(c + "_cov") for c in new]).to_dicts())
res.write_parquet(dst); print("wrote", dst, res.shape)
