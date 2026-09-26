"""E132 (26.09, hypothesis 68, merry-quicksand's record-ordering block re-check): where does a record sit in the id stream?

MVT_ID_mvt and FLIGHT_ID_mvt are assigned in record order. A movement whose id deviates from the ids of its time
neighbours was written out of sequence (late, edited, re-created). merry-quicksand: `o_dev_mvt` = (MVT_ID - rolling
median of MVT_ID over +-250 rows of the whole-airport movement stream) / max(p90 - p10 of that window, 1); same for
FLIGHT_ID; board -712 MSE. Our E27 (+4.37 s) used a different construction. Adds o_dev_mvt, o_dev_flt, o_absdev_mvt to a
features file; the stream is all movements (DEP + ARR) of the airport ordered by MVT_TIME, train and rank separately.

    FEATURES=data/features_e106.parquet FEATURES_OUT=data/features_e132.parquet python src/e132_record_order.py
"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

src = os.environ.get("FEATURES", "data/features_e106.parquet")
dst = os.environ.get("FEATURES_OUT", "data/features_e132.parquet")
W = int(os.environ.get("ORDER_WINDOW", "250"))
D = "data/prc-2026-datasets/"
raw = pl.concat([
    pl.read_parquet(D + "training_*.parquet", columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).with_columns(split=pl.lit("train")),
    pl.read_parquet(D + "ranking.parquet", columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).with_columns(split=pl.lit("rank")),
], how="vertical_relaxed").with_columns(airport=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")))


def rolling_dev(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(ids); dev = np.full(n, np.nan); sc = np.full(n, np.nan)
    if n == 0:
        return dev, sc
    x = ids.astype(np.float64)
    for i in range(n):
        lo, hi = max(0, i - W), min(n, i + W + 1)
        w = x[lo:hi]
        med = np.median(w); p10, p90 = np.percentile(w, [10, 90])
        s = max(p90 - p10, 1.0)
        dev[i] = (x[i] - med) / s; sc[i] = s
    return dev, sc


parts = []
for (sp, ap), g in raw.group_by(["split", "airport"]):
    g = g.sort("MVT_TIME_UTC_mvt")
    dm, _ = rolling_dev(g["MVT_ID_mvt"].to_numpy())
    fl = g["FLIGHT_ID_mvt"].to_numpy().astype(float)
    df = np.full(len(fl), np.nan)
    m = np.isfinite(fl)
    if m.sum() > 2 * W:
        d2, _ = rolling_dev(fl[m]); df[m] = d2
    parts.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(), "o_dev_mvt": dm, "o_dev_flt": df}))
    print(sp, ap, g.height, flush=True)
o = pl.concat(parts).with_columns(o_absdev_mvt=pl.col("o_dev_mvt").abs())
feat = pl.read_parquet(src)
o = o.with_columns(pl.col("MVT_ID_mvt").cast(feat.schema["MVT_ID_mvt"]))
res = feat.join(o, on="MVT_ID_mvt", how="left")
print(res.select(pl.col("o_dev_mvt").is_not_null().mean().alias("cov_mvt"), pl.col("o_dev_flt").is_not_null().mean().alias("cov_flt"),
                 pl.col("o_dev_mvt").abs().quantile(0.99).alias("p99_abs_dev")).to_dicts())
res.write_parquet(dst); print("wrote", dst, res.shape)
