"""Review 5, item 5 check (15.09): do the two LFPG rows without NM (targets 84 240 and 58 206 s) rank high by Isolation Forest
anomaly score among all LFPG departures without NM in 2025, using features not derived from time or the target? (Isolation Forest and, on request, Local Outlier Factor; features include day_nonm_rate, runway configuration and stand)
    FEATURES=data/features_final.parquet .venv/bin/python src/iforest_lfpg_probe.py"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import train as T

Y = T.TARGET
df = pl.read_parquet(T.FEATURES_PATH).filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LFPG") & pl.col("FLIGHT_ID_mvt").is_null())
cols = [c for c in df.columns if c.startswith(("wx_", "dep_", "arr_", "nb_ap_", "taxi_in_")) or c in ("AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "local_hour", "d_mvt_sched", "day_nonm_rate", "rwy_config", "ap_stand")]
X = df.select(cols).to_pandas()
for c in X.columns:
    if X[c].dtype == object or str(X[c].dtype).startswith("str"):
        X[c] = X[c].astype("category").cat.codes
X = X.astype(float).fillna(-999)
for seed in (0, 1, 2):
    score = -IsolationForest(n_estimators=500, random_state=seed).fit(X).score_samples(X)
    rank = (-score).argsort().argsort() + 1  # 1 = most anomalous
    big = df[Y].to_numpy() > 50000
    print(f"seed {seed}: LFPG no-NM rows {len(df)}, ranks of rows with y > 50 000: {sorted(rank[big].tolist())}, "
          f"their y {df[Y].to_numpy()[big].tolist()}, top-1% cutoff rank {int(len(df) * 0.01)}")
Xs = StandardScaler().fit_transform(X.replace(-999, np.nan).fillna(X.replace(-999, np.nan).median()))
big = df[Y].to_numpy() > 50000
for k in (10, 20, 50, 100):
    lof = LocalOutlierFactor(n_neighbors=k).fit(Xs)
    rank = (lof.negative_outlier_factor_).argsort().argsort() + 1
    print(f"LOF k={k}: ranks of rows with y > 50 000: {sorted(rank[big].tolist())} of {len(df)}")
print("features:", len(cols))
