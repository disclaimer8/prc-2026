"""E103 (23.09): importance weights for the 2025 -> 2026 shift, from an adversarial classifier.

    .venv/bin/python src/e103_propensity.py        -> data/e103_adv_w.parquet (MVT_ID_mvt, adv_w) for train rows
The adversarial AUC 0.84 (src/adv_val.py, 15.09) was recorded as a diagnostic only. Here it becomes a weight:
w = p / (1 - p) for a 2025 row, where p = P(row looks like ranking 2026 | features), cross-fitted by calendar day
(five day groups on each side, the row's own group never sees its label). Clipped to [0.2, 5] and normalised to mean 1.
Season is part of the shift (ranking is January and July), so the weight also leans towards 2025's January and July --
that is intended, not a leak: nothing about the target is used.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("FEATURES", "data/features_e77.parquet")
os.environ.setdefault("DROP_PREFIX", "day_,tq_,wxr_,month")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

df = pl.read_parquet(T.FEATURES_PATH)
cols = [c for c in T.feature_columns(df) if c != "month"]
day = (pl.col("t_mvt").cast(pl.Int64) // 86_400_000_000) if df.schema["t_mvt"] != pl.Float64 else (pl.col("t_mvt") // 86400).cast(pl.Int64)
df = df.with_columns(adv=(pl.col("split") == "rank").cast(pl.Int8), day=day)
X, _ = T.to_pandas(df, cols, None)
y = df["adv"].to_numpy()
days = df["day"].to_numpy()
rng = np.random.default_rng(42)
uniq = np.unique(days)
grp_of = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
grp = np.array([grp_of[d] for d in days])
cat_cols = [c for c in T.CATEGORICAL if c in cols]
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, verbose=-1, seed=42,
              deterministic=True, force_row_wise=True, num_threads=int(os.environ.get("NT", "4")))
p = np.zeros(len(y))
for g in range(5):
    tr = grp != g
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], categorical_feature=cat_cols), 400)
    p[~tr] = m.predict(X[~tr])
    print(f"group {g}: rows {int((~tr).sum())}", flush=True)
from sklearn.metrics import roc_auc_score  # noqa: E402
print(f"cross-fitted AUC {roc_auc_score(y, p):.4f}", flush=True)
trn = y == 0
odds = p[trn] / np.clip(1 - p[trn], 1e-6, None)
w = np.clip(odds / odds.mean(), 0.2, 5.0)
w = w / w.mean()
ess = w.sum() ** 2 / (w ** 2).sum()
out = pl.DataFrame({"MVT_ID_mvt": df.filter(pl.col("adv") == 0)["MVT_ID_mvt"], "adv_w": w,
                    "month": df.filter(pl.col("adv") == 0)["month"], "ADEP_mvt": df.filter(pl.col("adv") == 0)["ADEP_mvt"]})
print(f"weights: effective sample {ess / len(w):.3f} of rows, clipped low {np.mean(w <= 0.2 / 1.0 + 1e-9):.3f}")
print(out.group_by("month").agg(pl.col("adv_w").mean()).sort("month"))
print(out.group_by("ADEP_mvt").agg(pl.col("adv_w").mean()).sort("ADEP_mvt"))
out.select("MVT_ID_mvt", "adv_w").write_parquet("data/e103_adv_w.parquet")
print("wrote data/e103_adv_w.parquet")
