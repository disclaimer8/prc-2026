"""Adversarial validation 2025 vs 2026 (diagnostic only, reviews 3 and 5; draft by a reviewer, corrected 15.09).

    FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_ .venv/bin/python src/adv_val.py
Corrections to the draft: the model's own feature file and columns; the same season on both sides (January and July 2025 vs
January and July 2026), otherwise the classifier learns the season; train/validation split by calendar day, because rows of
one day share neighbour and runway-configuration features. Reports overall AUC, top features by gain, and single-feature AUC
of the top features (drift of that feature alone). No feature is dropped automatically.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import train as T

df = pl.read_parquet(T.FEATURES_PATH).filter(pl.col("month").is_in([1, 7]))
cols = [c for c in T.feature_columns(df) if c != "month"]
df = df.with_columns(adv=(pl.col("split") == "rank").cast(pl.Int8), day=pl.col("t_mvt").cast(pl.Int64) // 86_400_000_000
                     if df.schema["t_mvt"] != pl.Float64 else (pl.col("t_mvt") // 86400).cast(pl.Int64))
X, _ = T.to_pandas(df, cols, None)
y = df["adv"].to_numpy()
days = df["day"].to_numpy()
rng = np.random.default_rng(42)
uniq = np.unique(days)
val_days = rng.choice(uniq, size=len(uniq) // 5, replace=False)
va = np.isin(days, val_days)
cat_cols = [c for c in T.CATEGORICAL if c in cols]
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, seed=42,
              deterministic=True, num_threads=10)
clf = lgb.train(params, lgb.Dataset(X[~va], y[~va], categorical_feature=cat_cols), 1000,
                valid_sets=[lgb.Dataset(X[va], y[va], categorical_feature=cat_cols)], callbacks=[lgb.early_stopping(50, verbose=False)])
auc = clf.best_score["valid_0"]["auc"]
print(f"rows {len(y)} (2026 share {y.mean():.2f}), days {len(uniq)}, validation days {len(val_days)}")
print(f"Adversarial AUC (Jan+Jul 2025 vs Jan+Jul 2026, split by day): {auc:.4f}")
imp = sorted(zip(clf.feature_importance("gain"), cols), reverse=True)
total = sum(g for g, _ in imp)
print("\nTop-30 features by gain (share of gain, single-feature AUC on validation days):")
for g, name in imp[:30]:
    v = X.loc[va, name]
    if str(v.dtype) == "category":
        v = v.cat.codes
    v = v.astype(float).to_numpy()
    ok = ~np.isnan(v)
    a = roc_auc_score(y[va][ok], v[ok]) if ok.sum() > 100 and len(np.unique(y[va][ok])) == 2 else float("nan")
    print(f"{name:34s} {g / total:6.3f}  AUC {max(a, 1 - a):.3f}  missing {1 - ok.mean():.3f}")
