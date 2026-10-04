"""E185 board model: the movement-only member (src/e185_motion_only_cv.py) fit on all 2025 regular rows and applied to ranking,
seed-averaged, as src/e130_rank.py does for the E130 members.

    FEATURES=data/features_fin1b.parquet DROP_PREFIX=<production> OMP_NUM_THREADS=12 .venv/bin/python src/e185_motion_rank.py <n_seeds> <rounds>

Same view (src/e185_motion_view.py), target clip(y, 0, 7200), regular fit rows, categories of all fit rows; rounds = the mean of
the six nested fold choices printed by the CV log; seeds as src/e130_rank.py (seed 1000+s, bagging 2000+s, feature fraction 3000+s).
Writes E185_RANK_OUT (default data/e185_motion_rank.npz): mvt_id, reg (seed average), per_seed — every ranking row; only the rows
without NM outside LIRF are used (src/e185_apply.py).
"""
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import e185_motion_view as M  # noqa: E402
import train as T  # noqa: E402

n_seeds, rounds = int(sys.argv[1]), int(sys.argv[2])
OUT = os.environ.get("E185_RANK_OUT", "data/e185_motion_rank.npz")
PARAMS = {**T.REG_PARAMS, "num_threads": int(os.environ.get("OMP_NUM_THREADS", "6"))}
df, cols, cat_cols, replaced = M.load(["train", "rank"])
fm = pl.col("split") == "train"
cats = M.categories(df, fm, cat_cols)
reg_rows = fm & (pl.col("_lab") == 0)
ds = lgb.Dataset(M.matrix(df, reg_rows, cols, cats), M.label(df, reg_rows), feature_name=cols, categorical_feature=cat_cols,
                 free_raw_data=False)  # constructed by the first lgb.train, as in src/e130_rank.py
Xp = M.matrix(df, pl.col("split") == "rank", cols, cats)
ids = df.filter(pl.col("split") == "rank")["MVT_ID_mvt"].to_numpy()
nsub = int(df.filter(pl.col("split") == "rank").select(M.SUB).to_series().sum())
print(f"E185 rank: {T.FEATURES_PATH} + {M.MOTION}, {len(cols)} features, {int(df.select(reg_rows).to_series().sum())} regular fit rows, {len(ids)} ranking rows "
      f"({nsub} without NM outside LIRF), rounds {rounds}, threads {PARAMS['num_threads']}", flush=True)
preds = []
for s in range(n_seeds):
    t0 = time.time()
    params = {**PARAMS, "seed": 1000 + s, "bagging_seed": 2000 + s, "feature_fraction_seed": 3000 + s}
    model = lgb.train(params, ds, num_boost_round=rounds)
    preds.append(model.predict(Xp))
    np.savez(OUT, mvt_id=ids, reg=np.mean(preds, axis=0), per_seed=np.array(preds))
    print(f"seed {s} done ({(time.time() - t0) / 60:.1f} min), mean {preds[-1].mean():.1f}, saved {len(preds)}-seed average", flush=True)
print("done", flush=True)
