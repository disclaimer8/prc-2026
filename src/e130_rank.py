"""E130 board model (journal 166): the raw-target LightGBM regressor of src/e130_raw_reg_cv.py (E130_MODE=raw), fit on all
2025 regular rows and applied to ranking, seed-averaged.

    FEATURES=data/features_e106.parquet DROP_PREFIX=<E116> PROXY_SCALE=1 .venv/bin/python src/e130_rank.py <n_seeds> <rounds>

Same fit as the CV (regular rows, target clip(y, REG_CLIP), no observation weights, add_encodings on the fit rows); rounds = the
mean of the six nested fold choices (2380 3064 2370 3472 2383 2399 -> 2678). Seeds as src/seed_avg.py.
Writes submissions/e130_raw_rank.npz (mvt_id, reg, per_seed); blend.py mixes it in with E130_REG2=... E130_W=w.
E130_MODE=lobt (journal 198): the LOBT-anchored form of the CV (target y - clip(MVT - LOBT, 0, 7200) where LOBT is known,
prediction = model + anchor; CV -0.963, 6/6) -> submissions/e130_lobt_rank.npz.
"""
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

n_seeds, rounds = int(sys.argv[1]), int(sys.argv[2])
PARAMS = {**T.REG_PARAMS, "num_threads": int(os.environ.get("OMP_NUM_THREADS", "6"))}
df = pl.read_parquet(T.FEATURES_PATH)
fit, pred = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
lab = T.regime_label(fit)
fit, pred = T.add_encodings(fit, pred, lab == 0, fit_labels=lab)
cols = T.feature_columns(fit)
cat_cols = [c for c in T.CATEGORICAL if c in cols]
Xf, cats = T.to_pandas(fit, cols, None)
Xp, _ = T.to_pandas(pred, cols, cats)
m = lab == 0
MODE = os.environ.get("E130_MODE", "raw")
def anchor(d):
    if MODE == "raw":
        return np.zeros(d.height)
    l = d[f"d_mvt_{MODE}"].to_numpy().astype(float)  # lobt; sched / eobt / iobt for E172
    return np.where(np.isnan(l), 0.0, np.clip(np.nan_to_num(l), 0, 7200))
af, ap = anchor(fit), anchor(pred)
yf = np.clip(np.clip(fit[T.TARGET].to_numpy().astype(float), *T.REG_CLIP) - af, -7200, 7200)
dtrain = lgb.Dataset(Xf[m], yf[m], categorical_feature=cat_cols, free_raw_data=False)
print(f"E130 rank: features {T.FEATURES_PATH}, {len(cols)} columns, {int(m.sum())} regular fit rows, {pred.height} ranking rows, "
      f"rounds {rounds}, threads {PARAMS['num_threads']}", flush=True)
out = f"submissions/e130_{MODE}_rank.npz"
preds = []
for s in range(n_seeds):
    t0 = time.time()
    params = {**PARAMS, "seed": 1000 + s, "bagging_seed": 2000 + s, "feature_fraction_seed": 3000 + s}
    model = lgb.train(params, dtrain, num_boost_round=rounds)
    preds.append(model.predict(Xp) + ap)
    np.savez(out, mvt_id=pred["MVT_ID_mvt"].to_numpy(), reg=np.mean(preds, axis=0), per_seed=np.array(preds))
    print(f"seed {s} done ({(time.time() - t0) / 60:.1f} min), mean {preds[-1].mean():.1f}, saved {len(preds)}-seed average", flush=True)
print("done", flush=True)
