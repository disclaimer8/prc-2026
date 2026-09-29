"""E130 / external Hypothesis 5 (journal 151): a second regressor on the RAW target, blended into the regular candidate.

    FEATURES=data/features_e106.parquet DROP_PREFIX=<E116> PROXY_SCALE=1 E130_MODE=raw CV_OUT=data/cv_e130raw \\
        .venv/bin/python src/e130_raw_reg_cv.py

Both production regressors (LightGBM src/train.py:293, CatBoost src/catboost_reg.py:38) learn the residual to the calibrated
NM proxy (proxy_base). kind-mango went the other way (raw-y model + an AOBT-anchored member, blended 0.5: board -0.70 for the
anchored CatBoost alone) and elegant-alligator blended a LOBT-anchored member at 0.63 (-1.97). For us the diverse member is the
one WITHOUT the anchor. E130_MODE=raw: target clip(y, 0, 7200); E130_MODE=lobt: y - clip(MVT-LOBT, 0, 7200) where LOBT is known,
raw elsewhere (alligator's form).

Regressor only (the regime classifier does not depend on the regressor target, so the control's probabilities are reused).
Same folds and the same nested round selection as src/cv_fold_preds_nested.py: rounds early-stopped on the inner months a+1, b+1,
x1.1, refit on all 10 fit months, predict months a, b. Writes {CV_OUT}/fold_a_b.npz (mvt_id, reg2). Blending: src/e130_blend_cv.py.
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

MODE = os.environ.get("E130_MODE", "raw")
OUT = os.environ.get("CV_OUT", f"data/cv_e130{MODE}")
PARAMS = {**T.REG_PARAMS, "num_threads": int(os.environ.get("OMP_NUM_THREADS", "6"))}
FOLDS = [(a, a + 6) for a in range(1, 7)]
if os.environ.get("E130_FOLDS"):
    FOLDS = [f for f in FOLDS if f[0] in {int(x) for x in os.environ["E130_FOLDS"].split(",")}]
SMOKE = os.environ.get("E130_SMOKE") == "1"  # 5% of rows, <= 60 rounds: plumbing check only
os.makedirs(OUT, exist_ok=True)


def target(d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(target, anchor) with prediction = model + anchor."""
    y = np.clip(d[T.TARGET].to_numpy().astype(float), *T.REG_CLIP)
    if MODE == "raw":
        anchor = np.zeros(d.height)
    elif MODE == "lobt":
        l = d["d_mvt_lobt"].to_numpy().astype(float)
        anchor = np.where(np.isnan(l), 0.0, np.clip(np.nan_to_num(l), 0, 7200))
    else:
        raise SystemExit(f"unknown E130_MODE {MODE}")
    return np.clip(y - anchor, -7200, 7200), anchor


def fit_reg(fit: pl.DataFrame, pred: pl.DataFrame, rounds: int | None):
    lab = T.regime_label(fit)
    fit, pred = T.add_encodings(fit, pred, lab == 0, fit_labels=lab)
    cols = T.feature_columns(fit)
    cat_cols = [c for c in T.CATEGORICAL if c in cols]
    Xf, cats = T.to_pandas(fit, cols, None)
    Xp, _ = T.to_pandas(pred, cols, cats)
    m = lab == 0
    tf, _ = target(fit)
    tp, ap = target(pred)
    if rounds is None:
        vm = T.regime_label(pred) == 0
        reg = lgb.train(PARAMS, lgb.Dataset(Xf[m], tf[m], categorical_feature=cat_cols), num_boost_round=60 if SMOKE else 5000,
                        valid_sets=[lgb.Dataset(Xp[vm], tp[vm])],
                        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(200, verbose=False)])
        return reg.best_iteration, None
    reg = lgb.train(PARAMS, lgb.Dataset(Xf[m], tf[m], categorical_feature=cat_cols), rounds)
    return rounds, reg.predict(Xp) + ap


tr = pl.read_parquet(T.FEATURES_PATH).filter(pl.col("split") == "train")
if SMOKE:
    tr = tr.sample(fraction=0.05, seed=1)
print(f"E130 mode {MODE}, features {T.FEATURES_PATH}, threads {PARAMS['num_threads']}, folds {FOLDS}", flush=True)
for a, b in FOLDS:
    t0 = time.time()
    inner = [a % 12 + 1, b % 12 + 1]
    f, v = tr.filter(~pl.col("month").is_in([a, b])), tr.filter(pl.col("month").is_in([a, b]))
    best, _ = fit_reg(f.filter(~pl.col("month").is_in(inner)), f.filter(pl.col("month").is_in(inner)), None)
    rounds = int(best * 1.1)
    _, reg2 = fit_reg(f, v, rounds)
    y = v[T.TARGET].to_numpy().astype(float)
    k = (T.regime_label(v) == 0) & (y <= 7200)
    np.savez(f"{OUT}/fold_{a}_{b}.npz", mvt_id=v["MVT_ID_mvt"].to_numpy(), reg2=reg2)
    print(f"fold {a}+{b}: rounds {rounds} (inner best {best}), regular rows y<=7200 RMSE {np.sqrt(np.mean((reg2[k] - y[k]) ** 2)):.2f}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
print("done", flush=True)
