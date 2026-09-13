"""Regressor experiments on regular-regime departures (fit: 2025 minus Jan/Jul, val: Jan+Jul 2025).

    uv run python src/exp_reg.py base
    uv run python src/exp_reg.py te            # + target encodings
    uv run python src/exp_reg.py te cat        # CatBoost instead of LightGBM
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import polars as pl

import train as T
from features import CATEGORICAL, TARGET
from target_encoding import add_encodings


def main() -> None:
    args = set(sys.argv[1:])
    df = pl.read_parquet(os.environ.get("FEATURES", "data/features_wx.parquet"))
    train = df.filter(pl.col("split") == "train")
    is_val = pl.col("month").is_in([1, 7])
    fit, val = train.filter(~is_val), train.filter(is_val)
    fit_lab, val_lab = T.regime_label(fit), T.regime_label(val)

    if "te" in args:
        t0 = time.time()
        fit, val = add_encodings(fit, val, fit_lab == 0)
        print(f"encodings {time.time() - t0:.0f}s")

    cols = T.feature_columns(fit)
    fit_r, val_r = fit.filter(pl.Series(fit_lab == 0)), val.filter(pl.Series(val_lab == 0))
    if "filt" in args:  # plausibility filter on the regressor's training rows only (validation untouched)
        fit_r = fit_r.filter(pl.col(TARGET).is_between(60, 3600))
    yf = np.clip(fit_r[TARGET].to_numpy(), *T.REG_CLIP)
    yv = val_r[TARGET].to_numpy()
    base_f = base_v = np.zeros(len(yv))
    if "resid" in args:  # learn the residual against a calibrated NM off-block proxy
        base_f, base_v = T.proxy_base(fit_r, val_r)
        yf = np.clip(yf - base_f, -7200, 7200)
    clean = yv <= T.REG_CLIP[1]
    t0 = time.time()

    if "xgb" in args:
        import xgboost as xgb

        Xf, cats = T.to_pandas(fit_r, cols, None)
        Xv, _ = T.to_pandas(val_r, cols, cats)
        dfit = xgb.DMatrix(Xf, label=yf, enable_categorical=True)
        dval = xgb.DMatrix(Xv, label=np.clip(yv, *T.REG_CLIP) - base_v, enable_categorical=True)
        params = dict(
            objective="reg:squarederror", tree_method="hist", max_depth=10, eta=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=50, reg_lambda=10.0, max_cat_to_onehot=8, max_bin=255,
            nthread=int(os.environ.get("XGB_THREADS", "5")),
        )
        model = xgb.train(params, dfit, num_boost_round=6000, evals=[(dval, "val")], early_stopping_rounds=200, verbose_eval=500)
        pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        best = model.best_iteration
    elif "cat" in args:
        from catboost import CatBoostRegressor, Pool

        cat_cols = [c for c in CATEGORICAL if c in cols]
        def pool(d, y=None):
            pdf = d.select(cols).to_pandas()
            for c in cat_cols:
                pdf[c] = pdf[c].fillna("NA").astype(str)
            return Pool(pdf, y, cat_features=cat_cols)
        model = CatBoostRegressor(
            iterations=6000, learning_rate=0.08, depth=8, loss_function="RMSE", thread_count=12,
            od_type="Iter", od_wait=200, verbose=500,
        )
        model.fit(pool(fit_r, yf), eval_set=pool(val_r, np.clip(yv, *T.REG_CLIP) - base_v), use_best_model=True)
        pred = model.predict(pool(val_r))
        best = model.get_best_iteration()
    else:
        params = dict(T.REG_PARAMS)
        params["num_threads"] = int(os.environ.get("LGB_THREADS", params["num_threads"]))
        if "slow" in args:
            params.update(learning_rate=0.03, num_leaves=511, min_data_in_leaf=50)
        if "regul" in args:  # small trees, strong regularisation (PRC 2024 winner style)
            params.update(learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, lambda_l1=50.0, lambda_l2=200.0,
                          cat_l2=100.0, cat_smooth=100.0, max_bin=511, feature_fraction=0.75)
        Xf, cats = T.to_pandas(fit_r, cols, None)
        Xv, _ = T.to_pandas(val_r, cols, cats)
        model = lgb.train(
            params,
            lgb.Dataset(Xf, yf, categorical_feature=[c for c in CATEGORICAL if c in cols]),
            num_boost_round=8000,
            valid_sets=[lgb.Dataset(Xv, np.clip(yv, *T.REG_CLIP) - base_v)],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(1000)],
        )
        pred = model.predict(Xv, num_iteration=model.best_iteration)
        best = model.best_iteration
        imp = sorted(zip(model.feature_importance("gain"), cols), reverse=True)[:15]
        print("top gain:", [c for _, c in imp])

    pred = pred + base_v
    tag = "_".join(sorted(args)) or "base"
    np.save(f"data/exp_reg_{tag}.npy", pred)
    print(
        f"[{tag}] best={best} rmse_regular={T.rmse(pred, yv):.1f} rmse_regular_clean={T.rmse(pred[clean], yv[clean]):.1f} "
        f"n={len(yv)} ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
