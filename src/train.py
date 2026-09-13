"""Train the taxi-out model.

    uv run python src/train.py validate      # hold out Jan+Jul 2025, report RMSE
    uv run python src/train.py submit N      # fit on all 2025, write submissions/gentle-igloo_vN.parquet

Model = mixture of LightGBM models. For some departures (mostly long delays)
the reported block time is not an actual off-block but a planning time
(schedule, IOBT, LOBT, EOBT) or is dated one day early, so the recorded
taxi-out equals MVT minus that time. A multiclass model estimates which regime
applies; a regressor predicts ordinary taxi-out. Under squared error the best
point prediction is sum_k p_k * candidate_k, with candidate_regular = reg.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import lightgbm as lgb
import numpy as np
import polars as pl

from features import CATEGORICAL, TARGET

FEATURES_PATH = os.environ.get("FEATURES", "data/features.parquet")
REG_CLIP = (0, 7200)
TOL = 30  # seconds: |target - candidate| within this => that regime
# regime name -> feature holding MVT minus the reported planning time
CANDIDATES = {"sched": "d_mvt_sched", "iobt": "d_mvt_iobt", "lobt": "d_mvt_lobt", "eobt": "d_mvt_eobt"}
DAY = 86400.0
CLASSES = ["regular", *CANDIDATES, "dayshift"]

DROP = {
    TARGET,
    "split",
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_mvt",
    "CALLSIGN_flt",
    "BLOCK_TIME_UTC_mvt",
    "airport",
    "t_block",
    # absolute timestamps do not extrapolate from 2025 to 2026
    "t_mvt",
    "t_aobt",
    "doy",
}

REG_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    cat_smooth=20,
    min_data_per_group=50,
    max_cat_to_onehot=8,
    verbose=-1,
    num_threads=10,
)
CLF_PARAMS = {**REG_PARAMS, "objective": "multiclass", "num_class": len(CLASSES), "num_leaves": 127}


def feature_columns(df: pl.DataFrame) -> list[str]:
    cols = []
    for name, dtype in df.schema.items():
        if name in DROP or isinstance(dtype, (pl.Datetime, pl.Date)):
            continue
        if dtype == pl.String and name not in CATEGORICAL:
            continue
        cols.append(name)
    return cols


def to_pandas(df: pl.DataFrame, cols: list[str], categories: dict | None):
    pdf = df.select(cols).to_pandas()
    categories = categories if categories is not None else {}
    for c in CATEGORICAL:
        if c not in pdf:
            continue
        if c not in categories:
            categories[c] = sorted(pdf[c].dropna().unique().tolist())
        pdf[c] = pdf[c].astype("category").cat.set_categories(categories[c])
    return pdf, categories


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def regime_label(df: pl.DataFrame) -> np.ndarray:
    """Class index per row; the first matching candidate wins, dayshift is y in [1 day, 1 day + 2 h]."""
    y = pl.col(TARGET)
    expr = pl.when((y >= DAY) & (y <= DAY + 7200)).then(len(CLASSES) - 1)
    for i, col in enumerate(CANDIDATES.values(), start=1):
        expr = expr.when(((y - pl.col(col)).abs() <= TOL).fill_null(False)).then(i)
    return df.select(expr.otherwise(0)).to_series().to_numpy()


def fit_predict(fit: pl.DataFrame, pred: pl.DataFrame, cols: list[str], rounds: dict | None, val_y=None):
    """Fit regime classifier + regressor on `fit`, return (prediction, parts, best rounds, models)."""
    cat_cols = [c for c in CATEGORICAL if c in cols]
    Xf, cats = to_pandas(fit, cols, None)
    Xp, _ = to_pandas(pred, cols, cats)
    lab = regime_label(fit)
    y = fit[TARGET].to_numpy()
    reg_mask = lab == 0

    callbacks = [lgb.log_evaluation(500)]
    best = {}
    if rounds is None:
        val_lab = regime_label(pred)
        clf = lgb.train(
            CLF_PARAMS,
            lgb.Dataset(Xf, lab, categorical_feature=cat_cols),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(Xp, val_lab)],
            callbacks=callbacks + [lgb.early_stopping(200, verbose=False)],
        )
        reg = lgb.train(
            REG_PARAMS,
            lgb.Dataset(Xf[reg_mask], np.clip(y[reg_mask], *REG_CLIP), categorical_feature=cat_cols),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(Xp[val_lab == 0], np.clip(val_y[val_lab == 0], *REG_CLIP))],
            callbacks=callbacks + [lgb.early_stopping(200, verbose=False)],
        )
        best = {"clf": clf.best_iteration, "reg": reg.best_iteration}
    else:
        clf = lgb.train(CLF_PARAMS, lgb.Dataset(Xf, lab, categorical_feature=cat_cols), rounds["clf"])
        reg = lgb.train(
            REG_PARAMS,
            lgb.Dataset(Xf[reg_mask], np.clip(y[reg_mask], *REG_CLIP), categorical_feature=cat_cols),
            rounds["reg"],
        )

    proba = clf.predict(Xp)  # (n, n_classes)
    r = reg.predict(Xp)
    values = [r]
    for col in CANDIDATES.values():
        values.append(pred[col].to_numpy())
    values.append(r + DAY)
    values = np.column_stack(values)
    missing = np.isnan(values)
    proba = np.where(missing, 0.0, proba)
    proba = proba / proba.sum(axis=1, keepdims=True)
    out = np.sum(proba * np.nan_to_num(values), axis=1)
    return np.clip(out, 0, None), {"p": 1 - proba[:, 0], "reg": r}, best, (clf, reg)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    df = pl.read_parquet(FEATURES_PATH)
    cols = feature_columns(df)
    train = df.filter(pl.col("split") == "train")

    if mode == "validate":
        is_val = pl.col("month").is_in([1, 7])
        fit, val = train.filter(~is_val), train.filter(is_val)
        yv = val[TARGET].to_numpy()
        t0 = time.time()
        pred, parts, best, (clf, reg) = fit_predict(fit, val, cols, None, val_y=yv)
        clean = yv <= REG_CLIP[1]
        print(
            f"best={best} rmse={rmse(pred, yv):.1f} rmse_reg_only={rmse(parts['reg'], yv):.1f} "
            f"rmse_y<={REG_CLIP[1]}={rmse(pred[clean], yv[clean]):.1f} ({time.time() - t0:.0f}s)"
        )
        res = val.select("MVT_ID_mvt", "ADEP_mvt", "month", TARGET).with_columns(
            pred=pl.Series(pred), p=pl.Series(parts["p"]), reg=pl.Series(parts["reg"])
        )
        res.write_parquet("data/last_validation_pred.parquet")
        print(
            res.group_by("ADEP_mvt")
            .agg(rmse=((pl.col("pred") - pl.col(TARGET)) ** 2).mean().sqrt().round(0), n=pl.len())
            .sort("ADEP_mvt")
            .to_dicts()
        )
        imp = sorted(zip(reg.feature_importance("gain"), cols), reverse=True)[:20]
        print("reg top gain:", [c for _, c in imp])
        json.dump({"best": best, "rmse": rmse(pred, yv)}, open("data/last_validation.json", "w"))
        return

    if mode == "submit":
        version = int(sys.argv[2])
        best = json.load(open("data/last_validation.json"))["best"]
        rounds = {k: int(v * 1.1) for k, v in best.items()}
        rank = df.filter(pl.col("split") == "rank")
        pred, _, _, (clf, reg) = fit_predict(train, rank, cols, rounds)
        template = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
        preds = rank.select("MVT_ID_mvt").with_columns(pl.Series(TARGET, pred))
        sub = template.select("MVT_ID_mvt").join(preds, on="MVT_ID_mvt", how="left")
        assert sub.height == template.height, "row count mismatch"
        assert sub[TARGET].null_count() == 0, "missing predictions"
        sub = sub.with_columns(pl.col(TARGET).round(0).cast(template.schema[TARGET]))
        os.makedirs("submissions", exist_ok=True)
        path = f"submissions/gentle-igloo_v{version}.parquet"
        sub.write_parquet(path)
        clf.save_model(f"submissions/clf_v{version}.txt")
        reg.save_model(f"submissions/reg_v{version}.txt")
        print(f"wrote {path} rows={sub.height} rounds={rounds} mean={sub[TARGET].mean():.0f} max={sub[TARGET].max()}")


if __name__ == "__main__":
    main()
