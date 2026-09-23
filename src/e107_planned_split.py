"""E107 (23.09): re-split the planned-regime mass among SCHED / IOBT / LOBT / EOBT row by row, keeping its total.

    FOLDS=1,2 .venv/bin/python src/e107_planned_split.py        -> data/cv_preds_e107_split
The E89 classifier's regular and day-shift probabilities, and the SUM of the four planned ones, are kept exactly; only
the split inside the four changes, so the change cannot move mass towards the long regimes (the failure of E86/E97).
A binary LightGBM scores (row, candidate) pairs of planned-regime fit rows: target = the label matches this candidate
within the labelling tolerance. Several candidates can match (partial labels) and all of them count as right. Pair
features: the row's origin features plus the candidate's own offset and its distance to every other candidate and to
the AOBT proxy. At prediction the feasible candidates' scores are normalised within the row and share the planned mass.
Oracle ceiling after E89 (known answers, not a forecast): 5.43 s (external review, 23.09).
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("FEATURES", "data/features_e77.parquet")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

CTRL = os.environ.get("E107_CTRL", "data/cv_preds_e89_cand")
OUT = os.environ.get("E107_OUT", "data/cv_preds_e107_split")
CANDS = list(T.CANDIDATES.values())  # sched, iobt, lobt, eobt -> class 1..4
NUM = ["local_hour", "local_dow", "d_mvt_aobt", "d_aobt_eobt", "flight_time_plan", "stand_prev_arr_age", "dep_before_30",
       "offblocks_around_ob_30", "taxi_in_mean_60"] + CANDS
CAT = ["ADEP_mvt", "airline_prefix", "MARKET_SEGMENT_flt", "stand_prefix", "FLIGHT_TYPE_flt"]


def pairs(df: pl.DataFrame):
    """Long table of (row, feasible candidate) with pair features; returns X, row index, candidate index."""
    C = np.column_stack([df[c].to_numpy().astype(float) for c in CANDS])
    d_l = df["d_mvt_lobt"].to_numpy().astype(float)[:, None]
    feas = ~np.isnan(C) & ~(~np.isnan(d_l) & (np.abs(C - np.nan_to_num(d_l)) > T.LOBT_WINDOW))
    r, k = np.nonzero(feas)
    base = df.select(NUM).to_numpy().astype(float)[r]
    own = C[r, k]
    others = C[r] - own[:, None]                                 # distance to every other candidate (0 for itself)
    aobt = df["d_mvt_aobt"].to_numpy().astype(float)[r]
    extra = np.column_stack([k, own, own - aobt, others, np.sum(np.abs(others) <= T.TOL, axis=1)])
    cats = {c: df[c].cast(pl.String).to_numpy()[r] for c in CAT}
    return base, extra, cats, r, k, C


def frame(base, extra, cats, codes=None):
    import pandas as pd
    X = pd.DataFrame(np.column_stack([base, extra]),
                     columns=NUM + ["cand", "own", "own_minus_aobt"] + [f"to_{c}" for c in CANDS] + ["n_coincide"])
    codes = codes or {}
    for c, v in cats.items():
        if c not in codes:
            codes[c] = pd.CategoricalDtype(sorted(set(x for x in v if x is not None)))
        X[c] = pd.Series(v).astype(codes[c]).values
    return X, codes


def fit_split(fit: pl.DataFrame):
    lab = T.regime_label(fit)
    pl_rows = (lab >= 1) & (lab <= 4)
    f = fit.filter(pl.Series(pl_rows))
    base, extra, cats, r, k, C = pairs(f)
    y = f[T.TARGET].to_numpy().astype(float)
    target = (np.abs(C[r, k] - y[r]) <= T.TOL).astype(int)
    X, codes = frame(base, extra, cats)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, seed=7, deterministic=True, force_row_wise=True, verbose=-1,
                  num_threads=int(os.environ.get("NT", "4")))
    m = lgb.train(params, lgb.Dataset(X, target, categorical_feature=CAT), int(os.environ.get("E107_ROUNDS", "400")))
    print(f"  planned fit rows {len(f)}, pairs {len(target)}, positive share {target.mean():.3f}", flush=True)
    return m, codes


def resplit(m, codes, rows: pl.DataFrame, proba: np.ndarray) -> np.ndarray:
    base, extra, cats, r, k, _ = pairs(rows)
    X, _ = frame(base, extra, cats, codes)
    s = np.clip(m.predict(X), 1e-6, 1)
    S = np.zeros((rows.height, 4))
    S[r, k] = s
    tot = S.sum(1, keepdims=True)
    q = np.where(tot > 0, S / np.where(tot > 0, tot, 1), 0)
    out = proba.copy()
    mass = proba[:, 1:5].sum(1, keepdims=True)
    ok = (tot[:, 0] > 0)
    out[ok, 1:5] = mass[ok] * q[ok]
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    folds = [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5,6").split(",")]
    tr = pl.read_parquet(T.FEATURES_PATH).filter(pl.col("split") == "train")
    for a in folds:
        b = a + 6
        path = f"{OUT}/fold_{a}_{b}.npz"
        if os.path.exists(path):
            continue
        fit, val = tr.filter(~pl.col("month").is_in([a, b])), tr.filter(pl.col("month").is_in([a, b]))
        print(f"fold {a}+{b}", flush=True)
        m, codes = fit_split(fit)
        c = np.load(f"{CTRL}/fold_{a}_{b}.npz")
        pos = {mm: i for i, mm in enumerate(val["MVT_ID_mvt"].to_numpy())}
        pv = val[np.array([pos[mm] for mm in c["mvt_id"]])]
        values = c["values"].astype(np.float64)
        proba = resplit(m, codes, pv, c["proba"].astype(np.float64))
        pred = np.clip(T.lobt_clip(np.sum(proba * np.nan_to_num(values), axis=1), pv), 0, None)
        y = c["y"]
        print(f"  mixture RMSE ctrl {np.sqrt(np.mean((c['pred'] - y) ** 2)):.3f} split {np.sqrt(np.mean((pred - y) ** 2)):.3f}; "
              f"RMS change {np.sqrt(np.mean((pred - c['pred']) ** 2)):.2f} s", flush=True)
        np.savez(path, mvt_id=c["mvt_id"], pred=pred, reg=c["reg"], y=y, proba=proba.astype(np.float32), values=c["values"])
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
