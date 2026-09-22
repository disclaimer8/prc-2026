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
from target_encoding import add_encodings

FEATURES_PATH = os.environ.get("FEATURES", "data/features.parquet")
# E56 (17.09): the regressor answers for the regular regime and the mixture owns the tail, so the clip is a
# calibration choice, not a property of the data: only 0.028% of training rows exceed 7200 s and 0.2% exceed
# 3600 s. REG_CLIP_HI moves it.
REG_CLIP = (0, int(os.environ.get("REG_CLIP_HI", "7200")))
# A planned time substitutes the block time within +-6 s (a plateau in the histogram of BLOCK - SCHED seconds);
# EDDF, EHAM and LSZH record block time to the second and show no plateau, so planned regimes are not labelled there.
TOL = int(os.environ.get("REGIME_TOL", "6"))
PLANNED_REGIME_EXCLUDED = set(os.environ.get("REGIME_EXCLUDE", "EDDF,EHAM,LSZH").split(",")) - {""}
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
    learning_rate=0.03,
    num_leaves=511,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    cat_smooth=20,
    min_data_per_group=50,
    max_cat_to_onehot=8,
    verbose=-1,
    num_threads=10,
    # reproducibility: fixed seeds and deterministic histogram construction
    seed=42,
    deterministic=True,
    force_row_wise=True,
)
CLF_PARAMS = {**REG_PARAMS, "objective": "multiclass", "num_class": len(CLASSES), "num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 100}


def feature_columns(df: pl.DataFrame) -> list[str]:
    # DROP_PREFIX: comma-separated column prefixes to leave out (ablation experiments)
    drop_prefix = tuple(p for p in os.environ.get("DROP_PREFIX", "").split(",") if p)
    cols = []
    for name, dtype in df.schema.items():
        if name in DROP or isinstance(dtype, (pl.Datetime, pl.Date)):
            continue
        if drop_prefix and name.startswith(drop_prefix):
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
    planned_ok = ~pl.col("ADEP_mvt").is_in(list(PLANNED_REGIME_EXCLUDED))
    for i, col in enumerate(CANDIDATES.values(), start=1):
        expr = expr.when((((y - pl.col(col)).abs() <= TOL) & planned_ok).fill_null(False)).then(i)
    return df.select(expr.otherwise(0)).to_series().to_numpy()


LOBT_WINDOW = 3606  # NM flight matching window: |BLOCK - LOBT| <= 3606 s on every 2025 row that has LOBT


def lobt_clip(pred_values: np.ndarray, df: pl.DataFrame) -> np.ndarray:
    """Clip taxi-out into [MVT-LOBT-window, MVT-LOBT+window] where LOBT is known."""
    d = df["d_mvt_lobt"].to_numpy()
    lo, hi = d - LOBT_WINDOW, d + LOBT_WINDOW
    return np.where(np.isnan(d), pred_values, np.clip(pred_values, lo, hi))


def proxy_base(fit_r: pl.DataFrame, pred: pl.DataFrame):
    """Proxy chain MVT-AOBT_3 -> MVT-LOBT -> MVT-EOBT_1 (each only when in [0, 7200]), calibrated by a
    per-airport, per-source median offset learned on the fit rows; airport median taxi when nothing is available."""
    def chain(d):
        def ok(c):
            if c == "d_mvt_aobt" and os.environ.get("NO_AOBT") == "1":  # abstinence measurement: no AOBT link
                return pl.when(pl.lit(False)).then(pl.col(c))
            cond = pl.col(c).is_between(0, 7200)
            if c == "d_mvt_aobt" and os.environ.get("AOBT_GATE", "1") == "1":  # AOBT_3 outside the LOBT window cannot be the off-block time
                cond = cond & ((pl.col("d_mvt_aobt") - pl.col("d_mvt_lobt")).abs() <= LOBT_WINDOW).fill_null(True)
            return pl.when(cond).then(pl.col(c))
        return d.select(
            "ADEP_mvt",
            raw=pl.coalesce(ok("d_mvt_aobt"), ok("d_mvt_lobt"), ok("d_mvt_eobt")),
            src=pl.when(ok("d_mvt_aobt").is_not_null()).then(pl.lit("aobt"))
            .when(ok("d_mvt_lobt").is_not_null()).then(pl.lit("lobt"))
            .when(ok("d_mvt_eobt").is_not_null()).then(pl.lit("eobt"))
            .otherwise(pl.lit("none")),
        )
    cf = chain(fit_r).with_columns(y=fit_r[TARGET].clip(0, 7200))
    off = cf.filter(pl.col("raw").is_not_null()).group_by("ADEP_mvt", "src").agg(off=(pl.col("y") - pl.col("raw")).median())
    med = cf.group_by("ADEP_mvt").agg(med=pl.col("y").median())
    # PROXY_SCALE=1 (E46, 16.09): instead of a median offset, scale the proxy by a per-airport, per-source coefficient
    # "how much of MVT - filed time is really taxi", fitted by least squares on the fit rows and shrunk to the global
    # coefficient with PROXY_SCALE_RIDGE pseudo-rows (a competitor reports 0.23 at LSZH vs 0.86 at EGLL).
    # PROXY_ISO=1 (E55, 17.09): the proxy does not err proportionally — it is accurate while it is small and
    # over-predicts badly once it grows (AOBT: ~0 s below 1200 s, +580 s at 2400-3600; LOBT: +64 s below 1200 s,
    # +1502 s at 2400-3600). A single coefficient (E46) fits a straight line through that, so calibrate the
    # shape instead: an isotonic curve per airport and source, fitted on the fit rows, falling back to the
    # source-wide curve and then to the E46 coefficient where a group is too small to carry one.
    iso = None
    if os.environ.get("PROXY_ISO") == "1":
        from sklearn.isotonic import IsotonicRegression
        min_rows = int(os.environ.get("PROXY_ISO_MIN", "500"))
        cs = cf.filter(pl.col("raw").is_not_null() & (pl.col("raw") >= 0))
        iso = {}
        for src_name in cs["src"].unique().to_list():
            g = cs.filter(pl.col("src") == src_name)
            iso[(None, src_name)] = IsotonicRegression(out_of_bounds="clip").fit(g["raw"].to_numpy(), g["y"].to_numpy())
        for ap, src_name in cs.select("ADEP_mvt", "src").unique().iter_rows():
            g = cs.filter((pl.col("ADEP_mvt") == ap) & (pl.col("src") == src_name))
            if g.height >= min_rows:
                iso[(ap, src_name)] = IsotonicRegression(out_of_bounds="clip").fit(g["raw"].to_numpy(), g["y"].to_numpy())
        print(f"proxy isotonic: {len(iso)} curves ({sum(1 for k in iso if k[0] is None)} source-wide), min rows {min_rows}", flush=True)

    # PROXY_AFFINE=1 (E57, 17.09): E46 scales the proxy through the origin, so all of "how much of MVT - filed
    # time is really taxi" has to be carried by the slope. Least squares says the relation has a fixed part too:
    # with an intercept the slopes fall to 0.29-0.93 and the intercepts land at +25 s (EDDF) to +592 s (EDDM),
    # and the held-out proxy RMSE drops in all six folds (358->339, 345->335, 323->300, 321->300, 322->301,
    # 332->311). Unlike the isotonic curve of E55 this stays a smooth, shape-preserving map - one more parameter,
    # not a step function - and both parameters are stable across fit sets (intercept spread <= 24 s at nine
    # airports; LTFM, the January-disruption one, spreads 159 s). Shrinkage matches E46: PROXY_SCALE_RIDGE
    # pseudo-rows drawn from the global design moment, shrinking each group towards the global (a, b).
    affine = None
    if os.environ.get("PROXY_AFFINE") == "1":
        k = float(os.environ.get("PROXY_SCALE_RIDGE", "1000"))
        cs = cf.filter(pl.col("raw").is_not_null() & (pl.col("raw") > 0))
        xg = cs["raw"].to_numpy().astype(float)
        Xg = np.column_stack([xg, np.ones_like(xg)])
        gb = np.linalg.solve(Xg.T @ Xg, Xg.T @ cs["y"].to_numpy().astype(float))
        moment = (Xg.T @ Xg) / len(xg)  # average per-row design moment: k of these = k pseudo-rows
        rows = []
        for key, g in cs.group_by(["ADEP_mvt", "src"]):
            xi = g["raw"].to_numpy().astype(float)
            Xi = np.column_stack([xi, np.ones_like(xi)])
            sol = np.linalg.solve(Xi.T @ Xi + k * moment, Xi.T @ g["y"].to_numpy().astype(float) + k * moment @ gb)
            rows.append({"ADEP_mvt": key[0], "src": key[1], "a": float(sol[0]), "b": float(sol[1])})
        affine = pl.DataFrame(rows), gb
        print("proxy affine: global a %.3f b %+.0f, per airport-source %s" % (gb[0], gb[1], affine[0].sort("b").to_dicts()), flush=True)

    scale = None
    if os.environ.get("PROXY_SCALE") == "1":
        k = float(os.environ.get("PROXY_SCALE_RIDGE", "1000"))
        cs = cf.filter(pl.col("raw").is_not_null() & (pl.col("raw") > 0))
        g = cs.select(sxy=(pl.col("raw") * pl.col("y")).sum(), sxx=(pl.col("raw") ** 2).sum())
        a_global = float(g["sxy"][0] / max(float(g["sxx"][0]), 1e-9))
        scale = cs.group_by("ADEP_mvt", "src").agg(sxy=(pl.col("raw") * pl.col("y")).sum(), sxx=(pl.col("raw") ** 2).sum()).with_columns(
            a=(pl.col("sxy") + k * a_global * 1e6) / (pl.col("sxx") + k * 1e6)).select("ADEP_mvt", "src", "a")
        print("proxy scale: global %.3f, per airport-source %s" % (a_global, scale.sort("a").to_dicts()), flush=True)
    def apply(d):
        c = chain(d).with_row_index("_i").join(off, on=["ADEP_mvt", "src"], how="left").join(med, on="ADEP_mvt", how="left")
        if iso is not None:
            c = c.sort("_i")
            r = c["raw"].to_numpy(); aps = c["ADEP_mvt"].to_numpy(); srcs = c["src"].to_numpy()
            out = c["med"].to_numpy().astype(float).copy()
            ok = np.isfinite(r.astype(float))
            for key in set(zip(aps[ok], srcs[ok])):
                m = ok & (aps == key[0]) & (srcs == key[1])
                curve = iso.get(key) or iso.get((None, key[1]))
                if curve is None:
                    continue
                out[m] = curve.predict(r[m].astype(float))
            return np.clip(out, 0, 7200)
        if affine is not None:
            tab, gb = affine
            c = c.join(tab, on=["ADEP_mvt", "src"], how="left").sort("_i")
            fit_val = pl.col("raw") * pl.col("a").fill_null(gb[0]) + pl.col("b").fill_null(gb[1])
            return c.select(pl.coalesce(fit_val, pl.col("med")).clip(0, 7200)).to_series().to_numpy()
        if scale is not None:
            c = c.join(scale, on=["ADEP_mvt", "src"], how="left").sort("_i")
            return c.select(pl.coalesce(pl.col("raw") * pl.col("a").fill_null(1.0), pl.col("med"))).to_series().to_numpy()
        c = c.sort("_i")
        return c.select(pl.coalesce(pl.col("raw") + pl.col("off").fill_null(0), pl.col("med"))).to_series().to_numpy()
    return apply(fit_r), apply(pred)


def mask_infeasible(proba: np.ndarray, values: np.ndarray, df: pl.DataFrame) -> np.ndarray:
    """Zero the probability of non-regular regimes whose candidate lies outside the LOBT window."""
    d = df["d_mvt_lobt"].to_numpy()[:, None]
    bad = ~np.isnan(d) & (np.abs(np.nan_to_num(values) - np.nan_to_num(d)) > LOBT_WINDOW)
    bad[:, 0] = False
    out = np.where(bad, 0.0, proba)
    s = out.sum(axis=1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1), proba)


def fit_predict(fit: pl.DataFrame, pred: pl.DataFrame, cols: list[str], rounds: dict | None, val_y=None):
    """Fit regime classifier + regressor on `fit`, return (prediction, parts, best rounds, models)."""
    lab = regime_label(fit)
    fit, pred = add_encodings(fit, pred, lab == 0, fit_labels=lab)
    cols = feature_columns(fit)
    cat_cols = [c for c in CATEGORICAL if c in cols]
    Xf, cats = to_pandas(fit, cols, None)
    Xp, _ = to_pandas(pred, cols, cats)
    y = fit[TARGET].to_numpy()
    reg_mask = lab == 0
    # E73 / MB3 (kind-mango, -5.99 s on their leaderboard, 18.09): the LIRF tail table overwrites every LIRF
    # departure with no NM record and D >= 4 h, so the regressor is currently being shaped by labels whose
    # predictions never reach a submission. Same population fold_metrics.decided_map already excludes from
    # the gate. Their own hold-out could not price this -- it showed up only on the board.
    if os.environ.get("SERVED_ONLY") == "1":
        tail = ((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()
                & (pl.col("d_mvt_sched") >= 14400)).fill_null(False)
        drop = fit.select(tail.alias("t"))["t"].to_numpy()
        reg_mask = reg_mask & ~drop
        print(f"SERVED_ONLY: regressor drops {int(drop.sum())} rows the tail table overwrites", flush=True)

    # the regressor learns the residual against a calibrated NM off-block proxy
    base_f, base_p = proxy_base(fit.filter(pl.Series(reg_mask)), pred)
    resid_target = np.clip(np.clip(y[reg_mask], *REG_CLIP) - base_f, -7200, 7200)

    # E86: the classifier is scored by the squared error of the MIXTURE, but trained on plain multiclass
    # log-loss, which prices a probability slip identically whether the candidates lie 100 s or 10 000 s
    # apart. CLF_COST_WEIGHT=1 weights each row by how expensive a regime mistake is there: the squared
    # spread of its candidates, normalised to the median and clipped so no row dominates the fit.
    clf_w = None
    if os.environ.get("CLF_COST_WEIGHT") == "1":
        # the regular candidate stands at the airport's own median taxi on the fit rows, not a constant
        reg_med = (fit.filter(pl.Series(lab == 0)).group_by("ADEP_mvt")
                   .agg(pl.col(TARGET).median().alias("m")))
        mm = dict(zip(reg_med["ADEP_mvt"].to_list(), reg_med["m"].to_list()))
        reg_cand = np.array([mm.get(a, 900.0) for a in fit["ADEP_mvt"].to_list()], dtype=float)
        cand = np.column_stack([fit[c].to_numpy().astype(float) for c in CANDIDATES.values()])
        # a planned candidate outside the LOBT window can never be chosen (mask_infeasible), so it must not
        # inflate the cost of a mistake either
        d_lobt = fit["d_mvt_lobt"].to_numpy().astype(float)[:, None]
        far = ~np.isnan(d_lobt) & (np.abs(np.nan_to_num(cand) - np.nan_to_num(d_lobt)) > LOBT_WINDOW)
        cand = np.where(far, np.nan, cand)
        cand = np.column_stack([cand, reg_cand])
        with np.errstate(invalid="ignore"):
            spread = np.nanmax(cand, axis=1) - np.nanmin(cand, axis=1)
        spread = np.nan_to_num(spread, nan=0.0)
        med = float(np.median(spread[spread > 0])) or 1.0
        # the ceiling is 400, not 20: at 20 the clip flattened 92.8 % of the LIRF no-NM 0-4 h rows, exactly
        # the group whose cost of a mistake varies 74-fold inside itself
        clf_w = np.clip((spread / med) ** 2, 0.05, 400.0)
        # E89: E86 bought the tail with ordinary rows (y <= 7200 +6.96 s, worse in 6 of 6): down-weighting the
        # light majority broke the calibration where it worked. CLF_COST_MIN_SPREAD=T keeps every row whose
        # candidates lie within T seconds at weight exactly 1 and prices only the rows beyond, relative to T.
        t_min = float(os.environ.get("CLF_COST_MIN_SPREAD", "0"))
        if t_min > 0:
            clf_w = np.where(spread <= t_min, 1.0, np.clip((spread / t_min) ** 2, 1.0, 400.0))
            print(f"CLF_COST_MIN_SPREAD={t_min:.0f}: вес выше 1 у {100 * np.mean(clf_w > 1):.2f} % строк", flush=True)
        print(f"CLF_COST_WEIGHT: медиана разброса {med:.0f} с, вес {clf_w.min():.2f}..{clf_w.max():.1f}, "
              f"на верхней обрезке {100 * np.mean(clf_w >= 399.9):.2f} %", flush=True)


    def _reg_weight(models):
        """E94 (22.09, form proposed by the external review): the regressor's error is paid only in proportion
        to the probability that its own candidate is the one used, so rows whose candidates lie within an hour
        keep weight 1 and wider rows are priced by p_regular^2. Weight only, nothing else moves."""
        if os.environ.get("REG_COST_WEIGHT") != "1":
            return None
        cand_r = np.column_stack([fit[c].to_numpy().astype(float) for c in CANDIDATES.values()])
        d_l = fit["d_mvt_lobt"].to_numpy().astype(float)[:, None]
        far_r = ~np.isnan(d_l) & (np.abs(np.nan_to_num(cand_r) - np.nan_to_num(d_l)) > LOBT_WINDOW)
        cand_r = np.where(far_r, np.nan, cand_r)
        with np.errstate(invalid="ignore"):
            spread_r = np.nan_to_num(np.nanmax(cand_r, axis=1) - np.nanmin(cand_r, axis=1), nan=0.0)
        p_reg = np.mean([m.predict(Xf) for m in models], axis=0)[:, 0]
        thr = float(os.environ.get("REG_COST_MIN_SPREAD", "3600"))
        w = np.where(spread_r <= thr, 1.0, np.clip(p_reg ** 2, 0.05, 1.0))[reg_mask]
        print(f"REG_COST_WEIGHT: ниже 1 у {100 * np.mean(w < 1):.2f} % строк регрессора, минимум {w.min():.3f}", flush=True)
        return w

    callbacks = [lgb.log_evaluation(500)]
    best = {}
    if rounds is None:
        val_lab = regime_label(pred)
        clf = lgb.train(
            CLF_PARAMS,
            lgb.Dataset(Xf, lab, weight=clf_w, categorical_feature=cat_cols),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(Xp, val_lab)],
            callbacks=callbacks + [lgb.early_stopping(200, verbose=False)],
        )
        clf = [clf]
        reg = lgb.train(
            REG_PARAMS,
            lgb.Dataset(Xf[reg_mask], resid_target, weight=_reg_weight(clf), categorical_feature=cat_cols),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(Xp[val_lab == 0], np.clip(val_y[val_lab == 0], *REG_CLIP) - base_p[val_lab == 0])],
            callbacks=callbacks + [lgb.early_stopping(200, verbose=False)],
        )
        best = {"clf": clf[0].best_iteration, "reg": reg.best_iteration}
    else:
        # E95: seed-average the classifier (CLF_SEEDS=k). Variance reduction, no added capacity -- the same
        # change already accepted for the regressor as E37. Only the sampling seeds move; rounds stay as chosen.
        seeds = range(int(os.environ.get("CLF_SEEDS", "1")))
        clf = [lgb.train(CLF_PARAMS if i == 0 else  # i=0 must stay bit-identical to the pre-E95 model
                         {**CLF_PARAMS, "seed": 42 + i, "bagging_seed": 3 + i, "feature_fraction_seed": 2 + i},
                         lgb.Dataset(Xf, lab, weight=clf_w, categorical_feature=cat_cols), rounds["clf"])
               for i in seeds]
        reg = lgb.train(
            REG_PARAMS,
            lgb.Dataset(Xf[reg_mask], resid_target, weight=_reg_weight(clf), categorical_feature=cat_cols),
            rounds["reg"],
        )

    proba = np.mean([c.predict(Xp) for c in clf], axis=0)  # (n, n_classes), averaged over CLF_SEEDS models
    r = reg.predict(Xp) + base_p
    values = [r]
    for col in CANDIDATES.values():
        values.append(pred[col].to_numpy())
    values.append(r + DAY)
    values = np.column_stack(values)
    missing = np.isnan(values)
    proba = np.where(missing, 0.0, proba)
    proba = mask_infeasible(proba, values, pred)
    proba = proba / proba.sum(axis=1, keepdims=True)
    out = lobt_clip(np.sum(proba * np.nan_to_num(values), axis=1), pred)
    return np.clip(out, 0, None), {"p": 1 - proba[:, 0], "reg": r, "proba": proba, "values": values}, best, (clf, reg, cols)


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
        pred, parts, best, (clf, reg, cols) = fit_predict(fit, val, cols, None, val_y=yv)
        clean = yv <= REG_CLIP[1]
        print(
            f"best={best} rmse={rmse(pred, yv):.1f} rmse_reg_only={rmse(parts['reg'], yv):.1f} "
            f"rmse_y<={REG_CLIP[1]}={rmse(pred[clean], yv[clean]):.1f} ({time.time() - t0:.0f}s)"
        )
        res = val.select("MVT_ID_mvt", "ADEP_mvt", "month", TARGET).with_columns(
            pred=pl.Series(pred), p=pl.Series(parts["p"]), reg=pl.Series(parts["reg"])
        )
        res.write_parquet("data/last_validation_pred.parquet")
        np.savez("data/last_validation_parts.npz", mvt_id=val["MVT_ID_mvt"].to_numpy(), proba=parts["proba"],
                 values=parts["values"], reg=parts["reg"], pred=pred, y=yv, features=os.environ.get("FEATURES", FEATURES_PATH))
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
        pred, parts, _, (clf, reg, cols) = fit_predict(train, rank, cols, rounds)
        np.savez(f"submissions/parts_v{version}.npz", mvt_id=rank["MVT_ID_mvt"].to_numpy(), proba=parts["proba"], values=parts["values"], reg=parts["reg"])
        template = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
        preds = rank.select("MVT_ID_mvt").with_columns(pl.Series(TARGET, pred))
        sub = template.select("MVT_ID_mvt").join(preds, on="MVT_ID_mvt", how="left")
        assert sub.height == template.height, "row count mismatch"
        assert sub[TARGET].null_count() == 0, "missing predictions"
        sub = sub.with_columns(pl.col(TARGET).round(0).cast(template.schema[TARGET]))
        os.makedirs("submissions", exist_ok=True)
        path = f"submissions/gentle-igloo_v{version}.parquet"
        sub.write_parquet(path)
        clf[0].save_model(f"submissions/clf_v{version}.txt")
        for i, c in enumerate(clf[1:], start=1):  # E95: extra seeds, read back by blend.py if present
            c.save_model(f"submissions/clf_v{version}_s{i}.txt")
        reg.save_model(f"submissions/reg_v{version}.txt")
        print(f"wrote {path} rows={sub.height} rounds={rounds} mean={sub[TARGET].mean():.0f} max={sub[TARGET].max()}")


if __name__ == "__main__":
    main()
