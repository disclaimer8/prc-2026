"""E185 (idea 3 of the 04.10 competitor sweep, merry-mushroom v8): a "movement-only" LightGBM member for the rows without an NM
record outside LIRF.

    FEATURES=data/features_fin1b.parquet DROP_PREFIX=<production> OMP_NUM_THREADS=12 .venv/bin/python src/e185_motion_only_cv.py

The production regressors see a row without NM through the NaN branches of the NM columns, i.e. they learn "no NM" from
~21 k training rows. This member is trained on ALL regular 2025 rows (the same rows, folds, target and nested round selection as
the E130 raw member, src/e130_raw_reg_cv.py) with the table turned into the view a row without NM has:
  * HIDE: every column that comes from the row's own NM record (null on 100 % of no-NM rows, checked 04.10 on fin1b): the four
    NM categoricals, the NM time differences, the NM-derived flags, `*_own_minus_*` against NM proxies, plan_* (ARVT_1) and
    e157_ladder_q (from AOBT_3 / LOBT). d_mvt_sched stays: SCHED is a movement field, known on every row.
  * REPLACE: the 36 columns that features.py anchors at the row's own AOBT_3 (congestion around off-block, taxi-in of recent
    arrivals, previous/next arrival on the stand, METAR at off-block) plus airline_prefix are taken from
    data/e185_motion_features.parquet (src/e185_motion_features.py): anchor MVT - 900 s and own AOBT out of the sets for every
    row, which reproduces the no-NM rows of fin1b bit for bit.
Target: raw clip(y, 0, 7200), prediction = model (no anchor). The anchored forms are not available: LOBT/EOBT/AOBT are NM, and the
SCHED anchor is useless on this population (no-NM rows outside LIRF: median MVT - SCHED 5 215 s against median y 912 s, |y - D| <= 6 s
on 1.9 %).
Writes {CV_OUT}/fold_a_b.npz (mvt_id, reg2) for all rows of the fold months, like every E130 member. Blending on the subset only:
src/e185_subset_grid_cv.py.

Env: E130_FOLDS=1,2 (folds by first month), E185_MAX_ROUNDS (round cap, smoke runs only), REG_OVERRIDE (train.py; probes only).
The view itself (HIDE / REPLACE) lives in src/e185_motion_view.py, shared with the ranking fit src/e185_motion_rank.py.
"""
import os
import resource
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import e185_motion_view as M  # noqa: E402  (sets DROP_PREFIX = production + M.HIDE)
import train as T  # noqa: E402

OUT = os.environ.get("CV_OUT", "data/cv_e130motion")
PARAMS = {**T.REG_PARAMS, "num_threads": int(os.environ.get("OMP_NUM_THREADS", "6"))}
FOLDS = [(a, a + 6) for a in range(1, 7)]
if os.environ.get("E130_FOLDS"):
    FOLDS = [f for f in FOLDS if f[0] in {int(x) for x in os.environ["E130_FOLDS"].split(",")}]
MAX_ROUNDS = int(os.environ.get("E185_MAX_ROUNDS", "5000"))
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
tr, cols, cat_cols, replaced = M.load(["train"])
print(f"E185 movement-only: {T.FEATURES_PATH} + {M.MOTION} ({len(replaced)} columns replaced), {len(cols)} features "
      f"({len(cat_cols)} categorical), hidden {M.HIDE}, threads {PARAMS['num_threads']}, folds {FOLDS}, max rounds {MAX_ROUNDS}, "
      f"params override {os.environ.get('REG_OVERRIDE', '-')}, rows {tr.height}, loaded in {time.time() - t0:.0f} s", flush=True)


def fit_reg(fit_months: list[int], pred_months: list[int], rounds: int | None):
    """src/e130_raw_reg_cv.py fit_reg, E130_MODE=raw (target clip(y, 0, 7200), regular rows only), without add_encodings (all of
    their columns are in DROP_PREFIX). Same categories (all fit rows), same params; the matrices are built one at a time and the
    Datasets constructed before the next one, so the peak holds one copy of the fit matrix instead of four."""
    fm, pm = pl.col("month").is_in(fit_months), pl.col("month").is_in(pred_months)
    cats = M.categories(tr, fm, cat_cols)
    t_m = time.time()
    reg_rows = fm & (pl.col("_lab") == 0)
    Xf = M.matrix(tr, reg_rows, cols, cats)
    ds = lgb.Dataset(Xf, M.label(tr, reg_rows), feature_name=cols, categorical_feature=cat_cols, params=PARAMS).construct()
    del Xf
    if rounds is None:
        val_rows = pm & (pl.col("_lab") == 0)
        Xv = M.matrix(tr, val_rows, cols, cats)
        dv = lgb.Dataset(Xv, M.label(tr, val_rows), reference=ds, feature_name=cols, categorical_feature=cat_cols, params=PARAMS).construct()
        del Xv
        t_t = time.time()
        seen = [0]

        def count(env):  # the booster returned after early stopping holds only best_iteration trees
            seen[0] = env.iteration + 1
        count.order = 0  # lgb.train sorts callbacks by .order; run before early stopping raises
        reg = lgb.train(PARAMS, ds, num_boost_round=MAX_ROUNDS, valid_sets=[dv],
                        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(200, verbose=False), count])
        n = seen[0]
        print(f"  inner fit: matrices {t_t - t_m:.0f} s, {n} rounds in {time.time() - t_t:.0f} s ({(time.time() - t_t) / n:.3f} s/round)", flush=True)
        return reg.best_iteration, None
    t_t = time.time()
    reg = lgb.train(PARAMS, ds, rounds)
    t_p = time.time()
    del ds
    pred = reg.predict(M.matrix(tr, pm, cols, cats))
    print(f"  refit: matrices {t_t - t_m:.0f} s, {rounds} rounds in {t_p - t_t:.0f} s ({(t_p - t_t) / rounds:.3f} s/round), "
          f"predict {time.time() - t_p:.0f} s", flush=True)
    return rounds, pred


for a, b in FOLDS:
    t1 = time.time()
    inner = [a % 12 + 1, b % 12 + 1]
    rest = [m for m in range(1, 13) if m not in (a, b)]
    best, _ = fit_reg([m for m in rest if m not in inner], inner, None)
    t2 = time.time()
    rounds = min(int(best * 1.1), MAX_ROUNDS)
    _, reg2 = fit_reg(rest, [a, b], rounds)
    v = tr.filter(pl.col("month").is_in([a, b])).select("MVT_ID_mvt", "_lab", T.TARGET, sub=M.SUB)
    y = v[T.TARGET].to_numpy().astype(float)
    k = (v["_lab"].to_numpy() == 0) & (y <= 7200)
    sub = v["sub"].to_numpy()
    np.savez(f"{OUT}/fold_{a}_{b}.npz", mvt_id=v["MVT_ID_mvt"].to_numpy(), reg2=reg2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    print(f"fold {a}+{b}: rounds {rounds} (inner best {best}), regular rows y<=7200 RMSE {np.sqrt(np.mean((reg2[k] - y[k]) ** 2)):.2f}, "
          f"no-NM outside LIRF ({int((k & sub).sum())} rows) {np.sqrt(np.mean((reg2[k & sub] - y[k & sub]) ** 2)):.2f}; "
          f"inner {(t2 - t1) / 60:.1f} min, refit {(time.time() - t2) / 60:.1f} min, peak RSS {rss:.1f} GB", flush=True)
print("done", flush=True)
