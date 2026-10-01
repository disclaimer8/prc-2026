"""E178 (journal 232): a kNN regressor as an E130 member -- another model family after the MLP (E176), as in the stacking of
Wu et al. 2026 (CatBoost + SVR + kNN + MLP). Regular-regime rows, target = residual to the calibrated NM proxy (as the MLP and the
production regressor), clipped as train.py. Neighbours are searched within the departure airport, on 20 numeric features (the top
of the production regressor by gain), rank-gauss from the fit quantiles, NaN -> 0. Distance-weighted mean of K neighbours.
Same month-pair folds as the members' CV; output data/cv_e130knn/fold_a_b.npz (mvt_id, reg2) for src/e172_members_cv.py.

    FEATURES=data/features_e106.parquet PROXY_SCALE=1 .venv/bin/python src/e178_knn_member_cv.py
"""
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

import train as T  # noqa: E402

K = int(os.environ.get("KNN_K", "50"))
JOBS = int(os.environ.get("KNN_JOBS", "4"))
OUT = os.environ.get("CV_OUT", "data/cv_e130knn")
FEATS = ["d_aobt_eobt", "d_mvt_aobt", "nb_own_minus_ap", "d_aobt_sched", "nb_own_minus_rwy", "stand_prev_arr_age", "d_mvt_eobt",
         "nbf_rwy_30_30_mean", "geo_dist_m", "geo_dist_across", "d_mvt_sched", "fw_ap_60_mean", "wx_temp", "nbf_own_minus_rwy",
         "nbs_rwy_30_30_mean", "takeoffs_between_ob_mvt", "d_mvt_lobt", "stand_next_arr_gap", "d_mvt_iobt", "local_hour"]
os.makedirs(OUT, exist_ok=True)


def gauss(fit_col: np.ndarray, cols: list[np.ndarray]) -> list[np.ndarray]:
    ok = np.isfinite(fit_col)
    q = np.unique(np.quantile(fit_col[ok], np.linspace(0, 1, 257))) if ok.sum() > 100 else np.array([0.0, 1.0])
    if len(q) < 2:
        q = np.array([q[0], q[0] + 1.0])
    return [np.where(np.isfinite(c), np.interp(c, q, np.linspace(-2, 2, len(q))), 0.0) for c in cols]


tr = pl.read_parquet(T.FEATURES_PATH).filter(pl.col("split") == "train")
print(f"E178 kNN member: K {K}, features {len(FEATS)}, rows {tr.height}", flush=True)
for a in range(1, 7):
    b = a + 6
    path = f"{OUT}/fold_{a}_{b}.npz"
    if os.path.exists(path):
        continue
    t0 = time.time()
    fit, val = tr.filter(~pl.col("month").is_in([a, b])), tr.filter(pl.col("month").is_in([a, b]))
    lab = T.regime_label(fit)
    reg_fit = fit.filter(pl.Series(lab == 0))
    base_f, base_p = T.proxy_base(reg_fit, val)
    y = np.clip(np.clip(reg_fit[T.TARGET].to_numpy(), *T.REG_CLIP) - base_f, -7200, 7200)
    pred = np.zeros(val.height)
    ap_f, ap_v = reg_fit["ADEP_mvt"].to_numpy(), val["ADEP_mvt"].to_numpy()
    for ap in np.unique(ap_v):
        mf, mv = ap_f == ap, ap_v == ap
        Xf_cols, Xv_cols = [], []
        for c in FEATS:
            cf = reg_fit[c].cast(pl.Float64).to_numpy()[mf]; cv = val[c].cast(pl.Float64).to_numpy()[mv]
            gf, gv = gauss(cf, [cf, cv]); Xf_cols.append(gf); Xv_cols.append(gv)
        knn = KNeighborsRegressor(n_neighbors=K, weights="distance", n_jobs=JOBS).fit(np.column_stack(Xf_cols), y[mf])
        pred[mv] = knn.predict(np.column_stack(Xv_cols))
    reg2 = pred + base_p
    yv = val[T.TARGET].to_numpy().astype(float)
    k = (T.regime_label(val) == 0) & (yv <= 7200)
    np.savez(path, mvt_id=val["MVT_ID_mvt"].to_numpy(), reg2=reg2)
    print(f"fold {a}+{b}: regular rows y<=7200 RMSE {np.sqrt(np.mean((reg2[k] - yv[k]) ** 2)):.2f}, {(time.time() - t0) / 60:.1f} min", flush=True)
print("done", flush=True)
