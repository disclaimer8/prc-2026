"""E161 step 2 (journal 179): the keep-all E70 stage (CTRL + HEAD + E141) with and without monotone constraints (+1 on
mvt_first_move, mvt_move, sw_mvt_first_move, gs_mvt_leave; method advanced), one seed each, six month-pair folds on the
production-like stage base. Pre-probe: +60 s on those columns lowers the stage output on 5-20 % of rows, by ~1-2 s on average.
Gate as 21bz via fold_metrics.compare (decided metric, 4/6, trimmed). Data assembly copied from src/e134_stage_cv.py."""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
_argv, sys.argv = sys.argv, [sys.argv[0], "import"]
import e36_adsb_stand_geometry as E36  # noqa: E402
sys.argv = _argv
from adsb_visit import NAMES  # noqa: E402
from e69_visit_merge import NAMES_M  # noqa: E402
from fold_metrics import compare  # noqa: E402

SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]
HEAD = ["hd_gs0", "hd_gs_mean5", "hd_moving0", "hd_accel30", "hd_frac_mov60", "hd_t_first_stop", "hd_turn5", "hd_path_frac"]
E141 = ["e141_d_missed", "e141_t_rough"]
DIRS = {"ctrl": "data/cv_preds_e134_ctrl", "cand_a": "data/cv_preds_e134_a", "cand_b": "data/cv_preds_e134_b",
        "cand_c": "data/cv_preds_e140", "cand_d": "data/cv_preds_e141",
        "cand_e": "data/cv_preds_e142_ctrl5", "cand_f": "data/cv_preds_e142_hd5", "cand_g": "data/cv_preds_e147_nodrift5"}
DRIFT = ["n_pts", "sf_n_seg", "vi_n_pts", "vi_prev_pts", "vi_alt_first", "vm_n_pts", "vm_prev_pts"]
DIRS["cand_h"] = "data/cv_preds_e148_twofam"  # E148 (journal 121): 0.5 x cand_f (LightGBM x5) + 0.5 x CatBoost stage, same features
DIRS["cand_i"] = "data/cv_preds_e150_keep"  # E150 (journal 127): cand_d but the stage also trains on rows with |y - base| > 3600
DIRS["cand_j"] = "data/cv_preds_e150b_w025"  # E150b (journal 128): as cand_i, the added rows weighted 0.25
DIRS["cand_k"] = "data/cv_preds_e150d_clip3600"  # E150d (journal 129): as cand_i, target clipped to +-3600


def fit_stage_keep(tr, feats, w_contra=1.0, clip=1800):
    import lightgbm as lgb
    t = tr.filter(pl.col("coverage") >= 2)  # no |y - base| <= 3600 filter: rows where the record contradicts the sensor stay
    target = np.clip((t[E36.Y] - t["base"]).to_numpy(), -clip, clip)
    w = np.where(np.abs((t[E36.Y] - t["base"]).to_numpy()) > 3600, w_contra, 1.0)
    m = lgb.train(E36.PARAMS30, lgb.Dataset(t.select(feats).to_numpy(), target, weight=w, categorical_feature=[0]), 800)
    m.feats = feats
    return m


def fit_stage_cat(tr, feats, seed=0):
    from catboost import CatBoostRegressor
    t = tr.filter((pl.col("coverage") >= 2) & ((pl.col(E36.Y) - pl.col("base")).abs() <= 3600))
    target = np.clip((t[E36.Y] - t["base"]).to_numpy(), -1800, 1800)
    X = t.select(feats).to_pandas(); X[feats[0]] = X[feats[0]].fillna(-1).astype(int)
    m = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, loss_function="RMSE", random_seed=seed,
                          thread_count=int(os.environ.get("OMP_NUM_THREADS", "6")), verbose=0, cat_features=[0])
    m.fit(X, target)
    m.feats = feats
    return m


def correct_cat(model, d):
    X = d.select(model.feats).to_pandas(); X[model.feats[0]] = X[model.feats[0]].fillna(-1).astype(int)
    use = d["coverage"].to_numpy() >= 2
    return np.clip(d["base"].to_numpy() + np.where(use, model.predict(X), 0.0), 0, None)
SEEDS5 = [42, 1, 2, 3, 4]
E140 = ["nb_full", "nb_pace", "nb_n", "e140_missing", "e140_total"]
ONLY = [c for c in os.environ.get("E134_ONLY", "").split(",") if c]


def e140_features(d: pl.DataFrame, half: float = 1800.0) -> pl.DataFrame:
    """E140 (journal 89): the unseen start rebuilt from OTHER aircraft. Neighbours = coverage-3 departures (stand dwell
    and first movement seen) at the same airport with |MVT - MVT_i| <= half, self excluded. nb_full = their mean
    first-movement-to-take-off time; nb_pace = their mean seconds per metre of straight stand-runway distance;
    e140_missing = distance already behind at the first point (gs_d_first) x nb_pace; e140_total = MVT - t_first + missing."""
    full = (pl.col("t_to") - pl.col("t_move"))
    x = d.with_columns(
        _ok=((pl.col("coverage") == 3) & full.is_between(60, 3600)).cast(pl.Float64),
        _full=pl.when((pl.col("coverage") == 3) & full.is_between(60, 3600)).then(full).otherwise(0.0),
        _pace=pl.when((pl.col("coverage") == 3) & full.is_between(60, 3600) & (pl.col("gs_d_stand_rwy") > 200))
               .then(full / pl.col("gs_d_stand_rwy")).otherwise(None))
    x = x.with_columns(_okp=pl.col("_pace").is_not_null().cast(pl.Float64), _pace=pl.col("_pace").fill_null(0.0))
    parts = []
    for _, g in x.sort("to").group_by("ADEP_mvt", maintain_order=True):
        t = g["to"].to_numpy()
        lo, hi = np.searchsorted(t, t - half, "left"), np.searchsorted(t, t + half, "right")
        cols = {}
        for c in ("_ok", "_full", "_okp", "_pace"):
            v = g[c].to_numpy()
            cs = np.concatenate([[0.0], np.cumsum(v)])
            cols[c] = cs[hi] - cs[lo] - v
        n, np_ = cols["_ok"], cols["_okp"]
        parts.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "nb_n": n,
                                   "nb_full": np.where(n > 0, cols["_full"] / np.maximum(n, 1), np.nan),
                                   "nb_pace": np.where(np_ > 0, cols["_pace"] / np.maximum(np_, 1), np.nan)}))
    nb = pl.concat(parts)
    return d.join(nb, on="MVT_ID_mvt", how="left").with_columns(
        e140_missing=pl.col("gs_d_first") * pl.col("nb_pace")).with_columns(
        e140_total=pl.col("mvt_first") + pl.col("e140_missing"))

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
base = E36.oof_base()
_reg = {}
for _p in sorted(glob.glob(f"{os.environ.get('E36_BASE_DIR', 'data/cv_preds_nomonth')}/fold_*.npz")):
    _z = np.load(_p)
    _reg.update(zip(_z["mvt_id"].tolist(), _z["reg"].tolist()))
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])
E36.ADSB_DIR = "data/adsblol"
d = E36.build(raw, base, E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).join(
    vis, on="MVT_ID_mvt", how="left").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                    for f in sorted(glob.glob("data/adsblol_standwalk/offblock2_2025-*.parquet"))]).rename(
    {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
d = d.join(walked, on="MVT_ID_mvt", how="left").with_columns(
    sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
    sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
    sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))
d = d.join(pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_m_2025-*.parquet"))]), on="MVT_ID_mvt", how="left")
hd = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/head_2025-*.parquet"))]).unique("MVT_ID_mvt")
d = e140_features(d.join(hd, on="MVT_ID_mvt", how="left"))
d = d.with_columns(
    e141_d_missed=pl.col("gs_d_first"),
    e141_t_rough=(pl.col("gs_d_first") * 2.0 /
                  (pl.col("hd_gs0").clip(lower_bound=5.0) * 0.5144)).fill_nan(None),
)
print("E140 coverage on coverage>=2:", d.filter(pl.col("coverage") >= 2).select(
    pl.col("nb_full").is_not_null().mean(), pl.col("e140_missing").is_not_null().mean()).to_dicts(), flush=True)

trunc = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/head_trunc_2025-*.parquet"))]).with_columns(
    n_cand=pl.lit(1), gs_t_leave=pl.lit(None, dtype=pl.Float64))
aug = E36.build(raw, base, trunc).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())

CTRL = E36.FEATS + NAMES + SW + NEW + NAMES_M
missing = [c for c in CTRL + HEAD if c not in d.columns]
if missing:
    sys.exit(f"missing columns: {missing}")
cov2 = d.filter(pl.col("coverage") >= 2)
print(f"rows {d.height}; coverage>=2 {cov2.height}; HEAD known on coverage>=2 {int(cov2['hd_gs0'].is_not_null().sum())}; "
      f"truncated copies {aug.height}", flush=True)

import lightgbm as lgb  # noqa: E402
FEATS = CTRL + HEAD + E141
MONO = {"mvt_first_move": 1, "mvt_move": 1, "sw_mvt_first_move": 1, "gs_mvt_leave": 1}
OUTS = {"ctrl": "data/cv_preds_e161_ctrl", "mono": "data/cv_preds_e161_mono"}
for p in OUTS.values():
    os.makedirs(p, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    t = tr.filter(pl.col("coverage") >= 2)
    target = np.clip((t[E36.Y] - t["base"]).to_numpy(), -1800, 1800)
    for name, extra in (("ctrl", {}), ("mono", {"monotone_constraints": [MONO.get(c, 0) for c in FEATS], "monotone_constraints_method": "advanced"})):
        m = lgb.train({**E36.PARAMS30, **extra}, lgb.Dataset(t.select(FEATS).to_numpy(), target, categorical_feature=[0]), 800)
        m.feats = FEATS
        np.savez(f"{OUTS[name]}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=E36.correct(m, te), y=te[E36.Y].to_numpy())
    print(f"fold {a}+{b} done", flush=True)
print(compare(OUTS["ctrl"], OUTS["mono"]), flush=True)
