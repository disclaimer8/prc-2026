"""E35: E30 stage extended to departures WITHOUT their own surface track: neighbour ADS-B taxi times (take-off minus first
movement, and minus the end of the last stop) of covered departures at the same airport within +-30 min and on the same runway
within +-60 min (count, mean), added to the stage features; a second stage model is trained on uncovered rows (coverage < 2)
with the neighbour features only. Control: the E30 stage (covered rows only). Derived from src/e30_adsb_stage_v2.py (15.09).

E30: E25 stage with extra surface-track features (offblock3: path length, moving time, stops, last stop, speed) and a larger
stage model (63 leaves, 800 rounds). Control: the E25 stage recomputed on the same days. Based on src/e25_adsb_stage.py (15.09).

E25 description follows.

    python src/e35_adsb_neighbours.py cv      6 folds: E30 + uncovered-row stage against E30 -> decision (apply not implemented yet)
    .venv/bin/python src/e30_adsb_stage_v2.py apply <base> <new>  (base must NOT already contain E25)
Base: out-of-fold predictions of the canonical model without `month` (data/cv_preds_nomonth). Features per departure from
data/adsblol/offblock2_<day>.parquet (coverage, first surface point, first movement after the stand, last stop end, surface
time, stops) relative to AOBT_3, LOBT, MVT and to the base prediction; plus airport, D, non-regular probability and the base
prediction. Target: residual y - base, clipped to +-1800 s; training rows with |residual| > 3600 s are dropped (regime tail).
The correction is applied only to rows with coverage >= 2. Stage model: small LightGBM (31 leaves, min_data 200, 400 rounds,
lr 0.03), deterministic. For fold (a, b) the stage is trained on the other 10 months only.
Gate (REVIEW_BRIEF.md, E35): total SSE lower than the E30 stage by >= 0.3 s LB-equivalent and lower in >= 4 of 6 folds;
concentration (top-3 rows) reported, not gated. Months without a surface file for a day keep the base prediction.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
LB_SCALE = 6 * 2 * 273.41 * 344841
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FEATS = ["ap", "coverage", "D", "p_nonreg", "base", "has_nm", "stat_total", "n_stops", "n_pts",
         "mvt_first", "mvt_first_move", "mvt_move", "mvt_to", "aobt_first", "aobt_first_move", "aobt_move", "lobt_first_move",
         "base_vs_first_move", "base_vs_first"]
SF = ["sf_path_km", "sf_straight_km", "sf_move_time", "sf_longest_stop", "sf_last_stop", "sf_mean_speed", "sf_n_seg", "sf_to_gap"]
FEATS25 = list(FEATS)
FEATS = FEATS + SF
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, deterministic=True, force_row_wise=True, num_threads=8)


def load_adsb(pattern: str) -> pl.DataFrame:
    files = sorted(glob.glob(pattern))
    o2 = pl.concat([pl.read_parquet(f) for f in files])
    o3 = pl.concat([pl.read_parquet(f.replace("offblock2_", "offblock3_")) for f in files if os.path.exists(f.replace("offblock2_", "offblock3_"))])
    return o2.join(o3, on="MVT_ID_mvt", how="left")


def build(raw: pl.DataFrame, base: pl.DataFrame, adsb: pl.DataFrame) -> pl.DataFrame:
    s = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
    d = raw.join(base, on="MVT_ID_mvt", how="inner").join(adsb, on="MVT_ID_mvt", how="left").with_columns(
        to=s("MVT_TIME_UTC_mvt"), aobt=s("AOBT_3_flt"), lobt=s("LOBT_flt"), sched=s("SCHED_TIME_UTC_mvt"),
        coverage=pl.col("coverage").fill_null(-1), day_file=pl.col("n_cand").is_not_null())
    return d.with_columns(
        ap=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(AIRPORTS)}, return_dtype=pl.Int32),
        D=pl.col("to") - pl.col("sched"), has_nm=pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8),
        mvt_first=pl.col("to") - pl.col("t_first"), mvt_first_move=pl.col("to") - pl.col("t_first_move"),
        mvt_move=pl.col("to") - pl.col("t_move"), mvt_to=pl.col("to") - pl.col("t_to"),
        aobt_first=pl.col("aobt") - pl.col("t_first"), aobt_first_move=pl.col("aobt") - pl.col("t_first_move"),
        aobt_move=pl.col("aobt") - pl.col("t_move"), lobt_first_move=pl.col("lobt") - pl.col("t_first_move"),
    ).with_columns(base_vs_first_move=pl.col("base") - pl.col("mvt_first_move"), base_vs_first=pl.col("base") - pl.col("mvt_first"))


PARAMS30 = {**PARAMS, "num_leaves": 63}


NBF = ["nb_adsb_ap_n", "nb_adsb_ap_move", "nb_adsb_ap_first", "nb_adsb_rwy_n", "nb_adsb_rwy_move", "nb_adsb_rwy_first"]
FEATS_UNCOV = ["ap", "D", "p_nonreg", "base", "has_nm"] + NBF


def add_neighbours(d: pl.DataFrame) -> pl.DataFrame:
    """Neighbour ADS-B taxi times from covered departures (the row itself excluded)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from features import _window_stats
    d = d.with_columns(v_move=pl.when(pl.col("coverage") >= 2).then(pl.col("mvt_move")), v_first=pl.when(pl.col("coverage") >= 2).then(pl.col("mvt_first_move")))
    parts = []
    for key, keys, w in (("ap", ["ADEP_mvt"], 1800), ("rwy", ["ADEP_mvt", "RUNWAY_mvt"], 3600)):
        for _, g in d.select("MVT_ID_mvt", "to", "v_move", "v_first", *keys).drop_nulls(keys).group_by(keys):
            q = g["to"].to_numpy()
            cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
            for name, vcol in (("move", "v_move"), ("first", "v_first")):
                obs = g.drop_nulls(vcol).sort("to")
                n, mean, _ = _window_stats(obs["to"].to_numpy(), obs[vcol].to_numpy(), q, -w, w, g[vcol].to_numpy().astype(float))
                cols[f"nb_adsb_{key}_{name}"] = mean
                if name == "move":
                    cols[f"nb_adsb_{key}_n"] = n
            parts.append((key, pl.DataFrame(cols)))
    for key in ("ap", "rwy"):
        d = d.join(pl.concat([p for k, p in parts if k == key], how="diagonal_relaxed"), on="MVT_ID_mvt", how="left")
    return d


def fit_stage(tr: pl.DataFrame, feats=None, params=None, rounds=800):
    feats, params = feats or FEATS, params or PARAMS30
    t = tr.filter((pl.col("coverage") >= 2) & ((pl.col(Y) - pl.col("base")).abs() <= 3600))
    target = np.clip((t[Y] - t["base"]).to_numpy(), -1800, 1800)
    m = lgb.train(params, lgb.Dataset(t.select(feats).to_numpy(), target, categorical_feature=[0]), rounds)
    m.feats = feats
    return m


def correct(model, d: pl.DataFrame) -> np.ndarray:
    corr = model.predict(d.select(model.feats).to_numpy())
    use = d["coverage"].to_numpy() >= 2
    return np.clip(d["base"].to_numpy() + np.where(use, corr, 0.0), 0, None)


def oof_base(folder=os.environ.get("E30_BASE_DIR", "data/cv_preds_nomonth")):
    parts = []
    for p in sorted(glob.glob(f"{folder}/fold_*.npz")):
        z = np.load(p)
        parts.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "base": z["pred"], "p_nonreg": 1 - z["proba"][:, 0]}))
    return pl.concat(parts)


cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", Y]
mode = sys.argv[1]
if mode == "cv":
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
    d = add_neighbours(build(raw, oof_base(), load_adsb("data/adsblol/offblock2_2025-*.parquet"))).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
    covered = d.group_by("month").agg(files=pl.col("day_file").mean().round(2), cov2=(pl.col("coverage") >= 2).mean().round(2)).sort("month")
    print("share of rows with a surface file / coverage >= 2 by month:", covered.to_dicts())
    res, base_sse = [], 0.0
    for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
        tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
        m = fit_stage(tr)
        new, y = correct(m, te), te[Y].to_numpy()
        old = new.copy()  # control = E30 stage, covered rows only
        tu = tr.filter((pl.col("coverage") < 2) & ((pl.col(Y) - pl.col("base")).abs() <= 3600) & (pl.col("nb_adsb_ap_n") > 0))
        mu = lgb.train(PARAMS, lgb.Dataset(tu.select(FEATS_UNCOV).to_numpy(), np.clip((tu[Y] - tu["base"]).to_numpy(), -1800, 1800), categorical_feature=[0]), 400)
        unc = (te["coverage"].to_numpy() < 2) & (te["nb_adsb_ap_n"].fill_null(0).to_numpy() > 0)
        new = np.where(unc, np.clip(te["base"].to_numpy() + mu.predict(te.select(FEATS_UNCOV).to_numpy()), 0, None), new)
        dl = (new - y) ** 2 - (old - y) ** 2
        top3 = np.sort(dl)[:3].sum()
        per_ap = te.with_columns(dl=pl.Series(dl)).group_by("ADEP_mvt").agg(pl.col("dl").sum()).sort("ADEP_mvt")
        res.append((a, b, dl.sum(), top3))
        if os.environ.get("E35_SAVE_DIR"):  # corrected out-of-fold predictions of the E30 stage (15.09, default off)
            os.makedirs(os.environ["E35_SAVE_DIR"], exist_ok=True)
            np.savez(f"{os.environ['E35_SAVE_DIR']}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=new, y=y)
        if os.environ.get("E35_CTRL_DIR"):  # the control arm too (E60, 17.09): re-judging needs both arms on disk
            os.makedirs(os.environ["E35_CTRL_DIR"], exist_ok=True)
            np.savez(f"{os.environ['E35_CTRL_DIR']}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=old, y=y)
        print(f"fold {a}+{b}: rows {te.height} (coverage>=2 {int((te['coverage'] >= 2).sum())}), SSE change {dl.sum():+.3e}, best 3 rows {top3:+.3e}; "
              f"by airport {dict(zip(per_ap['ADEP_mvt'], np.round(per_ap['dl'].to_numpy() / 1e6)))} (1e6)", flush=True)
    tot = sum(r[2] for r in res)
    lower = sum(r[2] < 0 for r in res)
    ok = -tot / LB_SCALE >= 0.3 and lower >= 4
    print(f"DECISION E35 (vs E30 stage): {'ACCEPT' if ok else 'REJECT'} (lower in {lower}/6, total SSE change {tot:+.3e} = {tot / LB_SCALE:+.2f} s LB-equivalent RMSE; "
          f"without best 3 rows per fold {(tot - sum(r[3] for r in res)) / LB_SCALE:+.2f} s)")
elif mode == "apply":
    base_v, new_v = int(sys.argv[2]), int(sys.argv[3])
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
    m = fit_stage(build(raw, oof_base(), load_adsb("data/adsblol/offblock2_2025-*.parquet")))
    rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=[c for c in cols if c != Y]).filter(pl.col("PHASE_mvt") == "DEP")
    sub = pl.read_parquet(f"submissions/gentle-igloo_v{base_v}.parquet")
    parts = np.load(f"submissions/parts_v{base_v - 2}.npz") if os.path.exists(f"submissions/parts_v{base_v - 2}.npz") else None
    pn = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "p_nonreg": 1 - parts["proba"][:, 0]}) if parts is not None else None
    base = sub.rename({Y: "base"}).with_columns(pl.col("base").cast(pl.Float64))
    base = base.join(pn, on="MVT_ID_mvt", how="left") if pn is not None else base.with_columns(p_nonreg=pl.lit(0.0))
    d = build(rk.with_columns(pl.lit(None, dtype=pl.Int32).alias(Y)), base, load_adsb("data/adsblol/offblock2_2026-*.parquet"))
    new = correct(m, d)
    out = sub.join(pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "new": new}), on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce(pl.col("new").round(0).cast(sub.schema[Y]), pl.col(Y)).alias(Y)).select(sub.columns)
    dd = (out.sort("MVT_ID_mvt")[Y].cast(float) - sub.sort("MVT_ID_mvt")[Y].cast(float)).to_numpy()
    out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
    print(f"wrote v{new_v}: rows with coverage>=2 {int((d['coverage'] >= 2).sum())}, changed {(dd != 0).sum()}, RMS change {np.sqrt((dd ** 2).mean()):.2f} s")
