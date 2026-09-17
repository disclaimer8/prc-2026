"""E44: the E36 stage plus how much of the surface track was received rather than repeated (17.09).

readsb marks a trace position stale (bit 0 of p[6]) when nothing has been heard for 20 s and the point simply
repeats the last known position. A stand held by fresh receptions and a stand held only by repeats are not the
same evidence, yet the off-block estimate treats them alike: a long repeat stretch can invent a stop that was
never observed. src/adsb_stale_features.py turns each departure's segment into stale_share, stale_first,
stale_at_move, stale_run_max, stale_before_to and fresh_before_move; this stage adds those six columns to E36
and changes nothing else.

Why this and not another feature set: of the five experiments with a known leaderboard outcome, the ones that
transferred (E25 -15.2, E30 -2.0, E36 -3.7, E46 -1.0) either calibrate a fixed transformation or read a
physical observation of the flight itself, while everything that added learned capacity over training-period
aggregates lost on the leaderboard (E32 +0.33, E38 +3.79, E50 +5.78). Staleness is an observation of the
flight, measured identically in both years.

    .venv/bin/python src/e44_adsb_staleness.py cv            6 folds against the E36 stage -> decision
    PARTS=... .venv/bin/python src/e44_adsb_staleness.py apply <base> <new>

Gate (REVIEW_BRIEF.md, E44): SSE over the rows the base model decides lower than the E36 stage by >= 0.3 s
LB-equivalent and lower in >= 4 of 6 folds, with the strict, full and y <= 7200 numbers reported alongside.
Requires every day of the 2025 training months and both ranking months to carry the flag; the loader fails
loudly if a day is missing rather than silently training on a subset.
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
FEATS30 = FEATS + SF
GS = ["gs_stand_known", "gs_d_first", "gs_d_firstmove", "gs_d_lastmove", "gs_d_lastmove_rwy", "gs_dwell_stand", "gs_seen_at_stand", "gs_mvt_leave", "gs_aobt_leave"]
FEATS36 = FEATS30 + GS
# what the receiver actually heard, as opposed to what readsb repeated: src/adsb_stale_features.py
ST = ["stale_share", "stale_first", "stale_at_move", "stale_run_max", "stale_before_to", "fresh_before_move"]
FEATS = FEATS36 + ST
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, deterministic=True, force_row_wise=True, num_threads=8)


def load_adsb(pattern: str) -> pl.DataFrame:
    files = sorted(glob.glob(pattern))
    o2 = pl.concat([pl.read_parquet(f) for f in files])
    o3 = pl.concat([pl.read_parquet(f.replace("offblock2_", "offblock3_")) for f in files if os.path.exists(f.replace("offblock2_", "offblock3_"))])
    o5 = pl.concat([pl.read_parquet(f.replace("offblock2_", "offblock5_")) for f in files if os.path.exists(f.replace("offblock2_", "offblock5_"))])
    stale_files = [f.replace("offblock2_", "stale_") for f in files]
    missing = [f for f in stale_files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(files)} days have no staleness features yet "
                         f"(first: {os.path.basename(missing[0])}) — run scripts/stale_features_loop.sh; "
                         "training on a subset would compare two different day sets")
    st = pl.concat([pl.read_parquet(f) for f in stale_files])
    return (o2.join(o3, on="MVT_ID_mvt", how="left").join(o5, on="MVT_ID_mvt", how="left")
            .join(st, on="MVT_ID_mvt", how="left"))


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
    ).with_columns(base_vs_first_move=pl.col("base") - pl.col("mvt_first_move"), base_vs_first=pl.col("base") - pl.col("mvt_first"),
                   gs_mvt_leave=pl.col("to") - pl.col("gs_t_leave"), gs_aobt_leave=pl.col("aobt") - pl.col("gs_t_leave"))


PARAMS30 = {**PARAMS, "num_leaves": 63}


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


def oof_base(folder=os.environ.get("E36_BASE_DIR", "data/cv_preds_nomonth")):
    parts = []
    for p in sorted(glob.glob(f"{folder}/fold_*.npz")):
        z = np.load(p)
        parts.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "base": z["pred"], "p_nonreg": 1 - z["proba"][:, 0]}))
    return pl.concat(parts)


cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", Y]
mode = sys.argv[1]
if mode == "cv":
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
    d = build(raw, oof_base(), load_adsb("data/adsblol/offblock2_2025-*.parquet")).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
    covered = d.group_by("month").agg(files=pl.col("day_file").mean().round(2), cov2=(pl.col("coverage") >= 2).mean().round(2)).sort("month")
    print("share of rows with a surface file / coverage >= 2 by month:", covered.to_dicts())
    res, base_sse = [], 0.0
    for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
        tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
        m = fit_stage(tr)
        m30 = fit_stage(tr, FEATS36)   # control: the E36 stage without the reception features
        new, y, old = correct(m, te), te[Y].to_numpy(), correct(m30, te)
        dl = (new - y) ** 2 - (old - y) ** 2
        top3 = np.sort(dl)[:3].sum()
        per_ap = te.with_columns(dl=pl.Series(dl)).group_by("ADEP_mvt").agg(pl.col("dl").sum()).sort("ADEP_mvt")
        res.append((a, b, dl.sum(), top3))
        if os.environ.get("E36_SAVE_DIR"):  # corrected out-of-fold predictions of the E30 stage (15.09, default off)
            os.makedirs(os.environ["E36_SAVE_DIR"], exist_ok=True)
            np.savez(f"{os.environ['E36_SAVE_DIR']}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=new, y=y)
        print(f"fold {a}+{b}: rows {te.height} (coverage>=2 {int((te['coverage'] >= 2).sum())}), SSE change {dl.sum():+.3e}, best 3 rows {top3:+.3e}; "
              f"by airport {dict(zip(per_ap['ADEP_mvt'], np.round(per_ap['dl'].to_numpy() / 1e6)))} (1e6)", flush=True)
    tot = sum(r[2] for r in res)
    lower = sum(r[2] < 0 for r in res)
    ok = -tot / LB_SCALE >= 0.3 and lower >= 4
    print(f"DECISION E44 (staleness vs the E36 stage): {'ACCEPT' if ok else 'REJECT'} (lower in {lower}/6, total SSE change {tot:+.3e} = {tot / LB_SCALE:+.2f} s LB-equivalent RMSE; "
          f"without best 3 rows per fold {(tot - sum(r[3] for r in res)) / LB_SCALE:+.2f} s)")
elif mode == "apply":
    base_v, new_v = int(sys.argv[2]), int(sys.argv[3])
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
    m = fit_stage(build(raw, oof_base(), load_adsb("data/adsblol/offblock2_2025-*.parquet")))
    rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=[c for c in cols if c != Y]).filter(pl.col("PHASE_mvt") == "DEP")
    sub = pl.read_parquet(f"submissions/gentle-igloo_v{base_v}.parquet")
    # regime probabilities of the base: explicit file (PARTS=submissions/parts_vN.npz, the blend step of the base chain); no silent zero
    parts_path = os.environ.get("PARTS", f"submissions/parts_v{base_v - 2}.npz")
    if not os.path.exists(parts_path):
        sys.exit(f"missing {parts_path}: set PARTS to the parts file of the blend that produced v{base_v}")
    parts = np.load(parts_path)
    print("regime probabilities from", parts_path)
    pn = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "p_nonreg": 1 - parts["proba"][:, 0]})
    base = sub.rename({Y: "base"}).with_columns(pl.col("base").cast(pl.Float64))
    base = base.join(pn, on="MVT_ID_mvt", how="left") if pn is not None else base.with_columns(p_nonreg=pl.lit(0.0))
    d = build(rk.with_columns(pl.lit(None, dtype=pl.Int32).alias(Y)), base, load_adsb("data/adsblol/offblock2_2026-*.parquet"))
    new = correct(m, d)
    out = sub.join(pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "new": new}), on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce(pl.col("new").round(0).cast(sub.schema[Y]), pl.col(Y)).alias(Y)).select(sub.columns)
    dd = (out.sort("MVT_ID_mvt")[Y].cast(float) - sub.sort("MVT_ID_mvt")[Y].cast(float)).to_numpy()
    out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
    print(f"wrote v{new_v}: rows with coverage>=2 {int((d['coverage'] >= 2).sum())}, changed {(dd != 0).sum()}, RMS change {np.sqrt((dd ** 2).mean()):.2f} s")
