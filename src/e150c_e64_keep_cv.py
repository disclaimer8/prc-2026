"""E150c (28.09, journal 128): E64 x5 seeds trained also on |y - base| > 3600 rows vs the filtered E64 x5.
Based on E143 (27.09, journal 107): the E64 visit stage averaged over 5 LightGBM seeds (42, 1, 2, 3, 4) against the accepted
one-seed E64, on the E64 CV (same base, rows, features, clips, rounds). Capacity-free, like E142 on the E70 stage
(-0.052 s after E14, 6 of 6). Criterion: lower in >= 4 of 6 folds and not carried by a few rows; the board sizes it.
Prediction: -0.01 ... -0.05 s.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_visit import NAMES  # noqa: E402
from fold_metrics import compare  # noqa: E402

Y = "TAXITIME_SEC_mvt"
CTRL_DIR, CAND_DIR = "data/cv_preds_e150c_e64_filter", "data/cv_preds_e150c_e64_keep"
SEEDS = [42, 1, 2, 3, 4]
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FEATS = ["ap", "coverage", "D", "p_nonreg", "base", "has_nm"] + NAMES
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=42, deterministic=True,
              force_row_wise=True, num_threads=8)
ROUNDS = 400

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", Y]
raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
base = pl.concat([pl.DataFrame({"MVT_ID_mvt": np.load(p)["mvt_id"], "base": np.load(p)["pred"],
                                "p_nonreg": 1 - np.load(p)["proba"][:, 0]})
                  for p in sorted(glob.glob("data/cv_preds_nomonth/fold_*.npz"))])
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol/offblock2_2025-*.parquet"))])
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])

d = (raw.join(base, on="MVT_ID_mvt", how="inner").join(cov, on="MVT_ID_mvt", how="left")
        .join(vis, on="MVT_ID_mvt", how="left")
        .with_columns(coverage=pl.col("coverage").fill_null(-1),
                      month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                      ap=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(AIRPORTS)}, return_dtype=pl.Int32),
                      has_nm=pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8),
                      D=(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s") - pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s")).cast(pl.Float64)))
usable = ((pl.col("coverage") < 2) & (pl.col("vi_has_prev") == 1)).fill_null(False)
print(f"rows {d.height}; coverage < 2: {d.filter(pl.col('coverage') < 2).height}; "
      f"of them with an identified airframe and a previous visit: {d.filter(usable).height}", flush=True)

for x in (CTRL_DIR, CAND_DIR):
    os.makedirs(x, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    Xte = te.select(FEATS).to_numpy()
    def fit(t):
        tgt = np.clip((t[Y] - t["base"]).to_numpy(), -1800, 1800)
        return np.mean([lgb.train({**PARAMS, "seed": sd}, lgb.Dataset(t.select(FEATS).to_numpy(), tgt, categorical_feature=[0]), ROUNDS).predict(Xte) for sd in SEEDS], axis=0)
    use = te.select(usable).to_series().to_numpy()
    b0 = te["base"].to_numpy()
    ctrl = np.clip(b0 + np.where(use, fit(tr.filter(usable & ((pl.col(Y) - pl.col("base")).abs() <= 3600))), 0.0), 0, None)
    cand = np.clip(b0 + np.where(use, fit(tr.filter(usable)), 0.0), 0, None)
    y = te[Y].to_numpy(); mvt = te["MVT_ID_mvt"].to_numpy()
    np.savez(f"{CTRL_DIR}/fold_{a}_{b}.npz", mvt_id=mvt, pred=ctrl, y=y)
    np.savez(f"{CAND_DIR}/fold_{a}_{b}.npz", mvt_id=mvt, pred=cand, y=y)
    print(f"fold {a}+{b}: corrected {int(use.sum())} of {te.height} rows, "
          f"RMSE {np.sqrt(np.mean((ctrl-y)**2)):.2f} -> {np.sqrt(np.mean((cand-y)**2)):.2f}", flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and dec < 0 and lower >= 4 and small < 0 and trim < 0  # magnitude is left to the board (REVIEW_BRIEF 21bv)
print(f"DECISION E150c (E64 x5 keep-all vs E64 x5 with the filter): {'TO BOARD' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
