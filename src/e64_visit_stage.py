"""E64: a stage for the departures that have no ground segment at all (18.09).

Population: coverage < 2 - 824369 rows holding 61.83 s of the 108.66 s decided-row budget, which no stage has
ever touched. The stage of E30/E36 corrects only coverage >= 2; E35/E60 tried to describe these rows through
OTHER aircraft's taxi times and failed (-0.09 s against a -0.3 gate). The difference here is that the aircraft
being observed is the flight's own: src/adsb_visit.py identifies its icao24 by call sign and reads the airborne
points we already download, so the features describe this flight, not its neighbours. That is the same step that
took E31 (someone else's airframe, matched by take-off time) to E25 (its own track), which was worth -15.2 s.

Measured before registering, 5 days of 2025, coverage <= 1, n = 4615: the turnaround is useless as a bound
(median 6625 s against a median taxi-out of 964 s, binds in 0.2% of rows, correlation -0.02 with the residual),
but the time between the airframe reappearing and take-off correlates +0.11 - four times anything else measured
on 17-18.09 - and mean |residual| rises 117 -> 143 -> 155 -> 171 s across turnaround quartiles. The airframe is
identified for 41% of this population, so roughly 25 s of budget is in reach.

Control is the chain as it stands: on these rows nothing corrects the base at all. Only additions are made -
no existing value is substituted and no coverage class is rewritten, per the rule of E61 -> E62 -> E63.

Gate: SSE over the rows the base model decides lower by >= 0.3 s LB-equivalent and lower in >= 4 of 6 folds; the
same on y <= 7200; and the gain must survive dropping the three best and three worst rows of each fold.
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
CTRL_DIR, CAND_DIR = "data/cv_preds_e64_ctrl", "data/cv_preds_e64"
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
    t = tr.filter(usable & ((pl.col(Y) - pl.col("base")).abs() <= 3600))
    m = lgb.train(PARAMS, lgb.Dataset(t.select(FEATS).to_numpy(),
                                      np.clip((t[Y] - t["base"]).to_numpy(), -1800, 1800), categorical_feature=[0]), ROUNDS)
    use = te.select(usable).to_series().to_numpy()
    corr = m.predict(te.select(FEATS).to_numpy())
    ctrl = te["base"].to_numpy()
    cand = np.clip(ctrl + np.where(use, corr, 0.0), 0, None)
    y = te[Y].to_numpy(); mvt = te["MVT_ID_mvt"].to_numpy()
    np.savez(f"{CTRL_DIR}/fold_{a}_{b}.npz", mvt_id=mvt, pred=ctrl, y=y)
    np.savez(f"{CAND_DIR}/fold_{a}_{b}.npz", mvt_id=mvt, pred=cand, y=y)
    print(f"fold {a}+{b}: corrected {int(use.sum())} of {te.height} rows, "
          f"RMSE {np.sqrt(np.mean((ctrl-y)**2)):.2f} -> {np.sqrt(np.mean((cand-y)**2)):.2f}", flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E64: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent)")
