"""Apply E14 (accepted 15.09: SSE lower in 6/6 folds, 0.95 of the gain kept without top-3, full RMSE not worse 6/6) to ranking.

    FEATURES=data/features_final.parquet .venv/bin/python src/e14_apply.py <base submission> <parts version> <new>
The logistic model is fitted on all 2025 LIRF departures without NM at 0 < D < 4 h with out-of-month flight history;
ranking rows use the history of all 2025. Applied only where the flight has >= 3 history rows:
prediction = p * D + (1 - p) * regular (regular part from submissions/parts_v<parts>.npz). The TabPFN rows (D >= 4 h) are untouched.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.argv, (base, parts_v, new) = sys.argv[:1], sys.argv[1:4]

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

src = open("src/e14_flight_history.py").read().split("\nresults = []")[0]
g = {"__name__": "e14"}
exec(src, g)
f, history, design, Y = g["f"], g["history"], g["design"], g["Y"]
window = (pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400)

parts = [history(f.filter(pl.col("month") != m), f.filter(pl.col("month") == m)) for m in f["month"].unique().to_list()]
fit = pl.concat(parts).filter(window)
model = LogisticRegression(C=1.0, max_iter=2000).fit(design(fit), fit["sched"].to_numpy())

cols = ["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "local_hour", "day_nonm_rate"]
rank = pl.read_parquet(os.environ.get("FEATURES", "data/features_final.parquet"), columns=cols).filter(
    (pl.col("split") == "rank") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).filter(window)
rank = history(f, rank).filter(pl.col("n") >= 3)
z = np.load(f"submissions/parts_v{parts_v}.npz")
rank = rank.join(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "reg": z["reg"]}), on="MVT_ID_mvt", how="left")
assert rank["reg"].null_count() == 0
p = model.predict_proba(design(rank))[:, 1]
rank = rank.with_columns(new=pl.Series(p * rank["d_mvt_sched"].to_numpy() + (1 - p) * rank["reg"].to_numpy()))

sub = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
out = sub.join(rank.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left").with_columns(
    pl.coalesce(pl.col("new").round(0).cast(sub.schema["TAXITIME_SEC_mvt"]), pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt")).select(sub.columns)
changed = (out.sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"] != sub.sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"]).sum()
d = (out.sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].cast(float) - sub.sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].cast(float)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
print(f"wrote v{new}: rule rows {rank.height}, changed {changed}, RMS change {np.sqrt((d ** 2).mean()):.2f} s, mean p {p.mean():.3f}")
