"""E153 (journal 141): E14 with the airline prefix (U2 of the rejection audit, journal 137; Discord 28.09 "the echo of the planned
time in the off-block is mostly a carrier thing when there's no NM match").

    FEATURES=data/features_e106.parquet .venv/bin/python src/e153_apply.py <base submission> <parts version> <new>

Same as src/e14_apply.py (logistic "block = schedule" for LIRF departures without NM at 0 < D < 4 h, fitted on all 2025 with
out-of-month history, prediction p * D + (1 - p) * regular), plus two design columns: the out-of-month logit rate of
block = schedule for the airline prefix (E113 lookup, data/features_e113.parquet) and log(1 + prefix rows); applied where the
flight number has >= 3 history rows OR the prefix has >= 20. LOMO probe (src/u2_e14_prefix_probe.py): vs E14 -1.33 s summed over
6 folds (lower 4/6), without the 3 largest changes per fold -1.53 s (4/6); 2026 rows touched 43 -> 256.
E153_PFX=0 reproduces src/e14_apply.py exactly (checked before use).
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
base, parts_v, new = sys.argv[1:4]

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

PFX = os.environ.get("E153_PFX", "1") == "1"
Y = "TAXITIME_SEC_mvt"
FEATURES = os.environ.get("FEATURES", "data/features_final.parquet")
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "local_hour", "day_nonm_rate", Y]
allf = pl.read_parquet(FEATURES, columns=cols).filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
allf = allf.join(pl.read_parquet(os.environ.get("PFX_FEATURES", "data/features_e113.parquet"), columns=["MVT_ID_mvt", "airline_prefix"]),
                 on="MVT_ID_mvt", how="left").with_columns(pfx=pl.col("airline_prefix").fill_null("?"))
f = allf.filter(pl.col("split") == "train").with_columns(
    sched=((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).cast(pl.Int8), delayed=(pl.col("d_mvt_sched") > 1800))
window = (pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400)


def history(src, rows):
    h = src.group_by("FLIGHT_mvt").agg(n=pl.len(), k=pl.col("sched").sum(), nd=pl.col("delayed").sum(), kd=(pl.col("sched") * pl.col("delayed")).sum())
    g = src.group_by("pfx").agg(np_=pl.len(), kp=pl.col("sched").sum())
    return (rows.join(h, on="FLIGHT_mvt", how="left").join(g, on="pfx", how="left").with_columns(pl.col("n", "k", "nd", "kd", "np_", "kp").fill_null(0))
            .with_columns(ff=(pl.col("k") + 0.5) / (pl.col("n") + 1), ffd=(pl.col("kd") + 0.5) / (pl.col("nd") + 1),
                          fp=(pl.col("kp") + 0.5) / (pl.col("np_") + 1)))


def design(d):
    lg = lambda c: np.log(d[c].to_numpy() / (1 - d[c].to_numpy()))
    D = d["d_mvt_sched"].to_numpy()
    c = [lg("ff"), lg("ffd"), np.log1p(d["n"].to_numpy()), d["day_nonm_rate"].to_numpy(), np.log(np.clip(D, 60, None)), (D > 7200).astype(float), d["local_hour"].to_numpy()]
    if PFX:
        c += [lg("fp"), np.log1p(d["np_"].to_numpy())]
    return np.column_stack(c)


fit = pl.concat([history(f.filter(pl.col("month") != m), f.filter(pl.col("month") == m)) for m in f["month"].unique().to_list()]).filter(window)
model = LogisticRegression(C=1.0, max_iter=3000 if PFX else 2000).fit(design(fit), fit["sched"].to_numpy())
rank = history(f, allf.filter(pl.col("split") == "rank").filter(window))
rank = rank.filter((pl.col("n") >= 3) | (pl.col("np_") >= 20)) if PFX else rank.filter(pl.col("n") >= 3)
z = np.load(f"submissions/parts_v{parts_v}.npz")
rank = rank.join(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "reg": z["reg"]}), on="MVT_ID_mvt", how="left")
assert rank["reg"].null_count() == 0
p = model.predict_proba(design(rank))[:, 1]
rank = rank.with_columns(new=pl.Series(p * rank["d_mvt_sched"].to_numpy() + (1 - p) * rank["reg"].to_numpy()))
sub = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
out = sub.join(rank.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left").with_columns(
    pl.coalesce(pl.col("new").round(0).cast(sub.schema[Y]), pl.col(Y)).alias(Y)).select(sub.columns)
changed = (out.sort("MVT_ID_mvt")[Y] != sub.sort("MVT_ID_mvt")[Y]).sum()
d = (out.sort("MVT_ID_mvt")[Y].cast(float) - sub.sort("MVT_ID_mvt")[Y].cast(float)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
print(f"wrote v{new}: E153 prefix {PFX}, rule rows {rank.height}, changed {changed}, RMS change {np.sqrt((d ** 2).mean()):.2f} s, mean p {p.mean():.3f}")
