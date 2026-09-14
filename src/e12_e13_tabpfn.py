"""E12 / E13: TabPFN-2 regime probabilities for departures without NM, against the current regime classifier.

    TABPFN_ALLOW_CPU_LARGE_DATASET=1 uv run --with tabpfn python src/e12_e13_tabpfn.py E12|E13

Validation: January + July 2025 rows; TabPFN is fitted on the other 10 months; the current model's predictions and
regular-regime values come from data/val_parts_v6_baseline.npz (same regime classifier as v7/v13).
E12: LIRF without NM, delay from schedule < 4 h (disjoint from E11). E13: the other nine airports without NM.
Pre-registered gate: SSE >= 5% below the current model, still below without the top-3 rows, better in both months.
"""
import os
import re
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

Y = "TAXITIME_SEC_mvt"
MODE = sys.argv[1]
APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "d_mvt_sched",
        "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y]
f = pl.read_parquet("data/features_v9.parquet", columns=cols).filter((pl.col("split") == "train") & pl.col("FLIGHT_ID_mvt").is_null())
if MODE in ("E12", "E12b"):
    f = f.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("d_mvt_sched") < 14400))
else:
    f = f.filter(pl.col("ADEP_mvt") != "LIRF")
f = f.with_columns(
    cls=pl.when((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).then(0).when(pl.col(Y).is_between(86400, 93600)).then(1).otherwise(2),
    ap=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(APS)}, return_dtype=pl.Float64),
    stand_num=pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    rwy_head=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64),
    fn_num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    fn_len=pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float64),
    fn_garbled=pl.col("FLIGHT_mvt").map_elements(lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64),
)
feats = ["d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", "stand_num", "rwy_head", "fn_num", "fn_len", "fn_garbled"]
if MODE == "E13":
    feats = ["ap"] + feats
fit, val = f.filter(~pl.col("month").is_in([1, 7])), f.filter(pl.col("month").is_in([1, 7]))
print(MODE, "fit rows", fit.height, "classes", np.bincount(fit["cls"].to_numpy(), minlength=3).tolist(), "val rows", val.height)
if fit.height > 8000:  # TabPFN-2 context limit; keep every non-regular row, subsample regular ones
    rare = fit.filter(pl.col("cls") != 2)
    fit = pl.concat([rare, fit.filter(pl.col("cls") == 2).sample(8000 - rare.height, seed=0)])
    print("subsampled fit rows", fit.height)

parts = np.load("data/val_parts_v6_baseline.npz")
base = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "pred": parts["pred"], "reg": parts["reg"]})
val = val.join(base, on="MVT_ID_mvt", how="inner")
day_taxi = 951.0

clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
clf.fit(fit.select(feats).to_numpy(), fit["cls"].to_numpy())
raw = clf.predict_proba(val.select(feats).to_numpy())
p = np.zeros((val.height, 3))
for j, c in enumerate(clf.classes_):
    p[:, int(c)] = raw[:, j]
d, y, r = val["d_mvt_sched"].to_numpy(), val[Y].to_numpy(), val["reg"].to_numpy()
print("p_day on validation: mean", round(float(p[:, 1].mean()), 4), "max", round(float(p[:, 1].max()), 4), "rows with p_day>0.05:", int((p[:, 1] > 0.05).sum()))
if MODE.endswith("b"):  # E12b: the one-day shift was only observed at delays >= 14 h in 2025
    short = d < 43200
    p[short, 2] += p[short, 1]
    p[short, 1] = 0.0
new = p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * r
old = val["pred"].to_numpy()
se_old, se_new = (old - y) ** 2, (new - y) ** 2
top3 = np.argsort(-se_old)[:3]
mask = np.ones(len(y), bool)
mask[top3] = False
months = val["month"].to_numpy()
by_month = {m: (se_old[months == m].sum(), se_new[months == m].sum()) for m in (1, 7)}
gain = 100 * (se_new.sum() / se_old.sum() - 1)
gain_wo = 100 * (se_new[mask].sum() / se_old[mask].sum() - 1)
both = all(b < a for a, b in by_month.values())
print(f"{MODE}: SSE current={se_old.sum():.3e} tabpfn={se_new.sum():.3e} ({gain:+.1f}%), without top-3 ({gain_wo:+.1f}%), months {by_month}")
n_val_total = parts["y"].shape[0]
print(f"full-validation RMSE effect if applied: {np.sqrt(((parts['pred'] - parts['y']) ** 2).sum() / n_val_total):.2f} -> "
      f"{np.sqrt((((parts['pred'] - parts['y']) ** 2).sum() - se_old.sum() + se_new.sum()) / n_val_total):.2f}")
print("DECISION:", "ADOPT" if gain <= -5 and gain_wo < 0 and both else "REJECT")
