"""Validation predictions equivalent to v13: v6-lineage mixture (data/val_parts_v6_baseline.npz) with TabPFN-2 regime
probabilities for LIRF departures without NM at 4-30 h delay, TabPFN fitted on the 10 non-validation months.
Writes data/val_pred_v13eq.parquet for residual diagnostics."""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import lirf_table as L
from lirf_tabpfn import FEATS, prepare

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "d_mvt_sched",
        "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y]
f = pl.read_parquet("data/features_v9.parquet", columns=cols).filter(pl.col("split") == "train")
fit = prepare(f.filter(~pl.col("month").is_in([1, 7]))).with_columns(
    cls=pl.when((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).then(0).when(pl.col(Y).is_between(86400, 93600)).then(1).otherwise(2)
)
val = prepare(f.filter(pl.col("month").is_in([1, 7]))).filter(pl.col("b").is_in(L.OVERRIDE))
clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
clf.fit(fit.select(FEATS).to_numpy(), fit["cls"].to_numpy())
raw = clf.predict_proba(val.select(FEATS).to_numpy())
p = np.zeros((val.height, 3))
for j, c in enumerate(clf.classes_):
    p[:, int(c)] = raw[:, j]

parts = np.load("data/val_parts_v6_baseline.npz")
base = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "y": parts["y"], "pred_v6": parts["pred"], "reg": parts["reg"],
                     **{f"p_{k}": parts["proba"][:, i] for i, k in enumerate(["regular", "sched", "iobt", "lobt", "eobt", "day"])}})
reg = dict(zip(parts["mvt_id"], parts["reg"]))
over = pl.DataFrame({"MVT_ID_mvt": val["MVT_ID_mvt"],
                     "pred_tabpfn": p[:, 0] * val["d_mvt_sched"].to_numpy() + p[:, 1] * (86400 + 951.0) + p[:, 2] * np.array([reg[m] for m in val["MVT_ID_mvt"].to_numpy()])})
out = base.join(over, on="MVT_ID_mvt", how="left").with_columns(pred=pl.coalesce("pred_tabpfn", "pred_v6"), tabpfn_row=pl.col("pred_tabpfn").is_not_null())
out.write_parquet("data/val_pred_v13eq.parquet")
rm = lambda c: float(np.sqrt(((out[c] - out["y"]) ** 2).mean()))
print(f"validation RMSE: v6 {rm('pred_v6'):.2f} -> v13-equivalent {rm('pred'):.2f}; TabPFN rows {int(out['tabpfn_row'].sum())}")
