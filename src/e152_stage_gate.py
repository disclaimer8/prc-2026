"""E152 (journal 134): the ADS-B stages must not move rows where the base is sure of a filed-time regime.

    .venv/bin/python src/e152_stage_gate.py <pre-stage version> <staged version> <parts version> <new version>

Where the base mixture (submissions/parts_v<parts>.npz) gives max P(sched | iobt | lobt | eobt) >= GATE (0.95), the label
is a filed time, not a physical taxi, so a correction learned from the surface track cannot apply: keep the pre-stage
prediction (the E14 output, before E141/E70 and E64). Everywhere else the staged file is kept as is.
CV (base OOF cv_preds_nomonth, stages cv_preds_e141 / cv_preds_e150_keep): gate at 0.95 lower in 6/6 folds for both,
-0.003 ... -0.034 s per fold; at 0.9 5/6; at 0.99 almost no rows. Found tracing ISR342 (2026): base 8074 at D = 8099,
p_sched 0.996, and E141 / E64 lifted it to 8631-10856.
"""
import os
import sys

import numpy as np
import polars as pl

pre, staged, parts_v, new = (int(x) for x in sys.argv[1:5])
GATE = float(os.environ.get("E152_GATE", "0.95"))
Y = "TAXITIME_SEC_mvt"
z = np.load(f"submissions/parts_v{parts_v}.npz")
conf = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "conf": z["proba"][:, 1:5].max(axis=1)})
a = pl.read_parquet(f"submissions/gentle-igloo_v{pre}.parquet")
b = pl.read_parquet(f"submissions/gentle-igloo_v{staged}.parquet")
j = b.join(a.rename({Y: "pre"}), on="MVT_ID_mvt", how="left").join(conf, on="MVT_ID_mvt", how="left")
assert j.height == b.height and j["pre"].null_count() == 0 and j["conf"].null_count() == 0
gate = pl.col("conf") >= GATE
out = j.with_columns(pl.when(gate).then(pl.col("pre")).otherwise(pl.col(Y)).cast(b.schema[Y]).alias(Y))
d = (out[Y].cast(float) - b[Y].cast(float)).to_numpy()
print(f"E152 gate {GATE}: {int(j.filter(gate).height)} confident rows, {int((d != 0).sum())} moved back, RMS change {np.sqrt((d ** 2).mean()):.3f} s, max |d| {np.abs(d).max():.0f} s")
out.select(b.columns).write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
print(f"wrote v{new} = v{staged} with the pre-stage v{pre} on confident filed-time rows")
