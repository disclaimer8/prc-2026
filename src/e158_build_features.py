"""E158 (journal 169): classifier-only stacking inputs. data/features_e106.parquet + stk_* columns for the 2025 rows:
stk_reg = production OOF regressor (cv_preds_e116prod "reg"), stk_reg2 = E130 raw-target OOF (cv_e130raw "reg2"), and each filed
candidate minus stk_reg. Used with CLF_ONLY_PREFIX=stk_ so only the regime classifier sees them (src/train.py).
Ranking rows stay null here (CV only); a board build would fill them with the production regressor and src/e130_rank.py.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

a = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(p))["mvt_id"], "stk_reg": z["reg"].astype(float)}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))])
b = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(p))["mvt_id"], "stk_reg2": z["reg2"].astype(float)}) for p in sorted(glob.glob("data/cv_e130raw/fold_*.npz"))])
f = pl.read_parquet("data/features_e106.parquet")
f = f.join(a, on="MVT_ID_mvt", how="left").join(b, on="MVT_ID_mvt", how="left").with_columns(
    [(pl.col(c) - pl.col("stk_reg")).alias(f"stk_{c[6:]}_minus_reg") for c in ["d_mvt_sched", "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"]])
tr = f.filter(pl.col("split") == "train")
print("train rows", tr.height, "stk_reg null", tr["stk_reg"].null_count(), "stk_reg2 null", tr["stk_reg2"].null_count())
f.write_parquet("data/features_e158.parquet")
print("wrote data/features_e158.parquet", f.shape)
