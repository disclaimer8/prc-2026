"""E158 board table (journal 169): data/features_e158.parquet with the ranking rows filled.
stk_reg (ranking) = the E116 LightGBM regressor of the production base (submissions/parts_v240.npz "reg", the same model type as the
OOF used for 2025), stk_reg2 (ranking) = the E130 raw regressor (submissions/e130_raw_rank.npz, 5 seeds). Writes
data/features_e158_full.parquet.
"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

f = pl.read_parquet("data/features_e158.parquet")
z1, z2 = np.load("submissions/parts_v240.npz"), np.load("submissions/e130_raw_rank.npz")
r = pl.DataFrame({"MVT_ID_mvt": z1["mvt_id"], "r1": z1["reg"].astype(float)}).join(
    pl.DataFrame({"MVT_ID_mvt": z2["mvt_id"], "r2": z2["reg"].astype(float)}), on="MVT_ID_mvt", how="full", coalesce=True)
f = f.join(r, on="MVT_ID_mvt", how="left").with_columns(
    stk_reg=pl.when(pl.col("split") == "rank").then(pl.col("r1")).otherwise(pl.col("stk_reg")),
    stk_reg2=pl.when(pl.col("split") == "rank").then(pl.col("r2")).otherwise(pl.col("stk_reg2"))).drop("r1", "r2")
f = f.with_columns([(pl.col(c) - pl.col("stk_reg")).alias(f"stk_{c[6:]}_minus_reg") for c in ["d_mvt_sched", "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"]])
rk = f.filter(pl.col("split") == "rank")
print("rank rows", rk.height, "stk_reg null", rk["stk_reg"].null_count(), "stk_reg2 null", rk["stk_reg2"].null_count(),
      "corr", round(float(np.corrcoef(rk["stk_reg"].to_numpy(), rk["stk_reg2"].to_numpy())[0, 1]), 4))
f.write_parquet("data/features_e158_full.parquet")
print("wrote data/features_e158_full.parquet", f.shape)
