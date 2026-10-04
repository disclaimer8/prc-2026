"""E185 on the board: the movement-only member mixed into the regular candidate on the ranking rows without an NM record outside
LIRF, on a parts file (the same pattern as src/e167_nm_window_rule.py). Every other row keeps its value bit for bit.

    E185_W=0.xx E185_RANK=data/e185_motion_rank.npz .venv/bin/python src/e185_apply.py <parts version> <new version>

Goes right after src/blend.py (before E167, which reads the parts it writes): reg' = (1 - w) reg + w motion on the subset, values[:, 0]
= reg', values[:, -1] = reg' + 1 day, prediction = clip(lobt_clip(sum(proba * values)), 0) with the stored probabilities, exactly as
blend.py and src/e185_subset_grid_cv.py. Writes parts_v<new>.npz and gentle-igloo_v<new>.parquet.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

src_v, new_v = int(sys.argv[1]), int(sys.argv[2])
W = float(os.environ["E185_W"])
RANK = os.environ.get("E185_RANK", "data/e185_motion_rank.npz")
FEAT = os.environ.get("E185_SUBSET_FEATURES", "data/features_e106.parquet")
Y = "TAXITIME_SEC_mvt"
SUBS = os.environ.get("E185_SUBDIR", "submissions")  # a scratch directory for dry runs

z = dict(np.load(f"{SUBS}/parts_v{src_v}.npz"))
ids = z["mvt_id"]
f = pl.read_parquet(FEAT, columns=["MVT_ID_mvt", "split", "ADEP_mvt", "LOBT_flt", "d_mvt_lobt"]).filter(pl.col("split") == "rank")
d = pl.DataFrame({"MVT_ID_mvt": ids}).join(f.drop("split"), on="MVT_ID_mvt", how="left", maintain_order="left")
assert d["ADEP_mvt"].null_count() == 0, "parts rows missing from the feature table"
S = (d["LOBT_flt"].is_null() & (d["ADEP_mvt"] != "LIRF")).to_numpy()
m = np.load(RANK)
mm = dict(zip(m["mvt_id"], m["reg"]))
motion = np.array([mm[k] for k in ids[S]])
reg = z["reg"].astype(float).copy()
values = z["values"].astype(float).copy()
reg[S] = (1 - W) * reg[S] + W * motion
values[S, 0], values[S, -1] = reg[S], reg[S] + T.DAY
pred = np.clip(T.lobt_clip(np.sum(z["proba"].astype(float) * np.nan_to_num(values), axis=1), d), 0, None)
old = z["pred"].astype(float)
assert np.allclose(pred[~S], old[~S], atol=1e-6), "rows outside the subset moved: the parts file is not a blend.py mixture"
print(f"subset rows {int(S.sum())} (by airport {dict(sorted(d.filter(pl.Series(S)).group_by('ADEP_mvt').len().rows()))}); w {W}; "
      f"corr(motion, reg) {np.corrcoef(motion, z['reg'][S])[0, 1]:.4f}, mean(motion - reg) {np.mean(motion - z['reg'][S]):+.1f} s; "
      f"prediction change on the subset: mean {np.mean(pred[S] - old[S]):+.1f} s, RMS {np.sqrt(np.mean((pred[S] - old[S]) ** 2)):.1f} s")
z.update(reg=reg, values=values.astype(z["values"].dtype), pred=np.where(S, pred, old))
np.savez(f"{SUBS}/parts_v{new_v}.npz", **z)
sub = pl.read_parquet(f"{SUBS}/gentle-igloo_v{src_v}.parquet")
upd = pl.DataFrame({"MVT_ID_mvt": ids[S], "new": np.round(pred[S])})
out = sub.join(upd, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("new", Y).cast(sub.schema[Y]).alias(Y)).select(sub.columns)
assert out.height == sub.height and out[Y].null_count() == 0
out.write_parquet(f"{SUBS}/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v} and parts_v{new_v}")
