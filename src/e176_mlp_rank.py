"""E176 (journal 222-223): the E100 MLP regular-regime regressor as an E130 member on the ranking set.

    env -u FEATURES -u DROP_PREFIX NN_SEEDS=3 .venv/bin/python src/e176_mlp_rank.py
-> submissions/e130_mlp_rank.npz (mvt_id, reg), the format of src/e130_rank.py, read by blend.py E130_MEMBERS.
Same configuration as the OOF arm that passed as the 6th member (data/cv_preds_e100_nn: features_e77, DROP_PREFIX
day_,tq_,wxr_,month, PROXY_SCALE=1, fold target encodings, residual to the calibrated NM proxy), fitted on all of 2025.
NN_SEEDS=3 averages three networks (E101: three seeds = one seed in CV, -0.646 vs -0.657), which only removes draw noise.
E181a (journal 244-246): the same network on the final table,
    env -u DROP_PREFIX FEATURES=data/features_fin1b.parquet E176_OUT=submissions/e130_mlpa_rank.npz NN_SEEDS=3 ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import e100_nn_partner as N  # noqa: E402  (sets the E100 FEATURES / DROP_PREFIX / PROXY_SCALE defaults)
import train as T  # noqa: E402

assert os.environ["DROP_PREFIX"] == "day_,tq_,wxr_,month", os.environ["DROP_PREFIX"]
assert os.environ["FEATURES"].endswith(("features_e77.parquet", "features_fin1b.parquet")), os.environ["FEATURES"]
df = pl.read_parquet(T.FEATURES_PATH)
fit, rank = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
del df
r = N.regular_candidate(fit, rank)
out = os.environ.get("E176_OUT", "submissions/e130_mlp_rank.npz")
np.savez(out, mvt_id=rank["MVT_ID_mvt"].to_numpy(), reg=r)
print(f"wrote {out}: {len(r)} rows, mean {float(np.mean(r)):.1f}", flush=True)
