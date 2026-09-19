"""E64 apply: fit the visit stage on all of 2025 and correct the ranking rows that have no ground segment.

    PARTS=submissions/parts_v52.npz .venv/bin/python src/e64_apply.py 55 76

The E36 stage in v55 corrects coverage >= 2; this one corrects coverage < 2, so the two populations are disjoint
and this applies straight on top. Base, features, clips and rounds are exactly those of src/e64_visit_stage.py.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb
import numpy as np
import polars as pl

# E74: coverage comes from the extraction the stage was fitted on; the 1 kt run lives in another directory,
# and E64 must read the SAME coverage as E70 or their populations stop being disjoint.
ADSB_DIR = os.environ.get("ADSB_DIR", "data/adsblol")

sys.path.insert(0, os.path.dirname(__file__))
from adsb_visit import NAMES  # noqa: E402
from e64_visit_stage import AIRPORTS, FEATS, PARAMS, ROUNDS, Y  # noqa: E402

base_v, new_v = int(sys.argv[1]), int(sys.argv[2])
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]


def build(raw, base, cov, vis):
    return (raw.join(base, on="MVT_ID_mvt", how="inner").join(cov, on="MVT_ID_mvt", how="left")
            .join(vis, on="MVT_ID_mvt", how="left")
            .with_columns(coverage=pl.col("coverage").fill_null(-1),
                          ap=pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(AIRPORTS)}, return_dtype=pl.Int32),
                          has_nm=pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8),
                          D=(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s") - pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s")).cast(pl.Float64)))


usable = ((pl.col("coverage") < 2) & (pl.col("vi_has_prev") == 1)).fill_null(False)

tr_raw = pl.concat([pl.read_parquet(f, columns=cols + [Y]) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
tr_base = pl.concat([pl.DataFrame({"MVT_ID_mvt": np.load(p)["mvt_id"], "base": np.load(p)["pred"],
                                   "p_nonreg": 1 - np.load(p)["proba"][:, 0]})
                     for p in sorted(glob.glob("data/cv_preds_nomonth/fold_*.npz"))])
tr = build(tr_raw,
           tr_base,
           pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob(f"{ADSB_DIR}/offblock2_2025-*.parquet"))]),
           pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))]))
t = tr.filter(usable & ((pl.col(Y) - pl.col("base")).abs() <= 3600))
print(f"stage trained on {t.height} rows of 2025", flush=True)
m = lgb.train(PARAMS, lgb.Dataset(t.select(FEATS).to_numpy(),
                                  np.clip((t[Y] - t["base"]).to_numpy(), -1800, 1800), categorical_feature=[0]), ROUNDS)

rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=cols).filter(pl.col("PHASE_mvt") == "DEP")
sub = pl.read_parquet(f"submissions/gentle-igloo_v{base_v}.parquet")
parts_path = os.environ.get("PARTS", f"submissions/parts_v{base_v - 3}.npz")
if not os.path.exists(parts_path):
    sys.exit(f"missing {parts_path}: set PARTS to the parts file of the blend that produced v{base_v}")
parts = np.load(parts_path)
print("regime probabilities from", parts_path)
rk_base = (sub.rename({Y: "base"}).with_columns(pl.col("base").cast(pl.Float64))
           .join(pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "p_nonreg": 1 - parts["proba"][:, 0]}), on="MVT_ID_mvt", how="left"))
d = build(rk, rk_base,
          pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob(f"{ADSB_DIR}/offblock2_2026-*.parquet"))]),
          pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2026-*.parquet"))]))
use = d.select(usable).to_series().to_numpy()
new = np.clip(d["base"].to_numpy() + np.where(use, m.predict(d.select(FEATS).to_numpy()), 0.0), 0, None)
out = sub.join(pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "new": new}), on="MVT_ID_mvt", how="left").with_columns(
    pl.coalesce(pl.col("new").round(0).cast(sub.schema[Y]), pl.col(Y)).alias(Y)).select(sub.columns)
dd = (out.sort("MVT_ID_mvt")[Y].cast(float) - sub.sort("MVT_ID_mvt")[Y].cast(float)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v}: ranking rows {d.height}, corrected {int(use.sum())} ({100*use.mean():.1f}%), "
      f"changed {(dd != 0).sum()}, RMS change {np.sqrt((dd ** 2).mean()):.2f} s")
