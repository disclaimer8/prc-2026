"""E141 apply (27.09): the E70 stage plus HEAD (src/e134_head_features.py) and the E141 missing-time estimate,
optionally averaged over LightGBM seeds (STAGE_SEEDS=42,1,2,3,4 = E142). Same inputs and call as e70_apply.py.
CV (journal 98): after E14 -0.137 s, lower in 6 of 6 folds.

Original E70 docstring follows.
Apply the accepted E70 stage to a submission (18.09).

E70 = the accepted E65 configuration plus the whole sub-threshold family measured together: E63's extended
backward pass over the ground track (five columns), E66's "where on the taxi the aircraft was caught and how
far is left" (three columns already sitting in offblock5), and E69's repair of a reception gap that split a
taxi in two (eleven columns in visit_m). Measured -0.38 s, lower in all six folds, strict -0.42, y <= 7200
-0.39, trimmed -0.35 (src/e70_all_subthreshold.py).

Written as its own file rather than another env branch inside e36_adsb_stand_geometry.py, so the path that
produced everything on the leaderboard so far is not touched.

    PARTS=submissions/parts_vN.npz .venv/bin/python src/e70_apply.py <base version> <new version>

The base must NOT already carry an ADS-B stage, exactly as for e36 apply: this fits the stage on 2025 and
writes the corrected ranking predictions over the base submission.
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

_argv, sys.argv = sys.argv, [sys.argv[0], "import"]
import e36_adsb_stand_geometry as E36  # noqa: E402
sys.argv = _argv
from e69_visit_merge import NAMES_M  # noqa: E402

SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]
HEAD = ["hd_gs0", "hd_gs_mean5", "hd_moving0", "hd_accel30", "hd_frac_mov60", "hd_t_first_stop", "hd_turn5", "hd_path_frac"]
E141 = ["e141_d_missed", "e141_t_rough"]
FEATS = E36.FEATS + E36.VISIT + SW + NEW + NAMES_M + HEAD + E141
SEEDS = [int(x) for x in os.environ.get("STAGE_SEEDS", "42").split(",")]

base_v, new_v = int(sys.argv[1]), int(sys.argv[2])


def augment(d: pl.DataFrame, year: int) -> pl.DataFrame:
    vd = os.environ.get("VISIT_DIR", "data/adsblol")  # E75: visits re-extracted across midnight live elsewhere
    vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{vd}/visit_{year}-*.parquet"))])
    vim = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{vd}/visit_m_{year}-*.parquet"))])
    walk = os.environ.get("WALK_DIR", "data/adsblol_standwalk")  # E74: the 1 kt stand-walk lives elsewhere
    walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                        for f in sorted(glob.glob(f"{walk}/offblock2_{year}-*.parquet"))]).rename(
        {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
    d = d.join(vis, on="MVT_ID_mvt", how="left").join(vim, on="MVT_ID_mvt", how="left").join(walked, on="MVT_ID_mvt", how="left")
    d = d.with_columns(
        sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
        sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
        sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
    ).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))
    hd = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{os.environ.get('HEAD_DIR', 'data/adsblol')}/head_{year}-*.parquet"))]).unique("MVT_ID_mvt")
    d = d.join(hd, on="MVT_ID_mvt", how="left").with_columns(
        e141_d_missed=pl.col("gs_d_first"),
        e141_t_rough=(pl.col("gs_d_first") * 2.0 / (pl.col("hd_gs0").clip(lower_bound=5.0) * 0.5144)).fill_nan(None))
    print(f"{year}: HEAD known on coverage>=2: {int(d.filter(pl.col('coverage') >= 2)['hd_gs0'].is_not_null().sum())} of {int((d['coverage'] >= 2).sum())}", flush=True)
    missing = [c for c in FEATS if c not in d.columns]
    if missing:
        sys.exit(f"{year}: missing columns {missing}")
    return d


raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
tr = augment(E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")), 2025)
print(f"fit rows {tr.height}; visit known {int(tr['vi_first_seen'].is_not_null().sum())}, "
      f"merged {int((tr['vm_merged'] > 0).sum())}, walk extended {int(tr['sw_recovered'].sum())}", flush=True)
models = [E36.fit_stage(tr, FEATS, {**E36.PARAMS30, "seed": sd}) for sd in SEEDS]
print("stage seeds", SEEDS, flush=True)

rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet",
                     columns=[c for c in E36.cols if c != E36.Y]).filter(pl.col("PHASE_mvt") == "DEP")
sub = pl.read_parquet(f"submissions/gentle-igloo_v{base_v}.parquet")
parts_path = os.environ.get("PARTS", f"submissions/parts_v{base_v - 2}.npz")
if not os.path.exists(parts_path):
    sys.exit(f"missing {parts_path}: set PARTS to the parts file of the blend that produced v{base_v}")
parts = np.load(parts_path)
print("regime probabilities from", parts_path)
pn = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "p_nonreg": 1 - parts["proba"][:, 0]})
base = sub.rename({E36.Y: "base"}).with_columns(pl.col("base").cast(pl.Float64)).join(pn, on="MVT_ID_mvt", how="left")
d = augment(E36.build(rk.with_columns(pl.lit(None, dtype=pl.Int32).alias(E36.Y)), base,
                      E36.load_adsb("data/adsblol/offblock2_2026-*.parquet")), 2026)
new = np.mean([E36.correct(m, d) for m in models], axis=0)
out = sub.join(pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "new": new}), on="MVT_ID_mvt", how="left").with_columns(
    pl.coalesce(pl.col("new").round(0).cast(sub.schema[E36.Y]), pl.col(E36.Y)).alias(E36.Y)).select(sub.columns)
dd = (out.sort("MVT_ID_mvt")[E36.Y].cast(float) - sub.sort("MVT_ID_mvt")[E36.Y].cast(float)).to_numpy()
out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v}: rows with coverage>=2 {int((d['coverage'] >= 2).sum())}, merged {int((d['vm_merged'] > 0).sum())}, "
      f"changed {(dd != 0).sum()}, RMS change {np.sqrt((dd ** 2).mean()):.2f} s")
