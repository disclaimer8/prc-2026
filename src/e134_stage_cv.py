"""E134 stage CV (26.09): does the HEAD of the surface track (src/e134_head_features.py) tell the E70 stage anything
about the unseen start? Pre-registered in REVIEW_BRIEF.md 21bz before the run.

Control: the accepted E70 stage (E36.FEATS + visit + stand-walk + geometry + merged visit), rebuilt here on the same days.
  cand_a: control features + HEAD, trained on real rows only (the natural partial tracks, coverage 2).
  cand_b: cand_a plus the E39 truncated copies (head_trunc_<day>: cut 60-600 s after first movement, offblock2/3
          and HEAD recomputed); the copies carry no visit/geometry features (NaN). Evaluated on real rows only.
Gate (21bz): decided metric >= 0.3 s lower, lower in >= 4 of 6 folds, the same on y <= 7200, and the gain survives
dropping each fold's three best and three worst rows; then post_e14 after the chain tail (reported by the shell).
Prediction before the run: -0.3 ... +0.2, centre -0.05.\ncand_c (E140, journal 89): control + neighbour-pace reconstruction of the unseen start, same gate; prediction -0.2 ... +0.05.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
_argv, sys.argv = sys.argv, [sys.argv[0], "import"]
import e36_adsb_stand_geometry as E36  # noqa: E402
sys.argv = _argv
from adsb_visit import NAMES  # noqa: E402
from e69_visit_merge import NAMES_M  # noqa: E402
from fold_metrics import compare  # noqa: E402

SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]
HEAD = ["hd_gs0", "hd_gs_mean5", "hd_moving0", "hd_accel30", "hd_frac_mov60", "hd_t_first_stop", "hd_turn5", "hd_path_frac"]
DIRS = {"ctrl": "data/cv_preds_e134_ctrl", "cand_a": "data/cv_preds_e134_a", "cand_b": "data/cv_preds_e134_b",
        "cand_c": "data/cv_preds_e140"}
E140 = ["nb_full", "nb_pace", "nb_n", "e140_missing", "e140_total"]


def e140_features(d: pl.DataFrame, half: float = 1800.0) -> pl.DataFrame:
    """E140 (journal 89): the unseen start rebuilt from OTHER aircraft. Neighbours = coverage-3 departures (stand dwell
    and first movement seen) at the same airport with |MVT - MVT_i| <= half, self excluded. nb_full = their mean
    first-movement-to-take-off time; nb_pace = their mean seconds per metre of straight stand-runway distance;
    e140_missing = distance already behind at the first point (gs_d_first) x nb_pace; e140_total = MVT - t_first + missing."""
    full = (pl.col("t_to") - pl.col("t_move"))
    x = d.with_columns(
        _ok=((pl.col("coverage") == 3) & full.is_between(60, 3600)).cast(pl.Float64),
        _full=pl.when((pl.col("coverage") == 3) & full.is_between(60, 3600)).then(full).otherwise(0.0),
        _pace=pl.when((pl.col("coverage") == 3) & full.is_between(60, 3600) & (pl.col("gs_d_stand_rwy") > 200))
               .then(full / pl.col("gs_d_stand_rwy")).otherwise(None))
    x = x.with_columns(_okp=pl.col("_pace").is_not_null().cast(pl.Float64), _pace=pl.col("_pace").fill_null(0.0))
    parts = []
    for _, g in x.sort("to").group_by("ADEP_mvt", maintain_order=True):
        t = g["to"].to_numpy()
        lo, hi = np.searchsorted(t, t - half, "left"), np.searchsorted(t, t + half, "right")
        cols = {}
        for c in ("_ok", "_full", "_okp", "_pace"):
            v = g[c].to_numpy()
            cs = np.concatenate([[0.0], np.cumsum(v)])
            cols[c] = cs[hi] - cs[lo] - v
        n, np_ = cols["_ok"], cols["_okp"]
        parts.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "nb_n": n,
                                   "nb_full": np.where(n > 0, cols["_full"] / np.maximum(n, 1), np.nan),
                                   "nb_pace": np.where(np_ > 0, cols["_pace"] / np.maximum(np_, 1), np.nan)}))
    nb = pl.concat(parts)
    return d.join(nb, on="MVT_ID_mvt", how="left").with_columns(
        e140_missing=pl.col("gs_d_first") * pl.col("nb_pace")).with_columns(
        e140_total=pl.col("mvt_first") + pl.col("e140_missing"))

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
base = E36.oof_base()
_reg = {}
for _p in sorted(glob.glob(f"{os.environ.get('E36_BASE_DIR', 'data/cv_preds_nomonth')}/fold_*.npz")):
    _z = np.load(_p)
    _reg.update(zip(_z["mvt_id"].tolist(), _z["reg"].tolist()))
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])
E36.ADSB_DIR = "data/adsblol"
d = E36.build(raw, base, E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).join(
    vis, on="MVT_ID_mvt", how="left").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                    for f in sorted(glob.glob("data/adsblol_standwalk/offblock2_2025-*.parquet"))]).rename(
    {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
d = d.join(walked, on="MVT_ID_mvt", how="left").with_columns(
    sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
    sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
    sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))
d = d.join(pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_m_2025-*.parquet"))]), on="MVT_ID_mvt", how="left")
hd = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/head_2025-*.parquet"))]).unique("MVT_ID_mvt")
d = e140_features(d.join(hd, on="MVT_ID_mvt", how="left"))
print("E140 coverage on coverage>=2:", d.filter(pl.col("coverage") >= 2).select(
    pl.col("nb_full").is_not_null().mean(), pl.col("e140_missing").is_not_null().mean()).to_dicts(), flush=True)

trunc = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/head_trunc_2025-*.parquet"))]).with_columns(
    n_cand=pl.lit(1), gs_t_leave=pl.lit(None, dtype=pl.Float64))
aug = E36.build(raw, base, trunc).with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())

CTRL = E36.FEATS + NAMES + SW + NEW + NAMES_M
missing = [c for c in CTRL + HEAD if c not in d.columns]
if missing:
    sys.exit(f"missing columns: {missing}")
cov2 = d.filter(pl.col("coverage") >= 2)
print(f"rows {d.height}; coverage>=2 {cov2.height}; HEAD known on coverage>=2 {int(cov2['hd_gs0'].is_not_null().sum())}; "
      f"truncated copies {aug.height}", flush=True)

for p in DIRS.values():
    os.makedirs(p, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    tr_aug = pl.concat([tr, aug.filter(~pl.col("month").is_in([a, b]))], how="diagonal_relaxed")
    y = te[E36.Y].to_numpy()
    line = []
    for name, train, feats in (("ctrl", tr, CTRL), ("cand_a", tr, CTRL + HEAD), ("cand_b", tr_aug, CTRL + HEAD),
                                ("cand_c", tr, CTRL + E140)):
        pred = E36.correct(E36.fit_stage(train, feats), te)
        ids = te["MVT_ID_mvt"].to_numpy()
        np.savez(f"{DIRS[name]}/fold_{a}_{b}.npz", mvt_id=ids, pred=pred, y=y, reg=np.array([_reg.get(m, np.nan) for m in ids.tolist()]))
        line.append(f"{name} {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

for name in ("cand_a", "cand_b", "cand_c"):
    dec, tot, small, trim, lower, n, strict = compare(DIRS["ctrl"], DIRS[name])
    ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
    print(f"DECISION {'E140' if name == 'cand_c' else 'E134 ' + name}: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided {dec:+.2f} s, "
          f"strict {strict:+.2f} s, all {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
