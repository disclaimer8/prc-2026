"""E146 (27.09, journal 115): drift of the ADS-B stage inputs between 2025 and 2026 (GREKI hint: adsb.lol's receiver
network changed; features describing where an aircraft was heard did not transfer, features describing its movement did).
Same calendar months (Jan, Jul) and same airport, coverage >= 2: per-feature AUC of "row is from 2026", averaged over
the 10 airports x 2 months weighted by rows. Base-model columns are excluded (OOF in 2025 vs a full fit in 2026).
"""
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
FEATS = E36.FEATS + E36.VISIT + SW + NEW + NAMES_M + HEAD
SKIP = {"base", "p_nonreg", "base_vs_first_move", "base_vs_first", "ap"}


def augment(d, year):
    vd = "data/adsblol"
    vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{vd}/visit_{year}-*.parquet"))])
    vim = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{vd}/visit_m_{year}-*.parquet"))])
    walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                        for f in sorted(glob.glob(f"{os.environ.get('WALK_DIR', 'data/adsblol_standwalk')}/offblock2_{year}-*.parquet"))]).rename(
        {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
    hd = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"data/adsblol/head_{year}-*.parquet"))]).unique("MVT_ID_mvt")
    d = d.join(vis, on="MVT_ID_mvt", how="left").join(vim, on="MVT_ID_mvt", how="left").join(walked, on="MVT_ID_mvt", how="left").join(hd, on="MVT_ID_mvt", how="left")
    return d.with_columns(sw_mvt_first=pl.col("to") - pl.col("t_first_sw"), sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
                          sw_extra_s=pl.col("t_first") - pl.col("t_first_sw")).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))


raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
raw = raw.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
a = augment(E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")), 2025)
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=[c for c in E36.cols if c != E36.Y]).filter(pl.col("PHASE_mvt") == "DEP")
sub = pl.read_parquet("submissions/gentle-igloo_v362.parquet").rename({E36.Y: "base"}).with_columns(pl.col("base").cast(pl.Float64))
parts = np.load("submissions/parts_v291.npz")
base = sub.join(pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "p_nonreg": 1 - parts["proba"][:, 0]}), on="MVT_ID_mvt", how="left")
b = augment(E36.build(rk.with_columns(pl.lit(None, dtype=pl.Int32).alias(E36.Y)), base, E36.load_adsb("data/adsblol/offblock2_2026-*.parquet")), 2026)
feats = [c for c in FEATS if c not in SKIP and c in a.columns and c in b.columns]
a = a.filter(pl.col("coverage") >= 2).with_columns(yr=pl.lit(0), m=pl.col("MVT_TIME_UTC_mvt").dt.month())
b = b.filter(pl.col("coverage") >= 2).with_columns(yr=pl.lit(1), m=pl.col("MVT_TIME_UTC_mvt").dt.month())
print(f"coverage>=2 rows: 2025 Jan+Jul {a.height}, 2026 Jan+Jul {b.height}")
print("coverage>=2 share of departures by airport-month, 2025 vs 2026:")
cov = lambda x, y: x.group_by("ADEP_mvt", "m").len().rename({"len": y})
print(cov(a, "n25").join(cov(b, "n26"), on=["ADEP_mvt", "m"]).sort("ADEP_mvt", "m").to_pandas().to_string())


def auc(x, y):  # P(x2026 > x2025), ties half; nulls dropped
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 50 or len(y) < 50:
        return np.nan, 0
    z = np.concatenate([x, y]); r = np.argsort(np.argsort(z, kind="mergesort"), kind="mergesort").astype(float)
    # average ranks for ties
    _, inv, cnt = np.unique(z, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    r = (start + csum - 1) / 2.0
    r = r[inv]
    ry = r[len(x):].sum()
    return (ry - len(y) * (len(y) - 1) / 2) / (len(x) * len(y)), len(x) + len(y)


rows = []
for f in feats:
    num = den = 0.0
    nul = []
    for (ap, m), ga in a.group_by("ADEP_mvt", "m"):
        gb = b.filter((pl.col("ADEP_mvt") == ap) & (pl.col("m") == m))
        if gb.height < 50:
            continue
        xa, xb = ga[f].cast(pl.Float64).fill_null(np.nan).to_numpy(), gb[f].cast(pl.Float64).fill_null(np.nan).to_numpy()
        v, n = auc(xa, xb)
        if not np.isnan(v):
            num += abs(v - 0.5) * n; den += n
        nul.append((np.isnan(xa).mean(), np.isnan(xb).mean()))
    rows.append((f, round(0.5 + num / den, 3) if den else None, round(float(np.mean([u for u, _ in nul])), 3), round(float(np.mean([w for _, w in nul])), 3)))
rows.sort(key=lambda r: -(r[1] or 0))
print("\nfeature | mean |AUC-0.5|+0.5 within airport-month (1.0 = separates years perfectly) | null share 2025 | 2026")
for r in rows:
    print(r)
