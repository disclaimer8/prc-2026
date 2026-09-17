"""Where the remaining SSE of the post-stage model sits (17.09).

    .venv/bin/python src/residual_where.py [pred_dir]     default data/cv_preds_e28_e30

src/residual_bias.py showed the post-stage model has almost no systematic bias left in any slice (every grouping
under 0,1 s of leaderboard-equivalent gain), so the remaining error is variance. This prints where that variance
lives: share of total SSE by group, and how much of it a perfect handling of the worst rows would free.
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import decided_map  # noqa: E402
from residual_bias import coverage_map  # noqa: E402

LB = 6 * 2 * 252.43 * 344841
FEATURES = "data/features_e28.parquet"


def main() -> None:
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "data/cv_preds_e28_e30"
    dec = decided_map(FEATURES)
    ft = pl.read_parquet(FEATURES, columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "d_mvt_sched", "TAXITIME_SEC_mvt"])
    ft = ft.with_columns(coverage=pl.col("MVT_ID_mvt").replace_strict(coverage_map(), default=-1))
    rows = []
    for p in sorted(glob.glob(f"{pred_dir}/fold_*.npz")):
        z = np.load(p)
        m = np.array([dec[k] for k in z["mvt_id"]])
        rows.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"][m], "err": (z["pred"] - z["y"])[m], "y": z["y"][m].astype(float)}))
    df = pl.concat(rows).join(ft.drop("TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="left").with_columns(e2=pl.col("err") ** 2)
    sse = float(df["e2"].sum())
    n = df.height
    print(f"{pred_dir}: {n} decided rows, RMSE {np.sqrt(sse / n):.1f} s, total {sse / LB:.2f} s of leaderboard budget\n")

    a = np.sort(df["e2"].to_numpy())[::-1]
    c = np.cumsum(a) / sse
    print("--- concentration: the worst rows carry")
    for k in (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5):
        i = int(n * k)
        print(f"    top {k * 100:6.2f}% ({i:7d} rows): {c[i - 1] * 100:5.1f}% of SSE  = {a[:i].sum() / LB:6.2f} s")
    print()

    for keys, label in ((["ADEP_mvt"], "airport"), (["coverage"], "coverage")):
        g = df.group_by(keys).agg(n=pl.len(), sse=pl.col("e2").sum(), rmse=(pl.col("e2").mean() ** 0.5)).sort("sse", descending=True)
        print(f"--- SSE by {label}")
        for r in g.iter_rows(named=True):
            print(f"    {str(r[keys[0]]):8s} n={r['n']:7d}  RMSE {r['rmse']:6.1f}  {r['sse'] / sse * 100:5.1f}% = {r['sse'] / LB:6.2f} s")
        print()

    q = df.with_columns(band=pl.when(pl.col("y") <= 300).then(pl.lit("0-5 min")).when(pl.col("y") <= 600).then(pl.lit("5-10"))
                        .when(pl.col("y") <= 900).then(pl.lit("10-15")).when(pl.col("y") <= 1200).then(pl.lit("15-20"))
                        .when(pl.col("y") <= 1800).then(pl.lit("20-30")).when(pl.col("y") <= 3600).then(pl.lit("30-60"))
                        .otherwise(pl.lit("60+ min")))
    g = q.group_by("band").agg(n=pl.len(), sse=pl.col("e2").sum(), rmse=(pl.col("e2").mean() ** 0.5), bias=pl.col("err").mean()).sort("sse", descending=True)
    print("--- SSE by true taxi time")
    for r in g.iter_rows(named=True):
        print(f"    {r['band']:8s} n={r['n']:7d}  RMSE {r['rmse']:6.1f}  bias {r['bias']:+7.1f}  {r['sse'] / sse * 100:5.1f}% = {r['sse'] / LB:6.2f} s")


if __name__ == "__main__":
    main()
