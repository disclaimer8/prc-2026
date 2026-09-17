"""Where the remaining error of the current best model is systematic (17.09).

    .venv/bin/python src/residual_bias.py [pred_dir]      default data/cv_preds_e46 (= v55 base)

The transfer rule says only two kinds of change survive onto the leaderboard: calibration of a fixed
transformation, and a stage built on physical observation of the flight. Both need the same input: a group of rows
where the model is wrong *in the same direction every month*. A bias that flips sign between folds is composition
of the training period and correcting it is the third, non-transferring kind.

So for every grouping this prints, per group:
    n           rows over the six folds
    bias        mean signed error (prediction - truth), seconds
    signs       how many of the six folds share the sign of the pooled bias
    gain        SSE that removing the bias would free, in leaderboard-equivalent seconds

Only rows the base model actually decides are counted (src/fold_metrics.py): the LIRF tail table overwrites the
rest, so a bias there costs nothing.
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

LB = 2 * 252.43 * 344841  # SSE -> leaderboard seconds, per fold
FEATURES = "data/features_e28.parquet"
MIN_N = 2000  # a group smaller than this is noise at this scale


def coverage_map() -> dict:
    out = {}
    for f in sorted(glob.glob("data/adsblol/offblock2_*.parquet")):
        d = pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"])
        out.update(dict(zip(d["MVT_ID_mvt"].to_list(), d["coverage"].to_list())))
    return out


def main() -> None:
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "data/cv_preds_e46"
    dec = decided_map(FEATURES)
    cols = ["MVT_ID_mvt", "ADEP_mvt", "local_hour", "AIRCRAFT_TYPE_mvt", "airline_prefix", "stand_prefix",
            "RUNWAY_mvt", "d_mvt_sched", "FLIGHT_ID_mvt", "MARKET_SEGMENT_flt", "wx_snow", "wx_thunder", "wx_fog"]
    ft = pl.read_parquet(FEATURES, columns=cols)
    cov = coverage_map()
    ft = ft.with_columns(coverage=pl.col("MVT_ID_mvt").replace_strict(cov, default=-1))

    rows = []
    for i, p in enumerate(sorted(glob.glob(f"{pred_dir}/fold_*.npz"))):
        z = np.load(p)
        m = np.array([dec[k] for k in z["mvt_id"]])
        rows.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"][m], "err": (z["pred"] - z["y"])[m],
                                  "fold": np.full(int(m.sum()), i, dtype=np.int8)}))
    df = pl.concat(rows).join(ft, on="MVT_ID_mvt", how="left")
    df = df.with_columns(
        no_nm=pl.col("FLIGHT_ID_mvt").is_null(),
        sched_h=(pl.col("d_mvt_sched") / 3600).round(0).clip(-2, 8),
        wx_bad=(pl.col("wx_snow").fill_null(0) + pl.col("wx_thunder").fill_null(0) + pl.col("wx_fog").fill_null(0)) > 0,
    )
    n_all, sse_all = df.height, float((df["err"] ** 2).sum())
    print(f"{pred_dir}: {n_all} decided rows over 6 folds, RMSE {np.sqrt(sse_all / n_all):.1f} s, "
          f"pooled bias {float(df['err'].mean()):+.1f} s\n")

    for keys in (["ADEP_mvt"], ["coverage"], ["local_hour"], ["no_nm"], ["sched_h"], ["MARKET_SEGMENT_flt"],
                 ["wx_bad"], ["ADEP_mvt", "no_nm"], ["ADEP_mvt", "coverage"], ["airline_prefix"],
                 ["AIRCRAFT_TYPE_mvt"], ["stand_prefix"], ["ADEP_mvt", "RUNWAY_mvt"]):
        g = df.group_by(keys).agg(n=pl.len(), bias=pl.col("err").mean(),
                                  per_fold=pl.col("err").mean().over(keys + ["fold"]).unique())
        per = (df.group_by(keys + ["fold"]).agg(b=pl.col("err").mean())
                 .group_by(keys).agg(signs_pos=(pl.col("b") > 0).sum(), folds=pl.len()))
        g = g.join(per, on=keys).with_columns(
            signs=pl.max_horizontal(pl.col("signs_pos"), pl.col("folds") - pl.col("signs_pos")),
            gain=-(pl.col("n") * pl.col("bias") ** 2) / (6 * LB))
        g = g.filter(pl.col("n") >= MIN_N).sort("gain")
        head = g.head(8)
        tot = float(g.filter((pl.col("signs") == pl.col("folds")) & (pl.col("folds") == 6))["gain"].sum())
        print(f"--- {'+'.join(keys)}  (sign-stable groups together: {tot:+.2f} s)")
        for r in head.iter_rows(named=True):
            k = " ".join(str(r[c]) for c in keys)
            print(f"    {k:28s} n={r['n']:7d}  bias {r['bias']:+7.1f} s  signs {r['signs']}/{r['folds']}  "
                  f"gain {r['gain']:+.2f} s")
        print()


if __name__ == "__main__":
    main()
