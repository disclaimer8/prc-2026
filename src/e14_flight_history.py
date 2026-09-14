"""E14: flight-number history for LIRF departures without NM at 0 < D < 4 h (residual diagnostics, 14.09).

For each fold, a logistic model "block = schedule vs regular" is fitted on the fold's training months with features:
logit of the out-of-month frequency of block = schedule for the same FLIGHT_mvt (all rows and delayed > 30 min),
log(1 + n history rows), day_nonm_rate, log D, D > 2 h, local hour. It is applied only to rows whose flight number has
>= 3 history rows; prediction = p * D + (1 - p) * regular prediction (no one-day shift in this range).
Control: the full model's out-of-fold prediction (data/cv_preds). Pre-registered gate: SSE on the target rows lower in
>= 5 of 6 folds, >= 50% of the gain kept without the top-3 rows, full RMSE not worse in >= 4 of 6 folds.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

Y = "TAXITIME_SEC_mvt"
f = pl.read_parquet(os.environ.get("FEATURES", "data/features_final.parquet"),
                    columns=["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "local_hour", "day_nonm_rate", Y])
f = f.filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
f = f.with_columns(sched=((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).cast(pl.Int8), delayed=(pl.col("d_mvt_sched") > 1800))


def history(src: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    h = src.group_by("FLIGHT_mvt").agg(n=pl.len(), k=pl.col("sched").sum(), nd=pl.col("delayed").sum(), kd=(pl.col("sched") * pl.col("delayed")).sum())
    return rows.join(h, on="FLIGHT_mvt", how="left").with_columns(pl.col("n", "k", "nd", "kd").fill_null(0)).with_columns(
        ff=((pl.col("k") + 0.5) / (pl.col("n") + 1)), ffd=((pl.col("kd") + 0.5) / (pl.col("nd") + 1))
    )


def design(d: pl.DataFrame) -> np.ndarray:
    logit = lambda c: np.log(d[c].to_numpy() / (1 - d[c].to_numpy()))
    D = d["d_mvt_sched"].to_numpy()
    return np.column_stack([logit("ff"), logit("ffd"), np.log1p(d["n"].to_numpy()), d["day_nonm_rate"].to_numpy(),
                            np.log(np.clip(D, 60, None)), (D > 7200).astype(float), d["local_hour"].to_numpy()])


results = []
for path in sorted(glob.glob("data/cv_preds/fold_*.npz")):
    z = np.load(path)
    a, b = [int(x) for x in os.path.basename(path)[5:-4].split("_")]
    base = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "reg": z["reg"], "y": z["y"]})
    fit = f.filter(~pl.col("month").is_in([a, b]))
    # out-of-month history for training rows
    parts = []
    for m in fit["month"].unique().to_list():
        parts.append(history(fit.filter(pl.col("month") != m), fit.filter(pl.col("month") == m)))
    fit_h = pl.concat(parts).filter((pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400))
    model = LogisticRegression(C=1.0, max_iter=2000).fit(design(fit_h), fit_h["sched"].to_numpy())
    val = history(fit, f.filter(pl.col("month").is_in([a, b]))).filter((pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400))
    val = val.join(base, on="MVT_ID_mvt", how="inner")
    p = model.predict_proba(design(val))[:, 1]
    use = val["n"].to_numpy() >= 3
    new = np.where(use, p * val["d_mvt_sched"].to_numpy() + (1 - p) * val["reg"].to_numpy(), val["pred"].to_numpy())
    y, old = val["y"].to_numpy(), val["pred"].to_numpy()
    se_old, se_new = (old - y) ** 2, (new - y) ** 2
    gain = se_old.sum() - se_new.sum()
    top3 = np.argsort(-np.abs(se_old - se_new))[:3]
    mask = np.ones(len(y), bool)
    mask[top3] = False
    gain_wo = se_old[mask].sum() - se_new[mask].sum()
    n_all = len(base)
    full_old = np.sqrt(((base["pred"] - base["y"]) ** 2).sum() / n_all)
    full_new = np.sqrt((((base["pred"] - base["y"]) ** 2).sum() - gain) / n_all)
    results.append((a, b, gain, gain_wo, full_new - full_old))
    print(f"fold {a}+{b}: target rows {len(y)} (applied {int(use.sum())}), SSE {se_old.sum():.3e} -> {se_new.sum():.3e}, "
          f"gain kept without top-3 {gain_wo / gain if gain else float('nan'):.2f}, full RMSE {full_old:.2f} -> {full_new:.2f}")

lower = sum(g > 0 for _, _, g, _, _ in results)
total, total_wo = sum(r[2] for r in results), sum(r[3] for r in results)
not_worse = sum(d <= 0 for *_, d in results)
ok = lower >= 5 and total > 0 and total_wo >= 0.5 * total and not_worse >= 4
print(f"DECISION E14: {'ACCEPT' if ok else 'REJECT'} (SSE lower {lower}/6, kept without top-3 {total_wo / total if total else 0:.2f}, full not worse {not_worse}/6)")
