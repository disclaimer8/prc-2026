"""E15: label-free flight identity for LIRF departures without NM at 0 < D < 4 h (review "top-3 levers", M1/M3, 14.09).

Same-month counts over all LIRF departures of the same split (so they exist for 2026 flight numbers unseen in 2025):
n_rk (departures of this FLIGHT_mvt in the month), share of them matched to NM, orphan flag (never matched), garbled call sign.
Basis (src/crux_m3_probe.py, 2025 labels): block = schedule for 86% of never-matched flights vs 31% for flights matched on
>= 10 days of the month. A logistic model "block = schedule" is fitted per fold on the training months with these counts plus
log D, D > 2 h, local hour and day_nonm_rate, and applied to every target row: prediction = p * D + (1 - p) * regular.
Control and gate are the same as E14: OOF prediction of the full model (data/cv_preds), SSE on target rows lower in >= 5 of 6
folds, >= 50% of the gain kept without the top-3 rows, full RMSE not worse in >= 4 of 6.
    FEATURES=data/features_final.parquet .venv/bin/python src/e15_flight_identity.py
"""
import glob
import os
import re

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "local_hour", "day_nonm_rate", Y]
lirf = pl.read_parquet(os.environ.get("FEATURES", "data/features_final.parquet"), columns=cols).filter(
    (pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LIRF"))
counts = lirf.group_by("FLIGHT_mvt", "month").agg(n_rk=pl.len(), n_nm=pl.col("FLIGHT_ID_mvt").is_not_null().sum())
# rows without a flight number (54 in 2025) have no identity at all and are treated as orphans: n_rk = 1, n_nm = 0
f = lirf.filter(pl.col("FLIGHT_ID_mvt").is_null()).join(counts, on=["FLIGHT_mvt", "month"], how="left").with_columns(
    pl.col("n_rk").fill_null(1), pl.col("n_nm").fill_null(0)).with_columns(
    sched=((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).cast(pl.Int8),
    share_nm=pl.col("n_nm") / pl.col("n_rk"),
    orphan=(pl.col("n_nm") == 0).cast(pl.Float64),
    garbled=pl.col("FLIGHT_mvt").map_elements(
        lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64).fill_null(0.0),
).filter((pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400))


def design(d: pl.DataFrame) -> np.ndarray:
    D = d["d_mvt_sched"].to_numpy()
    return np.column_stack([np.log1p(d["n_rk"].to_numpy()), d["share_nm"].to_numpy(), d["orphan"].to_numpy(), d["garbled"].to_numpy(),
                            np.log(np.clip(D, 60, None)), (D > 7200).astype(float), d["local_hour"].to_numpy(), d["day_nonm_rate"].to_numpy()])


results = []
for path in sorted(glob.glob("data/cv_preds/fold_*.npz")):
    z = np.load(path)
    a, b = [int(x) for x in os.path.basename(path)[5:-4].split("_")]
    base = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "reg": z["reg"], "y": z["y"]})
    fit = f.filter(~pl.col("month").is_in([a, b]))
    model = LogisticRegression(C=1.0, max_iter=2000).fit(design(fit), fit["sched"].to_numpy())
    val = f.filter(pl.col("month").is_in([a, b])).join(base, on="MVT_ID_mvt", how="inner")
    p = model.predict_proba(design(val))[:, 1]
    new = p * val["d_mvt_sched"].to_numpy() + (1 - p) * val["reg"].to_numpy()
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
    orph = val["orphan"].to_numpy() == 1
    results.append((a, b, gain, gain_wo, full_new - full_old))
    print(f"fold {a}+{b}: target rows {len(y)} (orphans {int(orph.sum())}), SSE {se_old.sum():.3e} -> {se_new.sum():.3e} "
          f"(orphans {se_old[orph].sum():.2e} -> {se_new[orph].sum():.2e}), full RMSE {full_old:.2f} -> {full_new:.2f}")

lower = sum(g > 0 for _, _, g, _, _ in results)
total, total_wo = sum(r[2] for r in results), sum(r[3] for r in results)
not_worse = sum(d <= 0 for *_, d in results)
ok = lower >= 5 and total > 0 and total_wo >= 0.5 * total and not_worse >= 4
print(f"DECISION E15: {'ACCEPT' if ok else 'REJECT'} (SSE lower {lower}/6, kept without top-3 {total_wo / total if total else 0:.2f}, "
      f"full not worse {not_worse}/6, total SSE gain {total:.3e} = {total / (2 * 274.93 * 344841):.2f} s LB-equivalent)")
