"""E21: arrival linkage for departures without NM (review 6, 14.09; pre-registered 14.09 in REVIEW_BRIEF.md, first as E15, renumbered E21 on 15.09).

A departure without NM whose flight (same FLIGHT_mvt, ADEP, ADES) has exactly one NM-matched arrival at the destination hub
0-16 h after take-off gets LOBT from that arrival. If |D| > 3606 s and |SCHED - LOBT_arrival| <= 3606 s, the probabilities
of the SCHED and one-day-shift regimes are set to 0 and renormalised (2025: SCHED in 0 of 773 such rows).
Control: OOF prediction of the full model (data/cv_preds, needs proba/values). Gate as E14. Rows of LIRF at 4-30 h are
TabPFN-overridden in submissions, so they are reported separately (control here is the classifier).
    .venv/bin/python src/e21_arrival_link.py
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
raw = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))], how="diagonal_relaxed")
dep = raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_null() & pl.col("ADES_mvt").is_in(HUBS))
arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("FLIGHT_ID_mvt").is_not_null()).select(
    "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", a_mvt=pl.col("MVT_TIME_UTC_mvt"), a_lobt=pl.col("LOBT_flt"))
link = dep.join(arr, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], how="inner").filter(
    (pl.col("a_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().is_between(0, 16 * 3600))
link = link.filter(pl.len().over("MVT_ID_mvt") == 1).with_columns(
    D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
    gap=(pl.col("SCHED_TIME_UTC_mvt") - pl.col("a_lobt")).dt.total_seconds().abs())
rule = link.filter((pl.col("D").abs() > 3606) & (pl.col("gap") <= 3606)).with_columns(
    tabpfn_zone=(pl.col("ADEP_mvt") == "LIRF") & pl.col("D").is_between(4 * 3600, 30 * 3600)).select("MVT_ID_mvt", "ADEP_mvt", "tabpfn_zone")
print("rule rows in 2025:", rule.height, "of which LIRF 4-30 h:", rule["tabpfn_zone"].sum())

SCHED, DAYSHIFT = 1, 5  # train.CLASSES = regular, sched, iobt, lobt, eobt, dayshift
results = []
for path in sorted(glob.glob("data/cv_preds/fold_*.npz")):
    z = np.load(path)
    a, b = [int(x) for x in os.path.basename(path)[5:-4].split("_")]
    ids = z["mvt_id"]
    hit = np.isin(ids, rule["MVT_ID_mvt"].to_numpy())
    zone = np.isin(ids, rule.filter(pl.col("tabpfn_zone"))["MVT_ID_mvt"].to_numpy())
    proba, values, y, old = z["proba"].copy(), z["values"], z["y"], z["pred"].copy()
    proba[hit, SCHED] = 0.0
    proba[hit, DAYSHIFT] = 0.0
    s = proba[hit].sum(axis=1, keepdims=True)
    mix = np.where(s[:, 0] > 0, (proba[hit] / np.where(s > 0, s, 1) * np.nan_to_num(values[hit])).sum(axis=1), z["reg"][hit])
    new = old.copy()
    new[hit] = np.clip(mix, 0, None)
    se_old, se_new = (old - y) ** 2, (new - y) ** 2
    gain = se_old[hit].sum() - se_new[hit].sum()
    d = (se_old - se_new)[hit]
    top3 = np.argsort(-np.abs(d))[:3]
    gain_wo = gain - d[top3].sum()
    gain_zone = se_old[zone].sum() - se_new[zone].sum()
    full_old, full_new = np.sqrt(se_old.mean()), np.sqrt(se_new.mean())
    results.append((a, b, gain, gain_wo, full_new - full_old))
    print(f"fold {a}+{b}: rule rows {int(hit.sum())} (LIRF 4-30 h {int(zone.sum())}, gain there {gain_zone:.2e}), "
          f"SSE gain {gain:.3e}, full RMSE {full_old:.2f} -> {full_new:.2f}")

lower = sum(g > 0 for _, _, g, _, _ in results)
total, total_wo = sum(r[2] for r in results), sum(r[3] for r in results)
not_worse = sum(dd <= 0 for *_, dd in results)
ok = lower >= 5 and total > 0 and total_wo >= 0.5 * total and not_worse >= 4
print(f"DECISION E21: {'ACCEPT' if ok else 'REJECT'} (SSE lower {lower}/6, kept without top-3 {total_wo / total if total else 0:.2f}, "
      f"full not worse {not_worse}/6, total SSE gain {total:.3e} = {total / (2 * 274.93 * 344841):.2f} s LB-equivalent)")
