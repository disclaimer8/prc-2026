"""E58: does recalibrating the schedule-regime probability change anything? (reopening E6, 17.09)

    .venv/bin/python src/e58_psched_recalibration.py            control data/cv_preds_e28 (saved proba + values)

E6 (monotone recalibration of p_sched per airport and delay bin) was closed on 14.09 without a run, on the argument
that cross-month recalibrations elsewhere gave +0,01…+0,18 s and the classifier already looks calibrated. It was
closed BEFORE the gate metric was fixed on 16.09, so it deserves a number rather than an argument — E25 was also
closed by mistake once and later gave -15,22 s.

The test needs no retraining: the fold files of data/cv_preds_e28 carry the regime probabilities and the candidate
values, so the mixture can be rebuilt exactly (verified against the saved prediction before anything is changed).
For every fold the recalibration curve is fitted on the OTHER ten months only, per airport, as an isotonic map of
p_sched onto the observed share of the schedule regime; the remaining classes keep their relative weights.

Gate, written before the run: ACCEPT only if the decided-rows metric is at least 0,3 s LB-equivalent lower, lower in
at least 4 of 6 folds, and the y <= 7200 number is also at least 0,3 s lower. Anything else is a REJECT and E6 stays
closed with a number.
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__))
import train as T  # noqa: E402
from fold_metrics import decided_map  # noqa: E402

LB = 6 * 2 * 252.43 * 344841
CTRL = os.environ.get("E58_CTRL", "data/cv_preds_e28")
SCHED = 1  # T.CLASSES.index("sched")
FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]


def mixture(proba: np.ndarray, values: np.ndarray, d_lobt: np.ndarray) -> np.ndarray:
    """The same arithmetic as train.fit_predict: mask infeasible, renormalise, mix, clip to the LOBT window."""
    bad = ~np.isnan(d_lobt[:, None]) & (np.abs(np.nan_to_num(values) - np.nan_to_num(d_lobt[:, None])) > T.LOBT_WINDOW)
    bad[:, 0] = False
    p = np.where(bad, 0.0, proba)
    s = p.sum(axis=1, keepdims=True)
    p = np.where(s > 0, p / np.where(s > 0, s, 1), proba)
    p = p / p.sum(axis=1, keepdims=True)
    out = np.sum(p * np.nan_to_num(values), axis=1)
    lo, hi = d_lobt - T.LOBT_WINDOW, d_lobt + T.LOBT_WINDOW
    return np.clip(np.where(np.isnan(d_lobt), out, np.clip(out, lo, hi)), 0, None)


def main() -> None:
    dec = decided_map("data/features_e28.parquet")
    df = pl.read_parquet("data/features_e28.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "month", "TAXITIME_SEC_mvt", "d_mvt_sched",
                                  "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt"])
    df = df.with_columns(true_sched=pl.Series(T.regime_label(df) == SCHED))
    meta = {r[0]: (r[1], r[2], r[3]) for r in df.select("MVT_ID_mvt", "ADEP_mvt", "d_mvt_lobt", "true_sched").iter_rows()}

    tot = small = 0.0
    lower = 0
    for (a, b), path in zip(FOLDS, sorted(glob.glob(f"{CTRL}/fold_*.npz"))):
        z = np.load(path)
        mvt, proba, values, y = z["mvt_id"], z["proba"], z["values"], z["y"].astype(float)
        ap = np.array([meta[m][0] for m in mvt])
        d_lobt = np.array([meta[m][1] if meta[m][1] is not None else np.nan for m in mvt], dtype=float)
        rebuilt = mixture(proba, values, d_lobt)
        off = np.abs(rebuilt - z["pred"]).max()
        if off > 1.0:
            sys.exit(f"{path}: the mixture cannot be rebuilt from the saved parts (max difference {off:.1f} s) — "
                     "this test would compare against something the saved prediction is not")
        fit = df.filter(~pl.col("month").is_in([a, b]))
        fit_ap = fit["ADEP_mvt"].to_numpy()
        fit_true = fit["true_sched"].to_numpy()
        fit_id = fit["MVT_ID_mvt"].to_numpy()
        # the fit rows' own probabilities come from the same saved folds (each row is out-of-fold exactly once)
        pool = {}
        for q in sorted(glob.glob(f"{CTRL}/fold_*.npz")):
            w = np.load(q)
            pool.update(dict(zip(w["mvt_id"], w["proba"][:, SCHED])))
        fit_p = np.array([pool.get(i, np.nan) for i in fit_id])
        ok = ~np.isnan(fit_p)
        new_p = proba[:, SCHED].copy()
        for airport in np.unique(ap):
            m = ok & (fit_ap == airport)
            if m.sum() < 5000:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(fit_p[m], fit_true[m].astype(float))
            t = ap == airport
            new_p[t] = iso.predict(proba[t, SCHED])
        p2 = proba.copy()
        rest = proba.sum(axis=1) - proba[:, SCHED]
        scale = np.where(rest > 0, (1 - new_p) / np.where(rest > 0, rest, 1), 0.0)
        p2 *= scale[:, None]
        p2[:, SCHED] = new_p
        cand = mixture(p2, values, d_lobt)

        keep = np.array([dec[m] for m in mvt])
        d = ((cand - y) ** 2 - (rebuilt - y) ** 2)
        tot += d[keep].sum()
        small += d[keep & (y <= 7200)].sum()
        lower += d[keep].sum() < 0
        print(f"fold {a}+{b}: decided {d[keep].sum() / LB:+.2f} s | y<=7200 {d[keep & (y <= 7200)].sum() / LB:+.2f} | "
              f"rows moved {(np.abs(cand - rebuilt) > 1).sum()}", flush=True)
    ok = -tot / LB >= 0.3 and lower >= 4 and -small / LB >= 0.3
    print(f"DECISION E58: {'ACCEPT' if ok else 'REJECT'} (folds 6, lower in {lower}; decided {tot / LB:+.2f} s, "
          f"y<=7200 {small / LB:+.2f} s)")


if __name__ == "__main__":
    main()
