"""E98 (23.09): correct the departure prediction by the LEVEL SHIFT that arrivals reveal in the same months.

The model never sees `month` (DROP_PREFIX drops it on purpose, so 2025 artefacts cannot leak), therefore it
predicts a seasonal average. Arrivals do not hide their taxi time in any month, including the two ranking
months of 2026 -- 344 693 labelled rows. If a month runs slow at an airport, arrivals show it, and the 2025
data says how much of that carries to departures: per-airport correlation of monthly medians is +0.82 EDDF,
+0.76 EDDM, +0.72 LEMD, +0.68 LFPG, +0.57 LIRF, and negative only at EHAM and LTFM.

Correction, per fold and per airport, fitted only on the fit months:
    slope_ap  = OLS slope of monthly median departure taxi on monthly median arrival taxi (fit months)
    delta_ap  = median arrival taxi of the held-out months - median arrival taxi of the fit months
    shift     = slope_ap * delta_ap, applied to every departure prediction of that airport
Airports whose fit-month correlation is below MIN_CORR keep shift 0 -- the relation must be visible where it
is fitted, not assumed. This is calibration of a fixed transform against an observation, the class that has
transferred best (E46: CV -0.40 -> board -1.00).

REGISTERED BEFORE THE RUN. Gate: the standing one on decided rows against data/cv_preds_e89_cand.
Prediction: -0.10 .. -0.80 s, centre -0.30. Slot only at -0.5 or better (rule 21ao).
    .venv/bin/python src/e98_arrival_drift.py [MIN_CORR]
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from fold_metrics import compare  # noqa: E402

Y = "TAXITIME_SEC_mvt"
FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
MIN_CORR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
SRC = "data/cv_preds_e89_cand"
OUT = f"data/cv_preds_e98_drift{int(MIN_CORR * 100)}"


def monthly() -> pl.DataFrame:
    cols = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", Y]
    tr = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
    tr = tr.with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                         ap=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")))
    tr = tr.filter(pl.col(Y) <= 7200)
    return (tr.group_by("ap", "month", "PHASE_mvt").agg(med=pl.col(Y).median())
            .pivot(values="med", index=["ap", "month"], on="PHASE_mvt").drop_nulls())


def main() -> None:
    w = monthly()
    ids = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt"]) for f in
                     sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).unique(subset="MVT_ID_mvt")
    ap_of = dict(zip(ids["MVT_ID_mvt"].to_list(), ids["ADEP_mvt"].to_list()))
    os.makedirs(OUT, exist_ok=True)
    for a, b in FOLDS:
        z = np.load(f"{SRC}/fold_{a}_{b}.npz")
        pred = z["pred"].astype(float).copy()
        aps = np.array([ap_of.get(m, "") for m in z["mvt_id"]])
        shifts = {}
        for ap, g in w.group_by("ap"):
            ap = ap[0]
            fit = g.filter(~pl.col("month").is_in([a, b]))
            te = g.filter(pl.col("month").is_in([a, b]))
            if fit.height < 6 or te.height == 0:
                continue
            x, y = fit["ARR"].to_numpy(), fit["DEP"].to_numpy()
            c = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(c) or c < MIN_CORR:
                shifts[ap] = 0.0
                continue
            slope = float(np.polyfit(x, y, 1)[0])
            shifts[ap] = slope * float(te["ARR"].mean() - x.mean())
        for ap, s in shifts.items():
            if s:
                pred[aps == ap] += s
        np.savez(f"{OUT}/fold_{a}_{b}.npz", mvt_id=z["mvt_id"], pred=np.clip(pred, 0, None), y=z["y"],
                 reg=z["reg"], proba=z["proba"], values=z["values"])
        print(f"fold {a}+{b}: сдвиги " + ", ".join(f"{k} {v:+.0f}" for k, v in sorted(shifts.items()) if v), flush=True)
    dec, tot, small, trim, lower, n, strict = compare(SRC, OUT, lb_rmse=242.6556)
    ok = n == 6 and -dec >= 0.3 and lower >= 4 and -trim >= 0.3 and -small >= 0.3
    print(f"DECISION E98 (MIN_CORR={MIN_CORR}): {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; "
          f"decided {dec:+.2f} s, strict {strict:+.2f} s, all {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")


if __name__ == "__main__":
    main()
