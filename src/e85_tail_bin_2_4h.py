"""E85: should the LIRF tail table also own the 2-4 h delay bin? (21.09)

Found by the 21.09 data-first pass (REVIEW_BRIEF 21y), not by a hypothesis. On decided rows, LIRF no-NM rows
with truth above 3 h are 68 rows worth **9.21 s** in gate units -- more than three times all of E77 -- and 64
of them sit in the 2-4 h delay bin, where the model predicts a median 8 440 s against a truth of 11 856.

`src/lirf_table.py` already bins exactly there (BINS has 7200 and 14400) and already estimates that bin's
regime probabilities; it simply does not apply them: OVERRIDE = ["4-8h", "8-12h", "12-30h"].

Why the outcome is not obvious. The share of "block = schedule" rows rises monotonically with the delay --
39.5 % below 2 h, **59.6 % at 2-4 h**, 80.5 % at 4-8 h, 100 % at 8-12 h -- so the table owns the bins where
the regime is nearly certain and leaves the mixed one to the model. At the median of the 2-4 h bin the model
already predicts slightly HIGH (8 440 against a truth of 7 736 with a delay of 9 178); the 9.21 s live in the
bin's tail, not at its centre. So overriding could just as well trade a tail gain for a centre loss.

REGISTERED BEFORE THE RUN. Probabilities are estimated on the ten fit months and applied to the two held-out
ones, six month-pair folds, exactly as the table itself is validated. Gate: SSE on the affected rows falls by
**>= 0.30 s** in gate units and falls in **>= 4 of 6** folds. Prediction: **-1.5 .. +1.0 s, centre -0.2** --
wide and barely negative on purpose, because the bin is a near-even mixture and the measured level error at
its centre points the wrong way.

    .venv/bin/python src/e85_tail_bin_2_4h.py
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import fold_metrics as FM  # noqa: E402

LB = 6 * 2 * 243.8347 * 344841
FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
LO, HI = 7200, 14400          # the 2-4 h bin, as lirf_table.py bins it
ALPHA, ALPHA_DAY = 1.0, 0.2   # the table's own Dirichlet pseudo-counts
DAY = 86400


def main() -> None:
    f = pl.read_parquet("data/features_e77.parquet",
                        columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt",
                                 "d_mvt_sched", "MVT_TIME_UTC_mvt", "split"])
    f = f.filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
    f = f.with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())
    bin_ = f.filter((pl.col("d_mvt_sched") >= LO) & (pl.col("d_mvt_sched") < HI))
    print(f"строк в корзине 2-4 ч: {bin_.height}")

    oof = {}
    for a, b in FOLDS:
        d = np.load(f"data/cv_preds_e77_stage/fold_{a}_{b}.npz")
        oof.update(dict(zip(d["mvt_id"].tolist(), d["pred"].tolist())))

    tot_now = tot_new = 0.0
    lower = 0
    print(f"\n{'фолд':8} {'строк':>6} {'p_sched':>8} {'SSE сейчас':>12} {'SSE с таблицей':>15} {'разница, с':>11}")
    for a, b in FOLDS:
        fit = bin_.filter(~pl.col("month").is_in([a, b]))
        te = bin_.filter(pl.col("month").is_in([a, b]))
        if te.height == 0:
            continue
        y_f, d_f = fit["TAXITIME_SEC_mvt"].to_numpy().astype(float), fit["d_mvt_sched"].to_numpy().astype(float)
        n_sched = int(np.sum(np.abs(y_f - d_f) <= 600))
        n_day = int(np.sum(np.abs(y_f - (DAY + d_f)) <= 600))
        n_oth = len(y_f) - n_sched - n_day
        tot = n_sched + ALPHA + n_day + ALPHA_DAY + n_oth + ALPHA
        p_s, p_d = (n_sched + ALPHA) / tot, (n_day + ALPHA_DAY) / tot
        p_o = 1.0 - p_s - p_d
        med_day = float(np.median(y_f[np.abs(y_f - (DAY + d_f)) <= 600])) if n_day else DAY

        y = te["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        D = te["d_mvt_sched"].to_numpy().astype(float)
        base = np.array([oof.get(m, np.nan) for m in te["MVT_ID_mvt"].to_list()], dtype=float)
        ok = np.isfinite(base)
        y, D, base = y[ok], D[ok], base[ok]
        new = p_s * D + p_d * med_day + p_o * base
        s_now, s_new = float(((base - y) ** 2).sum()), float(((new - y) ** 2).sum())
        tot_now += s_now; tot_new += s_new
        lower += s_new < s_now
        print(f"{a}+{b:<5} {len(y):6} {p_s:8.3f} {s_now / LB:12.2f} {s_new / LB:15.2f} {(s_new - s_now) / LB:+11.2f}")

    delta = (tot_new - tot_now) / LB
    ok = -delta >= 0.30 and lower >= 4
    print(f"\nИТОГО: {delta:+.2f} с, ниже в {lower} из 6")
    print(f"DECISION E85: {'ACCEPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
