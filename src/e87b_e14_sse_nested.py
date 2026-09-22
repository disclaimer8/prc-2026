"""E87b: E87 with the split fixed (22.09). The external review found that E87 built the flight-number history
once for every month from the other eleven, BEFORE the fold split (e87_e14_sse.py:71): held-out January rows
saw July labels, and training rows saw the labels of both held-out months. E14 itself builds history inside
the fit part (e14_flight_history.py:59-63). E87b does exactly that, per fold:

    fit rows  -> out-of-month history within the ten fit months
    test rows -> history from the ten fit months only

Everything else is E87 unchanged: same features (design), population (0 < D < 4 h, n >= 3), regular candidate r
from the saved OOF of the accepted chain, logistic start, matched L2 (E87_L2=0.5). Known residual: r of the
training rows is their own OOF, fitted on months that include the held-out pair; r is not refitted here.

REGISTERED BEFORE THE RUN: standing E87 gate, SSE on applied rows lower by >= 0.30 s in gate units and in
>= 4 of 6 folds. Prediction -0.05 .. -0.40, centre -0.10. E89 (cost-priced classifier) moved the board by
-1.18 s, which raises the prior on the idea; the population is still ~1 300 rows and the E14 logistic is
calibrated there in the mean.
    .venv/bin/python src/e87b_e14_sse_nested.py
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
E = {}
exec(open("src/e87_e14_sse.py").read().split("\ndef main()")[0], E)
f, history, design, Y, WINDOW, FOLDS, LB, LAM = (E[k] for k in ("f", "history", "design", "Y", "WINDOW", "FOLDS", "LB", "LAM"))


def main() -> None:
    reg = E["regular_candidate"]()
    tot_old = tot_new = 0.0
    lower = n_rows = 0
    print(f"{'фолд':8} {'строк':>6} {'SSE логит':>11} {'SSE по SSE':>12} {'разница, с':>11}")
    for a, b in FOLDS:
        fit = f.filter(~pl.col("month").is_in([a, b]))
        tr = pl.concat([history(fit.filter(pl.col("month") != m), fit.filter(pl.col("month") == m))
                        for m in fit["month"].unique().to_list()]).filter(WINDOW).filter(pl.col("n") >= 3)
        te = history(fit, f.filter(pl.col("month").is_in([a, b]))).filter(WINDOW).filter(pl.col("n") >= 3)
        if te.height == 0:
            continue
        Xtr, Xte = design(tr), design(te)
        ytr, yte = tr[Y].to_numpy().astype(float), te[Y].to_numpy().astype(float)
        str_, ste = tr["d_mvt_sched"].to_numpy().astype(float), te["d_mvt_sched"].to_numpy().astype(float)
        rtr = np.array([reg.get(m, np.nan) for m in tr["MVT_ID_mvt"].to_list()], dtype=float)
        rte = np.array([reg.get(m, np.nan) for m in te["MVT_ID_mvt"].to_list()], dtype=float)
        ok_tr, ok_te = np.isfinite(rtr), np.isfinite(rte)
        lab = tr["sched"].to_numpy()[ok_tr]
        Xtr, ytr, str_, rtr = Xtr[ok_tr], ytr[ok_tr], str_[ok_tr], rtr[ok_tr]
        Xte, yte, ste, rte = Xte[ok_te], yte[ok_te], ste[ok_te], rte[ok_te]
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, lab)
        w0 = np.append(lr.coef_.ravel(), lr.intercept_[0])
        w = E["sse_fit"](np.column_stack([Xtr, np.ones(len(Xtr))]), ytr, rtr, str_, w0, LAM)
        p_old = lr.predict_proba(Xte)[:, 1]
        p_new = 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([Xte, np.ones(len(Xte))]) @ w, -30, 30)))
        e_old = float(np.sum((yte - (rte + p_old * (ste - rte))) ** 2))
        e_new = float(np.sum((yte - (rte + p_new * (ste - rte))) ** 2))
        tot_old += e_old; tot_new += e_new; lower += e_new < e_old; n_rows += len(yte)
        print(f"{a}+{b:<5} {len(yte):6} {e_old / LB:11.2f} {e_new / LB:12.2f} {(e_new - e_old) / LB:+11.2f}")
    delta = (tot_new - tot_old) / LB
    ok = -delta >= 0.30 and lower >= 4
    print(f"\nстрок применения {n_rows}; ИТОГО: {delta:+.2f} с, ниже в {lower} из 6")
    print(f"DECISION E87b: {'ACCEPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
