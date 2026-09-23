"""Compare two OOF arms AFTER applying E14 to both, as the submission chain does (23.09, hypothesis 24).

    PYTHONPATH=src .venv/bin/python src/post_e14.py data/cv_preds_ctrl data/cv_preds_cand
Why: E106 won -1.41 s on decided rows in CV, but applying the E14 flight-history logit to both arms (the chain applies
it to LIRF departures without NM, 0 < D < 4 h, flight history >= 3 rows) cut that to -0.58 s: the two carry part of the
same information. Any change that touches the no-NM Rome population must be judged on this leg too.
Prints decided / trimmed after E14 per fold and in total, in the same LB-equivalent seconds as fold_metrics.compare.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("FEATURES", "data/features_e77.parquet")
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import fold_metrics as FM  # noqa: E402

_src = open(os.path.join(os.path.dirname(__file__), "e14_flight_history.py")).read().split("\nresults = []")[0]
_ns: dict = {}
exec(compile(_src, "e14_flight_history.py", "exec"), _ns)
F, HISTORY, DESIGN = _ns["f"], _ns["history"], _ns["design"]


def after_e14(arm: str, a: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(f"{arm}/fold_{a}_{b}.npz")
    base = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "reg": z["reg"], "y": z["y"]})
    fit = F.filter(~pl.col("month").is_in([a, b]))
    parts = [HISTORY(fit.filter(pl.col("month") != m), fit.filter(pl.col("month") == m)) for m in fit["month"].unique().to_list()]
    band = (pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400)
    fit_h = pl.concat(parts).filter(band)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(DESIGN(fit_h), fit_h["sched"].to_numpy())
    val = HISTORY(fit, F.filter(pl.col("month").is_in([a, b]))).filter(band).join(base, on="MVT_ID_mvt", how="inner")
    p = model.predict_proba(DESIGN(val))[:, 1]
    new = np.where(val["n"].to_numpy() >= 3, p * val["d_mvt_sched"].to_numpy() + (1 - p) * val["reg"].to_numpy(), val["pred"].to_numpy())
    upd = dict(zip(val["MVT_ID_mvt"].to_list(), new))
    pred = np.array([upd.get(m, q) for m, q in zip(z["mvt_id"], z["pred"])])
    return z["mvt_id"], pred, z["y"]


def compare(ctrl: str, cand: str, lb_rmse: float = 242.2622) -> tuple[float, float, int]:
    dec = FM.decided_map()
    lb = 6 * 2 * lb_rmse * 344841
    tot = trim = 0.0
    lower = 0
    for a in range(1, 7):
        b = a + 6
        if not (os.path.exists(f"{ctrl}/fold_{a}_{b}.npz") and os.path.exists(f"{cand}/fold_{a}_{b}.npz")):
            continue
        ids, pc, y = after_e14(ctrl, a, b)
        ids2, pe, _ = after_e14(cand, a, b)
        o = {m: i for i, m in enumerate(ids2)}
        pe = pe[np.array([o[m] for m in ids])]
        m = np.array([dec[k] for k in ids])
        d = ((pe - y) ** 2 - (pc - y) ** 2)[m]
        s = np.sort(d)
        tot += d.sum() / lb
        trim += (d.sum() - s[:3].sum() - s[-3:].sum()) / lb
        lower += d.sum() < 0
        print(f"fold {a}+{b}: decided after E14 {d.sum() / lb:+.3f} | trimmed {(d.sum() - s[:3].sum() - s[-3:].sum()) / lb:+.3f}", flush=True)
    print(f"AFTER E14: decided {tot:+.3f}, trimmed {trim:+.3f}, lower {lower}/6")
    return tot, trim, lower


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
