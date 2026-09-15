"""E29: NM actual/estimated arrival ARVT_3 (unused so far; code review 15.09) added to the canonical model with E28 features.

    python src/e29_arvt3.py build      writes data/features_e29.parquet = features_e28 + arvt3 columns
    python src/e29_arvt3.py summary    decision against data/cv_preds_e28
New columns (departures with NM; null otherwise): arvt3_minus_mvt (ARVT_3 - take-off), arvt3_minus_arvt1 (ARVT_3 - ARVT_1),
flight_time_ratio = arvt3_minus_mvt / (ARVT_1 - EOBT_1), arvt3_minus_lobt_flight (ARVT_3 - LOBT - planned block-to-arrival).
Gate (REVIEW_BRIEF.md, E29): total SSE of the 6 folds lower by >= 0.25 s LB-equivalent and lower in >= 4 of 6 folds.
Runs on the home PC (Windows, Python 3.14) in parallel with the Mac.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

if sys.argv[1] == "build":
    cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ARVT_1_flt", "ARVT_3_flt", "EOBT_1_flt", "LOBT_flt"]
    files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in files]).filter(pl.col("PHASE_mvt") == "DEP")
    s = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)
    add = raw.select(
        "MVT_ID_mvt",
        arvt3_minus_mvt=s("ARVT_3_flt", "MVT_TIME_UTC_mvt"),
        arvt3_minus_arvt1=s("ARVT_3_flt", "ARVT_1_flt"),
        flight_time_ratio=s("ARVT_3_flt", "MVT_TIME_UTC_mvt") / s("ARVT_1_flt", "EOBT_1_flt").clip(600, None),
        arvt3_minus_lobt_flight=s("ARVT_3_flt", "LOBT_flt") - s("ARVT_1_flt", "EOBT_1_flt"),
    )
    feat = pl.read_parquet("data/features_e28.parquet")
    out = feat.join(add, on="MVT_ID_mvt", how="left")
    assert out.height == feat.height
    print("null share arvt3_minus_mvt", out["arvt3_minus_mvt"].null_count() / out.height)
    out.write_parquet("data/features_e29.parquet")
else:
    LB_SCALE = 6 * 2 * 273.41 * 344841
    rows = []
    for p in sorted(glob.glob("data/cv_preds_e28/fold_*.npz")):
        q = p.replace("cv_preds_e28", "cv_preds_e29")
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        assert np.array_equal(c["mvt_id"], e["mvt_id"])
        y = c["y"]
        dl = (e["pred"] - y) ** 2 - (c["pred"] - y) ** 2
        m = y <= 7200
        rows.append((os.path.basename(p), dl.sum()))
        print(f"{rows[-1][0]}: SSE change {dl.sum():+.3e}; regressor RMSE y<=7200 {np.sqrt(np.mean((c['reg'][m] - y[m]) ** 2)):.2f} -> {np.sqrt(np.mean((e['reg'][m] - y[m]) ** 2)):.2f}")
    tot = sum(r[1] for r in rows)
    lower = sum(r[1] < 0 for r in rows)
    ok = len(rows) == 6 and -tot / LB_SCALE >= 0.25 and lower >= 4
    print(f"DECISION E29: {'ACCEPT' if ok else 'REJECT'} (folds {len(rows)}, lower in {lower}, total {tot:+.3e} = {tot / LB_SCALE:+.2f} s LB-equivalent)")
