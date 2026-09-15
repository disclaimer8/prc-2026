"""E32: ADS-B surface-track timings as features of the main model instead of (or before) a second stage (15.09).

    .venv/bin/python src/e32_adsb_in_model.py build     writes data/features_e32.parquet = features_e28 + adsb_* columns
    .venv/bin/python src/e32_adsb_in_model.py summary   decision
Columns from offblock2 (call-sign match only, as E25): adsb_coverage, adsb_mvt_first, adsb_mvt_first_move, adsb_mvt_move,
adsb_mvt_to, adsb_aobt_first_move, adsb_aobt_move, adsb_lobt_first_move, adsb_stat_total, adsb_n_stops, adsb_n_pts
(times relative to take-off, AOBT_3, LOBT; null without a track).
Candidate: canonical model (no month) on features_e32 -> data/cv_preds_e32, then the E30 stage trained on its out-of-fold
predictions -> data/cv_preds_e32_e30. Control: E28 model + E30 stage -> data/cv_preds_e28_e30 (src/e30_adsb_stage_v2.py cv with
E30_BASE_DIR / E30_SAVE_DIR). Both stages are trained per fold on the other 10 months.
Gate (REVIEW_BRIEF.md, E32): total SSE of the candidate lower than the control by >= 0.5 s LB-equivalent and lower in >= 4 of 6.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

if sys.argv[1] == "build":
    o = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/offblock2_*.parquet"))])
    cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt"]
    files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
    raw = pl.concat([pl.read_parquet(f, columns=cols) for f in files]).filter(pl.col("PHASE_mvt") == "DEP")
    e = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
    add = raw.join(o, on="MVT_ID_mvt", how="inner").filter(pl.col("coverage") >= 2).select(
        "MVT_ID_mvt",
        adsb_coverage=pl.col("coverage").cast(pl.Float64),
        adsb_mvt_first=e("MVT_TIME_UTC_mvt") - pl.col("t_first"),
        adsb_mvt_first_move=e("MVT_TIME_UTC_mvt") - pl.col("t_first_move"),
        adsb_mvt_move=e("MVT_TIME_UTC_mvt") - pl.col("t_move"),
        adsb_mvt_to=e("MVT_TIME_UTC_mvt") - pl.col("t_to"),
        adsb_aobt_first_move=e("AOBT_3_flt") - pl.col("t_first_move"),
        adsb_aobt_move=e("AOBT_3_flt") - pl.col("t_move"),
        adsb_lobt_first_move=e("LOBT_flt") - pl.col("t_first_move"),
        adsb_stat_total=pl.col("stat_total"), adsb_n_stops=pl.col("n_stops").cast(pl.Float64), adsb_n_pts=pl.col("n_pts").cast(pl.Float64),
    )
    feat = pl.read_parquet("data/features_e28.parquet")
    out = feat.join(add, on="MVT_ID_mvt", how="left")
    assert out.height == feat.height
    print("rows with ADS-B:", out["adsb_coverage"].is_not_null().sum(), "of", out.height)
    out.write_parquet("data/features_e32.parquet")
else:
    LB_SCALE = 6 * 2 * 258.18 * 344841
    rows = []
    for p in sorted(glob.glob("data/cv_preds_e28_e30/fold_*.npz")):
        q = p.replace("cv_preds_e28_e30", "cv_preds_e32_e30")
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        mc = dict(zip(c["mvt_id"], c["pred"]))
        yc = dict(zip(c["mvt_id"], c["y"]))
        ids = e["mvt_id"]
        keep = np.array([i in mc for i in ids])
        y = e["y"][keep]
        ctrl = np.array([mc[i] for i in ids[keep]])
        dl = (e["pred"][keep] - y) ** 2 - (ctrl - y) ** 2
        rows.append((os.path.basename(p), dl.sum()))
        print(f"{rows[-1][0]}: rows {keep.sum()}, SSE change vs E28+E30 {dl.sum():+.3e}")
    tot = sum(r[1] for r in rows)
    lower = sum(r[1] < 0 for r in rows)
    ok = len(rows) == 6 and -tot / LB_SCALE >= 0.5 and lower >= 4
    print(f"DECISION E32: {'ACCEPT' if ok else 'REJECT'} (folds {len(rows)}, lower in {lower}, total {tot:+.3e} = {tot / LB_SCALE:+.2f} s LB-equivalent)")
