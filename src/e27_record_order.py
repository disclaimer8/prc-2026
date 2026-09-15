"""E27: record-ordering features (competitor idea, probe src/record_order_probe.py) added to the canonical model without `month`.

    .venv/bin/python src/e27_record_order.py build      writes data/features_e27.parquet = features_final + id_dev, fid_dev
    .venv/bin/python src/e27_record_order.py summary    decision against data/cv_preds_nomonth
id_dev: (MVT_ID - rolling median) / rolling (p90 - p10) over +-250 movements of the same airport, phase and calendar month,
sorted by MVT_TIME; fid_dev: the same for FLIGHT_ID over rows that have it (null otherwise). Computed identically for the
2025 files and ranking. Folds via src/cv_fold_preds.py with FEATURES=data/features_e27.parquet CV_OUT=data/cv_preds_e27.
Gate (REVIEW_BRIEF.md, E27): total SSE of the 6 folds lower by >= 0.25 s LB-equivalent and lower in >= 4 of 6 folds.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

W = 501


def dev(col: str, keys: list[str]) -> list[pl.Expr]:
    q = lambda p: pl.col(col).rolling_quantile(p, window_size=W, center=True, min_samples=50).over(keys)
    return [((pl.col(col) - q(0.5)) / (q(0.9) - q(0.1)).clip(1, None))]


def record_order(d: pl.DataFrame) -> pl.DataFrame:
    d = d.with_columns(ap=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
                       m=pl.col("MVT_TIME_UTC_mvt").dt.month()).sort("ap", "PHASE_mvt", "m", "MVT_TIME_UTC_mvt")
    keys = ["ap", "PHASE_mvt", "m"]
    a = d.with_columns(id_dev=dev("MVT_ID_mvt", keys)[0])
    f = d.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).with_columns(fid_dev=dev("FLIGHT_ID_mvt", keys)[0]).select("MVT_ID_mvt", "fid_dev")
    return a.join(f, on="MVT_ID_mvt", how="left").filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "id_dev", "fid_dev")


if sys.argv[1] == "build":
    cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
    files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
    ro = pl.concat([record_order(pl.read_parquet(f, columns=cols)) for f in files])
    feat = pl.read_parquet("data/features_final.parquet")
    out = feat.join(ro, on="MVT_ID_mvt", how="left")
    assert out.height == feat.height
    print("id_dev null share", out["id_dev"].null_count() / out.height, "fid_dev null share", out["fid_dev"].null_count() / out.height)
    out.write_parquet("data/features_e27.parquet")
else:
    LB_SCALE = 6 * 2 * 273.41 * 344841
    rows = []
    for p in sorted(glob.glob("data/cv_preds_nomonth/fold_*.npz")):
        q = p.replace("cv_preds_nomonth", "cv_preds_e27")
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        assert np.array_equal(c["mvt_id"], e["mvt_id"])
        y = c["y"]
        dl = (e["pred"] - y) ** 2 - (c["pred"] - y) ** 2
        rows.append((os.path.basename(p), dl.sum(), np.sort(dl)[:3].sum()))
        print(f"{rows[-1][0]}: SSE change {dl.sum():+.3e}, best 3 rows {rows[-1][2]:+.3e}")
    tot = sum(r[1] for r in rows)
    lower = sum(r[1] < 0 for r in rows)
    ok = len(rows) == 6 and -tot / LB_SCALE >= 0.25 and lower >= 4
    print(f"DECISION E27: {'ACCEPT' if ok else 'REJECT'} (folds {len(rows)}, lower in {lower}, total {tot:+.3e} = {tot / LB_SCALE:+.2f} s LB-equivalent; "
          f"without best 3 rows per fold {(tot - sum(r[2] for r in rows)) / LB_SCALE:+.2f} s)")
