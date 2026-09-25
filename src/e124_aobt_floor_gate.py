"""E124 gate (25.09, idea from team likable-river, hypothesis 61): AOBT_3-anchored floor on the prediction.

Rule: where MVT - AOBT_3 > 1800 s, pred = max(pred, (MVT - AOBT_3) - 3600). Their board attribution: -20.3 s (v5 336 -> v10 316),
on a chain without our LOBT window clip. Ours already clips to |BLOCK - LOBT| <= 3606; this anchors on the NM actual instead.
Measured here on the 2025 OOF of the E106 configuration (data/cv_preds_e106_air): how often the truth itself violates the floor,
and what applying the floor does to SSE, per fold and on y <= 7200, plus the LB-equivalent (6-fold SSE / 6 over 344 841 rows).
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

CV_DIR = os.environ.get("CV_DIR", "data/cv_preds_e106_air")
GAP = float(os.environ.get("FLOOR_GAP", "3600"))
MIN_D = float(os.environ.get("FLOOR_MIN_D", "1800"))
f = pl.read_parquet(os.environ.get("FEATURES", "data/features_e106.parquet"), columns=["MVT_ID_mvt", "split", "ADEP_mvt", "d_mvt_aobt", "d_mvt_lobt", "month"]).filter(pl.col("split") == "train")
tot = {"sse_before": 0.0, "sse_after": 0.0, "n": 0, "changed": 0, "viol": 0, "elig": 0}
per_ap = {}
print(f"floor: pred >= (MVT-AOBT_3) - {GAP:.0f} where MVT-AOBT_3 > {MIN_D:.0f}")
for p in sorted(glob.glob(f"{CV_DIR}/fold_*.npz")):
    d = np.load(p)
    o = pl.DataFrame({"MVT_ID_mvt": d["mvt_id"].astype(np.int64), "pred": d["pred"].astype(np.float64), "y": d["y"].astype(np.float64)}).with_columns(pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"]))
    j = o.join(f, on="MVT_ID_mvt", how="left")
    da = j["d_mvt_aobt"].to_numpy(); pred = j["pred"].to_numpy(); y = j["y"].to_numpy(); ap = j["ADEP_mvt"].to_numpy()
    elig = np.isfinite(da) & (da > MIN_D)
    floor = da - GAP
    viol = elig & (y < floor)
    new = np.where(elig, np.maximum(pred, floor), pred)
    ch = new != pred
    sse_b = float(((pred - y) ** 2).sum()); sse_a = float(((new - y) ** 2).sum())
    m7 = y <= 7200
    sse_b7 = float(((pred - y)[m7] ** 2).sum()); sse_a7 = float(((new - y)[m7] ** 2).sum())
    print(f"{os.path.basename(p)}: rows {len(y)}, eligible {int(elig.sum())}, truth violates floor {int(viol.sum())} ({100*viol.sum()/max(elig.sum(),1):.2f}% of eligible), "
          f"preds changed {int(ch.sum())}, SSE {sse_b:.4e} -> {sse_a:.4e} ({sse_a-sse_b:+.3e}), y<=7200: {sse_a7-sse_b7:+.3e}")
    for k, v in (("sse_before", sse_b), ("sse_after", sse_a), ("n", len(y)), ("changed", int(ch.sum())), ("viol", int(viol.sum())), ("elig", int(elig.sum()))):
        tot[k] += v
    for a in np.unique(ap[ch]):
        m = ch & (ap == a)
        per_ap[a] = per_ap.get(a, 0.0) + float((((new - y) ** 2)[m] - ((pred - y) ** 2)[m]).sum())
delta = tot["sse_after"] - tot["sse_before"]
lb = 239.9169
lb_eq = np.sqrt(max(lb ** 2 + delta / 6 / 344841, 0)) - lb
print(f"TOTAL: eligible {tot['elig']}, truth violations {tot['viol']}, preds changed {tot['changed']}, dSSE {delta:+.3e} -> LB-equivalent {lb_eq:+.3f} s")
print("dSSE by airport (changed rows):", {k: f"{v:+.2e}" for k, v in sorted(per_ap.items(), key=lambda kv: kv[1])})
