"""Recompute the E16 decision lines from the saved LOMO predictions (data/e16_lomo.parquet) with the committed sign convention
(RMSE change: negative = better). The first log of 15.09 10:26 was printed by an uncommitted version with the opposite sign.
    .venv/bin/python src/e16_summary.py"""
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
res = pl.read_parquet("data/e16_lomo.parquet")
se_c = ((res["pred_control"] - res[Y]) ** 2).to_numpy()
top3 = np.argsort(-se_c)[:3]
mask = np.ones(len(se_c), bool)
mask[top3] = False
months = res["month"].to_numpy()
for arm in ("e16a", "e16b"):
    se_a = ((res[f"pred_{arm}"] - res[Y]) ** 2).to_numpy()
    wins = sum(se_a[months == m].sum() < se_c[months == m].sum() for m in np.unique(months))
    gain, gain_wo = 100 * (se_a.sum() / se_c.sum() - 1), 100 * (se_a[mask].sum() / se_c[mask].sum() - 1)
    ok = gain <= -5 and gain_wo < 0 and wins >= 8
    print(f"{arm}: LOMO-SSE ({gain:+.1f}%), without top-3 ({gain_wo:+.1f}%), months better {wins}/{len(np.unique(months))}, "
          f"LB-equivalent RMSE change {(se_a.sum() - se_c.sum()) / 6 / (2 * 274.28 * 344841):+.2f} s -> {'PASS' if ok else 'FAIL'}")
