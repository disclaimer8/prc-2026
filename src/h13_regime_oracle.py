"""External review 30.09 (journal 168): ceiling of perfect regime recognition on today's production OOF, before retraining the
regime classifier with the regressors' OOF as inputs. On the rows the base decides (fold_metrics.decided_map), replace the
mixture by the true regime's candidate (regular -> the regressor, dayshift -> regressor + 1 day) and measure the RMSE change of
the full 2025 set. Also the same oracle only where the candidates spread by more than 3600 s (the reviewer's target population),
and the mean probability the classifier gives to the true regime there.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import numpy as np
import polars as pl

import train as T
from fold_metrics import decided_map

FEAT = os.environ.get("FEATURES", "data/features_e106.parquet")
cols = ["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", T.TARGET, *T.CANDIDATES.values()]
df = pl.read_parquet(FEAT, columns=cols).filter(pl.col("split") == "train")
lab = pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], "label": T.regime_label(df), "ap": df["ADEP_mvt"],
                    "nm": df["FLIGHT_ID_mvt"].is_not_null()})
dm = decided_map(FEAT)
Z = [dict(np.load(p)) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]
mv = np.concatenate([z["mvt_id"] for z in Z]); y = np.concatenate([z["y"] for z in Z]).astype(float)
pred = np.concatenate([z["pred"] for z in Z]).astype(float); P = np.concatenate([z["proba"] for z in Z]).astype(float)
V = np.concatenate([z["values"] for z in Z]).astype(float)
L = pl.DataFrame({"MVT_ID_mvt": mv}).join(lab, on="MVT_ID_mvt", how="left", maintain_order="left")
k = L["label"].to_numpy(); ap = L["ap"].to_numpy(); nm = L["nm"].to_numpy()
dec = np.array([dm.get(m, True) for m in mv])
n = len(y)
orc = V[np.arange(n), k]                                   # values: [reg, sched, iobt, lobt, eobt, reg + day]
orc = np.where(np.isnan(orc), pred, orc)
spread = np.nanmax(V[:, :5], axis=1) - np.nanmin(V[:, :5], axis=1)
sse0 = float(((pred - y) ** 2).sum()); r0 = np.sqrt(sse0 / n)
def gain(mask, label):
    new = np.where(mask, orc, pred)
    d = float(((new - y) ** 2).sum()) - sse0
    print(f"{label:58s} rows {int(mask.sum()):8d}  dSSE {d / 1e9:+8.3f}e9  RMSE {r0:.3f} -> {np.sqrt((sse0 + d) / n):.3f} ({np.sqrt((sse0 + d) / n) - r0:+.3f} s)")
print(f"production OOF (cv_preds_e116prod), 2025, rows {n}, RMSE {r0:.3f}; decided {int(dec.sum())}")
gain(dec, "oracle regime, all decided rows")
gain(dec & (spread > 3600), "oracle regime, decided rows with candidate spread > 3600")
gain(dec & (spread > 3600) & (k != 0), "  ... of which the truth is a filed/dayshift regime")
gain(dec & (spread > 3600) & (k == 0), "  ... of which the truth is regular taxi")
gain(dec & (spread <= 3600), "oracle regime, decided rows with spread <= 3600")
for cls in range(6):
    gain(dec & (k == cls), f"  truth = {T.CLASSES[cls]}")
p_true = P[np.arange(n), k]
m = dec & (spread > 3600)
print(f"spread > 3600 rows: truth regular {np.mean(k[m] == 0):.3f}; mean P(true regime) {p_true[m].mean():.3f}; "
      f"where truth is filed: mean P(true) {p_true[m & (k != 0)].mean():.3f}, median {np.median(p_true[m & (k != 0)]):.3f}")
print("by airport x NM (oracle on decided rows, spread > 3600): dSSE e9")
g = pl.DataFrame({"ap": ap, "nm": nm, "d": np.where(m, (orc - y) ** 2 - (pred - y) ** 2, 0.0)}).group_by("ap", "nm").agg(
    dsse=(pl.col("d").sum() / 1e9)).sort("dsse").head(10)
print(g)
