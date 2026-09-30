"""E167 on the 2025 production OOF (journal 196): the NM-window rule through the arrival record (src/e167_nm_window_rule.py logic)
applied to data/cv_preds_e116prod (proba/values), per month and in total, A alone and A + B. Mirrors the ranking script: links
FLIGHT_mvt/ADEP/ADES to exactly one NM arrival 0-16 h after take-off; TabPFN zone (LIRF 4-30 h) and MVT - LOBT < 0 skipped.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import numpy as np
import polars as pl

import train as T

Q = 0.9
HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
raw = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))], how="diagonal_relaxed")
dep = raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_null() & pl.col("ADES_mvt").is_in(HUBS)).select(
    "MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", mvt=ep("MVT_TIME_UTC_mvt"), sched=ep("SCHED_TIME_UTC_mvt"), month=pl.col("MVT_TIME_UTC_mvt").dt.month())
arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("FLIGHT_ID_mvt").is_not_null()).select(
    "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", a_mvt=ep("MVT_TIME_UTC_mvt"), a_lobt=ep("LOBT_flt"), a_aobt=ep("AOBT_3_flt"))
link = dep.join(arr, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], how="inner").filter((pl.col("a_mvt") - pl.col("mvt")).is_between(0, 16 * 3600))
link = link.filter(pl.len().over("MVT_ID_mvt") == 1).with_columns(D=pl.col("mvt") - pl.col("sched"), gap=(pl.col("sched") - pl.col("a_lobt")).abs(),
                                                                 agap=(pl.col("a_aobt") - pl.col("a_lobt")).abs(), d_lobt=pl.col("mvt") - pl.col("a_lobt"))
link = link.filter(pl.col("a_lobt").is_not_null() & (pl.col("d_lobt") >= 0) & ~((pl.col("ADEP_mvt") == "LIRF") & pl.col("D").is_between(4 * 3600, 30 * 3600, closed="left")))
A = link.filter((pl.col("D").abs() > 3606) & (pl.col("gap") <= 3606))
B = link.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("D").abs() > 3606) & (pl.col("gap") > 3606) & (pl.col("agap") <= 3606))
print(f"2025 linked {link.height}; rule A {A.height}; rule B {B.height}")
Z = [dict(np.load(p)) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]
ids = np.concatenate([z["mvt_id"] for z in Z]); P = np.concatenate([z["proba"] for z in Z]).astype(float); V = np.concatenate([z["values"] for z in Z]).astype(float)
y = np.concatenate([z["y"] for z in Z]).astype(float); pred = np.concatenate([z["pred"] for z in Z]).astype(float)
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "d_mvt_lobt", "month"]).filter(pl.col("split") == "train")
d = pl.DataFrame({"MVT_ID_mvt": ids}).join(f.drop("split"), on="MVT_ID_mvt", how="left", maintain_order="left")
month = d["month"].to_numpy()
n = len(y); rm = np.sqrt(np.mean((pred - y) ** 2))
for name, useB in (("A", False), ("A+B", True)):
    Pn = P.copy(); iA = np.isin(ids, A["MVT_ID_mvt"].to_numpy()); iB = np.isin(ids, B["MVT_ID_mvt"].to_numpy()) if useB else np.zeros(n, bool)
    Pn[iA, 1] = 0; Pn[iA, 5] = 0
    rest = np.clip(1 - Pn[iB, 1], 1e-9, None); Pn[iB] = Pn[iB] * ((1 - Q) / rest)[:, None]; Pn[iB, 1] = Q
    Pn = np.where(np.isnan(V), 0, Pn); Pn = Pn / Pn.sum(1, keepdims=True)
    new = np.where(iA | iB, np.clip(T.lobt_clip(np.sum(Pn * np.nan_to_num(V), 1), d), 0, None), pred)
    dd = (new - y) ** 2 - (pred - y) ** 2
    per = [dd[month == m].sum() for m in range(1, 13)]
    s = np.sort(dd[iA | iB]); tr = s[3:-3].sum() if len(s) > 6 else s.sum()
    print(f"{name}: rows {int((iA | iB).sum())}; LB-equivalent {dd.sum() / 6 / n / (2 * rm):+.3f} s (6 folds cover 12 months, ranking 2); "
          f"months lower {sum(x < 0 for x in per)}/{sum(x != 0 for x in per)}; trimmed(3/3) {tr / 6 / n / (2 * rm):+.3f} s")
