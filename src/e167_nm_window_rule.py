"""E167 (journal 192-193; Fable review 3.1 = E21-A + E21-B): the NM matching-window rule through the arrival record, on a parts file.

    .venv/bin/python src/e167_nm_window_rule.py <parts version> <new version>

A ranking departure without NM whose flight (FLIGHT_mvt, ADEP, ADES) has exactly one NM-matched arrival at our hub 0-16 h after
take-off gets that plan's LOBT / AOBT_3. The window |BLOCK - LOBT| <= 3606 s holds on 100 % of 2 062 577 matched rows, so an
unmatched departure's BLOCK lies OUTSIDE its plan's window:
  A: |D| > 3606 and |SCHED - LOBT| <= 3606  ->  BLOCK != SCHED: P(sched) = P(day shift) = 0, renormalised (2025 LIRF 0/68);
  B: LIRF only, |D| > 3606, |SCHED - LOBT| > 3606 and |AOBT_3 - LOBT| <= 3606  ->  the real off-block was inside, BLOCK is not: P(sched) = Q
     (0.9), the rest scaled (2025 LIRF 21/23). Rows with MVT - LOBT < 0 (plan after take-off) are skipped.
LIRF rows with 4 h <= D < 30 h (TabPFN zone) are left alone. Writes parts_v<new>.npz and gentle-igloo_v<new>.parquet (the mixture
recomputed with the LOBT clip, as blend.py), every other row unchanged.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import train as T  # noqa: E402

src_v, new_v = int(sys.argv[1]), int(sys.argv[2])
Q = float(os.environ.get("E167_Q", "0.9"))
USE_B = os.environ.get("E167_B", "1") == "1"
HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
Y = "TAXITIME_SEC_mvt"
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet")
dep = rk.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_null() & pl.col("ADES_mvt").is_in(HUBS)).select(
    "MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", mvt=ep("MVT_TIME_UTC_mvt"), sched=ep("SCHED_TIME_UTC_mvt"))
arr = rk.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("FLIGHT_ID_mvt").is_not_null()).select(
    "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", a_mvt=ep("MVT_TIME_UTC_mvt"), a_lobt=ep("LOBT_flt"), a_aobt=ep("AOBT_3_flt"))
link = dep.join(arr, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], how="inner").filter((pl.col("a_mvt") - pl.col("mvt")).is_between(0, 16 * 3600))
link = link.filter(pl.len().over("MVT_ID_mvt") == 1).with_columns(
    D=pl.col("mvt") - pl.col("sched"), gap=(pl.col("sched") - pl.col("a_lobt")).abs(), agap=(pl.col("a_aobt") - pl.col("a_lobt")).abs(),
    d_lobt=pl.col("mvt") - pl.col("a_lobt"))
link = link.filter(pl.col("a_lobt").is_not_null() & (pl.col("d_lobt") >= 0) &
                   ~((pl.col("ADEP_mvt") == "LIRF") & pl.col("D").is_between(4 * 3600, 30 * 3600, closed="left")))
ruleA = link.filter((pl.col("D").abs() > 3606) & (pl.col("gap") <= 3606))
ruleB = link.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("D").abs() > 3606) & (pl.col("gap") > 3606) & (pl.col("agap") <= 3606)) if USE_B else link.head(0)  # 2025: 21/23 at D > 3606, only 8/11 below
print(f"linked no-NM departures {link.height} (by airport {dict(link.group_by('ADEP_mvt').len().rows())}); rule A rows {ruleA.height}, rule B rows {ruleB.height}")

z = dict(np.load(f"submissions/parts_v{src_v}.npz"))
ids = z["mvt_id"]; P = z["proba"].astype(float).copy(); V = z["values"].astype(float)
iA = np.isin(ids, ruleA["MVT_ID_mvt"].to_numpy()); iB = np.isin(ids, ruleB["MVT_ID_mvt"].to_numpy())
SCHED, DAY = 1, 5
before = P.copy()
P[iA, SCHED] = 0.0; P[iA, DAY] = 0.0
rest = np.clip(1.0 - P[iB, SCHED], 1e-9, None)
P[iB] = P[iB] * ((1.0 - Q) / rest)[:, None]; P[iB, SCHED] = Q
P = np.where(np.isnan(V), 0.0, P); P = P / P.sum(axis=1, keepdims=True)
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "rank")
d = pl.DataFrame({"MVT_ID_mvt": ids}).join(f.drop("split"), on="MVT_ID_mvt", how="left", maintain_order="left")
pred = np.clip(T.lobt_clip(np.sum(P * np.nan_to_num(V), axis=1), d), 0, None)
chg = iA | iB
old = z["pred"].astype(float)
print(f"P(sched) mass removed by A {before[iA, SCHED].sum():.2f} (+ day {before[iA, DAY].sum():.2f}); B mean P(sched) {before[iB, SCHED].mean() if iB.any() else 0:.2f} -> {Q}")
print(f"changed rows {int((np.abs(pred - old) > 0.5).sum())}; RMS change on them {np.sqrt(np.mean((pred - old)[chg] ** 2)) if chg.any() else 0:.0f} s")
z.update(proba=P, pred=np.where(chg, pred, old))
np.savez(f"submissions/parts_v{new_v}.npz", **z)
sub = pl.read_parquet(f"submissions/gentle-igloo_v{src_v}.parquet")
upd = pl.DataFrame({"MVT_ID_mvt": ids[chg], "new": np.round(pred[chg])})
out = sub.join(upd, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce("new", Y).cast(sub.schema[Y]).alias(Y)).select(sub.columns)
assert out.height == 344841 and out[Y].null_count() == 0
out.write_parquet(f"submissions/gentle-igloo_v{new_v}.parquet")
print(f"wrote v{new_v} and parts_v{new_v}")
