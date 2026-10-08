"""Label-free audit of no-NM departures on disruption windows in the 2026 final inputs (E209, journal №395).

    .venv/bin/python analysis/nonm_disruption_floor_20261008/audit_2026.py <scratch dir with level_2026.parquet> <prediction parquet> [months]

Prediction parquet: MVT_ID_mvt + TAXITIME_SEC_mvt (a submission file). Jan/Jul: the v12090 canary
`data/final_tail_dryrun_janjul_20261008/work/gentle-igloo_v12090.parquet`. Feb/Jun hook: pass the final candidate
`data/<final work>/submissions/gentle-igloo_final.parquet` (scripts/final_tail_assembly_20261008.py, mode final) and `2,6`.
Reads MVT/AOBT_3/ADEP/SCHED only (live_level.py); never BLOCK/TAXITIME of the ranking rows. Rule parameters fixed in №395.
"""
import sys
import polars as pl

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220)
S, PRED = sys.argv[1], sys.argv[2]
MONTHS = [int(m) for m in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
THR, NMIN, F = 1500.0, 5, 0.95
l26 = pl.read_parquet(f"{S}/level_2026.parquet")
hot = (pl.col("level") > THR) & (pl.col("n_nb") >= NMIN)
print("== 2026 no-NM rows by month: live level > 1500 s with >= 5 NM neighbours in +-30 min ==")
print(l26.filter(~pl.col("has_nm")).group_by("month").agg(n=pl.len(), hot=hot.sum(), hot_nonlirf=(hot & (pl.col("ADEP_mvt") != "LIRF")).sum()).sort("month"))
p = pl.read_parquet(PRED, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"]).rename({"TAXITIME_SEC_mvt": "pred"})
j = l26.join(p, on="MVT_ID_mvt", how="inner")
if MONTHS:
    j = j.filter(pl.col("month").is_in(MONTHS))
print("joined rows", j.height, "months", sorted(j["month"].unique().to_list()))
t = j.filter(~pl.col("has_nm") & hot & (pl.col("ADEP_mvt") != "LIRF")).with_columns(
    floor=(F * pl.col("level")).round(0), lift=(F * pl.col("level") - pl.col("pred")).clip(0, None))
print("== no-NM hot rows (non-LIRF): prediction vs live level ==")
print(t.select(n=pl.len(), below_floor=(pl.col("pred") < pl.col("floor")).sum(), pred_med=pl.col("pred").median(), level_med=pl.col("level").median(),
               ratio_med=(pl.col("pred") / pl.col("level")).median(), lift_sum=pl.col("lift").sum(),
               lift_mean_touched=pl.col("lift").filter(pl.col("lift") > 0).mean(), lift_max=pl.col("lift").max()))
print(t.group_by("ADEP_mvt", "day").agg(n=pl.len(), below=(pl.col("pred") < pl.col("floor")).sum(), pred_med=pl.col("pred").median(),
                                       level_med=pl.col("level").median().round(0), lift_sum=pl.col("lift").sum()).sort("day", "ADEP_mvt"))
print("== ratio pred/level by D bucket ==")
print(t.with_columns(Db=pl.when(pl.col("D") < 3606).then(pl.lit("D<1h")).when(pl.col("D") < 4 * 3600).then(pl.lit("1-4h")).otherwise(pl.lit(">=4h")))
      .group_by("Db").agg(n=pl.len(), below=(pl.col("pred") < pl.col("floor")).sum(), ratio_med=(pl.col("pred") / pl.col("level")).median(), lift_sum=pl.col("lift").sum()).sort("Db"))
w = j.filter(pl.col("has_nm") & hot).with_columns(r=pl.col("pred") / pl.col("ta"))
print("== NM rows inside hot windows: pred / (MVT - AOBT_3) (label-free witness) ==")
print(w.select(n=pl.len(), r_med=pl.col("r").median(), r_q25=pl.col("r").quantile(.25), r_q75=pl.col("r").quantile(.75)))
