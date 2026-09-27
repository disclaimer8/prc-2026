import glob
import numpy as np
import polars as pl

raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                             "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "pred": z["pred"]})
                 for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    mvt_aobt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64),
    bs=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6),
    fp=pl.col("FLIGHT_ID_mvt").is_not_null(), nm=pl.col("AOBT_3_flt").is_not_null())
d = d.with_columns(err=pl.col("pred") - pl.col("y"))
N_TR, tot = d.height, float((d["err"] ** 2).sum())
rm = np.sqrt(tot / N_TR)
print(f"train DEP {N_TR}, OOF RMSE {rm:.1f}")
L = d.filter(pl.col("ADEP_mvt") == "LIRF")
j = L.filter((pl.col("month") == 7) & pl.col("fp") & pl.col("nm") & pl.col("bs") & (pl.col("y") > 3600))
print(f"\nclaim set: LIRF July, FLIGHT_ID present, AOBT_3 present, block=SCHED(+-6 s), y>3600: {j.height} rows")
print(j.select(med_mvt_aobt_min=(pl.col("mvt_aobt").median() / 60).round(1), q10=(pl.col("mvt_aobt").quantile(.1) / 60).round(1),
               q90=(pl.col("mvt_aobt").quantile(.9) / 60).round(1), med_y=pl.col("y").median(), med_pred=pl.col("pred").median(),
               mean_pred=pl.col("pred").mean().round(0), med_err=pl.col("err").median().round(0), sse=(pl.col("err") ** 2).sum()))
sse = float((j["err"] ** 2).sum())
print(f"their SSE {sse:.3g}; if ALL were fixed, train OOF RMSE {rm:.2f} -> {np.sqrt((tot - sse) / N_TR):.2f} "
      f"(board-scale via the text's formula with N=344841: {sse / (344841 * 2 * 240):.2f} s, but these rows are 2025 training rows)")
print("\nsame definition, all months:")
print(L.filter(pl.col("fp") & pl.col("nm") & pl.col("bs") & (pl.col("y") > 3600)).group_by("month").agg(
    n=pl.len(), med_pred=pl.col("pred").median().round(0), med_y=pl.col("y").median()).sort("month"))
print("\nprecision problem: LIRF rows with NM, observable D = MVT - SCHED in 1 h ... 2 h (where the claim set lives):")
band = L.filter(pl.col("nm") & pl.col("D").is_between(3600, 7200))
print(band.group_by("month").agg(n=pl.len(), share_block_sched=pl.col("bs").mean().round(3), mean_y=pl.col("y").mean().round(0),
                                 mean_pred=pl.col("pred").mean().round(0), mean_err=pl.col("err").mean().round(0)).sort("month"))
print(band.select(n=pl.len(), share_block_sched=pl.col("bs").mean().round(3), mean_y=pl.col("y").mean().round(0),
                  mean_pred=pl.col("pred").mean().round(0), mean_err=pl.col("err").mean().round(1)))
print("\ncalibration of the model on LIRF NM rows by D band (mean residual):")
print(L.filter(pl.col("nm")).with_columns(b=pl.col("D").cut([0, 1800, 3600, 7200, 14400])).group_by("b").agg(
    n=pl.len(), share_bs=pl.col("bs").mean().round(3), mean_y=pl.col("y").mean().round(0), mean_pred=pl.col("pred").mean().round(0),
    mean_err=pl.col("err").mean().round(1)).sort("b"))
