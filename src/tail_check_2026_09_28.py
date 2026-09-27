import glob
import numpy as np
import polars as pl

S = 1_000_000
raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                             "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
d = raw.with_columns(
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
    day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    nm=pl.col("AOBT_3_flt").is_not_null(),
    mvt_aobt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds() / 60,
    blk_aobt=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds() / 60,
    blk_sched=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6),
)
oof = []
for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz")):
    z = np.load(f)
    oof.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"]}))
d = d.join(pl.concat(oof).with_columns(pl.col("MVT_ID_mvt").cast(d["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
d = d.with_columns(err=pl.col("pred") - pl.col("y"))
tot_sse = float((d["err"] ** 2).sum())
print(f"DEP rows {d.height}, with OOF {int(d['pred'].is_not_null().sum())}, overall RMSE {np.sqrt(tot_sse / d.height):.1f}")

t = d.filter(pl.col("y") > 3600)
print(f"\n1) tail y>3600: {t.height} rows, SSE share {float((t['err']**2).sum()) / tot_sse:.1%}")
print(t.group_by("ADEP_mvt").agg(n=pl.len(), nm=pl.col("nm").mean().round(2), blk_sched=pl.col("blk_sched").mean().round(2),
                                 sse_share=((pl.col("err") ** 2).sum() / tot_sse).round(4)).sort("n", descending=True))

print("\n2) top airport-days in the tail:")
print(t.group_by("ADEP_mvt", "day").agg(n=pl.len(), nm=pl.col("nm").mean().round(2), blk_sched=pl.col("blk_sched").mean().round(2))
      .sort("n", descending=True).head(15))

print("\n3) claimed clusters (tail rows only):")
for ap, dd in [("LFPG", "2025-01-17"), ("LFPG", "2025-01-21"), ("EHAM", "2025-01-05"), ("EDDF", "2025-01-05")]:
    c = d.filter((pl.col("ADEP_mvt") == ap) & (pl.col("day") == pl.lit(dd).str.to_date()))
    ct = c.filter(pl.col("y") > 3600)
    cn = ct.filter(pl.col("nm"))
    print(f"{ap} {dd}: day DEP {c.height}, tail {ct.height} (NM {cn.height}); tail median y {ct['y'].median()} s; "
          f"NM tail: median MVT-AOBT3 {cn['mvt_aobt'].median()} min, median BLOCK-AOBT3 {cn['blk_aobt'].median()} min; "
          f"block=SCHED {ct['blk_sched'].mean()}; OOF median pred {ct['pred'].median()} s, median err {ct['err'].median()} s")

print("\n4) type 1 mechanism across the tail with NM, outside LIRF: does AOBT_3 lag the block time?")
tn = t.filter(pl.col("nm") & (pl.col("ADEP_mvt") != "LIRF"))
print(tn.select(n=pl.len(), med_mvt_aobt=pl.col("mvt_aobt").median(), med_blk_aobt=pl.col("blk_aobt").median(),
                share_aobt_after_block_20min=(pl.col("blk_aobt") < -20).mean(), med_err=pl.col("err").median(),
                sse_share=(pl.col("err") ** 2).sum() / tot_sse))
print("same, among those where AOBT_3 is > 20 min after BLOCK (the claimed lag):")
lag = tn.filter(pl.col("blk_aobt") < -20)
print(lag.select(n=pl.len(), med_y=pl.col("y").median(), med_pred=pl.col("pred").median(), med_err=pl.col("err").median(),
                 sse_share=(pl.col("err") ** 2).sum() / tot_sse))

print("\n5) type 2 (LIRF tail):")
tl = t.filter(pl.col("ADEP_mvt") == "LIRF")
print(tl.group_by("nm").agg(n=pl.len(), blk_sched=pl.col("blk_sched").mean().round(3), med_y=pl.col("y").median(),
                            med_mvt_aobt=pl.col("mvt_aobt").median(), med_err=pl.col("err").median(),
                            sse_share=((pl.col("err") ** 2).sum() / tot_sse).round(4)))
