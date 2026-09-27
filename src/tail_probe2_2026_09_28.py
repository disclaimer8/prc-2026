import glob
import numpy as np
import polars as pl

raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                             "TAXITIME_SEC_mvt", "AOBT_3_flt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "pred": z["pred"]})
                 for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
    to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
    mvt_aobt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64),
    lag=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds() < -1200),
    month=pl.col("MVT_TIME_UTC_mvt").dt.month())
d = d.with_columns(err=pl.col("pred") - pl.col("y"))
tot = float((d["err"] ** 2).sum())
n = d.filter(pl.col("AOBT_3_flt").is_not_null() & (pl.col("ADEP_mvt") != "LIRF"))
print(f"NM rows outside LIRF: {n.height}; AOBT_3 > 20 min after block: {int(n['lag'].sum())} ({n['lag'].mean():.2%}); "
      f"of them y>3600: {int(n.filter(pl.col('lag'))['y'].gt(3600).sum())}; lag rows SSE share {float((n.filter(pl.col('lag'))['err']**2).sum())/tot:.2%}")
print("lag rate and mean residual by y band:")
print(n.with_columns(band=pl.col("y").cut([900, 1800, 3600, 7200])).group_by("band").agg(
    rows=pl.len(), lag=pl.col("lag").mean().round(4), mean_err=pl.col("err").mean().round(1)).sort("band"))

# observable neighbour signal: same airport, other departures with MVT within +-60 min, median MVT-AOBT3 (self excluded ~ mean)
parts = []
for _, g in n.sort("to").group_by("ADEP_mvt", maintain_order=True):
    t, v = g["to"].to_numpy(), g["mvt_aobt"].to_numpy()
    lo, hi = np.searchsorted(t, t - 3600, "left"), np.searchsorted(t, t + 3600, "right")
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cnt = hi - lo - 1
    parts.append(g.select("MVT_ID_mvt").with_columns(nb_mvt_aobt=pl.Series(np.where(cnt > 0, (cs[hi] - cs[lo] - v) / np.maximum(cnt, 1), np.nan))))
n = n.join(pl.concat(parts), on="MVT_ID_mvt")
print("\nresidual of the current model by decile of the neighbour mean MVT-AOBT3 (+-60 min, same airport):")
q = n.with_columns(dec=pl.col("nb_mvt_aobt").qcut(10, labels=[str(i) for i in range(10)]))
print(q.group_by("dec").agg(rows=pl.len(), nb_min=(pl.col("nb_mvt_aobt").mean() / 60).round(1), lag=pl.col("lag").mean().round(4),
                            tail=(pl.col("y") > 3600).mean().round(4), mean_err=pl.col("err").mean().round(1),
                            sse_share=((pl.col("err") ** 2).sum() / tot).round(4)).sort("dec"))
top = q.filter(pl.col("dec") == "9")
print("top decile by month (mean residual, lag rate):")
print(top.group_by("month").agg(rows=pl.len(), lag=pl.col("lag").mean().round(3), mean_err=pl.col("err").mean().round(1)).sort("month"))
print("corr(residual, nb_mvt_aobt) on NM rows outside LIRF:", round(float(np.corrcoef(n.drop_nulls('nb_mvt_aobt').drop_nans('nb_mvt_aobt')['err'], n.drop_nulls('nb_mvt_aobt').drop_nans('nb_mvt_aobt')['nb_mvt_aobt'])[0, 1]), 4))
