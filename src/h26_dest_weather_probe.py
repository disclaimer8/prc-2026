"""H26 (journal 234, external review 01.10, item 3): destination weather at the planned arrival hour.

Departures to one of our ten hubs (local IEM METAR archive, public domain): the destination METAR at the planned arrival ARVT_1
(latest report at or before ARVT_1, within 90 min) and the worst of the window ARVT_1 +- 1 h: visibility, lowest layer, gust,
wind, thunderstorm / fog-mist / snow flags, low-visibility flag, and the change of visibility and layer over the hour before.
Probe on the residual of the production chain OOF (data/cv_preds_fin1b_hz, E163 inside), 2025, y <= 7200, residual clipped
+-1800, rows with destination weather only. Half-year swap stage (LightGBM on the residual): controls (own NM times, E163, hour,
own-origin weather) vs controls + destination weather; then the same on top of the H25 wave (controls + dw_*). Close if the
stage is not lower in both directions (the review's order: beyond E163 first, then beyond H25).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

Y = "TAXITIME_SEC_mvt"
HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)

met = []
for h in HUBS:
    m = pl.read_csv(f"data/metar/{h}.csv", null_values=["null", "M", ""], infer_schema_length=0)
    m = m.select(pl.lit(h).alias("ADES_mvt"),
                 t=pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").dt.epoch("s").cast(pl.Float64),
                 vsby=pl.col("vsby").cast(pl.Float64, strict=False), sky=pl.col("skyl1").cast(pl.Float64, strict=False),
                 gust=pl.col("gust").cast(pl.Float64, strict=False), sknt=pl.col("sknt").cast(pl.Float64, strict=False),
                 wx=pl.col("wxcodes").fill_null(""))
    met.append(m.with_columns(ts=pl.col("wx").str.contains("TS").cast(pl.Float64), fg=pl.col("wx").str.contains("FG|BR").cast(pl.Float64),
                              sn=pl.col("wx").str.contains("SN").cast(pl.Float64)).drop("wx").drop_nulls("t").unique(["ADES_mvt", "t"]).sort("t"))
met = pl.concat(met).sort(["ADES_mvt", "t"])
# change over the hour before each report and the worst of the +-1 h window around it
met = met.with_columns(
    vsby_prev=pl.col("vsby").shift(2).over("ADES_mvt"), sky_prev=pl.col("sky").shift(2).over("ADES_mvt"))  # half-hourly reports
met = met.with_columns(d_vsby=pl.col("vsby") - pl.col("vsby_prev"), d_sky=pl.col("sky") - pl.col("sky_prev")).drop("vsby_prev", "sky_prev")
met = met.with_columns(t_i=pl.col("t").cast(pl.Int64)).sort(["ADES_mvt", "t_i"])
win = (met.rolling(index_column="t_i", period="7200i", offset="-3600i", group_by="ADES_mvt")
       .agg(vsby_min=pl.col("vsby").min(), gust_max=pl.col("gust").max(), ts_any=pl.col("ts").max(), fg_any=pl.col("fg").max()))
met = met.join(win, on=["ADES_mvt", "t_i"], how="left").drop("t_i")
WX = ["vsby", "sky", "gust", "sknt", "ts", "fg", "sn", "d_vsby", "d_sky", "vsby_min", "gust_max", "ts_any", "fg_any"]

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "ARVT_1_flt"]
raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]
                + [pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=cols)], how="vertical_relaxed")
d = raw.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADES_mvt").is_in(HUBS) & pl.col("ARVT_1_flt").is_not_null()).select(
    "MVT_ID_mvt", "ADES_mvt", t=ep("ARVT_1_flt")).sort("t")
dw = d.join_asof(met.rename({"t": "t_met"}).with_columns(t=pl.col("t_met")).sort("t"), on="t", by="ADES_mvt", strategy="backward", tolerance=5400.0)
dw = dw.filter(pl.col("t_met").is_not_null()).select("MVT_ID_mvt", *[pl.col(c).alias(f"dwx_{c}") for c in WX],
                                                     dwx_lowvis=(pl.col("vsby") < 1.9).cast(pl.Float64))
print(f"departures to our hubs with destination METAR: {dw.height} (of {d.height})", flush=True)
dw.write_parquet("data/h26_dest_weather.parquet")
DWX = [c for c in dw.columns if c.startswith("dwx_")]

ids, preds = [], []
for p in sorted(glob.glob("data/cv_preds_fin1b_hz/fold_*.npz")):
    z = np.load(p); ids.append(z["mvt_id"]); preds.append(z["pred"])
oof = pl.DataFrame({"MVT_ID_mvt": np.concatenate(ids), "oof": np.concatenate(preds)})
fs = pl.read_parquet_schema("data/features_fin1b.parquet")
own_wx = [c for c in fs if c.startswith("wx_")]
f = pl.read_parquet("data/features_fin1b.parquet", columns=["MVT_ID_mvt", "split", "month", "local_hour", "d_mvt_aobt", "d_mvt_eobt", "d_mvt_lobt",
                                                             "dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr", "dst_known", Y] + own_wx
                    ).filter(pl.col("split") == "train")
key = f.schema["MVT_ID_mvt"]
h25 = pl.read_parquet("data/h25_dest_wave.parquet").with_columns(pl.col("MVT_ID_mvt").cast(key))
t = (f.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(key)), on="MVT_ID_mvt", how="inner")
      .join(dw.with_columns(pl.col("MVT_ID_mvt").cast(key)), on="MVT_ID_mvt", how="inner")
      .join(h25, on="MVT_ID_mvt", how="left")
      .filter(pl.col(Y) <= 7200)
      .with_columns(res=(pl.col(Y) - pl.col("oof")).clip(-1800, 1800), half=pl.when(pl.col("month") <= 6).then(1).otherwise(2)))
print(f"2025 rows with destination weather, y <= 7200, OOF: {t.height}; own-origin weather columns {len(own_wx)}", flush=True)
ctrl = ["local_hour", "d_mvt_aobt", "d_mvt_eobt", "d_mvt_lobt", "dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr", "dst_known"] + own_wx
dwf = ["dw_mean", "dw_med", "dw_n", "dw_nhub", "dw_pos"]
rank_rows = 51626
P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=0)
for base_name, base in (("beyond E163", ctrl), ("beyond E163 + H25", ctrl + dwf)):
    for a, b in ((1, 2), (2, 1)):
        tr, te = t.filter(pl.col("half") == a), t.filter(pl.col("half") == b)
        sse = {}
        for name, fsel in (("base", base), ("+ dest weather", base + DWX)):
            m = lgb.train(P, lgb.Dataset(tr.select(fsel).to_pandas(), tr["res"].to_numpy()), 300)
            sse[name] = float(((te["res"].to_numpy() - m.predict(te.select(fsel).to_pandas())) ** 2).sum())
        dd = sse["+ dest weather"] - sse["base"]
        print(f"[{base_name}] fit H{a} -> score H{b}: SSE {sse['base']:.4e} -> {sse['+ dest weather']:.4e}, diff {100 * dd / sse['base']:+.3f} %, "
              f"board-scale {dd / te.height * rank_rows / 344841 / (2 * 233.6):+.4f} s (x {rank_rows} ranking rows)", flush=True)
