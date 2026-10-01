"""H25 (journal 229, external review 01.10, item 2): an intraday destination delay wave read from OTHER hubs' departures.

For departure i: neighbours = departures from our other hubs (ADEP != ADEP_i) to the same ADES with |ARVT_1_j - ARVT_1_i| <= 1 h.
Neighbour delay = (AOBT_3_j - EOBT_1_j) minus the background of j's own hub (median AOBT_3 - EOBT_1 of all departures of
ADEP_j in the same UTC date-hour of EOBT_1). Features: dw_mean, dw_med, dw_n, dw_nhub, dw_pos (share of neighbours > 15 min).
Covered = dw_n >= 5 and dw_nhub >= 3 (123 399 ranking rows, verified 01.10).

Probe on the residual of the production chain OOF (data/cv_preds_fin1b_hz, the FIN1b table: E163 daily destination ATFM inside),
2025 departures with y <= 7200, residual clipped +-1800, covered rows only:
  (a) correlation of the residual with dw_mean, per half-year (Jan-Jun / Jul-Dec);
  (b) partial correlation after the own NM times (d_mvt_aobt, d_mvt_eobt, d_mvt_lobt, own AOBT_3 - EOBT_1), E163 (dst_*) and hour;
  (c) half-year swap stage: LightGBM on the residual, controls only vs controls + dw_*, fit on one half, scored on the other.
Close if (b) changes sign between halves or (c) is not lower in both directions (the review's own stop rule).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

Y = "TAXITIME_SEC_mvt"
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", Y]
raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]
                + [pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=cols)], how="vertical_relaxed")
d = raw.filter(pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt").with_columns(
    push=(ep("AOBT_3_flt") - ep("EOBT_1_flt")), arr=ep("ARVT_1_flt"),
    hour_key=pl.col("EOBT_1_flt").dt.truncate("1h"))
d = d.with_columns(bg=pl.col("push").median().over(["ADEP_mvt", "hour_key"])).with_columns(dl=pl.col("push") - pl.col("bg"))
d = d.filter(pl.col("arr").is_not_null() & pl.col("dl").is_not_null())
print(f"departures with ARVT_1 and AOBT_3/EOBT_1: {d.height}", flush=True)

hubs = sorted(d["ADEP_mvt"].unique().to_list()); hub_code = {h: i for i, h in enumerate(hubs)}
recs = []
for (ades,), g in d.select("MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "arr", "dl").group_by(["ADES_mvt"]):
    if g.height < 6:
        continue
    g = g.sort("arr")
    a = g["arr"].to_numpy(); ap = np.array([hub_code[x] for x in g["ADEP_mvt"].to_list()]); dl = g["dl"].to_numpy(); ids = g["MVT_ID_mvt"].to_numpy()
    lo = np.searchsorted(a, a - 3600, "left"); hi = np.searchsorted(a, a + 3600, "right")
    for i in range(len(a)):
        if hi[i] - lo[i] < 6:
            continue
        w_ap = ap[lo[i]:hi[i]]; m = w_ap != ap[i]
        if m.sum() < 5:
            continue
        w = dl[lo[i]:hi[i]][m]
        recs.append((ids[i], float(w.mean()), float(np.median(w)), int(m.sum()), len(np.unique(w_ap[m])), float((w > 900).mean())))
dw = pl.DataFrame(recs, schema=["MVT_ID_mvt", "dw_mean", "dw_med", "dw_n", "dw_nhub", "dw_pos"], orient="row").with_columns(
    pl.col("MVT_ID_mvt").cast(d.schema["MVT_ID_mvt"]))
print(f"rows with >= 5 neighbours: {dw.height}; covered (>= 3 hubs): {dw.filter(pl.col('dw_nhub') >= 3).height}", flush=True)
dw.write_parquet("data/h25_dest_wave.parquet")

# residual of the production chain OOF (FIN1b) on 2025 departures
ids, preds = [], []
for p in sorted(glob.glob("data/cv_preds_fin1b_hz/fold_*.npz")):
    z = np.load(p); ids.append(z["mvt_id"]); preds.append(z["pred"])
oof = pl.DataFrame({"MVT_ID_mvt": np.concatenate(ids), "oof": np.concatenate(preds)})
f = pl.read_parquet("data/features_fin1b.parquet", columns=["MVT_ID_mvt", "split", "month", "local_hour", "d_mvt_aobt", "d_mvt_eobt", "d_mvt_lobt",
                                                             "dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr", "dst_known", Y]).filter(pl.col("split") == "train")
oof = oof.with_columns(pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"]))
t = (f.join(oof, on="MVT_ID_mvt", how="inner").join(d.select(pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"]), "push"), on="MVT_ID_mvt", how="left")
      .join(dw.with_columns(pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"])), on="MVT_ID_mvt", how="inner")
      .filter((pl.col(Y) <= 7200) & (pl.col("dw_nhub") >= 3))
      .with_columns(res=(pl.col(Y) - pl.col("oof")).clip(-1800, 1800), half=pl.when(pl.col("month") <= 6).then(1).otherwise(2)))
print(f"covered 2025 rows with y <= 7200 and OOF: {t.height}", flush=True)
ctrl = ["local_hour", "d_mvt_aobt", "d_mvt_eobt", "d_mvt_lobt", "push", "dst_dly_per_arr", "dst_share_dly", "dst_w_per_arr", "dst_cap_per_arr", "dst_known"]
dwf = ["dw_mean", "dw_med", "dw_n", "dw_nhub", "dw_pos"]


def partial(x, y, Z):
    Z = np.column_stack([np.ones(len(x)), np.nan_to_num(Z)])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


for h in (1, 2):
    s = t.filter(pl.col("half") == h)
    x, y = s["dw_mean"].to_numpy(), s["res"].to_numpy()
    print(f"half {h}: n {s.height}, corr(res, dw_mean) {np.corrcoef(x, y)[0, 1]:+.4f}, partial after own NM times + E163 + hour "
          f"{partial(x, y, s.select(ctrl).to_numpy().astype(float)):+.4f}", flush=True)

P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=0)
for a, b in ((1, 2), (2, 1)):
    tr, te = t.filter(pl.col("half") == a), t.filter(pl.col("half") == b)
    sse = {}
    for name, fs in (("controls", ctrl), ("controls + dw", ctrl + dwf)):
        m = lgb.train(P, lgb.Dataset(tr.select(fs).to_pandas(), tr["res"].to_numpy()), 300)
        sse[name] = float(((te["res"].to_numpy() - m.predict(te.select(fs).to_pandas())) ** 2).sum())
    d_sse = sse["controls + dw"] - sse["controls"]
    print(f"fit H{a} -> score H{b}: SSE controls {sse['controls']:.4e}, + dw {sse['controls + dw']:.4e}, "
          f"diff {100 * d_sse / sse['controls']:+.3f} %, board-scale {d_sse / te.height * 123399 / 344841 / (2 * 233.6):+.4f} s "
          f"(per-row dSSE x 123 399 covered ranking rows)", flush=True)
