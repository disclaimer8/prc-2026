"""U2 (journal 137, 140): does the airline prefix help the E14 logit for LIRF departures without NM at 0 < D < 4 h?
Same LOMO frame and control as src/e14_flight_history.py (data/cv_preds). Arms, SSE on the E14 target rows per fold:
  e14  : the accepted design, applied where the flight number has >= 3 history rows
  u2a  : e14 design + out-of-month logit rate of block = schedule for the airline prefix (E113 lookup) and log(1 + n prefix rows)
  u2b  : u2a, applied also where the flight number has < 3 history rows if the prefix has >= 20
Discord 28.09 (Vudueprajacu): "the echo of the planned time in the off-block is mostly a carrier thing when there's no NM match".
"""
import glob
import os
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
Y = "TAXITIME_SEC_mvt"
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "local_hour", "day_nonm_rate", Y])
f = f.filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null())
f = f.join(pl.read_parquet("data/features_e113.parquet", columns=["MVT_ID_mvt", "airline_prefix"]), on="MVT_ID_mvt", how="left")
f = f.with_columns(sched=((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).cast(pl.Int8), delayed=(pl.col("d_mvt_sched") > 1800),
                   pfx=pl.col("airline_prefix").fill_null("?"))
print("LIRF no-NM rows", f.height, "prefix known", f.filter(pl.col("pfx") != "?").height)


def history(src, rows):
    h = src.group_by("FLIGHT_mvt").agg(n=pl.len(), k=pl.col("sched").sum(), nd=pl.col("delayed").sum(), kd=(pl.col("sched") * pl.col("delayed")).sum())
    g = src.group_by("pfx").agg(np_=pl.len(), kp=pl.col("sched").sum())
    return (rows.join(h, on="FLIGHT_mvt", how="left").join(g, on="pfx", how="left").with_columns(pl.col("n", "k", "nd", "kd", "np_", "kp").fill_null(0))
            .with_columns(ff=(pl.col("k") + 0.5) / (pl.col("n") + 1), ffd=(pl.col("kd") + 0.5) / (pl.col("nd") + 1),
                          fp=(pl.col("kp") + 0.5) / (pl.col("np_") + 1)))


def design(d, pfx):
    lg = lambda c: np.log(d[c].to_numpy() / (1 - d[c].to_numpy()))
    D = d["d_mvt_sched"].to_numpy()
    cols = [lg("ff"), lg("ffd"), np.log1p(d["n"].to_numpy()), d["day_nonm_rate"].to_numpy(), np.log(np.clip(D, 60, None)), (D > 7200).astype(float), d["local_hour"].to_numpy()]
    if pfx:
        cols += [lg("fp"), np.log1p(d["np_"].to_numpy())]
    return np.column_stack(cols)


win = (pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400)
tot = {"e14": 0.0, "u2a": 0.0, "u2b": 0.0}
low = {"u2a": 0, "u2b": 0}
trim = {}
base_vs = []
CVDIR = os.environ.get("CVDIR", "data/cv_preds")
MINP = int(os.environ.get("MINP", "20"))
print("control", CVDIR, "prefix threshold", MINP)
for path in sorted(glob.glob(f"{CVDIR}/fold_*.npz")):
    z = np.load(path)
    a, b = [int(x) for x in os.path.basename(path)[5:-4].split("_")]
    base = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "reg": z["reg"], "y": z["y"]})
    fit = f.filter(~pl.col("month").is_in([a, b]))
    fit_h = pl.concat([history(fit.filter(pl.col("month") != m), fit.filter(pl.col("month") == m)) for m in fit["month"].unique().to_list()]).filter(win)
    val = history(fit, f.filter(pl.col("month").is_in([a, b]))).filter(win).join(base, on="MVT_ID_mvt", how="inner")
    y, old, D, reg = val["y"].to_numpy(), val["pred"].to_numpy(), val["d_mvt_sched"].to_numpy(), val["reg"].to_numpy()
    n, npx = val["n"].to_numpy(), val["np_"].to_numpy()
    out = {}
    for arm, pfx in (("e14", False), ("u2a", True)):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(design(fit_h, pfx), fit_h["sched"].to_numpy())
        p = m.predict_proba(design(val, pfx))[:, 1]
        out[arm] = np.where(n >= 3, p * D + (1 - p) * reg, old)
        if pfx:
            out["u2b"] = np.where((n >= 3) | (npx >= MINP), p * D + (1 - p) * reg, old)
    sse = {k: float(((v - y) ** 2).sum()) for k, v in out.items()}
    sse["base"] = float(((old - y) ** 2).sum())
    base_vs.append(((sse["e14"] - sse["base"]) / (2 * 344841 * 235.35), (sse["u2a"] - sse["base"]) / (2 * 344841 * 235.35)))
    for k in ("u2a", "u2b"):
        d = (out[k] - y) ** 2 - (out["e14"] - y) ** 2
        keep = np.ones(len(d), bool); keep[np.argsort(-np.abs(d))[:3]] = False
        trim.setdefault(k, []).append(float(d[keep].sum()))
    lb = 2 * 344841 * 235.35
    for k in tot:
        tot[k] += sse[k]
    for k in low:
        low[k] += sse[k] < sse["e14"]
    print(f"fold {a}+{b}: rows {len(y)}, applied e14 {(n >= 3).sum()} u2b {((n >= 3) | (npx >= MINP)).sum()} | vs e14: u2a {(sse['u2a'] - sse['e14']) / lb:+.3f} s, u2b {(sse['u2b'] - sse['e14']) / lb:+.3f} s (board-scale)")
lb = 2 * 344841 * 235.35
print(f"TOTAL vs e14: u2a {(tot['u2a'] - tot['e14']) / lb:+.3f} s (lower {low['u2a']}/6), u2b {(tot['u2b'] - tot['e14']) / lb:+.3f} s (lower {low['u2b']}/6)")
print("trimmed (without the 3 largest changes per fold) vs e14:", {k: f"{sum(v) / lb:+.3f} s, lower {sum(x < 0 for x in v)}/6" for k, v in trim.items()})
# 2026 ranking: rows each arm would touch (history = all of 2025)
rk = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched"]).filter(
    (pl.col("split") == "rank") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).join(
    pl.read_parquet("data/features_e113.parquet", columns=["MVT_ID_mvt", "airline_prefix"]), on="MVT_ID_mvt", how="left").with_columns(pfx=pl.col("airline_prefix").fill_null("?")).filter(win)
rk = rk.join(f.group_by("FLIGHT_mvt").agg(n=pl.len()), on="FLIGHT_mvt", how="left").join(f.group_by("pfx").agg(np_=pl.len()), on="pfx", how="left").with_columns(pl.col("n", "np_").fill_null(0))
print(f"2026 ranking LIRF no-NM 0<D<4h: {rk.height}; e14 applies to {rk.filter(pl.col('n') >= 3).height}; u2b to {rk.filter((pl.col('n') >= 3) | (pl.col('np_') >= 20)).height}")
print("vs the control's own prediction (no E14): e14 " + " ".join(f"{a:+.3f}" for a, _ in base_vs) + f" = {sum(a for a, _ in base_vs):+.3f} s; u2a " + " ".join(f"{b:+.3f}" for _, b in base_vs) + f" = {sum(b for _, b in base_vs):+.3f} s")
