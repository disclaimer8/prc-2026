"""E184 probe, no model training: do the `ret_*` columns (src/e184_ret_overdue.py) say anything about the residual of the
production OOF of FIN1b (data/cv_preds_fin1b_hz, the control of the E184 CV)?

Rows: 2025 OOF rows the base model decides (fold_metrics.decided_map: not LIRF / no FLIGHT_ID / D >= 4 h), 0 < y <= 7200,
own AOBT_3 present (the rows without it are reported separately). r = y - pred, clipped to +-1800 s.
Reported per airport, per month and per half-year:
  raw    corr(feature, r);
  within corr after removing airport x UTC-day means (what is left once the day level is known);
  partial corr after per-airport OLS on what the model already has for the same mechanism: pred, own MVT - AOBT_3,
         own AOBT_3 - EOBT_1, own AOBT_3 - SCHED, taxiing_at_ob, offblocks_around_ob_30/60, dep_after_30, local hour;
  linear the three columns as a linear correction of the partial residual, fitted per airport on one half-year and applied
         to the other (both directions), in LB-equivalent seconds with the normalisation of src/fold_metrics.py.
Plus drift: per-airport means Jan+Jul 2025 vs ranking Jan+Jul 2026, and correlation with existing queue columns.

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e184_probe.py
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

NEW = ["ret_overdue_apt", "ret_overdue_rwy", "ret_atraso_15m"]
CTRL = ["pred", "prox", "d_aobt_eobt", "d_aobt_sched", "taxiing_at_ob", "offblocks_around_ob_30", "offblocks_around_ob_60",
        "dep_after_30", "local_hour"]
LB = 2 * 252.43 * 344841 * 6  # fold_metrics: sum of d over the six folds / LB = LB-equivalent seconds

oof = []
for p in sorted(glob.glob("data/cv_preds_fin1b_hz/fold_*.npz")):
    z = np.load(p)
    oof.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "y": z["y"].astype(np.float64)}))
oof = pl.concat(oof)
fcols = ["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "month", "t_mvt", "t_aobt", "d_mvt_sched", "d_mvt_aobt",
         "d_aobt_eobt", "d_aobt_sched", "taxiing_at_ob", "offblocks_around_ob_30", "offblocks_around_ob_60", "dep_after_30",
         "local_hour", "nb_ap_30_0_mean", "nbs_ap_30_0_mean", "nbf_ap_30_0_mean", "day_da_med"]
feat = pl.read_parquet("data/features_fin1b.parquet", columns=fcols).join(pl.read_parquet("data/features_e184.parquet"),
                                                                         on="MVT_ID_mvt", how="left")
assert feat["ret_overdue_apt"].null_count() == 0

# ---- drift: Jan + Jul 2025 against ranking (Jan + Jul 2026)
jj = feat.filter((pl.col("split") == "rank") | pl.col("month").is_in([1, 7])).with_columns(
    per=pl.when(pl.col("split") == "rank").then(pl.lit("2026 rank")).otherwise(pl.lit("2025 Jan+Jul")))
with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=200):
    print("DRIFT Jan+Jul 2025 vs ranking (means; atraso median)")
    print(jj.group_by("ADEP_mvt", "per").agg(n=pl.len(), ovd_apt=pl.col("ret_overdue_apt").mean().round(2),
                                             ovd_rwy=pl.col("ret_overdue_rwy").mean().round(2),
                                             atr_med=pl.col("ret_atraso_15m").median().round(0),
                                             taxiing=pl.col("taxiing_at_ob").cast(pl.Float64).mean().round(2))
          .pivot(on="per", index="ADEP_mvt", values=["ovd_apt", "ovd_rwy", "atr_med", "taxiing"]).sort("ADEP_mvt"))

d = oof.join(feat, on="MVT_ID_mvt", how="inner")
print("OOF rows", oof.height, "joined", d.height)
decided = ~((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null() & (pl.col("d_mvt_sched") >= 14400)).fill_null(False)
d = d.filter(decided & (pl.col("y") > 0) & (pl.col("y") <= 7200)).with_columns(
    r=(pl.col("y") - pl.col("pred")).clip(-1800, 1800),
    r_raw=pl.col("y") - pl.col("pred"),
    day=(pl.col("t_mvt") // 86400),
    half=pl.when(pl.col("month") <= 6).then(1).otherwise(2),
    prox=pl.col("d_mvt_aobt").clip(0, 7200),
    d_aobt_eobt=pl.col("d_aobt_eobt").clip(-3600, 7200),
    d_aobt_sched=pl.col("d_aobt_sched").clip(-3600, 7200),
)
noaobt = d.filter(pl.col("t_aobt").is_null())
d = d.filter(pl.col("t_aobt").is_not_null())
print(f"probe rows (decided, 0<y<=7200, own AOBT_3): {d.height}; without own AOBT_3: {noaobt.height}")

# outlier exposure of the unclipped atraso mean
print("atraso |value| > 3600:", round(float((d["ret_atraso_15m"].abs() > 3600).mean()), 4),
      " > 7200:", round(float((d["ret_atraso_15m"].abs() > 7200).mean()), 4))

# correlation with what the model already sees
print("CORR new vs existing (Pearson, probe rows)")
ex = ["taxiing_at_ob", "offblocks_around_ob_30", "offblocks_around_ob_60", "dep_after_30", "d_aobt_eobt", "d_aobt_sched",
      "nb_ap_30_0_mean", "nbs_ap_30_0_mean", "nbf_ap_30_0_mean", "day_da_med"]
cm = d.select(NEW + ex).fill_nan(None).fill_null(strategy="mean").to_numpy().astype(float)
cc = np.corrcoef(cm.T)
for i, c in enumerate(NEW):
    print(f"  {c:16s}", "  ".join(f"{e}={cc[i, len(NEW) + j]:+.2f}" for j, e in enumerate(ex)),
          f" | ovd_apt~ovd_rwy {cc[0, 1]:+.2f}" if i == 0 else "")


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def resid(X, v):
    m = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    b = np.linalg.lstsq(X[m], v[m], rcond=None)[0]
    out[m] = v[m] - X[m] @ b
    return out


# per-airport partial residuals (controls fitted per airport on all its probe rows)
parts = []
for (ap,), g in d.group_by("ADEP_mvt"):
    Xc = g.select(CTRL).fill_null(strategy="mean").to_numpy().astype(float)
    hr = g["local_hour"].to_numpy()
    X = np.column_stack([np.ones(g.height), Xc, *[(hr == h).astype(float) for h in range(5, 24)]])
    cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(), "pr": resid(X, g["r"].to_numpy())}
    for c in NEW:
        v = g[c].to_numpy().astype(float)
        fill = np.where(np.isfinite(v), v, np.nanmean(v))
        cols[f"p_{c}"] = resid(X, fill)
    parts.append(pl.DataFrame(cols))
d = d.join(pl.concat(parts), on="MVT_ID_mvt", how="left")
# airport x day demeaned
d = d.with_columns(*[(pl.col(c) - pl.col(c).mean().over("ADEP_mvt", "day")).alias(f"w_{c}") for c in NEW + ["r"]])


def table(by: str):
    rows = []
    for k, g in sorted(d.group_by(by), key=lambda kv: kv[0][0]):
        row = {by: k[0], "n": g.height}
        for c in NEW:
            row[f"raw_{c[4:]}"] = round(corr(g[c].to_numpy().astype(float), g["r"].to_numpy()), 4)
            row[f"within_{c[4:]}"] = round(corr(g[f"w_{c}"].to_numpy().astype(float), g["w_r"].to_numpy()), 4)
            row[f"partial_{c[4:]}"] = round(corr(g[f"p_{c}"].to_numpy(), g["pr"].to_numpy()), 4)
        rows.append(row)
    return pl.DataFrame(rows)


with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=250):
    for c in NEW:
        print(f"ALL {c}: raw {corr(d[c].to_numpy().astype(float), d['r'].to_numpy()):+.4f} "
              f"(unclipped r {corr(d[c].to_numpy().astype(float), d['r_raw'].to_numpy()):+.4f}) | "
              f"within airport-day {corr(d[f'w_{c}'].to_numpy().astype(float), d['w_r'].to_numpy()):+.4f} | "
              f"partial {corr(d[f'p_{c}'].to_numpy(), d['pr'].to_numpy()):+.4f}")
    for by in ("ADEP_mvt", "month", "half"):
        t = table(by)
        print(f"BY {by}")
        print(t.select(by, "n", *[c for c in t.columns if c.startswith("partial_")], *[c for c in t.columns if c.startswith("raw_")],
                       *[c for c in t.columns if c.startswith("within_")]))
    # airport x half-year partial: sign stability
    rows = []
    for (ap, h), g in d.group_by("ADEP_mvt", "half"):
        rows.append({"ADEP_mvt": ap, "half": h, **{f"partial_{c[4:]}": round(corr(g[f"p_{c}"].to_numpy(), g["pr"].to_numpy()), 4) for c in NEW}})
    print("PARTIAL by airport x half-year")
    print(pl.DataFrame(rows).pivot(on="half", index="ADEP_mvt", values=[f"partial_{c[4:]}" for c in NEW]).sort("ADEP_mvt"))

# linear correction, per airport, half-year swap, on the partial residual; scored on unclipped y - pred
gain = {}
for name, use in (("overdue_apt+rwy", NEW[:2]), ("atraso", NEW[2:]), ("all three", NEW)):
    tot = 0.0
    for (ap,), g in d.group_by("ADEP_mvt"):
        for h_fit, h_app in ((1, 2), (2, 1)):
            a, b = g.filter(pl.col("half") == h_fit), g.filter(pl.col("half") == h_app)
            Xa = np.column_stack([np.ones(a.height)] + [a[f"p_{c}"].to_numpy() for c in use])
            Xb = np.column_stack([np.ones(b.height)] + [b[f"p_{c}"].to_numpy() for c in use])
            coef = np.linalg.lstsq(Xa, a["pr"].to_numpy(), rcond=None)[0]
            coef[0] = 0.0  # no level shift: only what the columns say
            corr_b = Xb @ coef
            rb = b["r_raw"].to_numpy()
            tot += float(((rb - corr_b) ** 2).sum() - (rb ** 2).sum())
    gain[name] = tot / LB
print("LINEAR half-year swap, LB-equivalent s on decided 0<y<=7200 rows:", {k: round(v, 4) for k, v in gain.items()})
if noaobt.height:
    print("rows without own AOBT_3 (t = MVT - 900):",
          {c: round(corr(noaobt[c].to_numpy().astype(float), noaobt["r"].to_numpy()), 4) for c in NEW})
