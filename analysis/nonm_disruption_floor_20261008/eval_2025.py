"""No-NM disruption floor, 2025 out-of-fold evaluation (pre-registered №395). Rule fixed in advance:
scope = DEP, LOBT_flt null, ADEP != LIRF, live level L > 1500 s with >= 5 NM neighbours in +-30 min;
pred' = max(pred, round(0.95 L)). NM rows are never touched. Gate on E195 `control` (primary):
paired dSSE < 0 in >= 4/6 folds, pooled dSSE < 0, pooled trim20 dSSE < 0. FIN1b `pred` = secondary witness.
"""
import json, sys
import numpy as np
import polars as pl

S = sys.argv[1]
THR, NMIN, F = 1500.0, 5, 0.95
lv = pl.read_parquet(f"{S}/level_2025.parquet")
scope_expr = (~pl.col("has_nm")) & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("level") > THR) & (pl.col("n_nb") >= NMIN)
lv = lv.with_columns(scope=scope_expr.fill_null(False))


def evaluate(frame, pred_col, f=F, thr=THR, nmin=NMIN):
    d = frame.with_columns(scope=((~pl.col("has_nm")) & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("level") > thr) & (pl.col("n_nb") >= nmin)).fill_null(False))
    pred = d[pred_col].to_numpy().astype(float); y = d["y"].to_numpy().astype(float); month = d["month"].to_numpy()
    scope = d["scope"].to_numpy(); L = np.nan_to_num(d["level"].to_numpy())
    new = pred.copy(); new[scope] = np.maximum(pred[scope], np.rint(f * L[scope]))
    assert np.array_equal(new[d["has_nm"].to_numpy()], pred[d["has_nm"].to_numpy()]), "NM row touched"
    delta = (new - y) ** 2 - (pred - y) ** 2
    touched = new != pred
    out = dict(rule=dict(f=f, thr=thr, nmin=nmin, half_window_s=1800), scope_rows=int(scope.sum()), touched_rows=int(touched.sum()), folds={}, wins=0)
    for a in range(1, 7):
        m = (month == a) | (month == a + 6)
        dm = delta[m]; n = int(m.sum())
        sse0 = float(((pred[m] - y[m]) ** 2).sum()); sse1 = float(((new[m] - y[m]) ** 2).sum())
        out["folds"][f"{a}+{a + 6}"] = dict(n=n, touched=int(touched[m].sum()), delta_sse=float(dm.sum()), delta_rmse=float(np.sqrt(sse1 / n) - np.sqrt(sse0 / n)),
                                           rmse_control=float(np.sqrt(sse0 / n)), body_delta_sse=float(dm[y[m] <= 7200].sum()),
                                           touched_bias_before=float((y[m & touched] - pred[m & touched]).mean()) if touched[m].any() else None,
                                           touched_bias_after=float((y[m & touched] - new[m & touched]).mean()) if touched[m].any() else None)
        out["wins"] += int(dm.sum() < 0)
    ds = np.sort(delta[touched])
    out["pooled_delta_sse"] = float(delta.sum()); out["pooled_delta_rmse"] = float(np.sqrt(((new - y) ** 2).mean()) - np.sqrt(((pred - y) ** 2).mean()))
    out["pooled_trim20_delta_sse"] = float(ds[20:-20].sum()) if len(ds) > 40 else None
    out["pooled_trim3_delta_sse"] = float(ds[3:-3].sum()) if len(ds) > 6 else None
    out["body_pooled_delta_sse"] = float(delta[y <= 7200].sum())
    out["touched_rows_better"] = int((delta[touched] < 0).sum()); out["touched_rows_worse"] = int((delta[touched] > 0).sum())
    # bias diagnostics on scope rows
    s = scope
    out["scope_bias"] = dict(mean_y_minus_pred=float((y[s] - pred[s]).mean()), median_y_minus_pred=float(np.median(y[s] - pred[s])),
                             median_y_over_level=float(np.median(y[s] / L[s])), median_pred_over_level=float(np.median(pred[s] / L[s])),
                             frac_truth_below_floor=float((y[s] < f * L[s]).mean()), frac_pred_below_floor=float((pred[s] < np.rint(f * L[s])).mean()),
                             median_y=float(np.median(y[s])), median_pred=float(np.median(pred[s])), median_level=float(np.median(L[s])))
    out["gate"] = dict(wins_ge4=out["wins"] >= 4, pooled_neg=out["pooled_delta_sse"] < 0,
                       trimmed_neg=(out["pooled_trim20_delta_sse"] is not None and out["pooled_trim20_delta_sse"] < 0))
    out["gate"]["pass"] = all(out["gate"].values())
    return out


# primary: E195 control (current chain pre-E81: premix + E185 + E167 + E141/E64/E188 + E152 + LOBT clip)
e = pl.read_parquet("data/e195_pre_e81_20261007_v2/predictions.parquet", columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "month", "control", "evaluation_mask"]).rename({"TAXITIME_SEC_mvt": "y"})
j = lv.drop("month").join(e, on="MVT_ID_mvt", how="inner")
assert j.height == 2085047, j.height
results = {"primary_e195_control": evaluate(j, "control")}
# secondary: FIN1b production mixture OOF (pred)
parts = []
for a in range(1, 7):
    z = np.load(f"data/cv_preds_fin1b_hz/fold_{a}_{a + 6}.npz")
    parts.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "fin_pred": z["pred"].astype(float), "y_fin": z["y"].astype(np.int32)}))
fin = pl.concat(parts)
jf = lv.join(fin, on="MVT_ID_mvt", how="inner").rename({"y_fin": "y"})
assert jf.height == 2085047, jf.height
results["secondary_fin1b_pred"] = evaluate(jf, "fin_pred")
# descriptive sensitivity (NOT for selection; registered as descriptive only)
sens = {}
for f_, thr_, nmin_ in [(0.9, 1500, 5), (1.0, 1500, 5), (0.95, 1200, 5), (0.95, 2000, 5), (0.95, 1500, 10)]:
    r = evaluate(j, "control", f=f_, thr=thr_, nmin=nmin_)
    sens[f"f{f_}_thr{thr_}_n{nmin_}"] = dict(touched=r["touched_rows"], wins=r["wins"], pooled=r["pooled_delta_sse"], trim20=r["pooled_trim20_delta_sse"],
                                           body=r["body_pooled_delta_sse"], folds={k: round(v["delta_sse"]) for k, v in r["folds"].items()})
results["sensitivity_descriptive"] = sens
# per-airport breakdown of the primary rule (descriptive)
d = j.with_columns(scope=scope_expr.fill_null(False)).filter(pl.col("scope")).with_columns(
    new=pl.max_horizontal(pl.col("control"), (F * pl.col("level")).round(0)))
d = d.with_columns(delta=(pl.col("new") - pl.col("y")) ** 2 - (pl.col("control") - pl.col("y")) ** 2)
results["per_airport_primary"] = d.group_by("ADEP_mvt").agg(n=pl.len(), touched=(pl.col("new") != pl.col("control")).sum(), delta_sse=pl.col("delta").sum(),
                                                             med_y=pl.col("y").median(), med_pred=pl.col("control").median(), med_level=pl.col("level").median()).sort("n", descending=True).to_dicts()
results["per_month_primary"] = d.group_by("month").agg(n=pl.len(), touched=(pl.col("new") != pl.col("control")).sum(), delta_sse=pl.col("delta").sum(),
                                                        med_y=pl.col("y").median(), med_pred=pl.col("control").median()).sort("month").to_dicts()
# the 10 largest |delta| rows (where the verdict is decided)
results["top_abs_delta_rows"] = d.with_columns(ad=pl.col("delta").abs()).sort("ad", descending=True).head(10).select("ADEP_mvt", "day", "D", "level", "y", "control", "new", "delta").with_columns(pl.col("day").cast(pl.String)).to_dicts()
json.dump(results, open(f"{S}/eval_2025_results.json", "w"), indent=1, default=str)
for k in ("primary_e195_control", "secondary_fin1b_pred"):
    r = results[k]
    print(k, "scope", r["scope_rows"], "touched", r["touched_rows"], "wins", r["wins"], "pooled", round(r["pooled_delta_sse"]), "dRMSE", round(r["pooled_delta_rmse"], 4),
          "trim20", round(r["pooled_trim20_delta_sse"] or 0), "body", round(r["body_pooled_delta_sse"]), "gate", r["gate"])
    for fk, fv in r["folds"].items():
        print("  ", fk, fv)
    print("  bias", r["scope_bias"])
print(json.dumps(sens, indent=0))
print(results["per_airport_primary"]); print(results["per_month_primary"]); print(results["top_abs_delta_rows"])
