"""Journal 155: three checks from the external review of 29.09 20:09.

    PART=1|2|3 .venv/bin/python src/h9_three_checks.py

1. LIRF no-NM tail (D >= 4 h, the TabPFN zone; E14 does not reach it): flight-number habit rule, leave-one-month-out over 2025.
   History = the same FLIGHT_mvt in the OTHER months, LIRF rows without NM at D >= 1800 s (sched = |y - D| <= 6, day = y in
   [1 d, 1 d + 2 h]). n >= 3, sched share >= 0.8 and no day -> D; day share >= 0.8 -> day candidate (86 400 + 951); else keep TabPFN.
   Baseline = TabPFN LOMO, arm A0 of src/e156_tabpfn_ensemble_lomo.py (canonical prefix codes, seed 0). Gate from the review:
   precision on the held-out month >= 0.9 and full-OOF change <= -0.5 s; otherwise it repeats E133's transfer.
2. LIRF delay bin 1-2 h, rows where the schedule candidate lies > 1000 s from the regressor: mean P(sched) vs the frequency.
   If miscalibrated: replace P(sched) by the out-of-month frequency of that (NM, D-bin) cell, renormalise, rebuild the mixture,
   write fold files for the usual gates (fold_metrics, post_e14).
3. CES570 at LFPG without NM: 2025 history by month, and its 2026 rows.
"""
import glob
import os
import re
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")
import numpy as np
import polars as pl

if os.environ.get("PART", "1") == "1":  # torch must load before polars starts its thread pool (segfault otherwise, 29.09)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
Y, DAY, DAY_TAXI = "TAXITIME_SEC_mvt", 86400.0, 951.0
PART = os.environ.get("PART", "1")
N_OOF, RMSE_OOF, N_RANK, RMSE_LB = 2085047, None, 344841, 235.17
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "d_mvt_sched", "d_mvt_lobt", Y]
F = pl.read_parquet("data/features_e106.parquet", columns=cols)


def oof_frame():
    parts = []
    for i, p in enumerate(sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))):
        z = np.load(p)
        parts.append(pl.DataFrame({"MVT_ID_mvt": z["mvt_id"], "pred": z["pred"], "reg": z["reg"], "y": z["y"],
                                   "p_sched": z["proba"][:, 1].astype(float), "fold": np.full(len(z["y"]), i)}))
    return pl.concat(parts)


if PART == "1":
    import lirf_table as L
    oof = oof_frame()
    rmse_oof = float(np.sqrt(((oof["pred"] - oof["y"]) ** 2).mean()))
    f = F.filter(pl.col("split") == "train")
    lirf = f.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
        sched=((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6), day=pl.col(Y).is_between(DAY, DAY + 7200))
    hist_src = lirf.filter(pl.col("d_mvt_sched") >= 1800)

    def rule(rows, src):
        h = src.group_by("FLIGHT_mvt").agg(n=pl.len(), ks=pl.col("sched").sum(), kd=pl.col("day").sum())
        r = rows.join(h, on="FLIGHT_mvt", how="left").with_columns(pl.col("n", "ks", "kd").fill_null(0))
        return r.with_columns(rule=pl.when((pl.col("n") >= 3) & (pl.col("ks") >= 0.8 * pl.col("n")) & (pl.col("kd") == 0)).then(pl.lit("sched"))
                              .when((pl.col("n") >= 3) & (pl.col("kd") >= 0.8 * pl.col("n"))).then(pl.lit("day")).otherwise(pl.lit(None)))

    # TabPFN A0 baseline, same data preparation as src/e156_tabpfn_ensemble_lomo.py
    ex = pl.read_parquet(os.environ.get("TAB_FEATURES", "data/features_v9.parquet"),
                         columns=["MVT_ID_mvt", "split", "STAND_mvt", "RUNWAY_mvt", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30"])
    t = (lirf.join(ex.drop("split"), on="MVT_ID_mvt", how="left")
         .join(pl.read_parquet("data/features_e113.parquet", columns=["MVT_ID_mvt", "airline_prefix"]), on="MVT_ID_mvt", how="left")
         .with_columns(cls=pl.when(pl.col("sched")).then(0).when(pl.col("day")).then(1).otherwise(2),
                       b=pl.col("d_mvt_sched").cut(L.BINS, labels=L.LABELS),
                       stand_num=pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
                       rwy_head=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64),
                       fn_num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
                       fn_len=pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float64),
                       fn_garbled=pl.col("FLIGHT_mvt").map_elements(lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64))
         .join(oof.select("MVT_ID_mvt", "reg"), on="MVT_ID_mvt", how="left"))
    feats = ["d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", "stand_num", "rwy_head", "fn_num", "fn_len", "fn_garbled", "pfx"]
    vc = t["airline_prefix"].value_counts().sort(["count", "airline_prefix"], descending=[True, False], nulls_last=True)
    keep = vc.filter(pl.col("count") >= 5)["airline_prefix"].drop_nulls().to_list()
    code = {p: i + 1 for i, p in enumerate(keep)}
    reg_c = float(t.filter(pl.col("cls") == 2).select(pl.col(Y).clip(0, 7200).median()).item())
    t = t.with_columns(pfx=pl.col("airline_prefix").map_elements(lambda p: float(code.get(p, 0)), return_dtype=pl.Float64).fill_null(0.0),
                       reg=pl.col("reg").clip(0, 7200).fill_null(reg_c))
    out = []
    for m in range(1, 13):
        tr, te = t.filter(pl.col("month") != m), t.filter((pl.col("month") == m) & pl.col("b").is_in(L.OVERRIDE))
        if te.height == 0:
            continue
        c = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0, categorical_features_indices=[len(feats) - 1])
        c.fit(tr.select(feats).to_numpy(), tr["cls"].to_numpy())
        pr = np.zeros((te.height, 3))
        for j, k in enumerate(c.classes_):
            pr[:, int(k)] = c.predict_proba(te.select(feats).to_numpy())[:, j]
        d, r = te["d_mvt_sched"].to_numpy(), te["reg"].to_numpy()
        te = te.with_columns(tab=pl.Series(pr[:, 0] * d + pr[:, 1] * (DAY + DAY_TAXI) + pr[:, 2] * r), p_day=pl.Series(pr[:, 1]))
        te = rule(te, hist_src.filter(pl.col("month") != m))
        out.append(te)
        print(f"month {m:2d}: tail rows {te.height}, rule fires {te['rule'].is_not_null().sum()}", flush=True)
    res = pl.concat(out).with_columns(
        new=pl.when(pl.col("rule") == "sched").then(pl.col("d_mvt_sched")).when(pl.col("rule") == "day").then(DAY + DAY_TAXI).otherwise(pl.col("tab")),
        truth=pl.when(pl.col("sched")).then(pl.lit("sched")).when(pl.col("day")).then(pl.lit("day")).otherwise(pl.lit("other")))
    fired = res.filter(pl.col("rule").is_not_null())
    print(f"tail rows {res.height} (truth: {dict(res['truth'].value_counts().rows())}); rule fired on {fired.height}")
    print(fired.group_by("rule", "truth").agg(n=pl.len()).sort("rule", "truth"))
    prec = (fired["rule"] == fired["truth"]).mean() if fired.height else float("nan")
    ds = float(((res["new"] - res[Y]) ** 2).sum() - ((res["tab"] - res[Y]) ** 2).sum())
    print(f"precision {prec:.3f}; SSE change vs TabPFN A0 {ds:+.4e} = full OOF {ds / (2 * N_OOF * rmse_oof):+.3f} s (OOF RMSE {rmse_oof:.1f}); "
          f"board scale (12 months of such rows) {ds / (2 * N_RANK * RMSE_LB):+.3f} s, per two months {ds / 6 / (2 * N_RANK * RMSE_LB):+.3f} s")
    by_m = res.group_by("month").agg(d=((pl.col("new") - pl.col(Y)) ** 2 - (pl.col("tab") - pl.col(Y)) ** 2).sum()).sort("month")
    print("months better / worse / equal:", (by_m["d"] < 0).sum(), (by_m["d"] > 0).sum(), (by_m["d"] == 0).sum())
    print(fired.select("month", "FLIGHT_mvt", "d_mvt_sched", Y, "tab", "p_day", "n", "ks", "kd", "rule", "truth").sort("month"))
    day_rows = res.filter(pl.col("truth") == "day")
    print("2025 day-shift tail rows:", day_rows.select("month", "FLIGHT_mvt", "d_mvt_sched", Y, "tab", "p_day").sort("d_mvt_sched"))
    # 2026 ranking rows the rule would touch (history = all 2025)
    rk = F.filter((pl.col("split") == "rank") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null() & (pl.col("d_mvt_sched") >= 14400))
    rk = rule(rk, hist_src).join(pl.read_parquet("submissions/gentle-igloo_v507.parquet").select("MVT_ID_mvt", v507=pl.col(Y)), on="MVT_ID_mvt")
    print("2026 rows the rule fires on:", rk.filter(pl.col("rule").is_not_null()).select("month", "FLIGHT_mvt", "d_mvt_sched", "v507", "n", "ks", "kd", "rule"))
    print("2026 day-shift share in 2025 by D (tail): ", lirf.filter(pl.col("d_mvt_sched") >= 14400).group_by(pl.col("d_mvt_sched").cut([43200, 57600, 72000]).alias("D"))
          .agg(n=pl.len(), day=pl.col("day").mean(), sched=pl.col("sched").mean()).sort("D").rows())

if PART == "2":
    import train as T
    oof = oof_frame()
    d = (F.filter(pl.col("split") == "train").select("MVT_ID_mvt", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "d_mvt_sched", Y)
         .join(oof, on="MVT_ID_mvt").with_columns(nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
                                                   is_sched=((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6),
                                                   far=(pl.col("d_mvt_sched") - pl.col("reg")).abs() > 1000,
                                                   Dh=pl.col("d_mvt_sched").cut([1800, 3600, 7200, 14400])))
    li = d.filter(pl.col("ADEP_mvt") == "LIRF")
    print("LIRF: mean P(sched) vs frequency by NM x D bin x far (|D - reg| > 1000)")
    print(li.group_by("nm", "Dh", "far").agg(n=pl.len(), p_mean=pl.col("p_sched").mean(), freq=pl.col("is_sched").mean(),
                                               p_med=pl.col("p_sched").median()).sort("nm", "Dh", "far"))
    b = li.filter(pl.col("d_mvt_sched").is_between(3600, 7200) & pl.col("far"))
    reg_rows = b.filter(~pl.col("is_sched") & ((pl.col("pred") - pl.col("reg")).abs() > 1000))
    print(f"bin 1-2 h, far: rows {b.height}; regular rows moved > 1000 s from the regressor {reg_rows.height}, median P(sched) {reg_rows['p_sched'].median():.3f}")
    # the fix: out-of-month frequency per (nm, D-bin) among far rows, applied only there; rebuild the mixture exactly
    li_far = li.filter(pl.col("far") & pl.col("d_mvt_sched").is_between(1800, 14400))
    os.makedirs("data/cv_preds_h9b", exist_ok=True)
    feat_l = F.filter(pl.col("split") == "train").select("MVT_ID_mvt", "d_mvt_lobt")
    tot = 0.0
    for i, p in enumerate(sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))):
        z = dict(np.load(p))
        months = li.filter(pl.col("fold") == i)["month"].unique().to_list()
        freq = li_far.filter(~pl.col("month").is_in(months)).group_by("nm", "Dh").agg(fq=pl.col("is_sched").mean())
        rows = li_far.filter(pl.col("fold") == i).join(freq, on=["nm", "Dh"], how="left").filter(pl.col("Dh").cast(pl.Utf8) == "(3600, 7200]")
        order = {k: j for j, k in enumerate(z["mvt_id"])}
        idx = np.array([order[k] for k in rows["MVT_ID_mvt"].to_numpy()], dtype=int)
        pr = z["proba"][idx].astype(float)
        fq = rows["fq"].to_numpy()
        rest = 1 - pr[:, 1]
        scale = np.where(rest > 0, (1 - fq) / np.where(rest > 0, rest, 1), 0.0)
        pn = pr * scale[:, None]
        pn[:, 1] = fq
        pn = pn / pn.sum(axis=1, keepdims=True)
        v = np.nan_to_num(z["values"][idx].astype(float))
        dl = feat_l.join(pl.DataFrame({"MVT_ID_mvt": rows["MVT_ID_mvt"]}), on="MVT_ID_mvt", how="right").select("d_mvt_lobt", "MVT_ID_mvt")
        mix = lambda P: np.clip(T.lobt_clip(np.sum(P * v, axis=1), dl), 0, None)
        new = z["pred"].copy()
        new[idx] = z["pred"][idx] + mix(pn) - mix(pr)
        y = z["y"]
        ch = float(((new - y) ** 2).sum() - ((z["pred"] - y) ** 2).sum())
        tot += ch
        print(f"{os.path.basename(p)}: rows changed {len(idx)}, mean P(sched) {pr[:, 1].mean():.3f} -> {pn[:, 1].mean():.3f}, SSE change {ch / 1e9:+.4f}e9")
        z["pred"] = new
        np.savez(f"data/cv_preds_h9b/{os.path.basename(p)}", **z)
    print(f"total SSE change {tot / 1e9:+.4f}e9 = full OOF {tot / (2 * N_OOF * float(np.sqrt(((oof['pred'] - oof['y']) ** 2).mean()))):+.3f} s (before E14)")

if PART == "3":
    oof = oof_frame()
    c = F.filter(pl.col("FLIGHT_mvt") == "CES570").with_columns(nm=pl.col("FLIGHT_ID_mvt").is_not_null())
    print("CES570 rows by split / airport / NM:", c.group_by("split", "ADEP_mvt", "nm").agg(n=pl.len()).sort("split", "ADEP_mvt", "nm").rows())
    tr = c.filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LFPG") & ~pl.col("nm")).join(oof, on="MVT_ID_mvt", how="left")
    print("2025 LFPG without NM:", tr.select("month", "d_mvt_sched", "d_mvt_lobt", Y, "pred").sort("month"))
    tn = c.filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LFPG") & pl.col("nm")).join(oof, on="MVT_ID_mvt", how="left")
    print(f"2025 LFPG with NM: n {tn.height}, y median {tn[Y].median()}, RMSE {np.sqrt(((tn['pred'] - tn[Y]) ** 2).mean()):.0f}, y > 3600: {(tn[Y] > 3600).sum()}")
    rk = c.filter((pl.col("split") == "rank") & (pl.col("ADEP_mvt") == "LFPG") & ~pl.col("nm")).join(
        pl.read_parquet("submissions/gentle-igloo_v507.parquet").select("MVT_ID_mvt", v507=pl.col(Y)), on="MVT_ID_mvt")
    print("2026 LFPG without NM:", rk.select("month", "d_mvt_sched", "d_mvt_lobt", "v507").sort("month"))
    lf = F.filter((pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LFPG") & pl.col("FLIGHT_ID_mvt").is_null()).join(oof, on="MVT_ID_mvt")
    sse = ((lf["pred"] - lf[Y]) ** 2)
    top = lf.with_columns(se=sse).sort("se", descending=True).head(8)
    print(f"LFPG no-NM 2025: rows {lf.height}, SSE {sse.sum():.3e}; top rows:", top.select("month", "FLIGHT_mvt", "d_mvt_sched", Y, "pred", "se").rows())
