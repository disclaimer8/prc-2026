#!/usr/bin/env python
"""Two label-free checks of the final file (journal 448). Diagnostic only: nothing here enters the model, nothing is uploaded.

1. PRU indicator on the 2026 months. The PRU-style monthly "additional taxi-out time" (mean of taxi-out minus the
   unimpeded reference per airport x stand x runway) is computed from the final predictions with the exact method of
   docs/pru-view.md / scripts/regime_quality_report.py ("as recorded" variant: reference = 10th percentile of 2025
   taxi-out over all records per stand x runway, valid with >= 10 flights at or below it; PRU filters) and compared with
   the values EUROCONTROL publishes (taxi_out_additional_time_2026.csv). The same procedure is run on the 2025
   out-of-fold predictions against taxi_out_additional_time_2025.csv to measure what a deviation of a sound
   prediction looks like (de-icing exclusion, reference window), month by month.
2. Regime-share drift. Mean P(planning copy) per airport x month in 2026 (final premix classifier, parts_v970100.npz)
   against the same months of 2025 (production out-of-fold, cv_preds_e116prod).

Inputs are read only. Outputs: analysis/final_pru_and_regime_check_20261009/{result.json, pru_2026.csv,
pru_2025_procedure.csv, regime_share.csv}.

    POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
        .venv/bin/python analysis/final_pru_and_regime_check_20261009/check.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rqr", os.path.join(ROOT, "scripts", "regime_quality_report.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
T = R.T

FINAL = os.path.join(ROOT, "data/final_tail_20261009/work/gentle-igloo_final.parquet")
FINAL_SHA = "76c32426d750040690ff68bacd6ecbf3a7aa1e63340df5b9271a324b7064d554"
PARTS = os.path.join(ROOT, "data/final_tail_20261009/work/submissions/parts_v970100.npz")  # premix (receipt 00_premix parts_sha256)
PARTS_SHA = "1dccd50bfcdabd7c1229b752b37f2dda2f6c8ed2fb33bfcd29390c9de462dcf9"
RANKING = os.path.join(ROOT, "data/final_tail_20261009/work/data/prc-2026-datasets/ranking.parquet")
OOF = os.path.join(ROOT, "data/cv_preds_e116prod")
OOF_ALT = os.path.join(ROOT, "data/cv_preds_fin1b_hz")  # OOF behind docs/pru-view.md, second 2025 reference
PRU25 = os.path.join(ROOT, "data/taxi_out_additional_time_2025.csv")
PRU26 = os.path.join(ROOT, "data/taxi_out_additional_time_2026.csv")
MONTHS = [1, 2, 6, 7]
DOC_MAE_2025 = 3.3  # docs/data-quality-regimes.md 4.1: emulation vs published, April-October 2025
THR_S = 3 * DOC_MAE_2025
THR_PP = 3.0
KEY = ["ADEP_mvt", "month"]
GRP = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pru_ok(col: str) -> pl.Expr:
    """PRU filters exactly as regime_quality_report.pru_frame (null aircraft type -> null -> row dropped, as there)."""
    heli = pl.col("AIRCRAFT_TYPE_mvt").is_in(list(R.HELICOPTER_TYPES))
    return (pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null() & ~heli
            & (pl.col(col) >= 0) & (pl.col(col) <= R.PRU_MAX_TAXI))


def read_pru(path: str) -> pl.DataFrame:
    p = pl.read_csv(path, schema_overrides={"MONTH_NUM": pl.Int32, "COMMENT": pl.Utf8}).filter(pl.col("APT_ICAO").is_in(R.AIRPORTS))
    return p.select(
        pl.col("APT_ICAO").alias("ADEP_mvt"), pl.col("MONTH_NUM").alias("month"),
        pru_tf=pl.col("TF"), pru_valid_fl=pl.col("VALID_FL"), pru_ref_fl=pl.col("TOTAL_REF_NB_FL"),
        pru_add_s=pl.col("TOTAL_ADD_TIME_MIN") * 60 / pl.col("TOTAL_REF_NB_FL"),
        pru_ref_s=pl.col("TOTAL_REF_TIME_MIN") * 60 / pl.col("TOTAL_REF_NB_FL"),
    ).with_columns(pru_taxi_s=pl.col("pru_add_s") + pl.col("pru_ref_s"))


def kpi(df: pl.DataFrame, okcol: str, xcol: str, name: str) -> pl.DataFrame:
    v = df.filter(pl.col(okcol) & pl.col("valid_all"))
    return v.group_by(KEY).agg(
        pl.len().alias(f"n_valid_{name}"),
        (pl.col(xcol) - pl.col("ref_all")).mean().alias(f"kpi_{name}"),
        pl.col("ref_all").mean().alias(f"ref_{name}"),
        pl.col(xcol).mean().alias(f"taxi_{name}"),
    )


def summarise(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    return {"cells": int(len(d)), "mae_s": float(np.mean(np.abs(d))), "bias_s": float(np.mean(d)),
            "median_abs_s": float(np.median(np.abs(d))), "max_abs_s": float(np.max(np.abs(d))),
            "corr": float(np.corrcoef(a, b)[0, 1]) if len(d) > 2 else float("nan")}


def main() -> None:
    assert sha256(FINAL) == FINAL_SHA, "final file changed"
    assert sha256(PARTS) == PARTS_SHA, "premix parts changed"
    res: dict = {"final_sha256": FINAL_SHA, "parts_sha256": PARTS_SHA, "pru26_sha256": sha256(PRU26), "pru25_sha256": sha256(PRU25),
                 "threshold_s": THR_S, "threshold_pp": THR_PP}

    # ---------------------------------------------------------------- 2025: reference and the procedure on OOF
    oof, oof_meta = R.load_oof(OOF)
    res["oof"] = [{"file": os.path.relpath(m["file"], ROOT), "rows": m["rows"], "md5": m["md5"]} for m in oof_meta]
    dep = R.load_departures(os.path.join(ROOT, "data"))
    df = oof.join(dep, on="MVT_ID_mvt", how="inner")
    assert df.height == oof.height, (df.height, oof.height)
    assert float((df["y_oof"] - df[T.TARGET]).abs().max()) == 0.0
    df = df.with_columns(label_class=pl.Series(T.regime_label(df)).cast(pl.Int32),
                         p_planning_copy=(1 - pl.col("p_regular")).clip(0, 1))
    pf = R.pru_frame(df)  # adds y, pru_ok (label-based), ref_all / valid_all (2025 calendar year, all records)
    chk = pf.with_columns(ok_y=pru_ok("y").fill_null(False))
    assert (chk["ok_y"] == chk["pru_ok"].fill_null(False)).all(), "pru_ok re-implementation differs"
    ref = pf.filter(pl.col("pru_ok")).select(GRP + ["ref_all", "valid_all"]).unique(GRP)
    assert ref.select(GRP).is_duplicated().sum() == 0
    res["reference"] = {"groups": ref.height, "valid_groups": int(ref["valid_all"].sum())}

    pf = pf.with_columns(ok_y=pl.col("pru_ok").fill_null(False), ok_pred=pru_ok("pred").fill_null(False))
    k25 = (kpi(pf, "ok_y", "y", "label").join(kpi(pf, "ok_y", "pred", "pred_yfilter"), on=KEY)
           .join(kpi(pf, "ok_pred", "pred", "pred"), on=KEY).join(read_pru(PRU25), on=KEY, how="left").sort(KEY))
    k25 = k25.with_columns(dev_label=pl.col("kpi_label") - pl.col("pru_add_s"),
                           dev_pred=pl.col("kpi_pred") - pl.col("pru_add_s"),
                           dev_pred_vs_label=pl.col("kpi_pred") - pl.col("kpi_label"))
    # second 2025 OOF (the one behind docs/pru-view.md) through the same 2026 procedure, for robustness of the baseline
    alt, _ = R.load_oof(OOF_ALT)
    pa = pf.select(["MVT_ID_mvt", "ADEP_mvt", "month", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ref_all", "valid_all"]).join(
        alt.select("MVT_ID_mvt", pl.col("pred").alias("pred_alt")), on="MVT_ID_mvt", how="inner")
    pa = pa.with_columns(ok_alt=pru_ok("pred_alt").fill_null(False))
    k25 = k25.join(kpi(pa, "ok_alt", "pred_alt", "pred_fin1b").select(KEY + ["kpi_pred_fin1b"]), on=KEY, how="left").with_columns(
        dev_pred_fin1b=pl.col("kpi_pred_fin1b") - pl.col("pru_add_s"))
    k25.write_csv(os.path.join(OUT, "pru_2025_procedure.csv"), float_precision=2)

    def block(f: pl.DataFrame) -> dict:
        f = f.drop_nulls("pru_add_s")
        return {"emulation_from_labels": summarise(f["pru_add_s"].to_numpy(), f["kpi_label"].to_numpy()),
                "procedure_from_pred_e116prod": summarise(f["pru_add_s"].to_numpy(), f["kpi_pred"].to_numpy()),
                "procedure_from_pred_fin1b": summarise(f["pru_add_s"].to_numpy(), f["kpi_pred_fin1b"].to_numpy()),
                "pred_vs_label_indicator": summarise(f["kpi_label"].to_numpy(), f["kpi_pred"].to_numpy())}
    res["pru_2025"] = {"apr_oct": block(k25.filter(pl.col("month").is_between(4, 10))),
                       "nov_mar": block(k25.filter(~pl.col("month").is_between(4, 10))),
                       **{f"m{m}": block(k25.filter(pl.col("month") == m)) for m in MONTHS}}

    # ---------------------------------------------------------------- 2026: final predictions
    fin = pl.read_parquet(FINAL).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"))
    z = np.load(PARTS)
    pm = pl.DataFrame({"MVT_ID_mvt": z["mvt_id"].astype(np.int64), "pred_premix": z["pred"], "reg": z["reg"],
                       "p_planning_copy": 1 - z["proba"][:, 0], **{c: z["proba"][:, i] for i, c in enumerate(R.PROB_COLS)}})

    def secs(a, b):
        return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)
    rk = (pl.scan_parquet(RANKING).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col("MVT_ID_mvt").cast(pl.Int64), "ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32).alias("month"),
                  "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", d_mvt_sched=secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
                  d_mvt_lobt=secs("MVT_TIME_UTC_mvt", "LOBT_flt"), d_mvt_eobt=secs("MVT_TIME_UTC_mvt", "EOBT_1_flt")).collect())
    d26 = rk.join(fin, on="MVT_ID_mvt", how="inner").join(pm, on="MVT_ID_mvt", how="inner")
    assert d26.height == fin.height == rk.height == pm.height == 670_790, (d26.height, fin.height, rk.height, pm.height)
    d26 = d26.join(ref, on=GRP, how="left").with_columns(pl.col("valid_all").fill_null(False),
                                                         ok_pred=pru_ok("pred").fill_null(False),
                                                         ok_premix=pru_ok("pred_premix").fill_null(False))
    cnt = d26.group_by(KEY).agg(n_dep=pl.len(), n_pru_ok=pl.col("ok_pred").sum(),
                                n_no_type=pl.col("AIRCRAFT_TYPE_mvt").is_null().sum(),
                                n_pred_gt_7200=(pl.col("pred") > R.PRU_MAX_TAXI).sum(),
                                n_no_2025_group=(pl.col("ok_pred") & pl.col("ref_all").is_null()).sum())
    k26 = (cnt.join(kpi(d26, "ok_pred", "pred", "final"), on=KEY).join(kpi(d26, "ok_premix", "pred_premix", "premix").select(KEY + ["kpi_premix"]), on=KEY)
           .join(read_pru(PRU26), on=KEY, how="left")
           .join(k25.select(KEY + ["dev_pred", "dev_label", "dev_pred_fin1b", "pru_add_s", "kpi_pred"]).rename(
               {"dev_pred": "dev2025", "dev_label": "dev2025_label", "dev_pred_fin1b": "dev2025_fin1b", "pru_add_s": "pru_add_2025", "kpi_pred": "kpi_pred_2025"}), on=KEY, how="left")
           .sort(KEY))
    k26 = k26.with_columns(
        share_valid=pl.col("n_valid_final") / pl.col("n_dep"),
        tf_diff=pl.col("n_dep") - pl.col("pru_tf"),
        dev=pl.col("kpi_final") - pl.col("pru_add_s"),
        dev_premix=pl.col("kpi_premix") - pl.col("pru_add_s"),
        dev_taxi=pl.col("taxi_final") - pl.col("pru_taxi_s"),
        dev_ref=pl.col("ref_final") - pl.col("pru_ref_s"),
    ).with_columns(
        did=pl.col("dev") - pl.col("dev2025"),
        flag_dev=pl.col("dev").abs() > THR_S,
        flag_did=(pl.col("dev") - pl.col("dev2025")).abs() > THR_S,
    )
    # de-icing: PRU drops de-iced flights (flag not in the challenge data); its excluded share 1 - VALID_FL/TF above the
    # airport's April-October 2025 level is a label-free proxy for that share. D = extra additional time per excluded
    # flight, estimated per airport from 2025 (emulation from labels vs published, cells with excess share >= 2 %);
    # expected deviation of a sound prediction = D x excess share; band = min..max of the per-cell D of that airport.
    pr = pl.concat([read_pru(PRU25).with_columns(year=pl.lit(2025)), read_pru(PRU26).with_columns(year=pl.lit(2026))]).with_columns(
        excl=1 - pl.col("pru_valid_fl") / pl.col("pru_tf"))
    base = pr.filter((pl.col("year") == 2025) & pl.col("month").is_between(4, 10)).group_by("ADEP_mvt").agg(excl_base=pl.col("excl").mean())
    pr = pr.join(base, on="ADEP_mvt").with_columns(xs=(pl.col("excl") - pl.col("excl_base")).clip(0, None))
    dc = pr.filter(pl.col("year") == 2025).join(k25.select(KEY + ["dev_label"]), on=KEY).filter(pl.col("xs") >= 0.02).with_columns(
        D=pl.col("dev_label") / pl.col("xs"))
    dtab = dc.group_by("ADEP_mvt").agg(D=pl.col("D").median(), D_lo=pl.col("D").min(), D_hi=pl.col("D").max(), D_cells=pl.len())
    # fewer than 3 cells (LEMD, LTFM): the band is the pooled range of all airports' 2025 cells
    lo_all, hi_all = float(dc["D"].min()), float(dc["D"].max())
    dtab = dtab.with_columns(D_lo=pl.when(pl.col("D_cells") < 3).then(pl.lit(lo_all)).otherwise(pl.col("D_lo")),
                             D_hi=pl.when(pl.col("D_cells") < 3).then(pl.lit(hi_all)).otherwise(pl.col("D_hi")))
    res["deicing_D_2025"] = dtab.sort("ADEP_mvt").to_dicts()
    res["deicing_D_2025_pooled_range"] = [lo_all, hi_all]
    k26 = k26.join(pr.filter(pl.col("year") == 2026).select(KEY + ["excl", "xs"]), on=KEY, how="left").join(dtab, on="ADEP_mvt", how="left").with_columns(
        pl.col("D").fill_null(0.0), pl.col("D_lo").fill_null(0.0), pl.col("D_hi").fill_null(0.0))
    k26 = k26.with_columns(
        dev_adj=pl.col("dev") - pl.col("D") * pl.col("xs"),
        dev_adj_lo=pl.col("dev") - pl.col("D_hi") * pl.col("xs"),
        dev_adj_hi=pl.col("dev") - pl.col("D_lo") * pl.col("xs"),
    ).with_columns(flag_adj=(pl.col("dev_adj_lo") > THR_S) | (pl.col("dev_adj_hi") < -THR_S))
    k25a = k25.join(pr.filter(pl.col("year") == 2025).select(KEY + ["xs"]), on=KEY, how="left").join(dtab, on="ADEP_mvt", how="left").with_columns(
        dev_pred_adj=pl.col("dev_pred") - pl.col("D").fill_null(0.0) * pl.col("xs"))
    res["pru_2025_deicing_adjusted"] = {f"m{m}": summarise(np.zeros(10), k25a.filter(pl.col("month") == m)["dev_pred_adj"].to_numpy()) for m in MONTHS}
    res["pru_2025_deicing_adjusted"]["all12"] = summarise(np.zeros(120), k25a["dev_pred_adj"].to_numpy())
    k26.write_csv(os.path.join(OUT, "pru_2026.csv"), float_precision=2)
    f26 = k26.drop_nulls("pru_add_s")
    res["pru_2026"] = {
        "cells": f26.height,
        "all": summarise(f26["pru_add_s"].to_numpy(), f26["kpi_final"].to_numpy()),
        "premix": summarise(f26["pru_add_s"].to_numpy(), f26["kpi_premix"].to_numpy()),
        "did_all": {"mae_s": float(f26["did"].abs().mean()), "bias_s": float(f26["did"].mean()), "max_abs_s": float(f26["did"].abs().max())},
        **{f"m{m}": {**summarise(g["pru_add_s"].to_numpy(), g["kpi_final"].to_numpy()),
                     "did_mae_s": float(g["did"].abs().mean()), "did_bias_s": float(g["did"].mean()),
                     "same_month_2025_mae_s": float(g["dev2025"].abs().mean()), "same_month_2025_bias_s": float(g["dev2025"].mean())}
           for m in MONTHS for g in [f26.filter(pl.col("month") == m)]},
        "flags_dev": f26.filter(pl.col("flag_dev")).select(KEY + ["dev", "dev2025", "did"]).to_dicts(),
        "flags_did": f26.filter(pl.col("flag_did")).select(KEY + ["dev", "dev2025", "did"]).to_dicts(),
        "flags_deicing_adjusted": f26.filter(pl.col("flag_adj")).select(KEY + ["dev", "xs", "dev_adj", "dev_adj_lo", "dev_adj_hi"]).to_dicts(),
        "deicing_adjusted": {f"m{m}": {"mae_s": float(g["dev_adj"].abs().mean()), "bias_s": float(g["dev_adj"].mean()),
                                       "max_abs_s": float(g["dev_adj"].abs().max())} for m in MONTHS for g in [f26.filter(pl.col("month") == m)]},
        "rows": {"dep": int(k26["n_dep"].sum()), "pru_ok": int(k26["n_pru_ok"].sum()), "valid_ref": int(k26["n_valid_final"].sum()),
                 "no_aircraft_type": int(k26["n_no_type"].sum()), "pred_gt_7200": int(k26["n_pred_gt_7200"].sum()),
                 "stand_runway_group_absent_in_2025": int(k26["n_no_2025_group"].sum()),
                 "max_abs_tf_diff": int(k26["tf_diff"].abs().max())},
    }
    by_ap = f26.group_by("ADEP_mvt").agg(mae_s=pl.col("dev").abs().mean(), bias_s=pl.col("dev").mean(),
                                         did_mae_s=pl.col("did").abs().mean(), did_bias_s=pl.col("did").mean()).sort("ADEP_mvt")
    res["pru_2026"]["by_airport"] = by_ap.to_dicts()

    # ---------------------------------------------------------------- part 2: regime shares
    s25 = pf.filter(pl.col("month").is_in(MONTHS)).group_by(KEY).agg(
        n_2025=pl.len(), p_2025=pl.col("p_planning_copy").mean(), label_2025=(pl.col("label_class") > 0).mean(),
        flag50_2025=(pl.col("p_planning_copy") >= 0.5).mean())
    alt_p = pf.select("MVT_ID_mvt", "ADEP_mvt", "month").filter(pl.col("month").is_in(MONTHS)).join(
        alt.select("MVT_ID_mvt", (1 - pl.col("p_regular")).clip(0, 1).alias("p_alt")), on="MVT_ID_mvt", how="inner"
    ).group_by(KEY).agg(p_2025_fin1b=pl.col("p_alt").mean())
    s26 = d26.group_by(KEY).agg(n_2026=pl.len(), p_2026=pl.col("p_planning_copy").mean(),
                                flag50_2026=(pl.col("p_planning_copy") >= 0.5).mean())
    sh = s26.join(s25, on=KEY, how="left").join(alt_p, on=KEY, how="left").sort(KEY).with_columns(
        d_pp=100 * (pl.col("p_2026") - pl.col("p_2025")),
        d_pp_fin1b=100 * (pl.col("p_2026") - pl.col("p_2025_fin1b")),
        d_flag50_pp=100 * (pl.col("flag50_2026") - pl.col("flag50_2025")),
    ).with_columns(flag=pl.col("d_pp").abs() > THR_PP)
    sh.write_csv(os.path.join(OUT, "regime_share.csv"), float_precision=5)
    res["regime_share"] = {
        "cells": sh.height,
        "mean_p_2026": float(d26["p_planning_copy"].mean()),
        "mean_p_2025_same_months": float(pf.filter(pl.col("month").is_in(MONTHS))["p_planning_copy"].mean()),
        "max_abs_d_pp": float(sh["d_pp"].abs().max()),
        "flags": sh.filter(pl.col("flag")).select(KEY + ["p_2025", "p_2026", "d_pp", "label_2025"]).to_dicts(),
        "by_month": sh.group_by("month").agg(mean_abs_d_pp=pl.col("d_pp").abs().mean(), max_abs_d_pp=pl.col("d_pp").abs().max()).sort("month").to_dicts(),
    }
    # does the regime-share change line up with the indicator deviation?
    j = k26.join(sh.select(KEY + ["d_pp"]), on=KEY).drop_nulls("did")
    res["corr_did_vs_d_pp"] = float(np.corrcoef(j["did"].to_numpy(), j["d_pp"].to_numpy())[0, 1])

    # ---------------------------------------------------------------- label-free drivers per airport x month, both years
    def drivers(d: pl.DataFrame, pcol: str) -> pl.DataFrame:
        return d.group_by(KEY).agg(
            mean_pred=pl.col(pcol).mean(), mean_reg=pl.col("reg").mean(), uplift=(pl.col(pcol) - pl.col("reg")).mean(),
            p_copy_pct=100 * pl.col("p_planning_copy").mean(),
            **{f"{c}_pct": 100 * pl.col(c).mean() for c in R.PROB_COLS[1:]},
            dsched_mean=pl.col("d_mvt_sched").mean(), dsched_gt1800_pct=100 * (pl.col("d_mvt_sched") > 1800).mean(),
            no_lobt_pct=100 * pl.col("d_mvt_lobt").is_null().mean(),
            eobt_ne_lobt_pct=100 * (pl.col("d_mvt_eobt") != pl.col("d_mvt_lobt")).fill_null(False).mean(),
            eobt_taxi_window_pct=100 * pl.col("d_mvt_eobt").is_between(240, 1800).fill_null(False).mean())
    dr25 = drivers(pf.filter(pl.col("month").is_in(MONTHS)), "pred").with_columns(year=pl.lit(2025))
    dr26 = drivers(d26.rename({"pred": "pred_final"}).rename({"pred_premix": "pred"}), "pred").with_columns(year=pl.lit(2026))
    dr = pl.concat([dr25, dr26]).sort(KEY + ["year"])
    dr.write_csv(os.path.join(OUT, "drivers.csv"), float_precision=3)
    res["drivers_LIRF"] = dr.filter(pl.col("ADEP_mvt") == "LIRF").to_dicts()
    # flagged regime cells: how much of the change is the delay mix (2025 P per departure-delay bucket, 2026 bucket shares)?
    br = [0.0, 300.0, 600.0, 900.0, 1200.0, 1800.0, 3600.0]
    comp = []
    for row in sh.filter(pl.col("flag") | (pl.col("d_pp").abs() > 1.9)).iter_rows(named=True):
        ap, mo = row["ADEP_mvt"], row["month"]
        a = pf.filter((pl.col("ADEP_mvt") == ap) & (pl.col("month") == mo)).with_columns(bk=pl.col("d_mvt_sched").cut(br))
        b = d26.filter((pl.col("ADEP_mvt") == ap) & (pl.col("month") == mo)).with_columns(bk=pl.col("d_mvt_sched").cut(br))
        ga = a.group_by("bk").agg(p25=pl.col("p_planning_copy").mean())
        gb = b.group_by("bk").agg(n26=pl.len())
        g = gb.join(ga, on="bk", how="left")
        comp.append({"ADEP_mvt": ap, "month": mo, "p_2025_pct": 100 * row["p_2025"], "p_2026_pct": 100 * row["p_2026"],
                     "p_2026_if_only_delay_mix_changed_pct": float(100 * (g["n26"] * g["p25"]).sum() / g["n26"].sum())})
    res["regime_flag_composition"] = comp

    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    # markdown tables for the report (seconds, decimal comma, thin-space thousands; "!" marks a flag)
    mon = {1: "янв", 2: "фев", 6: "июн", 7: "июл"}

    def n(x: float, d: int = 1, sign: bool = False) -> str:
        return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "−").replace(".", ",")

    def k(x: int) -> str:
        return f"{x:,}".replace(",", "\u202f")
    lines = ["| аэропорт | мес | вылетов / TF PRU | доля с референсом | PRU 2026, с | наш, с | откл., с | откл. премикса, с "
             "| откл. 2025 (OOF, тот же метод), с | искл. PRU сверх лета-2025 | откл. за вычетом де-айсинга [полоса], с |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in k26.iter_rows(named=True):
        fl = "**!** " if r["flag_adj"] else ""
        band = f" [{n(r['dev_adj_lo'], 0, True)}…{n(r['dev_adj_hi'], 0, True)}]" if r["xs"] and r["xs"] >= 0.005 and r["D_hi"] > r["D_lo"] else ""
        lines.append(f"| {r['ADEP_mvt']} | {mon[r['month']]} | {k(r['n_dep'])} / {k(r['pru_tf'])} | {n(100 * r['share_valid'])} % | {n(r['pru_add_s'])} | "
                     f"{n(r['kpi_final'])} | {n(r['dev'], 1, True)} | {n(r['dev_premix'], 1, True)} | {n(r['dev2025'], 1, True)} | {n(100 * r['xs'])} % | "
                     f"{fl}{n(r['dev_adj'], 1, True)}{band} |")
    with open(os.path.join(OUT, "table_pru_2026.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    lines = ["| аэропорт | мес | P 2026, % | P 2025 OOF e116prod, % | P 2025 OOF fin1b_hz, % | правило ±6 с 2025, % | Δ 2026 − 2025, п.п. |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in sh.iter_rows(named=True):
        lines.append(f"| {r['ADEP_mvt']} | {mon[r['month']]} | {n(100 * r['p_2026'], 2)} | {n(100 * r['p_2025'], 2)} | {n(100 * r['p_2025_fin1b'], 2)} | "
                     f"{n(100 * r['label_2025'], 2)} | {'**!** ' if r['flag'] else ''}{n(r['d_pp'], 2, True)} |")
    with open(os.path.join(OUT, "table_regime.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    print(json.dumps({k: res[k] for k in ["reference", "pru_2025"]}, indent=1, default=float))
    print(k26.select(KEY + ["n_dep", "pru_tf", "share_valid", "kpi_final", "pru_add_s", "dev", "dev_premix", "dev_taxi", "dev_ref", "dev2025", "did", "flag_dev", "flag_did"]))
    print(json.dumps({k: v for k, v in res["pru_2026"].items() if k not in ("flags_dev", "flags_did")}, indent=1, default=float))
    print(sh)
    print(json.dumps({k: v for k, v in res["regime_share"].items()}, indent=1, default=float), res["corr_did_vs_d_pp"])


if __name__ == "__main__":
    main()
