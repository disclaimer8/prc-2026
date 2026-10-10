"""E218: why did E145 (PRU monthly calibration, journal 114/123) fail on the board, and is there a narrow form that passes?

Action `diagnose` (no upload, no fit, read-only inputs):
  1. 2025 production chain OOF (E216 `current` = E212 m10 at NN 0.35, i.e. the final configuration up to E152, before
     E81): per airport x month, the PRU-style indicator implied by our predictions (method of docs/pru-view.md "as
     recorded", functions of scripts/regime_quality_report.py, filter on the prediction as for 2026) against the value
     EUROCONTROL published for 2025 (data/taxi_out_additional_time_2025.csv); the same from labels (emulation offset);
     the airport-month mean residual of the chain; how much of the residual's airport-month mean the gap explains.
  2. E145 on the board: reconstruct the shift E145 applied (submissions v448 - v447 on the initial ranking months
     Jan/Jul 2026), the v447-implied indicator against PRU 2026, and what the published board delta (+0.114 s) says about
     the alignment of that shift with v447's true airport-month bias.
Action `evaluate` / `board` are added at pre-registration (journal 451).

    POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
        .venv/bin/python scripts/e218_pru_level_recheck_20261010.py diagnose
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

for _k, _v in {"POLARS_MAX_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "e218_pru_level_recheck_20261010")
_spec = importlib.util.spec_from_file_location("rqr", os.path.join(ROOT, "scripts", "regime_quality_report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
T = R.T

CHAIN = os.path.join(ROOT, "data/e203_pre_e81_20261007/predictions.parquet")      # `ordered` + masks (E203/E212/E216 chain base)
E216 = os.path.join(ROOT, "data/e216_chain_20261009")                              # fold_*/chain_e216.npz, delta[0] = current config
FIN1B = os.path.join(ROOT, "data/cv_preds_fin1b_hz")                                # cached FIN1b regime probabilities (chain input)
PRU25 = os.path.join(ROOT, "data/taxi_out_additional_time_2025.csv")
PRU26 = os.path.join(ROOT, "data/taxi_out_additional_time_2026.csv")
RANK_INITIAL = os.path.join(ROOT, "data/prc-2026-datasets/ranking.parquet")         # Jan/Jul 2026 (board months, initial phase)
RANK_FINAL = os.path.join(ROOT, "data/prc-2026-datasets/final_ranking.parquet")     # Jan/Feb/Jun/Jul 2026 (final phase)
FINAL = os.path.join(ROOT, "data/final_tail_20261009/work/gentle-igloo_final.parquet")   # read only
FINAL_SHA = "76c32426d750040690ff68bacd6ecbf3a7aa1e63340df5b9271a324b7064d554"
P26 = os.path.join(ROOT, "data/final_pru_regime_check_20261009/regime_probabilities_2026.parquet")  # final premix probabilities (№448 export)
V447 = os.path.join(ROOT, "submissions/gentle-igloo_v447.parquet")
V448 = os.path.join(ROOT, "submissions/gentle-igloo_v448.parquet")
BOARD = {"v447": 235.3498, "v448": 235.4634}      # journal 123
N_BOARD = 344_841                                   # rows scored on the initial board (journal 407c)
FOLDS = (1, 2, 3, 4, 5, 6)
KEY = ["ADEP_mvt", "month"]
GRP = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
E145_APS = ["EGLL", "LEMD", "LEBL", "LIRF", "EHAM"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pru_ok(col: str) -> pl.Expr:
    heli = pl.col("AIRCRAFT_TYPE_mvt").is_in(list(R.HELICOPTER_TYPES))
    return (pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null() & ~heli
            & (pl.col(col) >= 0) & (pl.col(col) <= R.PRU_MAX_TAXI)).fill_null(False)


def read_pru(path: str) -> pl.DataFrame:
    p = pl.read_csv(path, schema_overrides={"MONTH_NUM": pl.Int32, "COMMENT": pl.Utf8}).filter(pl.col("APT_ICAO").is_in(R.AIRPORTS))
    return p.select(
        pl.col("APT_ICAO").alias("ADEP_mvt"), pl.col("MONTH_NUM").alias("month"),
        pru_tf=pl.col("TF"), pru_valid_fl=pl.col("VALID_FL"),
        pru_add_s=pl.col("TOTAL_ADD_TIME_MIN") * 60 / pl.col("TOTAL_REF_NB_FL"),
        pru_ref_s=pl.col("TOTAL_REF_TIME_MIN") * 60 / pl.col("TOTAL_REF_NB_FL"),
    ).with_columns(pru_taxi_s=pl.col("pru_add_s") + pl.col("pru_ref_s"), excl=1 - pl.col("pru_valid_fl") / pl.col("pru_tf"))


def excess_exclusion() -> pl.DataFrame:
    """PRU excluded share above the airport's April-October 2025 level (label-free de-icing proxy, as in №448)."""
    pr = pl.concat([read_pru(PRU25).with_columns(year=pl.lit(2025)), read_pru(PRU26).with_columns(year=pl.lit(2026))])
    base = pr.filter((pl.col("year") == 2025) & pl.col("month").is_between(4, 10)).group_by("ADEP_mvt").agg(excl_base=pl.col("excl").mean())
    return pr.join(base, on="ADEP_mvt").with_columns(xs=(pl.col("excl") - pl.col("excl_base")).clip(0, None)).select("ADEP_mvt", "month", "year", "excl", "xs")


def reference(lab: pl.DataFrame) -> pl.DataFrame:
    """PRU-style unimpeded reference from labelled rows `lab` (needs y, STAND/RUNWAY/type): P10 per stand x runway over
    all records passing the PRU filters, valid with >= 10 flights at or below (regime_quality_report helpers)."""
    ok = lab.with_columns(w_all=pl.lit(1.0)).filter(pru_ok("y"))
    q = R.weighted_quantile_by_group(ok, GRP, "w_all", R.PRU_PCT, "ref")
    return q.with_columns(valid=(pl.col("n_at_or_below") >= R.PRU_MIN_AT_OR_BELOW)).select(GRP + ["ref", "valid"])


def indicator(df: pl.DataFrame, ref: pl.DataFrame, col: str, name: str, filter_col: str | None = None) -> pl.DataFrame:
    """Mean (col - ref) per airport x month over rows passing the PRU filters on `filter_col` (default col) with a valid ref."""
    fc = filter_col or col
    d = df.join(ref, on=GRP, how="left").filter(pru_ok(fc) & pl.col("valid").fill_null(False))
    return d.group_by(KEY).agg(**{f"ind_{name}": (pl.col(col) - pl.col("ref")).mean(), f"nv_{name}": pl.len(),
                                  f"taxi_{name}": pl.col(col).mean(), f"refm_{name}": pl.col("ref").mean()})


# ------------------------------------------------------------------------------------------------------------- 2025 data
def load_2025() -> pl.DataFrame:
    chain = pl.read_parquet(CHAIN).sort(ID)
    assert chain.height == 2_085_047
    parts = []
    for f in FOLDS:
        name = f"fold_{f}_{f + 6}"
        old = chain.filter(pl.col("month").is_in([f, f + 6]))
        with np.load(os.path.join(E216, name, "chain_e216.npz")) as z:
            assert str(z["names"][0]) == "current"
            assert np.array_equal(z["mvt_id"], old[ID].to_numpy())
            assert np.array_equal(z["excluded_e203"], ~old["evaluation_mask"].to_numpy())
            delta, unstable = z["delta"][0].astype(np.float64), z["unstable"][0].copy()
        parts.append(old.with_columns(pred=pl.Series(old["ordered"].to_numpy() + delta), unstable=pl.Series(unstable), pair=pl.lit(f)))
    d = pl.concat(parts).with_columns(pl.col(ID).cast(pl.Int64), pl.col(Y).cast(pl.Float64).alias("y"))
    dep = R.load_departures(os.path.join(ROOT, "data")).select(ID, "month", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                                                               pl.col(T.TARGET).alias("y_dep"), "d_mvt_sched", "d_mvt_iobt", "d_mvt_lobt", "d_mvt_eobt")
    d = d.join(dep.rename({"month": "month_dep"}), on=ID, how="inner")
    assert d.height == 2_085_047 and float((d["y"] - d["y_dep"]).abs().max()) == 0.0
    assert (d["month"].cast(pl.Int32) == d["month_dep"]).all()
    fin, _ = R.load_oof(FIN1B)
    d = d.join(fin.select(ID, "p_regular", pl.col("reg").alias("reg_fin1b"), pl.col("pred").alias("pred_fin1b")), on=ID, how="left")
    assert d["p_regular"].null_count() == 0
    assert T.TARGET == Y
    lab = pl.Series(T.regime_label(d)).cast(pl.Int32)
    d = d.with_columns(label_class=lab, month=pl.col("month").cast(pl.Int32))
    lirf_open = (pl.col("ADEP_mvt") == "LIRF") & ~pl.col("has_nm") & (pl.col("D") >= 14400)
    d = d.with_columns(ok=pl.col("evaluation_mask") & ~pl.col("unstable")).with_columns(
        decided=pl.col("ok") & ~lirf_open, body=pl.col("ok") & (pl.col("y") <= 7200))
    return d.drop("y_dep", "month_dep")


def cell_table_2025(d: pl.DataFrame, ref: pl.DataFrame) -> pl.DataFrame:
    pru = read_pru(PRU25)
    xs = excess_exclusion().filter(pl.col("year") == 2025).drop("year")
    t = (indicator(d, ref, "pred", "pred").join(indicator(d, ref, "y", "label"), on=KEY)
         .join(indicator(d, ref, "pred", "pred_yf", filter_col="y"), on=KEY))
    res = d.group_by(KEY).agg(
        n=pl.len(), n_dec=pl.col("decided").sum(),
        mr_dec=(pl.col("pred") - pl.col("y")).filter(pl.col("decided")).mean(),
        mr_body=(pl.col("pred") - pl.col("y")).filter(pl.col("body")).mean(),
        mr_reg=(pl.col("pred") - pl.col("y")).filter(pl.col("decided") & (pl.col("label_class") == 0)).mean(),
        mr_preg90=(pl.col("pred") - pl.col("y")).filter(pl.col("decided") & (pl.col("p_regular") >= 0.9)).mean(),
        n_preg90=(pl.col("decided") & (pl.col("p_regular") >= 0.9)).sum(),
        mean_preg=pl.col("p_regular").filter(pl.col("decided")).mean(),
        mean_pred=pl.col("pred").filter(pl.col("decided")).mean(), mean_y=pl.col("y").filter(pl.col("decided")).mean(),
        sse_dec=((pl.col("pred") - pl.col("y")) ** 2).filter(pl.col("decided")).sum(),
        wr_preg=((pl.col("pred") - pl.col("y")) * pl.col("p_regular")).filter(pl.col("decided")).sum() / pl.col("p_regular").filter(pl.col("decided")).sum())
    t = t.join(res, on=KEY).join(pru, on=KEY, how="left").join(xs, on=KEY, how="left").sort(KEY)
    return t.with_columns(gap=pl.col("ind_pred") - pl.col("pru_add_s"), emu=pl.col("ind_label") - pl.col("pru_add_s"),
                          bias_valid=pl.col("ind_pred_yf") - pl.col("ind_label"),
                          emu_taxi=pl.col("taxi_label") - pl.col("pru_taxi_s"), gap_taxi=pl.col("taxi_pred") - pl.col("pru_taxi_s"))


def explained(t: pl.DataFrame, xcol: str, ycol: str = "mr_dec", wcol: str = "n_dec") -> dict:
    x, yv, w = t[xcol].to_numpy(), t[ycol].to_numpy(), t[wcol].to_numpy().astype(float)
    m = np.isfinite(x) & np.isfinite(yv)
    x, yv, w = x[m], yv[m], w[m]
    beta = float((w * x * yv).sum() / (w * x * x).sum())                       # through the origin, weighted by rows
    oracle = float((w * yv * yv).sum())                                          # SSE removable by the exact cell mean
    full = float((w * (yv * yv - (yv - x) ** 2)).sum())                           # removing the whole gap
    best = float((w * (yv * yv - (yv - beta * x) ** 2)).sum())
    half = float((w * (yv * yv - (yv - 0.5 * x) ** 2)).sum())
    return {"cells": int(m.sum()), "corr": float(np.corrcoef(x, yv)[0, 1]), "corr_w": float(np.cov(x, yv, aweights=w)[0, 1] / np.sqrt(np.cov(x, x, aweights=w)[0, 0] * np.cov(yv, yv, aweights=w)[0, 0])),
            "beta_origin_w": beta, "sse_removable_oracle": oracle, "sse_removed_full_gap": full, "sse_removed_beta": best, "sse_removed_half_gap": half,
            "share_of_oracle_full": full / oracle, "share_of_oracle_beta": best / oracle, "share_of_oracle_half": half / oracle}


# ------------------------------------------------------------------------------------------------------------- 2026 data
def rank_frame(path: str) -> pl.DataFrame:
    return (pl.scan_parquet(path).filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col(ID).cast(pl.Int64), "ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32).alias("month"),
                    "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt").collect())


def read_sub(path: str, name: str) -> pl.DataFrame:
    s = pl.read_parquet(path)
    return s.select(pl.col(ID).cast(pl.Int64), pl.col(Y).cast(pl.Float64).alias(name))


def diagnose() -> None:
    os.makedirs(OUT, exist_ok=True)
    res: dict = {"inputs": {"pru25_sha256": sha256(PRU25), "pru26_sha256": sha256(PRU26), "final_sha256": sha256(FINAL),
                            "v447_sha256": sha256(V447), "v448_sha256": sha256(V448)}}
    assert res["inputs"]["final_sha256"] == FINAL_SHA
    d = load_2025()
    res["population_2025"] = {"rows": d.height, "ok": int(d["ok"].sum()), "decided": int(d["decided"].sum()), "body": int(d["body"].sum())}
    rmse = float(np.sqrt(d.filter("decided").select(((pl.col("pred") - pl.col("y")) ** 2).mean()).item()))
    res["population_2025"]["decided_rmse"] = rmse
    ref = reference(d)
    t = cell_table_2025(d, ref)
    t.write_csv(os.path.join(OUT, "cells_2025.csv"), float_precision=3)
    N = int(d["decided"].sum())
    to_s = lambda sse: sse / N / (2 * rmse)                                       # first-order dRMSE (s) of a pooled SSE change
    nodeice = t.filter(pl.col("xs") < 0.005)
    summ = {}
    for nm, sub in [("all120", t), ("no_deicing_xs_lt_0.5pct", nodeice), ("apr_oct", t.filter(pl.col("month").is_between(4, 10))),
                    ("LIRF", t.filter(pl.col("ADEP_mvt") == "LIRF")), ("no_LIRF_no_deicing", nodeice.filter(pl.col("ADEP_mvt") != "LIRF"))]:
        e = {}
        for xc in ["gap", "bias_valid", "emu"]:
            e[xc] = explained(sub, xc)
            for kk in ["sse_removable_oracle", "sse_removed_full_gap", "sse_removed_beta", "sse_removed_half_gap"]:
                e[xc][kk.replace("sse_", "drmse_s_")] = -to_s(e[xc][kk])
        e["gap_vs_wr_preg"] = explained(sub, "gap", ycol="wr_preg")
        e["gap_taxi"] = explained(sub, "gap_taxi")
        e["sd"] = {c: float(sub[c].std()) for c in ["gap", "emu", "mr_dec", "bias_valid", "emu_taxi", "gap_taxi", "wr_preg"]}
        e["mean"] = {c: float(sub[c].mean()) for c in ["gap", "emu", "mr_dec", "bias_valid", "emu_taxi", "gap_taxi", "wr_preg"]}
        e["mae"] = {c: float(sub[c].abs().mean()) for c in ["gap", "emu", "mr_dec", "bias_valid", "emu_taxi", "gap_taxi", "wr_preg"]}
        summ[nm] = e
    res["cells_2025"] = summ
    # emulation offset persistence: does an airport's emu in other months predict this month's emu? (what E145 relied on)
    emu_lopo = []
    for f in FOLDS:
        tr = t.filter(~pl.col("month").is_in([f, f + 6]) & (pl.col("xs") < 0.005)).group_by("ADEP_mvt").agg(emu_hat=pl.col("emu").mean(), gap_hat=pl.col("gap").mean())
        emu_lopo.append(t.filter(pl.col("month").is_in([f, f + 6])).join(tr, on="ADEP_mvt", how="left"))
    el = pl.concat(emu_lopo).filter(pl.col("xs") < 0.005)
    res["emu_persistence_2025"] = {"corr_emu_vs_lopo_mean": float(np.corrcoef(el["emu"], el["emu_hat"])[0, 1]),
                                   "sd_emu": float(el["emu"].std()), "sd_emu_minus_hat": float((el["emu"] - el["emu_hat"]).std()),
                                   "corr_mr_vs_gap_minus_gaphat": float(np.corrcoef(el["mr_dec"], el["gap"] - el["gap_hat"])[0, 1]),
                                   "corr_mr_vs_gap_minus_emuhat": float(np.corrcoef(el["mr_dec"], el["gap"] - el["emu_hat"])[0, 1])}

    # ------------------------------------------------------------------------------- E145 reconstruction on the board months
    rk = rank_frame(RANK_INITIAL)
    assert rk.height == N_BOARD
    b = rk.join(read_sub(V447, "p447"), on=ID).join(read_sub(V448, "p448"), on=ID)
    assert b.height == N_BOARD
    b = b.with_columns(s=pl.col("p448") - pl.col("p447"))
    ref26 = ref                                                                    # 2025 calendar-year reference, as in №448
    pru26 = read_pru(PRU26)
    xs26 = excess_exclusion().filter(pl.col("year") == 2026).drop("year")
    k447 = indicator(b, ref26, "p447", "p447").join(indicator(b, ref26, "p448", "p448"), on=KEY)
    sh = b.group_by(KEY).agg(n=pl.len(), s_mean=pl.col("s").mean(), s_rows=(pl.col("s") != 0).sum(), p447_mean=pl.col("p447").mean())
    g25 = t.select(KEY + [pl.col("gap").alias("gap2025"), pl.col("emu").alias("emu2025"), pl.col("mr_dec").alias("mr2025")])
    e = (sh.join(k447, on=KEY).join(pru26, on=KEY, how="left").join(xs26, on=KEY, how="left").join(g25, on=KEY, how="left")
         .with_columns(gap447=pl.col("ind_p447") - pl.col("pru_add_s"), gap448=pl.col("ind_p448") - pl.col("pru_add_s")).sort(KEY))
    # E145 itself: target = PRU 2026 mean taxi + same-month 2025 offset (mean label <= 7200 over ALL rows minus PRU 2025 taxi);
    # shift = target - mean prediction (<= 7200) over ALL rows. Reconstruct its terms with the chain labels (E145 used e106 OOF).
    lab = d.filter(pl.col("y") <= 7200).group_by(KEY).agg(ymean_f=pl.col("y").mean())
    p25 = read_pru(PRU25).select(KEY + [pl.col("pru_taxi_s").alias("pru_taxi25"), pl.col("pru_add_s").alias("pru_add25"), pl.col("pru_ref_s").alias("pru_ref25")])
    pmf = b.filter(pl.col("p447") <= 7200).group_by(KEY).agg(pmean_f=pl.col("p447").mean())
    e = e.join(lab, on=KEY, how="left").join(p25, on=KEY, how="left").join(pmf, on=KEY, how="left").with_columns(
        off2025=pl.col("ymean_f") - pl.col("pru_taxi25"),
        e145_shift_rebuilt=pl.col("pru_taxi_s") + pl.col("ymean_f") - pl.col("pru_taxi25") - pl.col("pmean_f"))
    # E145 shift minus the indicator-based shift (same-month 2025 emulation offset) = (PRU reference change - our reference
    # mix change) + (change of the valid-row minus all-row mean, 2026 predictions against 2025 labels): exact identity
    s25 = indicator(d, ref, "y", "y25").select(KEY + ["taxi_y25", "refm_y25"]).join(lab, on=KEY)
    s26 = indicator(b, ref26, "p447", "v447").select(KEY + ["taxi_v447", "refm_v447"]).join(pmf, on=KEY)
    p26r = pru26.select(KEY + [pl.col("pru_ref_s").alias("pru_ref26")])
    e = e.join(s25, on=KEY, how="left", suffix="_25").join(s26, on=KEY, how="left").join(p26r, on=KEY, how="left").with_columns(
        ind_shift=-((pl.col("ind_p447") - pl.col("pru_add_s")) - (pl.col("taxi_y25") - pl.col("refm_y25") - pl.col("pru_add25"))),
        d_pru_ref=pl.col("pru_ref26") - pl.col("pru_ref25"), d_our_ref=pl.col("refm_v447") - pl.col("refm_y25"),
        d_pop=(pl.col("taxi_v447") - pl.col("pmean_f")) - (pl.col("taxi_y25") - pl.col("ymean_f")))
    e = e.with_columns(identity_check=pl.col("e145_shift_rebuilt") - pl.col("ind_shift") - (pl.col("d_pru_ref") - pl.col("d_our_ref") + pl.col("d_pop")))
    assert float(e.filter(pl.col("s_rows") > 0)["identity_check"].abs().max()) < 1e-6
    res["e145_decomposition"] = e.filter(pl.col("s_rows") > 0).select(KEY + ["s_mean", "e145_shift_rebuilt", "ind_shift", "d_pru_ref", "d_our_ref", "d_pop"]).to_dicts()
    e.write_csv(os.path.join(OUT, "e145_board_cells.csv"), float_precision=3)
    dmse = BOARD["v448"] ** 2 - BOARD["v447"] ** 2
    s2 = float((b["s"] ** 2).sum() / N_BOARD)
    sb = (dmse - s2) / 2                                                           # = sum_rows s*(p447 - y) / N
    hyp = {}
    shifted = e.filter(pl.col("s_rows") > 0)
    for hn, expr in [("bias=gap2026", pl.col("gap447")), ("bias=gap2026-gap2025", pl.col("gap447") - pl.col("gap2025")),
                     ("bias=gap2026-emu2025", pl.col("gap447") - pl.col("emu2025"))]:
        x = shifted.with_columns(bh=expr)
        sbh = float((x["n"] * x["s_mean"] * x["bh"]).sum() / N_BOARD)
        pred_dmse = s2 + 2 * sbh
        hyp[hn] = {"sum_s_bias_per_row": sbh, "pred_board_dmse_s2": pred_dmse,
                   "pred_board_drmse_s": float(np.sqrt(BOARD["v447"] ** 2 + pred_dmse) - BOARD["v447"]),
                   "per_cell": x.select(KEY + ["n", "s_mean", "bh", "gap447", "gap2025", "emu2025"]).to_dicts()}
    res["e145_board"] = {"board_dmse_s2": dmse, "mean_s2": s2, "sum_s_bias_per_row_observed": sb, "b_eff_observed": sb / s2,
                         "optimal_scale": -sb / s2, "drmse_at_optimal_scale_s": float(np.sqrt(BOARD["v447"] ** 2 - sb * sb / s2) - BOARD["v447"]),
                         "hypotheses": hyp}
    # where the shift came from: rebuilt shift vs actual, and E145's population mismatch (rows with y > 7200 dropped from the label mean,
    # but no row dropped from the prediction mean)
    big = d.group_by(KEY).agg(share_y_gt7200=(pl.col("y") > 7200).mean(), share_pred_gt7200=(pl.col("pred") > 7200).mean(),
                              mean_y_all=pl.col("y").mean(), mean_pred_all=pl.col("pred").mean(),
                              mean_pred_yle=pl.col("pred").filter(pl.col("y") <= 7200).mean(), mean_pred_ple=pl.col("pred").filter(pl.col("pred") <= 7200).mean(),
                              mean_y_le=pl.col("y").filter(pl.col("y") <= 7200).mean())
    big = big.with_columns(e145_population_artifact=pl.col("mean_pred_ple") - pl.col("mean_pred_yle"))
    big.sort(KEY).write_csv(os.path.join(OUT, "e145_population_2025.csv"), float_precision=3)
    res["e145_population_artifact_2025"] = big.filter(pl.col("ADEP_mvt").is_in(E145_APS)).group_by("ADEP_mvt").agg(
        artifact_mean=pl.col("e145_population_artifact").mean(), artifact_min=pl.col("e145_population_artifact").min(),
        artifact_max=pl.col("e145_population_artifact").max(), share_y_gt7200=pl.col("share_y_gt7200").mean()).sort("ADEP_mvt").to_dicts()

    # ------------------------------------------------------------------------------- final file on the 2026 months
    rf = rank_frame(RANK_FINAL)
    fin = read_sub(FINAL, "pfin")
    p26 = pl.read_parquet(P26).select(pl.col(ID).cast(pl.Int64), pl.col("p_regular").cast(pl.Float64))
    f = rf.join(fin, on=ID, how="inner").join(p26, on=ID, how="left")
    assert f.height == 670_790 and f["p_regular"].null_count() == 0
    kf = indicator(f, ref26, "pfin", "fin").join(f.group_by(KEY).agg(n=pl.len(), mean_preg=pl.col("p_regular").mean(), mean_pfin=pl.col("pfin").mean()), on=KEY)
    kf = kf.join(pru26, on=KEY, how="left").join(xs26, on=KEY, how="left").join(g25, on=KEY, how="left").with_columns(
        gap=pl.col("ind_fin") - pl.col("pru_add_s")).sort(KEY)
    kf.write_csv(os.path.join(OUT, "cells_2026_final.csv"), float_precision=3)
    res["final_2026"] = kf.select(KEY + ["n", "gap", "gap2025", "emu2025", "xs", "mean_preg"]).to_dicts()
    with open(os.path.join(OUT, "diagnose.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    pl.Config.set_tbl_rows(130); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
    print(json.dumps({k: res[k] for k in ["population_2025", "cells_2025", "emu_persistence_2025"]}, indent=1, default=float))
    print(t.select(KEY + ["gap", "emu", "bias_valid", "mr_dec", "mr_reg", "mr_preg90", "xs"]).with_columns(pl.col(pl.Float64).round(1)))
    print(json.dumps({k: v for k, v in res["e145_board"].items() if k != "hypotheses"}, indent=1, default=float))
    for hn, h in res["e145_board"]["hypotheses"].items():
        print(hn, {k: v for k, v in h.items() if k != "per_cell"})
    print(e.select(KEY + ["n", "s_mean", "e145_shift_rebuilt", "gap447", "gap448", "gap2025", "emu2025", "off2025", "pmean_f", "ymean_f"]).with_columns(pl.col(pl.Float64).round(1)))
    print(json.dumps(res["e145_population_artifact_2025"], indent=1, default=float))
    print(pl.DataFrame(res["e145_decomposition"]).with_columns(pl.col(pl.Float64).round(1)))
    print(kf.select(KEY + ["n", "gap", "gap2025", "emu2025", "xs", "mean_preg"]).with_columns(pl.col(pl.Float64).round(2)))


# ============================================================================================ E218 registered rule (№451)
PROTOCOL = "E218-PRU-level-halfgap-nodeice-additive-v1"
XS_MAX = 0.005            # cell eligible only if PRU's excluded share is < 0.5 pp above the airport's Apr-Oct 2025 level (no de-icing)
K_PRIMARY = 0.5           # fixed fraction of the (gap - expected emulation offset); not tuned
VARIANTS = {               # first = registered rule; the others are reported only, they decide nothing
    "primary": dict(k=0.5, emu="hat", scope="nodeice", weight="flat"),
    "k0.25": dict(k=0.25, emu="hat", scope="nodeice", weight="flat"),
    "k0.75": dict(k=0.75, emu="hat", scope="nodeice", weight="flat"),
    "k1.0": dict(k=1.0, emu="hat", scope="nodeice", weight="flat"),
    "emu0": dict(k=0.5, emu="zero", scope="nodeice", weight="flat"),
    "lirf_only": dict(k=0.5, emu="hat", scope="LIRF", weight="flat"),
    "all_cells": dict(k=0.5, emu="hat", scope="all", weight="flat"),
    "preg": dict(k=0.5, emu="hat", scope="nodeice", weight="preg"),
}
TRIM = 20
SERVED = os.path.join(ROOT, "data/final_tail_20261009/work/e156/run_0/served.json")   # E156-protected rows: never shifted (read only)
DRYRUN = os.path.join(ROOT, "data/e218_pru_level_20261010")


def emu_hat(train: pl.DataFrame, ref: pl.DataFrame, pru: pl.DataFrame, xs: pl.DataFrame) -> pl.DataFrame:
    """Expected emulation offset per airport: mean over the TRAINING months without de-icing of (label indicator - published)."""
    li = indicator(train, ref, "y", "lab").join(pru, on=KEY, how="inner").join(xs, on=KEY, how="left")
    return li.filter(pl.col("xs") < XS_MAX).group_by("ADEP_mvt").agg(emu_hat=(pl.col("ind_lab") - pl.col("pru_add_s")).mean(), emu_months=pl.len())


def cell_shifts(ev: pl.DataFrame, ref: pl.DataFrame, emu: pl.DataFrame, pru: pl.DataFrame, xs: pl.DataFrame, v: dict) -> pl.DataFrame:
    """Per airport x month of `ev`: gap = indicator(pred) - published; c = -k (gap - emu_hat) on eligible cells, else 0."""
    ind = indicator(ev, ref, "pred", "ev").join(pru.select(KEY + ["pru_add_s"]), on=KEY, how="left").join(xs, on=KEY, how="left").join(emu, on="ADEP_mvt", how="left")
    e_hat = pl.lit(0.0) if v["emu"] == "zero" else pl.col("emu_hat").fill_null(0.0)
    if v["scope"] == "nodeice":
        elig = pl.col("pru_add_s").is_not_null() & (pl.col("xs") < XS_MAX)
    elif v["scope"] == "LIRF":
        elig = pl.col("pru_add_s").is_not_null() & (pl.col("xs") < XS_MAX) & (pl.col("ADEP_mvt") == "LIRF")
    elif v["scope"] == "all":
        elig = pl.col("pru_add_s").is_not_null()
    else:
        raise ValueError(v["scope"])
    return ind.with_columns(gap=pl.col("ind_ev") - pl.col("pru_add_s"), emu_used=e_hat, eligible=elig.fill_null(False)).with_columns(
        c=pl.when(pl.col("eligible")).then(-v["k"] * (pl.col("gap") - pl.col("emu_used"))).otherwise(0.0))


def apply_rows(ev: pl.DataFrame, ref: pl.DataFrame, cells: pl.DataFrame, weight: str, protected: pl.Series | None = None) -> pl.DataFrame:
    """Row shift on the indicator population P of each cell (PRU filters on the prediction, valid reference); other rows unchanged.
    flat: every P row moves by c; preg: P rows move by c * p_regular / mean_P(p_regular) (same mean shift c over P)."""
    x = ev.select([ID, "pred", "p_regular", "month"] + GRP + ["AIRCRAFT_TYPE_mvt"]).join(ref, on=GRP, how="left").with_columns(
        inP=pru_ok("pred") & pl.col("valid").fill_null(False))
    if protected is not None:
        x = x.with_columns(inP=pl.col("inP") & ~pl.col(ID).is_in(protected.to_list()))
    x = x.join(cells.select(KEY + ["c"]), on=KEY, how="left").with_columns(pl.col("c").fill_null(0.0))
    if weight == "flat":
        x = x.with_columns(shift=pl.when(pl.col("inP")).then(pl.col("c")).otherwise(0.0))
    elif weight == "preg":
        x = x.with_columns(mp=pl.col("p_regular").filter(pl.col("inP")).mean().over(KEY)).with_columns(
            shift=pl.when(pl.col("inP")).then(pl.col("c") * pl.col("p_regular") / pl.col("mp")).otherwise(0.0))
    else:
        raise ValueError(weight)
    x = x.with_columns(new=(pl.col("pred") + pl.col("shift")).round(0).clip(0, None))
    return x.select(ID, "shift", "new", "inP")


def trim(a: np.ndarray, k: int = TRIM) -> float:
    s = np.sort(a)
    return float(s[k:-k].sum()) if len(s) > 2 * k else 0.0


def leg(d: pl.DataFrame, train_months: list[int], eval_months: list[int], v: dict, cache: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One leg: reference and emu_hat from the training months' labels, the shift on the evaluation months."""
    key = tuple(sorted(train_months))
    if key not in cache:
        tr = d.filter(pl.col("month").is_in(train_months))
        ref = reference(tr)
        cache[key] = (ref, emu_hat(tr, ref, read_pru(PRU25), excess_exclusion().filter(pl.col("year") == 2025).drop("year")))
    ref, emu = cache[key]
    ev = d.filter(pl.col("month").is_in(eval_months)).sort(ID)
    cells = cell_shifts(ev, ref, emu, read_pru(PRU25), excess_exclusion().filter(pl.col("year") == 2025).drop("year"), v)
    rows = apply_rows(ev, ref, cells, v["weight"])
    out = ev.select(ID, "month", "ADEP_mvt", "y", "pred", "decided", "body").join(rows, on=ID, how="left")
    assert out.height == ev.height and out["new"].null_count() == 0
    return out.with_columns(dse=(pl.col("new") - pl.col("y")) ** 2 - (pl.col("pred") - pl.col("y")) ** 2), cells


def evaluate() -> None:
    os.makedirs(OUT, exist_ok=True)
    d = load_2025()
    dec = d.filter("decided")
    base_m = dec.group_by("month").agg(sse=((pl.col("pred") - pl.col("y")) ** 2).sum(), n=pl.len()).sort("month")
    base = {int(r["month"]): (float(r["sse"]), int(r["n"])) for r in base_m.iter_rows(named=True)}
    n_dec, sse_dec = sum(v[1] for v in base.values()), sum(v[0] for v in base.values())
    body = d.filter("body")
    n_body, sse_body = body.height, float(((body["pred"] - body["y"]) ** 2).sum())
    rm = lambda b, n, ds: float(np.sqrt((b + ds) / n) - np.sqrt(b / n))
    res = {"protocol": PROTOCOL, "base": {"decided_rmse": float(np.sqrt(sse_dec / n_dec)), "body_rmse": float(np.sqrt(sse_body / n_body)),
                                          "n_decided": n_dec, "n_body": n_body}, "variants": {}}
    cache: dict = {}
    allcells = []
    for name, v in VARIANTS.items():
        r = {"params": v}
        # six month-pair folds (LOPO): reference and emu_hat from the other ten months
        pairs, months, dec_tot, body_tot, t20 = [], {}, 0.0, 0.0, 0.0
        for f in FOLDS:
            tm = [m for m in range(1, 13) if m not in (f, f + 6)]
            out, cells = leg(d, tm, [f, f + 6], v, cache)
            dd = out.filter("decided")["dse"].to_numpy()
            db = float(out.filter("body")["dse"].sum())
            pairs.append({"pair": f"{f}+{f + 6}", "decided_dsse": float(dd.sum()), "body_dsse": db, "trim20": trim(dd),
                          "rows_shifted": int((out["shift"] != 0).sum()), "cells_shifted": int((cells["c"] != 0).sum())})
            dec_tot += float(dd.sum()); body_tot += db; t20 += trim(dd)
            for m in (f, f + 6):
                ds = float(out.filter(pl.col("decided") & (pl.col("month") == m))["dse"].sum())
                months[m] = {"dsse": ds, "drmse": rm(*base[m], ds)}
            if name == "primary":
                pm = out.filter(pl.col("inP")).group_by(KEY).agg(nP=pl.len(), mr_P=(pl.col("pred") - pl.col("y")).mean(),
                                                                  mean_shift=pl.col("shift").mean(), dsse_cell=pl.col("dse").sum())
                allcells.append(cells.join(pm, on=KEY, how="left").with_columns(pair=pl.lit(f)))
        r["lopo"] = {"pairs": pairs, "pairs_lower": sum(p["decided_dsse"] < 0 for p in pairs),
                     "pooled_decided_dsse": dec_tot, "pooled_decided_drmse": rm(sse_dec, n_dec, dec_tot),
                     "pooled_body_dsse": body_tot, "pooled_body_drmse": rm(sse_body, n_body, body_tot), "pooled_trim20": t20,
                     "months": {str(m): months[m] for m in range(1, 13)}, "months_not_worse": sum(months[m]["dsse"] <= 0 for m in range(1, 13)),
                     "months_lower": sum(months[m]["dsse"] < 0 for m in range(1, 13))}
        # forward legs
        out, _ = leg(d, list(range(1, 7)), list(range(7, 13)), v, cache)
        hs = float(out.filter("decided")["dse"].sum())
        b2 = sum(base[m][0] for m in range(7, 13)); n2 = sum(base[m][1] for m in range(7, 13))
        r["forward_half"] = {"dsse": hs, "drmse": rm(b2, n2, hs)}
        ex, exm = 0.0, {}
        for k in range(4, 13):
            out, _ = leg(d, list(range(1, k)), [k], v, cache)
            ds = float(out.filter("decided")["dse"].sum()); ex += ds; exm[str(k)] = {"dsse": ds, "drmse": rm(*base[k], ds)}
        b3 = sum(base[m][0] for m in range(4, 13)); n3 = sum(base[m][1] for m in range(4, 13))
        r["forward_expanding"] = {"dsse": ex, "drmse": rm(b3, n3, ex), "months": exm, "months_lower": sum(x["dsse"] < 0 for x in exm.values())}
        L = r["lopo"]
        g = {"pairs_ge_4_of_6": L["pairs_lower"] >= 4, "pooled_decided_lt0": L["pooled_decided_dsse"] < 0, "pooled_body_lt0": L["pooled_body_dsse"] < 0,
             "pooled_trim20_lt0": L["pooled_trim20"] < 0, "forward_half_lt0": r["forward_half"]["dsse"] < 0,
             "forward_expanding_lt0": r["forward_expanding"]["dsse"] < 0, "months_not_worse_ge_10": L["months_not_worse"] >= 10}
        g["passes"] = all(g.values())
        r["gate"] = g
        res["variants"][name] = r
        print(f"{name:10s} LOPO dec {L['pooled_decided_drmse']:+.4f} s body {L['pooled_body_drmse']:+.4f} trim20 {L['pooled_trim20']:+.3e} pairs {L['pairs_lower']}/6 "
              f"months<=0 {L['months_not_worse']}/12 | half {r['forward_half']['drmse']:+.4f} expanding {r['forward_expanding']['drmse']:+.4f} "
              f"({r['forward_expanding']['months_lower']}/9) | passes {g['passes']}", flush=True)
    ac = pl.concat(allcells, how="diagonal_relaxed").sort(KEY)
    ac.write_csv(os.path.join(OUT, "lopo_cells_primary.csv"), float_precision=3)
    # 2025 calibration of the bias model used for the 2026 estimate: cell mean residual over P on (gap - emu_hat), eligible cells
    el = ac.filter(pl.col("eligible") & pl.col("nP").is_not_null())
    xv, yv, w = (el["gap"] - el["emu_used"]).to_numpy(), el["mr_P"].to_numpy(), el["nP"].to_numpy().astype(float)
    beta = float((w * xv * yv).sum() / (w * xv * xv).sum())
    resid = yv - beta * xv
    aps = el["ADEP_mvt"].to_numpy()
    jack = []
    for a in np.unique(aps):                      # leave-one-airport-out jackknife of the slope
        m = aps != a
        jack.append(float((w[m] * xv[m] * yv[m]).sum() / (w[m] * xv[m] * xv[m]).sum()))
    jack = np.array(jack); na = len(jack)
    se = float(np.sqrt((na - 1) / na * ((jack - jack.mean()) ** 2).sum()))
    res["bias_model_2025"] = {"cells": int(len(xv)), "beta": beta, "beta_se_jackknife_airport": se, "beta_loo_airport_min": float(jack.min()),
                              "beta_loo_airport_max": float(jack.max()), "corr": float(np.corrcoef(xv, yv)[0, 1]),
                              "resid_sd_w": float(np.sqrt((w * resid ** 2).sum() / w.sum())), "sd_x": float(xv.std()), "sd_y": float(yv.std())}
    res["verdict_primary"] = res["variants"]["primary"]["gate"]
    with open(os.path.join(OUT, "evaluate.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res["bias_model_2025"], indent=1), flush=True)
    print("PRIMARY GATE", json.dumps(res["verdict_primary"]), flush=True)


def board() -> None:
    """2026: the registered rule on the final file (read only), per cell, and the expected effect on Jan/Jul (board) and Feb/Jun with
    a corridor from the 2025 bias model; plus the same bias model checked against E145's measured board delta. Writes a dry-run file
    under data/e218_pru_level_20261010/ (never uploaded) and the per-cell table."""
    ev_json = json.load(open(os.path.join(OUT, "evaluate.json")))
    if not ev_json["verdict_primary"]["passes"]:
        sys.exit("registered gate failed: no board estimate, no patch (journal 452)")
    bm = ev_json["bias_model_2025"]
    assert sha256(FINAL) == FINAL_SHA
    d = load_2025()
    ref = reference(d)
    pru25, pru26 = read_pru(PRU25), read_pru(PRU26)
    xs_all = excess_exclusion()
    xs25, xs26 = xs_all.filter(pl.col("year") == 2025).drop("year"), xs_all.filter(pl.col("year") == 2026).drop("year")
    emu = emu_hat(d, ref, pru25, xs25)
    rf = rank_frame(RANK_FINAL)
    p26 = pl.read_parquet(P26).select(pl.col(ID).cast(pl.Int64), pl.col("p_regular").cast(pl.Float64))
    f = rf.join(read_sub(FINAL, "pred"), on=ID, how="inner").join(p26, on=ID, how="left").sort(ID)
    assert f.height == 670_790 and f["p_regular"].null_count() == 0
    served = json.load(open(SERVED))
    prot = pl.Series(np.array(served["ids"], dtype=np.float64).astype(np.int64))
    v = VARIANTS["primary"]
    cells = cell_shifts(f, ref, emu, pru26, xs26, v)
    rows = apply_rows(f, ref, cells, v["weight"], protected=prot)
    x = f.join(rows, on=ID, how="left")
    assert x.height == f.height and x["new"].null_count() == 0
    assert x.filter(pl.col(ID).is_in(prot.to_list())).height == len(prot) and x.filter(pl.col(ID).is_in(prot.to_list()))["shift"].abs().max() == 0
    pc = x.filter(pl.col("inP")).group_by(KEY).agg(nP=pl.len(), mean_shift=pl.col("shift").mean())
    cells = cells.join(pc, on=KEY, how="left").with_columns(pl.col("nP").fill_null(0)).sort(KEY)
    cells.write_csv(os.path.join(OUT, "board_cells_2026.csv"), float_precision=3)
    # expected dSSE per cell for a true mean residual over P equal to beta * (gap - emu_hat):  nP (c^2 + 2 c b)
    B = bm["beta"]; se = bm["beta_se_jackknife_airport"]; sd_e = bm["resid_sd_w"]
    scen = {"beta_hat": B, "beta_lo_2se": max(B - 2 * se, 0.0), "beta_hi_2se": B + 2 * se, "beta_1": 1.0, "beta_0_gap_is_noise": 0.0}
    groups = {"board_jan_jul": [1, 7], "blind_feb_jun": [2, 6], "all_four": [1, 2, 6, 7]}
    n_rows = {g: int(f.filter(pl.col("month").is_in(ms)).height) for g, ms in groups.items()}
    R0 = {"board_jan_jul": 231.1805, "blind_feb_jun": 231.1805, "all_four": 231.1805}   # control v12080 (journal 445) as the level
    est = {}
    for g, ms in groups.items():
        cg = cells.filter(pl.col("month").is_in(ms) & pl.col("eligible"))
        xg, cg_c, nP = (cg["gap"] - cg["emu_used"]).to_numpy(), cg["c"].to_numpy(), cg["nP"].to_numpy().astype(float)
        e = {}
        for sn, bb in scen.items():
            dsse = float((nP * (cg_c ** 2 + 2 * cg_c * bb * xg)).sum())
            dmse = dsse / n_rows[g]
            e[sn] = {"dmse": dmse, "drmse": float(np.sqrt(R0[g] ** 2 + dmse) - R0[g])}
        lirf = (cg["ADEP_mvt"] == "LIRF").to_numpy()
        dsse = float((nP * (cg_c ** 2 + 2 * cg_c * np.where(lirf, 0.0, B * xg))).sum())
        e["lirf_gap_is_artifact"] = {"dmse": dsse / n_rows[g], "drmse": float(np.sqrt(R0[g] ** 2 + dsse / n_rows[g]) - R0[g])}
        sd = float(np.sqrt((nP * 4 * cg_c ** 2 * sd_e ** 2).sum()) / n_rows[g])     # cell-level noise of the bias model (1 sd, dMSE)
        e["noise_sd_dmse"] = sd
        e["noise_sd_drmse"] = sd / (2 * R0[g])
        e["cells_eligible"] = int(cg.height); e["rows_shifted"] = int(x.filter(pl.col("month").is_in(ms) & (pl.col("shift") != 0)).height)
        e["rows"] = n_rows[g]
        est[g] = e
    # same bias model against E145's measured board delta (+0.114 s): cells without de-icing take beta*(gap447 - emu_hat); LEMD Jan
    # (PRU excluded 3.1 pp above its summer level: de-icing) takes the band gap447 - D*xs with D in the pooled 2025 range (№448: 359-1298 s)
    rk = rank_frame(RANK_INITIAL)
    b = rk.join(read_sub(V447, "pred"), on=ID).join(read_sub(V448, "p448"), on=ID).with_columns(s=pl.col("p448") - pl.col("pred"))
    gi = indicator(b, ref, "pred", "v447").join(pru26, on=KEY, how="left").join(xs26, on=KEY, how="left").join(emu, on="ADEP_mvt", how="left").with_columns(
        gap447=pl.col("ind_v447") - pl.col("pru_add_s"))
    sc = b.group_by(KEY).agg(n=pl.len(), s_mean=pl.col("s").mean(), s_sd=pl.col("s").std()).join(gi, on=KEY).filter(pl.col("s_mean").abs() > 0)
    Nb = rk.height
    obs = (BOARD["v448"] ** 2 - BOARD["v447"] ** 2)
    s2 = float((b["s"] ** 2).sum() / Nb)
    chk = {}
    for sn, bb in {"beta_hat": B, "beta_1": 1.0, "beta_0": 0.0}.items():
        for dn, D in {"D_359": 359.0, "D_1298": 1298.0}.items():
            bias = sc.with_columns(bh=pl.when(pl.col("xs") < XS_MAX).then(bb * (pl.col("gap447") - pl.col("emu_hat").fill_null(0.0)))
                                   .otherwise(pl.col("gap447") - D * pl.col("xs")))
            sb = float((bias["n"] * bias["s_mean"] * bias["bh"]).sum() / Nb)
            pred = s2 + 2 * sb
            chk[f"{sn}|{dn}"] = {"pred_dmse": pred, "pred_drmse": float(np.sqrt(BOARD["v447"] ** 2 + pred) - BOARD["v447"])}
    res = {"protocol": PROTOCOL, "bias_model_2025": bm, "scenarios_beta": scen, "estimate": est,
           "e145_check": {"observed_dmse": obs, "observed_drmse": BOARD["v448"] - BOARD["v447"], "mean_s2": s2, "cases": chk,
                          "cells": sc.select(KEY + ["n", "s_mean", "gap447", "xs", "emu_hat"]).to_dicts()},
           "protected_rows_skipped": int(len(prot)), "final_sha256": FINAL_SHA}
    os.makedirs(DRYRUN, exist_ok=True)
    out = os.path.join(DRYRUN, "dryrun_final_plus_e218_pru_level.parquet")
    tmpl = pl.read_parquet(FINAL)
    new = tmpl.select(ID).with_columns(pl.col(ID).cast(pl.Int64).alias("_k")).join(x.select(pl.col(ID).alias("_k"), "new"), on="_k", how="left")
    assert new["new"].null_count() == 0 and new.height == tmpl.height
    tmpl.with_columns(pl.Series(Y, new["new"].to_numpy()).cast(tmpl.schema[Y])).write_parquet(out)
    res["dryrun"] = {"path": os.path.relpath(out, ROOT), "sha256": sha256(out), "rows_changed": int((new["new"].to_numpy() != tmpl[Y].to_numpy()).sum()),
                     "rms_change_s": float(np.sqrt(((new["new"].to_numpy() - tmpl[Y].to_numpy().astype(float)) ** 2).mean())),
                     "uploaded": False}
    assert sha256(FINAL) == FINAL_SHA
    with open(os.path.join(OUT, "board.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    print(cells.select(KEY + ["gap", "emu_used", "xs", "eligible", "c", "nP", "mean_shift"]).with_columns(pl.col(pl.Float64).round(2)))
    print(json.dumps({k: res[k] for k in ["estimate", "e145_check", "dryrun"]}, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["diagnose", "evaluate", "board"])
    a = ap.parse_args()
    {"diagnose": diagnose, "evaluate": evaluate, "board": board}[a.action]()


if __name__ == "__main__":
    main()
