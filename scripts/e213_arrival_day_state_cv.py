"""E213 (journal 407): two label-free day-state probes from ARRIVALS (ideas 2 and 3 of the external review, journal 406),
measured as an INCREMENT over what the production chain already knows. Definitions, descriptive stop rule and the gate are
written here BEFORE the run and committed together with the journal line. No model is fitted; a fixed one-parameter rule
per airport at most (kappa or beta), fitted on the training months of each fold only.

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=4 .venv/bin/python scripts/e213_arrival_day_state_cv.py > data/e213_arrival_day_state_20261008/run.log

Data: 2025 training files (DEP and ARR rows), production control OOF of the chain `data/e195_pre_e81_20261007_v2/predictions.parquet`
(column `joint`, through E152, pre-E81) and the FIN1b base OOF `data/cv_preds_fin1b_hz` (regime probabilities `proba`, class order
regular, sched, iobt, lobt, eobt, dayshift; candidate `values`). Day = UTC day of MVT_TIME (take-off for DEP, landing for ARR).

PROBE 2 — day-level recording state from arrivals.
  Arrival copy signatures (fixed): A_sched |BLOCK - SCHED| <= 6 s (primary); A_arvt1 |BLOCK - ARVT_1| <= 6 s; A_arvt3 |BLOCK - ARVT_3| <= 6 s.
  arr_share(a, d) = mean of A_sched over arrivals of airport a on day d, defined when >= 20 arrivals.
  Departure label: regime_label of src/train.py (first of |y - (MVT - SCHED/IOBT/LOBT/EOBT_1)| <= 6 s at the seven labelled airports;
  dayshift y in [86400, 93600]); label_i = 1 when the label is any planning class or dayshift. p_i = 1 - P(regular) of the FIN1b OOF.
  Increment of the day beyond the classifier: inc_i = label_i - p_i.
  x_day (primary) = arr_share(a, d) - mean over the TRAINING months of arr_share at airport a (airport mean; in the descriptive step
  the mean over all 2025); x_dm (reported) = arr_share(a, d) - mean over airport x month (the within-month fluctuation only).
  Descriptive: per airport, row-level Pearson corr(inc_i, x_day(i)) over departures with FIN1b proba and evaluation_mask; also
  corr(label_i, x_day), day-level corr of (label share - mean p) with x_day, and the same with x_dm and with A_arvt1 / A_arvt3.
  Coordinator amendment (08.10, before the run): the increment must survive DAY-LEVEL CONGESTION CONTROLS, since a bad day raises
  every day statistic at once. Controls C(a, d) = daily departure count, daily arrival count, daily mean arrival taxi-in, daily median
  of the chain's own prediction (joint) — all observable in 2026. Partial correlation = corr of the residuals of inc_i and of x after
  OLS (with intercept) on C within the airport, everything first centred by airport x month (so (a) the airport-month level the
  classifier absorbs and (b) congestion are both removed); inc = label - p already controls for the classifier's own mean P(copy).
  STOP RULE (fixed): the rule arm is built only if max |partial corr(inc, x | C, airport x month)| over {LIRF, LFPG, LTFM, EGLL} >= 0.05;
  otherwise null.
  Rule: planning-class logit shift. For the four planning classes p'_k = p_k exp(kappa_a x_day) / Z, regular and dayshift p'_k = p_k / Z.
  delta_i = sum_{k>=1} (p'_k - p_k) (v_k - v_0) with the FIN1b candidate values (NaN candidates carry p = 0); pred' = rint(joint + delta).
  kappa_a: grid -3 ... 3 step 0.05 minimising the SSE of pred' on the training months (decided rows), per airport; x_day undefined -> 0.
  Applied to held-out decided rows except LIRF without an NM record (tail chain) and rows without FIN1b proba.

PROBE 3 — NM clock offset of the day.
  off_arr(a, d) = median(ARVT_3 - MVT) over arrivals of airport a on day d with ARVT_3 (>= 20 rows); off_c = off_arr - mean over
  airport x month of off_arr (observable in 2026: arrivals carry ARVT_3 and MVT there too). Reported too: off_a = off_arr - airport mean.
  Departure anchor residual of the day: res_dep(a, d) = median(AOBT_3 - BLOCK) over departures with AOBT_3, centred by airport x month.
  Anchor rows: AOBT_3 present, 0 <= MVT - AOBT_3 <= 7200, P(regular) >= 0.5 (FIN1b), joint <= 7200, evaluation_mask, not LIRF without NM.
  Descriptive: per airport, day-level corr(res_dep_c, off_c) (days with both defined; raw anchor residual, reported only) and
  row-level corr of the chain residual (y - joint, 0 < y <= 7200, centred by airport x month) with off_c on anchor rows.
  Coordinator amendment (08.10, before the run): the test statistic is the CHAIN residual (y - joint), never the raw anchor residual,
  so what the model already corrects (the airport x month anchor levels, -118 s LIRF ... +296 s LTFM) is not counted twice; and the
  increment is the partial correlation after the same congestion controls C as probe 2 (daily departure and arrival counts, daily
  mean arrival taxi-in, daily median joint prediction), everything centred by airport x month, both at row level (anchor rows) and at
  day level (mean chain residual of the day over anchor body rows vs off_c | C).
  STOP RULE (fixed): the rule arm is built only if max |row-level partial corr(resid, off_c | C, airport x month)| over
  {LIRF, LFPG, LTFM, EGLL} >= 0.05; otherwise null.
  Rule: pred' = rint(joint + beta_a off_c) on anchor rows; beta_a by OLS without intercept of y - joint on off_c, training months only,
  rows evaluation_mask & 0 < y <= 7200 & joint <= 7200 & anchor; off_c undefined -> 0.

METRIC: paired squared error against the unchanged chain, LB-equivalent seconds = sum / (6 * 2 * 252.43 * 344841) (as E211).
Folds: the six month pairs (1,7) ... (6,12), training = the other ten months.
GATE (per rule, all required): decided lower in >= 4/6 pairs; pooled decided < 0; pooled body (y <= 7200) < 0; pooled trim20 < 0
(decided without the 20 best and 20 worst rows of each fold); pairs (2,8) and (6,12) decided <= 0; AND forward-in-time legs not worse:
half-year fit 1-6 -> apply 7-12 decided <= 0; expanding window fit 1..k-1 -> k, k = 4..12, pooled decided <= 0; one-month gap fit
1..k-2 -> k, k = 4..12, pooled decided <= 0. Same primary arm on the FIN1b base OOF is reported, not gating.
PREDICTION before the run: probe 2 row-level |partial corr(inc, x_day | C)| 0.01-0.05 at LIRF, < 0.03 elsewhere -> probably null; if a
rule is built, decided -0.02 ... +0.05 s with forward legs >= 0. Probe 3 day-level raw corr 0.1-0.3 (shared day effects of the two
clocks) but row-level partial |corr| < 0.03 -> null; if built, |effect| < 0.02 s.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

for k, v in {"OMP_NUM_THREADS": "2", "POLARS_MAX_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
             "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl

R = Path(__file__).resolve().parents[1]
OUT = R / "data/e213_arrival_day_state_20261008"
JOINT = R / "data/e195_pre_e81_20261007_v2/predictions.parquet"
FIN1B = R / "data/cv_preds_fin1b_hz"
LB = 6 * 2 * 252.43 * 344841
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
TOL, MIN_DAY = 6, 20
KAPPA_GRID = np.round(np.arange(-3.0, 3.0001, 0.05), 2)
KEY_AIRPORTS = ["LIRF", "LFPG", "LTFM", "EGLL"]
EXCLUDED = {"EDDF", "EHAM", "LSZH"}
STOP = 0.05


def ep(c):
    return pl.col(c).dt.epoch("s").cast(pl.Float64)


def load_raw(paths):
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", Y]
    raw = pl.concat([pl.read_parquet(p, columns=cols) for p in paths])
    y = pl.col(Y).cast(pl.Float64)
    planned_ok = ~pl.col("ADEP_mvt").is_in(list(EXCLUDED))
    lab = pl.when((y >= 86400) & (y <= 86400 + 7200)).then(5)
    for i, c in enumerate(["SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt"], start=1):
        lab = lab.when((((y - (ep("MVT_TIME_UTC_mvt") - ep(c))).abs() <= TOL) & planned_ok).fill_null(False)).then(i)
    dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col(ID).cast(pl.Int64), "ADEP_mvt", has_nm=pl.col("FLIGHT_ID_mvt").is_not_null(), t=ep("MVT_TIME_UTC_mvt"),
        day=(ep("MVT_TIME_UTC_mvt") // 86400).cast(pl.Int64), month=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64),
        label_cls=lab.otherwise(0), own_p=ep("MVT_TIME_UTC_mvt") - ep("AOBT_3_flt"), res_ab=ep("AOBT_3_flt") - ep("BLOCK_TIME_UTC_mvt"))
    arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
        ap=pl.col("ADES_mvt"), day=(ep("MVT_TIME_UTC_mvt") // 86400).cast(pl.Int64), month=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64),
        a_sched=((ep("BLOCK_TIME_UTC_mvt") - ep("SCHED_TIME_UTC_mvt")).abs() <= TOL).fill_null(False),
        a_arvt1=((ep("BLOCK_TIME_UTC_mvt") - ep("ARVT_1_flt")).abs() <= TOL).fill_null(False),
        a_arvt3=((ep("BLOCK_TIME_UTC_mvt") - ep("ARVT_3_flt")).abs() <= TOL).fill_null(False),
        off=ep("ARVT_3_flt") - ep("MVT_TIME_UTC_mvt"), taxi_in=pl.col(Y).cast(pl.Float64))
    return dep, arr


def day_table(arr: pl.DataFrame) -> pl.DataFrame:
    """Per (airport, day): arrival copy shares and the NM-vs-APDF landing offset median; undefined when < MIN_DAY rows."""
    g = arr.group_by("ap", "day").agg(n_arr=pl.len(), month=pl.col("month").first(), a_sched=pl.col("a_sched").mean(),
                                      a_arvt1=pl.col("a_arvt1").mean(), a_arvt3=pl.col("a_arvt3").mean(),
                                      n_off=pl.col("off").is_not_null().sum(), off=pl.col("off").median(),
                                      arr_taxi_mean=pl.col("taxi_in").filter(pl.col("taxi_in").is_between(0, 7200)).mean())
    return g.with_columns([pl.when(pl.col("n_arr") >= MIN_DAY).then(pl.col(c)).alias(c) for c in ("a_sched", "a_arvt1", "a_arvt3")]
                          + [pl.when(pl.col("n_off") >= MIN_DAY).then(pl.col("off")).alias("off")])


def centre_cols(d: pl.DataFrame, months_ref, col: str, by_month: bool) -> pl.Series:
    """col minus its mean over (airport[, month]); the airport mean uses only months_ref (training months). Undefined -> 0."""
    if by_month:
        m = pl.col(col).mean().over("ADEP_mvt", "month")
    else:
        m = pl.when(pl.col("month").is_in(list(months_ref))).then(pl.col(col)).mean().over("ADEP_mvt")
    return d.select((pl.col(col) - m).fill_null(0.0).alias(col)).to_series()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


CONTROLS = ["c_ndep", "c_narr", "c_arrtaxi", "c_medpred"]


def partial_corr(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    """corr of the OLS residuals of a and b on [1, C] (rows with everything finite)."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(C).all(axis=1)
    if m.sum() < C.shape[1] + 3:
        return float("nan")
    X = np.column_stack([np.ones(m.sum()), C[m]])
    ba, *_ = np.linalg.lstsq(X, a[m], rcond=None)
    bb, *_ = np.linalg.lstsq(X, b[m], rcond=None)
    return corr(a[m] - X @ ba, b[m] - X @ bb)


def add_controls(d: pl.DataFrame) -> pl.DataFrame:
    """Day-level congestion controls, centred by airport x month: departures, arrivals, mean arrival taxi-in, median chain prediction."""
    g = d.group_by("ADEP_mvt", "day").agg(c_ndep=pl.len().cast(pl.Float64), c_medpred=pl.col("s").median())
    d = d.join(g, on=["ADEP_mvt", "day"], how="left").with_columns(c_narr=pl.col("n_arr").cast(pl.Float64), c_arrtaxi=pl.col("arr_taxi_mean"))
    return d.with_columns([(pl.col(c) - pl.col(c).mean().over("ADEP_mvt", "month")).alias(c) for c in CONTROLS])


def descriptive(d: pl.DataFrame, days: pl.DataFrame) -> dict:
    """Both probes; centring by the airport mean over all 2025 (x_day/off_a) and by airport x month (x_dm/off_c); partial correlations
    after the day-level congestion controls (airport x month centred) are the test statistics of the stop rule."""
    out = {"probe2": {}, "probe3": {}}
    allm = list(range(1, 13))
    dd = d.filter(pl.col("ok") & pl.col("has_proba")).with_columns(inc=pl.col("label") - pl.col("p_copy"))
    for sig in ("a_sched", "a_arvt1", "a_arvt3"):
        dd = dd.with_columns(pl.Series(f"{sig}_day", centre_cols(dd, allm, sig, False)), pl.Series(f"{sig}_dm", centre_cols(dd, allm, sig, True)))
    dd = dd.with_columns(inc_c=pl.col("inc") - pl.col("inc").mean().over("ADEP_mvt", "month"))
    for (ap,), g in sorted(dd.group_by(["ADEP_mvt"]), key=lambda z: z[0]):
        row = {"n": int(g.height), "label_share": float(g["label"].mean()), "mean_p": float(g["p_copy"].mean())}
        Cm = g.select(CONTROLS).to_numpy()
        for sig in ("a_sched", "a_arvt1", "a_arvt3"):
            row[f"pcorr_inc_{sig}"] = partial_corr(g["inc_c"].to_numpy(), g[f"{sig}_dm"].to_numpy(), Cm)
        row["pcorr_inc_a_sched_nocontrols"] = corr(g["inc_c"].to_numpy(), g["a_sched_dm"].to_numpy())
        for sig in ("a_sched", "a_arvt1", "a_arvt3"):
            for c in ("day", "dm"):
                x = g[f"{sig}_{c}"].to_numpy()
                row[f"corr_inc_{sig}_{c}"] = corr(g["inc"].to_numpy(), x)
                row[f"corr_label_{sig}_{c}"] = corr(g["label"].to_numpy().astype(float), x)
                row[f"corr_p_{sig}_{c}"] = corr(g["p_copy"].to_numpy(), x)
        dl = g.group_by("day").agg(ls=pl.col("label").mean(), mp=pl.col("p_copy").mean(), x=pl.col("a_sched_day").first(),
                                   xm=pl.col("a_sched_dm").first(), n=pl.len(), inc_c=pl.col("inc_c").mean(),
                                   *[pl.col(c).first() for c in CONTROLS]).filter(pl.col("n") >= MIN_DAY)
        row["days"] = int(dl.height)
        row["daylevel_pcorr_inc"] = partial_corr(dl["inc_c"].to_numpy(), dl["xm"].to_numpy(), dl.select(CONTROLS).to_numpy())
        row["daylevel_corr_inc_day"] = corr((dl["ls"] - dl["mp"]).to_numpy(), dl["x"].to_numpy())
        row["daylevel_corr_inc_dm"] = corr((dl["ls"] - dl["mp"]).to_numpy(), dl["xm"].to_numpy())
        row["daylevel_corr_label_day"] = corr(dl["ls"].to_numpy(), dl["x"].to_numpy())
        out["probe2"][ap] = row
    # probe 3
    d3 = d.filter(pl.col("ok") & pl.col("anchor") & pl.col(Y).is_between(1, 7200) & (pl.col("s") <= 7200)).with_columns(
        r=pl.col(Y).cast(pl.Float64) - pl.col("s"))
    d3 = d3.with_columns(rc=pl.col("r") - pl.col("r").mean().over("ADEP_mvt", "month"),
                         off_c=pl.Series(centre_cols(d3, allm, "off", True)), off_a=pl.Series(centre_cols(d3, allm, "off", False)))
    dayres = d.filter(pl.col("res_ab").is_not_null()).group_by("ADEP_mvt", "day").agg(res=pl.col("res_ab").median(), n=pl.len(),
                                                                                       month=pl.col("month").first())
    dayres = dayres.filter(pl.col("n") >= MIN_DAY).with_columns(res_c=pl.col("res") - pl.col("res").mean().over("ADEP_mvt", "month"))
    dayres = dayres.join(days.select(pl.col("ap").alias("ADEP_mvt"), "day", "off"), on=["ADEP_mvt", "day"], how="inner").drop_nulls("off")
    dayres = dayres.with_columns(off_c=pl.col("off") - pl.col("off").mean().over("ADEP_mvt", "month"))
    for (ap,), g in sorted(d3.group_by(["ADEP_mvt"]), key=lambda z: z[0]):
        h = dayres.filter(pl.col("ADEP_mvt") == ap)
        Cm = g.select(CONTROLS).to_numpy()
        dl = g.group_by("day").agg(rc=pl.col("rc").mean(), off_c=pl.col("off_c").first(), n=pl.len(),
                                   *[pl.col(c).first() for c in CONTROLS]).filter(pl.col("n") >= MIN_DAY)
        out["probe3"][ap] = {"n_rows": int(g.height), "pcorr_resid_off_c": partial_corr(g["rc"].to_numpy(), g["off_c"].to_numpy(), Cm),
                             "corr_resid_off_c": corr(g["rc"].to_numpy(), g["off_c"].to_numpy()),
                             "corr_resid_off_a": corr(g["r"].to_numpy(), g["off_a"].to_numpy()),
                             "daylevel_pcorr_chainresid_off_c": partial_corr(dl["rc"].to_numpy(), dl["off_c"].to_numpy(), dl.select(CONTROLS).to_numpy()),
                             "daylevel_corr_chainresid_off_c": corr(dl["rc"].to_numpy(), dl["off_c"].to_numpy()),
                             "days": int(h.height), "daylevel_corr_resdep_off_c": corr(h["res_c"].to_numpy(), h["off_c"].to_numpy()),
                             "off_sd_within_month": float(h["off_c"].std()) if h.height > 2 else float("nan"),
                             "resdep_sd_within_month": float(h["res_c"].std()) if h.height > 2 else float("nan")}
    return out


def shifted_delta(p: np.ndarray, v: np.ndarray, kappa_x: np.ndarray) -> np.ndarray:
    """delta of the mixture when the four planning-class logits are shifted by kappa_x (per row)."""
    w = np.ones_like(p)
    w[:, 1:5] = np.exp(kappa_x)[:, None]
    pp = p * w
    pp /= pp.sum(axis=1, keepdims=True)
    dv = np.nan_to_num(v - v[:, :1])
    return ((pp - p) * dv).sum(axis=1)


def fit_kappa(g: pl.DataFrame) -> float:
    p, v, x = g["proba"].to_numpy(), g["values"].to_numpy(), g["x"].to_numpy()
    s, y = g["s"].to_numpy(), g[Y].cast(pl.Float64).to_numpy()
    best, bk = None, 0.0
    for k in KAPPA_GRID:
        sse = float((((np.rint(s + shifted_delta(p, v, k * x)) - y) ** 2)).sum())
        if best is None or sse < best - 1e-9:
            best, bk = sse, float(k)
    return bk


def eval_split(d: pl.DataFrame, train_months, test_months, probe: int) -> dict:
    if probe == 2:
        x = centre_cols(d, train_months, "a_sched", False)
        dd = d.with_columns(x=x)
        fitpop = dd.filter(pl.col("month").is_in(train_months) & pl.col("decided") & pl.col("has_proba") & ~pl.col("lirf_nonm"))
        param = {ap: fit_kappa(g) for (ap,), g in fitpop.group_by(["ADEP_mvt"])}
        te = dd.filter(pl.col("month").is_in(test_months) & pl.col("ok"))
        appl = (te["decided"] & te["has_proba"] & ~te["lirf_nonm"]).to_numpy()
        kx = np.array([param.get(a, 0.0) for a in te["ADEP_mvt"].to_list()]) * te["x"].to_numpy()
        corr_v = np.where(appl, shifted_delta(te["proba"].to_numpy(), te["values"].to_numpy(), kx), 0.0)
    else:
        x = centre_cols(d, train_months, "off", True)
        dd = d.with_columns(x=x)
        fitpop = dd.filter(pl.col("month").is_in(train_months) & pl.col("ok") & pl.col("anchor") & pl.col(Y).is_between(1, 7200)
                           & (pl.col("s") <= 7200))
        param = {}
        for (ap,), g in fitpop.group_by(["ADEP_mvt"]):
            xx, rr = g["x"].to_numpy(), (g[Y].cast(pl.Float64) - g["s"]).to_numpy()
            param[ap] = float((xx * rr).sum() / (xx * xx).sum()) if (xx * xx).sum() > 0 else 0.0
        te = dd.filter(pl.col("month").is_in(test_months) & pl.col("ok"))
        appl = (te["anchor"] & te["decided"]).to_numpy()
        corr_v = np.where(appl, np.array([param.get(a, 0.0) for a in te["ADEP_mvt"].to_list()]) * te["x"].to_numpy(), 0.0)
    s, y = te["s"].to_numpy(), te[Y].cast(pl.Float64).to_numpy()
    p2 = np.clip(np.rint(s + corr_v), 0, None)
    dsq = (p2 - y) ** 2 - (s - y) ** 2
    dec, body = te["decided"].to_numpy(), y <= 7200
    sd = np.sort(dsq[dec])
    return dict(test=list(test_months), param=param, rows=int(te.height), applied=int(appl.sum()),
                rms_corr=float(np.sqrt(np.mean(corr_v[appl] ** 2))) if appl.any() else 0.0, moved=int((p2 != s).sum()),
                all=float(dsq.sum()), decided=float(dsq[dec].sum()), body=float(dsq[body].sum()),
                trim20=float(sd[20:-20].sum()), trim3=float(sd[3:-3].sum()),
                by_airport={ap: float(dsq[(te["ADEP_mvt"] == ap).to_numpy() & dec].sum()) for ap in sorted(set(te["ADEP_mvt"].to_list()))})


def run_arm(d: pl.DataFrame, splits, probe: int) -> dict:
    folds = [eval_split(d, tr, te, probe) for tr, te in splits]
    pooled = {k: sum(f[k] for f in folds) / LB for k in ("all", "decided", "body", "trim20", "trim3")}
    pooled["lower_decided"] = int(sum(f["decided"] < 0 for f in folds))
    pooled["by_airport"] = {ap: sum(f["by_airport"].get(ap, 0.0) for f in folds) / LB for ap in folds[0]["by_airport"]}
    return dict(folds=folds, pooled=pooled)


def gate(pairs: dict, half: dict, expanding: dict, gap: dict) -> dict:
    p, f = pairs["pooled"], pairs["folds"]
    by_pair = {tuple(x["test"]): x["decided"] for x in f}
    legs = dict(pairs_lower=p["lower_decided"] >= 4, pooled_decided=p["decided"] < 0, pooled_body=p["body"] < 0, pooled_trim20=p["trim20"] < 0,
                pair_2_8=by_pair[(2, 8)] <= 0, pair_6_12=by_pair[(6, 12)] <= 0, half_forward=half["folds"][0]["decided"] <= 0,
                expanding=expanding["pooled"]["decided"] <= 0, gap=gap["pooled"]["decided"] <= 0)
    legs["PASS"] = all(legs.values())
    return legs


def fmt(res: dict) -> str:
    p = res["pooled"]
    return (f"decided {p['decided']:+.3f} s ({p['lower_decided']}/{len(res['folds'])}) | trim20 {p['trim20']:+.3f} | body {p['body']:+.3f} | "
            f"all {p['all']:+.3f} | folds " + " ".join(f"{tuple(f['test'])}:{f['decided'] / LB:+.3f}" for f in res["folds"]))


def run_probe(d: pl.DataFrame, probe: int, tag: str, result: dict):
    pair_splits = [([m for m in range(1, 13) if m not in p], list(p)) for p in PAIRS]
    pairs = run_arm(d, pair_splits, probe)
    half = run_arm(d, [([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]), ([7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6])], probe)
    expanding = run_arm(d, [(list(range(1, k)), [k]) for k in range(4, 13)], probe)
    gap = run_arm(d, [(list(range(1, k - 1)), [k]) for k in range(4, 13)], probe)
    g = gate(pairs, half, expanding, gap)
    result[tag] = dict(pairs=pairs, half_year=half, expanding=expanding, gap=gap, gate=g)
    print(f"[{tag}/pairs] {fmt(pairs)}", flush=True)
    print(f"[{tag}/pairs] by airport: " + " ".join(f"{a} {v:+.3f}" for a, v in pairs["pooled"]["by_airport"].items()), flush=True)
    print(f"[{tag}/pairs] params: " + "; ".join(f"{tuple(f['test'])}: " + " ".join(f"{a} {v:+.2f}" for a, v in sorted(f["param"].items()))
                                              for f in pairs["folds"]), flush=True)
    print(f"[{tag}/half-year] forward {half['folds'][0]['decided'] / LB:+.3f} (body {half['folds'][0]['body'] / LB:+.3f}); "
          f"backward {half['folds'][1]['decided'] / LB:+.3f}", flush=True)
    print(f"[{tag}/expanding 1..k-1 -> k] {fmt(expanding)}", flush=True)
    print(f"[{tag}/gap 1..k-2 -> k] {fmt(gap)}", flush=True)
    print(f"[{tag}] GATE {'PASS' if g['PASS'] else 'FAIL'} {g}", flush=True)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dep, arr = load_raw(sorted(glob.glob(str(R / "data/prc-2026-datasets/training_*.parquet"))))
    days = day_table(arr)
    oof = pl.read_parquet(JOINT).with_columns(pl.col(ID).cast(pl.Int64))
    d = oof.join(dep, on=ID, how="inner", maintain_order="left")
    if d.height != oof.height:
        raise ValueError("departure rows missing for the joint OOF")
    fin = {}
    for f in sorted(glob.glob(str(FIN1B / "fold_*.npz"))):
        z = np.load(f)
        ids = z["mvt_id"].astype(np.int64)
        fin.update({int(i): (p, v, q) for i, p, v, q in zip(ids.tolist(), z["proba"].astype(np.float64), z["values"].astype(np.float64),
                                                              z["pred"].astype(np.float64))})
    keys = d[ID].to_list()
    has = np.array([k in fin for k in keys])
    P = np.array([fin[k][0] if k in fin else np.full(6, np.nan) for k in keys])
    V = np.array([fin[k][1] if k in fin else np.full(6, np.nan) for k in keys])
    fpred = np.array([fin[k][2] if k in fin else np.nan for k in keys])
    d = d.with_columns(pl.Series("has_proba", has), pl.Series("proba", P), pl.Series("values", V), pl.Series("fin1b", fpred),
                       pl.Series("p_copy", 1.0 - P[:, 0]), label=(pl.col("label_cls") > 0).cast(pl.Float64),
                       ok=pl.col("evaluation_mask"), s=pl.col("joint").cast(pl.Float64), lirf_nonm=(pl.col("ADEP_mvt") == "LIRF") & ~pl.col("has_nm"))
    d = d.with_columns(decided=pl.col("ok") & ~(pl.col("lirf_nonm") & (pl.col("D") >= 14400)))
    d = d.join(days.select(pl.col("ap").alias("ADEP_mvt"), "day", "a_sched", "a_arvt1", "a_arvt3", "off", "n_arr", "arr_taxi_mean"),
               on=["ADEP_mvt", "day"], how="left")
    d = add_controls(d)
    d = d.with_columns(anchor=pl.col("own_p").is_between(0, 7200) & (pl.col("proba").arr.get(0) >= 0.5) & (pl.col("s") <= 7200)
                       & ~pl.col("lirf_nonm") & pl.col("has_proba"))
    print(f"rows {d.height:,}; with FIN1b proba {int(has.sum()):,}; arrivals {arr.height:,}; airport-days {days.height:,} "
          f"(a_sched defined {int(days['a_sched'].is_not_null().sum()):,}, off defined {int(days['off'].is_not_null().sum()):,}); "
          f"label share {d['label'].mean():.4f}, mean p_copy {float(np.nanmean(d['p_copy'].to_numpy())):.4f}; anchor rows {int(d['anchor'].sum()):,}; "
          f"load {time.time() - t0:.0f}s", flush=True)
    result = dict(protocol="E213-arrival-day-state-v1-20261008", tol=TOL, min_day=MIN_DAY, stop=STOP, key_airports=KEY_AIRPORTS)
    result["arrival_shares_by_airport"] = {ap: {k: float(v) for k, v in row.items() if k != "ap"} for ap, row in
                                           ((r["ap"], r) for r in arr.group_by("ap").agg(a_sched=pl.col("a_sched").mean(), a_arvt1=pl.col("a_arvt1").mean(),
                                                                                       a_arvt3=pl.col("a_arvt3").mean(), off_med=pl.col("off").median(),
                                                                                       off_q10=pl.col("off").quantile(0.1), off_q90=pl.col("off").quantile(0.9)).sort("ap").to_dicts())}
    desc = descriptive(d, days)
    result["descriptive"] = desc
    print("PROBE 2 descriptive (pcorr = row-level partial corr of inc = label - p with the arrival share | congestion controls, airport x month centred;"
          " raw corr with x_day = airport-mean centred, x_dm = airport x month centred):", flush=True)
    for ap, row in desc["probe2"].items():
        print(f"  {ap}: n {row['n']:,} label {row['label_share']:.4f} p {row['mean_p']:.4f} | PCORR inc~a_sched {row['pcorr_inc_a_sched']:+.4f} "
              f"(no controls {row['pcorr_inc_a_sched_nocontrols']:+.4f}; arvt1 {row['pcorr_inc_a_arvt1']:+.4f} arvt3 {row['pcorr_inc_a_arvt3']:+.4f}; "
              f"day-level {row['daylevel_pcorr_inc']:+.3f}) | raw: inc~a_sched day {row['corr_inc_a_sched_day']:+.4f} "
              f"dm {row['corr_inc_a_sched_dm']:+.4f} | label~a_sched day {row['corr_label_a_sched_day']:+.4f} | p~a_sched day {row['corr_p_a_sched_day']:+.4f} | "
              f"inc~arvt1 day {row['corr_inc_a_arvt1_day']:+.4f} inc~arvt3 day {row['corr_inc_a_arvt3_day']:+.4f} | "
              f"day-level inc {row['daylevel_corr_inc_day']:+.3f} (dm {row['daylevel_corr_inc_dm']:+.3f}) label {row['daylevel_corr_label_day']:+.3f} days {row['days']}", flush=True)
    print("PROBE 3 descriptive (row-level corr of chain residual with off_c; day-level corr of median(AOBT_3 - BLOCK) with off_c):", flush=True)
    for ap, row in desc["probe3"].items():
        print(f"  {ap}: rows {row['n_rows']:,} PCORR resid~off_c|C {row['pcorr_resid_off_c']:+.4f} (raw {row['corr_resid_off_c']:+.4f}, off_a {row['corr_resid_off_a']:+.4f}; "
              f"day-level chain resid pcorr {row['daylevel_pcorr_chainresid_off_c']:+.3f} raw {row['daylevel_corr_chainresid_off_c']:+.3f}) | raw anchor "
              f"day-level {row['daylevel_corr_resdep_off_c']:+.3f} (days {row['days']}, sd off_c {row['off_sd_within_month']:.1f} s, "
              f"sd res_dep_c {row['resdep_sd_within_month']:.1f} s)", flush=True)
    key2 = max(abs(desc["probe2"][a]["pcorr_inc_a_sched"]) for a in KEY_AIRPORTS if a in desc["probe2"])
    key3 = max(abs(desc["probe3"][a]["pcorr_resid_off_c"]) for a in KEY_AIRPORTS if a in desc["probe3"])
    result["stop"] = {"probe2_max_key_corr": key2, "probe3_max_key_corr": key3, "probe2_build": bool(key2 >= STOP), "probe3_build": bool(key3 >= STOP)}
    print(f"STOP RULE (partial corr | controls): probe 2 max |pcorr| over key airports {key2:.4f} -> {'BUILD' if key2 >= STOP else 'NULL, no rule'}; "
          f"probe 3 {key3:.4f} -> {'BUILD' if key3 >= STOP else 'NULL, no rule'}", flush=True)
    if key2 >= STOP:
        run_probe(d, 2, "probe2/joint", result)
        run_probe(d.filter(pl.col("has_proba")).with_columns(s=pl.col("fin1b")), 2, "probe2/fin1b", result)
    if key3 >= STOP:
        run_probe(d, 3, "probe3/joint", result)
        run_probe(d.filter(pl.col("has_proba")).with_columns(s=pl.col("fin1b")), 3, "probe3/fin1b", result)
    result["seconds"] = time.time() - t0
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float))
    days.write_parquet(OUT / "airport_day_table.parquet")
    print(f"done {result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
