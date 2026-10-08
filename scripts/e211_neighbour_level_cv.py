"""E211 (journal 399): level correction by the neighbours' label-free model residual, decayed by time distance and centred
within (airport, UTC day) — the Motoki/CityLearn "alpha^h level correction" in the smart-jigsaw §70–72 form, applied to the
production chain OOF (E195 joint, through E152, pre-E81). Definitions and the gate are written here BEFORE the run and
committed together with the journal line.

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python scripts/e211_neighbour_level_cv.py > data/e211_neighbour_level_20261008/run.log

Observable at prediction time (identical for 2025 OOF and the 2026 inputs — no label of the row, no label of any departure):
  S1  nbp_apt : departures j != i of the same airport, v_j = clip((MVT - AOBT_3)_j - s_j, +-1800) where (MVT - AOBT_3)_j in [1, 5399];
                weight w = exp(-|t_i - t_j| / TAU), TAU = 1800 s (alpha per 10 min = exp(-600/1800) = 0.717), |t_i - t_j| <= W = 3600 s,
                defined when >= 2 neighbours carry a value; s_j = the same chain prediction that is being corrected.
  S2  arrx    : arrivals of the same airport, v_j = clip(taxi_in_j - normal_j, +-1800); normal = median taxi-in per (ADES, RUNWAY, STAND)
                (>= 20 rows) else (ADES, RUNWAY) else ADES from the OTHER months of 2025 (leave-month-out, as U1); same kernel on
                |t_i - landing_j|. Arrivals carry their taxi-in in the ranking file, so this is observable in 2026 as well.
  S3  nbp_rwy : S1 restricted to the same (airport, runway).
  Each S_k = decayed mean minus its mean over (airport, UTC day of MVT_TIME) among defined rows; undefined -> 0.
Correction: pred' = rint(pred + sum_k beta_{airport,k} S_k); beta by OLS (no intercept) of r = y - pred on (S1, S2, S3) per airport, fitted
ONLY on the training months of each fold (the ten months outside the pair), rows evaluation_mask & 0 < y <= 7200 & pred <= 7200.
Applied to held-out rows with evaluation_mask & pred <= 7200, except LIRF without an NM record (the tail chain owns them).
Metric: paired squared error against the unchanged chain, per fold; LB-equivalent seconds = sum / (6 * 2 * 252.43 * 344841).
GATE (primary arm = joint OOF, S1+S2+S3, beta per airport): decided lower in >= 4/6 folds; pooled decided < 0; pooled trim20 (decided,
without the 20 best and 20 worst rows of each fold) < 0; pooled body (y <= 7200) < 0; folds 2+8 and 6+12 decided <= 0.
Reported, not gating: single-signal arms, global beta, the half-year transfer (fit Jan–Jun -> apply Jul–Dec and the reverse; E183's
NM proxy failed exactly this leg), and the same primary arm on the FIN1b base OOF (data/cv_preds_fin1b_hz).
Prediction written before the run: pooled decided -0.05 … -0.20 s, 4–5/6 folds; half-year forward leg weaker or positive.
Also writes beta fitted on all twelve months (for the optional default-off stage) and the per-airport/month correlations.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

for k, v in {"OMP_NUM_THREADS": "2", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
             "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl

R = Path(__file__).resolve().parents[1]
OUT = R / "data/e211_neighbour_level_20261008"
JOINT = R / "data/e195_pre_e81_20261007_v2/predictions.parquet"
FIN1B = R / "data/cv_preds_fin1b_hz"
TAU, W, MIN_NB, CLIP = 1800.0, 3600.0, 2, 1800.0
LB = 6 * 2 * 252.43 * 344841
PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
HALVES = [([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]), ([7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6])]
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
SIGNALS = ["S1", "S2", "S3"]


def ep(c):
    return pl.col(c).dt.epoch("s").cast(pl.Float64)


def load_raw(paths):
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", Y]
    raw = pl.concat([pl.read_parquet(p, columns=cols) for p in paths])
    dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select(
        pl.col(ID).cast(pl.Int64), "ADEP_mvt", "RUNWAY_mvt", has_nm=pl.col("FLIGHT_ID_mvt").is_not_null(),
        t=ep("MVT_TIME_UTC_mvt"), own_p=ep("MVT_TIME_UTC_mvt") - ep("AOBT_3_flt"))
    arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
        ap=pl.col("ADES_mvt"), rwy=pl.col("RUNWAY_mvt"), stand=pl.col("STAND_mvt"), t=ep("MVT_TIME_UTC_mvt"),
        month=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64), x=pl.col(Y).cast(pl.Float64))
    return dep, arr


def arrival_residuals(arr: pl.DataFrame, normal_from: pl.DataFrame | None = None) -> pl.DataFrame:
    """clip(taxi_in - normal, +-CLIP); normal leave-month-out from arr itself, or from normal_from (the 2026 stage use)."""
    arr = arr.filter(pl.col("x").is_between(0, 7200))
    parts = []
    for m in sorted(arr["month"].unique().to_list()):
        o = arr.filter(pl.col("month") != m) if normal_from is None else normal_from.filter(pl.col("x").is_between(0, 7200))
        n3 = o.group_by("ap", "rwy", "stand").agg(n3=pl.col("x").median(), c3=pl.len())
        n2 = o.group_by("ap", "rwy").agg(n2=pl.col("x").median())
        n1 = o.group_by("ap").agg(n1=pl.col("x").median())
        a = arr.filter(pl.col("month") == m).join(n3, on=["ap", "rwy", "stand"], how="left").join(n2, on=["ap", "rwy"], how="left") \
            .join(n1, on="ap", how="left")
        parts.append(a.with_columns(normal=pl.when(pl.col("c3") >= 20).then(pl.col("n3")).otherwise(pl.coalesce("n2", "n1"))))
    return pl.concat(parts).with_columns(v=(pl.col("x") - pl.col("normal")).clip(-CLIP, CLIP)).drop_nulls("v").select("ap", "t", "v")


def one_side(t: np.ndarray, v: np.ndarray, is_src: np.ndarray):
    """Processing order given. For every position: sum of exp(-(t_k - t_j)/TAU) v_j, the weight sum and the count over SOURCE positions
    j < k with t_k - t_j <= W (t non-decreasing). Exact (amortised deque), brute-force checked in selftest()."""
    n = len(t)
    F, Wt, N = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)
    f = wt = 0.0
    cnt = 0
    dq: deque = deque()
    tl, vl, sl = t.tolist(), v.tolist(), is_src.tolist()
    last = None
    for k in range(n):
        tk = tl[k]
        if last is not None and tk != last:
            dec = math.exp(-(tk - last) / TAU)
            f *= dec
            wt *= dec
        while dq and tk - dq[0][0] > W:
            tj, vj = dq.popleft()
            w = math.exp(-(tk - tj) / TAU)
            f -= w * vj
            wt -= w
            cnt -= 1
        F[k], Wt[k], N[k] = f, wt, cnt
        if sl[k]:
            f += vl[k]
            wt += 1.0
            cnt += 1
            dq.append((tk, vl[k]))
        last = tk
    return F, Wt, N


def decayed_mean(t_q: np.ndarray, t_s: np.ndarray, v_s: np.ndarray, same_set: bool) -> np.ndarray:
    """Decayed mean of v_s over sources within W of each query (both sides, self excluded when same_set); NaN unless >= MIN_NB."""
    ok = ~np.isnan(v_s)
    if same_set:
        # one array: sources are the queries themselves (where defined). Self excluded by strict index order.
        order = np.argsort(t_q, kind="stable")
        t, v, src = t_q[order], np.where(ok, v_s, 0.0)[order], ok[order]
        Ff, Wf, Nf = one_side(t, v, src)
        Fb, Wb, Nb = one_side(-t[::-1], v[::-1], src[::-1])
        Fb, Wb, Nb = Fb[::-1], Wb[::-1], Nb[::-1]
        out = np.full(len(t_q), np.nan)
        num, den, cnt = Ff + Fb, Wf + Wb, Nf + Nb
        good = cnt >= MIN_NB
        res = np.where(good, num / np.where(den > 0, den, 1.0), np.nan)
        out[order] = res
        return out
    t_s, v_s = t_s[ok], v_s[ok]
    nq, ns = len(t_q), len(t_s)
    t = np.concatenate([t_s, t_q])
    v = np.concatenate([v_s, np.zeros(nq)])
    src = np.concatenate([np.ones(ns, bool), np.zeros(nq, bool)])
    flag = np.concatenate([np.zeros(ns), np.ones(nq)])  # sources first at ties
    order = np.lexsort((flag, t))
    Ff, Wf, Nf = one_side(t[order], v[order], src[order])
    order_b = np.lexsort((-flag, -t))  # queries first at ties: a source at exactly t_q is counted once (forward pass)
    Fb, Wb, Nb = one_side(-t[order_b], v[order_b], src[order_b])
    F = np.zeros(len(t))
    Wt = np.zeros(len(t))
    N = np.zeros(len(t), dtype=np.int64)
    F[order] += Ff
    Wt[order] += Wf
    N[order] += Nf
    F[order_b] += Fb
    Wt[order_b] += Wb
    N[order_b] += Nb
    Fq, Wq, Nq = F[ns:], Wt[ns:], N[ns:]
    return np.where(Nq >= MIN_NB, Fq / np.where(Wq > 0, Wq, 1.0), np.nan)


def centre(df: pl.DataFrame, col: str) -> pl.Series:
    m = pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col))
    return df.select((m - m.mean().over("ADEP_mvt", "day")).fill_null(0.0).alias(col)).to_series()


def build_signals(dep: pl.DataFrame, s: np.ndarray, arr_res: pl.DataFrame) -> pl.DataFrame:
    """dep: ID, ADEP_mvt, RUNWAY_mvt, t, own_p (row order kept). s: chain prediction aligned with dep. Returns S1..S3 centred + raw means."""
    d = dep.with_columns(s=pl.Series(s.astype(np.float64)), day=(pl.col("t") // 86400).cast(pl.Int64), _i=pl.int_range(pl.len()))
    d = d.with_columns(v=pl.when(pl.col("own_p").is_between(1, 5399)).then((pl.col("own_p") - pl.col("s")).clip(-CLIP, CLIP)))
    m1, m2, m3 = (np.full(d.height, np.nan) for _ in range(3))
    for (ap,), g in d.group_by(["ADEP_mvt"]):
        idx, t = g["_i"].to_numpy(), g["t"].to_numpy()
        v = g["v"].fill_null(np.nan).to_numpy().astype(np.float64)
        m1[idx] = decayed_mean(t, t, v, same_set=True)
        a = arr_res.filter(pl.col("ap") == ap)
        m2[idx] = decayed_mean(t, a["t"].to_numpy(), a["v"].to_numpy(), same_set=False)
        for (rw,), h in g.group_by(["RUNWAY_mvt"]):
            j, th = h["_i"].to_numpy(), h["t"].to_numpy()
            vh = h["v"].fill_null(np.nan).to_numpy().astype(np.float64)
            m3[j] = decayed_mean(th, th, vh, same_set=True)
    d = d.with_columns(m1=pl.Series(m1), m2=pl.Series(m2), m3=pl.Series(m3))
    return d.with_columns(S1=centre(d, "m1"), S2=centre(d, "m2"), S3=centre(d, "m3"))


def fit_beta(df: pl.DataFrame, cols, per_airport: bool) -> dict:
    """OLS without intercept of r on cols; rows already filtered to the fitting population."""
    out = {}
    groups = df.group_by(["ADEP_mvt"]) if per_airport else [(("ALL",), df)]
    for (ap,), g in groups:
        X = g.select(cols).to_numpy()
        r = (g[Y].cast(pl.Float64) - g["s"]).to_numpy()
        b, *_ = np.linalg.lstsq(X, r, rcond=None)
        out[ap] = [float(x) for x in b]
    return out


def apply_beta(df: pl.DataFrame, beta: dict, cols) -> np.ndarray:
    keys = df["ADEP_mvt"].to_list() if "ALL" not in beta else ["ALL"] * df.height
    B = np.array([beta.get(k, [0.0] * len(cols)) for k in keys])
    return (df.select(cols).to_numpy() * B).sum(axis=1)


def fold_eval(d: pl.DataFrame, test_months, train_months, cols, per_airport: bool) -> dict:
    fitpop = d.filter(pl.col("month").is_in(train_months) & pl.col("ok") & pl.col(Y).is_between(1, 7200) & (pl.col("s") <= 7200))
    beta = fit_beta(fitpop, cols, per_airport)
    te = d.filter(pl.col("month").is_in(test_months) & pl.col("ok"))
    appl = ((te["s"] <= 7200) & ~((te["ADEP_mvt"] == "LIRF") & ~te["has_nm"])).to_numpy()
    corr_v = np.where(appl, apply_beta(te, beta, cols), 0.0)
    s, y = te["s"].to_numpy(), te[Y].cast(pl.Float64).to_numpy()
    p2 = np.clip(np.rint(s + corr_v), 0, None)
    dsq = (p2 - y) ** 2 - (s - y) ** 2
    dec = te["decided"].to_numpy()
    body = y <= 7200
    sd = np.sort(dsq[dec])
    return dict(test=list(test_months), beta=beta, rows=int(te.height), applied=int(appl.sum()),
                rms_corr=float(np.sqrt(np.mean(corr_v[appl] ** 2))) if appl.any() else 0.0,
                all=float(dsq.sum()), decided=float(dsq[dec].sum()), body=float(dsq[body].sum()),
                trim20=float(sd[20:-20].sum()), trim3=float(sd[3:-3].sum()),
                by_airport={ap: float(dsq[(te["ADEP_mvt"] == ap).to_numpy() & dec].sum()) for ap in sorted(set(te["ADEP_mvt"].to_list()))})


def run_arm(d: pl.DataFrame, cols, per_airport: bool, splits) -> dict:
    folds = [fold_eval(d, te, tr, cols, per_airport) for tr, te in splits]
    pooled = {k: sum(f[k] for f in folds) / LB for k in ("all", "decided", "body", "trim20", "trim3")}
    pooled["lower_decided"] = int(sum(f["decided"] < 0 for f in folds))
    pooled["by_airport"] = {ap: sum(f["by_airport"].get(ap, 0.0) for f in folds) / LB for ap in folds[0]["by_airport"]}
    return dict(folds=folds, pooled=pooled)


def gate(res: dict) -> bool:
    p, f = res["pooled"], res["folds"]
    by_pair = {tuple(x["test"]): x["decided"] for x in f}
    return (p["lower_decided"] >= 4 and p["decided"] < 0 and p["trim20"] < 0 and p["body"] < 0
            and by_pair[(2, 8)] <= 0 and by_pair[(6, 12)] <= 0)


def correlations(d: pl.DataFrame) -> dict:
    """Centred (airport x day) correlation of r with each S, and the partial correlation given the other two, per airport and month."""
    out = {}
    d = d.filter(pl.col("ok") & pl.col(Y).is_between(1, 7200) & (pl.col("s") <= 7200)).with_columns(
        r=pl.col(Y).cast(pl.Float64) - pl.col("s"))
    d = d.with_columns((pl.col("r") - pl.col("r").mean().over("ADEP_mvt", "day")).alias("rc"))

    def block(g):
        X, r = g.select(SIGNALS).to_numpy(), g["rc"].to_numpy()
        row = {"n": int(g.height)}
        for i, c in enumerate(SIGNALS):
            x = X[:, i]
            row[c] = float(np.corrcoef(x, r)[0, 1]) if x.std() > 0 else float("nan")
            others = np.delete(X, i, axis=1)
            if others.std(axis=0).min() > 0 and x.std() > 0:
                bx, *_ = np.linalg.lstsq(others, x, rcond=None)
                br, *_ = np.linalg.lstsq(others, r, rcond=None)
                ex, er = x - others @ bx, r - others @ br
                row[c + "_partial"] = float(np.corrcoef(ex, er)[0, 1]) if ex.std() > 0 else float("nan")
        return row
    out["airport"] = {ap: block(g) for (ap,), g in sorted(d.group_by(["ADEP_mvt"]), key=lambda z: z[0])}
    out["month"] = {int(m): block(g) for (m,), g in sorted(d.group_by(["month"]), key=lambda z: z[0])}
    out["airport_month_S1"] = {ap: {int(m): block(h)["S1"] for (m,), h in sorted(g.group_by(["month"]), key=lambda z: z[0])}
                               for (ap,), g in sorted(d.group_by(["ADEP_mvt"]), key=lambda z: z[0])}
    out["all"] = block(d)
    return out


def selftest():
    rng = np.random.default_rng(211)
    t = np.sort(rng.uniform(0, 20000, 300))
    v = rng.normal(size=300)
    v[rng.random(300) < 0.3] = np.nan
    got = decayed_mean(t, t, v, same_set=True)
    for i in range(300):
        m = (np.abs(t - t[i]) <= W) & ~np.isnan(v)
        m[i] = False
        w = np.exp(-np.abs(t[i] - t[m]) / TAU)
        exp = np.sum(w * v[m]) / w.sum() if m.sum() >= MIN_NB else np.nan
        assert (np.isnan(exp) and np.isnan(got[i])) or abs(exp - got[i]) < 1e-9, (i, exp, got[i])
    tq = rng.uniform(0, 20000, 200)
    got = decayed_mean(tq, t, v, same_set=False)
    for i in range(200):
        m = (np.abs(t - tq[i]) <= W) & ~np.isnan(v)
        w = np.exp(-np.abs(tq[i] - t[m]) / TAU)
        exp = np.sum(w * v[m]) / w.sum() if m.sum() >= MIN_NB else np.nan
        assert (np.isnan(exp) and np.isnan(got[i])) or abs(exp - got[i]) < 1e-9, (i, exp, got[i])
    # ties: a query at exactly a source time sees that source (different sets); same-set excludes only itself
    t2 = np.array([0.0, 0.0, 100.0])
    v2 = np.array([1.0, 3.0, 5.0])
    g2 = decayed_mean(t2, t2, v2, same_set=True)
    w = math.exp(-100 / TAU)
    assert abs(g2[0] - (3 + w * 5) / (1 + w)) < 1e-12 and abs(g2[2] - 2.0) < 1e-12
    g3 = decayed_mean(np.array([0.0]), t2, v2, same_set=False)
    assert abs(g3[0] - (1 + 3 + w * 5) / (2 + w)) < 1e-12
    print("selftest passed: decayed means equal brute force; ties handled", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        return
    selftest()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dep, arr = load_raw(sorted(glob.glob(str(R / "data/prc-2026-datasets/training_*.parquet"))))
    oof = pl.read_parquet(JOINT).with_columns(pl.col(ID).cast(pl.Int64))
    d = oof.join(dep, on=ID, how="inner", maintain_order="left")
    if d.height != oof.height:
        raise ValueError("departure rows missing for the joint OOF")
    d = d.with_columns(ok=pl.col("evaluation_mask"), month=pl.col("month").cast(pl.Int64)).with_columns(
        decided=pl.col("ok") & ~((pl.col("ADEP_mvt") == "LIRF") & ~pl.col("has_nm") & (pl.col("D") >= 14400)))
    print(f"rows {d.height:,}; arrivals {arr.height:,}; load {time.time() - t0:.0f}s", flush=True)
    arr_res = arrival_residuals(arr)
    print(f"arrival residuals {arr_res.height:,}, median {arr_res['v'].median():+.0f}", flush=True)

    result = dict(protocol="E211-neighbour-level-correction-v1-20261008", tau=TAU, window=W, min_nb=MIN_NB, clip=CLIP,
                  alpha_per_10min=math.exp(-600 / TAU), arms={})
    bases = {"joint": d["joint"].cast(pl.Float64).to_numpy()}
    fin = {}
    for f in sorted(glob.glob(str(FIN1B / "fold_*.npz"))):
        z = np.load(f)
        fin.update(zip(z["mvt_id"].astype(np.int64).tolist(), z["pred"].astype(np.float64).tolist()))
    bases["fin1b"] = np.array([fin.get(k, np.nan) for k in d[ID].to_list()])
    pair_splits = [([m for m in range(1, 13) if m not in p], list(p)) for p in PAIRS]
    for base_name, s in bases.items():
        t1 = time.time()
        dd = d.select(ID, "ADEP_mvt", "RUNWAY_mvt", "t", "own_p", "has_nm", "month", "ok", "decided", Y)
        sig = build_signals(dd, s, arr_res)
        print(f"[{base_name}] signals built in {time.time() - t1:.0f}s; defined S1 {np.isfinite(sig['m1'].to_numpy()).mean():.3f} "
              f"S2 {np.isfinite(sig['m2'].to_numpy()).mean():.3f} S3 {np.isfinite(sig['m3'].to_numpy()).mean():.3f}", flush=True)
        if base_name == "fin1b":
            keep = np.isfinite(s)
            sig = sig.filter(pl.Series(keep))
            print(f"[fin1b] rows with FIN1b OOF {sig.height:,}", flush=True)
        arms = {"primary_S123_airport": (SIGNALS, True), "S1_airport": (["S1"], True), "S2_airport": (["S2"], True),
                "S3_airport": (["S3"], True), "S123_global": (SIGNALS, False)}
        if base_name == "fin1b":
            arms = {"primary_S123_airport": (SIGNALS, True)}
        for arm, (cols, pa) in arms.items():
            res = run_arm(sig, cols, pa, pair_splits)
            res["gate"] = gate(res)
            result["arms"][f"{base_name}/{arm}"] = res
            p = res["pooled"]
            print(f"[{base_name}/{arm}] decided {p['decided']:+.3f} s ({p['lower_decided']}/6) | trim20 {p['trim20']:+.3f} | "
                  f"body {p['body']:+.3f} | all {p['all']:+.3f} | folds "
                  + " ".join(f"{tuple(f['test'])}:{f['decided'] / LB:+.3f}" for f in res["folds"])
                  + f" | gate {'PASS' if res['gate'] else 'FAIL'}", flush=True)
        half = run_arm(sig, SIGNALS, True, HALVES)
        result["arms"][f"{base_name}/half_year_S123_airport"] = half
        print(f"[{base_name}/half-year] forward (fit 1-6 -> 7-12) decided {half['folds'][0]['decided'] / LB:+.3f}, body "
              f"{half['folds'][0]['body'] / LB:+.3f}; backward decided {half['folds'][1]['decided'] / LB:+.3f}, body "
              f"{half['folds'][1]['body'] / LB:+.3f}", flush=True)
        if base_name == "joint":
            result["correlations"] = correlations(sig)
            c = result["correlations"]
            print("centred corr(r, S) all rows:", {k: round(v, 4) for k, v in c["all"].items() if k != "n"}, flush=True)
            for ap, row in c["airport"].items():
                print(f"  {ap}: " + " ".join(f"{k} {v:+.3f}" for k, v in row.items() if k != "n") + f" n {row['n']:,}", flush=True)
            for m, row in c["month"].items():
                print(f"  month {m:2d}: " + " ".join(f"{k} {v:+.3f}" for k, v in row.items() if k != "n"), flush=True)
            fitpop = sig.filter(pl.col("ok") & pl.col(Y).is_between(1, 7200) & (pl.col("s") <= 7200))
            result["beta_all_2025_airport"] = fit_beta(fitpop, SIGNALS, True)
            print("beta all 2025 per airport:", {k: [round(x, 4) for x in v] for k, v in sorted(result["beta_all_2025_airport"].items())},
                  flush=True)
    result["gate_passed"] = result["arms"]["joint/primary_S123_airport"]["gate"]
    result["seconds"] = time.time() - t0
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print("GATE", "PASS" if result["gate_passed"] else "FAIL", f"| {result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
