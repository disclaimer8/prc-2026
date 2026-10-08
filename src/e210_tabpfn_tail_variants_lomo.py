"""E210 (journal 397): variants of the E156 TabPFN ensemble on the LIRF no-NM tail (tail-table bins, D 4-30 h),
leave-one-month-out over 2025, months 2 and 6 reported separately (the final adds Feb/Jun 2026).

Arms (all capacity-free, prefix design of E118 with deterministic codes, regular candidate = production OOF regressor):
  A0  seed 0                                   (one draw)
  A1  seeds 0-9 averaged                       (production E156, control against data/e156_lomo.log)
  B1  seeds 0-19 averaged
  C1  A1 + dither marginalisation: probabilities averaged over K=5 copies of each query row with
      d_mvt_sched + {-6,-3,0,3,6} s (the provider's take-off-time dither is ~U{-6..6} s, final_jitter_sensitivity §1)
  C2  B1 + the same marginalisation
Candidates and the squared error always use the observed D. Secondary (no gate): robustness to a re-draw of the
dither, query D + w', w' ~ U{-6..6} integer, 3 fixed draws per month: SSE and RMS change of the prediction per arm.

Gate against A1, fixed before the run: total LOMO-SSE lower; months 2 and 6 not higher; no month worse than A1 by
more than X = 10 %; without the top-3 rows (by A1 squared error) not higher.

    OMP_NUM_THREADS=4 .venv/bin/python src/e210_tabpfn_tail_variants_lomo.py
"""
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
import torch

THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
torch.set_num_threads(THREADS)
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

import lirf_table as L  # noqa: E402

Y = "TAXITIME_SEC_mvt"
FEATURES = os.environ.get("FEATURES", "data/features_e106.parquet")   # features_v9 is no longer on the Mac; same columns
CV_DIR = os.environ.get("CV_DIR", "data/cv_preds_e116prod")
CKPT = os.environ.get("TABPFN_CKPT", "data/tabpfn_v2_checkpoint/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt")
OUT = os.environ.get("OUT", "analysis/tail_e156_opdi_20261008/e210_lomo.json")
N_SEEDS = 20
DELTAS = [-6.0, -3.0, 0.0, 3.0, 6.0]
N_DRAWS = 3
X_MONTH = 0.10
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt",
        "d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y]
f = pl.read_parquet(FEATURES, columns=cols).filter(pl.col("split") == "train")
ap = pl.read_parquet("data/features_e113.parquet", columns=["MVT_ID_mvt", "airline_prefix"])
f = f.join(ap, on="MVT_ID_mvt", how="left")
ids, regs = [], []
for p in sorted(glob.glob(f"{CV_DIR}/fold_*.npz")):
    d = np.load(p)
    ids.append(d["mvt_id"]); regs.append(d["reg"])
oof = pl.DataFrame({"MVT_ID_mvt": np.concatenate(ids).astype(np.int64), "oof_reg": np.concatenate(regs).astype(np.float64)}).with_columns(
    pl.col("MVT_ID_mvt").cast(f.schema["MVT_ID_mvt"]))
t = f.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
    cls=pl.when((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).then(0)
    .when(pl.col(Y).is_between(86400, 93600)).then(1).otherwise(2),
    b=pl.col("d_mvt_sched").cut(L.BINS, labels=L.LABELS),
    stand_num=pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    rwy_head=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64),
    fn_num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    fn_len=pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float64),
    fn_garbled=pl.col("FLIGHT_mvt").map_elements(
        lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64
    ),
).join(oof, on="MVT_ID_mvt", how="left")
feats = ["d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", "stand_num", "rwy_head", "fn_num", "fn_len", "fn_garbled"]
vc = t["airline_prefix"].value_counts().sort(["count", "airline_prefix"], descending=[True, False], nulls_last=True)
keep = vc.filter(pl.col("count") >= 5)["airline_prefix"].drop_nulls().to_list()
code = {p: i + 1 for i, p in enumerate(keep)}
t = t.with_columns(pfx=pl.col("airline_prefix").map_elements(lambda p: float(code.get(p, 0)), return_dtype=pl.Float64).fill_null(0.0))
feats_p = feats + ["pfx"]
reg_c = float(t.filter(pl.col("cls") == 2).select(pl.col(Y).clip(0, 7200).median()).item())
t = t.with_columns(oof_reg=pl.col("oof_reg").clip(0, 7200).fill_null(reg_c))
day_taxi = 951.0
test_bins = L.OVERRIDE
tail = t.filter(pl.col("b").is_in(test_bins))
print(f"features {FEATURES}; OOF rows {oof.height} from {CV_DIR}; threads {torch.get_num_threads()}; checkpoint {CKPT}", flush=True)
print(f"LIRF no-NM rows {t.height} tail rows {tail.height} prefix codes {len(keep)} classes {np.bincount(t['cls'].to_numpy(), minlength=3).tolist()}", flush=True)


def expected(p, d, r):
    return p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * r


def fit_predict(tr, xq, seed):
    c = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed, model_path=CKPT,
                                                    categorical_features_indices=[len(feats_p) - 1])
    c.fit(tr.select(feats_p).to_numpy(), tr["cls"].to_numpy())
    r = c.predict_proba(xq)
    out = np.zeros((xq.shape[0], 3))
    for j, k in enumerate(c.classes_):
        out[:, int(k)] = r[:, j]
    return out


# query blocks: block (k, i) = draw k (0 = no re-draw), shift DELTAS[i]; all blocks in one predict_proba call
# (TabPFN scores query rows independently given the training context; the batch equivalence is checked below)
def blocks(te, draws):
    xq, labels = [], []
    x0 = te.select(feats_p).to_numpy()
    for k, w in enumerate([np.zeros(te.height)] + draws):
        for i, dl in enumerate(DELTAS):
            x = x0.copy(); x[:, 0] = x[:, 0] + w + dl
            xq.append(x); labels.append((k, i))
    return np.vstack(xq), labels


ARMS = ["A0", "A1", "B1", "C1", "C2"]
I0 = DELTAS.index(0.0)
res = dict(protocol="E210-lirf-tail-tabpfn-variants-lomo-20261008-v1", features=FEATURES, cv_dir=CV_DIR, checkpoint=CKPT,
           n_seeds=N_SEEDS, deltas=DELTAS, n_draws=N_DRAWS, x_month=X_MONTH, threads=torch.get_num_threads(), months=[],
           batch_equivalence=None, tail_rows=tail.height, started_unix=time.time())
se = {a: [] for a in ARMS}                     # squared errors per month (clean query)
se_jit = {a: [[] for _ in range(N_DRAWS)] for a in ARMS}   # squared errors with re-drawn dither
dpred = {a: [] for a in ARMS}                  # prediction change under re-draw (all draws pooled)
rows_m = []
for m in range(1, 13):
    tr0, te0 = t.filter(pl.col("month") != m), tail.filter(pl.col("month") == m)
    if te0.height == 0:
        continue
    t0 = time.time()
    rng = np.random.default_rng(1000 + m)
    draws = [rng.integers(-6, 7, te0.height).astype(float) for _ in range(N_DRAWS)]
    xq, labels = blocks(te0, draws)
    n = te0.height
    P = np.zeros((N_SEEDS, len(labels), n, 3))
    for s in range(N_SEEDS):
        p = fit_predict(tr0, xq, s)
        for bi in range(len(labels)):
            P[s, bi] = p[bi * n:(bi + 1) * n]
        if m == 1 and s == 0:
            alone = fit_predict(tr0, te0.select(feats_p).to_numpy(), 0)
            res["batch_equivalence"] = dict(max_abs_diff=float(np.abs(alone - P[0, labels.index((0, I0))]).max()), month=m, seed=0)
            print(f"batch equivalence (query alone vs stacked block, month {m} seed 0): max |dp| = {res['batch_equivalence']['max_abs_diff']:.2e}", flush=True)
    bi = {lab: i for i, lab in enumerate(labels)}
    d, y, r = te0["d_mvt_sched"].to_numpy(), te0[Y].to_numpy(), te0["oof_reg"].to_numpy()

    def arm(a, k):
        seeds = slice(0, 1) if a == "A0" else slice(0, 10) if a in ("A1", "C1") else slice(0, N_SEEDS)
        if a in ("C1", "C2"):
            return P[seeds][:, [bi[(k, i)] for i in range(len(DELTAS))]].mean(axis=(0, 1))
        return P[seeds, bi[(k, I0)]].mean(axis=0)

    row = dict(month=m, n=n, seconds=round(time.time() - t0, 1))
    for a in ARMS:
        pred = expected(arm(a, 0), d, r)
        se[a].append((pred - y) ** 2)
        row[a] = float(se[a][-1].sum())
        for k in range(N_DRAWS):
            pj = expected(arm(a, k + 1), d, r)
            se_jit[a][k].append((pj - y) ** 2)
            dpred[a].append(pj - pred)
    rows_m.append(row)
    res["months"].append(row)
    print(f"month {m:2d}: n={n:3d} " + " ".join(f"{a} {row[a]:.3e}" for a in ARMS) + f"  ({row['seconds']} s)", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

S = {a: np.concatenate(se[a]) for a in ARMS}
top3 = np.argsort(-S["A1"])[:3]
mask = np.ones(len(S["A1"]), bool); mask[top3] = False
mi = {row["month"]: i for i, row in enumerate(rows_m)}
res["totals"] = {a: float(S[a].sum()) for a in ARMS}
res["without_top3"] = {a: float(S[a][mask].sum()) for a in ARMS}
res["control_e156_log"] = dict(A0=5.077e9, A1=4.822e9, note="data/e156_lomo.log, features_v9")
print(f"[control] A0 {S['A0'].sum():.4e} (E156 log 5.077e9), A1 {S['A1'].sum():.4e} (E156 log 4.822e9), A1 vs A0 {100 * (S['A1'].sum() / S['A0'].sum() - 1):+.1f} %", flush=True)
res["gate"] = {}
for v in ("B1", "C1", "C2"):
    tot = S[v].sum() / S["A1"].sum() - 1
    m2 = rows_m[mi[2]][v] / rows_m[mi[2]]["A1"] - 1 if 2 in mi else float("nan")
    m6 = rows_m[mi[6]][v] / rows_m[mi[6]]["A1"] - 1 if 6 in mi else float("nan")
    worst = max((row[v] / row["A1"] - 1, row["month"]) for row in rows_m)
    wo = S[v][mask].sum() / S["A1"][mask].sum() - 1
    better = sum(1 for row in rows_m if row[v] < row["A1"])
    ok = tot < 0 and m2 <= 0 and m6 <= 0 and worst[0] <= X_MONTH and wo <= 0
    res["gate"][v] = dict(total_rel=tot, month2_rel=m2, month6_rel=m6, worst_month_rel=worst[0], worst_month=worst[1], without_top3_rel=wo,
                          months_better=better, verdict="ADOPT" if ok else "REJECT")
    print(f"[{v} vs A1] total {tot:+.2%}, month 2 {m2:+.2%}, month 6 {m6:+.2%}, worst month {worst[1]} {worst[0]:+.2%} (X={X_MONTH:.0%}), "
          f"without top-3 {wo:+.2%}, months better {better}/{len(rows_m)} -> {'ADOPT' if ok else 'REJECT'}", flush=True)
res["redraw"] = {}
for a in ARMS:
    dp = np.concatenate(dpred[a])
    sj = [float(np.concatenate(se_jit[a][k]).sum()) for k in range(N_DRAWS)]
    res["redraw"][a] = dict(rms_pred_change_s=float(np.sqrt(np.mean(dp ** 2))), max_abs_pred_change_s=float(np.abs(dp).max()),
                            sse_per_draw=sj, sse_mean=float(np.mean(sj)), sse_mean_rel_clean=float(np.mean(sj) / S[a].sum() - 1))
    print(f"[re-draw {a}] RMS prediction change {res['redraw'][a]['rms_pred_change_s']:.1f} s (max {res['redraw'][a]['max_abs_pred_change_s']:.0f}), "
          f"SSE mean over {N_DRAWS} draws {np.mean(sj):.4e} ({res['redraw'][a]['sse_mean_rel_clean']:+.2%} vs clean)", flush=True)
res["finished_unix"] = time.time()
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", OUT, flush=True)
