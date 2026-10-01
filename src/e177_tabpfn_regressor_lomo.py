"""E177 (journal 226, external review 01.10): TabPFN REGRESSOR on the LIRF no-NM tail instead of the three-regime classifier.
The method survey (analysis/method_survey_2026-09-16.md:129) credited E11 to TabPFN regression; the chain uses
TabPFNClassifier (src/lirf_tabpfn.py:85), SCHED / +day / regular, and a regressor was never tried. Same rows, features, codes and
LOMO as E156: A0 = classifier, canonical codes, seed 0 (production form); R1 = TabPFNRegressor on y (no clipping, mean of the
predictive distribution); R2 = TabPFNRegressor on y - D (D = MVT - SCHED), + D; M1/M2 = 0.5 A0 + 0.5 R1/R2 (weights fixed in
advance). Gate per arm against A0 as E11/E118: LOMO-SSE >= 5 % below A0, the gain holds without A0's top-3 rows, better in
>= 8 of 12 months. Five arms are compared, so a single pass at the edge is read as noise.

--- E156 header kept for reference ---
E156 (journal 148): TabPFN ensemble on the LIRF no-NM tail (bins of the tail table, D 4-30 h), LOMO over 2025 months.
Arms on the prefix design (E118, deterministic codes since journal 133): A0 canonical codes, seed 0 (the production draw);
A1 canonical codes, seeds 0-9 averaged; A2 10 random permutations of the prefix codes, seed 0, averaged. Regular candidate =
the production OOF regressor (CV_DIR default data/cv_preds_e116prod). Gate as E11/E118: LOMO-SSE >= 5 % below A0, the gain
holds without the top-3 rows, better in >= 8 of 12 months.

Based on E118b (25.09, hypothesis 50 / external review): E118 re-run with the regular candidate the chain actually uses.

E11 and E118 (src/e118_tabpfn_prefix.py) scored the regime probabilities with a single regular candidate, the
median taxi of the regular class (reg_c ~ 1031 s). The chain (src/lirf_tabpfn.py) uses the per-row regressor
prediction. Here the LOMO comparison is repeated with the per-row OOF regressor prediction from the nested CV of
the E106 configuration (CV_DIR, default data/cv_preds_e106_air), everything else identical; the reg_c figures are
printed alongside for reference. Gate unchanged (E8/E11): LOMO-SSE >= 5 % below the E11 arm, gain holds without
the top-3 rows, better in >= 8 of 12 months.
"""
import glob
import os
import re
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

import lirf_table as L

Y = "TAXITIME_SEC_mvt"
FEATURES = os.environ.get("FEATURES", "data/features_v9.parquet")
CV_DIR = os.environ.get("CV_DIR", "data/cv_preds_e116prod")
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
print(f"OOF rows {oof.height} from {CV_DIR}")
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
feats_p = feats + ["pfx"]
reg_c = float(t.filter(pl.col("cls") == 2).select(pl.col(Y).clip(0, 7200).median()).item())
t = t.with_columns(oof_reg=pl.col("oof_reg").clip(0, 7200).fill_null(reg_c))
day_taxi = 951.0
test_bins = L.OVERRIDE
rng = np.random.default_rng(0)
perms = [list(rng.permutation(len(keep))) for _ in range(10)]
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))


def with_codes(d, order):
    code = {p: order[i] + 1 for i, p in enumerate(keep)}
    return d.with_columns(pfx=pl.col("airline_prefix").map_elements(lambda p: float(code.get(p, 0)), return_dtype=pl.Float64).fill_null(0.0))


def expected(p, d, r):
    return p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * r


def pfn(tr, te, seed):
    c = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed, categorical_features_indices=[len(feats_p) - 1])
    c.fit(tr.select(feats_p).to_numpy(), tr["cls"].to_numpy())
    r = c.predict_proba(te.select(feats_p).to_numpy())
    out = np.zeros((te.height, 3))
    for j, k in enumerate(c.classes_):
        out[:, int(k)] = r[:, j]
    return out


canon = list(range(len(keep)))


def pfn_reg(tr, te, target, seed=0):
    r = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed, categorical_features_indices=[len(feats_p) - 1])
    r.fit(tr.select(feats_p).to_numpy(), target)
    return np.asarray(r.predict(te.select(feats_p).to_numpy()), dtype=float)   # default output: mean of the predictive distribution


ARMS = ("A0", "R1", "R2", "M1", "M2")
res = {k: [] for k in ARMS}
months = []
print("LIRF no-NM rows", t.height, "tail rows", t.filter(pl.col("b").is_in(test_bins)).height, "prefix codes", len(keep), flush=True)
for m in range(1, 13):
    tr0, te0 = t.filter(pl.col("month") != m), t.filter((pl.col("month") == m) & pl.col("b").is_in(test_bins))
    if te0.height == 0:
        continue
    trc, tec = with_codes(tr0, canon), with_codes(te0, canon)
    d, y, r = te0["d_mvt_sched"].to_numpy(), te0[Y].to_numpy(), te0["oof_reg"].to_numpy()
    ytr, dtr = trc[Y].to_numpy().astype(float), trc["d_mvt_sched"].to_numpy().astype(float)
    a0 = expected(pfn(trc, tec, 0), d, r)
    r1 = pfn_reg(trc, tec, ytr)
    r2 = pfn_reg(trc, tec, ytr - dtr) + d
    for k, p in (("A0", a0), ("R1", r1), ("R2", r2), ("M1", 0.5 * a0 + 0.5 * r1), ("M2", 0.5 * a0 + 0.5 * r2)):
        res[k].append((p - y) ** 2)
    months.append((m, te0.height, *[res[k][-1].sum() for k in ARMS]))
    print(f"month {m:2d}: n={te0.height:3d} " + " ".join(f"{k} {res[k][-1].sum():.3e}" for k in ARMS), flush=True)
s0 = np.concatenate(res["A0"])
top3 = np.argsort(-s0)[:3]; mask = np.ones(len(s0), bool); mask[top3] = False
for i, k in enumerate(ARMS[1:], start=3):
    sk = np.concatenate(res[k])
    wins = sum(1 for row in months if row[i] < row[2])
    gain = 100 * (sk.sum() / s0.sum() - 1); gain_wo = 100 * (sk[mask].sum() / s0[mask].sum() - 1)
    lb = 2 * 344841 * 235.35
    print(f"[{k} vs A0] LOMO-SSE {s0.sum():.3e} -> {sk.sum():.3e} ({gain:+.1f}%, board-scale {(sk.sum() - s0.sum()) / lb:+.3f} s over 12 months), without top-3 {gain_wo:+.1f}%, months better {wins}/{len(months)} -> "
          + ("ADOPT" if gain <= -5 and gain_wo < 0 and wins >= 8 else "REJECT"))
