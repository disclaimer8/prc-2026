"""E16: regime probabilities for LIRF departures without NM at D = 4-30 h, two pre-registered arms against TabPFN E11.

    FEATURES=data/features_final.parquet TABPFN_ALLOW_CPU_LARGE_DATASET=1 .venv/bin/python src/e16_tail_regimes.py
Control: TabPFN-2 with the E11 features. E16a: the same TabPFN plus flight identity in the month (share of days matched to NM,
never matched flag, log(1 + departures of the number)) and the day share of LIRF arrivals with block = schedule (+-6 s) and
taxi-in > 30 min. E16b: explicit Dirichlet table with the prior of the current table (1; 0.2; 1.8) over cells
4-8 h x {orphan or garbled call sign, other}, 8-12 h, 12-30 h. Leave-one-month-out on 2025, same candidates and expectation as E11.
Gate (REVIEW_BRIEF.md, E16): LOMO-SSE >= 5% lower, lower without the top-3 rows, better in >= 8 of 12 months; if both arms
pass, the larger gain. Per-row predictions are saved to data/e16_lomo.parquet.
"""
import glob
import os
import re
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, "src")

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import lirf_table as L

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "t_mvt",
        "d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y]
f = pl.read_parquet(os.environ.get("FEATURES", "data/features_final.parquet"), columns=cols).filter(
    (pl.col("split") == "train") & (pl.col("ADEP_mvt") == "LIRF"))
counts = f.group_by("FLIGHT_mvt", "month").agg(n_rk=pl.len(), n_nm=pl.col("FLIGHT_ID_mvt").is_not_null().sum())

raw = pl.concat([pl.read_parquet(p, columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", Y])
                 for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
arr = raw.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")).group_by(date=pl.col("MVT_TIME_UTC_mvt").dt.date()).agg(
    arr_sub=(((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6) & (pl.col(Y) > 1800)).mean())

t = f.filter(pl.col("FLIGHT_ID_mvt").is_null()).join(counts, on=["FLIGHT_mvt", "month"], how="left").with_columns(
    pl.col("n_rk").fill_null(1), pl.col("n_nm").fill_null(0),
    date=pl.from_epoch(pl.col("t_mvt").cast(pl.Int64), time_unit="s").dt.date(),
).join(arr, on="date", how="left").with_columns(
    cls=pl.when((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).then(0).when(pl.col(Y).is_between(86400, 93600)).then(1).otherwise(2),
    b=pl.col("d_mvt_sched").cut(L.BINS, labels=L.LABELS),
    stand_num=pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    rwy_head=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64),
    fn_num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    fn_len=pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float64),
    fn_garbled=pl.col("FLIGHT_mvt").map_elements(
        lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64).fill_null(0.0),
    share_nm=pl.col("n_nm") / pl.col("n_rk"),
    orphan=(pl.col("n_nm") == 0).cast(pl.Float64),
    log_n=(pl.col("n_rk") + 1).log(),
    arr_sub=pl.col("arr_sub").fill_null(0.0),
)
FE11 = ["d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", "stand_num", "rwy_head", "fn_num", "fn_len", "fn_garbled"]
FE16 = FE11 + ["share_nm", "orphan", "log_n", "arr_sub"]
print("LIRF no-NM rows 2025:", t.height, "classes", np.bincount(t["cls"].to_numpy(), minlength=3).tolist())
reg_c = float(t.filter(pl.col("cls") == 2).select(pl.col(Y).clip(0, 7200).median()).item())
day_taxi = 951.0


def expected(p, d):
    return p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * reg_c


def tabpfn(tr, te, feats):
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
    clf.fit(tr.select(feats).to_numpy(), tr["cls"].to_numpy())
    raw_p = clf.predict_proba(te.select(feats).to_numpy())
    p = np.zeros((te.height, 3))
    for j, c in enumerate(clf.classes_):
        p[:, int(c)] = raw_p[:, j]
    return p


def cell(d: pl.DataFrame) -> pl.Expr:
    grp = pl.when((pl.col("orphan") == 1) | (pl.col("fn_garbled") == 1)).then(pl.lit("og")).otherwise(pl.lit("other"))
    return pl.when(pl.col("b") == "4-8h").then(pl.lit("4-8h:") + grp).otherwise(pl.col("b").cast(pl.String))


def table(tr, te):
    tab = tr.with_columns(c=cell(tr)).group_by("c").agg(n=pl.len(), ns=(pl.col("cls") == 0).sum(), nd=(pl.col("cls") == 1).sum()).with_columns(
        ps=(pl.col("ns") + L.ALPHA) / (pl.col("n") + 3 * L.ALPHA), pd=(pl.col("nd") + L.ALPHA_DAY) / (pl.col("n") + 3 * L.ALPHA))
    j = te.with_columns(c=cell(te)).join(tab.select("c", "ps", "pd"), on="c", how="left").with_columns(
        pl.col("ps").fill_null(L.ALPHA / (3 * L.ALPHA)), pl.col("pd").fill_null(L.ALPHA_DAY / (3 * L.ALPHA)))
    return np.column_stack([j["ps"].to_numpy(), j["pd"].to_numpy(), 1 - j["ps"].to_numpy() - j["pd"].to_numpy()])


rows, months = [], []
for m in range(1, 13):
    tr, te = t.filter(pl.col("month") != m), t.filter((pl.col("month") == m) & pl.col("b").is_in(L.OVERRIDE))
    if te.height == 0:
        continue
    d, y = te["d_mvt_sched"].to_numpy(), te[Y].to_numpy()
    preds = {"control": expected(tabpfn(tr, te, FE11), d), "e16a": expected(tabpfn(tr, te, FE16), d), "e16b": expected(table(tr, te), d)}
    se = {k: (v - y) ** 2 for k, v in preds.items()}
    months.append((m, {k: v.sum() for k, v in se.items()}))
    rows.append(te.select("MVT_ID_mvt", "month", "b", "orphan", "fn_garbled", "d_mvt_sched", Y).with_columns(
        **{f"pred_{k}": pl.Series(v) for k, v in preds.items()}))
    print(f"month {m:2d}: n={te.height:3d} " + " ".join(f"{k} SSE={v.sum():.3e}" for k, v in se.items()), flush=True)

res = pl.concat(rows)
res.write_parquet("data/e16_lomo.parquet")
se_c = ((res["pred_control"] - res[Y]) ** 2).to_numpy()
top3 = np.argsort(-se_c)[:3]
mask = np.ones(len(se_c), bool)
mask[top3] = False
decisions = {}
for arm in ("e16a", "e16b"):
    se_a = ((res[f"pred_{arm}"] - res[Y]) ** 2).to_numpy()
    gain = 100 * (se_a.sum() / se_c.sum() - 1)
    gain_wo = 100 * (se_a[mask].sum() / se_c[mask].sum() - 1)
    wins = sum(1 for _, s in months if s[arm] < s["control"])
    ok = gain <= -5 and gain_wo < 0 and wins >= 8
    decisions[arm] = (ok, gain)
    print(f"{arm}: LOMO-SSE control={se_c.sum():.3e} arm={se_a.sum():.3e} ({gain:+.1f}%), without top-3 ({gain_wo:+.1f}%), "
          f"months better {wins}/{len(months)}, LB-equivalent RMSE change {(se_a.sum() - se_c.sum()) / 6 / (2 * 274.28 * 344841):+.2f} s -> {'PASS' if ok else 'FAIL'}")
passed = [a for a, (ok, _) in decisions.items() if ok]
print("DECISION E16:", "ACCEPT " + min(passed, key=lambda a: decisions[a][1]) if passed else "REJECT")
