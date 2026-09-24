"""E118 (25.09, external review 3.2): E11 with the airline prefix (E113 lookup) as a categorical feature, against E11 itself.
Based on E11: TabPFN-2 (Prior Labs License, Apache-2.0 based) regime probabilities for LIRF departures without NM,
versus the smoothed table (src/lirf_table.py), leave-one-month-out on 2025.

Pre-registered decision: adopt if LOMO-SSE is >= 5% below the table, the gain holds without the top-3 rows,
and TabPFN is better in >= 8 of the 12 months (E8 gate).
"""
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
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt",
        "d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y]
f = pl.read_parquet("data/features_v9.parquet", columns=cols).filter(pl.col("split") == "train")
ap = pl.read_parquet("data/features_e113.parquet", columns=["MVT_ID_mvt", "airline_prefix"])
f = f.join(ap, on="MVT_ID_mvt", how="left")
t = f.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
    cls=pl.when((pl.col(Y) - pl.col("d_mvt_sched")).abs() <= 6).then(0)
    .when(pl.col(Y).is_between(86400, 93600)).then(1).otherwise(2),
    b=pl.col("d_mvt_sched").cut(L.BINS, labels=L.LABELS),
    stand_num=pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    rwy_head=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64),
    fn_num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64),
    fn_len=pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float64),
    # garbled call signs like ITYTY680, RYRR90BN, EXSXS1GT: ICAO prefix followed by a repeat of its tail
    fn_garbled=pl.col("FLIGHT_mvt").map_elements(
        lambda s: float(bool(re.match(r"^([A-Z])([A-Z])([A-Z])(\3|\2\3)", s or ""))), return_dtype=pl.Float64
    ),
)
feats = ["d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", "stand_num", "rwy_head", "fn_num", "fn_len", "fn_garbled"]
vc = t["airline_prefix"].value_counts().sort("count", descending=True)
code = {p: i + 1 for i, p in enumerate(vc.filter(pl.col("count") >= 5)["airline_prefix"].drop_nulls().to_list())}
t = t.with_columns(pfx=pl.col("airline_prefix").map_elements(lambda p: float(code.get(p, 0)), return_dtype=pl.Float64).fill_null(0.0))
feats_p = feats + ["pfx"]
print("prefix codes:", len(code), "coverage", round(float((t["pfx"] > 0).mean()), 3))
print("LIRF no-NM rows 2025:", t.height, "classes", np.bincount(t["cls"].to_numpy(), minlength=3).tolist())

reg_c = float(t.filter(pl.col("cls") == 2).select(pl.col(Y).clip(0, 7200).median()).item())
day_taxi = 951.0
test_bins = L.OVERRIDE  # compare where the table is applied


def expected(p, d):
    return p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * reg_c


se_tab, se_pfn, months = [], [], []
for m in range(1, 13):
    tr, te = t.filter(pl.col("month") != m), t.filter((pl.col("month") == m) & pl.col("b").is_in(test_bins))
    if te.height == 0:
        continue
    def pfn(fs, cat):
        c = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0, categorical_features_indices=cat)
        c.fit(tr.select(fs).to_numpy(), tr["cls"].to_numpy())
        r = c.predict_proba(te.select(fs).to_numpy())
        out = np.zeros((te.height, 3))
        for j, k in enumerate(c.classes_):
            out[:, int(k)] = r[:, j]
        return out
    p_tab = pfn(feats, None)
    p_pfn = pfn(feats_p, [len(feats_p) - 1])
    d, y = te["d_mvt_sched"].to_numpy(), te[Y].to_numpy()
    se_tab.append((expected(p_tab, d) - y) ** 2)
    se_pfn.append((expected(p_pfn, d) - y) ** 2)
    months.append((m, se_tab[-1].sum(), se_pfn[-1].sum(), te.height))
    print(f"month {m:2d}: n={te.height:3d} E11 SSE={se_tab[-1].sum():.3e} E118 SSE={se_pfn[-1].sum():.3e}", flush=True)

st, sp = np.concatenate(se_tab), np.concatenate(se_pfn)
top3 = np.argsort(-st)[:3]
mask = np.ones(len(st), bool)
mask[top3] = False
wins = sum(1 for _, a, b, _ in months if b < a)
gain = 100 * (sp.sum() / st.sum() - 1)
gain_wo = 100 * (sp[mask].sum() / st[mask].sum() - 1)
print(f"LOMO-SSE E11={st.sum():.3e} E118={sp.sum():.3e} ({gain:+.1f}%), without top-3 ({gain_wo:+.1f}%), months better {wins}/{len(months)}")
print("DECISION:", "ADOPT" if gain <= -5 and gain_wo < 0 and wins >= 8 else "REJECT")
