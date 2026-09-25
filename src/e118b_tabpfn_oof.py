"""E118b (25.09, hypothesis 50 / external review): E118 re-run with the regular candidate the chain actually uses.

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
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import lirf_table as L

Y = "TAXITIME_SEC_mvt"
FEATURES = os.environ.get("FEATURES", "data/features_v9.parquet")
CV_DIR = os.environ.get("CV_DIR", "data/cv_preds_e106_air")
cols = ["MVT_ID_mvt", "split", "month", "ADEP_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt",
        "d_mvt_sched", "local_hour", "local_dow", "day_nonm_rate", "dep_before_30", "arr_before_30", Y]
f = pl.read_parquet(FEATURES, columns=cols).filter(pl.col("split") == "train")
ap = pl.read_parquet("data/features_e113.parquet", columns=["MVT_ID_mvt", "airline_prefix"])
f = f.join(ap, on="MVT_ID_mvt", how="left")
ids, regs = [], []
for p in sorted(glob.glob(f"{CV_DIR}/fold_*.npz")):
    d = np.load(p)
    ids.append(d["mvt_id"]); regs.append(d["reg"])
oof = pl.DataFrame({"MVT_ID_mvt": np.concatenate(ids).astype(np.int64), "oof_reg": np.concatenate(regs).astype(np.float64)})
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
vc = t["airline_prefix"].value_counts().sort("count", descending=True)
code = {p: i + 1 for i, p in enumerate(vc.filter(pl.col("count") >= 5)["airline_prefix"].drop_nulls().to_list())}
t = t.with_columns(pfx=pl.col("airline_prefix").map_elements(lambda p: float(code.get(p, 0)), return_dtype=pl.Float64).fill_null(0.0))
feats_p = feats + ["pfx"]
reg_c = float(t.filter(pl.col("cls") == 2).select(pl.col(Y).clip(0, 7200).median()).item())
t = t.with_columns(oof_reg=pl.col("oof_reg").clip(0, 7200).fill_null(reg_c))
day_taxi = 951.0
test_bins = L.OVERRIDE
tt = t.filter(pl.col("b").is_in(test_bins))
print("prefix codes:", len(code), "coverage", round(float((t["pfx"] > 0).mean()), 3))
print("LIRF no-NM rows 2025:", t.height, "classes", np.bincount(t["cls"].to_numpy(), minlength=3).tolist(),
      f"| tail rows {tt.height}, OOF missing {int(tt['oof_reg'].is_null().sum())}, reg_c {reg_c:.0f}, "
      f"RMS(oof_reg - reg_c) on tail {float(np.sqrt(np.mean((tt['oof_reg'].to_numpy() - reg_c) ** 2))):.0f} s")


def expected(p, d, r):
    return p[:, 0] * d + p[:, 1] * (86400 + day_taxi) + p[:, 2] * r


res = {k: [] for k in ("tab_c", "pfn_c", "tab_o", "pfn_o")}
months = []
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

    p_tab, p_pfn = pfn(feats, None), pfn(feats_p, [len(feats_p) - 1])
    d, y, r = te["d_mvt_sched"].to_numpy(), te[Y].to_numpy(), te["oof_reg"].to_numpy()
    res["tab_c"].append((expected(p_tab, d, reg_c) - y) ** 2); res["pfn_c"].append((expected(p_pfn, d, reg_c) - y) ** 2)
    res["tab_o"].append((expected(p_tab, d, r) - y) ** 2); res["pfn_o"].append((expected(p_pfn, d, r) - y) ** 2)
    months.append((m, te.height, *[res[k][-1].sum() for k in ("tab_c", "pfn_c", "tab_o", "pfn_o")]))
    print(f"month {m:2d}: n={te.height:3d} reg_c: E11 {months[-1][2]:.3e} E118 {months[-1][3]:.3e} | OOF: E11 {months[-1][4]:.3e} E118b {months[-1][5]:.3e}", flush=True)

for tag, a, b in (("reg_c (as E118)", "tab_c", "pfn_c"), ("OOF regressor (as the chain)", "tab_o", "pfn_o")):
    st, sp = np.concatenate(res[a]), np.concatenate(res[b])
    top3 = np.argsort(-st)[:3]
    mask = np.ones(len(st), bool); mask[top3] = False
    i_t, i_p = (2, 3) if a == "tab_c" else (4, 5)
    wins = sum(1 for row in months if row[i_p] < row[i_t])
    gain = 100 * (sp.sum() / st.sum() - 1); gain_wo = 100 * (sp[mask].sum() / st[mask].sum() - 1)
    print(f"[{tag}] LOMO-SSE E11={st.sum():.3e} E118={sp.sum():.3e} ({gain:+.1f}%), without top-3 ({gain_wo:+.1f}%), months better {wins}/{len(months)}")
    print(f"[{tag}] DECISION:", "ADOPT" if gain <= -5 and gain_wo < 0 and wins >= 8 else "REJECT")
# the E11 arm itself: does the candidate choice matter for the E11 table replacement at all?
st_c, st_o = np.concatenate(res["tab_c"]), np.concatenate(res["tab_o"])
print(f"E11 arm SSE with reg_c {st_c.sum():.3e} vs OOF {st_o.sum():.3e} ({100 * (st_o.sum() / st_c.sum() - 1):+.1f}%)")
