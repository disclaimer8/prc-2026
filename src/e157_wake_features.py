"""E157 (journal 152): wake-ladder features for the base model = external Hypothesis 6 + our G7 (heavy/super count ahead).

    .venv/bin/python src/e157_wake_features.py   ->  data/features_e157.parquet (= data/features_e106.parquet + 7 columns)

Per departure, on its own runway (ADEP, RUNWAY), from the movement records of the same split (2025 training / 2026 ranking,
the take-off times are known in both):
  e157_gap        seconds since the previous take-off (clip 0..1800)
  e157_req        median gap of that (leader, follower) wake pair at the airport, 2025 pairs with 30 < gap <= 600 s
                  (M behind M 110 s, M behind H 143 s, M behind J 181 s: GREKI's ladder reproduces)
  e157_hold       max(0, req - gap)          (the reviewer's "excess hold")
  e157_margin     gap - req, clip -600..1800 (runway saturation)
  e157_prev_heavy leader is H or J
  e157_ladder_q   sum of the required gaps of the take-offs on the runway between the row's off-block (AOBT_3, LOBT where
                  missing) and its own take-off
  e157_heavy_10   H/J take-offs on the runway in the 10 min before its own take-off (G7)
Wake category = WK_TBL_CAT_flt, else the most common category of the aircraft type, else M.
Probe (src/h6h7_wake_probe.py, production OOF): partial Spearman g7 +0.048/+0.043, prev_heavy +0.029/+0.024; stage over the
existing windows -0.34 / -0.35 s RMSE of the subset (E151's probe was -0.29 / -0.22 and E151 was neutral in production CV).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt"]
files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))
raw = pl.concat([pl.read_parquet(p, columns=cols).with_columns(part=pl.lit("train")) for p in files]
                + [pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=cols).with_columns(part=pl.lit("rank"))])
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
tmode = raw.drop_nulls("WK_TBL_CAT_flt").group_by("AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt").agg(n=pl.len()) \
    .sort(["n", "WK_TBL_CAT_flt"], descending=[True, False]).unique("AIRCRAFT_TYPE_mvt", keep="first", maintain_order=True) \
    .select("AIRCRAFT_TYPE_mvt", tcat="WK_TBL_CAT_flt")
dep = (raw.filter(pl.col("PHASE_mvt") == "DEP").join(tmode, on="AIRCRAFT_TYPE_mvt", how="left")
       .select("MVT_ID_mvt", "part", ap=pl.col("ADEP_mvt"), rwy=pl.col("RUNWAY_mvt"),
               cat=pl.coalesce("WK_TBL_CAT_flt", "tcat").fill_null("M"), t=ep("MVT_TIME_UTC_mvt"),
               t_ob=pl.coalesce(ep("AOBT_3_flt"), ep("LOBT_flt")))
       .sort(["part", "ap", "rwy", "t", "MVT_ID_mvt"])
       .with_columns(prev_cat=pl.col("cat").shift(1).over("part", "ap", "rwy"),
                     gap=pl.col("t") - pl.col("t").shift(1).over("part", "ap", "rwy")))
req = (dep.filter((pl.col("part") == "train") & pl.col("gap").is_between(30, 600)).group_by("ap", "prev_cat", "cat")
       .agg(req=pl.col("gap").median(), n=pl.len()).filter(pl.col("n") >= 50).drop("n"))
dep = dep.join(req, on=["ap", "prev_cat", "cat"], how="left").with_columns(
    e157_gap=pl.col("gap").clip(0, 1800), e157_req=pl.col("req"), e157_hold=(pl.col("req") - pl.col("gap")).clip(0, None),
    e157_margin=(pl.col("gap") - pl.col("req")).clip(-600, 1800), e157_prev_heavy=pl.col("prev_cat").is_in(["H", "J"]).cast(pl.Float64))
out = []
for _, g in dep.group_by("part", "ap", "rwy", maintain_order=True):
    t, q = g["t"].to_numpy(), g["t_ob"].to_numpy()
    rq = np.nan_to_num(g["req"].to_numpy().astype(float), nan=120.0)
    hv = g["cat"].is_in(["H", "J"]).to_numpy().astype(float)
    cr, ch = np.concatenate([[0.0], np.cumsum(rq)]), np.concatenate([[0.0], np.cumsum(hv)])
    hi = np.searchsorted(t, t, "left")
    lo = np.minimum(np.searchsorted(t, np.where(np.isnan(q), t, q), "right"), hi)
    l10 = np.minimum(np.searchsorted(t, t - 600, "right"), hi)
    out.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "e157_ladder_q": np.where(np.isnan(q), np.nan, cr[hi] - cr[lo]),
                             "e157_heavy_10": ch[hi] - ch[l10]}))
NEW = ["e157_gap", "e157_req", "e157_hold", "e157_margin", "e157_prev_heavy", "e157_ladder_q", "e157_heavy_10"]
feat = dep.select("MVT_ID_mvt", *NEW[:5]).join(pl.concat(out), on="MVT_ID_mvt")
assert feat["MVT_ID_mvt"].n_unique() == feat.height
base = pl.read_parquet("data/features_e106.parquet")
res = base.join(feat, on="MVT_ID_mvt", how="left")
assert res.height == base.height
res.write_parquet("data/features_e157.parquet")
for part in ("train", "rank"):
    s = res.filter(pl.col("split") == part)
    print(f"{part}: rows {s.height}, e157 null share " + ", ".join(f"{c} {s[c].null_count() / s.height:.4f}" for c in NEW))
print(f"wrote data/features_e157.parquet: {res.width} columns (+{len(NEW)})")
