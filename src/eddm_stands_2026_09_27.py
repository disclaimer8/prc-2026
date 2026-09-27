import glob
import numpy as np
import polars as pl

C = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "FLIGHT_ID_mvt"]
tr = pl.concat([pl.read_parquet(f, columns=C + ["TAXITIME_SEC_mvt"]) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=C)
f = lambda x: x.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "EDDM")).with_columns(
    ma=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64), m=pl.col("MVT_TIME_UTC_mvt").dt.month())
tr, rk = f(tr), f(rk)
known = set(tr["STAND_mvt"].drop_nulls().unique().to_list())
sub = pl.read_parquet("submissions/gentle-igloo_v366.parquet").rename({"TAXITIME_SEC_mvt": "v366"})
rk = rk.join(sub.with_columns(pl.col("MVT_ID_mvt").cast(rk["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left").with_columns(
    new=~pl.col("STAND_mvt").is_in(list(known)) & pl.col("STAND_mvt").is_not_null())
print("EDDM ranking by month: rows, unseen stands")
print(rk.group_by("m").agg(n=pl.len(), unseen=pl.col("new").sum(), stands=pl.col("STAND_mvt").filter(pl.col("new")).unique().sort().str.join(",")).sort("m"))
print("\nv366 - (MVT - AOBT_3), July 2026, by unseen vs known stand (NM rows):")
j = rk.filter((pl.col("m") == 7) & pl.col("ma").is_not_null())
print(j.group_by("new").agg(n=pl.len(), med_ma=pl.col("ma").median(), med_v366=pl.col("v366").median(), med_gap=(pl.col("v366") - pl.col("ma")).median(),
                            mean_gap=(pl.col("v366") - pl.col("ma")).mean().round(1)))
print(j.filter(pl.col("new")).group_by("STAND_mvt").agg(n=pl.len(), med_gap=(pl.col("v366") - pl.col("ma")).median()).sort("STAND_mvt"))
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(q))["mvt_id"], "p": z["pred"]}) for q in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
t = tr.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(tr["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").filter(pl.col("ma").is_not_null() & (pl.col("m") == 7))
t = t.with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), grp=pl.when(pl.col("STAND_mvt").str.contains(r"^1[0-2][0-9]$")).then(pl.lit("1xx")).otherwise(pl.lit("other")))
print("\nJuly 2025 EDDM (NM rows): truth and OOF vs MVT - AOBT_3, stands 1xx vs other")
print(t.group_by("grp").agg(n=pl.len(), med_y_gap=(pl.col("y") - pl.col("ma")).median(), mean_y_gap=(pl.col("y") - pl.col("ma")).mean().round(1),
                            med_oof_gap=(pl.col("p") - pl.col("ma")).median()))
print("\nfeature columns in features_e106 for unseen-stand rank rows (share null) vs known:")
fe = pl.read_parquet("data/features_e106.parquet")
cols = [c for c in fe.columns if "stand" in c.lower() or c.startswith(("ap_st", "ter_", "te_"))]
x = fe.filter(pl.col("split") == "rank").join(rk.select("MVT_ID_mvt", "new", "m").with_columns(pl.col("MVT_ID_mvt").cast(fe["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").filter(pl.col("m") == 7)
print(x.group_by("new").agg([pl.col(c).is_null().mean().round(2).alias(c) for c in cols[:25]]).transpose(include_header=True).to_pandas().to_string())
