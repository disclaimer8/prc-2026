import glob
import numpy as np
import polars as pl

APS = ["EGLL", "EHAM", "EDDF", "EDDM", "LFPG", "LEMD", "LEBL", "LIRF", "LTFM", "LSZH"]
def pru_year(y):
    return pl.read_csv(f"data/taxi_out_additional_time_{y}.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APS)).select(
        ADEP_mvt=pl.col("APT_ICAO"), m=pl.col("MONTH_NUM").cast(pl.Int8),
        pru=(pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) * 60 / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64),
        valid=pl.col("VALID_FL").cast(pl.Float64) / pl.col("TF").cast(pl.Float64))
pru = pru_year(2025)
raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "p": z["pred"], "y": z["y"]}) for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))
g = d.group_by("ADEP_mvt", "m").agg(ymean_f=pl.col("y").filter(pl.col("y") <= 7200).mean(), pmean_f=pl.col("p").filter(pl.col("p") <= 7200).mean()).join(pru, on=["ADEP_mvt", "m"])
g = g.with_columns(off=pl.col("ymean_f") - pl.col("pru"), bias=pl.col("pmean_f") - pl.col("ymean_f"))
folds = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
parts = []
for a, b in folds:
    tm = [x for x in range(1, 13) if x not in (a, b)]
    oh = g.filter(pl.col("m").is_in(tm)).group_by("ADEP_mvt").agg(off_hat=pl.col("off").mean(), off_sd=pl.col("off").std())
    for mm in (a, b):
        parts.append(g.filter(pl.col("m") == mm).join(oh, on="ADEP_mvt").select("ADEP_mvt", "m", "off_sd", shift=pl.col("pru") + pl.col("off_hat") - pl.col("pmean_f")))
s = pl.concat(parts)
x = d.join(s, on=["ADEP_mvt", "m"], how="left").with_columns(pl.col("shift").fill_null(0.0)).with_columns(
    p2=pl.when(pl.col("p") <= 7200).then(pl.col("p") + pl.col("shift")).otherwise(pl.col("p")))
x = x.with_columns(de=(pl.col("p2") - pl.col("y")) ** 2 - (pl.col("p") - pl.col("y")) ** 2,
                   fold=pl.col("m").replace_strict({mm: i for i, (a, b) in enumerate(folds) for mm in (a, b)}))
N, r0 = d.height, float(np.sqrt(((d["p"] - d["y"]) ** 2).mean()))
res = x.group_by("ADEP_mvt").agg(dRMSE=(pl.col("de").sum() / N / (2 * r0)).round(3),
                                 folds_lower=pl.col("de").sum().over("fold").lt(0).alias("x").first(), n=pl.len())
per_fold = x.group_by("ADEP_mvt", "fold").agg(pl.col("de").sum()).group_by("ADEP_mvt").agg(folds_lower=(pl.col("de") < 0).sum())
res = res.drop("folds_lower").join(per_fold, on="ADEP_mvt").join(g.group_by("ADEP_mvt").agg(off_sd=pl.col("off").std().round(1), off_mean=pl.col("off").mean().round(1),
    bias_sd=pl.col("bias").std().round(1), valid_min=pl.col("valid").min().round(3)), on="ADEP_mvt").sort("dRMSE")
print(f"OOF RMSE {r0:.3f}; per-airport effect of airport-month calibration to PRU (offset learned on the other months of the airport):")
print(res.to_pandas().to_string())
print("\nEGLL monthly detail 2025:")
print(g.filter(pl.col("ADEP_mvt") == "EGLL").sort("m").select("m", pl.col("pru").round(1), pl.col("ymean_f").round(1), pl.col("off").round(1), pl.col("bias").round(1), pl.col("valid").round(3)).to_pandas().to_string())
p26 = pru_year(2026)
print("\nPRU 2026 EGLL:", p26.filter(pl.col("ADEP_mvt") == "EGLL").sort("m").select("m", pl.col("pru").round(1), pl.col("valid").round(3)).to_dicts())
