import glob
import numpy as np
import polars as pl

APS = ["EGLL", "EHAM", "EDDF", "EDDM", "LFPG", "LEMD", "LEBL", "LIRF", "LTFM", "LSZH"]
pru = pl.read_csv("data/taxi_out_additional_time_2025.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APS)).select(
    ADEP_mvt=pl.col("APT_ICAO"), m=pl.col("MONTH_NUM").cast(pl.Int8), tf=pl.col("TF").cast(pl.Int64),
    pru=(pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) * 60 / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64))
raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "p": z["pred"], "y": z["y"]})
                 for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(
    m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))
g = d.group_by("ADEP_mvt", "m").agg(n=pl.len(), ymean_f=pl.col("y").filter(pl.col("y") <= 7200).mean(),
                                    pmean_f=pl.col("p").filter(pl.col("p") <= 7200).mean()).join(pru, on=["ADEP_mvt", "m"])
g = g.with_columns(off=pl.col("ymean_f") - pl.col("pru"))
print("row count vs PRU TF (ratio), Jan/Jul:")
print(g.filter(pl.col("m").is_in([1, 7])).select("ADEP_mvt", "m", (pl.col("n") / pl.col("tf")).round(3).alias("n/tf"), pl.col("off").round(1)).sort("ADEP_mvt", "m"))
folds = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
N, r0 = d.height, float(np.sqrt(((d["p"] - d["y"]) ** 2).mean()))
for mode in ("other_months", "same_season"):
    shifts = []
    for a, b in folds:
        for mm in (a, b):
            train_m = [x for x in range(1, 13) if x not in (a, b)]
            if mode == "same_season":
                win = [1, 2, 3, 11, 12] if mm in (1, 2, 3, 11, 12) else [4, 5, 6, 7, 8, 9, 10]
                train_m = [x for x in train_m if x in win]
            oh = g.filter(pl.col("m").is_in(train_m)).group_by("ADEP_mvt").agg(off_hat=pl.col("off").mean())
            t = g.filter(pl.col("m") == mm).join(oh, on="ADEP_mvt").with_columns(
                shift=pl.col("pru") + pl.col("off_hat") - pl.col("pmean_f")).select("ADEP_mvt", "m", "shift")
            shifts.append(t)
    s = pl.concat(shifts)
    x = d.join(s, on=["ADEP_mvt", "m"], how="left").with_columns(pl.col("shift").fill_null(0.0)).with_columns(
        p2=pl.when(pl.col("p") <= 7200).then(pl.col("p") + pl.col("shift")).otherwise(pl.col("p")))
    x = x.with_columns(e0=(pl.col("p") - pl.col("y")) ** 2, e1=(pl.col("p2") - pl.col("y")) ** 2,
                       fold=pl.col("m").replace_strict({mm: i for i, (a, b) in enumerate(folds) for mm in (a, b)}))
    r1 = float(np.sqrt(x["e1"].mean()))
    per = x.group_by("fold").agg(((pl.col("e1").sum() - pl.col("e0").sum()) / N).alias("dmse")).sort("fold")
    lower = int((per["dmse"] < 0).sum())
    jj = x.filter(pl.col("m").is_in([1, 7])).group_by("m").agg(((pl.col("e1").sum() - pl.col("e0").sum()) / N / (2 * r0)).alias("drmse_contrib"))
    print(f"\n{mode}: OOF RMSE {r0:.3f} -> {r1:.3f} ({r1 - r0:+.3f} s), folds lower {lower}/6, median |shift| {s['shift'].abs().median():.1f} s")
    print("  per fold dMSE/N:", [round(v, 2) for v in per["dmse"].to_list()], "| Jan/Jul contribution to dRMSE:", jj.sort("m").to_dicts())
