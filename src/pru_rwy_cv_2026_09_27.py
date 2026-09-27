import glob
import numpy as np
import polars as pl

w = pl.read_parquet("data/pru_runway_monthly.parquet").filter(pl.col("year") == 2025).rename({"ap": "ADEP_mvt", "rwy": "RUNWAY_mvt"})
raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "p": z["pred"], "y": z["y"]}) for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))
g = d.group_by("ADEP_mvt", "RUNWAY_mvt", "m").agg(n=pl.len(), ymean_f=pl.col("y").filter(pl.col("y") <= 7200).mean(),
                                                   pmean_f=pl.col("p").filter(pl.col("p") <= 7200).mean()).join(w, on=["ADEP_mvt", "RUNWAY_mvt", "m"], how="inner")
g = g.with_columns(off=pl.col("ymean_f") - pl.col("pru_s"))
print(f"runway-months matched: {g.height}; rows covered {int(g['n'].sum())} of {d.height}")
print("offset spread by airport (std over runway-months, s):", g.group_by("ADEP_mvt").agg(pl.col("off").std().round(1)).sort("ADEP_mvt").to_dicts())
folds = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
N, r0 = d.height, float(np.sqrt(((d["p"] - d["y"]) ** 2).mean()))
for mode in ("other_months", "same_season"):
    parts = []
    for a, b in folds:
        for mm in (a, b):
            tm = [x for x in range(1, 13) if x not in (a, b)]
            if mode == "same_season":
                win = [1, 2, 3, 11, 12] if mm in (1, 2, 3, 11, 12) else [4, 5, 6, 7, 8, 9, 10]
                tm = [x for x in tm if x in win]
            oh = g.filter(pl.col("m").is_in(tm)).group_by("ADEP_mvt", "RUNWAY_mvt").agg(off_hat=pl.col("off").mean())
            parts.append(g.filter(pl.col("m") == mm).join(oh, on=["ADEP_mvt", "RUNWAY_mvt"]).select(
                "ADEP_mvt", "RUNWAY_mvt", "m", shift=pl.col("pru_s") + pl.col("off_hat") - pl.col("pmean_f")))
    s = pl.concat(parts)
    x = d.join(s, on=["ADEP_mvt", "RUNWAY_mvt", "m"], how="left").with_columns(pl.col("shift").fill_null(0.0)).with_columns(
        p2=pl.when(pl.col("p") <= 7200).then(pl.col("p") + pl.col("shift")).otherwise(pl.col("p")))
    x = x.with_columns(e0=(pl.col("p") - pl.col("y")) ** 2, e1=(pl.col("p2") - pl.col("y")) ** 2,
                       fold=pl.col("m").replace_strict({mm: i for i, (a, b) in enumerate(folds) for mm in (a, b)}))
    r1 = float(np.sqrt(x["e1"].mean()))
    per = x.group_by("fold").agg(((pl.col("e1").sum() - pl.col("e0").sum()) / N).alias("dmse")).sort("fold")
    jul = x.filter(pl.col("m") == 7); jan = x.filter(pl.col("m") == 1)
    print(f"{mode}: OOF RMSE {r0:.3f} -> {r1:.3f} ({r1 - r0:+.3f} s), folds lower {int((per['dmse'] < 0).sum())}/6, median |shift| {s['shift'].abs().median():.1f} s; "
          f"Jan dSSE {float((jan['e1'] - jan['e0']).sum()):.3g}, Jul dSSE {float((jul['e1'] - jul['e0']).sum()):.3g}")
