import glob
import numpy as np
import polars as pl

raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "p": z["pred"], "y": z["y"]})
                 for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(
    m=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
N = d.height
base = float(((d["p"] - d["y"]) ** 2).sum())
r0 = np.sqrt(base / N)
print(f"OOF E106: N {N}, RMSE {r0:.3f}")
def run(key, pop, apply_to, label):
    b = d.filter(pop).group_by(key).agg(bias=(pl.col("p") - pl.col("y")).mean(), n=pl.len())
    x = d.join(b, on=key, how="left").with_columns(pl.col("bias").fill_null(0.0))
    x = x.with_columns(p2=pl.when(apply_to).then(pl.col("p") - pl.col("bias")).otherwise(pl.col("p")))
    sse = float(((x["p2"] - x["y"]) ** 2).sum())
    r = np.sqrt(sse / N)
    print(f"{label:70s} RMSE {r:.3f}  delta {r - r0:+.3f} s   (median |bias| {b['bias'].abs().median():.1f} s)")
clean_pop = pl.col("y") <= 3600
clean_apply = pl.col("p") <= 3600
run(["ADEP_mvt", "m"], clean_pop, clean_apply, "airport x month, bias on y<=3600, shift rows with pred<=3600")
run(["ADEP_mvt", "m"], pl.col("y") <= 7200, pl.col("p") <= 7200, "airport x month, bias on y<=7200, shift rows with pred<=7200")
run(["ADEP_mvt", "m"], pl.lit(True), pl.lit(True), "airport x month, bias on all rows, shift all rows")
run(["ADEP_mvt", "day"], clean_pop, clean_apply, "airport x day, bias on y<=3600, shift rows with pred<=3600")
run(["ADEP_mvt", "day"], pl.col("y") <= 7200, pl.col("p") <= 7200, "airport x day, bias on y<=7200, shift rows with pred<=7200")
run(["ADEP_mvt", "day"], pl.lit(True), pl.lit(True), "airport x day, bias on all rows, shift all rows")
print("\nper airport x month bias on y<=3600 (s), Jan and Jul:")
print(d.filter(clean_pop & pl.col("m").is_in([1, 7])).group_by("ADEP_mvt", "m").agg(bias=((pl.col("p") - pl.col("y")).mean()).round(1)).pivot(on="m", index="ADEP_mvt", values="bias").sort("ADEP_mvt"))
