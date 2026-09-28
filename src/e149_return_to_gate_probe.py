"""E149 probe (journal 111): return to gate. When the surface track shows a first movement, a long stop and then the final
movement, which moment does the APDF off-block (y = MVT - BLOCK) match: the first movement or the final one? And is the
current model already right on those rows? offblock2 (kt1): t_first_move = first movement, t_move = start of the final
continuous movement, t_to = take-off; coverage 3 = track seen from the stand.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ob = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first_move", "t_move", "t_to"])
                for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2025-*.parquet"))])
raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"])
                 for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "p": z["pred"]}) for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
d = raw.join(ob, on="MVT_ID_mvt", how="inner").join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").filter(pl.col("coverage") == 3)
to = pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)
d = d.with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), gap=pl.col("t_move") - pl.col("t_first_move"),
                   y_first=to - pl.col("t_first_move"), y_final=to - pl.col("t_move"),
                   aobt=(pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64))
d = d.with_columns(err=pl.col("p") - pl.col("y"))
tot = float((d["err"] ** 2).sum())
print(f"coverage-3 departures 2025: {d.height}")
print(d.with_columns(g=pl.col("gap").cut([60, 300, 600, 1200, 1800])).group_by("g").agg(
    n=pl.len(), med_y_minus_first=(pl.col("y") - pl.col("y_first")).median(), med_y_minus_final=(pl.col("y") - pl.col("y_final")).median(),
    share_y_near_first=((pl.col("y") - pl.col("y_first")).abs() < 120).mean().round(3), share_y_near_final=((pl.col("y") - pl.col("y_final")).abs() < 120).mean().round(3),
    med_aobt_minus_final=(pl.col("aobt") - pl.col("y_final")).median(), mean_err=pl.col("err").mean().round(1),
    sse_share=((pl.col("err") ** 2).sum() / tot).round(4)).sort("g").to_pandas().to_string())

print("\nrule candidate: for coverage-3 rows with gap >= G, predict y_first (+ median offset) instead of the model")
N_all = 2085047
r0 = 282.459
for G in (300, 600, 900, 1200):
    x = d.filter(pl.col("gap") >= G)
    off = float((x["y"] - x["y_first"]).median())
    e_rule = (x["y_first"] + off - x["y"]) ** 2
    e_mod = x["err"] ** 2
    blend = ((0.5 * (x["y_first"] + off) + 0.5 * x["p"]) - x["y"]) ** 2
    d_rule, d_blend = float(e_rule.sum() - e_mod.sum()), float(blend.sum() - e_mod.sum())
    print(f"gap >= {G:5d}: rows {x.height:6d}, offset {off:+.0f} s; dRMSE on the full OOF if replaced {d_rule / N_all / (2 * r0):+.3f} s, "
          f"50/50 with the model {d_blend / N_all / (2 * r0):+.3f} s; months where replace helps "
          f"{int((x.with_columns(dd=(pl.col('y_first') + off - pl.col('y')) ** 2 - pl.col('err') ** 2).group_by(pl.col('MVT_TIME_UTC_mvt').dt.month()).agg(pl.col('dd').sum())['dd'] < 0).sum())}/12")
ob26 = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first_move", "t_move"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_2026-*.parquet"))])
g26 = ob26.filter(pl.col("coverage") == 3).with_columns(gap=pl.col("t_move") - pl.col("t_first_move"))
print(f"\n2026 ranking coverage-3 rows: {g26.height}; gap >= 600: {g26.filter(pl.col('gap') >= 600).height}; gap >= 1200: {g26.filter(pl.col('gap') >= 1200).height}")
