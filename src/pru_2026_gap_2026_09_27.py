import glob
import numpy as np
import polars as pl

APS = ["EGLL", "LEMD", "LIRF", "LEBL", "EHAM"]
def pru_year(y):
    return pl.read_csv(f"data/taxi_out_additional_time_{y}.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APS)).select(
        ADEP_mvt=pl.col("APT_ICAO"), m=pl.col("MONTH_NUM").cast(pl.Int8),
        pru=(pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) * 60 / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64),
        tf=pl.col("TF").cast(pl.Int64))
raw = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f))["mvt_id"], "y": z["y"]}) for f in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz"))])
t = raw.join(oof.with_columns(pl.col("MVT_ID_mvt").cast(raw["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").with_columns(m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))
off = t.group_by("ADEP_mvt", "m").agg(ymean_f=pl.col("y").filter(pl.col("y") <= 7200).mean()).join(pru_year(2025), on=["ADEP_mvt", "m"]).with_columns(
    off=pl.col("ymean_f") - pl.col("pru")).select("ADEP_mvt", "m", "off")
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int8))
out = []
for v in (366, 414):
    s = pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet").rename({"TAXITIME_SEC_mvt": "p"})
    r = rk.join(s.with_columns(pl.col("MVT_ID_mvt").cast(rk["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt").filter(pl.col("ADEP_mvt").is_in(APS))
    g = r.group_by("ADEP_mvt", "m").agg(n=pl.len(), n_f=(pl.col("p") <= 7200).sum(), pmean_f=pl.col("p").filter(pl.col("p") <= 7200).mean())
    g = g.join(pru_year(2026), on=["ADEP_mvt", "m"]).join(off, on=["ADEP_mvt", "m"]).with_columns(
        target=pl.col("pru") + pl.col("off")).with_columns(gap=pl.col("pmean_f") - pl.col("target"))
    g = g.with_columns(gain_s=-(pl.col("gap") ** 2) * pl.col("n_f") / 344841 / (2 * 235.5))
    print(f"\nv{v}: mean prediction (pred<=7200) minus PRU-implied label mean (PRU 2026 + same-month 2025 offset), and the gain if the gap were a pure bias")
    print(g.sort("ADEP_mvt", "m").select("ADEP_mvt", "m", "n", (pl.col("n") / pl.col("tf")).round(3).alias("n/tf"), pl.col("pmean_f").round(1), pl.col("target").round(1), pl.col("gap").round(1), pl.col("gain_s").round(3)).to_pandas().to_string())
    print(f"total if every gap were a pure bias: {float(g['gain_s'].sum()):+.3f} s; EGLL only {float(g.filter(pl.col('ADEP_mvt') == 'EGLL')['gain_s'].sum()):+.3f} s")
