"""E163 (journal 181): destination arrival ATFM delay (apt_dly_*, ansperformance.eu/data) as base-model features.
Per departure (2025 and ranking): the destination's (ADES_mvt) daily arrival ATFM delay on the UTC day of take-off:
dst_dly_per_arr, dst_share_dly, dst_w_per_arr, dst_cap_per_arr, dst_known. Writes data/features_e163.parquet = features_e106 + dst_*.
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

raw_d = pl.concat([pl.read_csv(f"data/ext_pru/apt_dly_{y}.csv", infer_schema_length=0, encoding="utf8-lossy") for y in (2024, 2025, 2026)])
dc = [c for c in raw_d.columns if c.startswith("DLY_APT_ARR") or c.startswith("FLT_ARR")]
raw_d = raw_d.with_columns([pl.col(c).cast(pl.Float64) for c in dc])
per = lambda c: (pl.col(c) / pl.col("FLT_ARR_1")).fill_nan(0.0)
dst = raw_d.select(dest=pl.col("APT_ICAO"), day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date(), dst_dly_per_arr=per("DLY_APT_ARR_1"),
                   dst_share_dly=per("FLT_ARR_1_DLY"), dst_w_per_arr=per("DLY_APT_ARR_W_1"),
                   dst_cap_per_arr=((pl.col("DLY_APT_ARR_C_1") + pl.col("DLY_APT_ARR_G_1")) / pl.col("FLT_ARR_1")).fill_nan(0.0), dst_known=pl.lit(1.0)).unique(["dest", "day"])
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
files = sorted(glob.glob("data/prc-2026-datasets/training_*.parquet")) + ["data/prc-2026-datasets/ranking.parquet"]
dep = pl.concat([pl.read_parquet(p, columns=cols) for p in files]).filter(pl.col("PHASE_mvt") == "DEP").select(
    "MVT_ID_mvt", dest=pl.col("ADES_mvt"), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
feat = dep.join(dst, on=["dest", "day"], how="left").with_columns(pl.col("dst_known").fill_null(0.0)).drop("dest", "day")
f = pl.read_parquet("data/features_e106.parquet").join(feat, on="MVT_ID_mvt", how="left")
for s in ("train", "rank"):
    x = f.filter(pl.col("split") == s)
    print(s, x.height, "destination known", round(float(x["dst_known"].mean()), 3), "delayed day", round(float((x["dst_dly_per_arr"].fill_null(0) > 0).mean()), 3))
f.write_parquet("data/features_e163.parquet")
print("wrote data/features_e163.parquet", f.shape)
