"""Tail regimes for LIRF departures without an NM flight record, from an empirical table.

The multiclass LightGBM cannot represent the one-day-shift regime (14 rows in 2025, min_data_in_leaf=100)
and under-predicts the schedule regime at long delays. For LIRF rows without FLIGHT_ID and a delay
D = MVT - SCHED >= 4 h, the prediction is replaced by
    p_sched * D + p_day * (86400 + median day-shift taxi) + p_other * regressor,
with Laplace-smoothed probabilities per delay bin counted on 2025.

    FEATURES=data/features_v5.parquet uv run python src/lirf_table.py <base_version> <new_version>
Validation (probabilities from the 10 fit months, applied to Jan+Jul 2025): RMSE 390.3 -> 320.3.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

import train as T
from features import TARGET

BINS = [7200, 14400, 28800, 43200, 108000]
LABELS = ["<2h", "2-4h", "4-8h", "8-12h", "12-30h", ">30h"]
# ">30h" is left to the model: one 2025 row, never validated (second review)
OVERRIDE = ["4-8h", "8-12h", "12-30h"]
# Dirichlet pseudo-counts (sched, day, other) = (1, 0.2, 1.8): day-shift is absent below 12 h
# (1 of 1441 LIRF no-NM rows), so its prior mass is kept small where it was never observed.
ALPHA, ALPHA_DAY = 1.0, 0.2


def lirf_no_nm(d: pl.DataFrame) -> pl.DataFrame:
    return d.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_null()).with_columns(
        b=pl.col("d_mvt_sched").cut(BINS, labels=LABELS)
    )


def table(train: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    t = lirf_no_nm(train).with_columns(
        is_sched=((pl.col(TARGET) - pl.col("d_mvt_sched")).abs() <= T.TOL),
        is_day=pl.col(TARGET).is_between(T.DAY, T.DAY + 7200),
    )
    tab = t.group_by("b").agg(n=pl.len(), ns=pl.col("is_sched").sum(), nd=pl.col("is_day").sum()).with_columns(
        ps=(pl.col("ns") + ALPHA) / (pl.col("n") + 3 * ALPHA),
        pd=(pl.col("nd") + ALPHA_DAY) / (pl.col("n") + 3 * ALPHA),
    )
    day_taxi = float(train.filter(pl.col(TARGET).is_between(T.DAY, T.DAY + 7200)).select((pl.col(TARGET) - T.DAY).median()).item())
    return tab, day_taxi


def main() -> None:
    base, new = int(sys.argv[1]), int(sys.argv[2])
    df = pl.read_parquet(T.FEATURES_PATH, columns=["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "d_mvt_sched", "d_mvt_lobt", TARGET])
    tab, day_taxi = table(df.filter(pl.col("split") == "train"))
    print(tab.sort("b"), "day-shift taxi median", day_taxi)
    parts = np.load(f"submissions/parts_v{base}.npz")
    reg = pl.DataFrame({"MVT_ID_mvt": parts["mvt_id"], "reg": parts["reg"]})
    rank = lirf_no_nm(df.filter(pl.col("split") == "rank")).join(tab.select("b", "ps", "pd"), on="b", how="left").join(reg, on="MVT_ID_mvt")
    rank = rank.filter(pl.col("b").is_in(OVERRIDE)).with_columns(
        new=pl.col("ps") * pl.col("d_mvt_sched") + pl.col("pd") * (T.DAY + day_taxi) + (1 - pl.col("ps") - pl.col("pd")) * pl.col("reg")
    )
    print("overridden ranking rows by bin:", rank.group_by("b").len().sort("b").to_dicts())
    sub = pl.read_parquet(f"submissions/gentle-igloo_v{base}.parquet")
    out = sub.join(rank.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce(pl.col("new").round(0).cast(sub.schema[TARGET]), pl.col(TARGET)).alias(TARGET)
    ).select(sub.columns)
    assert out.height == sub.height and out[TARGET].null_count() == 0
    out.write_parquet(f"submissions/gentle-igloo_v{new}.parquet")
    changed = out.join(sub, on="MVT_ID_mvt", suffix="_old").filter(pl.col(TARGET) != pl.col(TARGET + "_old"))
    print(f"wrote v{new}: {changed.height} rows changed, mean |delta| {changed.select((pl.col(TARGET) - pl.col(TARGET + '_old')).abs().mean()).item():.0f}s")


if __name__ == "__main__":
    main()
