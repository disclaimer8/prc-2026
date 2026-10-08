"""Label-free live level of neighbouring NM-matched departures.

For every DEP row: median over NM-matched departures (AOBT_3 not null) at the same ADEP whose MVT_TIME lies within
+-HALF seconds of this row's MVT_TIME of (MVT_TIME - AOBT_3), and the neighbour count. Uses only MVT/AOBT_3/ADEP/SCHED
(never BLOCK/TAXITIME). Writes one compact parquet per year to the scratchpad.
"""
import os, sys, glob
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import pandas as pd
import polars as pl

HALF = 1800  # +-30 min
OUT = sys.argv[1]
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "FLIGHT_ID_mvt"]


def level(frame: pl.DataFrame) -> pl.DataFrame:
    d = frame.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        aobt=pl.col("AOBT_3_flt").dt.epoch("s").cast(pl.Float64),
        sched=pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        has_nm=pl.col("LOBT_flt").is_not_null(),
        flt_null=pl.col("FLIGHT_ID_mvt").is_null(),
    ).with_columns(ta=(pl.col("mvt") - pl.col("aobt")), D=pl.col("mvt") - pl.col("sched")).sort("ADEP_mvt", "mvt")
    out = []
    for ap, g in d.group_by("ADEP_mvt", maintain_order=True):
        gp = g.to_pandas()
        idx = pd.to_datetime(gp["mvt"].to_numpy(), unit="s")
        s = pd.Series(np.where(gp["has_nm"].to_numpy(), gp["ta"].to_numpy(), np.nan), index=idx)
        roll = s.rolling(f"{2 * HALF}s", center=True, min_periods=1)
        med = roll.median().to_numpy()
        cnt = roll.count().to_numpy()
        gp["level"] = med
        gp["n_nb"] = cnt.astype(np.int32)
        out.append(pl.from_pandas(gp[["MVT_ID_mvt", "ADEP_mvt", "mvt", "D", "ta", "has_nm", "flt_null", "level", "n_nb"]]))
    r = pl.concat(out)
    r = r.with_columns(
        month=pl.from_epoch(pl.col("mvt"), time_unit="s").dt.month().cast(pl.Int8),
        day=pl.from_epoch(pl.col("mvt"), time_unit="s").dt.date(),
    )
    return r


if __name__ == "__main__":
    files_2025 = sorted(glob.glob("data/prc-2026-datasets/training_2025-*.parquet"))
    r25 = level(pl.concat([pl.read_parquet(f, columns=COLS) for f in files_2025]))
    r25.write_parquet(f"{OUT}/level_2025.parquet")
    print("2025", r25.shape, "no-NM", (~r25["has_nm"]).sum())
    r26 = level(pl.read_parquet("data/prc-2026-datasets/final_ranking.parquet", columns=COLS))
    r26.write_parquet(f"{OUT}/level_2026.parquet")
    print("2026", r26.shape, "no-NM", (~r26["has_nm"]).sum())
