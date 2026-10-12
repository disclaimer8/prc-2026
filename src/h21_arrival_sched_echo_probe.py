"""H21: arrivals in ranking as a 2026 detector of the "block = schedule" recording practice.

Departure labels are hidden in ranking, but arrival rows carry BLOCK_TIME (on-block) and
SCHED_TIME. If an airport's AODB fills the schedule when the NM record is missing, the same
artefact should be visible on arrivals without an NM record -- with no labels needed.

Baseline: the same measurement on the 2025 training files, where departure labels confirm the
practice (LIRF no-NM, D >= 4 h: 79.5 %).

Read-only probe. Usage: uv run python src/h21_arrival_sched_echo_probe.py
"""

from __future__ import annotations

import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "prc-2026-datasets"

COLS = [
    "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt",
]


def frame(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(COLS + ["LOBT_flt"])
        .with_columns(
            # "no NM record" = the movement was not matched to a Network Manager flight (LOBT absent),
            # the same definition the pipeline uses (src/features.py:519)
            pl.col("LOBT_flt").is_null().alias("no_nm"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds().abs().alias("d_sched"),
            pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"),
            pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
        )
        .filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null() & pl.col("SCHED_TIME_UTC_mvt").is_not_null())
    )


def summarise(lf: pl.LazyFrame, by: list[str]) -> pl.DataFrame:
    return (
        lf.group_by(by + ["PHASE_mvt", "no_nm"])
        .agg(
            n=pl.len(),
            echo6=(pl.col("d_sched") <= 6).mean(),
            echo65=(pl.col("d_sched") <= 65).mean(),
            med_d=pl.col("d_sched").median(),
            med_y=pl.col("TAXITIME_SEC_mvt").median(),
        )
        .sort(by + ["PHASE_mvt", "no_nm"])
        .collect()
    )


def main() -> None:
    aps = ["LIRF", "LFPG", "LTFM", "EDDF", "EHAM", "EGLL", "LEMD", "LEBL", "LSZH", "EDDM"]

    rank = frame(DS / "ranking.parquet").filter(pl.col("ADEP_mvt").is_in(aps))
    print("=== RANKING (Jan+Jul 2026), by airport x month x phase x NM ===")
    print(summarise(rank, ["ADEP_mvt", "month"]).to_pandas().to_string(index=False))

    print("\n=== 2025 baseline, same measurement (all 12 months) ===")
    tr = pl.concat([frame(p) for p in sorted(DS.glob("training_*.parquet"))], how="vertical")
    tr = tr.filter(pl.col("ADEP_mvt").is_in(aps))
    base = summarise(tr, ["ADEP_mvt"])
    print(base.to_pandas().to_string(index=False))

    print("\n=== LIRF arrivals without NM: rate by month, 2025 vs 2026 ===")
    lf = (
        pl.concat([tr, rank], how="vertical")
        .filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("PHASE_mvt") == "ARR") & pl.col("no_nm"))
        .group_by("month")
        .agg(n=pl.len(), echo6=(pl.col("d_sched") <= 6).mean(), echo65=(pl.col("d_sched") <= 65).mean())
        .sort("month")
    )
    print(lf.collect().to_pandas().to_string(index=False))

    print("\n=== LIRF arrivals without NM: worst days in ranking (echo6 rate, n >= 5) ===")
    days = (
        rank.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("PHASE_mvt") == "ARR") & pl.col("no_nm"))
        .group_by("day")
        .agg(n=pl.len(), echo6=(pl.col("d_sched") <= 6).mean(), echo65=(pl.col("d_sched") <= 65).mean())
        .filter(pl.col("n") >= 5)
        .sort("echo6", descending=True)
        .head(15)
    )
    print(days.collect().to_pandas().to_string(index=False))

    print("\n=== Departures for reference: 2025 (labels visible) vs ranking (hidden) ===")
    dep = (
        pl.concat([tr, rank], how="vertical")
        .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("no_nm"))
        .group_by(["ADEP_mvt", pl.col("month").is_in(["2026-01", "2026-07"]).alias("is_rank")])
        .agg(n=pl.len(), echo6=(pl.col("d_sched") <= 6).mean(), echo65=(pl.col("d_sched") <= 65).mean())
        .sort(["ADEP_mvt", "is_rank"])
    )
    print(dep.collect().to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
