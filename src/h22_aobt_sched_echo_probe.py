"""H22: is |AOBT_3 - SCHED| <= 6 s an observable detector of the "block = schedule" echo?

The model already carries d_aobt_sched as a feature (src/features.py:501). This probe asks the
different question: as a RULE (recording-practice class, the one that transferred: E14, E106), how
precise is it, what is the oracle value on 2025, and how big is the pocket in ranking 2026, where
AOBT_3 and SCHED are both visible but BLOCK is not.

Read-only. Usage: uv run python src/h22_aobt_sched_echo_probe.py
"""

from __future__ import annotations

import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "prc-2026-datasets"
APS = ["LIRF", "LFPG", "LTFM", "EDDF", "EHAM", "EGLL", "LEMD", "LEBL", "LSZH", "EDDM"]

COLS = [
    "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt", "LOBT_flt", "AOBT_3_flt", "IOBT_flt", "EOBT_1_flt",
]


def frame(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(COLS)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .filter(pl.col("ADEP_mvt").is_in(APS))
        .with_columns(
            pl.col("LOBT_flt").is_null().alias("no_nm"),
            (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("d_aobt_sched"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("d_block_sched"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("D"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"),
        )
    )


def main() -> None:
    tr = pl.concat([frame(p) for p in sorted(DS.glob("training_*.parquet"))], how="vertical")
    rank = frame(DS / "ranking.parquet")

    # ---- 2025, rows with NM and a visible block: is the echo visible in AOBT_3? ----
    w = (
        tr.filter(~pl.col("no_nm") & pl.col("y").is_not_null())
        .with_columns(
            echo=pl.col("d_block_sched").abs() <= 6,
            det=pl.col("d_aobt_sched").abs() <= 6,
            det65=pl.col("d_aobt_sched").abs() <= 65,
        )
    ).collect()

    print("=== 2025, departures WITH NM: detector |AOBT_3-SCHED|<=6 vs truth block=sched ===")
    rows = []
    for ap in APS:
        s = w.filter(pl.col("ADEP_mvt") == ap)
        n = len(s)
        if n == 0:
            continue
        echo = int(s["echo"].sum())
        det = int(s["det"].sum())
        tp = int((s["det"] & s["echo"]).sum())
        rows.append(
            dict(ap=ap, n=n, echo=echo, echo_rate=round(echo / n, 4), det=det,
                 precision=round(tp / det, 3) if det else None,
                 recall=round(tp / echo, 3) if echo else None,
                 det65=int(s["det65"].sum()),
                 prec65=round(int((s["det65"] & s["echo"]).sum()) / s["det65"].sum(), 3) if s["det65"].sum() else None)
        )
    print(pl.DataFrame(rows).to_pandas().to_string(index=False))

    print("\n=== 2025 LIRF with NM: what does AOBT_3 look like on echo rows? ===")
    l = w.filter(pl.col("ADEP_mvt") == "LIRF")
    for lab, sub in [("echo (block=sched)", l.filter(pl.col("echo"))), ("not echo", l.filter(~pl.col("echo")))]:
        d = sub["d_aobt_sched"].drop_nulls()
        print(f"{lab:20s} n={len(sub):6d}  |d_aobt_sched|<=6: {(d.abs()<=6).mean():.3f}  "
              f"<=65: {(d.abs()<=65).mean():.3f}  median {d.median():8.0f} s")

    # ---- oracle value of the rule on 2025 (LIRF, with NM) ----
    print("\n=== Oracle: rule 'predict D' where detector fires (2025, LIRF with NM) ===")
    base_med = l.filter(~pl.col("echo"))["y"].median()
    for name, sub in [("detector fires", l.filter(pl.col("det"))), ("detector silent", l.filter(~pl.col("det")))]:
        sse_now = ((sub["y"] - base_med) ** 2).sum()
        sse_rule = ((sub["y"] - sub["D"]) ** 2).sum()
        print(f"{name:18s} n={len(sub):6d}  SSE(reg med {base_med:.0f}) {sse_now:12.3e} -> SSE(D) {sse_rule:12.3e} "
              f" d={100*(sse_rule-sse_now)/max(sse_now,1):+.1f}%")

    # ---- ranking 2026: pocket size, observable ----
    print("\n=== RANKING 2026: pocket size (AOBT_3 and SCHED visible, BLOCK hidden) ===")
    r = rank.collect()
    rows = []
    for ap in APS:
        s = r.filter(pl.col("ADEP_mvt") == ap)
        nm = s.filter(~s["no_nm"])
        rows.append(dict(
            ap=ap, dep=len(s), with_nm=len(nm),
            det6=int((nm["d_aobt_sched"].abs() <= 6).sum()),
            det65=int((nm["d_aobt_sched"].abs() <= 65).sum()),
            det6_and_D_gt_1h=int(((nm["d_aobt_sched"].abs() <= 6) & (nm["D"] > 3600)).sum()),
            no_nm=int(s["no_nm"].sum()),
        ))
    print(pl.DataFrame(rows).to_pandas().to_string(index=False))

    print("\n=== RANKING 2026 LIRF with NM, detector fires: D distribution ===")
    lr = r.filter((pl.col("ADEP_mvt") == "LIRF") & ~r["no_nm"] & (r["d_aobt_sched"].abs() <= 6))
    print(lr.select(pl.col("D").describe()).to_pandas().to_string(index=False))
    print("rows with D > 3600:", len(lr.filter(pl.col("D") > 3600)), " D > 14400:", len(lr.filter(pl.col("D") > 14400)))


if __name__ == "__main__":
    main()
