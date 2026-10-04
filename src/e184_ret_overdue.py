"""E184 (04.10, idea 2 of the 04.10 competitor sweep): "overdue at the gate" counts of outgoing-boat v34 (`src/pista.py`
`colunas_retencao`, lines 393-499 of EnioAguiar/prc-taxiout-2026), originally unique-umbrella's `overdue_runway_queue`.

At the row's own off-block moment t, over the OTHER departures of the same airport (runway):

  ret_overdue_apt / ret_overdue_rwy : departures already past EOBT_1 but not yet off-block, EOBT_1 <= t < AOBT_3.
        Counted as #{EOBT_1 <= t} - #{AOBT_3 <= t} over the guarded set of departures with both clocks,
        0 <= AOBT_3 - EOBT_1 < 7200 and MVT >= AOBT_3 (their `_retidas`); with EOBT <= AOBT for every member the
        difference is exactly the number of open intervals [EOBT, AOBT) containing t.
  ret_atraso_15m : mean AOBT_3 - SCHED_TIME of the departures of the airport whose AOBT_3 is in [t - 900, t)
        (their `_atraso_recente`: no clipping, null when the window is empty).

t = own AOBT_3. Without AOBT_3 (1.1 % of 2025 departures, 1.5 % of ranking) t = MVT - 900, the same off-block estimate
that `taxiing_at_ob` / `offblocks_around_ob_*` already use (src/features.py) and E137 used; outgoing-boat leaves the whole
block NaN there. The row itself never enters its own count: at t = own AOBT_3 its interval [EOBT, AOBT) is already closed
and its AOBT is not in [t - 900, t); a row without AOBT_3 is not in either set.

Inputs: MVT_TIME, RUNWAY, AOBT_3, EOBT_1 and SCHED_TIME of departures. In the ranking file only BLOCK_TIME and TAXITIME of
departures are blanked (journal 137), so every input is present there (RUNWAY 100 %, AOBT_3/EOBT_1 98.5 %, SCHED 100 %).
Neighbours come from the raw files (all departures of 2025 + ranking, one timeline per airport), not from a filtered table.

    OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e184_ret_overdue.py   -> data/features_e184.parquet
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

DATA = "data/prc-2026-datasets/"
OUT = os.environ.get("E184_OUT", "data/features_e184.parquet")
MAX_HOLD = 7200.0  # their MAX_RETIDO: AOBT - EOBT at or above this is a date artefact
WINDOW = 900.0  # their JANELA_ATRASO
FALLBACK = 900.0  # off-block estimate without AOBT_3 = MVT - 15 min (features.py)
NEW = ["ret_overdue_apt", "ret_overdue_rwy", "ret_atraso_15m"]

ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)  # noqa: E731
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"]
parts = [pl.read_parquet(p, columns=cols).with_columns(split=pl.lit("train")) for p in sorted(glob.glob(DATA + "training_*.parquet"))]
parts.append(pl.read_parquet(DATA + "ranking.parquet", columns=cols).with_columns(split=pl.lit("rank")))
dep = (
    pl.concat(parts)
    .filter(pl.col("PHASE_mvt") == "DEP")
    .select("MVT_ID_mvt", "split", "ADEP_mvt", "RUNWAY_mvt",
            t_mvt=ep("MVT_TIME_UTC_mvt"), t_aobt=ep("AOBT_3_flt"), t_eobt=ep("EOBT_1_flt"), t_sched=ep("SCHED_TIME_UTC_mvt"))
    .with_columns(t_ob=pl.col("t_aobt").fill_null(pl.col("t_mvt") - FALLBACK))
)
assert dep["MVT_ID_mvt"].is_duplicated().sum() == 0
# guards written with explicit null handling: a null comparison is null, and filter() would drop it silently
held = (pl.col("t_eobt").is_not_null() & pl.col("t_aobt").is_not_null() & pl.col("t_mvt").is_not_null()
        & ((pl.col("t_aobt") - pl.col("t_eobt")) >= 0) & ((pl.col("t_aobt") - pl.col("t_eobt")) < MAX_HOLD)
        & (pl.col("t_mvt") >= pl.col("t_aobt"))).fill_null(False)
dep = dep.with_columns(held=held, has_delay=(pl.col("t_aobt").is_not_null() & pl.col("t_sched").is_not_null()).fill_null(False))


def overdue(g: pl.DataFrame) -> np.ndarray:
    """#{EOBT <= t} - #{AOBT <= t} over the guarded departures of g, at every row's t_ob."""
    h = g.filter(pl.col("held"))
    e, a = np.sort(h["t_eobt"].to_numpy()), np.sort(h["t_aobt"].to_numpy())
    q = g["t_ob"].to_numpy()
    return (np.searchsorted(e, q, "right") - np.searchsorted(a, q, "right")).astype(np.float64)


def recent_delay(g: pl.DataFrame) -> np.ndarray:
    """Mean AOBT - SCHED of departures with AOBT in [t - 900, t); NaN when none."""
    h = g.filter(pl.col("has_delay")).sort("t_aobt")
    when = h["t_aobt"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum((h["t_aobt"] - h["t_sched"]).to_numpy())])
    q = g["t_ob"].to_numpy()
    hi = np.searchsorted(when, q, "left")
    lo = np.searchsorted(when, q - WINDOW, "left")
    n = hi - lo
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


out_ap, out_rw = [], []
for _, g in dep.group_by("ADEP_mvt"):
    out_ap.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "ret_overdue_apt": overdue(g), "ret_atraso_15m": recent_delay(g)}))
for _, g in dep.drop_nulls(["ADEP_mvt", "RUNWAY_mvt"]).group_by("ADEP_mvt", "RUNWAY_mvt"):
    out_rw.append(pl.DataFrame({"MVT_ID_mvt": g["MVT_ID_mvt"], "ret_overdue_rwy": overdue(g)}))
res = (dep.select("MVT_ID_mvt", "split", "ADEP_mvt")
       .join(pl.concat(out_ap), on="MVT_ID_mvt", how="left", maintain_order="left")
       .join(pl.concat(out_rw), on="MVT_ID_mvt", how="left", maintain_order="left")
       .with_columns(pl.col("ret_atraso_15m").fill_nan(None)))
assert res.height == dep.height and res["ret_overdue_apt"].min() >= 0 and res["ret_overdue_rwy"].min() >= 0

summ = res.group_by("split", "ADEP_mvt").agg(
    n=pl.len(),
    **{f"{c}_cov": pl.col(c).is_not_null().mean().round(4) for c in NEW},
    ovd_apt_mean=pl.col("ret_overdue_apt").mean().round(2), ovd_apt_p90=pl.col("ret_overdue_apt").quantile(0.9),
    ovd_rwy_mean=pl.col("ret_overdue_rwy").mean().round(2), ovd_rwy_p90=pl.col("ret_overdue_rwy").quantile(0.9),
    atr_med=pl.col("ret_atraso_15m").median().round(0), atr_p10=pl.col("ret_atraso_15m").quantile(0.1).round(0),
    atr_p90=pl.col("ret_atraso_15m").quantile(0.9).round(0),
).sort("ADEP_mvt", "split")
with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250):
    print(summ)
    print(res.group_by("split").agg(n=pl.len(), **{f"{c}_cov": pl.col(c).is_not_null().mean().round(4) for c in NEW},
                                    **{f"{c}_mean": pl.col(c).mean().round(2) for c in NEW}).sort("split"))
res.select("MVT_ID_mvt", *NEW).write_parquet(OUT)
print("wrote", OUT, res.height, "rows")
