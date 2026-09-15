"""Review 15.09 'winter disruption': (1) out-of-fold bias of the LTFM 19-23.02.2025 snow event (fold 2+8);
(2) OOF bias of departures without NM outside LIRF by delay bin, overall and on days with >= 20 such departures;
(3) which ranking rows E4 (v9 vs v8) changed, including EHAM 2-7.01.2026.
    .venv/bin/python src/snow_event_check.py"""
import glob
import os
from datetime import datetime, timezone

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

Y = "TAXITIME_SEC_mvt"
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", Y]
raw = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP")
oof = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(p))["mvt_id"], "pred": z["pred"]}) for p in sorted(glob.glob("data/cv_preds/fold_*.npz"))])
x = raw.join(oof, on="MVT_ID_mvt").with_columns(nonm=pl.col("FLIGHT_ID_mvt").is_null(), D=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                                                date=pl.col("MVT_TIME_UTC_mvt").dt.date())
ev = x.filter((pl.col("ADEP_mvt") == "LTFM") & pl.col("MVT_TIME_UTC_mvt").is_between(datetime(2025, 2, 19, tzinfo=timezone.utc), datetime(2025, 2, 24, tzinfo=timezone.utc), closed="left"))
print("LTFM 19-23.02.2025 (OOF):", ev.group_by("nonm").agg(n=pl.len(), y_mean=pl.col(Y).mean().round(0), pred_mean=pl.col("pred").mean().round(0),
      bias=(pl.col("pred") - pl.col(Y)).mean().round(0), sched_share=((pl.col(Y) - pl.col("D")).abs() <= 6).mean().round(3)).sort("nonm").to_dicts())
n = x.filter(pl.col("nonm") & (pl.col("ADEP_mvt") != "LIRF"))
n = n.join(n.group_by("ADEP_mvt", "date").agg(n_day=pl.len()), on=["ADEP_mvt", "date"]).with_columns(Dbin=pl.col("D").cut([0, 1800, 3600, 7200, 14400], labels=["<0", "0-30m", "30-60m", "1-2h", "2-4h", ">4h"]))
for name, d in (("all days", n), ("days with >= 20 no-NM departures", n.filter(pl.col("n_day") >= 20))):
    print(f"no-NM outside LIRF, {name}:", d.group_by("Dbin").agg(n=pl.len(), y_mean=pl.col(Y).mean().round(0), bias=(pl.col("pred") - pl.col(Y)).mean().round(0)).sort("Dbin").to_dicts())
a, b = pl.read_parquet("submissions/gentle-igloo_v8.parquet"), pl.read_parquet("submissions/gentle-igloo_v9.parquet")
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
d = a.join(b, on="MVT_ID_mvt", suffix="_9").filter(pl.col(Y) != pl.col(f"{Y}_9")).join(rk, on="MVT_ID_mvt")
e = d.filter((pl.col("ADEP_mvt") == "EHAM") & pl.col("MVT_TIME_UTC_mvt").dt.date().is_between(pl.date(2026, 1, 2), pl.date(2026, 1, 7)))
print(f"E4 changed {d.height} ranking rows; EHAM 2-7.01.2026: {e.height} rows, mean prediction v8 {e[Y].mean():.0f} -> v9 {e[f'{Y}_9'].mean():.0f}")
