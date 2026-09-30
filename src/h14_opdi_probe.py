"""OPDI v002 flight events as a second off-block source and a de-icing marker (journal 168/170).

Organiser 06.09: "OPDI milestones for now do not cover off-block, but you can eventually use it". Events downloaded by
scripts/get_opdi_events.sh (Jan/Jul 2025 and 2026, departures from our 10 airports). Matching: call sign (CALLSIGN_flt, else
FLIGHT_mvt, spaces removed) + ADEP, OPDI take-off within 900 s of MVT_TIME. Reports per airport and period:
coverage of exit-parking_position / entry-deicing_pad / take-off, coverage where our adsb.lol ground coverage < 2,
accuracy of exit-parking vs BLOCK (2025), and for de-icing-pad rows the production OOF residual (does the model already know?).
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
TYPES = ["exit-parking_position", "entry-taxiway", "entry-deicing_pad", "exit-deicing_pad", "entry-runway", "take-off", "first_seen", "first-xing-fl50"]
ev = pl.concat([pl.read_parquet(p, columns=["flight_id", "flt_id", "adep", "type", "t"]) for p in sorted(glob.glob("data/opdi/ev_*.parquet"))])
ev = ev.filter(pl.col("type").is_in(TYPES)).with_columns(t=pl.col("t").dt.epoch("s").cast(pl.Float64))
first = ev.group_by("flight_id", "type").agg(pl.col("t").min(), pl.col("flt_id").first(), pl.col("adep").first())
w = first.pivot(on="type", index=["flight_id", "flt_id", "adep"], values="t")
for c in TYPES:
    if c not in w.columns:
        w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
# anchor: take-off if seen, else the first FL50 crossing, else first_seen (OpenSky rarely hears the ground at our airports)
w = w.with_columns(anchor=pl.coalesce("take-off", "first-xing-fl50", "first_seen")).filter(pl.col("anchor").is_not_null())
print(f"OPDI flights from our airports with an anchor event: {w.height}; with take-off {w['take-off'].is_not_null().sum()}, exit-parking {w['exit-parking_position'].is_not_null().sum()}, de-icing pad {w['entry-deicing_pad'].is_not_null().sum()}")

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"]
tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=[c for c in cols if c not in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")])
rk = rk.with_columns(pl.lit(None, dtype=tr.schema["BLOCK_TIME_UTC_mvt"]).alias("BLOCK_TIME_UTC_mvt"), pl.lit(None, dtype=tr.schema["TAXITIME_SEC_mvt"]).alias("TAXITIME_SEC_mvt")).select(cols)
d = pl.concat([tr, rk]).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64), blk=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
    per=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", "")).filter(pl.col("per").is_in(["2025-01", "2025-07", "2026-01", "2026-07"]))
j = d.join(w, left_on=["cs", "ADEP_mvt"], right_on=["flt_id", "adep"], how="left").with_columns(dt_to=(pl.col("anchor") - pl.col("mvt")))
j = j.with_columns(pl.when(pl.col("dt_to").is_between(-900, 1800)).then(pl.col("dt_to").abs()).otherwise(None).alias("ok")).sort("MVT_ID_mvt", "ok", nulls_last=True).unique("MVT_ID_mvt", keep="first")
for c in TYPES:
    j = j.with_columns(pl.when(pl.col("ok").is_not_null()).then(pl.col(c)).otherwise(None).alias(c))
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_*.parquet"))]).unique("MVT_ID_mvt")
j = j.join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0))
x = pl.col("exit-parking_position")
print("\nCoverage by airport and month (share of departures):")
t = j.group_by("ADEP_mvt", "per").agg(n=pl.len(), opdi_to=pl.col("ok").is_not_null().mean(), exit_park=x.is_not_null().mean(),
                                       deice=pl.col("entry-deicing_pad").is_not_null().mean(), adsb_cov2=(pl.col("coverage") >= 2).mean(),
                                       exit_park_where_adsb_lt2=(x.is_not_null() & (pl.col("coverage") < 2)).mean(),
                                       n_new=(x.is_not_null() & (pl.col("coverage") < 2)).sum()).sort("ADEP_mvt", "per")
print(t.with_columns(pl.col(pl.Float64).round(3)))
print("\nAccuracy 2025: exit-parking minus BLOCK (s), and our adsb.lol t_first minus BLOCK on the same rows:")
a = j.filter(pl.col("blk").is_not_null() & x.is_not_null()).with_columns(e=x - pl.col("blk"), e_adsb=pl.col("t_first") - pl.col("blk"))
print(a.group_by("ADEP_mvt").agg(n=pl.len(), med=pl.col("e").median(), in60=(pl.col("e").abs() <= 60).mean(), in180=(pl.col("e").abs() <= 180).mean(),
                                 adsb_n=pl.col("e_adsb").is_not_null().sum(), adsb_in60=(pl.col("e_adsb").abs() <= 60).mean()).sort("ADEP_mvt")
      .with_columns(pl.col(pl.Float64).round(3)))
print("\nDe-icing pad rows (2025 January): taxi-out and production OOF residual")
oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")}) for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
dz = j.filter(pl.col("per").str.starts_with("2025")).join(oof, on="MVT_ID_mvt", how="inner").with_columns(r=pl.col("y") - pl.col("pred"),
                                                                                                            deiced=pl.col("entry-deicing_pad").is_not_null())
print(dz.group_by("ADEP_mvt", "per", "deiced").agg(n=pl.len(), y_med=pl.col("y").median(), r_mean=pl.col("r").mean(), r_rmse=(pl.col("r") ** 2).mean().sqrt())
      .filter(pl.col("deiced")).sort("ADEP_mvt", "per"))
print("de-icing pad rows in ranking by airport/month:", j.filter(pl.col("per").str.starts_with("2026") & pl.col("entry-deicing_pad").is_not_null())
      .group_by("ADEP_mvt", "per").len().sort("ADEP_mvt", "per").rows())
print("\n2026 check without labels: OPDI exit-parking minus our adsb.lol t_first where both exist (coverage >= 2):")
b = j.filter(pl.col("per").str.starts_with("2026") & x.is_not_null() & (pl.col("coverage") >= 2) & pl.col("t_first").is_not_null()).with_columns(e=x - pl.col("t_first"))
print(b.group_by("ADEP_mvt", "per").agg(n=pl.len(), med=pl.col("e").median(), in60=(pl.col("e").abs() <= 60).mean(), in180=(pl.col("e").abs() <= 180).mean()).sort("ADEP_mvt", "per")
      .with_columns(pl.col(pl.Float64).round(3)))
a2 = j.filter(pl.col("per").str.starts_with("2025") & x.is_not_null() & (pl.col("coverage") >= 2) & pl.col("t_first").is_not_null()).with_columns(e=x - pl.col("t_first"))
print("same on 2025:", a2.group_by("ADEP_mvt").agg(n=pl.len(), med=pl.col("e").median(), in60=(pl.col("e").abs() <= 60).mean()).sort("ADEP_mvt").with_columns(pl.col(pl.Float64).round(3)).rows())
