"""H23: OPDI ground-progress events beyond exit-parking (entry/exit-taxiway, entry-runway).

H14 downloaded entry-taxiway but only reported exit-parking coverage. The event counts differ by 5x
(EDDF 2026: entry-taxiway 54,779 vs exit-parking 11,473), so the E159-style stage may apply to a much
larger population than the 2,088 rows of v709 / 14-22k of v729-v719.

For each event type this reports, per airport and period:
  - coverage among matched departures, and among departures where our adsb.lol coverage < 2;
  - accuracy against BLOCK on 2025 (labels visible): median (event - BLOCK), share within 60/180 s;
  - the money number: RMSE of the naive predictor y = MVT - event against the production OOF RMSE on
    the same rows (rows with our coverage < 2 first, then all rows with the event).

Read-only. Usage: uv run python src/h23_opdi_taxiway_events_probe.py
"""
import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(24)

TYPES = ["exit-parking_position", "exit-apron", "entry-taxiway", "exit-taxiway", "entry-runway",
         "take-off", "first_seen", "first-xing-fl50", "entry-deicing_pad"]

ev = pl.concat([pl.read_parquet(p, columns=["flight_id", "flt_id", "adep", "type", "t"])
                for p in sorted(glob.glob("data/opdi/ev_*.parquet"))])
ev = ev.filter(pl.col("type").is_in(TYPES)).with_columns(t=pl.col("t").dt.epoch("s").cast(pl.Float64))
first = ev.group_by("flight_id", "type").agg(pl.col("t").min(), pl.col("flt_id").first(), pl.col("adep").first())
w = first.pivot(on="type", index=["flight_id", "flt_id", "adep"], values="t")
for c in TYPES:
    if c not in w.columns:
        w = w.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
w = w.with_columns(anchor=pl.coalesce("take-off", "first-xing-fl50", "first_seen")).filter(pl.col("anchor").is_not_null())

cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "CALLSIGN_flt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"]
tr = pl.concat([pl.read_parquet(p, columns=cols) for p in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))])
rk_cols = [c for c in cols if c not in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")]
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=rk_cols)
rk = rk.with_columns(pl.lit(None, dtype=tr.schema["BLOCK_TIME_UTC_mvt"]).alias("BLOCK_TIME_UTC_mvt"),
                     pl.lit(None, dtype=tr.schema["TAXITIME_SEC_mvt"]).alias("TAXITIME_SEC_mvt")).select(cols)
d = pl.concat([tr, rk]).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
    blk=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
    per=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    cs=pl.coalesce("CALLSIGN_flt", "FLIGHT_mvt").str.replace_all(" ", ""))
d = d.filter(pl.col("per").is_in(["2025-01", "2025-07", "2026-01", "2026-07"]))

j = d.join(w, left_on=["cs", "ADEP_mvt"], right_on=["flt_id", "adep"], how="left").with_columns(dt_to=(pl.col("anchor") - pl.col("mvt")))
j = (j.with_columns(pl.when(pl.col("dt_to").is_between(-900, 1800)).then(pl.col("dt_to").abs()).otherwise(None).alias("ok"))
       .sort("MVT_ID_mvt", "ok", nulls_last=True).unique("MVT_ID_mvt", keep="first"))
for c in TYPES:
    j = j.with_columns(pl.when(pl.col("ok").is_not_null()).then(pl.col(c)).otherwise(None).alias(c))
cov = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob("data/adsblol_kt1/offblock2_*.parquet"))]).unique("MVT_ID_mvt")
j = j.join(cov, on="MVT_ID_mvt", how="left").with_columns(pl.col("coverage").fill_null(0))

oof = pl.concat([pl.DataFrame({k: np.load(p)[k] for k in ("mvt_id", "pred", "y")})
                 for p in sorted(glob.glob("data/cv_preds_e116prod/fold_*.npz"))]).rename({"mvt_id": "MVT_ID_mvt"})
j = j.join(oof, on="MVT_ID_mvt", how="left")

print("=== Coverage of each OPDI event among departures, by airport x period ===")
agg = [pl.len().alias("n"), pl.col("ok").is_not_null().mean().alias("matched"), (pl.col("coverage") < 2).mean().alias("adsb_lt2")]
for c in ["exit-parking_position", "exit-apron", "entry-taxiway", "exit-taxiway", "entry-runway", "take-off"]:
    agg.append(pl.col(c).is_not_null().mean().alias(c[:14]))
t = j.group_by("ADEP_mvt", "per").agg(*agg).sort("ADEP_mvt", "per")
print(t.with_columns(pl.col(pl.Float64).round(3)).to_pandas().to_string(index=False))

print("\n=== Same, restricted to rows where our adsb.lol coverage < 2 (the blind pocket) ===")
t2 = (j.filter(pl.col("coverage") < 2).group_by("ADEP_mvt", "per")
        .agg(n=pl.len(),
             exit_park=pl.col("exit-parking_position").is_not_null().mean(),
             entry_tw=pl.col("entry-taxiway").is_not_null().mean(),
             exit_tw=pl.col("exit-taxiway").is_not_null().mean(),
             entry_rwy=pl.col("entry-runway").is_not_null().mean())
        .sort("ADEP_mvt", "per"))
print(t2.with_columns(pl.col(pl.Float64).round(3)).to_pandas().to_string(index=False))

print("\n=== 2025 accuracy: event minus BLOCK (s), and RMSE of naive y = MVT - event vs production OOF ===")
for c in ["exit-parking_position", "exit-apron", "entry-taxiway", "exit-taxiway", "entry-runway"]:
    s = j.filter(pl.col("per").str.starts_with("2025") & pl.col(c).is_not_null() & pl.col("blk").is_not_null() & pl.col("y").is_not_null())
    if len(s) == 0:
        print(f"{c}: no rows"); continue
    e = (s[c] - s["blk"])
    naive = (s["mvt"] - s[c])
    r_naive = float(np.sqrt(((naive - s["y"]) ** 2).mean()))
    r_oof = float(np.sqrt(((s["pred"] - s["y"]) ** 2).mean())) if s["pred"].null_count() == 0 else float("nan")
    blind = s.filter(pl.col("coverage") < 2)
    r_naive_b = float(np.sqrt((((blind["mvt"] - blind[c]) - blind["y"]) ** 2).mean())) if len(blind) else float("nan")
    r_oof_b = float(np.sqrt(((blind["pred"] - blind["y"]) ** 2).mean())) if len(blind) and blind["pred"].null_count() == 0 else float("nan")
    print(f"{c:24s} n={len(s):6d} med(e-blk)={e.median():7.0f} in60={float((e.abs()<=60).mean()):.3f} in180={float((e.abs()<=180).mean()):.3f} "
          f"| RMSE naive {r_naive:7.1f} vs OOF {r_oof:7.1f} | blind n={len(blind):5d} naive {r_naive_b:7.1f} vs OOF {r_oof_b:7.1f}")

print("\n=== 2026 (no labels): event minus our adsb.lol-free naive check -- median (MVT - event) by airport, Jan vs Jul ===")
for c in ["exit-parking_position", "entry-taxiway", "exit-taxiway"]:
    s = j.filter(pl.col("per").str.starts_with("2026") & pl.col(c).is_not_null())
    g = (s.group_by("ADEP_mvt", "per").agg(n=pl.len(), med_taxi=(pl.col("mvt") - pl.col(c)).median())
           .sort("ADEP_mvt", "per"))
    print(f"--- {c}")
    print(g.with_columns(pl.col(pl.Float64).round(0)).to_pandas().to_string(index=False))
