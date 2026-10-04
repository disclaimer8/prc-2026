"""E185 (idea 3 of the 04.10 competitor sweep, merry-mushroom v8): the "movement-only" view of the feature table.

    .venv/bin/python src/e185_motion_features.py      ->  data/e185_motion_features.parquet (MVT_ID_mvt + REPLACED columns)

A departure without an NM record (all *_flt null; 2025: 22 470 rows, ranking 5 290) gets its "around off-block" features from
the fallback anchor t_mvt - 900 (src/features.py: `t_aobt.fill_null(t_mvt - 900)`), and its own AOBT is absent from the
"other departures' off-blocks" sets. A row WITH an NM record is anchored at its own AOBT_3 and counts itself. A member trained
on all rows with only the NM columns hidden would therefore still read the row's own AOBT through these columns
(`takeoffs_between_ob_mvt` is a count of take-offs between AOBT_3 and MVT, `metar_age_min` + MVT minute gives AOBT_3 mod 30 min,
`taxiing_at_ob` / `offblocks_around_ob_*` count the row itself) and would learn on 2 M rows a meaning those columns do not have
on the rows it is meant for.

This script recomputes those columns for EVERY departure exactly as features.py computes them for a row without NM:
anchor t_mvt - 900, the row's own AOBT taken out of the off-block sets. On the rows without NM the result must equal
data/features_fin1b.parquet bit for bit (checked at the end; the script refuses to write otherwise), so the member sees on
2 M training rows the same representation the no-NM rows have. `airline_prefix` is rebuilt from FLIGHT_mvt for every row
(E106 rule `^([A-Z]{3})\\d`; NM rows have it from the callsign), checked the same way.

Not recomputed, hidden instead (src/e185_motion_only_cv.py, HIDE): e157_ladder_q (null without NM), the NM fields and their
differences. Columns anchored at t_mvt (dep_/arr_before/after, rwy_*, nb_*, nbf_*, nbs_*, fw_*, e157_* except ladder_q, geo_*,
route_*, dst_*) already mean the same on both kinds of rows: other departures only, the own value is excluded.
"""
import os
import sys
import time

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import features as F  # noqa: E402

OUT = os.environ.get("E185_OUT", "data/e185_motion_features.parquet")
REF = os.environ.get("E185_REF", "data/features_fin1b.parquet")
OB = 900.0  # features.py fallback: off-block = 15 min before take-off
WX = ["wx_temp", "wx_spread", "wx_wind", "wx_gust", "wx_vis_mi", "wx_ceiling_ft", "wx_snow", "wx_freezing", "wx_rain", "wx_thunder",
      "wx_fog", "wx_icing", "wx_snow_recent", "wx_icing_recent", "metar_age_min"]
STAND = ["stand_prev_arr_age", "stand_prev_arr_taxi_in", "stand_next_arr_gap", "stand_prev_arr_same_type"]
CONG = ([f"arr_around_ob_{m}" for m in (10, 20, 30, 60)] + [f"offblocks_around_ob_{m}" for m in (10, 20, 30, 60)]
        + ["taxiing_at_ob", "takeoffs_between_ob_mvt"] + [f"taxi_in_{s}_{m}" for m in (30, 60, 180) for s in ("mean", "n")])
REPLACED = CONG + STAND + WX + ["airline_prefix"]


def congestion(g: pl.DataFrame) -> pl.DataFrame:
    """features._airport_congestion with the anchor t_mvt - 900 for every departure and the row's own AOBT out of the sets."""
    dep = g.filter(pl.col("PHASE_mvt") == "DEP")
    arr = g.filter(pl.col("PHASE_mvt") == "ARR")
    dep_takeoff = np.sort(dep["t_mvt"].to_numpy())
    dep_aobt = np.sort(dep["t_aobt"].drop_nulls().to_numpy())
    dep_takeoff_with_aobt = np.sort(dep.filter(pl.col("t_aobt").is_not_null())["t_mvt"].to_numpy())
    arr_land = np.sort(arr["t_mvt"].to_numpy())
    arr_sorted = arr.select("t_mvt", "t_block", F.TARGET).drop_nulls().sort("t_block")
    inblock = arr_sorted["t_block"].to_numpy()
    taxi_in_cum = np.concatenate([[0.0], np.cumsum(arr_sorted[F.TARGET].cast(pl.Float64).to_numpy())])
    t = dep["t_mvt"].to_numpy()
    t_ob = t - OB
    own = dep["t_aobt"].to_numpy().astype(float)  # NaN without NM
    out = {"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()}
    for m in (10, 20, 30, 60):
        w = m * 60.0
        out[f"arr_around_ob_{m}"] = F._window_count(arr_land, t_ob, -w / 2, w / 2)
        own_in = (own >= t_ob - w / 2) & (own <= t_ob + w / 2)  # NaN compares False
        out[f"offblocks_around_ob_{m}"] = F._window_count(dep_aobt, t_ob, -w / 2, w / 2) - own_in
    # own take-off (t) is always after t_ob, so the row adds to the first count only, when its AOBT <= t_ob
    out["taxiing_at_ob"] = (np.searchsorted(dep_aobt, t_ob, side="right") - np.searchsorted(dep_takeoff_with_aobt, t_ob, side="left")
                            - (own <= t_ob))
    out["takeoffs_between_ob_mvt"] = np.searchsorted(dep_takeoff, t, side="left") - np.searchsorted(dep_takeoff, t_ob, side="left")
    for m in (30, 60, 180):
        hi = np.searchsorted(inblock, t_ob, side="right")
        lo = np.searchsorted(inblock, t_ob - m * 60.0, side="left")
        n = hi - lo
        s = taxi_in_cum[hi] - taxi_in_cum[lo]
        out[f"taxi_in_mean_{m}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out[f"taxi_in_n_{m}"] = n
    return pl.DataFrame(out)


def stand_arrivals(dep: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """The stand part of features.add_neighbor_features (previous / next arrival on the stand) at the anchor t_mvt - 900."""
    arr = (
        df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("t_block").is_not_null())
        .select(ADEP_mvt=pl.col("ADES_mvt"), STAND_mvt="STAND_mvt", inb="t_block", arr_taxi_in=pl.col(F.TARGET).cast(pl.Float64),
                arr_type=pl.col("AIRCRAFT_TYPE_mvt"))
        .sort("inb")
    )
    key = dep.select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "t_mvt", t_ob=pl.col("t_mvt") - OB).drop_nulls("STAND_mvt").sort("t_ob")
    prev = key.join_asof(arr, left_on="t_ob", right_on="inb", by=["ADEP_mvt", "STAND_mvt"], strategy="backward").select(
        "MVT_ID_mvt", stand_prev_arr_age=pl.col("t_ob") - pl.col("inb"), stand_prev_arr_taxi_in="arr_taxi_in",
        stand_prev_arr_type="arr_type")
    nxt = key.join_asof(arr, left_on="t_ob", right_on="inb", by=["ADEP_mvt", "STAND_mvt"], strategy="forward").select(
        "MVT_ID_mvt", stand_next_arr_gap=pl.col("t_mvt") - pl.col("inb"))
    stale = 3 * 86400
    o = dep.select("MVT_ID_mvt", "AIRCRAFT_TYPE_mvt").join(prev, on="MVT_ID_mvt", how="left").join(nxt, on="MVT_ID_mvt", how="left")
    o = o.with_columns(
        pl.when(pl.col("stand_prev_arr_age") > stale).then(None).otherwise(pl.col("stand_prev_arr_taxi_in")).alias("stand_prev_arr_taxi_in"),
        pl.when(pl.col("stand_prev_arr_age") > stale).then(None).otherwise(pl.col("stand_prev_arr_type")).alias("stand_prev_arr_type"),
        pl.when(pl.col("stand_prev_arr_age") > stale).then(None).otherwise(pl.col("stand_prev_arr_age")).alias("stand_prev_arr_age"),
        pl.when(pl.col("stand_next_arr_gap").abs() > stale).then(None).otherwise(pl.col("stand_next_arr_gap")).alias("stand_next_arr_gap"),
    )
    return o.with_columns(stand_prev_arr_same_type=(pl.col("stand_prev_arr_type") == pl.col("AIRCRAFT_TYPE_mvt")).cast(pl.Int8)).select(
        "MVT_ID_mvt", *STAND)


def weather(dep: pl.DataFrame) -> pl.DataFrame:
    """features.add_weather at MVT - 15 min for every departure (the fallback it uses when AOBT_3 is missing)."""
    m = F.load_metar()
    k = dep.select("MVT_ID_mvt", "ADEP_mvt", _ob_time=pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=15)).sort("_ob_time")
    o = k.join_asof(m.sort("metar_time"), left_on="_ob_time", right_on="metar_time", by="ADEP_mvt", strategy="backward", tolerance="3h")
    return o.with_columns(metar_age_min=(pl.col("_ob_time") - pl.col("metar_time")).dt.total_minutes().cast(pl.Float64)).select("MVT_ID_mvt", *WX)


if __name__ == "__main__":
    t0 = time.time()
    df = F.load().with_columns(t_mvt=F._epoch("MVT_TIME_UTC_mvt"), t_block=F._epoch("BLOCK_TIME_UTC_mvt"), t_aobt=F._epoch("AOBT_3_flt"))
    df = df.with_columns(airport=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")))
    cong = pl.concat([congestion(g) for _, g in df.group_by("airport")])
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    res = (dep.select("MVT_ID_mvt", airline_prefix=pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{3})\d", 1))
           .join(cong, on="MVT_ID_mvt", how="left").join(stand_arrivals(dep, df), on="MVT_ID_mvt", how="left")
           .join(weather(dep), on="MVT_ID_mvt", how="left").select("MVT_ID_mvt", *REPLACED))
    assert res["MVT_ID_mvt"].n_unique() == res.height == dep.height
    print(f"recomputed {len(REPLACED)} columns for {res.height} departures ({time.time() - t0:.0f} s)", flush=True)

    # parity: on rows WITHOUT an NM record the view must equal the reference table bit for bit
    ref = pl.read_parquet(REF, columns=["MVT_ID_mvt", "split", "LOBT_flt", *REPLACED])
    j = ref.join(res, on="MVT_ID_mvt", how="left", suffix="_m")
    assert j.height == ref.height and j["MVT_ID_mvt"].n_unique() == j.height
    nonm = j["LOBT_flt"].is_null().to_numpy()
    bad = []
    for c in REPLACED:
        a, b = j[c], j[f"{c}_m"].cast(j[c].dtype)
        same = ((a == b) | (a.is_null() & b.is_null())).fill_null(False).to_numpy()
        if a.dtype.is_float():
            same |= (a.is_nan() & b.is_nan()).fill_null(False).to_numpy()
        mis_nonm, chg_nm = int((~same & nonm).sum()), float((~same & ~nonm).mean() / max((~nonm).mean(), 1e-12))
        print(f"  {c:28s} differs on no-NM rows {mis_nonm:6d} / {int(nonm.sum())} | changed on NM rows {chg_nm:6.1%}")
        if mis_nonm:
            bad.append(c)
    if bad:
        sys.exit(f"refusing to write: the movement-only view does not reproduce the no-NM rows for {bad}")
    res.write_parquet(OUT)
    print(f"wrote {OUT} {res.shape} ({time.time() - t0:.0f} s)", flush=True)
