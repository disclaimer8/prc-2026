"""Feature engineering for PRC Data Challenge 2026 (taxi-out time).

Every feature is computed from columns that are present for *all* rows of the
ranking file: departure BLOCK_TIME / TAXITIME are blanked there, so they are
never read here for departures. Arrival block and taxi-in times stay available
and are used as congestion signals.
"""
from __future__ import annotations

import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

DATA = "data/prc-2026-datasets/"
TARGET = "TAXITIME_SEC_mvt"

AIRPORT_TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}

CATEGORICAL = [
    "ADEP_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "stand_prefix",
    "ap_runway",
    "ap_stand",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt",
    "ades_region",
    "airline_prefix",
    "rwy_config",
]


def load() -> pl.DataFrame:
    train = pl.read_parquet(DATA + "training_*.parquet").with_columns(pl.lit("train").alias("split"))
    rank = pl.read_parquet(DATA + "ranking.parquet").with_columns(pl.lit("rank").alias("split"))
    df = pl.concat([train, rank], how="vertical_relaxed")
    return df


def _secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def _epoch(col: str) -> pl.Expr:
    return (pl.col(col).dt.epoch("s")).cast(pl.Float64)


def _window_count(times_sorted: np.ndarray, t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Number of events with time in [t+lo, t+hi]."""
    return np.searchsorted(times_sorted, t + hi, side="right") - np.searchsorted(times_sorted, t + lo, side="left")


def _airport_congestion(g: pl.DataFrame) -> pl.DataFrame:
    """Traffic features for one airport; g holds all its movements."""
    dep = g.filter(pl.col("PHASE_mvt") == "DEP")
    arr = g.filter(pl.col("PHASE_mvt") == "ARR")

    dep_takeoff = np.sort(dep["t_mvt"].to_numpy())
    dep_aobt = np.sort(dep["t_aobt"].drop_nulls().to_numpy())
    arr_land = np.sort(arr["t_mvt"].to_numpy())

    arr_sorted = arr.select("t_mvt", "t_block", TARGET).drop_nulls().sort("t_block")
    inblock = arr_sorted["t_block"].to_numpy()
    taxi_in_cum = np.concatenate([[0.0], np.cumsum(arr_sorted[TARGET].cast(pl.Float64).to_numpy())])

    t = dep["t_mvt"].to_numpy()
    # off-block estimate: NM actual off-block, falling back to 15 min before take-off
    t_ob = dep["t_aobt"].fill_null(dep["t_mvt"] - 900).to_numpy()

    out = {"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy()}
    for m in (10, 20, 30, 60):
        w = m * 60.0
        out[f"dep_before_{m}"] = _window_count(dep_takeoff, t, -w, 0)
        out[f"dep_after_{m}"] = _window_count(dep_takeoff, t, 0, w)
        out[f"arr_before_{m}"] = _window_count(arr_land, t, -w, 0)
        out[f"arr_around_ob_{m}"] = _window_count(arr_land, t_ob, -w / 2, w / 2)
        out[f"offblocks_around_ob_{m}"] = _window_count(dep_aobt, t_ob, -w / 2, w / 2)
    # departures that have pushed back but not yet taken off at our off-block estimate
    out["taxiing_at_ob"] = np.searchsorted(dep_aobt, t_ob, side="right") - np.searchsorted(dep_takeoff, t_ob, side="left")
    out["takeoffs_between_ob_mvt"] = np.searchsorted(dep_takeoff, t, side="left") - np.searchsorted(dep_takeoff, t_ob, side="left")

    for m in (30, 60, 180):
        hi = np.searchsorted(inblock, t_ob, side="right")
        lo = np.searchsorted(inblock, t_ob - m * 60.0, side="left")
        n = hi - lo
        s = taxi_in_cum[hi] - taxi_in_cum[lo]
        out[f"taxi_in_mean_{m}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out[f"taxi_in_n_{m}"] = n

    return pl.DataFrame(out)


def _runway_congestion(dep: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for (_, _), g in dep.group_by(["ADEP_mvt", "RUNWAY_mvt"]):
        takeoff = np.sort(g["t_mvt"].to_numpy())
        t = g["t_mvt"].to_numpy()
        parts.append(
            pl.DataFrame(
                {
                    "MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(),
                    "rwy_dep_before_10": _window_count(takeoff, t, -600, 0),
                    "rwy_dep_before_30": _window_count(takeoff, t, -1800, 0),
                    "rwy_dep_after_10": _window_count(takeoff, t, 0, 600),
                }
            )
        )
    return pl.concat(parts)


def load_metar() -> pl.DataFrame:
    """Hourly-ish METAR observations (Iowa Environmental Mesonet ASOS archive)."""
    frames = []
    for ap in AIRPORT_TZ:
        m = pl.read_csv(f"data/metar/{ap}.csv", null_values=["null", "M"], infer_schema_length=0)
        frames.append(m)
    m = pl.concat(frames).rename({"station": "ADEP_mvt"})
    wx = pl.col("wxcodes").fill_null("")
    m = m.select(
        "ADEP_mvt",
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").dt.cast_time_unit("us").alias("metar_time"),
        pl.col("tmpc").cast(pl.Float64).alias("wx_temp"),
        (pl.col("tmpc").cast(pl.Float64) - pl.col("dwpc").cast(pl.Float64)).alias("wx_spread"),
        pl.col("sknt").cast(pl.Float64).alias("wx_wind"),
        pl.col("gust").cast(pl.Float64).alias("wx_gust"),
        pl.col("vsby").cast(pl.Float64).alias("wx_vis_mi"),
        pl.col("skyl1").cast(pl.Float64).alias("wx_ceiling_ft"),
        wx.str.contains("SN|SG|PL|GS").cast(pl.Int8).alias("wx_snow"),
        wx.str.contains("FZ").cast(pl.Int8).alias("wx_freezing"),
        wx.str.contains("RA|DZ|SH").cast(pl.Int8).alias("wx_rain"),
        wx.str.contains("TS|CB").cast(pl.Int8).alias("wx_thunder"),
        wx.str.contains("FG|BR|HZ").cast(pl.Int8).alias("wx_fog"),
    ).drop_nulls("metar_time")
    m = m.sort("ADEP_mvt", "metar_time").with_columns(
        # icing-condition proxy: near/below freezing with visible moisture
        ((pl.col("wx_temp") <= 3) & ((pl.col("wx_spread") <= 3) | (pl.col("wx_snow") == 1) | (pl.col("wx_freezing") == 1)))
        .cast(pl.Int8)
        .alias("wx_icing"),
    )
    m = m.with_columns(
        pl.col("wx_snow").rolling_max(6).over("ADEP_mvt").alias("wx_snow_recent"),
        pl.col("wx_icing").rolling_max(6).over("ADEP_mvt").alias("wx_icing_recent"),
    )
    return m


def add_weather(feat: pl.DataFrame) -> pl.DataFrame:
    m = load_metar()
    feat = feat.with_columns(
        pl.col("AOBT_3_flt").fill_null(pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=15)).alias("_ob_time")
    ).sort("_ob_time")
    out = feat.join_asof(
        m.sort("metar_time"), left_on="_ob_time", right_on="metar_time", by="ADEP_mvt", strategy="backward",
        tolerance="3h",
    )
    return out.with_columns(
        metar_age_min=(pl.col("_ob_time") - pl.col("metar_time")).dt.total_minutes().cast(pl.Float64)
    ).drop("_ob_time", "metar_time")


def add_runway_config(feat: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Runway configuration per airport-hour, inferred from the runway column of all movements.

    RUNWAY_mvt is filled for every ranking row, so this is available for 2026 as well and captures
    configuration changes (closures, new operating modes) that a stand x runway category cannot.
    """
    mv = df.with_columns(hour=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h")).drop_nulls("RUNWAY_mvt")
    per = mv.group_by("airport", "hour", "PHASE_mvt", "RUNWAY_mvt").len().filter(pl.col("len") >= 2)
    cfg = (
        per.sort("RUNWAY_mvt")
        .group_by("airport", "hour", "PHASE_mvt")
        .agg(rwys=pl.col("RUNWAY_mvt").str.join(","), n_rwys=pl.len(), total=pl.col("len").sum())
    )
    dep_cfg = cfg.filter(pl.col("PHASE_mvt") == "DEP").select("airport", "hour", dep_rwys="rwys", n_dep_rwys="n_rwys")
    arr_cfg = cfg.filter(pl.col("PHASE_mvt") == "ARR").select("airport", "hour", arr_rwys="rwys", n_arr_rwys="n_rwys")
    own = (
        mv.filter(pl.col("PHASE_mvt") == "DEP")
        .group_by("airport", "hour", "RUNWAY_mvt")
        .len()
        .with_columns(own_rwy_share_hour=pl.col("len") / pl.col("len").sum().over("airport", "hour"))
        .drop("len")
    )
    key = feat.select(
        "MVT_ID_mvt",
        "RUNWAY_mvt",
        airport=pl.col("ADEP_mvt"),
        hour=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"),
    )
    out = (
        key.join(dep_cfg, on=["airport", "hour"], how="left")
        .join(arr_cfg, on=["airport", "hour"], how="left")
        .join(own, on=["airport", "hour", "RUNWAY_mvt"], how="left")
        .select(
            "MVT_ID_mvt",
            "n_dep_rwys",
            "n_arr_rwys",
            "own_rwy_share_hour",
            rwy_config=pl.col("airport") + ":" + pl.col("dep_rwys").fill_null("") + "|" + pl.col("arr_rwys").fill_null(""),
        )
    )
    return feat.join(out, on="MVT_ID_mvt", how="left")


def build(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return one row per departure (train + ranking) with features and target."""
    if df is None:
        df = load()

    df = df.with_columns(
        t_mvt=_epoch("MVT_TIME_UTC_mvt"),
        t_block=_epoch("BLOCK_TIME_UTC_mvt"),
        t_aobt=_epoch("AOBT_3_flt"),
    )

    # airport of the movement: departures report ADEP, arrivals report ADES
    df = df.with_columns(
        airport=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt"))
    )
    congestion = pl.concat([_airport_congestion(g) for _, g in df.group_by("airport")])

    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    rwy = _runway_congestion(dep)

    local_parts = []
    for ap, tz in AIRPORT_TZ.items():
        loc = pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz)
        local_parts.append(
            dep.filter(pl.col("ADEP_mvt") == ap).select(
                "MVT_ID_mvt",
                local_hour=loc.dt.hour().cast(pl.Float64) + loc.dt.minute() / 60,
                local_dow=loc.dt.weekday(),
            )
        )
    local = pl.concat(local_parts)

    feat = (
        dep.join(congestion, on="MVT_ID_mvt", how="left")
        .join(rwy, on="MVT_ID_mvt", how="left")
        .join(local, on="MVT_ID_mvt", how="left")
        .with_columns(
            d_mvt_aobt=_secs("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
            d_mvt_eobt=_secs("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
            d_mvt_lobt=_secs("MVT_TIME_UTC_mvt", "LOBT_flt"),
            d_mvt_iobt=_secs("MVT_TIME_UTC_mvt", "IOBT_flt"),
            d_mvt_sched=_secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
            d_aobt_eobt=_secs("AOBT_3_flt", "EOBT_1_flt"),
            d_eobt_iobt=_secs("EOBT_1_flt", "IOBT_flt"),
            d_aobt_sched=_secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt"),
            flight_time_plan=_secs("ARVT_1_flt", "EOBT_1_flt"),
            diverted=(pl.col("ADES_flt") != pl.col("ADES_FILED_flt")).cast(pl.Int8),
            month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
            doy=pl.col("MVT_TIME_UTC_mvt").dt.ordinal_day(),
            stand_prefix=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)", 1),
            ap_runway=pl.col("ADEP_mvt") + "_" + pl.col("RUNWAY_mvt").fill_null("NA"),
            ap_stand=pl.col("ADEP_mvt") + "_" + pl.col("STAND_mvt").fill_null("NA"),
            ades_region=pl.col("ADES_mvt").str.slice(0, 2),
            airline_prefix=pl.col("CALLSIGN_flt").str.extract(r"^([A-Z]{3})", 1),
            aobt_minute_exact=(pl.col("AOBT_3_flt").dt.second() == 0).cast(pl.Int8),
        )
    )
    from geometry import add_geometry

    feat = add_runway_config(feat, df)
    feat = add_geometry(feat)
    return add_weather(feat)


if __name__ == "__main__":
    f = build()
    f.write_parquet(os.environ.get("FEATURES_OUT", "data/features_v4.parquet"))
    print(f.shape)
    print(f.filter(pl.col("split") == "rank").null_count().transpose(include_header=True).filter(pl.col("column_0") > 0))
