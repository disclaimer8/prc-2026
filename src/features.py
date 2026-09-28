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

# NO_AOBT=1 (18.09): measure what abstaining from AOBT_3_flt costs. Every place that reads the NM actual
# off-block time reads the filed LOBT instead, so the derived columns (congestion around off-block,
# neighbours, day median, target encoding) keep their names and meaning but no longer carry AOBT_3.
# The direct AOBT columns then duplicate their LOBT twins and are left out of the model matrix by
# DROP_PREFIX; the proxy chain skips the AOBT link (train.py).
# E72 (19.09): x_plan_res and x_hprox_* were measured and REJECTED (+0.06 s, lower in 3 of 6). They stay in
# the code because the measurement is worth keeping reproducible, but off by default: the canonical
# DROP_PREFIX is "day_,tq_,wxr_,month" and does not drop x_, so a rebuild would otherwise put three
# rejected features into the accepted configuration without anyone choosing it.
X_FEATS = os.environ.get("X_FEATS") == "1"
NO_AOBT = os.environ.get("NO_AOBT") == "1"
AOBT_SRC = "LOBT_flt" if NO_AOBT else "AOBT_3_flt"

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
    # take-offs of the same departures that have an AOBT: pushed-back-but-airborne must be counted on one set,
    # otherwise the difference accumulates every departure without AOBT and drifts over the calendar
    dep_takeoff_with_aobt = np.sort(dep.filter(pl.col("t_aobt").is_not_null())["t_mvt"].to_numpy())
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
    out["taxiing_at_ob"] = np.searchsorted(dep_aobt, t_ob, side="right") - np.searchsorted(dep_takeoff_with_aobt, t_ob, side="left")
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
        pl.col(AOBT_SRC).fill_null(pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=15)).alias("_ob_time")
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


def _window_stats(times: np.ndarray, vals: np.ndarray, q: np.ndarray, lo: float, hi: float, exclude_self: np.ndarray | None):
    """Count, mean and std of vals with time in [q+lo, q+hi] (times sorted); optionally removes the row itself."""
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    cs2 = np.concatenate([[0.0], np.cumsum(vals * vals)])
    a = np.searchsorted(times, q + lo, side="left")
    b = np.searchsorted(times, q + hi, side="right")
    n = (b - a).astype(float)
    s = cs[b] - cs[a]
    s2 = cs2[b] - cs2[a]
    if exclude_self is not None:
        inside = ~np.isnan(exclude_self)
        n = n - inside
        s = s - np.nan_to_num(exclude_self)
        s2 = s2 - np.nan_to_num(exclude_self) ** 2
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, (s2 - n * mean**2) / np.maximum(n - 1, 1), np.nan)
    return n, mean, np.sqrt(np.clip(var, 0, None))


def add_neighbor_features(feat: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Observed NM taxi proxies (take-off minus AOBT_3) of other departures, plus the previous arrival on the stand.

    Other movements are fully observed in the ranking file, and features built from them are allowed.
    """
    dep = feat.select(
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "t_mvt",
        prox=pl.when(pl.col("d_mvt_aobt").is_between(0, 7200) & (pl.col("aobt_infeasible") != 1)).then(pl.col("d_mvt_aobt")),
        t_ob=pl.col("t_aobt").fill_null(pl.col("t_mvt") - 900),
    )
    out = [dep.select("MVT_ID_mvt")]
    groups = {"ap": ["ADEP_mvt"], "rwy": ["ADEP_mvt", "RUNWAY_mvt"], "stand": ["ADEP_mvt", "STAND_mvt"]}
    windows = {"ap": [(-1800, 0), (-1800, 1800)], "rwy": [(-3600, 0), (-1800, 1800)], "stand": [(-86400 * 3, 0)]}
    for gname, keys in groups.items():
        parts = []
        for _, g in dep.drop_nulls(keys).group_by(keys):
            obs = g.drop_nulls("prox").sort("t_mvt")
            times, vals = obs["t_mvt"].to_numpy(), obs["prox"].to_numpy()
            q = g["t_mvt"].to_numpy()
            own = g["prox"].to_numpy().astype(float)
            cols = {"MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy()}
            for lo, hi in windows[gname]:
                self_in = np.where((lo <= 0) & (hi >= 0), own, np.nan)
                n, mean, std = _window_stats(times, vals, q, lo, hi, self_in)
                tag = f"nb_{gname}_{-lo // 60}_{hi // 60}"
                cols[f"{tag}_n"], cols[f"{tag}_mean"], cols[f"{tag}_std"] = n, mean, std
            # last observed neighbour strictly before take-off
            idx = np.searchsorted(times, q, side="left") - 1
            has = idx >= 0
            cols[f"nb_{gname}_last"] = np.where(has, vals[np.clip(idx, 0, None)] if len(vals) else np.nan, np.nan)
            cols[f"nb_{gname}_last_age"] = np.where(has, q - (times[np.clip(idx, 0, None)] if len(times) else 0), np.nan)
            parts.append(pl.DataFrame(cols))
        out.append(pl.concat(parts, how="diagonal_relaxed"))
    nb = out[0]
    for p_ in out[1:]:
        nb = nb.join(p_, on="MVT_ID_mvt", how="left")

    # previous and next arrival on the same stand around the off-block estimate
    arr = (
        df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("STAND_mvt").is_not_null() & pl.col("t_block").is_not_null())
        .select(ADEP_mvt=pl.col("ADES_mvt"), STAND_mvt="STAND_mvt", inb="t_block", arr_taxi_in=pl.col(TARGET).cast(pl.Float64),
                arr_type=pl.col("AIRCRAFT_TYPE_mvt"))
        .sort("inb")
    )
    key = dep.select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "t_ob", "t_mvt").drop_nulls("STAND_mvt").sort("t_ob")
    prev = key.join_asof(arr, left_on="t_ob", right_on="inb", by=["ADEP_mvt", "STAND_mvt"], strategy="backward").select(
        "MVT_ID_mvt", stand_prev_arr_age=pl.col("t_ob") - pl.col("inb"), stand_prev_arr_taxi_in="arr_taxi_in",
        stand_prev_arr_type="arr_type",
    )
    nxt = key.join_asof(arr, left_on="t_ob", right_on="inb", by=["ADEP_mvt", "STAND_mvt"], strategy="forward").select(
        "MVT_ID_mvt", stand_next_arr_gap=pl.col("t_mvt") - pl.col("inb"),
    )
    feat = feat.join(nb, on="MVT_ID_mvt", how="left").join(prev, on="MVT_ID_mvt", how="left").join(nxt, on="MVT_ID_mvt", how="left")
    # the ranking frame holds January and July 2026 back to back: a "previous" event more than 3 days away
    # (for early July that is late January) says nothing about the stand, so it is dropped
    stale = 3 * 86400
    feat = feat.with_columns(
        pl.when(pl.col("nb_stand_last_age") > stale).then(None).otherwise(pl.col("nb_stand_last")).alias("nb_stand_last"),
        pl.when(pl.col("nb_stand_last_age") > stale).then(None).otherwise(pl.col("nb_stand_last_age")).alias("nb_stand_last_age"),
        pl.when(pl.col("stand_prev_arr_age") > stale).then(None).otherwise(pl.col("stand_prev_arr_taxi_in")).alias("stand_prev_arr_taxi_in"),
        pl.when(pl.col("stand_prev_arr_age") > stale).then(None).otherwise(pl.col("stand_prev_arr_type")).alias("stand_prev_arr_type"),
        pl.when(pl.col("stand_prev_arr_age") > stale).then(None).otherwise(pl.col("stand_prev_arr_age")).alias("stand_prev_arr_age"),
        pl.when(pl.col("stand_next_arr_gap").abs() > stale).then(None).otherwise(pl.col("stand_next_arr_gap")).alias("stand_next_arr_gap"),
        pl.when(pl.col("nb_ap_last_age") > stale).then(None).otherwise(pl.col("nb_ap_last")).alias("nb_ap_last"),
        pl.when(pl.col("nb_rwy_last_age") > stale).then(None).otherwise(pl.col("nb_rwy_last")).alias("nb_rwy_last"),
    )
    return feat.with_columns(
        stand_prev_arr_same_type=(pl.col("stand_prev_arr_type") == pl.col("AIRCRAFT_TYPE_mvt")).cast(pl.Int8),
        nb_own_minus_ap=pl.col("d_mvt_aobt") - pl.col("nb_ap_30_30_mean"),
        nb_own_minus_rwy=pl.col("d_mvt_aobt") - pl.col("nb_rwy_30_30_mean"),
    ).drop("stand_prev_arr_type")


def add_runway_arrivals(feat: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Landings on the departure runway around take-off (mixed-mode runways interleave departures and arrivals)."""
    arr = df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("RUNWAY_mvt").is_not_null()).select(
        ADEP_mvt=pl.col("ADES_mvt"), RUNWAY_mvt="RUNWAY_mvt", t_land="t_mvt"
    )
    groups = {k: np.sort(g["t_land"].to_numpy()) for k, g in arr.group_by(["ADEP_mvt", "RUNWAY_mvt"])}
    parts = []
    for key, g in feat.select("MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "t_mvt").drop_nulls("RUNWAY_mvt").group_by(["ADEP_mvt", "RUNWAY_mvt"]):
        t = g["t_mvt"].to_numpy()
        land = groups.get(key)
        if land is None or len(land) == 0:
            land = np.array([np.inf])
        before = np.searchsorted(land, t, side="right")
        prev_t = np.where(before > 0, land[np.clip(before - 1, 0, None)], np.nan)
        next_t = np.where(before < len(land), land[np.clip(before, 0, len(land) - 1)], np.nan)
        parts.append(
            pl.DataFrame(
                {
                    "MVT_ID_mvt": g["MVT_ID_mvt"].to_numpy(),
                    "rwy_arr_before_5": _window_count(land, t, -300, 0),
                    "rwy_arr_before_10": _window_count(land, t, -600, 0),
                    "rwy_arr_around_10": _window_count(land, t, -600, 600),
                    "rwy_arr_before_30": _window_count(land, t, -1800, 0),
                    "rwy_since_last_arr": np.where(np.isfinite(prev_t), t - prev_t, np.nan),
                    "rwy_to_next_arr": np.where(np.isfinite(next_t), next_t - t, np.nan),
                }
            )
        )
    out = pl.concat(parts).with_columns(
        # the ranking frame holds January and July 2026 back to back: gaps across it are not meaningful
        pl.when(pl.col("rwy_since_last_arr") > 86400).then(None).otherwise(pl.col("rwy_since_last_arr")).alias("rwy_since_last_arr"),
        pl.when(pl.col("rwy_to_next_arr") > 86400).then(None).otherwise(pl.col("rwy_to_next_arr")).alias("rwy_to_next_arr"),
    )
    return feat.join(out, on="MVT_ID_mvt", how="left")


def add_day_context(feat: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Airport-day context: share of movements without an NM record (a matching outage) and the day's NM taxi proxy level.

    Both use only fields that the ranking file keeps for every movement.
    """
    parts = []
    for ap, tz in AIRPORT_TZ.items():
        parts.append(
            df.filter(pl.col("airport") == ap).select(
                "MVT_ID_mvt", "airport", "PHASE_mvt", "FLIGHT_ID_mvt",
                local_day=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz).dt.date(),
            )
        )
    mv = pl.concat(parts)
    day = mv.group_by("airport", "local_day").agg(day_nonm_rate=pl.col("FLIGHT_ID_mvt").is_null().mean(), day_movements=pl.len())
    day = day.with_columns(
        day_nonm_ratio=pl.col("day_nonm_rate") / pl.col("day_nonm_rate").median().over("airport").clip(lower_bound=1e-4)
    )
    dep_day = feat.select("MVT_ID_mvt", "d_mvt_aobt", "aobt_infeasible").join(
        mv.select("MVT_ID_mvt", "airport", "local_day"), on="MVT_ID_mvt", how="left"
    )
    level = (
        dep_day.filter(pl.col("d_mvt_aobt").is_between(0, 7200) & (pl.col("aobt_infeasible") != 1))
        .group_by("airport", "local_day")
        .agg(day_da_med=pl.col("d_mvt_aobt").median())
    )
    level = level.with_columns(day_da_ratio=pl.col("day_da_med") / pl.col("day_da_med").median().over("airport"))
    ctx = (
        dep_day.select("MVT_ID_mvt", "airport", "local_day")
        .join(day, on=["airport", "local_day"], how="left")
        .join(level, on=["airport", "local_day"], how="left")
        .select("MVT_ID_mvt", "day_nonm_rate", "day_nonm_ratio", "day_movements", "day_da_med", "day_da_ratio")
    )
    return feat.join(ctx, on="MVT_ID_mvt", how="left")


def add_hour_context(feat: pl.DataFrame) -> pl.DataFrame:
    """The NM taxi proxy level of the airport-HOUR, the hour-resolution twin of day_da_med (E72, 18.09).

    merry-quicksand credit this with -3.12 s on the leaderboard for the rows without an NM record, where the
    flight's own proxy is missing and the neighbours' is all there is. Our day_da_med is per airport-day,
    twenty-four times coarser. Computed over the matched rows of the same airport-hour, so it reads other
    flights' AOBT and never a label; null where fewer than five matched rows support it.
    """
    key = feat.select("MVT_ID_mvt", "ADEP_mvt", "d_mvt_aobt", "aobt_infeasible",
                      hour=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h"))
    stats = (
        key.filter(pl.col("d_mvt_aobt").is_between(0, 7200) & (pl.col("aobt_infeasible") != 1))
        .group_by("ADEP_mvt", "hour")
        .agg(x_hprox_med=pl.col("d_mvt_aobt").median(), x_hprox_n=pl.len())
    )
    ctx = (
        key.select("MVT_ID_mvt", "ADEP_mvt", "hour")
        .join(stats, on=["ADEP_mvt", "hour"], how="left")
        .with_columns(x_hprox_med=pl.when(pl.col("x_hprox_n") >= 5).then(pl.col("x_hprox_med")))
        .select("MVT_ID_mvt", "x_hprox_med", "x_hprox_n")
    )
    return feat.join(ctx, on="MVT_ID_mvt", how="left")


def add_runway_wind(feat: pl.DataFrame) -> pl.DataFrame:
    """Head- and crosswind on the departure runway from METAR wind (IEM archive) and the runway designator heading."""
    frames = [pl.read_csv(f"data/metar_wind/{ap}.csv", null_values=["null", "M"], infer_schema_length=0) for ap in AIRPORT_TZ]
    w = (
        pl.concat(frames)
        .select(
            ADEP_mvt=pl.col("station"),
            wind_time=pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").dt.cast_time_unit("us"),
            drct=pl.col("drct").cast(pl.Float64),
            sknt=pl.col("sknt").cast(pl.Float64),
        )
        .drop_nulls(["wind_time", "sknt"])
        .sort("wind_time")
    )
    base = feat.select(
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt",
        t=pl.col("MVT_TIME_UTC_mvt"),
        heading=pl.col("RUNWAY_mvt").str.extract(r"^(\d{2})", 1).cast(pl.Float64) * 10,
    ).sort("t")
    j = base.join_asof(w, left_on="t", right_on="wind_time", by="ADEP_mvt", strategy="backward", tolerance="3h")
    rad = ((pl.col("drct") - pl.col("heading")) * np.pi / 180)
    j = j.select(
        "MVT_ID_mvt",
        # wind direction is where it blows from: positive headwind when it blows along the runway heading's reverse
        wxr_headwind=pl.when(pl.col("drct") > 0).then(pl.col("sknt") * rad.cos()).otherwise(0.0),
        wxr_crosswind=pl.when(pl.col("drct") > 0).then((pl.col("sknt") * rad.sin()).abs()).otherwise(0.0),
        wxr_variable=((pl.col("drct") == 0) & (pl.col("sknt") > 0)).cast(pl.Int8),
    )
    return feat.join(j, on="MVT_ID_mvt", how="left")


def build(df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return one row per departure (train + ranking) with features and target."""
    if df is None:
        df = load()

    df = df.with_columns(
        t_mvt=_epoch("MVT_TIME_UTC_mvt"),
        t_block=_epoch("BLOCK_TIME_UTC_mvt"),
        t_aobt=_epoch(AOBT_SRC),
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
            d_mvt_aobt=_secs("MVT_TIME_UTC_mvt", AOBT_SRC),
            d_mvt_eobt=_secs("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
            d_mvt_lobt=_secs("MVT_TIME_UTC_mvt", "LOBT_flt"),
            d_mvt_iobt=_secs("MVT_TIME_UTC_mvt", "IOBT_flt"),
            d_mvt_sched=_secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
            d_aobt_eobt=_secs(AOBT_SRC, "EOBT_1_flt"),
            d_eobt_iobt=_secs("EOBT_1_flt", "IOBT_flt"),
            d_aobt_sched=_secs(AOBT_SRC, "SCHED_TIME_UTC_mvt"),
            flight_time_plan=_secs("ARVT_1_flt", "EOBT_1_flt"),
            # E72 (18.09): a second anchor clock, independent of the off-block family. kind-mango credit
            # MVT - ARVT_1 with -3.19 s on the leaderboard (v44 299.85 -> v45 296.13). Kept as the raw
            # difference rather than their route-median residual: the residual needs a reference period,
            # and out-of-fold target encoding already carries the route level.
            # 28.09: absent, not an all-null column, when X_FEATS is off: the accepted tables never had it, and an extra
            # column shifts which features feature_fraction=0.8 samples, so a rebuild would not reproduce the base
            **({"x_plan_res": _secs("MVT_TIME_UTC_mvt", "ARVT_1_flt")} if X_FEATS else {}),
            diverted=(pl.col("ADES_flt") != pl.col("ADES_FILED_flt")).cast(pl.Int8),
            month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
            doy=pl.col("MVT_TIME_UTC_mvt").dt.ordinal_day(),
            stand_prefix=pl.col("STAND_mvt").str.extract(r"^([A-Za-z]+)", 1),
            ap_runway=pl.col("ADEP_mvt") + "_" + pl.col("RUNWAY_mvt").fill_null("NA"),
            ap_stand=pl.col("ADEP_mvt") + "_" + pl.col("STAND_mvt").fill_null("NA"),
            ades_region=pl.col("ADES_mvt").str.slice(0, 2),
            airline_prefix=pl.col("CALLSIGN_flt").str.extract(r"^([A-Z]{3})", 1),
            aobt_minute_exact=(pl.col(AOBT_SRC).dt.second() == 0).cast(pl.Int8),
            # block time lies within 3606 s of LOBT on every 2025 row, so an AOBT_3 further than that from LOBT cannot be the off-block
            aobt_infeasible=((pl.col(AOBT_SRC) - pl.col("LOBT_flt")).dt.total_seconds().abs() > 3606).cast(pl.Int8),
        )
    )
    from geometry import add_geometry

    feat = add_runway_config(feat, df)
    feat = add_neighbor_features(feat, df)
    feat = add_runway_arrivals(feat, df)
    feat = add_day_context(feat, df)
    if X_FEATS:
        feat = add_hour_context(feat)
    feat = add_runway_wind(feat)
    feat = add_geometry(feat)
    return add_weather(feat)


if __name__ == "__main__":
    f = build()
    f.write_parquet(os.environ.get("FEATURES_OUT", "data/features_v9.parquet"))
    print(f.shape)
    print(f.filter(pl.col("split") == "rank").null_count().transpose(include_header=True).filter(pl.col("column_0") > 0))
