"""Shared loader for traffic analysis. Read-only on data/prc-2026-datasets."""
import glob
import polars as pl

import os
DATA = os.path.expanduser("~/prc-2026/data/prc-2026-datasets")
OUT = os.path.expanduser("~/prc-2026/analysis/traffic")
HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
      "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
      "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
      "LTFM": "Europe/Istanbul"}
CITY = {"EDDF": "Frankfurt", "EDDM": "Munich", "EGLL": "London Heathrow", "EHAM": "Amsterdam",
        "LEBL": "Barcelona", "LEMD": "Madrid", "LFPG": "Paris CDG", "LIRF": "Rome FCO",
        "LSZH": "Zurich", "LTFM": "Istanbul"}

COLS = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "CALLSIGN_flt", "ADEP_flt",
        "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
        "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "FLIGHT_RULE_mvt"]


def _prep(lf: pl.LazyFrame, src: str) -> pl.LazyFrame:
    lf = lf.select(COLS).with_columns(
        hub=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
        other=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADES_mvt")).otherwise(pl.col("ADEP_mvt")),
        # NM (flight-plan) counterpart; airport-reported codes contain non-airport/final-leg codes (see 07_unknown_codes)
        other_nm=pl.coalesce(pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADES_flt")).otherwise(pl.col("ADEP_flt")),
                             pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADES_mvt")).otherwise(pl.col("ADEP_mvt"))),
        src=pl.lit(src),
    )
    # local time per hub
    local = pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Berlin").dt.replace_time_zone(None)
    expr = None
    for h, tz in TZ.items():
        e = pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz).dt.replace_time_zone(None)
        expr = pl.when(pl.col("hub") == h).then(e) if expr is None else expr.when(pl.col("hub") == h).then(e)
    expr = expr.otherwise(local)
    return lf.with_columns(t_local=expr).with_columns(
        date_utc=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        date_local=pl.col("t_local").dt.date(),
        month=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    )


def load_training() -> pl.LazyFrame:
    files = sorted(glob.glob(f"{DATA}/training_2025-*.parquet"))
    return _prep(pl.scan_parquet(files), "training")


def load_ranking() -> pl.LazyFrame:
    return _prep(pl.scan_parquet(f"{DATA}/ranking.parquet"), "ranking")


def load_all() -> pl.LazyFrame:
    return pl.concat([load_training(), load_ranking()])


def airports() -> pl.DataFrame:
    """OurAirports airports.csv (public domain), keyed by ICAO ident."""
    a = pl.read_csv(OUT + "/ext/airports.csv", infer_schema_length=0)
    a = a.with_columns(pl.col("latitude_deg").cast(pl.Float64), pl.col("longitude_deg").cast(pl.Float64))
    # prefer ident; also map icao_code and gps_code when they differ
    keys = []
    for col in ["ident", "icao_code", "gps_code"]:
        keys.append(a.filter(pl.col(col).is_not_null() & (pl.col(col).str.len_chars() == 4))
                    .select(icao=pl.col(col), name="name", municipality="municipality", iso_country="iso_country",
                            continent="continent", lat="latitude_deg", lon="longitude_deg", atype="type"))
    k = pl.concat(keys).unique(subset="icao", keep="first", maintain_order=True)
    return k


def haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1; dl = np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


# ICAO type groupings (documented in REPORT)
FAM = {
    "A320neo family": ["A19N", "A20N", "A21N"],
    "A320ceo family": ["A318", "A319", "A320", "A321"],
    "737 MAX": ["B37M", "B38M", "B39M", "B3XM"],
    "737 NG/classic": ["B733", "B734", "B735", "B736", "B737", "B738", "B739"],
    "A220": ["BCS1", "BCS3"],
    "E-Jet": ["E170", "E175", "E190", "E195", "E290", "E295", "E75L", "E75S"],
    "CRJ": ["CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX"],
    "Turboprop": ["AT43", "AT45", "AT46", "AT72", "AT73", "AT75", "AT76", "DH8A", "DH8B", "DH8C", "DH8D", "SF34", "SB20", "D328", "J328"],
    "Widebody": ["A306", "A30B", "A310", "A332", "A333", "A337", "A338", "A339", "A342", "A343", "A345", "A346", "A359", "A35K", "A388",
                 "B742", "B744", "B748", "B74S", "B762", "B763", "B764", "B772", "B77L", "B77W", "B778", "B779", "B788", "B789", "B78X",
                 "MD11", "IL96", "A3ST", "A400"],
}
TYPE2FAM = {t: f for f, ts in FAM.items() for t in ts}
