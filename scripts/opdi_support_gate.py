#!/usr/bin/env python3
"""Unlabeled OPDI/PRC/ADS-B support gate; no targets, predictions or models."""
import os
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import statistics
import sys
import tempfile
import time

import polars as pl
import pyarrow.parquet as pq

from opdi_unique_match import unique_matches

APS = "EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM".split()
MONTHS = [f"2025{m:02}" for m in range(1, 13)] + ["202601", "202607"]
PLACES = ["parking_position", "taxiway", "runway", "apron", "deicing_pad", "hangar", "threshold"]
GROUND = [f"{a}-{p}" for p in PLACES for a in ("entry", "exit")]
ANCHORS = ["take-off", "first-xing-fl50", "first_seen"]
SHARDS = 16
READ_COLUMNS = {
    "flights": ["id", "flt_id", "icao24", "adep", "ades", "first_seen", "last_seen"],
    "events": ["opdi_event_id", "opdi_flight_id", "type", "event_time", "longitude", "latitude", "altitude", "info"],
    "prc": ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "adsb": ["MVT_ID_mvt", "coverage"],
}
INFO_SCHEMA = pl.Struct({"osm_id": pl.Int64, "osm_aeroway": pl.String, "osm_ref": pl.String,
                         "osm_airport": pl.String, "opdi_time_in_use_s": pl.Float64})


def collect(query):
    return query.collect(engine="streaming")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def projected_frame(path, kind):
    """Arrow projection is explicit before any PRC/ADS-B values are read."""
    if kind not in READ_COLUMNS:
        raise ValueError(f"Forbidden input kind: {kind}")
    source = pq.ParquetFile(path)
    columns = [c for c in READ_COLUMNS[kind] if c in source.schema_arrow.names]
    return pl.from_arrow(source.read(columns=columns))


def canonical_ids(series):
    if series.null_count() or not (series.dtype.is_integer() or series.dtype == pl.String):
        raise ValueError("OPDI identifiers must be non-null native integers or strings")
    result = series.cast(pl.String)
    if (result == "").any():
        raise ValueError("Empty OPDI identifier")
    return result


def prc_ids(series):
    if series.null_count():
        raise ValueError("Null PRC MVT ID")
    if series.dtype.is_float():
        if not (series.is_finite() & (series == series.floor()) & (series.abs() <= 2**53)).all():
            raise ValueError("Native DOUBLE PRC ID is not an exact finite integer within2^53")
        return series.cast(pl.Int64).cast(pl.String)
    return canonical_ids(series)


def seconds(frame, column):
    dtype = frame.schema[column]
    if isinstance(dtype, pl.Datetime):
        return pl.col(column).dt.epoch("s")
    if dtype == pl.String:
        return pl.col(column).str.to_datetime(strict=False, time_zone="UTC").dt.epoch("s")
    raise ValueError(f"Unsupported timestamp schema for {column}: {dtype}")


def normalize_cs(column):
    clean = pl.col(column).str.replace_all(r"\s+", "").str.to_uppercase()
    return pl.when(clean != "").then(clean)


def safe_info(value):
    """Fallback only when vectorized JSON decode encounters malformed source."""
    try:
        data = json.loads(value)
        if (not isinstance(data, dict) or isinstance(data.get("osm_id"), bool)
                or not isinstance(data.get("osm_id"), int) or not 0 < data["osm_id"] < 2**63):
            return None
        result = {name: data.get(name) for name in INFO_SCHEMA.to_schema()}
        if any(result[k] is not None and not isinstance(result[k], str) for k in ("osm_aeroway", "osm_ref", "osm_airport")):
            return None
        duration = result["opdi_time_in_use_s"]
        result["opdi_time_in_use_s"] = float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None
        return result
    except (ValueError, TypeError):
        return None


def prepared_events(frame):
    frame = frame.with_columns(canonical_ids(frame["opdi_event_id"]), canonical_ids(frame["opdi_flight_id"]))
    if "event_s" not in frame.columns:
        frame = frame.with_columns(seconds(frame, "event_time").alias("event_s"))
    # Legacy timestamp[us] and modern string sources must share one shard
    # schema. Downstream pairing/anchors use event_s; originals stay untouched.
    return frame.drop("event_time", strict=False)


def check_global_event_ids(paths):
    if not paths:
        return {"rows": 0, "duplicate_ids": 0}
    result = collect(pl.scan_parquet(paths).group_by("opdi_event_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1))
    if result.height:
        raise ValueError(f"Duplicate EVENT IDs across window/global scope: {result.height}; sample={result['opdi_event_id'].head(3).to_list()}")
    return {"duplicate_ids": 0}


def validate_pairs(events, flights, centers):
    """One exact flight/object/airport pair, with all relevant windows present."""
    events = prepared_events(events)
    context = events.group_by("opdi_flight_id").agg(
        pl.col("event_s").filter(pl.col("type") == "take-off").min().alias("takeoff_s"),
        pl.col("event_s").filter(pl.col("type") == "landing").min().alias("landing_s"))
    events = events.filter(pl.col("type").is_in(GROUND))
    if "altitude" not in events.columns:
        events = events.with_columns(pl.lit(None, dtype=pl.Float64).alias("altitude"))
    try:
        events = events.with_columns(pl.col("info").str.json_decode(INFO_SCHEMA).alias("decoded"),
                                     pl.col("info").str.json_path_match("$.osm_id").alias("osm_id_text"))
    except pl.exceptions.PolarsError:
        events = events.with_columns(pl.col("info").map_elements(safe_info, return_dtype=INFO_SCHEMA).alias("decoded"))
        events = events.with_columns(pl.col("decoded").struct.field("osm_id").cast(pl.String).alias("osm_id_text"))
    events = events.unnest("decoded").with_columns(
        pl.col("type").str.split_exact("-", 1).struct.field("field_0").alias("action"),
        pl.col("type").str.split_exact("-", 1).struct.field("field_1").alias("event_place"))
    info_ok = ((pl.col("osm_id") > 0) & pl.col("osm_id_text").str.contains(r"^[0-9]+$") & pl.col("osm_aeroway").is_in(PLACES)
               & (pl.col("event_place") == pl.col("osm_aeroway")) & pl.col("osm_airport").is_not_null())
    events = events.with_columns(info_ok.fill_null(False).alias("info_ok"))
    bad_flights = events.filter(~pl.col("info_ok")).select("opdi_flight_id").unique().with_columns(pl.lit(True).alias("has_invalid_info"))
    audit = {"ground_rows": events.height, "invalid_info_rows": events.filter(~pl.col("info_ok")).height,
             "outside_target_airport_rows": events.filter(pl.col("info_ok") & ~pl.col("osm_airport").is_in(APS)).height}
    events = events.filter(pl.col("info_ok") & pl.col("osm_airport").is_in(APS))
    events = events.join(centers, on="osm_airport", how="left").with_columns(
        (((pl.col("latitude") - pl.col("center_lat")) * 111195.).pow(2)
         + ((pl.col("longitude") - pl.col("center_lon")) * 111195.
            * (pl.col("center_lat") * math.pi / 180).cos()).pow(2)).alias("distance_squared"))
    events = events.with_columns((pl.col("distance_squared").is_finite()
                                  & (pl.col("distance_squared") <= 15000.**2)).fill_null(False).alias("geo_ok"))
    if "source_window" not in events.columns:
        events = events.with_columns(pl.lit("synthetic").alias("source_window"))
    grouped = events.group_by("opdi_flight_id", "osm_id", "osm_aeroway", "osm_airport").agg(
        (pl.col("action") == "entry").sum().alias("n_entry"),
        (pl.col("action") == "exit").sum().alias("n_exit"),
        pl.col("event_s").filter(pl.col("action") == "entry").min().alias("entry_s"),
        pl.col("event_s").filter(pl.col("action") == "exit").min().alias("exit_s"),
        pl.col("opdi_time_in_use_s").filter(pl.col("action") == "entry").first().alias("entry_duration"),
        pl.col("opdi_time_in_use_s").filter(pl.col("action") == "exit").first().alias("exit_duration"),
        pl.col("geo_ok").all(), pl.col("source_window").n_unique().alias("source_window_count"),
        pl.col("altitude").is_not_null().sum().alias("altitude_present_rows"),
        (pl.col("altitude").is_not_null() & ~pl.col("altitude").is_finite()).sum().alias("altitude_nonfinite_rows"),
        pl.col("altitude").filter(pl.col("altitude").is_finite()).min().alias("altitude_min"),
        pl.col("altitude").filter(pl.col("altitude").is_finite()).max().alias("altitude_max"))
    grouped = grouped.join(flights.select("opdi_flight_id", "adep", "ades", "month"), on="opdi_flight_id", how="left")
    grouped = grouped.join(context, on="opdi_flight_id", how="left")
    grouped = grouped.join(bad_flights, on="opdi_flight_id", how="left").with_columns(pl.col("has_invalid_info").fill_null(False))
    grouped = grouped.with_columns(
        ((pl.col("n_entry") == 0) | (pl.col("n_exit") == 0)).alias("one_sided"),
        ((pl.col("n_entry") > 1) | (pl.col("n_exit") > 1)).alias("repeated_visit"),
        ((pl.col("exit_s") - pl.col("entry_s")) > 0).fill_null(False).alias("positive_time"),
        (pl.col("osm_airport").is_in(APS) & ((pl.col("osm_airport") == pl.col("adep"))
         | (pl.col("osm_airport") == pl.col("ades")))).fill_null(False).alias("endpoint_ok"))
    grouped = grouped.with_columns(
        (pl.col("entry_duration").is_finite() & pl.col("exit_duration").is_finite()
         & (pl.col("entry_duration") == pl.col("exit_duration"))
         & (pl.col("entry_duration") == pl.col("exit_s") - pl.col("entry_s"))).fill_null(False).alias("duration_ok"))
    grouped = grouped.with_columns((~pl.col("has_invalid_info") & ~pl.col("one_sided") & ~pl.col("repeated_visit") & pl.col("positive_time")
                                    & pl.col("geo_ok") & pl.col("endpoint_ok") & pl.col("duration_ok")).alias("valid"))
    grouped = grouped.with_columns(
        (pl.col("exit_s") <= pl.col("takeoff_s")).fill_null(False).alias("ends_before_takeoff"),
        (pl.col("entry_s") >= pl.col("takeoff_s")).fill_null(False).alias("starts_after_takeoff"),
        ((pl.col("entry_s") < pl.col("takeoff_s")) & (pl.col("exit_s") > pl.col("takeoff_s"))).fill_null(False).alias("straddles_takeoff"),
        (pl.col("entry_s") >= pl.col("landing_s")).fill_null(False).alias("starts_after_landing"),
        (pl.col("exit_s") < pl.col("landing_s")).fill_null(False).alias("ends_before_landing"))
    return grouped, audit


def geometry(repo):
    rows, provenance = [], []
    for airport in APS:
        path = repo / "analysis/sources/samples/osm_overpass" / f"{airport}.json"
        points = []
        for element in json.loads(path.read_text())["elements"]:
            if element.get("tags", {}).get("aeroway") not in ("parking_position", "gate"):
                continue
            if "lat" in element and "lon" in element:
                points.append((element["lat"], element["lon"]))
            elif element.get("geometry"):
                points.append((statistics.mean(p["lat"] for p in element["geometry"]), statistics.mean(p["lon"] for p in element["geometry"])))
        if not points:
            raise ValueError(f"No OSM stand centers for {airport}")
        rows.append({"osm_airport": airport, "center_lat": statistics.median(p[0] for p in points), "center_lon": statistics.median(p[1] for p in points)})
        provenance.append({"path": str(path), "sha256": digest(path)})
    return pl.DataFrame(rows), provenance


def shard_write(frame, root, kind, name, key):
    if not frame.height:
        return
    frame = frame.with_columns((pl.col(key).hash(seed=42) % SHARDS).alias("_shard"))
    for (number,), part in frame.partition_by("_shard", as_dict=True).items():
        folder = root / kind / str(number)
        folder.mkdir(parents=True, exist_ok=True)
        part.drop("_shard").write_parquet(folder / f"{name}.parquet", compression="zstd")


def check_opdi(path, expected):
    if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
        raise ValueError(f"OPDI manifest/source hash mismatch: {path}")


def source_populations(flights):
    frames = []
    for role, airport, condition in (
        ("departure", "adep", pl.col("adep").is_in(APS)),
        ("arrival", "ades", pl.col("ades").is_in(APS)),
        ("arrival_only_new", "ades", pl.col("ades").is_in(APS) & ~pl.col("adep").is_in(APS).fill_null(False))):
        frames.append(flights.filter(condition).group_by("month", airport).len().rename({airport: "airport", "len": "source_flights"}).with_columns(pl.lit(role).alias("role")))
    return pl.concat(frames, how="vertical_relaxed")


def pair_populations(pairs):
    parts = []
    for role, mask in (("departure", pl.col("osm_airport") == pl.col("adep")),
                       ("arrival", pl.col("osm_airport") == pl.col("ades")),
                       ("arrival_only_new", (pl.col("osm_airport") == pl.col("ades")) & ~pl.col("adep").is_in(APS).fill_null(False))):
        group = pairs.filter(mask).group_by("month", "osm_airport", "osm_aeroway").agg(
            pl.len().alias("source_object_groups"), pl.col("valid").sum().alias("valid_polygon_pairs"),
            pl.col("opdi_flight_id").filter(pl.col("valid")).n_unique().alias("flights_with_valid_pairs"),
            pl.col("one_sided").sum(), pl.col("repeated_visit").sum(),
            (~pl.col("geo_ok")).sum().alias("geography_reject"), (~pl.col("duration_ok")).sum().alias("duration_reject"),
            (pl.col("valid") & pl.col("ends_before_takeoff")).sum().alias("valid_ends_before_takeoff"),
            (pl.col("valid") & pl.col("straddles_takeoff")).sum().alias("valid_straddles_takeoff"),
            (pl.col("valid") & pl.col("starts_after_takeoff")).sum().alias("valid_starts_after_takeoff"),
            (pl.col("valid") & pl.col("starts_after_landing")).sum().alias("valid_starts_after_landing"),
            (pl.col("valid") & pl.col("ends_before_landing")).sum().alias("valid_ends_before_landing"),
            pl.col("altitude_present_rows").sum(), pl.col("altitude_nonfinite_rows").sum(),
            pl.col("altitude_min").min(), pl.col("altitude_max").max())
        parts.append(group.rename({"osm_airport": "airport", "osm_aeroway": "object_type"}).with_columns(pl.lit(role).alias("role")))
    return pl.concat(parts, how="vertical_relaxed")


def extract_anchors(events, flights):
    frame = events.group_by("opdi_flight_id").agg(
        *[pl.col("event_s").filter(pl.col("type") == typ).min().alias(typ) for typ in ANCHORS])
    frame = frame.with_columns(pl.coalesce(ANCHORS).alias("anchor_s"),
        pl.when(pl.col("take-off").is_not_null()).then(pl.lit("take-off"))
        .when(pl.col("first-xing-fl50").is_not_null()).then(pl.lit("first-xing-fl50"))
        .otherwise(pl.lit("first_seen")).alias("anchor_type"))
    return frame.filter(pl.col("anchor_s").is_not_null()).join(
        flights.select("opdi_flight_id", "cs", pl.col("adep").alias("airport")), on="opdi_flight_id").filter(pl.col("airport").is_in(APS)).select(
        "opdi_flight_id", "cs", "airport", "anchor_s", "anchor_type")


def flight_pair_support(pairs):
    own = pl.col("osm_airport") == pl.col("adep")
    return pairs.group_by("opdi_flight_id").agg(
        *[(pl.col("valid") & own & (pl.col("osm_aeroway") == p)).sum().alias(f"valid_{p}_pairs") for p in PLACES],
        (own & pl.col("one_sided")).any().alias("has_one_sided_object"),
        (own & pl.col("repeated_visit")).any().alias("has_repeated_object"),
        (own & ~pl.col("duration_ok")).any().alias("has_duration_reject"),
        (own & ~pl.col("geo_ok")).any().alias("has_geography_reject"),
        (own & pl.col("valid") & (pl.col("osm_aeroway") == "runway") & pl.col("ends_before_takeoff")).sum().alias("runway_pairs_before_takeoff"))


def projected_provenance(path, frame, temporary):
    # Never hash the full label-bearing PRC file; hash only the projected copy.
    frame.write_parquet(temporary, compression="zstd")
    result = {"path": str(path), "file_bytes": path.stat().st_size, "file_mtime_ns": path.stat().st_mtime_ns,
              "read_columns": frame.columns, "projected_rows": frame.height, "projected_sha256": digest(temporary)}
    temporary.unlink()
    return result


def read_prc(repo, months, scratch, report):
    folder = repo / "data/prc-2026-datasets"
    paths = sorted(folder.glob("training_*.parquet")) if any(m.startswith("2025") for m in months) else []
    if any(m.startswith("2026") for m in months):
        paths.append(folder / "ranking.parquet")
    frames = []
    report["prc_projections"] = []
    for i, path in enumerate(paths):
        frame = projected_frame(path, "prc")
        report["prc_projections"].append(projected_provenance(path, frame, scratch / f"prc_projection_{i}.parquet"))
        frame = frame.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(APS))
        frame = frame.with_columns(prc_ids(frame["MVT_ID_mvt"]).alias("mvt_id"), seconds(frame, "MVT_TIME_UTC_mvt").alias("mvt_s"),
                                   pl.coalesce(normalize_cs("CALLSIGN_flt"), normalize_cs("FLIGHT_mvt")).alias("cs"))
        if frame["mvt_s"].null_count():
            raise ValueError("Unparseable PRC public MVT timestamp")
        frame = frame.with_columns(pl.from_epoch("mvt_s", time_unit="s").dt.strftime("%Y%m").alias("month"),
                                   pl.from_epoch("mvt_s", time_unit="s").dt.strftime("%Y-%m-%d").alias("day"))
        frames.append(frame.filter(pl.col("month").is_in(months)).select("mvt_id", "cs", pl.col("ADEP_mvt").alias("airport"), "mvt_s", "month", "day"))
    if not frames:
        raise ValueError("No allowed PRC source files")
    return pl.concat(frames, how="vertical_relaxed")


def read_coverage(repo, scratch, report):
    frames = []
    report["adsb_projections"] = []
    for i, path in enumerate(sorted((repo / "data/adsblol_kt1").glob("offblock2_*.parquet"))):
        frame = projected_frame(path, "adsb")
        report["adsb_projections"].append(projected_provenance(path, frame, scratch / f"adsb_projection_{i}.parquet"))
        frames.append(frame.select(prc_ids(frame["MVT_ID_mvt"]).alias("mvt_id"), pl.col("coverage").cast(pl.Int64)))
    if not frames:
        return pl.DataFrame(schema={"mvt_id": pl.String, "coverage": pl.Int64})
    coverage = pl.concat(frames, how="vertical_relaxed")
    report["adsb_duplicate_rows"] = coverage.height - coverage["mvt_id"].n_unique()
    return coverage.group_by("mvt_id").agg(pl.col("coverage").max())


def gap_flags(prc, available_days, missing_windows):
    missing_days = set()
    for window in missing_windows:
        first, last = [datetime.strptime(x, "%Y%m%d").date() for x in window.split("_")]
        while first < last:
            missing_days.add(first.isoformat())
            first += timedelta(days=1)
    days = available_days.unique().with_columns(pl.lit(True).alias("flight_day_available"))
    return prc.join(days, on=["airport", "day"], how="left").with_columns(
        (~pl.col("flight_day_available").fill_null(False)).alias("flight_source_day_missing"),
        pl.col("day").is_in(sorted(missing_days)).alias("mvt_day_in_missing_event_window"))


def summarize_mvt(frame):
    frame = frame.with_columns(pl.col("opdi_flight_id").is_not_null().alias("unique_opdi_match"),
                              pl.col("coverage").is_not_null().alias("adsb_known"))
    good_gap = ~pl.col("flight_source_day_missing") & ~pl.col("mvt_day_in_missing_event_window")
    aggregate = [pl.len().alias("prc_departures"), pl.col("unique_opdi_match").sum().alias("unique_mvt_matches"),
                 pl.col("flight_source_day_missing").sum(), pl.col("mvt_day_in_missing_event_window").sum(),
                 (~pl.col("adsb_known")).sum().alias("adsb_record_missing")]
    for place in PLACES:
        supported = (pl.col(f"valid_{place}_pairs").fill_null(0) > 0) & pl.col("unique_opdi_match")
        aggregate += [supported.sum().alias(f"valid_{place}_mvt"),
                      (supported & good_gap).sum().alias(f"valid_{place}_mvt_no_source_gap"),
                      (supported & good_gap & pl.col("adsb_known") & (pl.col("coverage") < 2)).sum().alias(f"valid_{place}_adsb_lt2_known"),
                      (supported & good_gap & ~pl.col("adsb_known")).sum().alias(f"valid_{place}_adsb_unknown")]
    aggregate += [(pl.col("runway_pairs_before_takeoff").fill_null(0) > 0).sum().alias("runway_mvt_before_public_takeoff")]
    return frame.group_by("airport", "month").agg(aggregate).sort("airport", "month")


def run_gate(repo, output, *, months=MONTHS):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Use a new output directory; existing audit reports are immutable")
    root = repo / "data/opdi_full_v002"
    manifest_path = root / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    started = time.monotonic()
    report = {"status": "running", "months": months, "manifest_sha256": manifest_hash, "script_sha256": digest(__file__),
              "matching_code_sha256": digest(Path(__file__).with_name("opdi_unique_match.py")),
              "labels_predictions_models_read": False, "allowed_columns": READ_COLUMNS,
              "matching_rule": "normalized uppercase whitespace-free callsign+ADEP; earliest take-off else first-xing-fl50 else first_seen; inclusive anchor-MVT[-900,+1800]seconds; prune both candidate graph degrees globally",
              "pair_meaning": "Coherent observed polygon entry/exit, not proof of ground contact or causal delay",
              "altitude_units": "Feet in official v002 generator; absolute cleaned barometric height, not AGL; no audit threshold applied",
              "source_semantics": {"commit": "8bf973d0997d3ef202e44e3ecc6426c714de0086",
                                   "path": "python/v2.0.0/04_opdi_flight_events_etl.py",
                                   "filter": "flight_level<=20 (absolute2000ft), H3res12 airport polygons; no on_ground/speed/phase/AGL check",
                                   "interval": "min/max matched timestamps, trace split only gap>300s; not interpolated crossing times",
                                   "caveat": "Pinned historical v002 generator; current publisher shard-by-shard source lineage not independently established"},
              "incremental_meaning": "Additional sensor support where known ADS-B coverage<2; not residual information or model gain",
              "duplicate_event_scope": "All used ground,anchor,landing EVENT IDs globally; duplicate IDs reject audit",
              "shards": SHARDS, "threads": 2, "opdi_inputs": []}
    centers, report["geometry_inputs"] = geometry(repo)
    populations, pair_counts, support_frames, anchor_frames, day_frames = [], [], [], [], []
    pair_audit, flight_audit = Counter(), Counter()
    requested = set(manifest["requested_windows"])
    if set(manifest["windows"]) - requested:
        raise ValueError("Manifest contains unrequested EVENT windows")
    if set(months) - set(manifest["flight_sources"]):
        raise ValueError("Manifest lacks a requested monthly FLIGHT source")
    missing_windows = sorted(w for w in requested if manifest["windows"].get(w, {}).get("events", {}).get("status") != "complete")
    report["missing_event_windows"] = missing_windows
    try:
        with tempfile.TemporaryDirectory(prefix=".support-shards-", dir=output) as temporary:
            scratch = Path(temporary)
            for month, record in sorted(manifest["flight_sources"].items()):
                path = root / "flights" / f"flight_list_{month}.parquet"
                check_opdi(path, record)
                report["opdi_inputs"].append({"path": str(path), "sha256": record["sha256"]})
                frame = projected_frame(path, "flights").filter(pl.col("adep").is_in(APS) | pl.col("ades").is_in(APS))
                frame = frame.with_columns(canonical_ids(frame["id"]).alias("opdi_flight_id"), normalize_cs("flt_id").alias("cs"),
                                           pl.col("first_seen").cast(pl.String).str.slice(0, 10).alias("day"),
                                           pl.col("first_seen").cast(pl.String).alias("first_seen_source"),
                                           pl.col("last_seen").cast(pl.String).alias("last_seen_source"),
                                           pl.col("flt_id").alias("callsign_source"))
                frame = frame.with_columns(pl.col("day").str.replace_all("-", "").str.slice(0, 6).alias("month"))
                frame = frame.select("opdi_flight_id", "cs", "icao24", "adep", "ades", "day", "month",
                                     "first_seen_source", "last_seen_source", "callsign_source")
                day_frames.append(frame.filter(pl.col("adep").is_in(APS)).select(pl.col("adep").alias("airport"), "day").unique())
                shard_write(frame, scratch, "flights", month, "opdi_flight_id")
            for window, stages in sorted(manifest["windows"].items()):
                if stages["events"]["status"] != "complete":
                    continue
                path = root / "events" / f"events_{window}.parquet"
                check_opdi(path, stages["events"]["output"])
                report["opdi_inputs"].append({"path": str(path), "sha256": stages["events"]["output"]["sha256"]})
                # Scan projection/filter first; never collect a whole unrestricted EVENT table.
                names = pq.ParquetFile(path).schema_arrow.names
                frame = collect(pl.scan_parquet(path).select([c for c in READ_COLUMNS["events"] if c in names]).filter(pl.col("type").is_in(GROUND + ANCHORS + ["landing"])))
                frame = prepared_events(frame).with_columns(pl.lit(window).alias("source_window"))
                shard_write(frame.select("opdi_event_id", "opdi_flight_id"), scratch, "event_ids", window, "opdi_event_id")
                shard_write(frame, scratch, "events", window, "opdi_flight_id")
                print(json.dumps({"stage": "project_sources", "window": window, "used_rows": frame.height}), flush=True)
            del frame
            for shard in range(SHARDS):
                check_global_event_ids(sorted((scratch / "event_ids" / str(shard)).glob("*.parquet")))
            for shard in range(SHARDS):
                paths = sorted((scratch / "flights" / str(shard)).glob("*.parquet"))
                if not paths:
                    continue
                flights = collect(pl.scan_parquet(paths)).unique()
                conflict = flights.group_by("opdi_flight_id").len().filter(pl.col("len") > 1)
                flight_audit["conflicting_native_ids_excluded"] += conflict.height
                flights = flights.join(conflict.select("opdi_flight_id"), on="opdi_flight_id", how="anti")
                flight_audit["unambiguous_source_flights"] += flights.height
                populations.append(source_populations(flights))
                paths = sorted((scratch / "events" / str(shard)).glob("*.parquet"))
                if not paths:
                    continue
                events = collect(pl.scan_parquet(paths))
                anchors = extract_anchors(events, flights)
                anchor_frames.append(anchors)
                pairs, audit = validate_pairs(events, flights, centers)
                pair_audit.update(audit)
                pair_audit["source_object_groups"] += pairs.height
                pair_audit["valid_polygon_pairs"] += pairs["valid"].sum()
                pair_audit["cross_window_object_groups"] += pairs.filter(pl.col("source_window_count") > 1).height
                pair_counts.append(pair_populations(pairs))
                support_frames.append(flight_pair_support(pairs))
                print(json.dumps({"stage": "validate_global_pairs", "shard": shard, "pairs": pairs.height, "valid_polygon_pairs": int(pairs['valid'].sum())}), flush=True)
            del events, flights, pairs
            report["pair_audit"], report["flight_audit"] = dict(pair_audit), dict(flight_audit)
            population = pl.concat(populations).group_by("month", "airport", "role").agg(pl.col("source_flights").sum()).filter(pl.col("month").is_in(months))
            report["source_support"] = population.sort("airport", "month", "role").to_dicts()
            counts = pl.concat(pair_counts)
            group = ["month", "airport", "object_type", "role"]
            counts = counts.group_by(group).agg(*[pl.col(c).sum() for c in counts.columns if c not in group + ["altitude_min", "altitude_max"]],
                                               pl.col("altitude_min").min(), pl.col("altitude_max").max()).filter(pl.col("month").is_in(months))
            report["validated_pair_support"] = counts.sort(group).to_dicts()
            prc = read_prc(repo, months, scratch, report)
            coverage = read_coverage(repo, scratch, report)
            anchors = pl.concat(anchor_frames, how="vertical_relaxed")
            matches, report["matching"] = unique_matches(prc.select("mvt_id", "cs", "airport", "mvt_s"), anchors)
            del anchor_frames, anchors
            joined = prc.join(matches, on="mvt_id", how="left").join(pl.concat(support_frames, how="vertical_relaxed"), on="opdi_flight_id", how="left").join(coverage, on="mvt_id", how="left")
            joined = gap_flags(joined, pl.concat(day_frames), missing_windows)
            report["mvt_support"] = summarize_mvt(joined).to_dicts()
            report["matched_anchor_support"] = joined.filter(pl.col("opdi_flight_id").is_not_null()).group_by("airport", "month", "anchor_type").agg(
                pl.len().alias("unique_mvt_matches"), (pl.col("valid_runway_pairs").fill_null(0) > 0).sum().alias("with_valid_runway_pair"),
                ((pl.col("valid_runway_pairs").fill_null(0) > 0) & pl.col("coverage").is_not_null() & (pl.col("coverage") < 2)).sum().alias("known_adsb_lt2_sensor_support")).sort("airport", "month", "anchor_type").to_dicts()
            support_path = output / "matched_support.parquet"
            joined.write_parquet(support_path, compression="zstd")
            report["matched_support_file"] = {"path": str(support_path), "sha256": digest(support_path), "rows": joined.height,
                                               "columns": joined.columns}
            stability = []
            for airport in APS:
                rows = [x for x in report["mvt_support"] if x["airport"] == airport]
                present = [x["month"] for x in rows if x["valid_runway_mvt_no_source_gap"] > 0]
                stability.append({"airport": airport, "months_with_unique_mvt_and_valid_runway_pair": present,
                                  "all_requested_months_have_some_support": set(months) <= set(present),
                                  "interpretation": "Presence only; neither stable capture probability nor validated ground phase"})
            report["airport_month_presence"] = stability
        if digest(manifest_path) != manifest_hash:
            raise ValueError("OPDI manifest changed during audit")
        report["status"] = "complete_unlabeled_audit"
    except Exception as error:
        report.update(status="invalid_or_incomplete", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "report.json", report)
        raise
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/a1/prc-2026"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_gate(args.repo, args.output)
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "peak_rss_mib", "matching")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
