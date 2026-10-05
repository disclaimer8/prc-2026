#!/usr/bin/env python3
"""Read-only OPDI source profile. Never opens MVT, features, labels, or predictions."""
import os
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time
import polars as pl

APS = "EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM".split()
GROUND = [f"{action}-{place}" for place in ("parking_position", "taxiway", "runway", "apron", "deicing_pad", "hangar", "threshold") for action in ("entry", "exit")]
FLAGS = GROUND + ["landing", "take-off"]


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(query):
    return query.collect(engine="streaming")


def centers(repo):
    result = []
    for ap in APS:
        path = repo / "analysis/sources/samples/osm_overpass" / f"{ap}.json"
        doc = json.loads(path.read_text())
        points = []
        for element in doc["elements"]:
            if element.get("tags", {}).get("aeroway") not in ("parking_position", "gate"):
                continue
            if "lat" in element and "lon" in element:
                points.append((element["lat"], element["lon"]))
            elif element.get("geometry"):
                g = element["geometry"]
                points.append((statistics.mean(x["lat"] for x in g), statistics.mean(x["lon"] for x in g)))
        if not points:
            raise ValueError(f"No OSM stand centers for {ap}")
        result.append({"airport": ap, "center_lat": statistics.median(p[0] for p in points),
                       "center_lon": statistics.median(p[1] for p in points), "osm_sha256": hash_file(path)})
    return result


def write_report(report, output):
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    temporary.replace(output)


def manifest_profile(manifest):
    types, ms_types = Counter(), Counter()
    totals = Counter()
    periods = {}
    incomplete = {}
    for window, stages in manifest["windows"].items():
        for kind, stage in stages.items():
            if stage.get("status") != "complete":
                incomplete[f"{window}/{kind}"] = stage.get("status")
                continue
            audit = stage["audit"]
            totals[f"{kind}_rows"] += audit["retained_rows"]
            if kind == "events":
                types.update(audit["retained_types"])
            else:
                ms_types.update(audit["retained_types"])
                for key in ("matched_event_ids", "events_without_measurements", "duplicate_event_type_rows", "conflicting_event_type_values"):
                    totals[key] += audit[key]
    for key, value in manifest.items():
        if "flight" in key and key not in ("flight_sources",):
            periods[key] = value
    return {"status": manifest["status"], "complete_windows": manifest["completed_windows"],
            "totals": dict(totals), "event_types": dict(types), "measurement_types": dict(ms_types),
            "incomplete": incomplete, "flight_audits": periods}


def ground_measurements(root, manifest, report, output):
    overall = []
    for window, stages in sorted(manifest["windows"].items()):
        if not all(stages[k]["status"] == "complete" for k in ("events", "measurements")):
            continue
        e = pl.scan_parquet(root / "events" / f"events_{window}.parquet").filter(pl.col("type").is_in(GROUND)).select("opdi_event_id", pl.col("type").alias("event_type"))
        m = pl.scan_parquet(root / "measurements" / f"measurements_{window}.parquet").select("opdi_event_id", pl.col("type").alias("measurement_type"), "value")
        table = collect(m.join(e, on="opdi_event_id", how="inner").group_by("event_type", "measurement_type").agg(
            pl.len().alias("rows"), pl.col("opdi_event_id").n_unique().alias("event_ids"),
            pl.col("value").min().alias("min_value"), pl.col("value").max().alias("max_value"),
            (~pl.col("value").is_finite()).sum().alias("nonfinite_values")))
        overall.extend(dict(row, window=window) for row in table.to_dicts())
        print(json.dumps({"stage": "ground_measurement_join", "window": window, "matched_rows": table["rows"].sum() if table.height else 0}), flush=True)
        report["ground_measurements_by_window"] = overall
        report["ground_measurement_windows_checked"] = report.get("ground_measurement_windows_checked", 0) + 1
        write_report(report, output)


def ranking_profile(repo, root, manifest, report, output):
    geometry = centers(repo)
    report["airport_centers"] = geometry
    flight_frames = []
    for month in ("202601", "202607"):
        frame = collect(pl.scan_parquet(root / "flights" / f"flight_list_{month}.parquet").select(
            pl.col("id").cast(pl.String).alias("opdi_flight_id"), "adep", "ades",
            pl.col("first_seen").cast(pl.String).str.slice(0, 10).alias("first_seen_day"))
            .filter(pl.col("adep").is_in(APS) | pl.col("ades").is_in(APS)))
        report.setdefault("ranking_flight_source_rows", {})[month] = frame.height
        report.setdefault("ranking_flight_days", {})[month] = frame.group_by("first_seen_day").len().sort("first_seen_day").to_dicts()
        flight_frames.append(frame.with_columns(pl.lit(month).alias("month")))
    flights = pl.concat(flight_frames).unique(subset="opdi_flight_id")
    roles = []
    for role, airport, mask in (("departure", "adep", pl.col("adep").is_in(APS)),
                                ("arrival", "ades", pl.col("ades").is_in(APS)),
                                ("arrival_only_new", "ades", pl.col("ades").is_in(APS) & ~pl.col("adep").is_in(APS).fill_null(False))):
        roles.append(flights.filter(mask).select("opdi_flight_id", "month", pl.col(airport).alias("airport"), pl.lit(role).alias("role")))
    scope = pl.concat(roles)
    population = scope.group_by("month", "airport", "role").len().rename({"len": "flight_population"})
    scope = scope.join(pl.DataFrame(geometry).drop("osm_sha256"), on="airport")
    report["ranking_flight_population"] = population.sort("month", "airport", "role").to_dicts()
    windows = []
    for window in manifest["requested_windows"]:
        start, end = (date(int(x[:4]), int(x[4:6]), int(x[6:8])) for x in window.split("_"))
        if (start < date(2026, 2, 1) and end > date(2026, 1, 1)) or (start < date(2026, 8, 1) and end > date(2026, 7, 1)):
            windows.append(window)
    presence, event_counts, infos = [], [], []
    for window in windows:
        if manifest["windows"][window]["events"]["status"] != "complete":
            continue
        path = root / "events" / f"events_{window}.parquet"
        events = pl.scan_parquet(path).select("opdi_flight_id", "type", "latitude", "longitude", "info")
        near = events.join(scope.lazy(), on="opdi_flight_id", how="inner").with_columns(
            (((pl.col("latitude") - pl.col("center_lat")) * 111195.0).pow(2)
             + ((pl.col("longitude") - pl.col("center_lon")) * 111195.0 * (pl.col("center_lat") * math.pi / 180).cos()).pow(2)).alias("distance_squared"))
        near = near.filter(pl.col("distance_squared") <= 15000.0 ** 2)
        # One row per observed flight/role; per-window results are collapsed across windows later.
        one = collect(near.group_by("opdi_flight_id", "month", "airport", "role").agg(
            *[(pl.col("type") == typ).any().alias(typ) for typ in FLAGS],
            (pl.col("info").is_not_null() & ~pl.col("info").is_in(["", "{}", "null"])).any().alias("has_info")))
        presence.append(one)
        counts = collect(near.group_by("month", "airport", "role", "type").agg(pl.len().alias("event_rows")))
        event_counts.append(counts)
        # Small inspectable samples reveal the original payload fields dropped by the old extractor.
        sample = collect(events.filter(pl.col("type").is_in(GROUND) & pl.col("info").is_not_null()
                           & ~pl.col("info").is_in(["", "{}", "null"]))
                         .group_by("type").agg(pl.col("info").first().alias("example_info")))
        infos.extend(dict(row, window=window) for row in sample.to_dicts())
        print(json.dumps({"stage": "ranking_ground", "window": window, "flight_role_rows": one.height}), flush=True)
    combined = pl.concat(presence).group_by("opdi_flight_id", "month", "airport", "role").agg(*[pl.col(c).any() for c in FLAGS + ["has_info"]])
    combined = combined.with_columns(
        (pl.col("entry-runway") & pl.col("exit-runway")).alias("both_runway_events"),
        (pl.col("entry-parking_position") & pl.col("exit-parking_position")).alias("both_parking_events"),
        (pl.col("landing") & pl.col("entry-parking_position")).alias("landing_and_parking"))
    measures = FLAGS + ["has_info", "both_runway_events", "both_parking_events", "landing_and_parking"]
    counts = combined.group_by("month", "airport", "role").agg(pl.len().alias("flight_with_any_near_event"), *[pl.col(c).sum() for c in measures])
    report["ranking_near_airport_flight_presence"] = population.join(counts, on=["month", "airport", "role"], how="left").sort("month", "airport", "role").to_dicts()
    report["ranking_near_airport_event_counts"] = pl.concat(event_counts).group_by("month", "airport", "role", "type").agg(pl.col("event_rows").sum()).sort("month", "airport", "role", "type").to_dicts()
    report["ground_info_examples"] = infos
    report["ranking_definition"] = "Native FLIGHT records from Jan/Jul2026; event within15km of median OSM stand center. This is source coverage, not a competition-row join or validated offblock/event pairing. Arrival_only_new means adep outside the ten airports or null."
    write_report(report, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/a1/prc-2026"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/opdi_full_unlabeled_profile_20261005.json"))
    parser.add_argument("--ranking", action="store_true")
    parser.add_argument("--ground-measurements", action="store_true")
    args = parser.parse_args()
    root = args.repo / "data/opdi_full_v002"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    started = time.monotonic()
    report = {"manifest_sha256": hash_file(manifest_path), "script_sha256": hash_file(Path(__file__)),
              "manifest_profile": manifest_profile(manifest), "reads": ["OPDI manifest", "OPDI FLIGHT/EVENT/MEASUREMENT", "OSM source geometry"],
              "labels_models_cv_read": False, "threads": 2}
    write_report(report, args.output)
    print(json.dumps(report["manifest_profile"], sort_keys=True), flush=True)
    if args.ranking:
        ranking_profile(args.repo, root, manifest, report, args.output)
    if args.ground_measurements:
        ground_measurements(root, manifest, report, args.output)
    report["elapsed_seconds"] = time.monotonic() - started
    report["complete"] = True
    write_report(report, args.output)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": report["elapsed_seconds"], "peak_rss_mib": report["peak_rss_mib"]}), flush=True)


if __name__ == "__main__":
    main()
