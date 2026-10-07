"""E194: label-free, sharded actual ADS-B visit sequences. No model training.

Preserve E64 same-day unique callsign/airport identity and 12h/+600s visit
boundaries. Serialize observed points <=MVT, never interpolate or create labels.
"""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

PROTOCOL = "E194-observed-visit-sequences-K64-sameday-2stream-v1"
DEP_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "CALLSIGN_flt",
            "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"]
POINT_COLS = ["icao", "airport", "t", "lat", "lon", "alt", "gs", "track",
              "on_ground", "callsign", "stale"]
NUM_NAMES = ["mvt_minus_t_s", "gap_from_previous_raw_s", "east_km", "north_km",
             "alt_ft", "speed_kt", "heading_sin", "heading_cos", "on_ground",
             "stale", "callsign_matches_query", "duplicate_timestamp"]
LOOKBACK, FUTURE_BOUNDARY, VISIT_GAP, K = 43200.0, 600.0, 1800.0, 64
MONTHS = [f"2025-{m:02}" for m in range(1, 13)] + ["2026-01", "2026-07"]
DAYS = [f"{m}-{d:02}" for m in MONTHS for d in range(
    1, calendar.monthrange(int(m[:4]), int(m[5:]))[1] + 1)]
AIRPORTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861),
            "EGLL": (51.4706, -0.4619), "EHAM": (52.3086, 4.7639),
            "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
            "LFPG": (49.0128, 2.5500), "LIRF": (41.8045, 12.2508),
            "LSZH": (47.4647, 8.5492), "LTFM": (41.2615, 28.7419)}
STATUSES = ["sensor_missing", "identity_absent", "identity_ambiguous",
            "too_few_window_points", "identified"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def assert_registration(root, registration):
    if not re.fullmatch(r"[0-9a-f]{40}", registration):
        raise ValueError("full registration commit SHA required")
    for name in ["scripts/e194_trajectory_sequence.py", "analysis/e194_trajectory_protocol_20261007.md"]:
        raw = subprocess.check_output(["git", "show", f"{registration}:{name}"], cwd=root)
        if hashlib.sha256(raw).hexdigest() != sha(root / name):
            raise ValueError(f"registered source drift: {name}")


def selected_days(index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard")
    return DAYS[index::count]


def ids_checked(frame):
    a = frame["MVT_ID_mvt"].to_numpy()
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError("departure IDs must be finite integers")
    ids = a.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate departure ID")
    return ids


def pack(root, output, registration, shard_index, shard_count):
    root, output = Path(root).resolve(), Path(output).resolve()
    assert_registration(root, registration)
    if output.exists():
        raise FileExistsError("fresh immutable bundle required")
    days = selected_days(shard_index, shard_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    for d in ["surface", "departures", "scripts", "analysis"]:
        (stage / d).mkdir()
    rawfiles = sorted((root / "data/prc-2026-datasets").glob("training_*.parquet"))
    if len(rawfiles) != 12:
        raise ValueError("all 12 raw training files required")
    rawfiles.append(root / "data/prc-2026-datasets/ranking.parquet")
    raw_hashes = {str(p.relative_to(root)): sha(p) for p in rawfiles}
    # Projection never reads TAXITIME/BLOCK. Actual UTC date avoids raw monthly
    # file boundary errors encountered in earlier extraction experiments.
    deps = pl.scan_parquet(rawfiles).select(DEP_COLS).filter(pl.col("PHASE_mvt") == "DEP").collect()
    ids_checked(deps)
    deps = deps.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").alias("_day"))
    if deps["_day"].null_count() or deps.filter(~pl.col("_day").is_in(DAYS)).height:
        raise ValueError("unexpected departure date outside protocol")
    point_meta, total_rows = {}, 0
    for day in days:
        dep = deps.filter(pl.col("_day") == day).drop("_day").sort("MVT_ID_mvt")
        ids_checked(dep)
        dep.write_parquet(stage / "departures" / f"{day}.parquet")
        total_rows += dep.height
        p = root / "data/adsblol" / f"surface_{day}.parquet"
        info = {"exists": p.exists(), "departure_rows": dep.height}
        if p.exists():
            schema = pl.scan_parquet(p).collect_schema()
            required = set(POINT_COLS) - {"stale"}
            if not required.issubset(schema):
                raise ValueError(f"point schema missing columns: {day}")
            info.update(sha256=sha(p), stale_column=("stale" in schema), bytes=p.stat().st_size)
            target = stage / "surface" / f"{day}.parquet"
            shutil.copy2(p, target)
            if sha(target) != info["sha256"] or sha(p) != info["sha256"]:
                raise ValueError("source/copy drift")
        elif day != "2025-12-31":
            raise FileNotFoundError(f"unexpected missing sensor day: {day}")
        point_meta[day] = info
    for name in ["scripts/e194_trajectory_sequence.py", "analysis/e194_trajectory_protocol_20261007.md"]:
        shutil.copy2(root / name, stage / name)
    assert_registration(root, registration)
    if any(sha(root / p) != h for p, h in raw_hashes.items()):
        raise ValueError("raw movement inputs changed while packing")
    files = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    save(stage / "manifest.json", {"protocol": PROTOCOL, "registration_commit": registration,
         "shard_index": shard_index, "shard_count": shard_count, "days": days,
         "departure_rows": total_rows, "raw_input_sha256": raw_hashes,
         "point_inputs": point_meta, "files": files, "departure_columns": DEP_COLS,
         "point_columns": POINT_COLS, "numeric_columns": NUM_NAMES,
         "streams": ["current_visit", "previous_visit"], "tokens_per_stream": K,
         "label_free": True, "accepted_for_training": False,
         "airport_reference_coordinates": AIRPORTS})
    verify_bundle(stage, registration)
    stage.rename(output)
    print(json.dumps({"bundle": str(output), "days": len(days), "rows": total_rows,
                      "manifest_sha256": sha(output / "manifest.json")}), flush=True)


def verify_bundle(bundle, registration):
    bundle = Path(bundle)
    m = read(bundle / "manifest.json")
    if (m["protocol"] != PROTOCOL or m["registration_commit"] != registration
            or m["days"] != selected_days(m["shard_index"], m["shard_count"])
            or m["departure_columns"] != DEP_COLS or m["point_columns"] != POINT_COLS
            or m["numeric_columns"] != NUM_NAMES or m["tokens_per_stream"] != K):
        raise ValueError("bundle protocol mismatch")
    expected = {f"departures/{d}.parquet" for d in m["days"]}
    expected |= {f"surface/{d}.parquet" for d in m["days"] if m["point_inputs"][d]["exists"]}
    expected |= {"scripts/e194_trajectory_sequence.py", "analysis/e194_trajectory_protocol_20261007.md"}
    if set(m["files"]) != expected:
        raise ValueError("bundle file inventory mismatch")
    for name, h in m["files"].items():
        if sha(bundle / name) != h:
            raise ValueError(f"bundle hash changed: {name}")
    if sha(Path(__file__)) != m["files"]["scripts/e194_trajectory_sequence.py"]:
        raise ValueError("executing source differs from bundled source")
    return m


def sample_time_indices(t):
    """Actual point indices near a uniform time grid; no interpolation."""
    if len(t) <= K:
        return np.arange(len(t), dtype=np.int64)
    grid = np.linspace(t[0], t[-1], K)
    selected = np.searchsorted(t, grid).clip(0, len(t)-1)
    selected[0], selected[-1] = 0, len(t)-1
    return np.unique(selected).astype(np.int64)


def visit_ranges(t, takeoff):
    """E64 boundary logic, with returned token slices restricted to t<=MVT."""
    lo = int(np.searchsorted(t, takeoff-LOOKBACK, side="left"))
    hi = int(np.searchsorted(t, takeoff+FUTURE_BOUNDARY, side="right"))
    if hi-lo < 2:
        return None
    gaps = np.flatnonzero(np.diff(t[lo:hi]) > VISIT_GAP) + lo
    current_start = int(gaps[-1])+1 if len(gaps) else lo
    # Past cutoff cannot add a point outside an E64 visit.
    past_end = int(np.searchsorted(t, takeoff, side="right"))
    current = (current_start, max(current_start, min(hi, past_end)))
    previous = (-1, -1)
    if len(gaps):
        start = int(gaps[-2])+1 if len(gaps) > 1 else lo
        end = int(gaps[-1])+1
        previous = (start, max(start, min(end, past_end)))
    return current, previous


def prepare_points(points):
    if "stale" not in points.columns:
        points = points.with_columns(pl.lit(None, dtype=pl.Boolean).alias("stale"))
    points = points.with_row_index("source_row")
    # Invalid timestamps have no defensible placement; they cannot establish
    # an identity or join a visit. Original row indices retain audit provenance.
    invalid = points.filter(~pl.col("t").is_finite() | pl.col("t").is_null()).height
    if invalid:
        raise ValueError(f"nonfinite point timestamps: {invalid}")
    points = points.sort(["icao", "airport", "t", "source_row"])
    cmap = {}
    for cs, ap, ic in points.filter(pl.col("callsign").is_not_null()).select(
            "callsign", "airport", "icao").unique().iter_rows():
        cmap.setdefault((cs, ap), set()).add(ic)
    # Global sorted row ranges provide an unbinned route to the complete raw
    # sequence using the immutable daily parquet, not just 64 selected points.
    groups = {}
    start = 0
    for key, group in points.group_by(["icao", "airport"], maintain_order=True):
        groups[key] = (start, group)
        start += group.height
    return points, cmap, groups


def encode_day(points, deps, day):
    ids = ids_checked(deps)
    n = len(ids)
    values = np.zeros((n, 2, K, len(NUM_NAMES)), dtype=np.float32)
    known = np.zeros_like(values, dtype=np.uint8)
    present = np.zeros((n, 2, K), dtype=np.uint8)
    source_rows = np.full((n, 2, K), -1, dtype=np.int32)
    point_times = np.full((n, 2, K), np.nan, dtype=np.float64)
    raw_ranges = np.full((n, 2, 2), -1, dtype=np.int64)
    counts = np.zeros((n, 2), dtype=np.int32)
    status = np.zeros(n, dtype=np.uint8)
    identities = np.full(n, "", dtype="U16")
    n_identity_candidates = np.zeros(n, dtype=np.int16)
    stale_schema = points is not None and "stale" in points.columns
    cmap, groups = {}, {}
    if points is not None:
        _, cmap, groups = prepare_points(points)
    deps = deps.with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        takeoff=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64))
    airports, takeoffs = [], []
    cache = {}
    for i, (ap, cs, takeoff) in enumerate(deps.select("ADEP_mvt", "cs", "takeoff").iter_rows()):
        if ap not in AIRPORTS or takeoff is None or not np.isfinite(takeoff):
            raise ValueError("unknown airport or nonfinite movement time")
        airports.append(ap); takeoffs.append(takeoff)
        if points is None:
            continue
        candidates = cmap.get((cs, ap), set())
        n_identity_candidates[i] = len(candidates)
        if not candidates:
            status[i] = 1; continue
        if len(candidates) != 1:
            status[i] = 2; continue
        ic = next(iter(candidates))
        if not isinstance(ic, str) or not ic or (ic, ap) not in groups:
            status[i] = 1; continue
        identities[i] = ic
        start, g = groups[(ic, ap)]
        if (ic, ap) not in cache:
            fields = {c: g[c].cast(pl.Float64).to_numpy() for c in
                      ["t", "lat", "lon", "alt", "gs", "track", "on_ground", "stale"]}
            fields["source_row"] = g["source_row"].to_numpy()
            fields["callsign"] = g["callsign"].to_list()
            cache[(ic, ap)] = fields
        f = cache[(ic, ap)]; t = f["t"]
        ranges = visit_ranges(t, takeoff)
        if ranges is None:
            status[i] = 3; continue
        status[i] = 4
        lat0, lon0 = AIRPORTS[ap]
        for stream, (lo, hi) in enumerate(ranges):
            if lo < 0:
                continue
            raw_ranges[i, stream] = [start+lo, start+hi]
            counts[i, stream] = hi-lo
            if lo == hi:
                continue
            chosen = sample_time_indices(t[lo:hi]) + lo
            size = len(chosen)
            if size > K:
                raise AssertionError("time sampler exceeds fixed capacity")
            tt = t[chosen]
            if np.any(tt > takeoff) or np.any(np.diff(tt) < 0):
                raise AssertionError("future or unordered token")
            gap = np.zeros(size)
            has_prev = chosen > lo
            gap[has_prev] = tt[has_prev] - t[chosen[has_prev]-1]
            dup = np.zeros(size)
            dup[has_prev] = (gap[has_prev] == 0).astype(np.uint8)
            has_next = chosen+1 < hi
            dup[has_next] = np.maximum(dup[has_next], t[chosen[has_next]+1] == tt[has_next])
            csign = [f["callsign"][int(j)] for j in chosen]
            cmatch = np.array([float(c == cs) if c is not None else np.nan for c in csign])
            heading = np.deg2rad(f["track"][chosen])
            block = np.column_stack([takeoff-tt, gap,
                     (f["lon"][chosen]-lon0)*111.2*np.cos(np.deg2rad(lat0)),
                     (f["lat"][chosen]-lat0)*111.2, f["alt"][chosen], f["gs"][chosen],
                     np.sin(heading), np.cos(heading), f["on_ground"][chosen],
                     f["stale"][chosen], cmatch, dup])
            finite = np.isfinite(block)
            values[i, stream, :size] = np.where(finite, block, 0).astype(np.float32)
            known[i, stream, :size] = finite
            present[i, stream, :size] = 1
            source_rows[i, stream, :size] = f["source_row"][chosen]
            point_times[i, stream, :size] = tt
    arrays = {"ids": ids, "values": values, "known": known, "present": present,
              "source_rows": source_rows, "point_times": point_times,
              "raw_ranges": raw_ranges, "raw_counts": counts, "status": status,
              "identity": identities, "identity_candidates": n_identity_candidates,
              "airport": np.array(airports, dtype="U4"), "takeoff": np.array(takeoffs)}
    if not np.isfinite(values).all() or np.any(known > present[..., None]):
        raise AssertionError("invalid tensor/mask contract")
    stats = {"day": day, "rows": n, "surface_present": points is not None,
             "stale_column_present": stale_schema, "statuses": {name: int((status == j).sum())
             for j, name in enumerate(STATUSES)}, "own_present": int((counts[:, 0] > 0).sum()),
             "previous_present": int((counts[:, 1] > 0).sum()),
             "both_present": int(((counts[:, 0] > 0) & (counts[:, 1] > 0)).sum()),
             "previous_only": int(((counts[:, 0] == 0) & (counts[:, 1] > 0)).sum()),
             "selected_points": int(present.sum()), "known_stale_points": int(known[..., 9].sum()),
             "selected_stale_points": int(((values[..., 9] == 1) & (known[..., 9] == 1)).sum())}
    stats["by_airport"] = {ap: {"rows": int((arrays["airport"] == ap).sum()),
        "own_present": int(((arrays["airport"] == ap) & (counts[:, 0] > 0)).sum()),
        "previous_present": int(((arrays["airport"] == ap) & (counts[:, 1] > 0)).sum())}
        for ap in sorted(set(airports))}
    return arrays, stats


def process_day(task):
    bundle, output, day, files = task
    bundle, output = Path(bundle), Path(output)
    began = time.monotonic()
    dp = bundle / "departures" / f"{day}.parquet"
    sp = bundle / "surface" / f"{day}.parquet"
    for p in [dp] + ([sp] if sp.exists() else []):
        if sha(p) != files[str(p.relative_to(bundle))]:
            raise ValueError(f"day input changed: {p}")
    deps = pl.read_parquet(dp, columns=DEP_COLS)
    if deps.filter(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d") != day).height:
        raise ValueError("departure shard contains another day")
    points = None
    if sp.exists():
        available = pl.scan_parquet(sp).collect_schema()
        points = pl.read_parquet(sp, columns=[c for c in POINT_COLS if c in available])
    arrays, stats = encode_day(points, deps, day)
    dest = output / day
    if dest.exists():
        raise FileExistsError("output day already exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{day}.partial-", dir=output))
    np.savez_compressed(stage / "sequence.npz", **arrays)
    stats.update(seconds=time.monotonic()-began, npz_sha256=sha(stage / "sequence.npz"),
                 protocol=PROTOCOL, source_sha256=sha(Path(__file__)))
    save(stage / "receipt.json", stats)
    for p in [dp] + ([sp] if sp.exists() else []):
        if sha(p) != files[str(p.relative_to(bundle))]:
            raise ValueError("input changed during day extraction")
    stage.rename(dest)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def run(bundle, output, registration, workers):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if not 1 <= workers <= 3:
        raise ValueError("one to three workers only")
    m = verify_bundle(bundle, registration)
    if output.exists():
        raise FileExistsError("new output required; no automatic resume")
    output.mkdir(parents=True)
    began = time.monotonic()
    tasks = [(str(bundle), str(output), d, m["files"]) for d in m["days"]]
    if workers == 1:
        reports = [process_day(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            reports = list(pool.map(process_day, tasks))
    if [x["day"] for x in reports] != m["days"] or sum(x["rows"] for x in reports) != m["departure_rows"]:
        raise ValueError("incomplete day/row coverage")
    save(output / "complete.json", {"protocol": PROTOCOL, "registration_commit": registration,
         "input_manifest_sha256": sha(bundle / "manifest.json"), "days": m["days"],
         "shard_index": m["shard_index"], "shard_count": m["shard_count"],
         "rows": sum(r["rows"] for r in reports), "seconds": time.monotonic()-began,
         "reports": reports, "numeric_columns": NUM_NAMES, "accepted_for_training": False,
         "label_free": True, "runtime": {"python": sys.version, "numpy": np.__version__,
         "polars": pl.__version__, "workers": workers, "polars_threads": pl.thread_pool_size()}})
    print(json.dumps({"complete": str(output / "complete.json"), "days": len(reports)}), flush=True)


def verify_output(bundle, output, registration):
    bundle, output = Path(bundle), Path(output)
    m = verify_bundle(bundle, registration)
    complete = read(output / "complete.json")
    if complete["days"] != m["days"] or complete["input_manifest_sha256"] != sha(bundle / "manifest.json"):
        raise ValueError("output/input mismatch")
    count = 0
    for day in m["days"]:
        receipt = read(output / day / "receipt.json")
        p = output / day / "sequence.npz"
        if sha(p) != receipt["npz_sha256"]:
            raise ValueError("output hash mismatch")
        with np.load(p, allow_pickle=False) as a:
            expected = ids_checked(pl.read_parquet(bundle / "departures" / f"{day}.parquet", columns=DEP_COLS))
            if not np.array_equal(a["ids"], expected) or not np.isfinite(a["values"]).all():
                raise ValueError("output ID/value mismatch")
            if np.any(a["known"] > a["present"][..., None]):
                raise ValueError("mask mismatch")
            if np.any(a["point_times"] > a["takeoff"][:, None, None]):
                raise ValueError("future observed point in tokens")
            count += len(expected)
    print(json.dumps({"verified_days": len(m["days"]), "verified_rows": count}), flush=True)


def selftest():
    # Only generated arrays: no real labels or source archive are loaded here.
    assert len(set(selected_days(0, 2)) & set(selected_days(1, 2))) == 0
    assert sorted(selected_days(0, 2)+selected_days(1, 2)) == sorted(DAYS)
    assert len(DAYS) == 427
    t = np.array([0., 10., 1900., 1910., 1920.])
    assert visit_ranges(t, 1915.) == ((2, 4), (0, 2))
    assert visit_ranges(np.array([0., 1800., 1801.]), 1801.) == ((0, 3), (-1, -1))
    assert visit_ranges(np.array([1000.]), 1000.) is None
    assert len(sample_time_indices(np.arange(1000.))) == K
    assert np.array_equal(sample_time_indices(np.array([0., 0., 1.])), [0, 1, 2])
    day = "2025-01-01"; anchor = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    base = anchor.timestamp()
    deps = pl.DataFrame({"MVT_ID_mvt": [1.], "PHASE_mvt": ["DEP"], "ADEP_mvt": ["LIRF"],
        "CALLSIGN_flt": ["TST1"], "FLIGHT_mvt": ["FALLBACK"],
        "MVT_TIME_UTC_mvt": [anchor+dt.timedelta(seconds=1915)], "FLIGHT_ID_mvt": [1.]})
    points = pl.DataFrame({"icao": ["abc123"]*5, "airport": ["LIRF"]*5,
        "t": base+t, "lat": [41.8045]*5, "lon": [12.2508]*5, "alt": [None]*5,
        "gs": [0., 1., 5., 6., 10.], "track": [0., 90., 180., 270., None],
        "on_ground": [True, True, True, None, False], "callsign": ["TST1"]*5})
    a, _ = encode_day(points, deps, day)
    assert a["raw_counts"].tolist() == [[2, 2]]
    assert a["present"].sum() == 4 and a["known"][..., 9].sum() == 0
    assert a["source_rows"][0, 0, :2].tolist() == [2, 3]
    assert a["known"][0, 0, 1, 8] == 0
    assert np.all(a["point_times"][np.isfinite(a["point_times"])] <= base+1915)
    missing, _ = encode_day(None, deps, day)
    assert missing["present"].sum() == 0 and missing["status"].tolist() == [0]
    ambiguous = pl.concat([points, points.with_columns(pl.lit("def456").alias("icao"))])
    a, _ = encode_day(ambiguous, deps, day)
    assert a["present"].sum() == 0 and a["status"].tolist() == [2]
    a, _ = encode_day(points.with_columns(pl.lit("OTHER").alias("airport")), deps, day)
    assert a["status"].tolist() == [1]
    tagged = points.with_columns(pl.Series("stale", [True, False, None, False, False]))
    a, _ = encode_day(tagged, deps, day)
    assert a["known"][0, 0, 0, 9] == 0 and a["values"][0, 1, 0, 9] == 1
    try:
        ids_checked(pl.concat([deps, deps]))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate IDs accepted")
    assert not {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"}.intersection(DEP_COLS)
    print(json.dumps({"selftest": "PASS", "cases": 16, "protocol": PROTOCOL}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("pack")
    p.add_argument("--root", default="."); p.add_argument("--output", required=True)
    p.add_argument("--registration-commit", required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-count", type=int, default=2)
    for name in ["run", "verify"]:
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True); p.add_argument("--output", required=True)
        p.add_argument("--registration-commit", required=True)
        if name == "run":
            p.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "selftest": selftest()
    elif args.command == "pack": pack(args.root, args.output, args.registration_commit, args.shard_index, args.shard_count)
    elif args.command == "run": run(args.bundle, args.output, args.registration_commit, args.workers)
    else: verify_output(args.bundle, args.output, args.registration_commit)


if __name__ == "__main__":
    main()
