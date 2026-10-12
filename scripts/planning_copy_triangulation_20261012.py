#!/usr/bin/env python3
"""Registered, descriptive alignment and independent stand-geometry triangulation.

No training, inference, downloads, predictions or submissions. `preflight` reads
only schemas, sizes and hashes. `run` requires this source, protocol, manifest and
journal marker in the specified commit. First movement is never used as AOBT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import signal
import resource
import sys

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

PROTOCOL = "planning-copy-triangulation-20261012-v1"
BASE = Path("analysis/planning_copy_triangulation_20261012")
SCRIPT = Path("scripts/planning_copy_triangulation_20261012.py")
TEST = Path("tests/test_planning_copy_triangulation_20261012.py")
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
PLANS = ["SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt", "EOBT_1_flt"]
ANCHORS = ["sched", "iobt", "lobt", "eobt"]
OFFSETS = (-180, -120, -60, 0, 60, 120, 180)
TOLERANCES = (1, 3, 6, 12, 30)
TRAIN_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
              "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "ADEP_flt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt", *PLANS]
STAND_COLS = ["MVT_ID_mvt", "gs_stand_known", "gs_seen_at_stand", "gs_d_first",
              "gs_d_firstmove", "gs_t_leave", "gs_dwell_stand"]
TRACK_COLS = ["MVT_ID_mvt", "coverage", "n_pts", "n_cand"]
RAW_DATES = [f"2025-{month:02d}-{day:02d}" for month in range(1, 13) for day in (5, 15, 25)]
RAW_ROOTS = [Path(f"/Volumes/PRCEXT/prc-2026-data/e194_inputs_20261007_s{shard}/surface") for shard in (0, 1)]
RAW_COLS = ["icao", "airport", "t", "lat", "lon", "gs", "on_ground", "alt", "callsign", "stale"]
ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12,
           "LEMD": 1998, "LFPG": 392, "LIRF": 13, "LSZH": 1416, "LTFM": 325}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def primary_population(y):
    y = np.asarray(y, dtype=float)
    return np.isfinite(y) & (y >= 0) & (y < 86400)


def matches(residuals, tolerance, offset=0):
    """Residual = (takeoff - recorded taxi duration) - planning timestamp, seconds.

    Positive offset moves the planning timestamp later. Eligibility is unchanged
    by offset/tolerance; union counts a row once even when anchors coincide.
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 2:
        raise ValueError("residuals must be rows x planning timestamps")
    finite = np.isfinite(r)
    hit = finite & (np.abs(r - offset) <= tolerance)
    return finite.any(axis=1), hit.any(axis=1), hit.sum(axis=1) > 1


def stand_eligibility(df):
    """Label-blind broad stand-neighbourhood evidence, not continuous stand truth.

    The v5 extractor associates callsign/airport and uses only geometry/track.
    Its inferred first-near time is leave - dwell, not an estimated block time.
    """
    a = {c: df[c].cast(pl.Float64).to_numpy() for c in ["takeoff", *STAND_COLS[1:], *TRACK_COLS[1:]]}
    near = a["gs_t_leave"] - a["gs_dwell_stand"]
    return ((a["gs_stand_known"] == 1) & (a["gs_seen_at_stand"] == 1)
            & (a["gs_d_first"] >= 0) & (a["gs_d_first"] <= 40)
            & (a["gs_d_firstmove"] >= 0) & (a["gs_d_firstmove"] <= 40)
            & (a["gs_dwell_stand"] >= 90) & (a["n_cand"] == 1)
            & (a["n_pts"] >= 10) & (a["coverage"] >= 2)
            & np.isfinite(near) & np.isfinite(a["gs_t_leave"])
            & (near >= a["takeoff"] - 7200) & (a["gs_t_leave"] <= a["takeoff"]))


def standardized_contrast(rows, min_per_arm=10):
    """Fixed common-support weighted risk difference; no model fit.

    Weights are pooled eligible population within airport x month x input-delay
    bin x NM-presence strata. Cells need >=10 flagged and >=10 unflagged.
    """
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], {})[bool(r["flagged"])] = r
    keep = [v for v in cells.values() if all(k in v and v[k]["n"] >= min_per_arm for k in (False, True))]
    total = sum(r["n"] for r in rows)
    denom = sum(v[True]["n"] + v[False]["n"] for v in keep)
    if not denom:
        return dict(cells=0, eligible_rows=0, discarded_rows=total, risk_difference=None,
                    retained_flagged_rows=0, retained_unflagged_rows=0, retained_months=[])
    rd = sum((v[True]["n"] + v[False]["n"]) *
             (v[True]["late"] / v[True]["n"] - v[False]["late"] / v[False]["n"]) for v in keep) / denom
    months = sorted({v[True]["month"] for v in keep if "month" in v[True]})
    return dict(cells=len(keep), eligible_rows=denom, discarded_rows=total-denom, risk_difference=rd,
                retained_flagged_rows=sum(v[True]["n"] for v in keep),
                retained_unflagged_rows=sum(v[False]["n"] for v in keep), retained_months=months)


def checked_ids(df, name):
    ids = df["MVT_ID_mvt"].to_numpy()
    if not np.all(np.isfinite(ids)) or not np.all(ids == np.rint(ids)):
        raise ValueError(f"{name}: missing or non-integral IDs")
    if df["MVT_ID_mvt"].n_unique() != df.height:
        raise ValueError(f"{name}: duplicate IDs")
    return df.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))


def local_distance(lat, lon, lat0, lon0):
    return np.hypot((np.asarray(lat)-lat0)*111200,
                    (np.asarray(lon)-lon0)*111200*np.cos(np.radians(lat0)))


def raw_stationary_prefix(track, takeoff, stand):
    """First contiguous stationary prefix of terminal ground segment, label blind.

    No search for a later favourable stop. No interpolation or gap bridging in
    the accepted stationary prefix. First-near timestamp is an observation.
    """
    t, surf = track["t"], track["surf"]
    hit = np.flatnonzero(surf & (t <= takeoff + 300))
    out = dict(accepted=False, reason="no_terminal_ground", first_near=None, dwell_s=0., n_stationary=0)
    if not len(hit) or takeoff-t[hit[-1]] > 900:
        return out
    end = int(hit[-1])
    start = end
    while start > 0 and surf[start-1] and 0 <= t[start]-t[start-1] <= 900:
        start -= 1
    lat, lon, gs = track["lat"], track["lon"], track["gs"]
    stale = track.get("stale", np.ones(len(t), dtype=bool))
    dstand = local_distance(lat, lon, *stand)
    dfirst = local_distance(lat, lon, lat[start], lon[start])
    good = (surf & np.isfinite(gs) & (gs >= 0) & (gs <= 1) & ~stale
            & np.isfinite(dstand) & (dstand <= 40) & (dfirst <= 25) & (t <= takeoff))
    stop = start
    while stop <= end and good[stop] and (stop == start or 0 < t[stop]-t[stop-1] <= 30):
        stop += 1
    n = stop-start
    span = float(t[stop-1]-t[start]) if n else 0.
    out.update(first_near=float(t[start]), dwell_s=span, n_stationary=n)
    if n >= 5 and span >= 90:
        out.update(accepted=True, reason="validated_stationary_prefix")
    else:
        out["reason"] = "insufficient_stationary_prefix"
    return out


def preflight(root):
    patterns = {"training": "data/prc-2026-datasets/training_*.parquet",
                "stand": "data/adsblol/offblock5_2025*.parquet",
                "track": "data/adsblol_kt1/offblock2_2025*.parquet"}
    requirements = {"training": TRAIN_COLS, "stand": STAND_COLS, "track": TRACK_COLS}
    records = []
    for kind, pattern in patterns.items():
        paths = sorted(root.glob(pattern))
        if kind == "training" and len(paths) != 12:
            raise RuntimeError("exactly twelve 2025 training files required")
        for path in paths:
            schema = pl.read_parquet_schema(path)
            if not set(requirements[kind]).issubset(schema):
                raise ValueError(f"{path}: absent required columns")
            if kind == "training" and "training_2025-" not in path.name:
                raise ValueError("only 2025 labels allowed")
            records.append(dict(kind=kind, path=str(path.relative_to(root)), bytes=path.stat().st_size,
                                sha256=sha(path), schema={k: str(schema[k]) for k in requirements[kind]}))
    raw_missing = []
    for day in RAW_DATES:
        options = [p/f"{day}.parquet" for p in RAW_ROOTS if (p/f"{day}.parquet").is_file()]
        if not options:
            raw_missing.append(day)
            continue
        hashes = [sha(p) for p in options]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"raw duplicate date differs across fixed roots: {day}")
        path = options[0].resolve()
        schema = pl.read_parquet_schema(path)
        if not (set(RAW_COLS)-{"stale"}).issubset(schema):
            raise RuntimeError(f"missing raw schema fields: {path}")
        records.append(dict(kind="raw", day=day, path=str(path), bytes=path.stat().st_size,
                            sha256=hashes[0], equal_duplicate_paths=[str(x.resolve()) for x in options[1:]],
                            stale_available="stale" in schema, schema={k: str(schema[k]) for k in RAW_COLS if k in schema}))
    if sum(r["bytes"] for r in records if r["kind"] == "raw") > 1_000_000_000:
        raise RuntimeError("fixed raw supplement exceeds 1GB compressed bound")
    for rel in (SCRIPT, TEST, BASE / "PROTOCOL.md", Path("src/train.py"),
                Path("src/adsb_offblock_v5.py"), Path("src/adsb_offblock_v2.py"), Path("src/geometry.py"),
                *[Path(f"analysis/sources/samples/osm_overpass/{a}.json") for a in AIRPORTS]):
        path = root / rel
        records.append(dict(kind="source", path=str(rel), bytes=path.stat().st_size, sha256=sha(path)))
    by_kind = {kind: [x for x in records if x["kind"] == kind] for kind in (*patterns, "raw", "source")}
    dates = {kind: {Path(x["path"]).stem.rsplit("_", 1)[-1] for x in by_kind[kind]} for kind in ("stand", "track")}
    import datetime as dt
    year = {(dt.date(2025, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(365)}
    out = dict(protocol=PROTOCOL, outcomes_read=False, files=records,
               inventory={k: dict(files=len(v), bytes=sum(x["bytes"] for x in v)) for k, v in by_kind.items()},
               missing_days={k: sorted(year - dates[k]) for k in dates},
               raw_surface_files_locally=len(list(root.glob("data/adsblol/surface_2025*.parquet"))),
               raw_dates_requested=RAW_DATES, raw_missing_dates=raw_missing,
               scientific_scope="Whole-year derived primary; fixed 36-date external-disk raw supplement; no network.")
    dest = root / BASE / "preflight.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("protocol", "outcomes_read", "inventory", "missing_days", "raw_surface_files_locally")}, indent=2))


def verify_registration(root, commit, manifest):
    if not commit or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("full preregistration commit required")
    for rel in (SCRIPT, TEST, BASE / "PROTOCOL.md", BASE / "preflight.json"):
        data = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)
        if hashlib.sha256(data).hexdigest() != sha(root / rel):
            raise RuntimeError(f"registered content changed: {rel}")
    journal = subprocess.check_output(["git", "show", f"{commit}:HYPOTHESES_LOG.md"], cwd=root).decode()
    if not any("| 464 |" in s and PROTOCOL in s for s in journal.splitlines()):
        raise RuntimeError("journal #464 protocol marker absent in preregistration commit")
    if manifest.get("protocol") != PROTOCOL or manifest.get("outcomes_read") is not False:
        raise RuntimeError("wrong or non-preflight manifest")
    for f in manifest["files"]:
        p = root / f["path"]
        if p.stat().st_size != f["bytes"] or sha(p) != f["sha256"]:
            raise RuntimeError(f"pinned input changed: {f['path']}")


def load_inputs(root, manifest):
    def paths(kind):
        return [root / x["path"] for x in manifest["files"] if x["kind"] == kind]
    src = (pl.concat([pl.scan_parquet(p).select(TRAIN_COLS) for p in paths("training")])
           .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(AIRPORTS)).collect())
    df = checked_ids(src, "departures").select(
        "MVT_ID_mvt", pl.col("ADEP_mvt").alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("us").cast(pl.Float64).truediv(1e6).alias("takeoff"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("us").cast(pl.Float64).truediv(1e6).alias("block_raw"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        pl.col("ADEP_flt").is_not_null().alias("nm_present"),
        pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", "").alias("callsign"),
        pl.col("STAND_mvt").alias("stand"),
        *[pl.col(p).dt.epoch("us").cast(pl.Float64).truediv(1e6).alias(a) for p, a in zip(PLANS, ANCHORS)])
    for kind, cols in (("stand", STAND_COLS), ("track", TRACK_COLS)):
        pp = paths(kind)
        if pp:
            d = checked_ids(pl.concat([pl.read_parquet(p, columns=cols) for p in pp]), kind)
            df = df.join(d.with_columns(pl.lit(True).alias(f"has_{kind}_row")), on="MVT_ID_mvt", how="left", validate="1:1")
        else:
            df = df.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in cols[1:]], pl.lit(False).alias(f"has_{kind}_row"))
    delay = pl.col("takeoff") - pl.col("sched")
    return df.with_columns(
        (pl.col("takeoff") - pl.col("y")).alias("block_implied"),
        pl.when(delay.is_null()).then(pl.lit("missing")).when(delay < 0).then(pl.lit("negative"))
        .when(delay <= 1800).then(pl.lit("0..1800"))
        .when(delay <= 3600).then(pl.lit("1800..3600"))
        .when(delay <= 7200).then(pl.lit("3600..7200"))
        .otherwise(pl.lit(">7200")).alias("delay_bin"),
        pl.col("coverage").fill_null(-1).cast(pl.Int64).alias("coverage_bin"),
        pl.col("has_stand_row").fill_null(False), pl.col("has_track_row").fill_null(False))


def group_indices(df):
    specs = {"all": [], "airport": ["airport"], "airport_month": ["airport", "month"],
             "airport_delay": ["airport", "delay_bin"], "airport_nm": ["airport", "nm_present"],
             "airport_coverage": ["airport", "coverage_bin"],
             "airport_month_delay": ["airport", "month", "delay_bin"]}
    out = {}
    for name, cols in specs.items():
        if not cols:
            out[name] = (np.zeros(df.height, dtype=np.int32), ["ALL"])
        else:
            values = df.select(cols).unique().sort(cols).with_row_index("gi")
            idx = df.select(cols).join(values, on=cols, how="left", maintain_order="left")["gi"].to_numpy()
            keys = ["/".join(map(str, row)) for row in values.select(cols).iter_rows()]
            out[name] = (idx, keys)
    return out


def alignment_tables(df, outdir):
    population = primary_population(df["y"].to_numpy())
    block = df["block_implied"].to_numpy()
    residuals = np.column_stack([block - df[a].to_numpy() for a in ANCHORS])
    groups = group_indices(df)
    rows = []
    for anchor in ["any", *ANCHORS]:
        r = residuals if anchor == "any" else residuals[:, [ANCHORS.index(anchor)]]
        eligible = np.isfinite(r).any(axis=1) & population
        counts = {scope: np.bincount(idx[eligible], minlength=len(keys)) for scope, (idx, keys) in groups.items()}
        for tolerance in TOLERANCES:
            for offset in OFFSETS:
                _, hit, multiple = matches(r, tolerance, offset)
                hit &= population
                multiple &= population
                for scope, (idx, keys) in groups.items():
                    sums = np.bincount(idx[hit], minlength=len(keys))
                    mults = np.bincount(idx[multiple], minlength=len(keys))
                    for i, key in enumerate(keys):
                        n = int(counts[scope][i])
                        rows.append(dict(scope=scope, key=key, anchor=anchor, tolerance_s=tolerance,
                                         offset_s=offset, eligible=n, matches=int(sums[i]),
                                         multiple_anchor_matches=int(mults[i]), rate=float(sums[i]/n) if n else None))
    tab = pl.DataFrame(rows)
    tab.write_csv(outdir / "alignment.csv")
    by = ["scope", "key", "anchor", "tolerance_s"]
    zero = tab.filter(pl.col("offset_s") == 0).drop("offset_s").rename({"rate": "zero_rate"})
    ctrl = tab.filter(pl.col("offset_s") != 0).group_by(by).agg(
        pl.col("rate").mean().alias("control_mean_rate"), pl.col("rate").min().alias("control_min_rate"),
        pl.col("rate").max().alias("control_max_rate"))
    contrast = zero.join(ctrl, on=by).with_columns(
        (pl.col("zero_rate") - pl.col("control_mean_rate")).alias("excess_rate"),
        (pl.col("zero_rate") > pl.col("control_max_rate")).alias("exceeds_all_controls"))
    contrast.sort(by).write_csv(outdir / "alignment_contrasts.csv")
    eligible, flag, multiple = matches(residuals, 6)
    return population & eligible, population & flag, multiple, contrast


def triangulation(df, eligible, flag, outdir):
    good = stand_eligibility(df)
    df = df.with_columns(pl.Series("stand_eligible", good), pl.Series("primary", eligible), pl.Series("flagged", flag),
                         (pl.col("gs_t_leave")-pl.col("gs_dwell_stand")-pl.col("block_implied")).alias("first_near_minus_block"),
                         (pl.col("gs_t_leave")-pl.col("block_implied")).alias("leave_minus_block"))
    funnel = df.group_by("airport", "month", "delay_bin", "nm_present", "coverage_bin").agg(
        pl.len().alias("departures"), pl.col("primary").sum().alias("primary"),
        pl.col("has_stand_row").sum().alias("stand_product_rows"), pl.col("has_track_row").sum().alias("track_product_rows"),
        (pl.col("gs_stand_known") == 1).sum().alias("known_stand"),
        (pl.col("gs_seen_at_stand") == 1).sum().alias("seen_near_stand"),
        pl.col("stand_eligible").sum().alias("eligible_geometry"),
        (pl.col("stand_eligible") & pl.col("primary")).sum().alias("primary_geometry"),
        (pl.col("stand_eligible") & pl.col("primary") & pl.col("flagged")).sum().alias("flagged_geometry"))
    funnel.sort("airport", "month", "delay_bin", "nm_present", "coverage_bin").write_csv(outdir / "availability.csv")
    q = df.filter(pl.col("stand_eligible") & pl.col("primary")).with_columns(
        (pl.col("first_near_minus_block") >= 300).alias("late_presence"),
        (pl.col("leave_minus_block") <= -300).alias("block_after_exit"))
    fields = ["airport", "month", "delay_bin", "nm_present", "flagged"]
    summary = q.group_by(fields).agg(pl.len().alias("n"), pl.col("late_presence").sum().alias("late"),
        pl.col("block_after_exit").sum().alias("block_after_exit"),
        pl.col("first_near_minus_block").quantile(.1).alias("first_near_minus_block_p10"),
        pl.col("first_near_minus_block").median().alias("first_near_minus_block_median"),
        pl.col("first_near_minus_block").quantile(.9).alias("first_near_minus_block_p90"),
        pl.col("leave_minus_block").median().alias("leave_minus_block_median"))
    summary.sort(fields).write_csv(outdir / "stand_geometry_summary.csv")
    output = {}
    for scope in ["ALL", *AIRPORTS, "LEBL_JulSep", "LEBL_JanJun", "LEBL_OctDec"]:
        sub = summary
        if scope.startswith("LEBL_"):
            months = {"LEBL_JulSep": [7,8,9], "LEBL_JanJun": list(range(1,7)), "LEBL_OctDec": [10,11,12]}[scope]
            sub = sub.filter((pl.col("airport") == "LEBL") & pl.col("month").is_in(months))
        elif scope != "ALL":
            sub = sub.filter(pl.col("airport") == scope)
        rows = [dict(cell="/".join(str(r[c]) for c in fields[:-1]), flagged=r["flagged"], n=r["n"], late=r["late"], month=r["month"]) for r in sub.iter_rows(named=True)]
        output[scope] = standardized_contrast(rows)
        output[scope]["available_flagged_rows"] = sum(r["n"] for r in rows if r["flagged"])
        output[scope]["available_unflagged_rows"] = sum(r["n"] for r in rows if not r["flagged"])
        output[scope]["months_with_any_flagged_geometry"] = sub.filter(pl.col("flagged"))["month"].n_unique()
        r = output[scope]
        enough = r["retained_flagged_rows"] >= 50 and r["retained_unflagged_rows"] >= 50 and len(r["retained_months"]) >= 3 and r["cells"] >= 3
        r["sufficiency_gate"] = enough
        r["interpretation"] = ("directionally_consistent" if r["risk_difference"] > 0 else "neutral_or_conflicting") if enough else "inconclusive_insufficient_common_support"
    (outdir / "stand_standardized.json").write_text(json.dumps(output, indent=2) + "\n")
    return df, output


def raw_supplement(root, manifest, enriched, outdir):
    """Stream fixed raw dates, inspect all derived-eligible visits before labels."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("triangulation_geometry", root / "src/geometry.py")
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    geo.OSM_DIR = root / "analysis/sources/samples/osm_overpass"
    st, _ = geo.load_tables()
    exact = {(a, k): (la, lo) for a, k, la, lo in st.iter_rows()}
    suffix = {}
    for a, k, la, lo in st.iter_rows():
        suffix.setdefault((a, geo.strip_suffix(k)), (la, lo))
    pool = enriched.filter(pl.col("stand_eligible") & pl.col("day").is_in(RAW_DATES))
    raw_by_day = {r["day"]: root/r["path"] for r in manifest["files"] if r["kind"] == "raw"}
    output, day_rows = [], []
    for day in RAW_DATES:
        visits = pool.filter(pl.col("day") == day)
        start = time.monotonic()
        raw_path = raw_by_day.get(day)
        if raw_path is None:
            for r in visits.iter_rows(named=True):
                output.append(dict(MVT_ID_mvt=r["MVT_ID_mvt"], day=day, airport=r["airport"], month=r["month"],
                                   delay_bin=r["delay_bin"], nm_present=r["nm_present"], primary=r["primary"],
                                   flagged=r["flagged"], accepted=False, reason="raw_date_missing", first_near=None,
                                   dwell_s=0., n_stationary=0, late=None, first_near_minus_block=None))
            day_rows.append(dict(day=day, derived_eligible=visits.height, raw_available=False, raw_points=0, accepted=0, seconds=0.))
            continue
        raw_schema = pl.read_parquet_schema(raw_path)
        stale_available = "stale" in raw_schema
        pts = pl.read_parquet(raw_path, columns=[c for c in RAW_COLS if c in raw_schema]).filter(pl.col("airport").is_in(AIRPORTS))
        if not stale_available:
            pts = pts.with_columns(pl.lit(True).alias("stale"))
        elev = pl.col("airport").replace_strict(ELEV_FT, return_dtype=pl.Float64)
        pts = pts.with_columns(
            (pl.col("on_ground") | ((pl.col("gs").fill_null(0) <= 40) & (pl.col("alt").is_null() | (pl.col("alt") <= elev+800))))
            .fill_null(False).alias("surf"),
            pl.col("callsign").str.replace_all(" ", ""), pl.col("stale").fill_null(True)).sort("airport", "icao", "t")
        candidates = {}
        for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
            candidates.setdefault((cs, ap), []).append(icao)
        tracks = {(ap, icao): {k: g[k].to_numpy() for k in ("t", "lat", "lon", "gs", "surf", "stale")}
                  for (ap, icao), g in pts.group_by("airport", "icao")}
        accepted = 0
        for r in visits.iter_rows(named=True):
            key = geo.norm_stand(r["stand"]) if r["stand"] else None
            stand = (exact.get((r["airport"], key)) or suffix.get((r["airport"], geo.strip_suffix(key)))) if key else None
            options = []
            for icao in candidates.get((r["callsign"], r["airport"]), []):
                track = tracks[(r["airport"], icao)]
                t = track["t"][track["surf"] & (track["t"] <= r["takeoff"]+300)]
                if len(t) and r["takeoff"]-t[-1] <= 900:
                    options.append(icao)
            result = dict(accepted=False, reason="no_unique_callsign_track", first_near=None, dwell_s=0., n_stationary=0)
            if not stale_available:
                result["reason"] = "missing_stale_status"
            elif stand is None:
                result["reason"] = "no_exact_or_suffix_stand"
            elif len(options) == 1:
                result = raw_stationary_prefix(tracks[(r["airport"], options[0])], r["takeoff"], stand)
            # Outcome/time comparison is read only after the label-blind selection.
            delta = result["first_near"]-r["block_implied"] if result["accepted"] else None
            output.append(dict(MVT_ID_mvt=r["MVT_ID_mvt"], day=day, airport=r["airport"], month=r["month"],
                               delay_bin=r["delay_bin"], nm_present=r["nm_present"], primary=r["primary"], flagged=r["flagged"],
                               **result, late=(delta >= 300) if delta is not None else None, first_near_minus_block=delta))
            accepted += int(result["accepted"])
        day_rows.append(dict(day=day, derived_eligible=visits.height, raw_available=True, raw_points=pts.height,
                             accepted=accepted, seconds=time.monotonic()-start))
        del pts, tracks, candidates
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > 6 * 1024**3:
            raise RuntimeError("raw supplement exceeded registered 6GiB peak RSS bound")
    pl.DataFrame(day_rows).write_csv(outdir/"raw_day_inventory.csv")
    if not output:
        result = dict(status="inconclusive_no_derived_eligible_visits_on_fixed_dates", sampled_dates=36, scopes={})
    else:
        rr = pl.DataFrame(output, schema_overrides={"first_near": pl.Float64, "first_near_minus_block": pl.Float64, "late": pl.Boolean})
        rr.write_csv(outdir/"raw_episode_audit.csv")
        rr.group_by("airport", "month", "reason").agg(pl.len().alias("n")).sort("airport", "month", "reason").write_csv(outdir/"raw_attrition.csv")
        valid = rr.filter(pl.col("accepted") & pl.col("primary"))
        group = valid.group_by("airport", "month", "delay_bin", "nm_present", "flagged").agg(pl.len().alias("n"), pl.col("late").sum().alias("late"))
        group.sort("airport", "month", "delay_bin", "nm_present", "flagged").write_csv(outdir/"raw_common_support_cells.csv")
        scopes = {}
        for scope in ["ALL", *AIRPORTS, "LEBL_JulSep", "LEBL_JanJun", "LEBL_OctDec"]:
            sub = group
            if scope.startswith("LEBL_"):
                months = {"LEBL_JulSep": [7,8,9], "LEBL_JanJun": list(range(1,7)), "LEBL_OctDec": [10,11,12]}[scope]
                sub = sub.filter((pl.col("airport") == "LEBL") & pl.col("month").is_in(months))
            elif scope != "ALL":
                sub = sub.filter(pl.col("airport") == scope)
            cells = [dict(cell="/".join(str(r[k]) for k in ("airport", "month", "delay_bin", "nm_present")),
                          flagged=r["flagged"], n=r["n"], late=r["late"], month=r["month"]) for r in sub.iter_rows(named=True)]
            z = standardized_contrast(cells)
            enough = z["retained_flagged_rows"] >= 50 and z["retained_unflagged_rows"] >= 50 and len(z["retained_months"]) >= 3 and z["cells"] >= 3
            z["sufficiency_gate"] = enough
            z["interpretation"] = ("directionally_consistent" if z["risk_difference"] > 0 else "neutral_or_conflicting") if enough else "inconclusive_insufficient_common_support"
            scopes[scope] = z
        result = dict(status="descriptive_raw_episode_validation_not_AOBT_truth", sampled_dates=36,
                      derived_eligible=pool.height, validated_episodes=int(rr["accepted"].sum()),
                      validated_primary=valid.height, scopes=scopes)
    (outdir/"raw_supplement.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def run(root, commit):
    t0 = time.monotonic()
    outdir = root / BASE
    if (outdir / "results.json").exists():
        raise RuntimeError("results already exist; refusing overwrite")
    manifest = json.loads((outdir / "preflight.json").read_text())
    verify_registration(root, commit, manifest)
    df = load_inputs(root, manifest)
    block_diff = np.abs(df["block_raw"].to_numpy() - df["block_implied"].to_numpy())
    if (block_diff > 1).any():
        raise RuntimeError(f"BLOCK vs MVT-y contract mismatch >1s on {int((block_diff > 1).sum())} rows; stop before analysis")
    eligible, flag, multiple, contrasts = alignment_tables(df, outdir)
    enriched, stand = triangulation(df, eligible, flag, outdir)
    raw = raw_supplement(root, manifest, enriched, outdir)
    # Fixed negative-control result; this is enrichment, never physical-copy precision.
    primary = contrasts.filter((pl.col("scope") == "airport") & (pl.col("anchor") == "any") & (pl.col("tolerance_s") == 6))
    summary = dict(protocol=PROTOCOL, registration=commit, input_manifest_sha256=sha(outdir/"preflight.json"),
        rows=df.height, primary_rows=int(eligible.sum()), flagged_rows=int(flag.sum()),
        missing_all_plans=int((primary_population(df["y"].to_numpy()) & ~eligible).sum()),
        excluded_y_not_finite_or_outside_0_86400=int((~primary_population(df["y"].to_numpy())).sum()),
        multiple_anchor_rows=int((multiple & eligible).sum()),
        block_timestamp_check=dict(comparable=int(np.isfinite(block_diff).sum()),
                                   disagree_over_1_second=int((block_diff > 1).sum())),
        primary_alignment=primary.to_dicts(), stand_standardized=stand, raw_supplement=raw,
        tests_of_physical_truth=0, model_fits=0, inference_calls=0,
        scientific_status="descriptive triangulation; exact matches and geometric evidence do not establish copying or AOBT truth",
        seconds=time.monotonic()-t0)
    outputs = sorted(p for p in outdir.glob("*") if p.name not in ("results.json", "PROTOCOL.md", "preflight.json"))
    summary["output_hashes"] = {p.name: sha(p) for p in outputs if p.is_file()}
    (outdir/"results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("protocol", "rows", "primary_rows", "flagged_rows", "seconds", "scientific_status")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight", "run"])
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--registration")
    a = p.parse_args()
    if a.action == "preflight":
        preflight(a.root)
    else:
        def timeout(*_):
            raise TimeoutError("registered 900-second outcome runtime bound reached")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(900)
        try:
            run(a.root, a.registration)
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
