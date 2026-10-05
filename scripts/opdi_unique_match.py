"""Unlabeled exact callsign/airport linkage with global mutual uniqueness.

The caller supplies one canonical public anchor per OPDI flight. No I/O, target
columns, nearest-neighbour choice, or numeric conversion of identifiers occurs.
The fixed inclusive window is anchor_s - mvt_s in [-900, +1800] seconds.
"""
from __future__ import annotations

import polars as pl

_OUTPUT = ["mvt_id", "opdi_flight_id", "anchor_type", "anchor_delta_s"]
# Python str.split includes the four ASCII information separators as whitespace;
# Rust regex's Unicode White_Space class needs those added explicitly.
_WHITESPACE = r"[\s\x1c-\x1f]+"


def normalize_callsign(value) -> str | None:
    """Remove every whitespace character and uppercase a string, preserving null."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("callsign must be a string or None")
    return "".join(value.split()).upper() or None


def _prepare(frame: pl.DataFrame, *, anchor: bool) -> tuple[pl.DataFrame, dict]:
    label = "anchor" if anchor else "prc"
    ident, time = ("opdi_flight_id", "anchor_s") if anchor else ("mvt_id", "mvt_s")
    required = [ident, "cs", "airport", time] + (["anchor_type"] if anchor else [])
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(f"{label} must be a polars DataFrame")
    if set(required) - set(frame.columns):
        raise ValueError(f"{label} missing columns {sorted(set(required)-set(frame.columns))}")
    # Check identity and time BEFORE any exclusions, so invalid/duplicate rows
    # cannot disappear because their callsign or airport happens to be absent.
    if frame.schema[ident] != pl.String or frame[ident].null_count():
        raise ValueError(f"{label} {ident} requires non-null canonical string IDs")
    if frame[ident].str.strip_chars().eq("").any():
        raise ValueError(f"{label} {ident} contains empty IDs")
    if frame[ident].n_unique() != frame.height:
        raise ValueError(f"{label} contains duplicate {ident}")
    if frame.schema[time] != pl.Int64 or frame[time].null_count():
        raise ValueError(f"{label} {time} requires non-null Int64 seconds, never float")
    for key in ("cs", "airport"):
        if frame.schema[key] not in (pl.String, pl.Null):
            raise ValueError(f"{label} {key} requires strings or nulls")
    if anchor and (frame.schema["anchor_type"] != pl.String or frame["anchor_type"].null_count()
                   or frame["anchor_type"].str.strip_chars().eq("").any()):
        raise ValueError("anchor_type requires a nonempty, non-null string")
    selected = frame.select(required).with_columns(
        pl.col("cs").cast(pl.String).str.replace_all(_WHITESPACE, "").str.to_uppercase(),
        pl.col("airport").cast(pl.String))
    selected = selected.with_columns(
        pl.when(pl.col("cs").eq("")).then(None).otherwise(pl.col("cs")).alias("cs"))
    missing_airport = pl.col("airport").is_null() | pl.col("airport").str.strip_chars().eq("")
    audit = {f"input_{label}_rows": frame.height,
             f"{label}_missing_callsign_rows": selected["cs"].null_count(),
             f"{label}_missing_airport_rows": int(selected.select(missing_airport.sum()).item())}
    selected = selected.filter(pl.col("cs").is_not_null() & ~missing_airport)
    audit[f"eligible_{label}_rows"] = selected.height
    return selected, audit


def unique_matches(prc: pl.DataFrame, anchors: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Return only edges whose two endpoints have degree one in the FULL graph.

    Inputs: PRC (mvt_id, cs, airport, mvt_s) and anchors
    (opdi_flight_id, cs, airport, anchor_s, anchor_type). IDs must already be
    canonical strings and times non-null Int64 seconds. Missing normalized
    callsigns/airports are excluded with counts; flights without a usable anchor
    must be excluded/audited by the caller before invoking this function.

    Audit *_ids fields are COUNTS of ambiguous vertices, not ID lists. No tie
    is broken by proximity, row order, or anchor type. Call once on the complete
    population: separately pruning monthly graphs could hide cross-month reuse.
    """
    p, prc_audit = _prepare(prc, anchor=False)
    a, anchor_audit = _prepare(anchors, anchor=True)
    # Only the +/-1 day keys expand. Streaming keeps the three-way expansion and
    # pre-window join lazy; retained edges contain only four required columns.
    p_days = p.lazy().with_columns((pl.col("mvt_s") // 86400).alias("_day"))
    p_days = p_days.with_columns(pl.concat_list(
        [pl.col("_day")-1, pl.col("_day"), pl.col("_day")+1]).alias("_day")).explode("_day", empty_as_null=True)
    a_days = a.lazy().with_columns((pl.col("anchor_s") // 86400).alias("_day"))
    joined = p_days.join(a_days, on=["cs", "airport", "_day"], how="inner")
    edges = (joined.with_columns((pl.col("anchor_s")-pl.col("mvt_s")).alias("anchor_delta_s"))
             .filter(pl.col("anchor_delta_s").is_between(-900, 1800, closed="both"))
             .select(_OUTPUT).collect(engine="streaming"))
    # Both degree columns are computed together on the full candidate graph,
    # BEFORE either side is removed. Sequential pruning would hide OPDI reuse.
    degrees = edges.with_columns(pl.len().over("mvt_id").alias("_mvt_degree"),
                                 pl.len().over("opdi_flight_id").alias("_opdi_degree"))
    matches = (degrees.filter((pl.col("_mvt_degree") == 1) & (pl.col("_opdi_degree") == 1))
               .select(_OUTPUT).sort("mvt_id"))
    with_candidates = edges["mvt_id"].n_unique()
    audit = {**prc_audit, **anchor_audit,
             "candidate_edges": edges.height,
             "mvt_with_candidates": with_candidates,
             "mvt_without_candidates": p.height-with_candidates,
             "anchors_without_candidates": a.height-edges["opdi_flight_id"].n_unique(),
             "ambiguous_mvt_ids": degrees.filter(pl.col("_mvt_degree") > 1)["mvt_id"].n_unique(),
             "reused_opdi_flight_ids": degrees.filter(pl.col("_opdi_degree") > 1)["opdi_flight_id"].n_unique(),
             "unique_matches": matches.height,
             "matched_anchor_types": dict(sorted(matches.group_by("anchor_type").len().iter_rows()))}
    return matches, audit
