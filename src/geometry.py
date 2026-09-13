"""Aerodrome geometry from OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0).

Builds stand coordinates (aeroway=parking_position/gate, matched on normalised ref) and
departure runway start points (the runway end opposite to its heading), then per
departure: straight distance stand -> runway start and the bearing offset between the
runway heading and the direction from stand to the runway start.

Overpass extracts live in analysis/sources/samples/osm_overpass/<ICAO>.json.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl

OSM_DIR = Path("analysis/sources/samples/osm_overpass")
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def norm_stand(s: str) -> str:
    s = str(s).upper().replace(" ", "").replace("-", "")
    m = re.fullmatch(r"([A-Z]*)0*(\d+)([A-Z]*)", s)
    return f"{m.group(1)}{int(m.group(2))}{m.group(3)}" if m else s


def strip_suffix(s: str) -> str:
    m = re.fullmatch(r"([A-Z]*\d+)[A-Z]?", s)
    return m.group(1) if m else s


def _center(e: dict):
    if e["type"] == "node":
        return e["lat"], e["lon"]
    g = e.get("geometry") or []
    if g:
        return float(np.mean([p["lat"] for p in g])), float(np.mean([p["lon"] for p in g]))
    b = e.get("bounds")
    return ((b["minlat"] + b["maxlat"]) / 2, (b["minlon"] + b["maxlon"]) / 2) if b else None


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def load_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    stands, runways = [], []
    for ap in AIRPORTS:
        doc = json.loads((OSM_DIR / f"{ap}.json").read_text())
        rw_points: dict[str, list] = {}
        for e in doc["elements"]:
            tags = e.get("tags", {})
            aw = tags.get("aeroway")
            if aw in ("parking_position", "gate"):
                c = _center(e)
                if c is None:
                    continue
                for key in ("ref", "name"):
                    if not tags.get(key):
                        continue
                    for v in re.split(r"[;,]", tags[key]):
                        v = v.strip()
                        if v:
                            prio = (0 if aw == "parking_position" else 1) + (0 if key == "ref" else 2)
                            stands.append((ap, norm_stand(v), c[0], c[1], prio))
            elif aw == "runway" and tags.get("ref") and e["type"] == "way" and e.get("geometry"):
                for rid in tags["ref"].split("/"):
                    rid = rid.strip()
                    if re.match(r"\d+", rid):
                        rw_points.setdefault(rid, []).extend((p["lat"], p["lon"]) for p in e["geometry"])
        for rid, pts in rw_points.items():
            heading = int(re.match(r"(\d+)", rid).group(1)) * 10
            hd = math.radians(heading)
            lat0 = np.mean([p[0] for p in pts])
            proj = [p[0] * math.cos(hd) + p[1] * math.cos(math.radians(lat0)) * math.sin(hd) for p in pts]
            start = pts[int(np.argmin(proj))]
            runways.append((ap, rid, start[0], start[1], float(heading)))
    st = (
        pl.DataFrame(stands, schema=["ADEP_mvt", "stand_key", "stand_lat", "stand_lon", "prio"], orient="row")
        .sort("prio")
        .unique(["ADEP_mvt", "stand_key"], keep="first", maintain_order=True)
        .drop("prio")
    )
    rw = pl.DataFrame(runways, schema=["ADEP_mvt", "RUNWAY_mvt", "rwy_lat", "rwy_lon", "rwy_heading"], orient="row")
    return st, rw


def add_geometry(dep: pl.DataFrame) -> pl.DataFrame:
    st, rw = load_tables()
    st_nosfx = (
        st.with_columns(stand_key=pl.col("stand_key").map_elements(strip_suffix, return_dtype=pl.String))
        .unique(["ADEP_mvt", "stand_key"], keep="first")
        .rename({"stand_lat": "lat2", "stand_lon": "lon2"})
    )
    # stand-prefix centroid as a last fallback
    prefix = (
        st.with_columns(pref=pl.col("stand_key").str.extract(r"^([A-Z]*\d?)", 1))
        .group_by("ADEP_mvt", "pref")
        .agg(lat3=pl.col("stand_lat").mean(), lon3=pl.col("stand_lon").mean())
    )
    keys = dep.select(
        "MVT_ID_mvt",
        "ADEP_mvt",
        "RUNWAY_mvt",
        stand_key=pl.col("STAND_mvt").map_elements(norm_stand, return_dtype=pl.String),
    ).with_columns(
        key_nosfx=pl.col("stand_key").map_elements(strip_suffix, return_dtype=pl.String),
        pref=pl.col("stand_key").str.extract(r"^([A-Z]*\d?)", 1),
    )
    g = (
        keys.join(st, on=["ADEP_mvt", "stand_key"], how="left")
        .join(st_nosfx.rename({"stand_key": "key_nosfx"}), on=["ADEP_mvt", "key_nosfx"], how="left")
        .join(prefix, on=["ADEP_mvt", "pref"], how="left")
        .join(rw, on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        .with_columns(
            geo_match=pl.when(pl.col("stand_lat").is_not_null()).then(2)
            .when(pl.col("lat2").is_not_null()).then(1)
            .when(pl.col("lat3").is_not_null()).then(0)
            .otherwise(None),
            slat=pl.coalesce("stand_lat", "lat2", "lat3"),
            slon=pl.coalesce("stand_lon", "lon2", "lon3"),
        )
    )
    slat, slon = g["slat"].to_numpy(), g["slon"].to_numpy()
    rlat, rlon = g["rwy_lat"].to_numpy(), g["rwy_lon"].to_numpy()
    dist = _haversine(slat, slon, rlat, rlon)
    brg = _bearing(slat, slon, rlat, rlon)
    offset = np.abs(((brg - g["rwy_heading"].to_numpy()) + 180) % 360 - 180)
    geo = g.select("MVT_ID_mvt", "geo_match").with_columns(
        geo_dist_m=pl.Series(dist),
        geo_bearing_offset=pl.Series(offset),
        geo_dist_along=pl.Series(dist * np.cos(np.radians(offset))),
        geo_dist_across=pl.Series(dist * np.sin(np.radians(offset))),
    )
    return dep.join(geo, on="MVT_ID_mvt", how="left")
