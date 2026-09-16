"""E51: taxi distance along the OpenStreetMap taxiway graph instead of the straight line (16.09).

    .venv/bin/python src/taxi_route.py        writes data/taxi_route.parquet (ADEP_mvt, stand_key, RUNWAY_mvt, route_m, route_detour)

The model knows the straight distance from the stand to the runway start (src/geometry.py). At LFPG, LTFM and EHAM the
stands sit behind terminals and the real taxi follows a path that can be far longer than the straight line, and the
detour differs per stand-runway pair rather than per airport. The graph is built from the same Overpass extracts that
already back the stand geometry: every `aeroway=taxiway` and `aeroway=runway` way becomes a chain of edges between its
OSM nodes (shared node ids join the ways), the stand and the runway start are attached to their nearest graph node
within ATTACH_M, and the route is the Dijkstra distance between them.

Rows whose stand or runway cannot be attached keep a null route; the pair table is small (a few thousand rows per
airport at most), so the whole thing is computed once and joined onto the features.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from collections import defaultdict

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

import geometry as G

OSM_DIR = str(G.OSM_DIR)
ATTACH_M = 300.0  # a stand or runway start further than this from any taxiway node is left without a route
ROUTE_WAYS = ("taxiway", "runway")


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return G._haversine(a[0], a[1], b[0], b[1])


def build_graph(airport: str):
    """node id -> (lat, lon), and node id -> [(neighbour, metres)]."""
    with open(f"{OSM_DIR}/{airport}.json") as f:
        elements = json.load(f)["elements"]
    pos: dict[int, tuple[float, float]] = {}
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for e in elements:
        if e["type"] != "way" or e.get("tags", {}).get("aeroway") not in ROUTE_WAYS:
            continue
        nodes, geom = e.get("nodes", []), e.get("geometry", [])
        if len(nodes) != len(geom) or len(nodes) < 2:
            continue
        for nid, g in zip(nodes, geom):
            pos[nid] = (g["lat"], g["lon"])
        for u, v in zip(nodes, nodes[1:]):
            d = _dist(pos[u], pos[v])
            adj[u].append((v, d))
            adj[v].append((u, d))
    return pos, adj


def nearest(pos: dict[int, tuple[float, float]], lat: float, lon: float):
    best, bd = None, ATTACH_M
    for nid, p in pos.items():
        d = _dist((lat, lon), p)
        if d < bd:
            best, bd = nid, d
    return best, bd


def route_lengths(adj, src: int, targets: set[int]) -> dict[int, float]:
    """Dijkstra from one stand node to every runway node we need."""
    left = set(targets)
    out: dict[int, float] = {}
    seen: set[int] = set()
    q = [(0.0, src)]
    while q and left:
        d, u = heapq.heappop(q)
        if u in seen:
            continue
        seen.add(u)
        if u in left:
            out[u] = d
            left.discard(u)
        for v, w in adj[u]:
            if v not in seen:
                heapq.heappush(q, (d + w, v))
    return out


def main() -> None:
    stands, runways = G.load_tables()
    rows = []
    for airport in sorted(set(stands["ADEP_mvt"].to_list())):
        if not os.path.exists(f"{OSM_DIR}/{airport}.json"):
            print(f"{airport}: no OSM extract", flush=True)
            continue
        pos, adj = build_graph(airport)
        st = stands.filter(pl.col("ADEP_mvt") == airport)
        rw = runways.filter(pl.col("ADEP_mvt") == airport)
        rw_nodes = {}
        for _, ref, lat, lon, _h in rw.iter_rows():
            nid, d = nearest(pos, lat, lon)
            if nid is not None:
                rw_nodes[ref] = (nid, d, lat, lon)
        n_ok = 0
        for _, stand, slat, slon in st.select("ADEP_mvt", "stand_key", "stand_lat", "stand_lon").iter_rows():
            snid, sd = nearest(pos, slat, slon)
            if snid is None or not rw_nodes:
                continue
            lengths = route_lengths(adj, snid, {v[0] for v in rw_nodes.values()})
            for ref, (nid, rd, rlat, rlon) in rw_nodes.items():
                if nid not in lengths:
                    continue
                route = lengths[nid] + sd + rd
                straight = G._haversine(slat, slon, rlat, rlon)
                rows.append((airport, stand, ref, route, route / straight if straight > 50 else None))
                n_ok += 1
        print(f"{airport}: {len(pos)} graph nodes, {st.height} stands, {rw.height} runways -> {n_ok} pairs", flush=True)
    out = pl.DataFrame(rows, schema={"ADEP_mvt": pl.Utf8, "stand_key": pl.Utf8, "RUNWAY_mvt": pl.Utf8,
                                     "route_m": pl.Float64, "route_detour": pl.Float64})
    out.write_parquet("data/taxi_route.parquet")
    print(f"wrote data/taxi_route.parquet rows={out.height} median detour={out['route_detour'].median():.2f}")


if __name__ == "__main__":
    main()
