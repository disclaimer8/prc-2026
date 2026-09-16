"""Extract airport-surface ADS-B points from one adsb.lol globe_history day.

    cat v2025.07.15-*.tar.a* | uv run python src/adsb_extract.py 2025-07-15

Data: adsb.lol globe_history (ODbL-1.0). Each tar member traces/xx/trace_full_<icao>.json
is a gzip-compressed readsb trace: {"icao", "timestamp", "trace": [[dt, lat, lon, alt|"ground",
gs, track, flags, vrate, details|null, ...], ...]}. Points within RADIUS_KM of one of the
challenge airports and below MAX_ALT_FT (or flagged on ground) are kept.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import tarfile

import polars as pl

AIRPORTS = {  # ICAO: (lat, lon) aerodrome reference points
    "EDDF": (50.0333, 8.5706),
    "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785),
    "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0128, 2.5500),
    "LIRF": (41.8045, 12.2508),
    "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2615, 28.7419),
}
RADIUS_KM = 7.0
MAX_ALT_FT = 2500


def near_airport(lat: float, lon: float) -> str | None:
    for code, (alat, alon) in AIRPORTS.items():
        dy = (lat - alat) * 111.2
        dx = (lon - alon) * 111.2 * math.cos(math.radians(alat))
        if dx * dx + dy * dy <= RADIUS_KM * RADIUS_KM:
            return code
    return None


def main() -> None:
    day = sys.argv[1]
    rows: dict[str, list] = {k: [] for k in ("icao", "airport", "t", "lat", "lon", "alt", "on_ground", "gs", "track", "callsign", "stale")}
    n_files = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
        for member in tar:
            if not member.isfile() or "trace_full_" not in member.name:
                continue
            raw = tar.extractfile(member).read()
            n_files += 1
            try:
                doc = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
            except (OSError, ValueError):
                continue
            base = doc.get("timestamp", 0)
            icao = doc.get("icao", "")
            callsign = None
            for p in doc.get("trace", []):
                if len(p) > 8 and isinstance(p[8], dict) and p[8].get("flight"):
                    callsign = p[8]["flight"].strip()
                lat, lon, alt = p[1], p[2], p[3]
                if lat is None or lon is None:
                    continue
                on_ground = alt == "ground"
                if not on_ground and (alt is None or alt > MAX_ALT_FT):
                    continue
                ap = near_airport(lat, lon)
                if ap is None:
                    continue
                rows["icao"].append(icao)
                rows["airport"].append(ap)
                rows["t"].append(base + p[0])
                rows["lat"].append(lat)
                rows["lon"].append(lon)
                rows["alt"].append(None if on_ground else float(alt))
                rows["on_ground"].append(on_ground)
                rows["gs"].append(None if p[4] is None else float(p[4]))
                rows["track"].append(None if p[5] is None else float(p[5]))
                rows["callsign"].append(callsign)
                # readsb trace flags (p[6]), bit 0: this position is stale — no reception for 20 s, the point repeats the
                # last known position. Kept because a repeated position at a stand is not the same evidence as a fresh one.
                rows["stale"].append(bool(int(p[6]) & 1) if len(p) > 6 and p[6] is not None else None)
            if n_files % 20000 == 0:
                print(f"{n_files} traces, {len(rows['t'])} points", file=sys.stderr, flush=True)
    df = pl.DataFrame(rows)
    out = f"data/adsblol/surface_{day}.parquet"
    df.write_parquet(out)
    print(f"{n_files} traces -> {df.height} points -> {out}")


if __name__ == "__main__":
    main()
