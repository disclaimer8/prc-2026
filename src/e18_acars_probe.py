"""E18: coverage of adsb.lol aircraft data links (ACARS/VDL2/HFDL, CC0) for LIRF departures (review "top-3 levers", 15.09).

    .venv/bin/python src/e18_acars_probe.py 2026-07-02 2026-07-20 2025-12-15
Needs data/acars/<day>/ (scripts/get_acars_days.sh) and data/ext/openflights_airlines.dat (ICAO -> IATA airline codes).
A departure is covered if a message with its flight number (ICAO callsign or IATA commercial form, numeric part compared
as an integer) arrives in [MVT - 3 h, MVT + 1 h]; OOOI = labels QP/QQ/QR/QS (ARINC 620 out/off/on/in).
Gate (REVIEW_BRIEF.md, E18): continue only if OOOI covers >= 5% of LIRF departures without NM on these days or >= 1% of all.
"""
import csv
import glob
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl
import zstandard

OOOI = {"QP", "QQ", "QR", "QS"}
icao2iata = {"ITY": "AZ"}
with open("data/ext/openflights_airlines.dat", newline="", encoding="utf-8") as fh:
    for row in csv.reader(fh):
        if len(row) > 7 and re.fullmatch(r"[A-Z]{3}", row[4] or "") and re.fullmatch(r"[A-Z0-9]{2}", row[3] or "") and row[7] == "Y":
            icao2iata.setdefault(row[4], row[3])


def keys_for_callsign(cs: str) -> set[str]:
    m = re.fullmatch(r"([A-Z]{3})0*(\d{1,4})([A-Z]?)", cs or "")
    if not m:
        return set()
    out = {f"{m.group(1)}{int(m.group(2))}"}
    if m.group(1) in icao2iata:
        out.add(f"{icao2iata[m.group(1)]}{int(m.group(2))}")
    return out


def key_for_acars_flight(fl: str) -> str | None:
    m = re.fullmatch(r"([A-Z]{3}|[A-Z0-9]{2})0*(\d{1,4})[A-Z]?", (fl or "").strip().upper())
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def find_acars(obj):
    """Yield dicts that look like an ACARS payload (have 'label')."""
    if isinstance(obj, dict):
        if "label" in obj and ("flight" in obj or "msg_text" in obj or "text" in obj or "tail" in obj or "reg" in obj):
            yield obj
        for v in obj.values():
            yield from find_acars(v)


def messages(day: str):
    for path in sorted(glob.glob(f"data/acars/{day}/*.zst")):
        with open(path, "rb") as fh:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("_adsblol", {}).get("received_at")
                for a in find_acars(rec):
                    yield t, a.get("flight"), a.get("label"), a.get("msg_text") or a.get("text") or ""


def departures(day: str) -> pl.DataFrame:
    d = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    files = ["data/prc-2026-datasets/ranking.parquet"] if day >= "2026" else sorted(glob.glob("data/prc-2026-datasets/training_2025-12-*.parquet"))
    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    df = pl.concat([pl.read_parquet(f, columns=cols) for f in files])
    return df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF")
                     & pl.col("MVT_TIME_UTC_mvt").is_between(d, d + timedelta(days=1), closed="left"))


for day in sys.argv[1:]:
    dep = departures(day)
    want = {}
    for r in dep.iter_rows(named=True):
        for k in keys_for_callsign(r["FLIGHT_mvt"]):
            want.setdefault(k, []).append(r)
    hits, oooi_hits, n_msg, n_rome, oooi_samples = {}, {}, 0, 0, []
    for t, fl, label, text in messages(day):
        n_msg += 1
        if "LIRF" in text or "FCO" in text:
            n_rome += 1
        k = key_for_acars_flight(fl)
        if not k or k not in want or not t:
            continue
        ts = datetime.fromisoformat(t)
        for r in want[k]:
            mvt = r["MVT_TIME_UTC_mvt"]
            if mvt - timedelta(hours=3) <= ts <= mvt + timedelta(hours=1):
                hits.setdefault(r["MVT_ID_mvt"], 0)
                hits[r["MVT_ID_mvt"]] += 1
                if label in OOOI:
                    oooi_hits.setdefault(r["MVT_ID_mvt"], []).append((ts, label, text[:80]))
    numeric = sum(1 for r in dep.iter_rows(named=True) if keys_for_callsign(r["FLIGHT_mvt"]))
    nonm = dep.filter(pl.col("FLIGHT_ID_mvt").is_null())
    nonm_ids = set(nonm["MVT_ID_mvt"].to_list())
    print(f"{day}: messages {n_msg}, mentioning LIRF/FCO {n_rome}; LIRF departures {dep.height} (numeric callsign {numeric}), without NM {nonm.height}")
    print(f"  covered by any message: {len(hits)} ({len(hits) / max(dep.height, 1):.1%}); without NM {len(nonm_ids & hits.keys())}")
    print(f"  covered by OOOI: {len(oooi_hits)} ({len(oooi_hits) / max(dep.height, 1):.1%}); without NM {len(nonm_ids & oooi_hits.keys())} "
          f"({len(nonm_ids & oooi_hits.keys()) / max(len(nonm_ids), 1):.1%})")
    if day < "2026":
        blk = {r["MVT_ID_mvt"]: r["BLOCK_TIME_UTC_mvt"] for r in dep.iter_rows(named=True)}
        for mid, ev in list(oooi_hits.items())[:10]:
            print("   ", [(lab, round((ts - blk[mid]).total_seconds())) for ts, lab, _ in ev], "s vs BLOCK")
    gate = len(nonm_ids & oooi_hits.keys()) >= 0.05 * max(len(nonm_ids), 1) or len(oooi_hits) >= 0.01 * dep.height
    print(f"  gate on this day: {'PASS' if gate else 'FAIL'}")
