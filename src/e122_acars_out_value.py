"""E122 (25.09, hypothesis 55): what would ACARS OUT messages be worth on the covered hubs?

For labelled 2025 days scanned from airframes.io (src/airframes_oooi_scan.py -> data/airframes/oooi_<day>.jsonl):
match OUT messages (labels QP/QA/QT) to our departures by airport + IATA flight number (ICAO callsign mapped via
OpenFlights, numeric part compared as an integer) within +-3 h of BLOCK, parse the OUT time (HHMM after the ICAO pair),
and report: matches per airport, OUT - BLOCK offset (median, IQR, share within 60 s), whether the ADS-B extractor
already saw the off-block on those rows (offblock2_<day>.parquet, t_move), and the current OOF residual of the E106
chain on the matched rows split by seen / not seen (data/cv_preds_e106_air). The SSE bound follows from the last two.

    python src/e122_acars_out_value.py 2025-07-02 2025-07-15 2025-01-15
"""
import csv
import glob
import json
import os
import re
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
icao2iata = {"ITY": "AZ"}
with open("data/ext/openflights_airlines.dat", newline="", encoding="utf-8") as fh:
    for row in csv.reader(fh):
        if len(row) > 4 and row[4] and row[4] != r"\N" and row[3] and row[3] != r"\N":
            icao2iata.setdefault(row[4], row[3])


def iata_forms(callsign: str):
    m = re.match(r"^([A-Z]{3})(\d{1,4})[A-Z]{0,2}$", callsign or "")
    if not m:
        return set()
    out = {f"{m.group(1)}{int(m.group(2))}"}
    if m.group(1) in icao2iata:
        out.add(f"{icao2iata[m.group(1)]}{int(m.group(2))}")
    return out


oof = {}
for p in sorted(glob.glob("data/cv_preds_e106_air/fold_*.npz")):
    d = np.load(p)
    for i, pr, y in zip(d["mvt_id"].astype(np.int64), d["pred"], d["y"]):
        oof[int(i)] = (float(pr), float(y))

tot = {"matched": 0, "seen": 0, "sse_seen": 0.0, "sse_unseen": 0.0, "n_unseen": 0}
for day in sys.argv[1:]:
    msgs = [json.loads(l) for l in open(f"data/airframes/oooi_{day}.jsonl")]
    outs = [m for m in msgs if m["label"] in ("QP", "QA", "QT") and m.get("dep_parsed") in HUBS and m.get("flightNumber")]
    tr = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_mvt", "t_block", "t_mvt", "TAXITIME_SEC_mvt"])
    t0 = pl.datetime(*[int(x) for x in day.split("-")], time_zone="UTC").dt.epoch("s")
    dep = tr.filter((pl.col("split") == "train") & (pl.col("t_mvt") >= pl.lit(t0)) & (pl.col("t_mvt") < pl.lit(t0) + 86400))
    ob = pl.read_parquet(f"data/adsblol/offblock2_{day}.parquet", columns=["MVT_ID_mvt", "t_move"]) if os.path.exists(f"data/adsblol/offblock2_{day}.parquet") else None
    if ob is not None:
        dep = dep.join(ob, on="MVT_ID_mvt", how="left")
    by_key = {}
    for r in dep.iter_rows(named=True):
        for f in iata_forms(r["FLIGHT_mvt"]):
            by_key.setdefault((r["ADEP_mvt"], f), []).append(r)
    per_ap, offsets, seen_flags, res = {}, [], [], {"seen": [], "unseen": []}
    for m in outs:
        fn = m["flightNumber"]; mm = re.match(r"^([A-Z0-9]{2})0*(\d{1,4})$", fn)
        key = (m["dep_parsed"], f"{mm.group(1)}{int(mm.group(2))}") if mm else None
        cands = by_key.get(key, []) if key else []
        ts = m["timestamp"]; tsec = int(pl.Series([ts]).str.to_datetime(time_zone="UTC").dt.epoch("s")[0])
        cands = [r for r in cands if abs(r["t_block"] - tsec) <= 3 * 3600]
        if not cands:
            continue
        r = min(cands, key=lambda r: abs(r["t_block"] - tsec))
        txt = (m.get("text") or "").strip()
        hm = re.match(r"^[A-Z]{4}[A-Z]{4}(\d{2})(\d{2})", txt)
        if hm:
            out_t = t0 if False else None
            import datetime as dt
            base = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            out_t = base.replace(hour=int(hm.group(1)), minute=int(hm.group(2))).timestamp()
            offsets.append(out_t - r["t_block"])
        ap = r["ADEP_mvt"]; per_ap[ap] = per_ap.get(ap, 0) + 1
        seen = r.get("t_move") is not None; seen_flags.append(seen)
        if r["MVT_ID_mvt"] in oof:
            pr, y = oof[r["MVT_ID_mvt"]]; res["seen" if seen else "unseen"].append((pr - y) ** 2)
    offsets = np.array(offsets)
    print(f"\n== {day}: OUT msgs at hubs {len(outs)}, matched {sum(per_ap.values())} {per_ap}")
    if len(offsets):
        print(f"   OUT - BLOCK: median {np.median(offsets):+.0f} s, IQR [{np.percentile(offsets,25):+.0f}, {np.percentile(offsets,75):+.0f}], within 60 s {np.mean(np.abs(offsets)<=60):.2f}, within 300 s {np.mean(np.abs(offsets)<=300):.2f} (n={len(offsets)})")
    print(f"   ADS-B off-block already seen on matched rows: {sum(seen_flags)}/{len(seen_flags)}")
    for k in ("seen", "unseen"):
        if res[k]:
            print(f"   current OOF RMSE on matched rows, {k}: {np.sqrt(np.mean(res[k])):.0f} s (n={len(res[k])})")
    tot["matched"] += sum(per_ap.values()); tot["seen"] += sum(seen_flags)
    tot["sse_unseen"] += sum(res["unseen"]); tot["n_unseen"] += len(res["unseen"]); tot["sse_seen"] += sum(res["seen"])
print("\nTOTAL", tot)
if tot["n_unseen"]:
    # bound: on unseen matched rows, replace the current error by the seen-row error level (85 s per journal 13)
    gain = tot["sse_unseen"] - tot["n_unseen"] * 85.0 ** 2
    print(f"SSE bound per day-equivalent on unseen matched rows: {gain / len(sys.argv[1:]):.3e} s^2/day -> over 62 ranking days / 344 841 rows: {np.sqrt(239.9169**2) - np.sqrt(max(239.9169**2 - gain / len(sys.argv[1:]) * 62 / 344841, 0)):.3f} s")
