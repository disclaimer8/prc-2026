"""Hypothesis 53 (25.09): how many OUT/OFF (OOOI) ACARS messages does airframes.io hold for departures from our ten hubs?

Scans one UTC day of the public airframes.io API (no key needed, 60 req/min, 100 rows/req, only the `labels`
filter is honoured) for the OOOI labels, parses the departure ICAO out of the message text (the QP/QQ text starts
with the ICAO pair, e.g. "EDFHGMMW0643 225" = EDFH -> GMMW, OUT 06:43) and counts per hub. Windows are split
until no window returns the 100-row cap. Output: data/airframes/oooi_<day>.jsonl and a per-hub summary.

    python src/airframes_oooi_scan.py 2026-07-02
"""
import collections
import json
import re
import sys
import time
import urllib.parse
import urllib.request

HUBS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
LABELS = "QA,QB,QP,QQ,QE,QF,QT"  # OUT/OFF families (ARINC 620): QA out/return-in, QB off, QP out report, QQ off report, QE/QF off, QT out
H = {"User-Agent": "Mozilla/5.0 (prc-2026 research probe)"}
day = sys.argv[1]
labels = sys.argv[2] if len(sys.argv) > 2 else LABELS


def get(since, until):
    u = "https://api.airframes.io/messages?" + urllib.parse.urlencode(dict(since=since, until=until, per_page=100, labels=labels))
    for attempt in range(5):
        try:
            resp = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60)
            return json.load(resp)
        except Exception as e:  # 429 or transient
            time.sleep(5 * (attempt + 1))
    return []


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


t0 = int(time.mktime(time.strptime(day, "%Y-%m-%d"))) - time.timezone
t0 = int(time.mktime(time.strptime(day + " UTC", "%Y-%m-%d %Z"))) if False else t0
import calendar
t0 = calendar.timegm(time.strptime(day, "%Y-%m-%d"))
stack = [(t0 + i * 600, t0 + (i + 1) * 600) for i in range(144)]
rows, req = [], 0
while stack:
    a, b = stack.pop(0)
    d = get(iso(a), iso(b)); req += 1
    time.sleep(1.02)
    if len(d) >= 100 and b - a > 2:
        m = (a + b) // 2
        stack[:0] = [(a, m), (m, b)]
        continue
    rows += d
seen = set()
out = open(f"data/airframes/oooi_{day}.jsonl", "w")
by_hub = collections.Counter(); by_hub_label = collections.Counter(); stations = collections.Counter(); labs = collections.Counter()
for x in rows:
    if x["id"] in seen:
        continue
    seen.add(x["id"])
    txt = (x.get("text") or "")
    m = re.match(r"^([A-Z]{4})([A-Z]{4})", txt.strip())
    dep = x.get("departingAirport") or (m.group(1) if m else None)
    rec = {k: x.get(k) for k in ("id", "timestamp", "tail", "flightNumber", "label", "text", "departingAirport", "destinationAirport", "frequency")}
    rec["dep_parsed"] = dep; rec["station"] = (x.get("station") or {}).get("nearestAirportIcao")
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    labs[x.get("label")] += 1
    if dep in HUBS:
        by_hub[dep] += 1; by_hub_label[(dep, x.get("label"))] += 1; stations[(dep, rec["station"])] += 1
out.close()
print(f"{day}: requests {req}, OOOI-family messages {len(seen)}, labels {labs.most_common()}")
print("per hub (departure parsed from text):", {h: by_hub.get(h, 0) for h in HUBS})
print("per hub x label:", sorted(by_hub_label.items()))
print("per hub x receiving station:", sorted(stations.items(), key=lambda kv: -kv[1])[:20])
