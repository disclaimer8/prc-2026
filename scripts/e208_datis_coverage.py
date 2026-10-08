"""E208: bounded CC0 D-ATIS availability, never labels or model fitting.

inventory = release metadata only; pack = projected departure metadata + verified
old scan reuse; run = separately preregistered bounded archive scan and coverage.
"""
from __future__ import annotations
import argparse
import bisect
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

PROTOCOL = "E208-CC0-DATIS-fixed16days-label-free-60min-v1"
BASE = Path("analysis/e208_datis_20261008")
RESEARCH = Path("research_notes/Новые сигналы PRC 8 октября")
HUBS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH")
DAYS = tuple(f"{year}-{month:02}-{day:02}" for day in (1, 8, 15, 22)
             for year, month in ((2025, 1), (2025, 7), (2026, 1), (2026, 7)))
CAP = 2 * 1024**3
ASSET_CAP = 32 * 1024**2
LINE_CAP = 1024**2
DECOMPRESSED_CAP = 512 * 1024**2
WALL = 2700
FRESH_SECONDS = 3600
LICENSE_URL = "https://www.adsb.lol/docs/open-data/aircraft-data-links/"
HEADER = re.compile(r"\b(" + "|".join(HUBS) + r")\s+(ARR|DEP)(?:/(ARR|DEP))?\s+ATIS\s+([A-Z])\s+([0-9]{4})Z\b")
HEADER_ALT = re.compile(r"\b(" + "|".join(HUBS) + r")\s+ATIS\s+(ARR|DEP)(?:/(ARR|DEP))?\s+([A-Z])\s+([0-9]{4})Z\b")
HUB_RE = re.compile(r"\b(" + "|".join(HUBS) + r")\b")
PROJECTION = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def utc(value):
    d = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timezone absent")
    return d.astimezone(dt.timezone.utc)


def clock_before(hhmm, received, max_age=6 * 3600):
    """Nearest nonfuture clock, only within six hours; future clocks are ambiguous."""
    if not re.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9]", hhmm):
        raise ValueError("invalid clock")
    candidate = received.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0, microsecond=0)
    if candidate > received:
        candidate -= dt.timedelta(days=1)
    if (received - candidate).total_seconds() > max_age:
        raise ValueError("ambiguous or old clock")
    return candidate


def operational(text, issued):
    closures = []
    # Bounded conservative grammar: a timing cue followed within 65 characters
    # by HHMM/HH:MM suppresses all current operational claims in this bulletin.
    # This deliberately also drops mixed current/scheduled statements; it does
    # not resolve time intervals or claim full coverage of aviation phrasing.
    scheduled = bool(re.search(r"\b(?:FROM|UNTIL|TILL|EXPECT\w*|CHANGE TO|AT|BTN|BETWEEN)\b.{0,65}\b(?:[01][0-9]|2[0-3]):?[0-5][0-9](?:Z| HOURS)?\b", text))
    for kind, prefix, identifier in (
        ("runway", r"RWY|RUNWAY", r"[0-3][0-9][LRC]?"),
        ("taxiway", r"TWY|TAXIWAY", r"[A-Z](?:[A-Z]?[0-9]{1,2}[A-Z]?)?"),
        ("exit", r"HIGH SPEED TURN OFF|RAPID EXIT|EXITS?", r"[A-Z](?:[A-Z]?[0-9]{1,2}[A-Z]?)?"),
    ):
        names = identifier + r"(?:(?:\s+(?:AND\s+)?|\s*[,/]\s*)" + identifier + r")*"
        rx = r"\b(?:" + prefix + r")\s*(" + names + r")\s+(?:IS |ARE )?(?:CLSD|CLOSED|UNAVAILABLE)\b"
        for m in re.finditer(rx, text):
            ids = re.findall(r"\b" + identifier + r"\b", m[1])
            closures.extend({"kind": kind, "object": item, "state": "closed"} for item in ids)
    conditions = []
    for m in re.finditer(r"\b(?:RWY|RUNWAY)\s*([0-3][0-9][LRC]?).{0,45}?\bCODES?\s*:?\s*([0-6]/[0-6]/[0-6])", text):
        conditions.append({"runway": m[1], "codes": m[2], "observed_at": None})
    if "/RSCD " in text:
        for m in re.finditer(r"\b([0-3][0-9][LRC]?)\s+([0-6]{3})\s+([0-9]{4})\s*Z\b", text.split("/RSCD ", 1)[1].split("/OPS", 1)[0]):
            try:
                observed = clock_before(m[3], issued).isoformat()
            except ValueError:
                observed = None
            conditions.append({"runway": m[1], "codes": "/".join(m[2]), "observed_at": observed,
                               "clock_status": "resolved" if observed else "ambiguous"})
    # A scheduled statement is not silently promoted to present operational state.
    if scheduled:
        closures, conditions = [], []
    return {"closures": closures,
            "runway_conditions": conditions,
            "scheduled_or_future_text": scheduled,
            "has_operational_state": bool(closures or any(c.get("clock_status") != "ambiguous" for c in conditions))}


def parse_message(message):
    if message.get("label") != "A9":
        return {"status": "not_a9"}
    text = re.sub(r"\s+", " ", str(message.get("text") or message.get("msg_text") or "")).strip()
    if len(text) > 16384:
        return {"status": "oversize_bulletin"}
    m = HEADER.search(text) or HEADER_ALT.search(text)
    if not m:
        return {"status": "unrecognised_header_or_role"}
    # Only routing prefixes may precede the actual bulletin, never clearance prose.
    prefix = text[:m.start()]
    if len(prefix) > 45 or re.search(r"[^A-Z0-9./ _-]", prefix) or " " in prefix.strip():
        return {"status": "non_bulletin_prefix"}
    if str(message.get("assembly_status", message.get("assstat", ""))).lower() != "complete":
        return {"status": "assembly_not_complete"}
    try:
        received = utc(message["received_at"])
        issued = clock_before(m[5], received)
    except (ValueError, TypeError, KeyError):
        return {"status": "ambiguous_timestamp"}
    # Persist only the bulletin beginning at ICAO, without receiver/tail/flight IDs.
    bulletin = text[m.start():]
    if "@" in bulletin or re.search(r"\b(?:TEL|PHONE|EMAIL|PAX|PASSENGER)\b", bulletin):
        return {"status": "unexpected_personal_content"}
    roles = sorted({m[2], *([m[3]] if m[3] else [])})
    return {"status": "accepted", "airport": m[1], "roles": roles, "designator": m[4],
            "received_at": received.isoformat(), "issued_at": issued.isoformat(),
            "expires_at": (issued + dt.timedelta(seconds=FRESH_SECONDS)).isoformat(),
            "text": bulletin, "text_sha256": hashlib.sha256(bulletin.encode()).hexdigest(),
            **operational(bulletin, issued)}


def walk(obj, depth=0):
    if depth > 24:
        raise ValueError("payload nesting limit")
    if isinstance(obj, dict):
        if "label" in obj and ("text" in obj or "msg_text" in obj):
            yield obj
        for value in obj.values():
            yield from walk(value, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value, depth + 1)


def candidate(text):
    # Exactly the candidate superset retained in the earlier complete scan.
    return "ATIS" in text and HUB_RE.search(text) is not None


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PRC-E208-public-metadata", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2:
        raise ValueError("metadata response cap")
    return json.loads(data)


def asset_record(asset, repo, day):
    prefix = f"https://github.com/adsblol/{repo}/releases/download/adsblol-adl-{day}/"
    if not asset["browser_download_url"].startswith(prefix):
        raise ValueError("unexpected source URL")
    return {k: asset.get(k) for k in ("id", "name", "size", "digest", "updated_at", "browser_download_url")}


def inventory(root):
    out = root / BASE / "inventory.json"
    if out.exists():
        raise FileExistsError("inventory is immutable")
    records = []
    for day in DAYS:
        repo = "aircraft-data-links-2025" if day < "2026-02-01" else "aircraft-data-links-2026"
        url = f"https://api.github.com/repos/adsblol/{repo}/releases/tags/adsblol-adl-{day}"
        old = root / RESEARCH / f"external_adl_release_correct_{day.replace('-', '')}.json"
        try:
            if old.exists():
                release = read(old)
                assets = release["assets"]
                source = {"local_release_sha256": sha(old), "local_release": str(old.relative_to(root))}
            else:
                release = get_json(url)
                assets = []
                for page in range(1, 11):
                    batch = get_json(release["assets_url"] + f"?per_page=100&page={page}")
                    assets.extend(batch)
                    if len(batch) < 100:
                        break
                else:
                    raise ValueError("asset pagination cap")
                source = {"api_url": url}
            if release["tag_name"] != "adsblol-adl-" + day:
                raise ValueError("release date mismatch")
            normal = [asset_record(a, repo, day) for a in assets]
            if len({a["id"] for a in normal}) != len(normal):
                raise ValueError("duplicate asset IDs")
            records.append({"day": day, "status": "listed", "repo": repo, "release_url": release["html_url"],
                            "published_at": release["published_at"], "assets": normal, **source})
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError) as error:
            records.append({"day": day, "status": "inventory_unavailable", "url": url, "error": str(error)[:200], "assets": []})
        print(json.dumps({"inventory_day": day, "status": records[-1]["status"], "assets": len(records[-1]["assets"])}), flush=True)
    write(out, {"protocol": PROTOCOL, "retrieved_utc": now(), "license": "CC0-1.0 (aircraft-data-links only)",
                "license_url": LICENSE_URL, "days": records,
                "declared_bytes": sum(a["size"] for r in records for a in r["assets"])})


def reuse_assets(root, inv):
    summary = read(root / RESEARCH / "external_archive_scan_summary.json")
    records = [json.loads(line) for line in (root / RESEARCH / "external_archive_scan_events.jsonl").open()]
    reused = {}
    if not summary.get("complete"):
        return reused
    for day in inv["days"]:
        key = day["day"].replace("-", "")
        if key not in summary["manifest"] or day["status"] != "listed":
            continue
        old = root / RESEARCH / f"external_adl_release_correct_{key}.json"
        if sha(old) != summary["manifest"][key]["sha256"]:
            raise ValueError("old release SHA drift")
        matched = [x for x in records if x["date"] == key]
        by_url = {x["url"]: x for x in matched}
        expected = {a["browser_download_url"] for a in day["assets"]}
        if len(by_url) != len(matched) or set(by_url) != expected:
            raise ValueError("old complete scan asset set mismatch")
        for asset in day["assets"]:
            old_record = by_url[asset["browser_download_url"]]
            if old_record["status"] != "ok" or old_record.get("malformed_json_lines") or old_record["actual_bytes"] != asset["size"]:
                raise ValueError("old scan incomplete")
            if asset.get("digest") != "sha256:" + old_record["compressed_sha256"]:
                raise ValueError("old asset digest mismatch")
            output = collections.Counter()
            accepted = []
            for m in old_record.get("hub_atis", []):
                p = parse_message(m)
                output[p["status"]] += 1
                if p["status"] == "accepted":
                    accepted.append(p)
            reused[asset["browser_download_url"]] = {
                "day": day["day"], "url": asset["browser_download_url"], "status": "ok", "reused": True,
                "compressed_sha256": old_record["compressed_sha256"], "actual_bytes": asset["size"],
                "network_bytes": 0, "json_lines": old_record["json_lines"], "payloads": old_record["payloads"],
                "all_A9": old_record["labels"].get("A9", 0), "candidate_messages": len(old_record.get("hub_atis", [])),
                "parse_counts": dict(output), "records": accepted}
    return reused


def registered(root, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full prereg commit required")
    files = ["scripts/e208_datis_coverage.py", "tests/test_e208_datis_coverage.py", str(BASE / "PROTOCOL.md"), str(BASE / "inventory.json")]
    checks = {}
    for name in files:
        frozen = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        checks[name] = hashlib.sha256(frozen).hexdigest()
        if sha(root / name) != checks[name]:
            raise ValueError("registered source drift: " + name)
    return checks


def pack(root, commit):
    import polars as pl
    checks = registered(root, commit)
    output = root / BASE / "inputs"
    output.mkdir(exist_ok=False)
    source = root / "data/prc-2026-datasets"
    paths = sorted(source.glob("training_*.parquet")) + [source / "ranking.parquet"]
    if len(paths) != 13:
        raise ValueError("canonical 12 training + ranking files required")
    # Explicit projection: PHASE is only used to select DEP, never retained.
    f = pl.scan_parquet(paths).select(PROJECTION).filter(pl.col("PHASE_mvt") == "DEP")
    f = f.filter(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").is_in(DAYS)).select(
        pl.col("MVT_ID_mvt").cast(pl.Int64).alias("id"), pl.col("ADEP_mvt").alias("airport"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("mvt")) .collect().sort("id")
    if f["id"].n_unique() != f.height or f.null_count().row(0) != (0, 0, 0) or not set(f["airport"]).issubset(HUBS):
        raise ValueError("canonical metadata invalid")
    f.write_parquet(output / "departures.parquet")
    inv = read(root / BASE / "inventory.json")
    write(output / "reuse.json", reuse_assets(root, inv))
    shutil.copyfile(root / BASE / "inventory.json", output / "inventory.json")
    write(output / "manifest.json", {"protocol": PROTOCOL, "registration_commit": commit, "source_sha256": checks,
        "columns": ["id", "airport", "mvt"], "rows": f.height, "days": DAYS,
        "canonical_projection": PROJECTION, "canonical_source_sha256": {str(p.relative_to(root)): sha(p) for p in paths},
        "research_receipts": {str(p.relative_to(root)): sha(p) for p in [root / RESEARCH / "external_archive_scan_summary.json", root / RESEARCH / "external_archive_scan_events.jsonl"]},
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    print("PACKED", f.height, "label-free departure rows", flush=True)


def scan_asset(task, deadline):
    day, asset = task
    url = asset["browser_download_url"]
    result = {"day": day, "url": url, "reused": False, "network_bytes": 0}
    decoder = None
    timer = None
    feeder = None
    try:
        if asset["size"] > ASSET_CAP:
            raise ValueError("per-asset compressed cap")
        if time.monotonic() >= deadline:
            raise TimeoutError("wall budget")
        request = urllib.request.Request(url, headers={"User-Agent": "PRC-E208-CC0-coverage"})
        body = bytearray()
        with urllib.request.urlopen(request, timeout=max(1, min(15, deadline - time.monotonic()))) as response:
            while len(body) < asset["size"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("wall budget")
                chunk = response.read1(min(65536, asset["size"] - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                result["network_bytes"] += len(chunk)
        digest = hashlib.sha256(body).hexdigest()
        result.update(actual_bytes=len(body), compressed_sha256=digest)
        if len(body) != asset["size"] or (asset.get("digest") and asset["digest"] != "sha256:" + digest):
            raise ValueError("asset size/SHA mismatch")
        decoder = subprocess.Popen([shutil.which("zstd") or "/usr/bin/zstd", "-dc", "-M128MB"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        def feed():
            try:
                decoder.stdin.write(body)
                decoder.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        timer = threading.Timer(max(.1, min(75, deadline - time.monotonic())), decoder.kill)
        timer.start()
        counts = collections.Counter()
        accepted = []
        lines = payloads = a9 = candidates = total = 0
        while True:
            line = decoder.stdout.readline(LINE_CAP + 1)
            if not line:
                break
            total += len(line)
            if len(line) > LINE_CAP or total > DECOMPRESSED_CAP:
                raise ValueError("decoded line/asset cap")
            if time.monotonic() >= deadline:
                raise TimeoutError("wall budget")
            raw = json.loads(line)
            lines += 1
            received = raw.get("_adsblol", {}).get("received_at") if isinstance(raw, dict) else None
            for item in walk(raw):
                payloads += 1
                a9 += item.get("label") == "A9"
                text = re.sub(r"\s+", " ", item.get("msg_text") or item.get("text") or "").strip()
                if not candidate(text):
                    continue
                candidates += 1
                parsed = parse_message({"text": text, "received_at": received, "label": item.get("label"), "assembly_status": item.get("assstat")})
                counts[parsed["status"]] += 1
                if parsed["status"] == "accepted":
                    accepted.append(parsed)
        if decoder.wait(timeout=5) != 0:
            raise ValueError("zstd decode failure")
        result.update(status="ok", json_lines=lines, payloads=payloads, all_A9=a9, candidate_messages=candidates,
                      parse_counts=dict(counts), records=accepted)
    except Exception as error:
        result.update(status="error", error=str(error)[:250], records=[])
    finally:
        if timer:
            timer.cancel()
        if decoder:
            if decoder.poll() is None:
                decoder.kill()
            decoder.wait(timeout=5)
            if feeder:
                feeder.join(timeout=5)
            decoder.stdout.close()
    return result


def unique_messages(records):
    unique = {}
    for m in records:
        key = (m["airport"], tuple(m["roles"]), m["issued_at"], m["text_sha256"])
        if key not in unique or m["received_at"] < unique[key]["received_at"]:
            unique[key] = m
    return list(unique.values())


def state_signature(message):
    return json.dumps({k: message[k] for k in ("closures", "runway_conditions", "scheduled_or_future_text")}, sort_keys=True)


def state_index(messages):
    """O(messages log messages) prefix state; queries are O(log messages)."""
    groups = collections.defaultdict(list)
    for m in unique_messages(messages):
        for role in m["roles"]:
            groups[(m["airport"], role)].append(m)
    result = {}
    for key, rows in groups.items():
        rows.sort(key=lambda m: (m["received_at"], m["issued_at"], m["text_sha256"]))
        times, snapshots, latest_issue, latest = [], [], "", []
        for m in rows:
            if m["issued_at"] > latest_issue:
                latest_issue, latest = m["issued_at"], [m]
            elif m["issued_at"] == latest_issue:
                latest = latest + [m]
            times.append(utc(m["received_at"]).timestamp())
            snapshots.append((len({state_signature(v) for v in latest}) > 1, latest[0]))
        result[key] = (times, snapshots)
    return result


def indexed_match(index, airport, role, timestamp):
    times, snapshots = index.get((airport, role), ([], []))
    position = bisect.bisect_right(times, timestamp) - 1
    if position < 0:
        return "no_prior_atis", None
    conflicting, m = snapshots[position]
    if utc(m["issued_at"]).timestamp() > timestamp or utc(m["received_at"]).timestamp() > timestamp:
        raise ValueError("future state in as-of index")
    if utc(m["expires_at"]).timestamp() <= timestamp:
        return "expired", None
    if conflicting:
        return "conflicting_same_issue", None
    return "covered", m


def match_state(messages, airport, role, timestamp):
    eligible = [m for m in messages if m["airport"] == airport and role in m["roles"]
                and utc(m["received_at"]).timestamp() <= timestamp and utc(m["issued_at"]).timestamp() <= timestamp]
    if not eligible:
        return "no_prior_atis", None
    issue = max(m["issued_at"] for m in eligible)
    latest = [m for m in eligible if m["issued_at"] == issue]
    if utc(latest[0]["expires_at"]).timestamp() <= timestamp:
        return "expired", None
    states = {state_signature(m) for m in latest}
    if len(states) > 1:
        return "conflicting_same_issue", None
    return "covered", min(latest, key=lambda m: m["received_at"])


def coverage(departures, records):
    unique = unique_messages(records)
    index = state_index(unique)
    counts = collections.defaultdict(collections.Counter)
    for flight in departures:
        day = dt.datetime.fromtimestamp(flight["mvt"], dt.timezone.utc).date().isoformat()
        c = counts[(day, flight["airport"])]
        c["flights"] += 1
        covered = operational_covered = False
        for role in ("ARR", "DEP"):
            state, message = indexed_match(index, flight["airport"], role, flight["mvt"])
            c[role + "_" + state] += 1
            if message:
                covered = True
                c[role + "_operational"] += message["has_operational_state"]
                operational_covered |= message["has_operational_state"]
        c["covered_any_role_unique_flights"] += covered
        c["covered_operational_unique_flights"] += operational_covered
        c["no_fresh_atis_unique_flights"] += not covered
        c["midnight_left_censored_flights"] += flight["mvt"] % 86400 < FRESH_SECONDS
    return [{"day": day, "airport": airport, **dict(c)} for (day, airport), c in sorted(counts.items())]


def run(root, commit):
    import pyarrow.parquet as pq
    inputs = root / BASE / "inputs"
    manifest = read(inputs / "manifest.json")
    if manifest["registration_commit"] != commit or manifest["protocol"] != PROTOCOL:
        raise ValueError("input registration mismatch")
    for name, expected in manifest["source_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("frozen source mismatch: " + name)
    for name, expected in manifest["files"].items():
        if sha(inputs / name) != expected:
            raise ValueError("input SHA mismatch: " + name)
    output = root / BASE / "execution"
    output.mkdir(exist_ok=False)
    started = now()
    deadline = time.monotonic() + WALL - 60
    inv, reused = read(inputs / "inventory.json"), read(inputs / "reuse.json")
    tasks, results = [], []
    reserved = 0
    for day in inv["days"]:
        for asset in sorted(day["assets"], key=lambda a: a["name"]):
            url = asset["browser_download_url"]
            if url in reused:
                results.append(reused[url])
            elif not asset["name"].endswith(".jsonl.zst"):
                results.append({"day": day["day"], "url": url, "status": "unsupported_asset", "network_bytes": 0})
            elif asset["size"] > ASSET_CAP or reserved + asset["size"] > CAP:
                results.append({"day": day["day"], "url": url, "status": "compressed_cap_skipped", "network_bytes": 0})
            else:
                tasks.append((day["day"], asset))
                reserved += asset["size"]
    with (output / "assets.jsonl").open("x") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(scan_asset, task, deadline) for task in tasks]
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                print(json.dumps({"assets_done": len(results), "day": r["day"], "status": r["status"],
                                  "accepted": len(r.get("records", []))}), flush=True)
    all_records = [m | {"source_url": r["url"], "source_sha256": r["compressed_sha256"]}
                   for r in results for m in r.get("records", [])]
    flights = pq.read_table(inputs / "departures.parquet", columns=["id", "airport", "mvt"]).to_pylist()
    days = []
    for d in inv["days"]:
        rr = [r for r in results if r["day"] == d["day"]]
        days.append({"day": d["day"], "complete": d["status"] == "listed" and bool(d["assets"]) and len(rr) == len(d["assets"]) and all(r["status"] == "ok" for r in rr),
                     "inventory_status": d["status"], "assets": len(rr), "status_counts": dict(collections.Counter(r["status"] for r in rr)),
                     "raw_candidate_messages": sum(r.get("candidate_messages", 0) for r in rr),
                     "accepted_complete_messages": sum(len(r.get("records", [])) for r in rr)})
    summary = {"protocol": PROTOCOL, "registration_commit": commit, "started_utc": started, "finished_utc": now(),
               "source_license": "CC0-1.0", "license_url": LICENSE_URL, "network_bytes": sum(r["network_bytes"] for r in results),
               "new_download_reserved_bytes": reserved, "days": days, "complete": all(d["complete"] for d in days),
               "accepted_messages": len(all_records), "unique_bulletins": len(unique_messages(all_records)),
               "coverage": coverage(flights, all_records), "target_reads": 0, "fits": 0,
               "freshness_assumption_seconds": FRESH_SECONDS, "missing_means_open": False}
    write(output / "summary.json", summary)
    write(output / "complete.json", {"status": "complete" if summary["complete"] else "partial", "summary_sha256": sha(output / "summary.json"), "accepted_for_model": False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("inventory", "pack", "run"))
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--registration-commit")
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == "inventory":
        inventory(root)
    elif args.command == "pack":
        pack(root, args.registration_commit)
    else:
        run(root, args.registration_commit)


if __name__ == "__main__":
    main()
