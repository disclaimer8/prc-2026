"""Cross-platform (Windows too) downloader+extractor for one adsb.lol globe_history day (ODbL 1.0), 16.09.

    python src/adsb_fetch_day.py 2025-07-15
Streams the day's tar parts straight into src/adsb_extract.py (no archive on disk), choosing the replica from the archive's
own PREFERRED_RELEASES.txt (cached in data/adsblol/PREFERRED_<year>.txt), then writes offblock2/3/5 for that day.
"""
import os
import subprocess
import sys
import urllib.request

day = sys.argv[1]
year = day[:4]
os.makedirs("data/adsblol", exist_ok=True)
pref = f"data/adsblol/PREFERRED_{year}.txt"
if not os.path.exists(pref) or os.path.getsize(pref) == 0:
    url = f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
    with urllib.request.urlopen(url, timeout=120) as r, open(pref, "wb") as f:
        f.write(r.read())
tag = f"/v{day.replace('-', '.')}-planes-readsb-"
urls = next((l.strip().split(",") for l in open(pref) if tag in l), None)
if not urls:
    sys.exit(f"NO RELEASE {day}")
print("preferred", urls[0].rsplit("/", 2)[1], flush=True)
extract = subprocess.Popen([sys.executable, "src/adsb_extract.py", day], stdin=subprocess.PIPE)
for u in urls:
    with urllib.request.urlopen(u.strip(), timeout=600) as r:
        while chunk := r.read(1 << 22):
            extract.stdin.write(chunk)
extract.stdin.close()
if extract.wait() != 0:
    sys.exit(f"EXTRACT FAILED {day}")
for script in ("src/adsb_offblock_v2.py", "src/adsb_offblock_v3.py", "src/adsb_offblock_v5.py"):
    subprocess.run([sys.executable, script, day], check=True, stdout=subprocess.DEVNULL)
print(f"{day} done", flush=True)
