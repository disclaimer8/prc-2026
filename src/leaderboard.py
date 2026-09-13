"""Print our submissions and current rank from the public leaderboard API."""
import json
import sys
import urllib.parse
import urllib.request

BASE = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
TEAM = "gentle-igloo"


def fetch_all():
    items, cursor = [], None
    while True:
        url = BASE + (f"?cursor={urllib.parse.quote(cursor)}" if cursor else "")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        page = json.load(urllib.request.urlopen(req, timeout=30))
        items += page["items"]
        cursor = page.get("nextCursor")
        if not cursor or not page["items"]:
            return items


items = fetch_all()
best = {}
for x in items:
    if x.get("score") is not None and (x["teamName"] not in best or x["score"] < best[x["teamName"]]["score"]):
        best[x["teamName"]] = x
ranked = sorted(best.values(), key=lambda x: x["score"])
ours = sorted((x for x in items if x["teamName"] == TEAM), key=lambda x: x["filename"])
for x in ours:
    print(x["filename"], x["score"], x["processedAt"])
if TEAM in best:
    pos = [x["teamName"] for x in ranked].index(TEAM) + 1
    print(f"rank {pos}/{len(ranked)} best={best[TEAM]['score']} leader={ranked[0]['score']}")
sys.exit(0 if ours else 1)
