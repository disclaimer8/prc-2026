"""What each 17.09 upload changed on the leaderboard, one line per candidate (16.09).

    .venv/bin/python src/score_attribution.py

Reads the scores the queue logs ("SCORE <version> Succeeded <rmse>") and prints the difference against v30, the
submission that is our current best, so each candidate is read as the effect of the one change it carries.
Candidates of 17.09: v45 = v26 + the E42 stage; v50 = the E32 features (re-decided as E47) + the E36 stage;
v55 = the E46 trust coefficient + the E36 stage; v60 = E32 and E46 together + the E36 stage.
"""
import glob
import re

WHAT = {
    30: "v26 + stand-geometry stage E36 (current best)",
    45: "v26 + per-airport stage E42",
    50: "ADS-B features in the model (E32/E47) + stage E36",
    55: "per-airport trust coefficient (E46) + stage E36",
    60: "E32 and E46 together + stage E36",
}
scores = {}
for f in glob.glob("data/*.log"):
    for m in re.finditer(r"^SCORE (\d+) Succeeded ([\d.]+)", open(f, errors="ignore").read(), re.M):
        scores[int(m.group(1))] = float(m.group(2))
base = scores.get(30)
print(f"{'version':>8}  {'RMSE':>9}  {'vs v30':>8}  what it carries")
for v in sorted(WHAT):
    s = scores.get(v)
    if s is None:
        print(f"{'v%d' % v:>8}  {'not yet':>9}  {'':>8}  {WHAT[v]}")
    else:
        print(f"{'v%d' % v:>8}  {s:9.4f}  {(s - base) if base else 0:+8.2f}  {WHAT[v]}")
