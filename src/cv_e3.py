"""E3: day-context features, 6 folds of month pairs (k, k+6); control = same features without day_*.

    FEATURES=data/features_v8.parquet uv run python src/cv_e3.py
Writes data/cv_e3.json. Decision rule (pre-registered): accept if RMSE on no-NM rows with y <= 7200
improves by >= 3 s in at least 4 of 6 folds.
"""
import json
import os
import subprocess
import sys

import numpy as np

PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
RUNNER = r'''
import os, sys, json
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
sys.path.insert(0, "src")
import train as T
months = [int(m) for m in sys.argv[1].split(",")]
df = pl.read_parquet(T.FEATURES_PATH)
tr = df.filter(pl.col("split") == "train")
fit, val = tr.filter(~pl.col("month").is_in(months)), tr.filter(pl.col("month").is_in(months))
yv = val[T.TARGET].to_numpy()
pred, parts, best, _ = T.fit_predict(fit, val, T.feature_columns(fit), None, val_y=yv)
nonm = val["FLIGHT_ID_mvt"].is_null().to_numpy()
r = lambda m: float(np.sqrt(np.mean((pred[m] - yv[m]) ** 2)))
out = {"full": r(np.ones(len(yv), bool)), "nonm_clean": r(nonm & (yv <= 7200)), "nm_clean": r(~nonm & (yv <= 7200)), "best": best}
print("RESULT " + json.dumps(out))
'''

results = {}
for a, b in PAIRS:
    for variant, drop in [("control", "day_"), ("day", "")]:
        env = {**os.environ, "DROP_PREFIX": drop, "POLARS_MAX_THREADS": "10"}
        proc = subprocess.run([sys.executable, "-c", RUNNER, f"{a},{b}"], env=env, capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(proc.stderr[-2000:]); sys.exit(1)
        results[f"{a}+{b}:{variant}"] = json.loads(line[0][7:])
        print(a, b, variant, results[f"{a}+{b}:{variant}"], flush=True)
        json.dump(results, open("data/cv_e3.json", "w"), indent=1)

wins = 0
for a, b in PAIRS:
    c, d = results[f"{a}+{b}:control"], results[f"{a}+{b}:day"]
    delta = d["nonm_clean"] - c["nonm_clean"]
    wins += delta <= -3
    print(f"fold {a}+{b}: no-NM clean {c['nonm_clean']:.1f} -> {d['nonm_clean']:.1f} ({delta:+.1f}); full {c['full']:.1f} -> {d['full']:.1f} ({d['full'] - c['full']:+.1f}); NM clean {d['nm_clean'] - c['nm_clean']:+.2f}")
print("DECISION:", "ACCEPT" if wins >= 4 else "REJECT", f"({wins}/6 folds with <= -3 s)")
