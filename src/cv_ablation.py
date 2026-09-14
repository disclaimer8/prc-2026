"""Ablation CV on 6 folds of month pairs (k, k+6) with a shared control.

    FEATURES=data/features_v9.parquet uv run python src/cv_ablation.py
Variants are defined by column prefixes left out of the model. Decision rule (pre-registered, per treatment):
accept if RMSE on NM rows with y <= 7200 improves by >= 0.5 s in >= 4 of 6 folds AND the mean full-RMSE delta <= 0.
"""
import json
import os
import subprocess
import sys

PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
VARIANTS = {"control": "day_,tq_,wxr_", "E9_p10": "day_,wxr_", "E10_wind": "day_,tq_"}
RUNNER = open(os.path.join(os.path.dirname(__file__), "cv_e3.py")).read().split("RUNNER = r'''")[1].split("'''")[0]
out_path = "data/cv_ablation.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}

for a, b in PAIRS:
    for variant, drop in VARIANTS.items():
        key = f"{a}+{b}:{variant}"
        if key in results:
            continue
        env = {**os.environ, "DROP_PREFIX": drop, "POLARS_MAX_THREADS": "10"}
        proc = subprocess.run([sys.executable, "-c", RUNNER, f"{a},{b}"], env=env, capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(proc.stderr[-2000:]); sys.exit(1)
        results[key] = json.loads(line[0][7:])
        print(key, results[key], flush=True)
        json.dump(results, open(out_path, "w"), indent=1)

for treat in [v for v in VARIANTS if v != "control"]:
    wins, full_deltas = 0, []
    for a, b in PAIRS:
        c, t = results[f"{a}+{b}:control"], results[f"{a}+{b}:{treat}"]
        d_nm = t["nm_clean"] - c["nm_clean"]
        d_full = t["full"] - c["full"]
        wins += d_nm <= -0.5
        full_deltas.append(d_full)
        print(f"{treat} fold {a}+{b}: NM clean {d_nm:+.2f}  full {d_full:+.2f}")
    mean_full = sum(full_deltas) / len(full_deltas)
    print(f"DECISION {treat}:", "ACCEPT" if wins >= 4 and mean_full <= 0 else "REJECT", f"(NM wins {wins}/6, mean full {mean_full:+.2f})")
