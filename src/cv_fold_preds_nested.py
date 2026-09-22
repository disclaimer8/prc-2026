"""Leak-free fold predictions: rounds are chosen on inner months, never on the months being scored (16.09).

    FEATURES=... DROP_PREFIX=... CV_OUT=data/cv_preds_nested [PER_AIRPORT=1] .venv/bin/python src/cv_fold_preds_nested.py
Why: src/cv_fold_preds.py passes the fold's own validation months to fit_predict, which early-stops on them. Control and
candidate share that, so most comparisons stay fair — but a candidate that early-stops MORE times (E38: one model per
airport, ten separate stops) gets more of the val months than the control does. E38 won CV by 10.98 s and then lost 3.79 s
on the leaderboard (v43 256.22 vs v30 252.43); this script checks whether the leak explains it.
Scheme per fold (a, b): inner = months a+1, b+1 of the fit set; rounds are chosen there, the model is refit on all 10 fit
months with rounds x 1.1 (as submissions do) and applied to months a, b.
"""
import os
import subprocess
import sys

ALL_PAIRS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
# FOLDS=1,3,5 runs only those month pairs, so the six independent folds can be split across machines
# (they never share state; the comparison scripts read whatever files are present and count them).
PAIRS = ([p for p in ALL_PAIRS if p[0] in {int(x) for x in os.environ["FOLDS"].split(",")}]
         if os.environ.get("FOLDS") else ALL_PAIRS)
RUNNER = r'''
import os, sys
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np, polars as pl
sys.path.insert(0, "src")
import train as T
a, b = [int(x) for x in sys.argv[1].split(",")]
inner = [a % 12 + 1, b % 12 + 1]
per_airport = os.environ.get("PER_AIRPORT") == "1"
df = pl.read_parquet(T.FEATURES_PATH)
tr = df.filter(pl.col("split") == "train")
del df
fit_all, val_all = tr.filter(~pl.col("month").is_in([a, b])), tr.filter(pl.col("month").is_in([a, b]))
# E96 (22.09): FIT_FRAC=f keeps a random f of the fit rows, same seed for every fold. Measures the learning
# curve -- how much the model still gains from more data of the kind it already has. Default 1.0 = unchanged.
frac = float(os.environ.get("FIT_FRAC", "1"))
if frac < 1.0:
    fit_all = fit_all.sample(fraction=frac, seed=42)
    print(f"FIT_FRAC={frac}: fit rows {fit_all.height}", flush=True)
groups = sorted(tr["ADEP_mvt"].unique().to_list()) if per_airport else [None]
ids, preds, regs, ys, probas, vals = [], [], [], [], [], []
for ap in groups:
    f = fit_all if ap is None else fit_all.filter(pl.col("ADEP_mvt") == ap)
    v = val_all if ap is None else val_all.filter(pl.col("ADEP_mvt") == ap)
    if v.height == 0:
        continue
    inner_fit, inner_val = f.filter(~pl.col("month").is_in(inner)), f.filter(pl.col("month").is_in(inner))
    _, _, best, _ = T.fit_predict(inner_fit, inner_val, T.feature_columns(inner_fit), None, val_y=inner_val[T.TARGET].to_numpy())
    rounds = {k: int(x * 1.1) for k, x in best.items()}
    pred, parts, _, _ = T.fit_predict(f, v, T.feature_columns(f), rounds)
    ids.append(v["MVT_ID_mvt"].to_numpy()); preds.append(pred); regs.append(parts["reg"]); ys.append(v[T.TARGET].to_numpy())
    probas.append(parts["proba"]); vals.append(parts["values"])
    print(ap or "all", rounds, float(np.sqrt(np.mean((pred - ys[-1]) ** 2))), flush=True)
out = os.environ.get("CV_OUT", "data/cv_preds_nested")
np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=np.concatenate(ids), pred=np.concatenate(preds), reg=np.concatenate(regs), y=np.concatenate(ys),
         proba=np.concatenate(probas).astype(np.float32), values=np.concatenate(vals).astype(np.float32))
print("saved", a, b, float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(ys)) ** 2))))
'''
out = os.environ.get("CV_OUT", "data/cv_preds_nested")
os.makedirs(out, exist_ok=True)
failed: list[tuple[int, int]] = []
for a, b in PAIRS:
    if os.path.exists(f"{out}/fold_{a}_{b}.npz"):
        continue
    proc = subprocess.run([sys.executable, "-c", RUNNER, f"{a},{b}"], env={**os.environ, "POLARS_MAX_THREADS": "10"}, capture_output=True, text=True)
    print(proc.stdout.strip()[-600:] or proc.stderr[-2000:], flush=True)
    # A fold can train fine and die while saving; stdout is then non-empty, so printing "stdout or stderr"
    # hides the traceback and the comparison silently runs on fewer folds (E53, 16.09: fold 3+9 vanished
    # this way and the result was reported on five folds). Check the artefact, not the exit code.
    if not os.path.exists(f"{out}/fold_{a}_{b}.npz"):
        print(f"FOLD {a}_{b} PRODUCED NO FILE (exit {proc.returncode})\n--- stderr tail ---\n{proc.stderr[-2000:]}", flush=True)
        failed.append((a, b))

if failed:
    raise SystemExit(f"{len(failed)} of {len(PAIRS)} folds produced no file: {failed} — the comparison would silently use fewer folds")
print(f"all {len(PAIRS)} folds saved in {out}", flush=True)
