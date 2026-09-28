"""Journal 130: compare the end-to-end rebuild (src/hz_final_e2e.sh, WORK=/srv/prc-final-e2e) with the board line v240 -> v447.

    .venv/bin/python src/e2e_compare.py [WORK]

Each built file is compared with its reference: equal bit for bit, or rows changed / RMS / max |d| / mean d.
Missing files are reported, not fatal (the run may still be going).
"""
import os
import sys

import numpy as np
import polars as pl

W = sys.argv[1] if len(sys.argv) > 1 else "/srv/prc-final-e2e"
M = "/root/prc-2026"
T = "TAXITIME_SEC_mvt"
PAIRS = [(900, 240, "base (train.py submit)"), (901, 441, "blend, 10 CatBoost seeds"),
         (912, 442, "TabPFN rows of v361"), (913, 443, "E14"), (914, 444, "E141 stage x5"), (915, 445, "E64 x5"),
         (916, 446, "E81 reuse"), (917, 447, "LOBT = FINAL, E81 reuse"),
         (922, 442, "tail 2: TabPFN"), (925, 445, "tail 2: E64"), (926, 446, "tail 2: E81 rebuild"), (927, 447, "tail 2: FINAL, E81 rebuild")]
PARTNERS = [(f"{W}/e81/submissions/gentle-igloo_v944.parquet", 44, "E81 partner v44 rebuilt"),
            (f"{W}/e81/submissions/gentle-igloo_v960.parquet", 60, "E81 partner v60 rebuilt")]


def load(path):
    return pl.read_parquet(path).sort("MVT_ID_mvt") if os.path.exists(path) else None


def cmp(new_path, ref_path, name):
    a, b = load(new_path), load(ref_path)
    if a is None or b is None:
        print(f"{name:34s} missing {'new' if a is None else 'ref'}: {new_path if a is None else ref_path}")
        return
    if not a["MVT_ID_mvt"].equals(b["MVT_ID_mvt"]):
        print(f"{name:34s} MVT_ID sets differ ({a.height} vs {b.height})")
        return
    if a.equals(b):
        print(f"{name:34s} identical")
        return
    d = a[T].cast(float).to_numpy() - b[T].cast(float).to_numpy()
    print(f"{name:34s} changed {int((d != 0).sum()):6d}, RMS {np.sqrt((d ** 2).mean()):.4f} s, max |d| {np.abs(d).max():.0f} s, mean {d.mean():+.4f} s")


for v, r, name in PAIRS:
    cmp(f"{W}/submissions/gentle-igloo_v{v}.parquet", f"{M}/submissions/gentle-igloo_v{r}.parquet", f"v{v} vs v{r} {name}")
for p, r, name in PARTNERS:
    cmp(p, f"{M}/submissions/gentle-igloo_v{r}.parquet", f"{name}")
for new, ref, key in ((f"{W}/submissions/lgb_reg_seeds_v900.npz", f"{M}/submissions/lgb_reg_seeds_v240.npz", "reg"),
                      (f"{W}/submissions/parts_v901.npz", f"{M}/submissions/parts_v441.npz", "pred")):
    if os.path.exists(new) and os.path.exists(ref):
        x, y = np.load(new), np.load(ref)
        same_ids = np.array_equal(x["mvt_id"], y["mvt_id"])
        d = x[key] - y[key] if same_ids else None
        print(f"{os.path.basename(new):34s} vs {os.path.basename(ref)}: ids equal {same_ids}"
              + (f", {key} max |d| {np.abs(d).max():.6f}" if d is not None else ""))
