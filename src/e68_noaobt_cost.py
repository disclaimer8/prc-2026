"""What abstaining from AOBT_3 costs (18.09).

    .venv/bin/python src/e68_noaobt_cost.py [ctrl_dir] [cand_dir]

Not a gated experiment: there is no threshold to pass, the run exists to put a number on a decision that is
Denys's to make (REVIEW_BRIEF.md, "Правила: проверено по первоисточнику"). Both arms are recomputed by today's
code -- control on data/features_e28.parquet, candidate on data/features_noaobt_e28.parquet, built with
NO_AOBT=1 so that every reader of the NM actual off-block time reads the filed LOBT instead.

Two numbers, because one of them stops being honest at this size:
* the **linearised** LB-equivalent seconds of src/fold_metrics.py, comparable with every gate in the brief,
  valid while the change is small against the leaderboard RMSE;
* the **multiplicative** estimate, pooled RMSE of the candidate over pooled RMSE of the control applied to the
  current leaderboard score -- the one to quote when the answer runs to tens of seconds.
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from fold_metrics import compare, decided_map

LB_NOW = 248.7737  # gentle-igloo v78 lineage: v55 + E64 + E65
CTRL = sys.argv[1] if len(sys.argv) > 1 else "data/cv_preds_noaobt_ctrl"
CAND = sys.argv[2] if len(sys.argv) > 2 else "data/cv_preds_noaobt"

dec = decided_map()
se_c = se_e = 0.0
n_rows = 0
print(f"{'fold':>6} | {'ctrl RMSE':>9} | {'cand RMSE':>9} | {'ratio':>6} | rows")
for p in sorted(glob.glob(f"{CTRL}/fold_*.npz")):
    q = p.replace(CTRL, CAND)
    if not os.path.exists(q):
        print(f"{os.path.basename(p)[5:-4]:>6} | candidate fold missing")
        continue
    c, e = np.load(p), np.load(q)
    order = {m: i for i, m in enumerate(e["mvt_id"])}
    idx = np.array([order[m] for m in c["mvt_id"]])
    m = np.array([dec[k] for k in c["mvt_id"]])
    sc = ((c["pred"] - c["y"]) ** 2)[m]
    se = ((e["pred"][idx] - c["y"]) ** 2)[m]
    se_c += sc.sum()
    se_e += se.sum()
    n_rows += int(m.sum())
    rc, re = float(np.sqrt(sc.mean())), float(np.sqrt(se.mean()))
    print(f"{os.path.basename(p)[5:-4]:>6} | {rc:9.2f} | {re:9.2f} | {re / rc:6.3f} | {int(m.sum())}")

if n_rows:
    rc, re = float(np.sqrt(se_c / n_rows)), float(np.sqrt(se_e / n_rows))
    print(f"\npooled over the rows the base model decides: ctrl {rc:.2f} -> cand {re:.2f}, ratio {re / rc:.4f}")
    print(f"multiplicative estimate: leaderboard {LB_NOW:.4f} -> {LB_NOW * re / rc:.2f}  "
          f"(+{LB_NOW * (re / rc - 1):.2f} s)")

print("\nlinearised gate metrics (src/fold_metrics.py, valid while the change is small):")
dd, tot, small, trim, lower, n, st = compare(CTRL, CAND, lb_rmse=LB_NOW)
print(f"COST OF ABSTAINING FROM AOBT_3 (folds {n}, worse in {n - lower}): decided {dd:+.2f} s, strict {st:+.2f} s, "
      f"all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s, all LB-equivalent")
