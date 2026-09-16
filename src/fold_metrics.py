"""Shared reading of fold predictions for every gate (16.09).

Three numbers per comparison instead of one, and the main one is not the total:

* **decided** — rows the base model actually decides. The final chain overwrites LIRF departures without an NM record
  and with D = MVT - SCHED >= 4 h by the empirical tail table (src/lirf_table.py, TabPFN probabilities), so whatever the
  base model predicts there never reaches a submission. Those rows are 10-45 per fold and carry up to 60% of the fold SSE.
* **total** — all rows, as the gates were computed until 16.09.
* **y <= 7200** and **trimmed** (without the 3 best and 3 worst rows of the fold) — how much of the gain is a handful of rows.

Why: E38 (a model per airport) won 10.98 s of CV and lost 3.79 s on the leaderboard. The early-stopping leak explains
about 2 s (E43, nested rounds: still -8.85 s). Excluding the rows the tail table overwrites leaves -1.40 s, i.e. 7.45 s
of the CV gain lived in rows no submission ever takes from the base model.
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

FEATURES = os.environ.get("FEATURES", "data/features_e28.parquet")
TAIL_TABLE_DELAY = 14400  # lirf_table.py: LIRF, no FLIGHT_ID, D >= 4 h


def decided_map(features_path: str = FEATURES, strict: bool = False) -> dict:
    """MVT_ID -> True when the base model's prediction survives into the submission.

    strict also drops the E14 population (the same LIRF rows at 0 < D < 4 h), where the submission keeps only the
    "regular" part of the base prediction and mixes it with the schedule by a logistic weight. Those rows are another
    5-28% of fold SSE, so the strict number says whether a gain survives outside every row a later stage touches.
    """
    df = pl.read_parquet(features_path, columns=["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "d_mvt_sched"])
    out = {}
    for mvt, ap, fid, d in df.iter_rows():
        tail = ap == "LIRF" and fid is None and d is not None
        out[mvt] = not (tail and (d >= TAIL_TABLE_DELAY or (strict and d > 0)))
    return out


def compare(ctrl_dir: str, cand_dir: str, lb_rmse: float = 252.43, features_path: str = FEATURES, quiet: bool = False):
    """Print per-fold numbers and return (decided, total, y<=7200, trimmed, lower_decided, n, strict) in LB-equivalent seconds."""
    dec = decided_map(features_path)
    strict = decided_map(features_path, strict=True)
    lb = 6 * 2 * lb_rmse * 344841
    tot = dd = small = trim = st = 0.0
    lower = n = 0
    for p in sorted(glob.glob(f"{ctrl_dir}/fold_*.npz")):
        q = p.replace(ctrl_dir, cand_dir)
        if not os.path.exists(q):
            continue
        c, e = np.load(p), np.load(q)
        order = {m: i for i, m in enumerate(e["mvt_id"])}
        idx = np.array([order[m] for m in c["mvt_id"]])
        d = (e["pred"][idx] - c["y"]) ** 2 - (c["pred"] - c["y"]) ** 2
        m = np.array([dec[k] for k in c["mvt_id"]])
        s = np.sort(d)
        ms = np.array([strict[k] for k in c["mvt_id"]])
        tot += d.sum()
        dd += d[m].sum()
        st += d[ms].sum()
        small += d[c["y"] <= 7200].sum()
        trim += d.sum() - s[:3].sum() - s[-3:].sum()
        lower += d[m].sum() < 0
        n += 1
        if not quiet:
            print(f"{os.path.basename(p)[5:-4]}: decided {d[m].sum() / lb:+.2f} s | strict {d[ms].sum() / lb:+.2f} | "
                  f"all {d.sum() / lb:+.2f} | y<=7200 {d[c['y'] <= 7200].sum() / lb:+.2f} | "
                  f"trimmed {(d.sum() - s[:3].sum() - s[-3:].sum()) / lb:+.2f} ({(~m).sum()} rows go to the tail table)")
    return dd / lb, tot / lb, small / lb, trim / lb, lower, n, st / lb
