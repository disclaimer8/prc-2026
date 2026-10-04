"""E186 primary fixed w=.15 addition to the saved FIN1b+E181a mixture.

No grid or weight optimization. This is the registered pre-ADS-B CV gate, not an
exact v1139 whole-chain OOF reconstruction. Rebuild regular/day candidates because
E172's saved values array can be stale while reg and pred are already updated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl
import torch  # before train/lightgbm (also imported by the shared I/O helpers).

from e186_nn_cv import atomic_json, atomic_npz, sha_file


def rebuild(proba, values, reg, d_lobt):
    v = np.asarray(values, dtype=np.float64).copy()
    v[:, 0], v[:, -1] = reg, reg + 86400
    pred = np.sum(proba.astype(np.float64) * np.nan_to_num(v), axis=1)
    pred = np.where(np.isnan(d_lobt), pred, np.clip(pred, d_lobt - 3606, d_lobt + 3606))
    return np.clip(pred, 0, None), v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", default="data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b")
    parser.add_argument("--member", default="data/cv_e130e186be")
    parser.add_argument("--features", default="data/features_fin1b.parquet")
    parser.add_argument("--out", default="data/cv_preds_e186be_w015_f1b")
    parser.add_argument("--allow-partial", action="store_true", help="Print diagnostics only; a partial run can never pass")
    args = parser.parse_args()
    member = Path(args.member)
    manifest = json.loads((member / "run_manifest.json").read_text())
    if sha_file(args.features) != manifest["spec"]["features_sha256"]:
        raise ValueError("Feature hash differs from the registered member run")
    paths = [member / f"fold_{a}_{a + 6}.npz" for a in range(1, 7)]
    if not args.allow_partial and not all(p.exists() for p in paths):
        raise ValueError("Primary gate requires all six member folds")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"experiment": "E186", "member_spec_sha256": manifest["spec_sha256"], "weight": 0.15,
            "control": str(Path(args.control).resolve()), "features_sha256": manifest["spec"]["features_sha256"],
            "code_sha256": sha_file(__file__)}
    spec_path = out / "gate_manifest.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Existing gate directory has a different manifest")
    atomic_json(spec_path, spec)
    fields = ["MVT_ID_mvt", "split", "d_mvt_lobt", "TAXITIME_SEC_mvt"]
    metadata = pl.read_parquet(args.features, columns=fields).filter(pl.col("split") == "train")
    for path in paths:
        if not path.exists():
            continue
        with np.load(path) as member_fold:
            m = {k: member_fold[k].copy() for k in member_fold.files}
        with np.load(Path(args.control) / path.name) as control_fold:
            c = {k: control_fold[k].copy() for k in control_fold.files}
        if str(m["spec_sha256"].item()) != manifest["spec_sha256"]:
            raise ValueError("Mixed member lineage")
        if len(np.unique(m["mvt_id"])) != len(m["mvt_id"]) or len(np.unique(c["mvt_id"])) != len(c["mvt_id"]):
            raise ValueError("Duplicate IDs")
        positions = {v: i for i, v in enumerate(m["mvt_id"])}
        if set(positions) != set(c["mvt_id"]):
            raise ValueError("Member/control ID set mismatch")
        idx = np.array([positions[v] for v in c["mvt_id"]])
        if not np.array_equal(m["y"][idx], c["y"]):
            raise ValueError("Member/control labels disagree")
        d = pl.DataFrame({"MVT_ID_mvt": c["mvt_id"]}).join(metadata, on="MVT_ID_mvt", how="left", maintain_order="left", validate="1:1")
        if not np.array_equal(d["TAXITIME_SEC_mvt"].to_numpy(), c["y"]):
            raise ValueError("Feature/control labels disagree")
        old, _ = rebuild(c["proba"], c["values"], c["reg"], d["d_mvt_lobt"].to_numpy())
        reg = 0.85 * c["reg"] + 0.15 * m["reg2"][idx]
        new, values = rebuild(c["proba"], c["values"], reg, d["d_mvt_lobt"].to_numpy())
        # The baseline can contain earlier fixed deltas; preserve them exactly.
        c["pred"] = c["pred"] + new - old
        c["reg"], c["values"] = reg, values.astype(np.float32)
        if not np.isfinite(c["pred"]).all():
            raise FloatingPointError("Nonfinite candidate")
        atomic_npz(out / path.name, **c)
    from fold_metrics import compare
    dd, total, body, trimmed, lower, n, strict = compare(args.control, args.out, features_path=args.features)
    passed = all(p.exists() for p in paths) and n == 6 and lower >= 4 and dd < 0 and trimmed < 0 and body < 0
    result = {"scope": "pre-ADS-B current FIN1b mixture", "weight": 0.15, "folds": n,
              "decided": dd, "all": total, "body_y_le_7200": body, "trimmed": trimmed,
              "strict": strict, "lower_decided_folds": int(lower), "pass": bool(passed),
              "units": "fold_metrics board-equivalent seconds, not exact RMSE difference"}
    atomic_json(out / "gate_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
