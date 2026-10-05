"""Registered E185-on-E182 additivity check; no training or weight search.

Run only after this code and the journal preregistration are committed together.
Weight .35 was selected in the earlier E185 CV: this is a registered additivity
check on reused OOF, not independent external validation. Ranking uses .35*2/3.
Exit 0 accepted by original E185 gate, 1 rejected, 2 invalid execution/inputs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

import strict_oof_gate as G

InvalidInput = G.InvalidOOF
FOLDS = G.FOLDS
ACCEPTED = "cv_preds_e172_rawf_lobtf_schedf_eobtf_aobtf_mlpa_f1b"
MEMBERS = ("rawf", "lobtf", "schedf", "eobtf", "aobtf", "mlpa")
# Frozen accepted E182 per-fold choices from data/e182_grid_5f.log, not pooled.
WEIGHTS = ((.05, .15, .15, .10, .15, .30), (.10, .20, .10, .15, .15, .20),
           *((0., .15, .15, .15, .15, .30),) * 4)
WEIGHT = .35
TOLERANCE = 1e-6
DAY = 86400.
WINDOW = 3606.


def mix(z, reg, lobt):
    values = z["values"].astype(np.float64).copy()
    values[:, 0], values[:, -1] = reg, reg + DAY
    expected = np.sum(z["proba"].astype(np.float64) * np.nan_to_num(values), axis=1)
    return np.clip(np.where(np.isnan(lobt), expected, np.clip(expected, lobt-WINDOW, lobt+WINDOW)), 0., None)


def synchronized(z):
    out = dict(z)
    out["values"] = z["values"].astype(np.float64).copy()
    out["values"][:, 0], out["values"][:, -1] = z["reg"], z["reg"] + DAY
    return out


def build_candidate(z, motion, subset, lobt):
    reg, pred = z["reg"].astype(np.float64).copy(), z["pred"].astype(np.float64).copy()
    reg[subset] = (1.-WEIGHT)*reg[subset] + WEIGHT*motion[subset]
    # Preserve every earlier stage's pred-minus-mixture correction. Assign only S.
    pred[subset] = (pred[subset] + mix(z, reg, lobt)[subset]
                    - mix(z, z["reg"], lobt)[subset])
    if not np.isfinite(reg).all() or not np.isfinite(pred).all():
        raise InvalidInput("nonfinite fixed-weight prediction")
    if not np.array_equal(pred[~subset], z["pred"][~subset]) or not np.array_equal(reg[~subset], z["reg"][~subset]):
        raise InvalidInput("prediction/reg changed outside the original subset")
    out = synchronized({**z, "reg": reg, "pred": pred})
    delta_error = np.abs((out["pred"] - mix(out, reg, lobt)) - (z["pred"] - mix(z, z["reg"], lobt)))
    if (delta_error > TOLERANCE).any():
        raise InvalidInput("pred-minus-mixture correction was not preserved")
    return out


def original_gate(report):
    """Journal 255: aggregate decided<0 is reported, not an extra threshold."""
    flags = {"four_of_six": report["metrics"]["lower_decided"] >= 4,
             "trimmed_lower": report["sse_delta"]["trimmed"] < 0,
             "body_lower": report["sse_delta"]["body"] < 0}
    return {"passed": all(flags.values()), "flags": flags,
            "definition": "decided lower in >=4/6; trimmed<0; body(y<=7200)<0"}


def read_npz(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise InvalidInput(f"{path}: duplicate NPZ fields")
            return {key: archive[key] for key in archive.files}
    except InvalidInput:
        raise
    except Exception as exc:
        raise InvalidInput(f"cannot read {path}: {exc}") from exc


def validate_arrays(z, path, *, member=False):
    required = {"mvt_id", "reg2"} if member else {"mvt_id", "y", "pred", "reg", "proba", "values"}
    if required - z.keys():
        raise InvalidInput(f"{path}: missing fields {sorted(required-z.keys())}")
    ids = G._integer_ids(z["mvt_id"], str(path))
    n = len(ids)
    if n == 0 or len(np.unique(ids)) != n:
        raise InvalidInput(f"{path}: empty or duplicate IDs")
    for key in (("reg2",) if member else ("y", "pred", "reg")):
        G._numeric(z[key], f"{path} {key}", (n,))
    if "month" in z:
        months = G._integer_ids(z["month"], f"{path} month")
        if months.shape != (n,):
            raise InvalidInput(f"{path}: wrong month shape")
    if not member:
        for key in ("values", "proba"):
            G._numeric(z[key], f"{path} {key}", (n, 6), allow_nan=(key == "values"))
        p = z["proba"]
        if (p < 0).any() or (p > 1).any() or (np.abs(p.sum(axis=1)-1.) > TOLERANCE).any():
            raise InvalidInput(f"{path}: invalid proba distribution")
    return ids


def aligned(z, ids, path, *, member=False):
    actual = validate_arrays(z, path, member=member)
    order = np.argsort(actual)
    if not np.array_equal(actual[order], np.sort(ids)):
        raise InvalidInput(f"{path}: ID set differs from accepted E182 fold")
    take = order[np.searchsorted(actual[order], ids)]
    return {key: value[take] if value.ndim and value.shape[0] == len(ids) else value
            for key, value in z.items()}


def load_metadata(root):
    paths = [root / "data" / name for name in ("features_fin1b.parquet", "features_e106.parquet")]
    metadata = [G._load_features(path) for path in paths]
    for key in metadata[0]:
        if not np.array_equal(metadata[0][key], metadata[1][key]):
            raise InvalidInput(f"feature parity failed: {key}")
    frames = []
    columns = ["MVT_ID_mvt", "ADEP_mvt", "LOBT_flt", "d_mvt_lobt"]
    for path in paths:
        frame = (pl.read_parquet(path, columns=[*columns, "split"])
                 .filter(pl.col("split") == "train").drop("split").sort("MVT_ID_mvt"))
        if not np.array_equal(G._integer_ids(frame["MVT_ID_mvt"].to_numpy(), str(path)), metadata[0]["ids"]):
            raise InvalidInput(f"feature parity failed: ID order in {path}")
        if frame["ADEP_mvt"].null_count():
            raise InvalidInput("feature parity cannot define subset with null airport")
        G._numeric(frame["d_mvt_lobt"].to_numpy(), f"{path} LOBT", (len(frame),), allow_nan=True)
        frames.append(frame)
    if not frames[0].equals(frames[1]):
        raise InvalidInput("feature parity failed: original E185 subset/LOBT fields")
    frame = frames[0]
    return {**metadata[0], "lobt": frame["d_mvt_lobt"].to_numpy(),
            "subset": (frame["LOBT_flt"].is_null() & frame["ADEP_mvt"].ne("LIRF")).to_numpy()}


def inventory(root):
    directories = ["cv_preds_fin1b_hz", ACCEPTED, *(f"cv_e130{m}" for m in (*MEMBERS, "motion"))]
    paths = [path for directory in directories for path in G._fold_paths(root / "data" / directory)]
    paths += [root / "data" / name for name in ("features_fin1b.parquet", "features_e106.parquet", "e182_grid_5f.log")]
    paths += [root / "src" / name for name in ("e172_members_cv.py", "e185_subset_grid_cv.py")]
    return paths


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_hashes(root, paths):
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def check_recorded_weights(root):
    lines = (root / "data/e182_grid_5f.log").read_text().splitlines()
    for name, wanted in zip(FOLDS, WEIGHTS):
        prefix = f"{ACCEPTED} {name}: weights "
        values = [line[len(prefix):] for line in lines if line.startswith(prefix)]
        try:
            valid = len(values) == 1 and ast.literal_eval(values[0]) == wanted
        except (SyntaxError, ValueError):
            valid = False
        if not valid:
            raise InvalidInput(f"{name}: recorded E182 weights do not match frozen protocol")


def prepare_fold(root, name, weights, meta, number):
    path = root / "data" / ACCEPTED / name
    accepted = read_npz(path)
    ids = validate_arrays(accepted, path)
    idx = np.searchsorted(meta["ids"], ids)
    if (idx >= len(meta["ids"])).any() or not np.array_equal(meta["ids"][idx], ids):
        raise InvalidInput(f"{name}: ID set outside authoritative features")
    months = meta["month"][idx]
    if not np.isin(months, (number, number+6)).all():
        raise InvalidInput(f"{name}: incorrect held-out month pair")
    base_path = root / "data/cv_preds_fin1b_hz" / name
    base = aligned(read_npz(base_path), ids, base_path)
    for label, z in (("accepted", accepted), ("base", base)):
        if not np.array_equal(z["y"], meta["y"][idx]):
            raise InvalidInput(f"{name}: {label} target differs from authoritative features")
        if "month" in z and not np.array_equal(z["month"], months):
            raise InvalidInput(f"{name}: {label} stored month mismatch")
    for key, column in (("proba", slice(None)), ("values", slice(1, -1))):
        if not np.array_equal(accepted[key][:, column], base[key][:, column], equal_nan=True):
            raise InvalidInput(f"{name}: {key} differs from E182 base lineage")
    members = []
    for member in (*MEMBERS, "motion"):
        path = root / "data" / f"cv_e130{member}" / name
        z = aligned(read_npz(path), ids, path, member=True)
        if "month" in z and not np.array_equal(z["month"], months):
            raise InvalidInput(f"{path}: stored month mismatch")
        if "y" in z and not np.array_equal(z["y"], meta["y"][idx]):
            raise InvalidInput(f"{path}: target mismatch")
        members.append(z["reg2"])
    w = np.asarray(weights, dtype=np.float64)
    rebuilt_reg = (1.-w.sum())*base["reg"] + np.column_stack(members[:-1]) @ w
    lobt = meta["lobt"][idx]
    rebuilt_pred = base["pred"] + mix(base, rebuilt_reg, lobt) - mix(base, base["reg"], lobt)
    errors = {key: float(np.max(np.abs(value-accepted[key])))
              for key, value in (("reg", rebuilt_reg), ("pred", rebuilt_pred))}
    if not all(np.isfinite(value) and value <= TOLERANCE for value in errors.values()):
        raise InvalidInput(f"{name}: faithful E182 rebuild failed: {errors}")
    control = synchronized(accepted)
    candidate = build_candidate(accepted, members[-1], meta["subset"][idx], lobt)
    return control, candidate, {"name": name, "weights": list(weights), "max_abs_rebuild": errors,
                                "rows": len(ids), "subset_rows": int(meta["subset"][idx].sum())}


def run(repo_root, output, *, registration):
    root, output = Path(repo_root).resolve(), Path(output).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", registration):
        raise InvalidInput("registration must be the full preregistration commit SHA")
    if output.exists():
        raise InvalidInput(f"output already exists: {output}")
    paths = inventory(root)
    before = input_hashes(root, paths)
    code_hashes = {str(path): sha256(path) for path in (Path(__file__).resolve(), Path(G.__file__).resolve())}
    check_recorded_weights(root)
    meta = load_metadata(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        (stage / "control").mkdir()
        (stage / "candidate").mkdir()
        all_ids, lineage = [], []
        for number, (name, weights) in enumerate(zip(FOLDS, WEIGHTS), 1):
            control, candidate, evidence = prepare_fold(root, name, weights, meta, number)
            all_ids.append(G._integer_ids(control["mvt_id"], name))
            lineage.append(evidence)
            np.savez(stage / "control" / name, **control)
            np.savez(stage / "candidate" / name, **candidate)
        ids = np.concatenate(all_ids)
        if len(np.unique(ids)) != len(ids) or not np.array_equal(np.sort(ids), meta["ids"]):
            raise InvalidInput("all-six ID set is duplicated or differs from full training coverage")
        if input_hashes(root, paths) != before:
            raise InvalidInput("input hashes changed while building")
        # All lineage/coverage validation has finished. First outcome calculation.
        report = G.evaluate(stage / "control", stage / "candidate", root / "data/features_fin1b.parquet",
                            audit_only=True, require_body=True, check_values=True)
        if not all(item.get("consistent", False) for folds in report["values_consistency"].values() for item in folds.values()):
            raise InvalidInput("synchronized values failed the mandatory consistency check")
        gate = original_gate(report)
        report.update(status="accepted" if gate["passed"] else "rejected", audit_only=False,
                      gate_passed=gate["passed"], gate_flags=gate["flags"], gate_definition=gate["definition"],
                      control=str(output / "control"), candidate=str(output / "candidate"))
        report["aggregate_decided_lower_diagnostic"] = report["sse_delta"]["decided"] < 0
        (stage / "verdict.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        outputs = {str(path.relative_to(stage)): sha256(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"registration_commit": registration, "weight": WEIGHT, "ranking_weight": WEIGHT*2/3,
                    "control_source": str(root / "data" / ACCEPTED), "member_source": str(root / "data/cv_e130motion"),
                    "subset": "LOBT_flt IS NULL AND ADEP_mvt != LIRF", "gate": gate,
                    "scope": "registered additivity check on reused OOF before ADS-B; not independent external validation",
                    "weight_selection": "0.35 selected on earlier E185 CV, journal 260; no retuning",
                    "input_sha256": before, "code_sha256": code_hashes, "output_sha256": outputs,
                    "faithful_rebuild_tolerance": TOLERANCE, "folds": lineage}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if input_hashes(root, paths) != before or any(sha256(path) != digest for path, digest in code_hashes.items()):
            raise InvalidInput("input/code hashes changed before publication")
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        if output.exists():
            raise InvalidInput(f"output already exists: {output}")
        stage.rename(output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", required=True, help="full SHA of code+journal preregistration commit")
    args = parser.parse_args(argv)
    try:
        report = run(args.repo_root, args.output, registration=args.registration)
    except (InvalidInput, OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "rows": report["rows"], "metrics": report["metrics"],
                      "gate": report["gate_definition"]}, indent=2, allow_nan=False))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
