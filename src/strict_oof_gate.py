"""Validate complete OOF artifacts before applying the legacy acceptance metrics.

No training imports or repairs. Exit 0: accepted (or valid audit), 1: valid
experiment rejected, 2: invalid inputs/execution. Months may be absent from
legacy NPZs; authoritative feature months still determine fold membership.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Bound standalone resource use before importing numerical libraries.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_name] = "2"
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

FOLDS = tuple(f"fold_{a}_{a + 6}.npz" for a in range(1, 7))
TARGET = "TAXITIME_SEC_mvt"
METRICS = ("decided", "strict", "all", "body", "trimmed")
LB_RMSE = 252.43
RANKING_ROWS = 344841
DENOMINATOR = 6 * 2 * LB_RMSE * RANKING_ROWS


class InvalidOOF(ValueError):
    """Malformed, incomplete, or inconsistent OOF input."""


def _integer_ids(array, label):
    array = np.asarray(array)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise InvalidOOF(f"{label}: integer IDs must have one-dimensional numeric shape")
    if array.dtype.kind == "f":
        if (not np.isfinite(array).all() or (np.abs(array) > 2**53).any()
                or (array != np.floor(array)).any()):
            raise InvalidOOF(f"{label}: IDs must be finite exact integers in the safe float range")
    if array.dtype.kind == "u" and (array > np.iinfo(np.int64).max).any():
        raise InvalidOOF(f"{label}: integer IDs exceed int64 range")
    return array.astype(np.int64)


def _numeric(array, label, shape, allow_nan=False):
    array = np.asarray(array)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise InvalidOOF(f"{label}: expected numeric shape {shape}, got {array.shape}")
    if (np.isinf(array).any() if allow_nan else not np.isfinite(array).all()):
        raise InvalidOOF(f"{label}: requires finite values" + " (NaN plans allowed)" * allow_nan)
    return array


def _load_fold(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise InvalidOOF(f"{path}: duplicate NPZ array names")
            data = {key: archive[key] for key in archive.files}
    except InvalidOOF:
        raise
    except Exception as exc:
        raise InvalidOOF(f"cannot read {path}: {exc}") from exc
    missing = {"mvt_id", "pred", "y"} - data.keys()
    if missing:
        raise InvalidOOF(f"{path}: missing fields {sorted(missing)}")
    ids = _integer_ids(data["mvt_id"], str(path))
    n = len(ids)
    if n == 0:
        raise InvalidOOF(f"{path}: empty fold")
    if len(np.unique(ids)) != n:
        raise InvalidOOF(f"{path}: duplicate IDs within fold")
    data["mvt_id"] = ids
    for key in ("pred", "y", "reg"):
        if key in data:
            data[key] = _numeric(data[key], f"{path} {key}", (n,))
    if "month" in data:
        data["month"] = _integer_ids(data["month"], f"{path} month")
        if data["month"].shape != (n,):
            raise InvalidOOF(f"{path}: wrong month shape")
    for key in ("proba", "values"):
        if key in data:
            data[key] = _numeric(data[key], f"{path} {key}", (n, 6), allow_nan=(key == "values"))
    return data


def _load_features(path):
    required = {"MVT_ID_mvt", "split", "ADEP_mvt", "FLIGHT_ID_mvt", "d_mvt_sched", TARGET}
    try:
        schema = pl.read_parquet_schema(path)
        missing = required - schema.keys()
        if missing:
            raise InvalidOOF(f"features missing columns {sorted(missing)}")
        month_column = "month" if "month" in schema else "MVT_TIME_UTC_mvt"
        if month_column not in schema:
            raise InvalidOOF("features missing month and MVT_TIME_UTC_mvt")
        frame = pl.read_parquet(path, columns=sorted(required | {month_column})).filter(pl.col("split") == "train")
        if month_column != "month":
            frame = frame.with_columns(pl.col(month_column).dt.month().alias("month"))
        ids = _integer_ids(frame["MVT_ID_mvt"].to_numpy(), "features")
        if not len(ids):
            raise InvalidOOF("features contain no training IDs")
        order = np.argsort(ids)
        ids = ids[order]
        if (ids[1:] == ids[:-1]).any():
            raise InvalidOOF("features contain duplicate training IDs")
        months = _integer_ids(frame["month"].to_numpy(), "features month")[order]
        if ((months < 1) | (months > 12)).any():
            raise InvalidOOF("features month must be in 1..12")
        y = _numeric(frame[TARGET].to_numpy(), "features target", (len(ids),))[order]
        delays = frame["d_mvt_sched"].cast(pl.Float64).to_numpy()[order]
        if np.isinf(delays).any():
            raise InvalidOOF("features delay must be finite or null/NaN")
        tail_population = (frame["ADEP_mvt"].eq("LIRF").fill_null(False).to_numpy()[order]
                           & frame["FLIGHT_ID_mvt"].is_null().to_numpy()[order])
        # NumPy comparisons make absent/NaN D false, as in decided_map's Python comparisons.
        decided = ~(tail_population & (delays >= 14400))
        strict = ~(tail_population & ((delays >= 14400) | (delays > 0)))
        return {"ids": ids, "month": months, "y": y, "decided": decided, "strict": strict}
    except InvalidOOF:
        raise
    except Exception as exc:
        raise InvalidOOF(f"cannot read features {path}: {exc}") from exc


def _fold_paths(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise InvalidOOF(f"OOF directory does not exist: {directory}")
    actual = {p.name for p in directory.glob("*.npz")}
    if actual != set(FOLDS):
        raise InvalidOOF(f"{directory}: expected exactly six folds; missing={sorted(set(FOLDS) - actual)}, "
                         f"extra={sorted(actual - set(FOLDS))}")
    return [directory / name for name in FOLDS]


def _values_consistency(data):
    if "values" not in data or "reg" not in data:
        return {"checked": False, "reason": "values or reg absent"}
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(data["values"][:, [0, 5]].astype(np.float64)
                       - np.column_stack([data["reg"], data["reg"] + 86400.]))
    finite = bool(np.isfinite(error).all())
    return {"checked": True, "consistent": finite and bool((error <= 1e-6).all()),
            "max_abs_seconds": float(error.max()) if finite else None,
            "tolerance_seconds": 1e-6}


def evaluate(control, candidate, features, *, require_body=False, audit_only=False, check_values=False):
    """Return JSON-safe evidence; all validation completes before computing SSE."""
    paths = {"control": _fold_paths(control), "candidate": _fold_paths(candidate)}
    metadata = _load_features(features)
    loaded = {side: [_load_fold(path) for path in files] for side, files in paths.items()}
    for side, folds in loaded.items():
        all_ids = np.concatenate([z["mvt_id"] for z in folds])
        if len(np.unique(all_ids)) != len(all_ids):
            raise InvalidOOF(f"{side}: duplicate IDs across folds")
        if not np.array_equal(np.sort(all_ids), metadata["ids"]):
            raise InvalidOOF(f"{side}: training ID set coverage differs from features")
    prepared = []
    for a, (control_fold, candidate_fold) in enumerate(zip(loaded["control"], loaded["candidate"]), 1):
        name = FOLDS[a - 1]
        ids = control_fold["mvt_id"]
        order = np.argsort(candidate_fold["mvt_id"])
        if not np.array_equal(np.sort(ids), candidate_fold["mvt_id"][order]):
            raise InvalidOOF(f"{name}: candidate/control ID set differs")
        idx = order[np.searchsorted(candidate_fold["mvt_id"][order], ids)]
        if not np.array_equal(control_fold["y"], candidate_fold["y"][idx]):
            raise InvalidOOF(f"{name}: candidate/control target differs")
        feature_idx = np.searchsorted(metadata["ids"], ids)
        months = metadata["month"][feature_idx]
        if not np.isin(months, (a, a + 6)).all():
            raise InvalidOOF(f"{name}: feature month outside held-out fold pair")
        for side, z, align in (("control", control_fold, slice(None)), ("candidate", candidate_fold, idx)):
            if "month" in z and not np.array_equal(z["month"][align], months):
                raise InvalidOOF(f"{name}: {side} month differs from features")
            if not np.array_equal(z["y"][align], metadata["y"][feature_idx]):
                raise InvalidOOF(f"{name}: {side} target differs from authoritative features")
        prepared.append((control_fold, candidate_fold, idx, feature_idx))
    sums = {key: 0. for key in METRICS}
    evidence_folds = []
    lower = 0
    with np.errstate(over="raise", invalid="raise"):
        try:
            for name, (ctrl, cand, idx, feature_idx) in zip(FOLDS, prepared):
                y = ctrl["y"].astype(np.float64)
                delta = (cand["pred"][idx].astype(np.float64) - y) ** 2 - (ctrl["pred"].astype(np.float64) - y) ** 2
                m, ms = metadata["decided"][feature_idx], metadata["strict"][feature_idx]
                sorted_delta = np.sort(delta)
                sse = {"all": float(delta.sum()), "decided": float(delta[m].sum()),
                       "strict": float(delta[ms].sum()), "body": float(delta[y <= 7200].sum()),
                       "trimmed": float(delta.sum() - sorted_delta[:3].sum() - sorted_delta[-3:].sum())}
                lower += sse["decided"] < 0
                for key in METRICS:
                    sums[key] += sse[key]
                evidence_folds.append({"name": name, "rows": len(y), "tail_rows": int((~m).sum()),
                                       "sse_delta": sse, "metrics": {k: v / DENOMINATOR for k, v in sse.items()}})
        except FloatingPointError as exc:
            raise InvalidOOF(f"nonfinite metric arithmetic: {exc}") from exc
    if not all(np.isfinite(v) for v in sums.values()):
        raise InvalidOOF("nonfinite aggregate metrics")
    flags = {"decided_lower": sums["decided"] < 0, "four_of_six": lower >= 4,
             "trimmed_lower": sums["trimmed"] < 0, "body_lower": sums["body"] < 0}
    passed = flags["decided_lower"] and flags["four_of_six"] and flags["trimmed_lower"] and (not require_body or flags["body_lower"])
    result = {"status": "audit_valid" if audit_only else "accepted" if passed else "rejected",
              "valid": True, "gate_passed": passed, "audit_only": audit_only, "require_body": require_body,
              "control": str(Path(control).resolve()), "candidate": str(Path(candidate).resolve()),
              "features": str(Path(features).resolve()), "fold_count": 6, "rows": len(metadata["ids"]),
              "target_verified_against_features": True, "sse_delta": sums,
              "metrics": {**{k: v / DENOMINATOR for k, v in sums.items()}, "lower_decided": lower},
              "gate_flags": flags, "folds": evidence_folds,
              "normalization": {"lb_rmse": LB_RMSE, "ranking_rows": RANKING_ROWS, "denominator": DENOMINATOR},
              "body_definition": "y <= 7200", "values_consistency": None}
    if check_values:
        result["values_consistency"] = {side: {name: _values_consistency(z) for name, z in zip(FOLDS, folds)}
                                        for side, folds in loaded.items()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control")
    parser.add_argument("candidate")
    parser.add_argument("--features", required=True)
    parser.add_argument("--require-body", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="validate/measure without requiring improvement")
    parser.add_argument("--check-values", action="store_true", help="report stale regular/day values; diagnostic only")
    parser.add_argument("--json", metavar="PATH", help="write evidence to a new PATH; '-' prints JSON only")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.control, args.candidate, args.features, require_body=args.require_body,
                          audit_only=args.audit_only, check_values=args.check_values)
        code = 0 if result["status"] in ("accepted", "audit_valid") else 1
    except InvalidOOF as exc:
        result, code = {"status": "invalid", "valid": False, "error": str(exc)}, 2
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.json == "-":
        print(rendered)
    else:
        if args.json:
            try:
                with Path(args.json).open("x") as output:
                    output.write(rendered + "\n")
            except OSError as exc:
                print(json.dumps({"status": "invalid", "valid": False, "error": f"cannot write JSON: {exc}"}))
                return 2
        if result["valid"]:
            for fold in result["folds"]:
                print(f"{fold['name']}: " + " | ".join(f"{k} {v:+.6f}s" for k, v in fold["metrics"].items()))
            m = result["metrics"]
            print(f"{result['status'].upper()}: decided {m['decided']:+.6f}s "
                  f"(lower {m['lower_decided']}/6) | strict {m['strict']:+.6f}s | all {m['all']:+.6f}s "
                  f"| y<=7200 {m['body']:+.6f}s | trimmed {m['trimmed']:+.6f}s | rows {result['rows']}")
            if args.check_values:
                print("VALUES (diagnostic only): " + json.dumps(result["values_consistency"]))
        else:
            print(f"INVALID: {result['error']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
