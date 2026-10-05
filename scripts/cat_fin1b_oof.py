"""Isolated retrospective CPU CatFIN1b component OOF, not full-v12048 validation.

prepare: frozen sources + six month-pair views, no model fitting.
run: fixed seed-major 6 x 15 fits; completed artifacts are verified read-only.
Root must register code and protocol together before run, and provide the
26 GiB service cgroup. No quality-based stopping, weight search or upload.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
PROTOCOL = "cat-fin1b-cpu-oof-15seed-v1"
FEATURE_SHA = "f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052"
SOURCES = {
    "catboost_seeds.py": "a52059bcdf622c5902d0120baac7893dc96d40fcf390707dc1d22e51ff23d5e5",
    "train.py": "b47dfcaef892bc9dc7dc4a913c3f38c8c94540f4becb4c9bfb66a7fec5d3d706",
    "features.py": "58e6b783df2123adf261022733e4c602dfd22ff97d2e98e9e03e8c12486a17a2",
    "target_encoding.py": "ac1b35fdd31139b531ca3760d9172f78112c77ed3ca3e72631100cd252b37d6c",
}
SCIENTIFIC = {
    "PROXY_SCALE": "1", "CLF_COST_WEIGHT": "1", "CLF_COST_MIN_SPREAD": "3600",
    "DROP_PREFIX": "day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config",
    "REGIME_RATES": "0", "CATBOOST_THREADS": "24", "OMP_NUM_THREADS": "24",
    "OMP_THREAD_LIMIT": "24", "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0",
    "POLARS_MAX_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
    "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
}


class Invalid(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    """Exclusive final creation; containing staging directory supplies atomicity."""
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def writer_lock(work):
    Path(work).mkdir(parents=True, exist_ok=True)
    with (Path(work) / ".writer.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Invalid("another writer owns this work directory") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def schedule():
    return [(seed, fold) for seed in range(15) for fold in range(1, 7)]


def integral_ids(ids):
    # Official PRC parquet stores exact integral IDs as Float64. Preserve its
    # dtype and order; reject fractional/nonfinite or inexact large float IDs.
    return (ids.ndim == 1 and ids.dtype.kind in "fiu" and np.isfinite(ids).all()
            and (ids.dtype.kind != "f" or ((np.abs(ids) <= 2**53).all()
                                           and np.equal(ids, np.floor(ids)).all())))


def fold_view(frame, fold):
    if fold not in range(1, 7):
        raise Invalid("invalid month pair")
    train = frame.filter(pl.col("split") == "train")
    ids = train[ID].to_numpy()
    y = train[TARGET].cast(pl.Float64).to_numpy()
    month = train["month"].to_numpy()
    if (not len(ids) or not integral_ids(ids) or len(np.unique(ids)) != len(ids)
            or not np.isfinite(y).all() or not np.isin(month, np.arange(1, 13)).all()):
        raise Invalid("invalid train IDs/targets/months")
    hold = np.isin(month, [fold, fold + 6])
    if not hold.any() or hold.all():
        raise Invalid("empty fit or heldout population")
    metadata = {"mvt_id": ids[hold], "y": y[hold], "month": month[hold]}
    view = train.with_columns(
        pl.when(pl.Series(hold)).then(pl.lit("rank")).otherwise(pl.lit("train")).alias("split"),
        pl.when(pl.Series(hold)).then(pl.lit(None)).otherwise(pl.col(TARGET)).alias(TARGET))
    return view, metadata


def prediction(path, ids):
    try:
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"mvt_id", "reg"}:
                raise Invalid("unexpected raw seed fields")
            actual, reg = z["mvt_id"], z["reg"]
            if (not integral_ids(actual) or not np.array_equal(actual, ids)
                    or len(np.unique(actual)) != len(ids) or reg.shape != ids.shape
                    or reg.dtype.kind not in "fiu" or not np.isfinite(reg).all()):
                raise Invalid("invalid seed IDs/order/shape/finite values")
            return reg.copy()
    except (ValueError, OSError, KeyError) as exc:
        raise Invalid(f"unreadable seed {path}: {exc}") from exc


def publish_seed(raw, final, ids, evidence):
    final = Path(final)
    if final.exists():
        raise Invalid(f"refusing seed overwrite: {final}")
    prediction(raw, ids)
    stage = final.parent / (".publish-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    shutil.copyfile(raw, stage / "prediction.npz")
    prediction(stage / "prediction.npz", ids)
    write_json(stage / "manifest.json", {"evidence": evidence, "rows": len(ids),
                                          "sha256": sha(stage / "prediction.npz")})
    os.rename(stage, final)


def verify_seed(final, ids, evidence):
    final = Path(final)
    try:
        m = read_json(final / "manifest.json")
        if (m["evidence"] != evidence or m["rows"] != len(ids)
                or m["sha256"] != sha(final / "prediction.npz")):
            raise Invalid(f"seed lineage or SHA changed: {final}")
        return prediction(final / "prediction.npz", ids)
    except (OSError, KeyError, ValueError) as exc:
        raise Invalid(f"invalid completed seed {final}: {exc}") from exc


def quarantine_attempts(work):
    work = Path(work)
    (work / "quarantine").mkdir(exist_ok=True)
    for directory in (work / "attempts", work / "inputs"):
        if not directory.exists():
            continue
        for path in list(directory.iterdir()):
            if directory.name == "inputs" and not path.name.startswith(".prepare-"):
                continue
            os.rename(path, work / "quarantine" / (path.name + "-" + uuid.uuid4().hex))
    for path in work.glob("fold_*/.publish-*"):
        os.rename(path, work / "quarantine" / (path.name + "-" + uuid.uuid4().hex))


def merge_fold(folder, metadata, lineage, fold):
    folder = Path(folder)
    ids = metadata["mvt_id"]
    arrays, hashes = [], []
    for seed in range(15):
        path = folder / f"seed_{seed:02}"
        arrays.append(verify_seed(path, ids, {"seed": seed, "lineage": lineage, "fold": fold}))
        hashes.append(sha(path / "prediction.npz"))
    stack = np.stack(arrays)
    expected = {"lineage": lineage, "fold": fold, "seeds": list(range(15)), "seed_sha256": hashes}
    final = folder / "complete"
    if final.exists():
        m = read_json(final / "manifest.json")
        if m["evidence"] != expected or sha(final / "prediction.npz") != m["sha256"]:
            raise Invalid("completed fold provenance changed")
        with np.load(final / "prediction.npz", allow_pickle=False) as z:
            wanted = {**metadata, "per_seed": stack, "reg2": stack.mean(axis=0)}
            if set(z.files) != set(wanted) or any(not np.array_equal(z[k], v) for k, v in wanted.items()):
                raise Invalid("completed mean or metadata changed")
        return
    stage = folder / (".publish-" + uuid.uuid4().hex)
    stage.mkdir()
    np.savez(stage / "prediction.npz", **metadata, per_seed=stack, reg2=stack.mean(axis=0))
    write_json(stage / "manifest.json", {"evidence": expected, "sha256": sha(stage / "prediction.npz")})
    os.rename(stage, final)


def child_environment(view, attempt):
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    return {**env, **SCIENTIFIC, "FEATURES": str(view),
            "CAT_SEED_OUT": str(Path(attempt) / "raw_seed{}.npz"),
            "CAT_MERGE_OUT": str(Path(attempt) / "unused_merge.npz")}


def packages():
    versions = {p: version(p) for p in ("catboost", "lightgbm", "numpy", "polars", "pandas", "pyarrow")}
    if versions["catboost"] != "1.2.10" or versions["lightgbm"] != "4.7.0":
        raise Invalid("CatBoost 1.2.10 and LightGBM 4.7.0 required; do not change active environments")
    return {"python": sys.version, "packages": versions}


def verify_sources(directory):
    for name, expected in SOURCES.items():
        if sha(Path(directory) / name) != expected:
            raise Invalid(f"source changed: {name}")


def inspect_columns(source, table, cwd):
    code = ("import json,sys,polars as pl;sys.path.insert(0,sys.argv[1]);import train as T;"
            "d=pl.DataFrame(schema=pl.read_parquet_schema(sys.argv[2]));c=T.feature_columns(d);"
            "print(json.dumps({'features':c,'categoricals':[x for x in T.CATEGORICAL if x in c]}))")
    result = subprocess.run([sys.executable, "-c", code, str(source), str(table)], cwd=cwd,
                            env=child_environment(table, cwd), check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def prepare(root, work, registration):
    if not re.fullmatch("[a-f0-9]{40}", registration):
        raise Invalid("full registration commit required")
    features = root / "data/features_fin1b.parquet"
    verify_sources(root / "src")
    if sha(features) != FEATURE_SHA:
        raise Invalid("not the pinned FIN1b table")
    runtime = packages()
    if (work / "manifest.json").exists():
        verify_run(work, registration)
        return
    if any(p.name != ".writer.lock" for p in work.iterdir()):
        raise Invalid("incomplete preparation preserved; use a fresh work directory")
    (work / "sources").mkdir()
    for name in SOURCES:
        shutil.copyfile(root / "src" / name, work / "sources" / name)
    reference = read_json(root / "src/e182_rank_lineage.json")
    columns = inspect_columns(work / "sources", features, work)
    if (len(columns["features"]) != 170 or columns["features"] != reference["feature_columns"]
            or columns["categoricals"] != reference["categorical_columns"]):
        raise Invalid("FIN1b feature/category lineage differs")
    shutil.copyfile(Path(__file__), work / "runner.py")
    (work / "inputs").mkdir()
    (work / "attempts").mkdir()
    (work / "logs").mkdir()
    frame = pl.read_parquet(features).filter(pl.col("split") == "train")
    if frame.height != 2085047 or set(frame["month"].unique()) != set(range(1, 13)):
        raise Invalid("FIN1b training coverage differs")
    inputs = {}
    for fold in range(1, 7):
        view, metadata = fold_view(frame, fold)
        name = f"fold_{fold}_{fold+6}"
        stage = work / "inputs" / (".prepare-" + name)
        stage.mkdir()
        view.write_parquet(stage / "features.parquet")
        np.savez(stage / "metadata.npz", **metadata)
        # A round-trip proves that serialization preserves row and column order.
        restored = pl.read_parquet(stage / "features.parquet")
        if not view.equals(restored):
            raise Invalid("fold view changed during serialization")
        del restored, view
        inputs[name] = {"rows": len(metadata["mvt_id"]),
                        **{file: sha(stage / file) for file in ("features.parquet", "metadata.npz")}}
        os.rename(stage, work / "inputs" / name)
        (work / name).mkdir()
        print(f"prepared {name}: {inputs[name]['rows']} heldout rows", flush=True)
    del frame
    verify_sources(work / "sources")
    if sha(features) != FEATURE_SHA:
        raise Invalid("original feature table changed during preparation")
    manifest = {"protocol": PROTOCOL, "registration": registration, "feature_sha256": FEATURE_SHA,
                "source_sha256": SOURCES, "runner_sha256": sha(work / "runner.py"),
                "environment": SCIENTIFIC, "runtime": runtime, "columns": columns, "inputs": inputs,
                "schedule": schedule(), "iterations": 5600, "task_type": "CPU (original trainer default)",
                "parameters": {"depth": 8, "learning_rate": .08, "loss_function": "RMSE"},
                "purpose": "Retrospective production-matching Cat component; not independent holdout or full v12048"}
    manifest["lineage"] = digest(manifest)
    write_json(work / "manifest.json", manifest)


def verify_run(work, registration):
    m = read_json(work / "manifest.json")
    body = {k: v for k, v in m.items() if k != "lineage"}
    if (digest(body) != m["lineage"] or m["protocol"] != PROTOCOL or m["registration"] != registration
            or m["source_sha256"] != SOURCES or m["environment"] != SCIENTIFIC
            or m["runtime"] != packages() or m["runner_sha256"] != sha(Path(__file__))
            or m["runner_sha256"] != sha(work / "runner.py") or m["feature_sha256"] != FEATURE_SHA
            or m["schedule"] != [list(x) for x in schedule()]):
        raise Invalid("run manifest, environment, package or wrapper lineage changed")
    verify_sources(work / "sources")
    for name, record in m["inputs"].items():
        for file in ("features.parquet", "metadata.npz"):
            if sha(work / "inputs" / name / file) != record[file]:
                raise Invalid(f"pinned fold input changed: {name}/{file}")
    return m


def run(work, registration):
    m = verify_run(work, registration)
    quarantine_attempts(work)
    metadata = {}
    for fold in range(1, 7):
        name = f"fold_{fold}_{fold+6}"
        with np.load(work / "inputs" / name / "metadata.npz", allow_pickle=False) as z:
            metadata[fold] = {key: z[key] for key in z.files}
    for index, (seed, fold) in enumerate(schedule()):
        name = f"fold_{fold}_{fold+6}"
        final = work / name / f"seed_{seed:02}"
        evidence = {"seed": seed, "lineage": m["lineage"], "fold": fold}
        ids = metadata[fold]["mvt_id"]
        if final.exists():
            verify_seed(final, ids, evidence)
            print(f"verified completed {name} seed={seed}", flush=True)
            continue
        verify_sources(work / "sources")
        view = work / "inputs" / name / "features.parquet"
        if sha(view) != m["inputs"][name]["features.parquet"]:
            raise Invalid("fold input changed before fit")
        attempt = work / "attempts" / f"{name}-s{seed:02}-{uuid.uuid4().hex}"
        attempt.mkdir()
        env = child_environment(view, attempt)
        started = time.monotonic()
        write_json(attempt / "invocation.json", {"evidence": evidence, "environment": env,
                   "command": [sys.executable, str(work / "sources/catboost_seeds.py"), "5600", str(seed)]})
        with (attempt / "fit.log").open("x") as log:
            result = subprocess.run([sys.executable, str(work / "sources/catboost_seeds.py"), "5600", str(seed)],
                                    cwd=attempt, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - started
        write_json(attempt / "exit.json", {"returncode": result.returncode, "seconds": elapsed})
        if result.returncode:
            raise Invalid(f"fit failed rc={result.returncode}; preserved {attempt}")
        verify_sources(work / "sources")
        if sha(view) != m["inputs"][name]["features.parquet"]:
            raise Invalid("fold input changed during fit")
        publish_seed(attempt / f"raw_seed{seed}.npz", final, ids, evidence)
        os.rename(attempt, work / "logs" / attempt.name)
        print(f"completed {index+1}/90 {name} seed={seed}, seconds={elapsed:.1f}; "
              f"remaining_at_this_rate_hours={elapsed*(89-index)/3600:.2f} (provisional)", flush=True)
    for fold in range(1, 7):
        merge_fold(work / f"fold_{fold}_{fold+6}", metadata[fold], m["lineage"], fold)
    summary = {"lineage": m["lineage"], "fits": 90, "folds": 6, "quality_gate": None,
               "fold_sha256": {str(f): sha(work / f"fold_{f}_{f+6}/complete/prediction.npz") for f in range(1, 7)}}
    if (work / "complete.json").exists():
        if read_json(work / "complete.json") != summary:
            raise Invalid("completed run summary differs")
    else:
        write_json(work / "complete.json", summary)
    print("All 90 fits and six complete means verified. Component OOF only.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--registration-commit", required=True)
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    if work == root or work == root / "data":
        raise Invalid("work must be a dedicated isolated directory")
    with writer_lock(work):
        if args.action == "prepare":
            prepare(root, work, args.registration_commit)
        elif args.action == "run":
            run(work, args.registration_commit)
        else:
            verify_run(work, args.registration_commit)
            print("Pinned inputs and environment verified; no fitting performed.")


if __name__ == "__main__":
    try:
        main()
    except (Invalid, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(2)
