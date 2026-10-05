"""Retrospective FIN1b production LGB5 component OOF, registration 283.

Six month pairs, five seeds per fold, fixed 2319 production rounds. Historical
round selection used Jan/Jul: this is not independent nested CV or a full-chain
quality gate. Preparation never fits; run never changes shared trainer sources.
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

os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
PROTOCOL = "lgb-fin1b-component-oof-5seed-v1"
ROUNDS, SEEDS = 2319, list(range(1000, 1005))
FEATURE_SHA = "f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052"
RANK_SHA = "01b1d9937a7753050fb91e4e15b4d25a9e33b39e30866725ef26e00838f83bbe"
SOURCES = {
    "seed_avg.py": "931aa4ab25b26e07f7d4bfb28f9272aa4faf2bbd00b58b429100fc95732491b0",
    "train.py": "248b74f238a7ff7c6deb0b784a4d52f030264e53ae3574aec02a128fca0ce5c5",
    "features.py": "58e6b783df2123adf261022733e4c602dfd22ff97d2e98e9e03e8c12486a17a2",
    "target_encoding.py": "ac1b35fdd31139b531ca3760d9172f78112c77ed3ca3e72631100cd252b37d6c",
}
PROOFS = {
    "validation_v870.json": "9089e9965d10ba12e127e5f3b43f914b526c61b20149f29029235d6db9b69556",
    "fin1b_lgbseeds.log": "5ace4afad2290e56fe737006d4a5ab6f38be1566c32845ed3a2a2f340aed9644",
    "fin1b_base.log": "7618eacc9124fa71e6dc484d9b3e29322b0707d7620310cf5fdc871e41f3d77e",
}
FEATURE_COLUMNS_SHA = "730ed1c410f17bcbb3ceb5684562bdae9f61ece6a32663919f688895533632cf"
CATEGORY_COLUMNS_SHA = "8a28688f8a63b4c0c4815587f4922aa96c03c2bee2d025dd522de37c26474dd3"
VERSIONS = {"lightgbm": "4.7.0", "numpy": "2.5.3", "polars": "1.44.2",
            "pandas": "3.0.5", "pyarrow": "25.0.1"}
PARAMETERS = {"objective": "regression", "learning_rate": .03, "num_leaves": 511,
              "min_data_in_leaf": 50, "feature_fraction": .8, "bagging_fraction": .8,
              "bagging_freq": 1, "lambda_l2": 10, "cat_smooth": 20,
              "min_data_per_group": 50, "max_cat_to_onehot": 8, "verbose": -1,
              "num_threads": 6, "seed": 42, "deterministic": True, "force_row_wise": True}
SCIENTIFIC = {
    "PROXY_SCALE": "1", "CLF_COST_WEIGHT": "1", "CLF_COST_MIN_SPREAD": "3600",
    "DROP_PREFIX": "day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config",
    "REGIME_RATES": "0", "REG_OVERRIDE": '{"num_threads":6}',
    "OMP_NUM_THREADS": "6", "OMP_THREAD_LIMIT": "6", "OMP_WAIT_POLICY": "PASSIVE",
    "GOMP_SPINCOUNT": "0", "POLARS_MAX_THREADS": "6", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
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


def integral_ids(ids):
    return (ids.ndim == 1 and ids.dtype.kind in "fiu" and np.isfinite(ids).all()
            and (ids.dtype.kind != "u" or (ids <= np.iinfo(np.int64).max).all())
            and (ids.dtype.kind != "f" or ((np.abs(ids) <= 2**53).all()
                                           and np.equal(ids, np.floor(ids)).all())))


def fold_view(frame, fold):
    if fold not in range(1, 7):
        raise Invalid("invalid month pair")
    train = frame.filter(pl.col("split") == "train")
    ids, month = train[ID].to_numpy(), train["month"].to_numpy()
    y = train[TARGET].cast(pl.Float64).to_numpy()
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


def validate_raw(path, ids):
    try:
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"mvt_id", "reg", "per_seed"}:
                raise Invalid("unexpected raw fields")
            actual, reg, seeds = z["mvt_id"], z["reg"], z["per_seed"]
            if (not integral_ids(actual) or not np.array_equal(actual, ids)
                    or len(np.unique(actual)) != len(ids) or reg.shape != ids.shape
                    or seeds.shape != (5, len(ids)) or seeds.dtype.kind not in "fiu"
                    or reg.dtype.kind not in "fiu" or not np.isfinite(seeds).all()
                    or not np.isfinite(reg).all() or not np.array_equal(reg, seeds.mean(axis=0))):
                raise Invalid("invalid IDs/order/shape/finite values/five-seed mean")
            return reg.copy(), seeds.copy()
    except (ValueError, OSError, KeyError) as exc:
        raise Invalid(f"unreadable raw prediction: {path}") from exc


def publish_fold(raw, final, metadata, evidence):
    final = Path(final)
    if final.exists():
        raise Invalid(f"refusing fold overwrite: {final}")
    reg, seeds = validate_raw(raw, metadata["mvt_id"])
    stage = final.parent / (".publish-" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    np.savez(stage / "prediction.npz", **metadata, reg2=reg, per_seed=seeds)
    write_json(stage / "manifest.json", {"evidence": evidence, "rows": len(reg),
                                          "sha256": sha(stage / "prediction.npz")})
    verify_fold(stage, metadata, evidence)
    os.rename(stage, final)


def verify_fold(final, metadata, evidence):
    final = Path(final)
    try:
        m = read_json(final / "manifest.json")
        if (m["evidence"] != evidence or m["rows"] != len(metadata["mvt_id"])
                or m["sha256"] != sha(final / "prediction.npz")):
            raise Invalid("completed fold lineage or SHA changed")
        with np.load(final / "prediction.npz", allow_pickle=False) as z:
            if (set(z.files) != {*metadata, "reg2", "per_seed"}
                    or any(not np.array_equal(z[k], v) for k, v in metadata.items())
                    or z["per_seed"].shape != (5, len(metadata["mvt_id"]))
                    or not np.isfinite(z["per_seed"]).all()
                    or not np.array_equal(z["reg2"], z["per_seed"].mean(axis=0))):
                raise Invalid("completed fold metadata or exact mean changed")
    except (OSError, KeyError, ValueError) as exc:
        raise Invalid(f"invalid completed fold: {final}") from exc


def validate_log(path):
    lines = Path(path).read_text().splitlines()
    completed = [int(m.group(1)) for line in lines if (m := re.fullmatch(r"seed (\d+) done \(.+s\)", line))]
    if completed != list(range(5)) or not lines or lines[-1] != "saved submissions/lgb_reg_seeds.npz":
        raise Invalid("trainer log does not prove all five seeds completed")


def quarantine_attempts(work):
    work = Path(work)
    (work / "quarantine").mkdir(exist_ok=True)
    paths = list((work / "attempts").glob("*")) + list(work.glob("fold_*/.publish-*"))
    for path in paths:
        os.rename(path, work / "quarantine" / (path.name + "-" + uuid.uuid4().hex))


def child_environment(view):
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    return {**env, **SCIENTIFIC, "FEATURES": str(view)}


def command(work):
    return [sys.executable, "-u", str(Path(work) / "sources/seed_avg.py"), "rank", "5", str(ROUNDS)]


def packages():
    versions = {p: version(p) for p in VERSIONS}
    if versions != VERSIONS:
        raise Invalid("production package versions required; do not change active environments")
    return {"python": sys.version, "packages": versions}


def verify_sources(directory):
    for name, expected in SOURCES.items():
        if sha(Path(directory) / name) != expected:
            raise Invalid(f"source changed: {name}")


def inspect_columns(source, table, cwd):
    code = ("import json,sys,polars as pl;sys.path.insert(0,sys.argv[1]);import train as T;"
            "d=pl.DataFrame(schema=pl.read_parquet_schema(sys.argv[2]));c=T.feature_columns(d);"
            "print(json.dumps({'features':c,'categoricals':[x for x in T.CATEGORICAL if x in c],"
            "'parameters':T.REG_PARAMS}))")
    result = subprocess.run([sys.executable, "-c", code, str(source), str(table)], cwd=cwd,
                            env=child_environment(table), check=True, capture_output=True, text=True)
    columns = json.loads(result.stdout.strip().splitlines()[-1])
    if (len(columns["features"]) != 170 or digest(columns["features"]) != FEATURE_COLUMNS_SHA
            or digest(columns["categoricals"]) != CATEGORY_COLUMNS_SHA
            or columns["parameters"] != PARAMETERS):
        raise Invalid("production feature/category/parameter lineage differs")
    return columns


def resource_guard():
    if sys.platform != "linux" or set(os.sched_getaffinity(0)) != {0, 2, 4, 6, 8, 10} or os.nice(0) < 10:
        raise Invalid("run requires Linux CPU affinity 0,2,4,6,8,10 and nice >= 10")
    relative = next((line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::")), None)
    if relative is None:
        raise Invalid("cgroup v2 with MemoryMax <= 24 GiB and MemorySwapMax=0 required")
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limit = (group / "memory.max").read_text().strip()
    swap = (group / "memory.swap.max").read_text().strip()
    if limit == "max" or int(limit) > 24 * 1024**3 or swap != "0":
        raise Invalid("service must set MemoryMax <= 24 GiB and MemorySwapMax=0")
    return {"affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0),
            "memory_max_bytes": int(limit), "memory_swap_max_bytes": 0}


def prepare(root, work, registration):
    if not re.fullmatch("[a-f0-9]{40}", registration):
        raise Invalid("full registration commit required")
    features, rank = root / "data/features_fin1b.parquet", root / "submissions/lgb_reg_seeds_v870.npz"
    verify_sources(root / "src")
    if sha(features) != FEATURE_SHA or sha(rank) != RANK_SHA:
        raise Invalid("not the pinned FIN1b table / production five-seed ranking artifact")
    for name, expected in PROOFS.items():
        if sha(root / "data" / name) != expected:
            raise Invalid(f"production evidence changed: {name}")
    validation = read_json(root / "data/validation_v870.json")
    if validation["best"]["reg"] != 2109 or int(validation["best"]["reg"] * 1.1) != ROUNDS:
        raise Invalid("historical production round selection differs")
    runtime = packages()
    if (work / "manifest.json").exists():
        verify_run(work, registration)
        return
    if any(p.name != ".writer.lock" for p in work.iterdir()):
        raise Invalid("incomplete preparation preserved; use a fresh work directory")
    for directory in ("sources", "proofs", "inputs", "attempts", "logs"):
        (work / directory).mkdir()
    for name in SOURCES:
        shutil.copyfile(root / "src" / name, work / "sources" / name)
    for name in PROOFS:
        shutil.copyfile(root / "data" / name, work / "proofs" / name)
    columns = inspect_columns(work / "sources", features, work)
    shutil.copyfile(Path(__file__), work / "runner.py")
    frame = pl.read_parquet(features)
    ranking_ids = frame.filter(pl.col("split") == "rank")[ID].to_numpy()
    if len(ranking_ids) != 344841:
        raise Invalid("production ranking coverage differs")
    validate_raw(rank, ranking_ids)
    frame = frame.filter(pl.col("split") == "train")
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
        restored = pl.read_parquet(stage / "features.parquet")
        if not view.equals(restored):
            raise Invalid("fold view changed during serialization")
        inspect_columns(work / "sources", stage / "features.parquet", work)
        del restored, view
        inputs[name] = {"rows": len(metadata["mvt_id"]),
                        **{file: sha(stage / file) for file in ("features.parquet", "metadata.npz")}}
        os.rename(stage, work / "inputs" / name)
        (work / name).mkdir()
        print(f"prepared {name}: {inputs[name]['rows']} heldout rows", flush=True)
    del frame
    verify_sources(root / "src")
    verify_sources(work / "sources")
    if sha(features) != FEATURE_SHA or sha(rank) != RANK_SHA:
        raise Invalid("original feature / ranking input changed during preparation")
    for name, expected in PROOFS.items():
        if sha(root / "data" / name) != expected or sha(work / "proofs" / name) != expected:
            raise Invalid("production evidence changed during preparation")
    manifest = {"protocol": PROTOCOL, "registration": registration, "feature_sha256": FEATURE_SHA,
                "production_rank_sha256": RANK_SHA, "source_sha256": SOURCES, "proof_sha256": PROOFS,
                "runner_sha256": sha(work / "runner.py"), "environment": SCIENTIFIC, "runtime": runtime,
                "columns": columns, "inputs": inputs, "rounds": ROUNDS, "seeds": SEEDS,
                "bagging_seeds": list(range(2000, 2005)), "feature_fraction_seeds": list(range(3000, 3005)),
                "schedule": list(range(1, 7)), "parameters": PARAMETERS,
                "purpose": "Retrospective production-matching component OOF; not nested or full-chain CV",
                "round_selection": "Fixed 2319 from historical Jan/Jul best=2109 times 1.1; retrospective"}
    manifest["lineage"] = digest(manifest)
    write_json(work / "manifest.json", manifest)


def verify_run(work, registration):
    m = read_json(work / "manifest.json")
    body = {k: v for k, v in m.items() if k != "lineage"}
    if (not re.fullmatch("[a-f0-9]{40}", registration) or digest(body) != m["lineage"]
            or m["protocol"] != PROTOCOL or m["registration"] != registration
            or m["source_sha256"] != SOURCES or m["proof_sha256"] != PROOFS
            or m["environment"] != SCIENTIFIC or m["runtime"] != packages()
            or m["runner_sha256"] != sha(Path(__file__)) or m["runner_sha256"] != sha(work / "runner.py")
            or m["feature_sha256"] != FEATURE_SHA or m["production_rank_sha256"] != RANK_SHA
            or m["rounds"] != ROUNDS or m["seeds"] != SEEDS or m["parameters"] != PARAMETERS
            or m["schedule"] != list(range(1, 7))
            or set(m["inputs"]) != {f"fold_{f}_{f+6}" for f in range(1, 7)}
            or digest(m["columns"]["features"]) != FEATURE_COLUMNS_SHA
            or digest(m["columns"]["categoricals"]) != CATEGORY_COLUMNS_SHA):
        raise Invalid("run manifest, environment, package or wrapper lineage changed")
    verify_sources(work / "sources")
    for name, expected in PROOFS.items():
        if sha(work / "proofs" / name) != expected:
            raise Invalid("production evidence snapshot changed")
    for name, record in m["inputs"].items():
        verify_input(work, name, record)
    return m


def verify_input(work, name, record):
    for file in ("features.parquet", "metadata.npz"):
        if sha(work / "inputs" / name / file) != record[file]:
            raise Invalid(f"pinned fold input changed: {name}/{file}")


def run(work, registration):
    resources = resource_guard()
    m = verify_run(work, registration)
    quarantine_attempts(work)
    hashes = {}
    for fold in range(1, 7):
        name = f"fold_{fold}_{fold+6}"
        with np.load(work / "inputs" / name / "metadata.npz", allow_pickle=False) as z:
            metadata = {key: z[key] for key in z.files}
        evidence = {"lineage": m["lineage"], "fold": fold, "rounds": ROUNDS, "seeds": SEEDS}
        final = work / name / "complete"
        if final.exists():
            verify_fold(final, metadata, evidence)
            print(f"verified completed {name}", flush=True)
        else:
            resource_guard()
            verify_sources(work / "sources")
            verify_input(work, name, m["inputs"][name])
            view = work / "inputs" / name / "features.parquet"
            attempt = work / "attempts" / f"{name}-{uuid.uuid4().hex}"
            (attempt / "submissions").mkdir(parents=True)
            env = child_environment(view)
            started = time.monotonic()
            write_json(attempt / "invocation.json", {"evidence": evidence, "environment": env,
                       "command": command(work), "resources": resources})
            with (attempt / "fit.log").open("x") as log:
                result = subprocess.run(command(work), cwd=attempt, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - started
            write_json(attempt / "exit.json", {"returncode": result.returncode, "seconds": elapsed})
            if result.returncode:
                raise Invalid(f"fit failed rc={result.returncode}; preserved {attempt}")
            resource_guard()
            verify_sources(work / "sources")
            verify_input(work, name, m["inputs"][name])
            validate_log(attempt / "fit.log")
            publish_fold(attempt / "submissions/lgb_reg_seeds.npz", final, metadata, evidence)
            os.rename(attempt, work / "logs" / attempt.name)
            print(f"completed {fold}/6 folds ({fold*5}/30 fits), seconds={elapsed:.1f}; "
                  f"remaining_at_this_rate_hours={elapsed*(6-fold)/3600:.2f} (provisional)", flush=True)
        hashes[str(fold)] = sha(final / "prediction.npz")
    summary = {"lineage": m["lineage"], "fits": 30, "folds": 6, "quality_gate": None, "fold_sha256": hashes}
    if (work / "complete.json").exists():
        if read_json(work / "complete.json") != summary:
            raise Invalid("completed run summary differs")
    else:
        write_json(work / "complete.json", summary)
    print("All 30 fits and six complete means verified. Retrospective component OOF only.", flush=True)


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
    except (Invalid, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(2)
