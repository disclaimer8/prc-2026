"""Registered prepare-only RAM probe; cannot train or publish predictions.

Snapshot unchanged E100 sources and the low-memory adapter, then execute the
original main() until its first fit_nn call. Run in a dedicated 12 GiB, CPU2,
swap-disabled systemd cgroup. Never mutate a Python environment or shared src.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile

PROTOCOL = "mlp-fin1b-rocm-prepare-only-v1"
FEATURE_SHA = "f061a441f6c97c232d9a403d612471e45020eb4ad56f501c2acfd7220a7d3052"
SOURCES = {
    "e100_nn_partner.py": "181073a73f9421982c9ccdb81dd281d7881e24f7704b57c5eecac4d2eaaa059a",
    "train.py": "b47dfcaef892bc9dc7dc4a913c3f38c8c94540f4becb4c9bfb66a7fec5d3d706",
    "target_encoding.py": "ac1b35fdd31139b531ca3760d9172f78112c77ed3ca3e72631100cd252b37d6c",
    "features.py": "58e6b783df2123adf261022733e4c602dfd22ff97d2e98e9e03e8c12486a17a2",
}
FIXED_ENV = {
    "NN_DEVICE": "cuda:0", "NN_SEEDS": "3", "NN_EPOCHS": "8", "NN_WIDTH": "512",
    "DROP_PREFIX": "day_,tq_,wxr_,month", "PROXY_SCALE": "1", "PROXY_SCALE_RIDGE": "1000",
    "AOBT_GATE": "1", "REG_CLIP_HI": "7200", "REGIME_TOL": "6",
    "REGIME_EXCLUDE": "EDDF,EHAM,LSZH", "SCHED_TOL_LONG": "0", "REGIME_RATES": "0",
    "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "2", "POLARS_MAX_THREADS": "2",
    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0",
    "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
    "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def child_environment(spec, work):
    env = {k: os.environ[k] for k in ("HOME", "PATH", "TMPDIR", "LANG") if k in os.environ}
    return {**env, **FIXED_ENV, "FEATURES": spec["feature"], "NN_CTRL": spec["control"],
            "NN_OUT": str(work / "unused-output"), "FOLDS": "1"}


def runtime_probe():
    # Import order is essential with the shared native OpenMP libraries.
    code = """import json,sys,torch,lightgbm,polars,numpy
assert torch.cuda.is_available() and torch.version.hip, 'ROCm device unavailable'
p=torch.cuda.get_device_properties(0)
print(json.dumps(dict(python=sys.version,torch=torch.__version__,hip=torch.version.hip,
 device=p.name,total_memory=p.total_memory,lightgbm=lightgbm.__version__,
 polars=polars.__version__,numpy=numpy.__version__)))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                            env=child_environment({"feature": "", "control": ""}, Path(".")))
    return json.loads(result.stdout)


def prepare(root, work, registration):
    if work.exists():
        raise ValueError("fresh work directory required")
    feature = root / "data/features_fin1b.parquet"
    control = root / "data/cv_preds_fin1b_hz"
    if sha(feature) != FEATURE_SHA:
        raise ValueError("feature SHA mismatch")
    for name, digest in SOURCES.items():
        if sha(root / "src" / name) != digest:
            raise ValueError(f"original source mismatch: {name}")
    runtime = runtime_probe()
    work.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{work.name}-", dir=work.parent))
    try:
        (stage / "src").mkdir(); (stage / "scripts").mkdir()
        for name in SOURCES:
            shutil.copy2(root / "src" / name, stage / "src" / name)
        launchers = [Path(__file__).resolve(), Path(__file__).with_name("mlp_lowmem_prepare.py").resolve()]
        for path in launchers:
            shutil.copy2(path, stage / "scripts" / path.name)
        spec = {"protocol": PROTOCOL, "registration_commit": registration,
                "feature": str(feature), "feature_sha256": FEATURE_SHA, "control": str(control),
                "sources": SOURCES, "scripts": {p.name: sha(p) for p in launchers},
                "fixed_environment": FIXED_ENV, "runtime": runtime, "python": sys.executable,
                "fold": 1, "no_fits": True, "decision": "peak RSS < 11 GiB then separate fit registration"}
        write_json(stage / "manifest.json", spec)
        verify_spec(stage, registration)
        if work.exists():
            raise ValueError("work directory appeared during prepare")
        stage.rename(work)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps({"prepared": str(work), "runtime": runtime}), flush=True)


def verify_spec(work, registration):
    spec = json.loads((work / "manifest.json").read_text())
    if (spec["protocol"] != PROTOCOL or spec["registration_commit"] != registration
            or spec["sources"] != SOURCES or spec["feature_sha256"] != FEATURE_SHA
            or spec["fixed_environment"] != FIXED_ENV or spec["fold"] != 1
            or spec["no_fits"] is not True or spec["python"] != sys.executable):
        raise ValueError("prepared manifest differs from protocol")
    for directory, hashes in (("src", SOURCES), ("scripts", spec["scripts"])):
        for name, digest in hashes.items():
            if sha(work / directory / name) != digest:
                raise ValueError(f"snapshot changed: {name}")
    for name, digest in spec["scripts"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"active launcher changed: {name}")
    if sha(spec["feature"]) != FEATURE_SHA:
        raise ValueError("feature changed")
    return spec


def verify_cgroup():
    entries = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next(line[3:] for line in entries if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    max_mem, max_swap = [(group / name).read_text().strip() for name in ("memory.max", "memory.swap.max")]
    cpu_quota, cpu_period = (group / "cpu.max").read_text().split()
    if (max_mem == "max" or int(max_mem) > 12 * 1024 ** 3 or max_swap != "0"
            or cpu_quota == "max" or int(cpu_quota) > 2 * int(cpu_period)):
        raise ValueError("dedicated MemoryMax<=12GiB, MemorySwapMax=0, CPUQuota<=200% required")
    return {"path": relative, "memory_max": int(max_mem), "swap_max": int(max_swap),
            "cpu_max": [int(cpu_quota), int(cpu_period)]}


def worker(work, registration):
    spec = verify_spec(work, registration)
    cgroup = verify_cgroup()
    if any(os.environ.get(k) != v for k, v in child_environment(spec, work).items()):
        raise ValueError("worker environment differs from fixed protocol")
    import torch  # must precede e100 -> train -> lightgbm
    if not torch.cuda.is_available() or not torch.version.hip:
        raise ValueError("ROCm accelerator unavailable")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Include ROCm context and a tiny synthetic GEMM initialization in host RSS.
    # No model, labels, gradients or optimizer are involved.
    warm = torch.ones((64, 64), device="cuda:0")
    (warm @ warm).sum().item()
    torch.cuda.synchronize()
    del warm
    sys.path.insert(0, str(work / "src"))
    import e100_nn_partner as N
    from mlp_lowmem_prepare import prepare as lowmem
    phases = []
    def sample(phase):
        item = {"phase": phase, "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2}
        phases.append(item)
        print(json.dumps(item), flush=True)
    class PreparedWithoutFit(Exception):
        pass
    original_encodings = N.add_encodings
    def encodings(*args, **kwargs):
        sample("before_target_encodings")
        result = original_encodings(*args, **kwargs)
        sample("after_target_encodings")
        return result
    def adapted(fit, pred, cols):
        sample("before_numeric_prepare")
        result = lowmem(fit, pred, cols, categorical=N.T.CATEGORICAL)
        sample("after_numeric_prepare")
        return result
    def stop_before_fit(xn, xc, y, card, seed=0):
        verify_spec(work, registration)
        sample("before_first_fit_STOP")
        peak = phases[-1]["peak_rss_gib"]
        report = {"protocol": PROTOCOL, "registration_commit": registration,
                  "manifest_sha256": sha(work / "manifest.json"), "runtime": spec["runtime"],
                  "cgroup": cgroup, "fit_calls": 0, "phases": phases,
                  "fit_shape_numeric": list(xn.shape), "fit_shape_categories": list(xc.shape),
                  "category_cardinalities": card, "peak_rss_gib": peak,
                  "memory_gate_passed": peak < 11, "training_authorized_by_this_probe": False}
        write_json(work / "probe.json", report)
        print(json.dumps(report), flush=True)
        raise PreparedWithoutFit
    N.add_encodings, N.prepare, N.fit_nn = encodings, adapted, stop_before_fit
    sample("startup")
    try:
        N.main()
    except PreparedWithoutFit:
        return
    raise RuntimeError("original main unexpectedly completed without reaching prepare/fit boundary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--registration-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.registration_commit):
        parser.error("full pre-registration commit required")
    work, root = args.work.resolve(), args.root.resolve()
    if args.command == "prepare":
        prepare(root, work, args.registration_commit)
    elif args.command == "worker":
        worker(work, args.registration_commit)
    else:
        with (work / ".writer.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            spec = verify_spec(work, args.registration_commit)
            if (work / "probe.json").exists():
                raise ValueError("probe already exists; no implicit rerun")
            if runtime_probe() != spec["runtime"]:
                raise ValueError("runtime changed since prepare")
            subprocess.run([spec["python"], str(work / "scripts" / Path(__file__).name), "worker",
                            "--work", str(work), "--registration-commit", args.registration_commit],
                           env=child_environment(spec, work), check=True, cwd=work)


if __name__ == "__main__":
    main()
