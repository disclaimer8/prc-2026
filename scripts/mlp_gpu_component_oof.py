"""Mac prepares one immutable FIN1b tensor fold; ROCm runs original E181a fits.

No fitting in prepare. No source parquet loading in GPU run. Root owns verified
rsync and cleanup of its Mac staging, one fold at a time.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile

import cat_fin1b_oof as C
import mlp_lowmem_prepare as L
import numpy as np
import polars as pl

Invalid = C.Invalid
PROTOCOL = "mlp-fin1b-split-rocm-oof-3seed-v1"
FEATURE_SHA = C.FEATURE_SHA
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
ARRAY_NAMES = ("xnf", "xcf", "xnp", "xcp", "y_scaled", "base_p", "control_indices")


def child_environment(feature="__prepared_inputs_only__"):
    env = {k: os.environ[k] for k in ("HOME", "PATH", "TMPDIR", "LANG") if k in os.environ}
    return {**env, **FIXED_ENV, "FEATURES": str(feature)}


def launcher_paths():
    return [Path(__file__).resolve(), Path(C.__file__).resolve(), Path(L.__file__).resolve()]


def verify_environment(feature="__prepared_inputs_only__"):
    if any(os.environ.get(k) != v for k, v in child_environment(feature).items()):
        raise Invalid("worker environment differs from fixed protocol")


def runtime(device=False):
    import torch  # must precede LightGBM on Mac and Linux
    import lightgbm
    result = {"python": sys.version, "torch": torch.__version__, "hip": torch.version.hip,
              "numpy": np.__version__, "polars": pl.__version__, "lightgbm": lightgbm.__version__}
    if device:
        if not torch.cuda.is_available() or not torch.version.hip:
            raise Invalid("ROCm accelerator unavailable; CPU fallback forbidden")
        p = torch.cuda.get_device_properties(0)
        result.update(device=p.name, total_memory=p.total_memory)
        validate_gpu_runtime(result)
    return result


def validate_gpu_runtime(value, expected=None):
    if not value.get("hip") or "7900 XTX" not in value.get("device", ""):
        raise Invalid("required home ROCm RX 7900 XTX unavailable")
    if expected is not None and value != expected:
        raise Invalid("GPU runtime differs from admitted immutable bundle")


def verify_cgroup():
    relative = next(s[3:] for s in Path("/proc/self/cgroup").read_text().splitlines() if s.startswith("0::"))
    group = Path("/sys/fs/cgroup") / relative.lstrip("/")
    mem, swap = [(group / s).read_text().strip() for s in ("memory.max", "memory.swap.max")]
    quota, period = (group / "cpu.max").read_text().split()
    if (mem == "max" or int(mem) > 12 * 1024 ** 3 or swap != "0"
            or quota == "max" or int(quota) > 2 * int(period)):
        raise Invalid("MemoryMax<=12GiB, MemorySwapMax=0, CPUQuota<=200% required")
    return {"path": relative, "memory_max": int(mem), "swap_max": int(swap),
            "cpu_max": [int(quota), int(period)]}


def verify_bundle(bundle, registration):
    bundle = Path(bundle)
    manifest = C.read_json(bundle / "manifest.json")
    spec = manifest["spec"]
    if (manifest.get("spec_sha256") != C.digest(spec) or spec.get("protocol") != PROTOCOL
            or spec.get("registration_commit") != registration or spec.get("feature_sha256") != FEATURE_SHA
            or spec.get("source_sha256") != SOURCES or spec.get("fixed_environment") != FIXED_ENV
            or spec.get("seeds") != [0, 1, 2] or spec.get("fold") not in range(1, 7)
            or set(spec.get("array_sha256", {})) != set(ARRAY_NAMES)):
        raise Invalid("tensor manifest differs from fixed protocol")
    validate_gpu_runtime(spec["gpu_runtime"])
    for name, digest in SOURCES.items():
        if C.sha(bundle / "src" / name) != digest:
            raise Invalid("snapshot source changed")
    active = {p.name: p for p in launcher_paths()}
    if set(spec.get("launcher_sha256", {})) != set(active):
        raise Invalid("launcher inventory differs")
    for name, digest in spec["launcher_sha256"].items():
        if C.sha(bundle / "scripts" / name) != digest or C.sha(active[name]) != digest:
            raise Invalid("snapshot or active launcher changed")
    for name, digest in spec["array_sha256"].items():
        if C.sha(bundle / f"{name}.npy") != digest:
            raise Invalid(f"prepared tensor changed: {name}")
    for name, key in (("metadata.npz", "metadata_sha256"), ("capacity_evidence.json", "capacity_evidence_sha256")):
        if C.sha(bundle / name) != spec[key]:
            raise Invalid(f"bundle evidence changed: {name}")
    return manifest


def load_arrays(bundle, spec):
    arrays = {n: np.load(Path(bundle) / f"{n}.npy", allow_pickle=False) for n in ARRAY_NAMES}
    with np.load(Path(bundle) / "metadata.npz", allow_pickle=False) as z:
        meta = dict(z)
    validate_arrays(arrays, meta, spec)
    return arrays, meta


def validate_arrays(a, meta, spec):
    n, p = len(a["xnf"]), len(a["xnp"])
    card, sd = spec.get("card", []), spec.get("sd")
    if (n < 4096 or p < 1 or not card or not isinstance(sd, (int, float))
            or not np.isfinite(sd) or sd <= 0 or any(not isinstance(k, int) or k < 1 for k in card)):
        raise Invalid("invalid prepared size/cardinalities/normalization")
    for name, rows in (("xnf", n), ("xnp", p)):
        x = a[name]
        if x.ndim != 2 or x.dtype != np.float32 or x.shape[1] < 1 or not x.flags.c_contiguous:
            raise Invalid("numeric matrix shape/dtype/layout mismatch")
        for start in range(0, rows, 65536):
            if not np.isfinite(x[start:start+65536]).all():
                raise Invalid("numeric matrix is nonfinite")
    if a["xnf"].shape[1] != a["xnp"].shape[1]:
        raise Invalid("train/prediction numeric widths differ")
    for name, rows in (("xcf", n), ("xcp", p)):
        x = a[name]
        if x.shape != (rows, len(card)) or x.dtype != np.int64 or not x.flags.c_contiguous:
            raise Invalid("categorical matrix shape/dtype/layout mismatch")
        for j, k in enumerate(card):
            if x[:, j].min() < 0 or x[:, j].max() >= k:
                raise Invalid("categorical code exceeds cardinality")
    for name, rows in (("y_scaled", n), ("base_p", p)):
        if a[name].shape != (rows,) or a[name].dtype != np.float64 or not np.isfinite(a[name]).all():
            raise Invalid("scaled target/proxy shape, dtype or finiteness mismatch")
    idx = a["control_indices"]
    if idx.dtype != np.int64 or idx.shape != (p,) or not np.array_equal(np.sort(idx), np.arange(p)):
        raise Invalid("control permutation invalid")
    if (set(meta) != {"mvt_id", "y", "month"} or meta["mvt_id"].dtype != np.float64
            or not C.integral_ids(meta["mvt_id"]) or len(np.unique(meta["mvt_id"])) != p
            or any(v.shape != (p,) for v in meta.values()) or not np.isfinite(meta["y"]).all()
            or not np.isin(meta["month"], [spec["fold"], spec["fold"]+6]).all()):
        raise Invalid("held-out metadata IDs, target or month mismatch")


def produce(root, bundle, registration, fold, capacity_evidence, gpu_runtime):
    """Preprocessing expressions match original regular_candidate; no fitting."""
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    feature = root / "data/features_fin1b.parquet"
    verify_environment(feature)
    if bundle.exists() or fold not in range(1, 7):
        raise Invalid("fresh tensor bundle and fixed month pair required")
    if C.sha(feature) != FEATURE_SHA or any(C.sha(root / "src" / n) != h for n, h in SOURCES.items()):
        raise Invalid("production feature/source SHA mismatch")
    capacity = C.read_json(capacity_evidence)
    if not capacity:
        raise Invalid("failed probe284 evidence required")
    gpu = C.read_json(gpu_runtime)
    gpu = gpu.get("runtime", gpu)
    validate_gpu_runtime(gpu)
    control = root / "data/cv_preds_fin1b_hz" / f"fold_{fold}_{fold+6}.npz"
    control_sha = C.sha(control)
    producer_runtime = runtime()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.partial-", dir=bundle.parent))
    (stage / "src").mkdir(); (stage / "scripts").mkdir()
    for name in SOURCES:
        shutil.copy2(root / "src" / name, stage / "src" / name)
    for path in launcher_paths():
        shutil.copy2(path, stage / "scripts" / path.name)
    shutil.copy2(capacity_evidence, stage / "capacity_evidence.json")
    sys.path.insert(0, str(stage / "src"))
    import e100_nn_partner as N
    tr = pl.read_parquet(feature).filter(pl.col("split") == "train")
    fit = tr.filter(~pl.col("month").is_in([fold, fold+6]))
    val = tr.filter(pl.col("month").is_in([fold, fold+6]))
    with np.load(control, allow_pickle=False) as z:
        ids, truth = z["mvt_id"].copy(), z["y"].copy()
    val_ids = val[C.ID].to_numpy()
    if (not C.integral_ids(ids) or len(ids) != len(val_ids) or len(np.unique(ids)) != len(ids)
            or not np.array_equal(np.sort(ids), np.sort(val_ids))):
        raise Invalid("control IDs differ from entire held-out population")
    order = np.argsort(val_ids)
    idx = order[np.searchsorted(val_ids[order], ids)].astype(np.int64)
    if not np.array_equal(truth, val[C.TARGET].to_numpy()[idx]):
        raise Invalid("control targets differ from authoritative feature table")
    meta = {"mvt_id": ids, "y": truth, "month": val["month"].to_numpy()[idx]}
    print(f"preparing fold {fold}+{fold+6}, no fitting", flush=True)
    lab = N.T.regime_label(fit)
    fit, pred = N.add_encodings(fit, val, lab == 0, fit_labels=lab)
    cols = N.T.feature_columns(fit)
    reg_mask = lab == 0
    base_f, base_p = N.T.proxy_base(fit.filter(pl.Series(reg_mask)), pred)
    y = np.clip(np.clip(fit[N.T.TARGET].to_numpy()[reg_mask], *N.T.REG_CLIP) - base_f, -7200, 7200)
    sd = float(y.std())
    xnf, xcf, xnp, xcp, card = L.prepare(fit.filter(pl.Series(reg_mask)), pred, cols,
                                      categorical=N.T.CATEGORICAL)
    arrays = dict(xnf=xnf, xcf=xcf, xnp=xnp, xcp=xcp, y_scaled=y / sd, base_p=base_p, control_indices=idx)
    spec = {"protocol": PROTOCOL, "registration_commit": registration, "feature_sha256": FEATURE_SHA,
            "source_sha256": SOURCES, "fixed_environment": FIXED_ENV, "fold": fold, "seeds": [0, 1, 2],
            "sd": sd, "card": card, "producer_runtime": producer_runtime, "gpu_runtime": gpu,
            "control_sha256": control_sha, "launcher_sha256": {p.name: C.sha(p) for p in launcher_paths()},
            "capacity_evidence_sha256": C.sha(stage / "capacity_evidence.json")}
    validate_arrays(arrays, meta, spec)
    for name, array in arrays.items():
        np.save(stage / f"{name}.npy", array, allow_pickle=False)
    np.savez(stage / "metadata.npz", **meta)
    spec.update(array_sha256={n: C.sha(stage / f"{n}.npy") for n in ARRAY_NAMES},
                metadata_sha256=C.sha(stage / "metadata.npz"))
    C.write_json(stage / "manifest.json", {"spec": spec, "spec_sha256": C.digest(spec)})
    verify_bundle(stage, registration)
    if C.sha(feature) != FEATURE_SHA or C.sha(control) != control_sha or bundle.exists():
        raise Invalid("inputs changed during prepare or destination appeared")
    stage.rename(bundle)
    print(json.dumps({"prepared": str(bundle), "fold": fold, "numeric_shape": list(xnf.shape),
                      "array_bytes": sum(a.nbytes for a in arrays.values()), "spec_sha256": C.digest(spec)}), flush=True)


def verify_complete(output, meta, evidence):
    manifest = C.read_json(output / "manifest.json")
    if manifest.get("evidence") != evidence or manifest.get("sha256") != C.sha(output / "prediction.npz"):
        raise Invalid("completed artifact or lineage changed")
    with np.load(output / "prediction.npz", allow_pickle=False) as z:
        if (set(z.files) != {"reg2", *meta} or not all(np.array_equal(z[k], v) for k, v in meta.items())
                or z["reg2"].shape != meta["y"].shape or not np.isfinite(z["reg2"]).all()):
            raise Invalid("completed component IDs/targets/months/prediction mismatch")
    return manifest


def publish(output, reg, meta, evidence, execution):
    if output.exists() or reg.shape != meta["y"].shape or not np.isfinite(reg).all():
        raise Invalid("completed output exists or prediction invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    np.savez(stage / "prediction.npz", reg2=reg, **meta)
    C.write_json(stage / "manifest.json", {"evidence": evidence, "execution": execution,
                                          "sha256": C.sha(stage / "prediction.npz")})
    if output.exists():
        raise Invalid("completed output appeared during publication")
    stage.rename(output)


def consume(bundle, output, registration):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    verify_environment()
    cgroup = verify_cgroup()
    actual_runtime = runtime(device=True)
    manifest = verify_bundle(bundle, registration)
    spec = manifest["spec"]
    validate_gpu_runtime(actual_runtime, spec["gpu_runtime"])
    evidence = {"protocol": PROTOCOL, "registration_commit": registration,
                "input_spec_sha256": manifest["spec_sha256"], "fold": spec["fold"],
                "seeds": [0, 1, 2], "runtime": actual_runtime}
    with np.load(bundle / "metadata.npz", allow_pickle=False) as z:
        meta = dict(z)
    if output.exists():
        verify_complete(output, meta, evidence)
        print(f"verified completed fold {spec['fold']}; no fit", flush=True)
        return
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    warm = torch.ones((64, 64), device="cuda:0")
    (warm @ warm).sum().item(); torch.cuda.synchronize(); del warm
    arrays, meta = load_arrays(bundle, spec)
    sys.path.insert(0, str(bundle / "src"))
    import e100_nn_partner as N
    if str(N.DEV) != "cuda:0" or N.EPOCHS != 8:
        raise Invalid("trainer device/epochs differ from fixed configuration")
    print(f"fold {spec['fold']}: ROCm seeds0/1/2, 8 epochs, width512; original fit_nn/predict", flush=True)
    predictions = [N.predict(N.fit_nn(arrays["xnf"], arrays["xcf"], arrays["y_scaled"], spec["card"], seed=s),
                             arrays["xnp"], arrays["xcp"]) for s in range(3)]
    reg = (np.mean(predictions, axis=0) * spec["sd"] + arrays["base_p"])[arrays["control_indices"]]
    verify_bundle(bundle, registration)
    execution = {"cgroup": cgroup, "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                 "gpu_allocated_bytes": torch.cuda.memory_allocated(), "gpu_reserved_bytes": torch.cuda.memory_reserved()}
    publish(output, reg, meta, evidence, execution)
    verify_complete(output, meta, evidence)
    print(json.dumps({"published": str(output), "rows": len(reg), **execution}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "produce", "verify", "run", "consume", "runtime"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registration-commit")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--capacity-evidence", type=Path)
    parser.add_argument("--gpu-runtime", type=Path)
    args = parser.parse_args(argv)
    if args.command == "runtime":
        print(json.dumps(runtime(device=True), sort_keys=True)); return 0
    if not args.input or not re.fullmatch(r"[0-9a-f]{40}", args.registration_commit or ""):
        parser.error("--input and full code+journal --registration-commit required")
    if args.command in {"prepare", "produce"} and (args.fold not in range(1, 7) or not args.capacity_evidence or not args.gpu_runtime):
        parser.error("prepare requires --fold 1..6, --capacity-evidence and --gpu-runtime")
    if args.command in {"run", "consume"} and not args.output:
        parser.error("run requires --output")
    if args.command in {"prepare", "run"}:
        target = (args.input if args.command == "prepare" else args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with (target.parent / f".{target.name}.writer.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = "produce" if args.command == "prepare" else "consume"
            env = child_environment(args.root.resolve() / "data/features_fin1b.parquet") if command == "produce" else child_environment()
            rest = list(sys.argv[2:] if argv is None else argv[1:])
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), command, *rest], check=True, env=env)
    elif args.command == "produce":
        produce(args.root, args.input, args.registration_commit, args.fold, args.capacity_evidence, args.gpu_runtime)
    elif args.command == "consume":
        consume(args.input, args.output, args.registration_commit)
    else:
        manifest = verify_bundle(args.input, args.registration_commit)
        print(json.dumps({"verified": str(args.input), "spec_sha256": manifest["spec_sha256"], "fold": manifest["spec"]["fold"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
