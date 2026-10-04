"""E186 fixed-configuration CV. Register code + journal in git BEFORE running.

Output is a new E130 member directory, not a replacement of any existing model.
First fold is a runtime/numerical check only; no tuning using its scored labels.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import torch  # MUST precede numpy/preprocessing/e100/train/lightgbm.
import numpy as np
import polars as pl

from e186_batch_ensemble import BatchEnsembleNet, individual_mse
from e186_runtime import (backend_provenance, configure_backend, empty_cache,
                          memory_guard, select_device, synchronize)

CONFIG = {"k": 16, "widths": [512, 512, 256], "epochs": 8, "batch_size": 4096, "micro_batch_size": 512,
          "predict_batch_size": 4096, "seed": 0, "lr": 0.002, "weight_decay": 1e-5,
          "dropout": 0.1, "drop_prefix": "day_,tq_,wxr_,month", "proxy_scale": "1",
          "regime_rates": "0"}


def pin_preprocessing_environment(features):
    """Pin the E181a feature contract before importing train/target_encoding.

    Return the same flags for the provenance manifest. REGIME_RATES is read by
    target_encoding.py, not train.py, and must not inherit a shell experiment.
    """
    pinned = {"FEATURES": str(Path(features).resolve()), "DROP_PREFIX": CONFIG["drop_prefix"],
              "PROXY_SCALE": CONFIG["proxy_scale"], "REGIME_RATES": CONFIG["regime_rates"]}
    os.environ.update(pinned)
    return pinned


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    a = np.asarray(a)
    if a.dtype.hasobject:
        a = a.astype("U")
    return hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode() + np.ascontiguousarray(a).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        temp = f.name
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as f:
        np.savez(f, **arrays)
        temp = f.name
    os.replace(temp, path)


@contextmanager
def locked_manifest(manifest_path, manifest):
    """Own the output writer lock before validating or publishing run lineage."""
    manifest_path = Path(manifest_path)
    lock = manifest_path.parent / ".writer.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError("Resume rejected: configuration/source/features/backend manifest changed")
        else:
            atomic_json(manifest_path, manifest)
        yield
    finally:
        lock.unlink(missing_ok=True)


def validate_saved_fold(path, spec_hash, ids, truth):
    with np.load(path) as saved:
        if (str(saved["spec_sha256"].item()) != spec_hash or
                not np.array_equal(saved["mvt_id"], ids) or not np.array_equal(saved["y"], truth) or
                saved["reg2"].shape != truth.shape or not np.isfinite(saved["reg2"]).all()):
            raise RuntimeError(f"Unsafe existing fold: {path}")


def train_network(xn, xc, y, card, device, *, max_rss_gib=16, max_device_gib=8):
    def guard(phase):
        return memory_guard(device, max_rss_gib=max_rss_gib, max_device_gib=max_device_gib, phase=phase)
    guard("before training")
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    net = BatchEnsembleNet(xn.shape[1], card, k=CONFIG["k"], widths=tuple(CONFIG["widths"])).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    n, bs = len(y), CONFIG["batch_size"]
    if n < bs:
        raise ValueError("Training sample is smaller than a full batch")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["lr"],
                                                  total_steps=CONFIG["epochs"] * (n // bs), pct_start=0.1)
    Xn, Xc, Y = torch.from_numpy(xn), torch.from_numpy(xc), torch.from_numpy(y.astype(np.float32))
    for ep in range(CONFIG["epochs"]):
        net.train()
        perm = rng.permutation(n)
        total, started = 0.0, time.monotonic()
        for i in range(n // bs):
            optimizer.zero_grad(set_to_none=True)
            micro = CONFIG["micro_batch_size"]
            for j in range(0, bs, micro):
                idx = torch.from_numpy(perm[i * bs + j:i * bs + j + micro])
                pred = net(Xn[idx].to(device), Xc[idx].to(device))
                loss = individual_mse(pred, Y[idx].to(device)) * (micro / bs)
                loss.backward()
                total += loss.item()
            optimizer.step()
            scheduler.step()
            if not np.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            if i % 256 == 0:
                guard("training")
        synchronize(device)
        usage = guard(f"epoch {ep + 1}")
        print(f"epoch {ep + 1}: mse={total / (n // bs):.6f} seconds={time.monotonic() - started:.1f} memory={json.dumps(usage)}", flush=True)
    return net


@torch.no_grad()
def predict(net, xn, xc, *, max_rss_gib=16, max_device_gib=8):
    net.eval()
    device = next(net.parameters()).device
    memory_guard(device, max_rss_gib=max_rss_gib, max_device_gib=max_device_gib, phase="before prediction")
    out = []
    bs = CONFIG["predict_batch_size"]
    for i in range(0, len(xn), bs):
        p = net(torch.from_numpy(xn[i:i + bs]).to(device), torch.from_numpy(xc[i:i + bs]).to(device))
        out.append(p.mean(dim=1).cpu().numpy())
        if (i // bs) % 32 == 0:
            memory_guard(device, max_rss_gib=max_rss_gib, max_device_gib=max_device_gib, phase="prediction")
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="1,2,3,4,5,6")
    parser.add_argument("--features", default="data/features_fin1b.parquet")
    parser.add_argument("--control", default="data/cv_preds_fin1b_hz")
    parser.add_argument("--out", default="data/cv_e130e186be")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cuda:<index>; CPU is forbidden for CV")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-rss-gib", type=float, default=16)
    parser.add_argument("--max-device-gib", type=float, default=8)
    args = parser.parse_args()
    folds = [int(s) for s in args.folds.split(",")]
    if not folds or len(set(folds)) != len(folds) or any(f not in range(1, 7) for f in folds):
        parser.error("folds must be unique integers from 1 to 6")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    device = select_device(args.device, require_accelerator=True)
    limits = {"max_rss_gib": args.max_rss_gib, "max_device_gib": args.max_device_gib}
    def guard(phase):
        return memory_guard(device, **limits, phase=phase)
    guard("startup")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(2, args.cpu_threads))
    for key in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.cpu_threads)
    configure_backend(device, max_device_gib=args.max_device_gib)
    runtime = {**backend_provenance(device), "cpu_threads": args.cpu_threads,
               "requested_device": args.device, **limits}
    print(json.dumps({"runtime": runtime}, sort_keys=True), flush=True)
    pinned_env = pin_preprocessing_environment(args.features)
    import e100_nn_partner as N
    T = N.T
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("e186_batch_ensemble.py"), Path(__file__).with_name("e186_runtime.py"),
                    Path(N.__file__), Path(T.__file__), Path(__file__).with_name("target_encoding.py"),
                    Path(__file__).with_name("features.py")]
    # Explicit pinned preprocessing flags cover target_encoding's REGIME_RATES too.
    # Never serialize unrelated environment variables or secrets.
    env_keys = sorted(set(re.findall(r'os\.environ\.get\("([A-Z_0-9]+)"', Path(T.__file__).read_text())))
    spec = {"experiment": "E186", "config": CONFIG, "torch": torch.__version__, "numpy": np.__version__,
            "polars": pl.__version__, "runtime": runtime, "features": str(Path(args.features).resolve()),
            "features_sha256": sha_file(args.features), "control": str(Path(args.control).resolve()),
            "sources": {p.name: sha_file(p) for p in source_paths},
            "train_env": {k: os.environ.get(k) for k in env_keys}, "preprocessing_env": pinned_env}
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    manifest = {"spec_sha256": spec_hash, "spec": spec}
    manifest_path = out / "run_manifest.json"
    with locked_manifest(manifest_path, manifest):
        guard("before feature loading")
        tr = pl.read_parquet(args.features).filter(pl.col("split") == "train")
        guard("after feature loading")
        for a in folds:
            name = f"fold_{a}_{a + 6}.npz"
            target_path = out / name
            val = tr.filter(pl.col("month").is_in([a, a + 6]))
            with np.load(Path(args.control) / name) as c:
                ids, truth = c["mvt_id"].copy(), c["y"].copy()
            positions = {m: i for i, m in enumerate(val["MVT_ID_mvt"].to_numpy())}
            if len(positions) != len(val) or len(np.unique(ids)) != len(ids) or set(ids) != set(positions):
                raise ValueError(f"{name}: IDs are not exactly the same unique set")
            idx = np.array([positions[m] for m in ids])
            if not np.array_equal(val[T.TARGET].to_numpy()[idx], truth):
                raise ValueError(f"{name}: target mismatch")
            if target_path.exists():
                validate_saved_fold(target_path, spec_hash, ids, truth)
                print(f"verified resume: {name}", flush=True)
                continue
            print(f"fold {a}+{a + 6}: prepare", flush=True)
            guard("before fold preprocessing")
            fit = tr.filter(~pl.col("month").is_in([a, a + 6]))
            lab = T.regime_label(fit)
            fit, encoded_val = N.add_encodings(fit, val, lab == 0, fit_labels=lab)
            guard("after target encoding")
            cols = T.feature_columns(fit)
            regular = fit.filter(pl.Series(lab == 0))
            base_fit, base_pred = T.proxy_base(regular, encoded_val)
            y = np.clip(np.clip(regular[T.TARGET].to_numpy(), *T.REG_CLIP) - base_fit, -7200, 7200)
            sd = float(y.std())
            if not np.isfinite(y).all() or not sd > 0:
                raise ValueError("Invalid standardized residual target")
            guard("before numeric/category preprocessing")
            xnf, xcf, xnp, xcp, card = N.prepare(regular, encoded_val, cols)
            guard("after numeric/category preprocessing")
            fold_info = {"fold": name, "train_rows": len(y), "score_rows": len(truth),
                         "numeric": xnf.shape[1], "categorical_cardinalities": card,
                         "feature_columns": cols, "residual_sd": sd,
                         "fit_ids_sha256": array_sha(fit["MVT_ID_mvt"].to_numpy()),
                         "score_ids_sha256": array_sha(ids), "score_y_sha256": array_sha(truth)}
            del fit, regular, encoded_val, lab, val
            gc.collect()
            print(json.dumps({k: v for k, v in fold_info.items() if k != "feature_columns"}), flush=True)
            guard("after fold preprocessing")
            net = train_network(xnf, xcf, y / sd, card, device, **limits)
            reg = (predict(net, xnp, xcp, **limits) * sd + base_pred)[idx]
            synchronize(device)
            guard("after prediction")
            if not np.isfinite(reg).all():
                raise FloatingPointError("Nonfinite held-out prediction")
            atomic_json(out / (name + ".json"), {"spec_sha256": spec_hash, **fold_info})
            atomic_npz(target_path, mvt_id=ids, y=truth, reg2=reg,
                       spec_sha256=np.array(spec_hash), fold_info=np.array(json.dumps(fold_info, sort_keys=True)))
            print(f"saved {target_path}", flush=True)
            del net, xnf, xcf, xnp, xcp, y, base_fit, base_pred
            gc.collect()
            empty_cache(device)
            guard("after fold cleanup")


if __name__ == "__main__":
    main()
