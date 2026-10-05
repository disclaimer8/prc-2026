"""Fixed six-fold Mac preparation -> verified home ROCm queue; retains all data."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import mlp_gpu_component_oof as M

HOME_ROOT = "/home/a1/prc-2026"
HOST = "home-pc-lan"
WORK = "data/mlp_split_20261005"


def call(argv, capture=False):
    return subprocess.run([str(a) for a in argv], check=True, text=True,
                          stdout=subprocess.PIPE if capture else None)


def remote(argv, capture=False):
    return call(["ssh", HOST, shlex.join([str(a) for a in argv])], capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registration-commit", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / WORK
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".queue.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return queue(root, work, args.registration_commit)


def queue(root, work, registration):
    script = root / "scripts/mlp_gpu_component_oof.py"
    hp = f"{HOME_ROOT}/.venv-rocm/bin/python"
    hs = f"{HOME_ROOT}/scripts/mlp_gpu_component_oof.py"
    remote(["mkdir", "-p", f"{HOME_ROOT}/{WORK}/inputs", f"{HOME_ROOT}/{WORK}/outputs"])
    for fold in range(1, 7):
        name = f"fold_{fold}_{fold+6}"
        local = work / "inputs" / name
        home = f"{HOME_ROOT}/{WORK}/inputs/{name}"
        output = f"{HOME_ROOT}/{WORK}/outputs/{name}"
        if not local.exists():
            if shutil.disk_usage(root).free < 7 * 1024 ** 3:
                raise RuntimeError("Need >=7GiB free before another ~2.2GiB bundle; nothing deleted")
            call([sys.executable, script, "prepare", "--root", root, "--input", local,
                  "--fold", fold, "--registration-commit", registration,
                  "--capacity-evidence", root / "analysis/mlp_gpu_prepare_failure_2026-10-05.json",
                  "--gpu-runtime", root / "data/mlp_gpu_prepare_20261005/manifest.json"])
        local_manifest = M.verify_bundle(local, registration)
        call(["rsync", "-a", local, f"{HOST}:{HOME_ROOT}/{WORK}/inputs/"])
        verified = json.loads(remote([hp, hs, "verify", "--input", home,
                                      "--registration-commit", registration], True).stdout)
        if verified["spec_sha256"] != local_manifest["spec_sha256"]:
            raise RuntimeError("Remote transfer differs from immutable local bundle")
        unit = f"prc-mlp-split-f{fold}-20261005"
        remote(["systemd-run", "--user", "--wait", "--pipe", f"--unit={unit}",
                f"--working-directory={HOME_ROOT}", "--property=MemoryMax=12G",
                "--property=MemorySwapMax=0", "--property=CPUQuota=200%", "--property=Nice=10",
                hp, hs, "run", "--input", home, "--output", output,
                "--registration-commit", registration])
        dest = work / "outputs"
        dest.mkdir(exist_ok=True)
        call(["rsync", "-a", f"{HOST}:{output}", str(dest) + "/"])
        manifest = M.C.read_json(dest / name / "manifest.json")
        spec = local_manifest["spec"]
        evidence = {"protocol": M.PROTOCOL, "registration_commit": registration,
                    "input_spec_sha256": local_manifest["spec_sha256"], "fold": fold,
                    "seeds": [0, 1, 2], "runtime": spec["gpu_runtime"]}
        with M.np.load(local / "metadata.npz", allow_pickle=False) as z:
            meta = dict(z)
        M.verify_complete(dest / name, meta, evidence)
        print(json.dumps({"complete_fold": fold, "sha256": manifest["sha256"]}), flush=True)
    print("COMPLETE: six fixed ROCm folds copied and verified on Mac", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
