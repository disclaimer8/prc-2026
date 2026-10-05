"""E184 resource-only restart using a private snapshot of the actual server code.

Run only after the operational amendment and this launcher are committed and
the previous E184 controller/worker have been stopped by the operator. No code
in the shared src tree is changed. The original nested CV RUNNER is reused;
only LightGBM thread counts and process resource settings change.
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
import subprocess
import sys

SOURCE_NAMES = ('train.py', 'features.py', 'target_encoding.py', 'cv_fold_preds_nested.py')
CPU_SET = {0, 2, 4, 6, 8, 10}
PAIRS = tuple((a, a + 6) for a in range(1, 7))
MODEL_ENV = {
    'PROXY_SCALE': '1', 'CLF_COST_WEIGHT': '1', 'CLF_COST_MIN_SPREAD': '3600',
    'DROP_PREFIX': 'day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config',
}
RESOURCE_ENV = {
    'OMP_NUM_THREADS': '6', 'OMP_THREAD_LIMIT': '6', 'OMP_WAIT_POLICY': 'PASSIVE',
    'GOMP_SPINCOUNT': '0', 'POLARS_MAX_THREADS': '6',
    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage', 'PYTHONUNBUFFERED': '1',
}
PATCH = ('T.REG_PARAMS = {**T.REG_PARAMS, "num_threads": 6}\n'
         'T.CLF_PARAMS = {**T.CLF_PARAMS, "num_threads": 6}\n')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def worker_environment(features, output):
    # These are exactly the active model flags of the original E184 process.
    # Unrelated inherited model flags, credentials and FOLDS never enter workers.
    return {'PATH': '/usr/bin:/bin', **MODEL_ENV, **RESOURCE_ENV,
            'FEATURES': str(features), 'CV_OUT': str(output)}


def extract_runner(source):
    assignments = [node.value for node in ast.parse(source).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'RUNNER' for t in node.targets)]
    if len(assignments) != 1:
        raise ValueError('Expected exactly one literal nested-CV RUNNER')
    runner = ast.literal_eval(assignments[0])
    if not isinstance(runner, str):
        raise ValueError('Nested-CV RUNNER must be source text')
    return runner


def resource_program(runner):
    anchor = 'import train as T\n'
    if runner.count(anchor) != 1:
        raise ValueError('Expected one train import before injecting resource limits')
    return runner.replace(anchor, anchor + PATCH, 1)


def validate_launch(retired_pids, *, proc_root=Path('/proc')):
    if not hasattr(os, 'sched_getaffinity') or set(os.sched_getaffinity(0)) != CPU_SET:
        raise RuntimeError('Launch under taskset -c 0,2,4,6,8,10 (six distinct physical cores)')
    for pid in retired_pids:
        if pid <= 0 or (proc_root / str(pid)).exists():
            raise RuntimeError(f'Previous E184 PID {pid} must be fully exited before starting another CV')


def prepare_snapshot(root, snapshot, output, registration_commit):
    root, snapshot, output = (Path(p).resolve() for p in (root, snapshot, output))
    if not re.fullmatch('[a-f0-9]{40}', registration_commit):
        raise ValueError('Full operational registration commit is required')
    if output == root / 'data/cv_preds_e184_hz' or snapshot == output:
        raise ValueError('The old E184 output and snapshot directory cannot be reused')
    if snapshot.exists() or output.exists():
        raise FileExistsError('Use fresh snapshot and OOF directories; automatic resume is forbidden')
    features = root / 'data/features_fin1b_e184.parquet'
    source_hashes = {name: sha256(root / 'src' / name) for name in SOURCE_NAMES}
    feature_hash = sha256(features)
    snapshot.mkdir(parents=True, exist_ok=False)
    (snapshot / 'src').mkdir()
    (snapshot / 'data').mkdir()
    (snapshot / 'logs').mkdir()
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_NAMES:
        shutil.copy2(root / 'src' / name, snapshot / 'src' / name)
        if sha256(snapshot / 'src' / name) != source_hashes[name]:
            raise RuntimeError(f'Source changed while copying snapshot: {name}')
    (snapshot / 'data/features_fin1b_e184.parquet').symlink_to(features)
    shutil.copy2(__file__, snapshot / 'launcher.py')
    manifest = {'experiment': 'E184', 'scope': 'resources only; original registration 254',
                'registration_commit': registration_commit, 'source_sha256': source_hashes,
                'launcher_sha256': sha256(__file__), 'features': str(features),
                'features_sha256': feature_hash, 'output': str(output),
                'cpu_affinity': sorted(CPU_SET), 'environment': worker_environment(features, output),
                'lightgbm_threads': {'REG_PARAMS': 6, 'CLF_PARAMS': 6},
                'folds': [list(pair) for pair in PAIRS], 'python': sys.version,
                'python_executable': sys.executable}
    with open(snapshot / 'manifest.json', 'x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return manifest


def run(root, snapshot, output, registration_commit, retired_pids):
    validate_launch(retired_pids)
    manifest = prepare_snapshot(root, snapshot, output, registration_commit)
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    original = extract_runner((snapshot / 'src/cv_fold_preds_nested.py').read_text())
    program = resource_program(original)
    (snapshot / 'worker.py').write_text(program)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    for a, b in PAIRS:
        if sha256(manifest['features']) != manifest['features_sha256']:
            raise RuntimeError('Feature data changed after snapshot registration')
        print(f'E184 fold {a}+{b}: starting with six LightGBM threads and passive OpenMP wait', flush=True)
        with open(snapshot / f'logs/fold_{a}_{b}.log', 'x') as log:
            process = subprocess.Popen([sys.executable, '-u', '-c', program, f'{a},{b}'],
                                       cwd=snapshot, env=manifest['environment'], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        if code or not (output / f'fold_{a}_{b}.npz').is_file():
            raise RuntimeError(f'E184 fold {a}+{b} failed (exit {code}); no gate may be run')
    expected = {f'fold_{a}_{b}.npz' for a, b in PAIRS}
    if {p.name for p in output.glob('*.npz')} != expected:
        raise RuntimeError('Expected exactly six OOF folds before audit')
    if sha256(manifest['features']) != manifest['features_sha256']:
        raise RuntimeError('Feature data changed during CV')
    with open(snapshot / 'completed.json', 'x') as stream:
        json.dump({'status': 'six folds saved; strict audit and original gate still required',
                   'outputs': {name: sha256(output / name) for name in sorted(expected)}}, stream, indent=2)
    print('All six E184 resource-restart folds saved. Apply strict audit then original registration 254 gate.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/prc-2026'))
    parser.add_argument('--snapshot', type=Path, default=Path('/root/prc-2026/data/e184_resource_20261005'))
    parser.add_argument('--out', type=Path, default=Path('/root/prc-2026/data/cv_preds_e184_hz_6t'))
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--retired-pids', required=True, help='comma-separated previous controller and worker PIDs')
    args = parser.parse_args()
    retired_pids = [int(p) for p in args.retired_pids.split(',')]
    run(args.root, args.snapshot, args.out, args.registration_commit, retired_pids)


if __name__ == '__main__':
    main()
