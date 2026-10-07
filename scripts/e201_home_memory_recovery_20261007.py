"""One explicitly registered E201 Home memory repair; never an automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

REG = 'd664e153b3f0732b4047011cc062f58f42581a6c'
MANIFEST = 'f8696b7326f196ba72f51e44be5a6b658c8f063fcac09e1bd54274c80218a2aa'
WORKER = '9da861964ddec3dade4c2a1438c20214ee21236388b9337b3fffb16cab594cbd'
ASSIGN = {'171': ('v44', [1, 2, 3], 6), '175': ('v44', [4, 5, 6], 6),
          'mac': ('v60', [1, 2], 8), 'home': ('v60', [3, 4], 6),
          'hetzner': ('v60', [5, 6], 6)}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--node', required=True, choices=['home'])
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    bundle, output = a.input.resolve(), a.output.resolve()
    worker = bundle / 'scripts/e201_donor_oof.py'
    if sha(bundle / 'manifest.json') != MANIFEST or sha(worker) != WORKER:
        raise ValueError('frozen E201 input/worker mismatch')
    if output.name != 'e201_home_20261007_memoryfix':
        raise ValueError('fresh named memoryfix output required')
    old = subprocess.run(['systemctl', '--user', 'show', 'prc-e201-donors-home-20261007.service',
                          '-p', 'ActiveState', '-p', 'MainPID', '-p', 'Result', '-p', 'InvocationID'],
                         capture_output=True, text=True, check=True)
    props = dict(line.split('=', 1) for line in old.stdout.splitlines() if '=' in line)
    if props != dict(ActiveState='failed', MainPID='0', Result='oom-kill', InvocationID='ebf174d4f933451987e314d91be999bd'):
        raise ValueError('expected original OOM unit not conclusively stopped: ' + str(props))
    output.mkdir(parents=True, exist_ok=False)
    donor, folds, threads = ASSIGN[a.node]
    start = time.time()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', POLARS_MAX_THREADS='2',
               OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', PYTHONUNBUFFERED='1',
               MALLOC_ARENA_MAX='2', MALLOC_TRIM_THRESHOLD_='131072')
    write(output / 'started.json', dict(node=a.node, donor=donor, folds=folds,
        threads=threads, pid=os.getpid(), python=sys.executable,
        registration_commit=REG, input_manifest_sha256=MANIFEST,
        worker_sha256=WORKER, queue_sha256=sha(__file__), original_failed_unit=props,
        technical_change='six threads plus process-local glibc arena/trim limits; unchanged frozen worker and inputs', started_unix=start))
    done = []
    try:
        for fold in folds:
            part = output / f'{donor}_fold{fold}'
            common = ['--input', str(bundle), '--output', str(part), '--registration-commit', REG]
            for action, extra in [('run-fold', ['--donor', donor, '--fold', str(fold), '--threads', str(threads)]), ('audit', [])]:
                command = [sys.executable, str(worker), action, *common, *extra]
                with (output / f'{donor}_fold{fold}_{action}.log').open('xb') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=max(1, 48 * 3600 - (time.time() - start)))
            proof = json.loads((part / 'transfer_receipt.json').read_text())
            packet = output / f'{donor}_fold{fold}_small.tar.gz'
            small = sorted(proof['small_files'])
            with tarfile.open(packet, 'x:gz') as archive:
                for name in [*small, 'complete.json', 'transfer_receipt.json']:
                    archive.add(part / name, arcname=f'{part.name}/{name}', recursive=False)
            record = dict(fold=fold, donor=donor, receipt_sha256=sha(part / 'complete.json'),
                          transfer_sha256=sha(part / 'transfer_receipt.json'),
                          packet=packet.name, packet_sha256=sha(packet))
            write(output / f'fold{fold}_ready.json', record)
            done.append(record)
            print(json.dumps(dict(event='audited_fold_ready', **record)), flush=True)
        write(output / 'complete.json', dict(node=a.node, registration_commit=REG,
            input_manifest_sha256=MANIFEST, folds=done, seconds=time.time() - start))
    except BaseException:
        write(output / 'failure.json', dict(node=a.node, completed=done,
            error=traceback.format_exc(), seconds=time.time() - start, auto_retry=False))
        raise

if __name__ == '__main__':
    main()
