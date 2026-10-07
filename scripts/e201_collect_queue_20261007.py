"""Collect audited E201 small packets; never start, stop, or retry model fits."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

REG = 'd664e153b3f0732b4047011cc062f58f42581a6c'
MANIFEST = 'f8696b7326f196ba72f51e44be5a6b658c8f063fcac09e1bd54274c80218a2aa'
WORKER = '9da861964ddec3dade4c2a1438c20214ee21236388b9337b3fffb16cab594cbd'
NODES = {
    'mac': (None, None, 'v60', (1, 2)),
    'home': ('home-pc-lan', '/home/a1/prc-2026', 'v60', (3, 4)),
    'hetzner': ('hetzner', '/root/prc-2026', 'v60', (5, 6)),
    '171': ('a1@192.168.1.171', '/home/a1/prc-2026', 'v44', (1, 2, 3)),
    '175': ('a1@192.168.1.175', '/home/a1/prc-2026', 'v44', (4, 5, 6)),
}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)

def optional(host, path):
    if host is None:
        return json.loads(path.read_text()) if path.exists() else None
    command = 'if test -f ' + shlex.quote(str(path)) + '; then cat ' + shlex.quote(str(path)) + '; else exit 3; fi'
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                       capture_output=True, text=True, timeout=45)
    if r.returncode == 3:
        return None
    if r.returncode:
        raise RuntimeError(f'SSH failed {host}: {r.stderr[:1000]}')
    return json.loads(r.stdout)

def unpack(packet, destination, expected_name):
    with tarfile.open(packet, 'r:gz') as archive:
        members = archive.getmembers()
        if sum(m.size for m in members) > 2 * 1024 ** 3:
            raise ValueError('unexpected packet size')
        names = set()
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or name.parts[0] != expected_name or member.name in names:
                raise ValueError('unsafe packet member')
            if name.suffix not in ('.json', '.parquet', '.npz'):
                raise ValueError('heavy/unexpected artifact in small packet')
            names.add(member.name)
        for member in members:
            p = destination / member.name
            p.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, p.open('xb') as out:
                shutil.copyfileobj(src, out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--state', required=True, type=Path)
    a = p.parse_args()
    root, state = a.root.resolve(), a.state.resolve()
    bundle = root / 'data/e201_inputs_20261007'
    worker = bundle / 'scripts/e201_donor_oof.py'
    if sha(bundle / 'manifest.json') != MANIFEST or sha(worker) != WORKER:
        raise ValueError('collector lineage mismatch')
    state.mkdir(parents=True, exist_ok=False)
    parts = state / 'parts'; parts.mkdir()
    start = time.time(); completed = {}
    save(state / 'started.json', dict(registration_commit=REG, input_manifest_sha256=MANIFEST,
         queue_sha256=sha(__file__), started_unix=start, timeout_hours=48))
    try:
        while len(completed) < 12:
            if time.time() - start > 48 * 3600:
                raise TimeoutError('48h bounded collector expired; no retries')
            for node, (host, remote_root, donor, folds) in NODES.items():
                base = (root if host is None else Path(remote_root)) / f'data/e201_{node}_20261007'
                failure = optional(host, base / 'failure.json')
                if failure is not None:
                    raise RuntimeError(f'Node {node} failed: {failure}')
                for fold in folds:
                    name = f'{donor}_fold{fold}'
                    if name in completed:
                        continue
                    ready = optional(host, base / f'fold{fold}_ready.json')
                    if ready is None:
                        continue
                    if ready['donor'] != donor or ready['fold'] != fold or ready['packet'] != name + '_small.tar.gz':
                        raise ValueError('unexpected packet mapping')
                    packet = state / (name + '_small.tar.gz')
                    if host is None:
                        shutil.copyfile(base / ready['packet'], packet)
                    else:
                        subprocess.run(['scp', '-o', 'BatchMode=yes', host + ':' + str(base / ready['packet']), str(packet)], check=True, timeout=900)
                    if sha(packet) != ready['packet_sha256']:
                        raise ValueError('packet checksum mismatch')
                    staging = Path(tempfile.mkdtemp(prefix='.unpack-', dir=parts))
                    unpack(packet, staging, name)
                    part = staging / name
                    if sha(part / 'complete.json') != ready['receipt_sha256'] or sha(part / 'transfer_receipt.json') != ready['transfer_sha256']:
                        raise ValueError('packet proof checksum mismatch')
                    proof = json.loads((part / 'transfer_receipt.json').read_text())
                    if proof['registration_commit'] != REG or proof['input_manifest_sha256'] != MANIFEST or proof['source_sha256'] != WORKER:
                        raise ValueError('packet proof lineage mismatch')
                    for file, digest in proof['small_files'].items():
                        if sha(part / file) != digest:
                            raise ValueError('small artifact checksum mismatch')
                    part.rename(parts / name); staging.rmdir()
                    completed[name] = ready
                    save(state / (name + '_collected.json'), ready)
                    print(json.dumps(dict(event='collected', node=node, **ready)), flush=True)
            if len(completed) < 12:
                time.sleep(60)
        command = [sys.executable, str(worker), 'collect', '--input', str(bundle),
                   '--output', str(root / 'data/e201_donors_20261007'), '--registration-commit', REG,
                   '--parts', *[str(parts / name) for name in sorted(completed)]]
        subprocess.run(command, check=True, timeout=1800)
        save(state / 'complete.json', dict(registration_commit=REG, input_manifest_sha256=MANIFEST,
            collected=completed, final_receipt_sha256=sha(root / 'data/e201_donors_20261007/complete.json'), seconds=time.time() - start))
    except BaseException:
        save(state / 'failure.json', dict(completed=completed, error=traceback.format_exc(), auto_retry=False))
        raise

if __name__ == '__main__':
    main()
