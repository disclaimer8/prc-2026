"""Oct 8 canary transport: exactly one PUT of the Jan/Jul dry-run candidate v12090.

The candidate was built by the registered final tail assembly adapter (№389,
E81 removed, fresh 10-seed E156 ensemble, cached stages replayed, 0 fits) and is
uploaded unchanged to measure the new final-pipeline code path against the real
Jan/Jul labels. inspect is local and network-free. seal exports verifiable Git
object proofs of the scientific and transport registrations. preflight/upload
verify those proofs without requiring a Git checkout on the canonical upload
host. No fitting, no weight choice, no retry.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'src'))
import numpy as np
import polars as pl
import submission_guard as G
import e185_submission as E

PROTOCOL = 'canary-v12090-uniform-tail-noE81-freshE156-janjul-single-PUT-20261008-v1'
SELF = 'scripts/submission_transport_20261008.py'
DAY = '2026-10-08'
CUTOFF = datetime(2026, 10, 9, tzinfo=timezone.utc)
VERSION = 12090
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
CANDIDATE_SHA = '180a185e232bfb5c13fe4f2dd518558f30005fc716024b2c77498665e9ed5fe3'
SCIENTIFIC_REGISTRATION = '9bd1f2e0d50381348eebffcf4de1ca068075077f'
SCIENTIFIC_PROTOCOL = 'final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1'
BUILDER = 'scripts/final_tail_assembly_20261008.py'
SOURCES = {
    BUILDER: '9ff246b8b22b9610404d99703ce392f907bd09416f9283822ebf72889f303c0f',
    'src/submission_guard.py': 'e2815dbbc2e16edbbe6140de3fbbb6cd4fcd937c7277bcf508ce4837595d0c57',
    'src/e185_submission.py': 'b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c',
    'src/e185_single_put_upload.py': 'd7cc42d6ff5a5b16a3eb8be5858308596feb3268d00b022cfae4d1339c8354a8',
}
REFERENCE = 'submissions/gentle-igloo_v12075.parquet'
REFERENCE_RMSE = 230.9387
INPUTS = {
    'data/prc-2026-datasets/submitting.parquet': '0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b',
    REFERENCE: '90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2',
}
RECEIPT = 'analysis/final_tail_assembly_20261008/dryrun_janjul/complete.json'
SERVED = 'analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'
RECEIPTS = {
    RECEIPT: 'da219ef0c81a42f398d9f9f21795dd7754ebb9e4db891940091316a6369ad633',
    SERVED: '80f5f261dd4d7be9ba6a15b951803d5025cefa42de3634dc6908e629d8797358',
}
# Pinned from analysis/final_tail_assembly_20261008/dryrun_janjul/validation_v12090_vs_v12075.json.
EXPECTED = dict(rows=344841, served_rows=42, changed=313616, rms_vs_v12075=18.208699347925087, max_abs_diff=2058.0)
FORECAST = dict(reference=REFERENCE_RMSE, central=231.6, corridor=[231.1, 232.3],
                gram_range=[212.7300, 249.1474], uncorrelated=231.6554,
                note='expert corridor, not a confidence interval; E81 alone was worth about -0.66 s on the board')


def now():
    return datetime.now(timezone.utc)


def require_window():
    if now().date().isoformat() != DAY or (CUTOFF-now()).total_seconds() < 120:
        raise G.GuardError('Oct 8 UTC only, with at least 120 seconds before cutoff')


def object_hash(kind, data):
    return hashlib.sha1(kind.encode()+b' '+str(len(data)).encode()+b'\0'+data).hexdigest()


def tree_entries(data):
    result = {}
    offset = 0
    while offset < len(data):
        end = data.index(b'\0', offset)
        mode, name = data[offset:end].split(b' ', 1)
        oid = data[end+1:end+21]
        if len(oid) != 20 or name in result:
            raise G.GuardError('invalid Git tree proof')
        result[name] = (mode, oid.hex())
        offset = end+21
    return result


def export_commit(root, commit, paths):
    def git(kind, oid):
        return subprocess.check_output(['git', 'cat-file', kind, oid], cwd=root)
    raw = git('commit', commit)
    tree = raw.split(b'\n', 1)[0].removeprefix(b'tree ').decode()
    proof = {'commit': base64.b64encode(raw).decode(), 'trees': {}, 'blobs': {}}
    for path in paths:
        oid = tree
        parts = path.split('/')
        for i, part in enumerate(parts):
            data = git('tree', oid)
            proof['trees'][oid] = base64.b64encode(data).decode()
            mode, oid = tree_entries(data)[part.encode()]
            if i < len(parts)-1 and mode != b'40000':
                raise G.GuardError('registration path is not a tree')
        proof['blobs'][path] = base64.b64encode(git('blob', oid)).decode()
    return proof


def verify_commit(proof, commit, required):
    def decoded(value):
        return base64.b64decode(value, validate=True)
    raw = decoded(proof['commit'])
    if object_hash('commit', raw) != commit or not raw.startswith(b'tree '):
        raise G.GuardError('registration commit object differs')
    first_tree = raw.split(b'\n', 1)[0][5:].decode()
    result = {}
    for path, expected in required.items():
        oid = first_tree
        parts = path.split('/')
        for i, part in enumerate(parts):
            data = decoded(proof['trees'][oid])
            if object_hash('tree', data) != oid:
                raise G.GuardError('registration tree object differs')
            mode, oid = tree_entries(data)[part.encode()]
            if mode != (b'100644' if i == len(parts)-1 else b'40000'):
                raise G.GuardError('registration requires regular source/journal files')
        data = decoded(proof['blobs'][path])
        if object_hash('blob', data) != oid or (expected is not None and hashlib.sha256(data).hexdigest() != expected):
            raise G.GuardError('registered source or journal differs')
        result[path] = data
    return result


def registration(proof, commit, source_sha):
    if not re.fullmatch('[a-f0-9]{40}', commit or '') or not re.fullmatch('[a-f0-9]{64}', source_sha or ''):
        raise G.GuardError('full transport registration and source SHA256 required')
    if proof.get('protocol') != PROTOCOL or proof.get('registration') != commit:
        raise G.GuardError('transport registration proof differs')
    current = verify_commit(proof['transport'], commit, {SELF: source_sha, 'HYPOTHESES_LOG.md': None})
    original = verify_commit(proof['scientific'], SCIENTIFIC_REGISTRATION,
                             {BUILDER: SOURCES[BUILDER], 'HYPOTHESES_LOG.md': None})
    if (PROTOCOL.encode() not in current['HYPOTHESES_LOG.md']
            or SCIENTIFIC_PROTOCOL.encode() not in original['HYPOTHESES_LOG.md']):
        raise G.GuardError('code and protocol are not registered together in journal')
    if G.sha256(Path(__file__).resolve()) != source_sha:
        raise G.GuardError('active transport differs from registration')


def load_json(root, name):
    return json.loads((root/name).read_text())


def validate(root):
    """Local, network-free checks of the unchanged candidate against pinned receipts."""
    G.verify_hashes(root, {**SOURCES, **INPUTS, **RECEIPTS, CANDIDATE: CANDIDATE_SHA})
    for path, module in (('src/submission_guard.py', G), ('src/e185_submission.py', E)):
        if G.sha256(Path(module.__file__).resolve()) != SOURCES[path]:
            raise G.GuardError('imported runtime differs: '+path)
    path = root/CANDIDATE
    if not 0 < path.stat().st_size <= 5*1024**2:
        raise G.GuardError('candidate must fit a single S3 PUT')
    receipt, served = load_json(root, RECEIPT), load_json(root, SERVED)
    fixed = dict(output_sha256=CANDIDATE_SHA, protocol=SCIENTIFIC_PROTOCOL, fits_performed=0,
                 e81_applied=False, donor_rows_copied=0, rows=EXPECTED['rows'], uploaded=False,
                 served_e156_ids_sha256=served.get('ids_sha256'))
    if any(receipt.get(k) != v for k, v in fixed.items()):
        raise G.GuardError('assembly receipt differs from the registered dry run')
    if (served.get('protocol') != SCIENTIFIC_PROTOCOL or served.get('rows') != EXPECTED['served_rows']
            or len(served['ids']) != EXPECTED['served_rows'] or len(served['values']) != EXPECTED['served_rows']):
        raise G.GuardError('E156 served-ID manifest differs')
    template = pl.read_parquet(root/E.TEMPLATE)
    candidate = pl.read_parquet(path)
    reference = pl.read_parquet(root/REFERENCE)
    for name, frame in (('candidate', candidate), ('reference', reference)):
        if frame.schema != template.schema or frame.columns != template.columns:
            raise G.GuardError(f'{name}: schema differs from template')
        G.exact_ids(frame[G.ID].to_numpy(), template[G.ID].to_numpy(), EXPECTED['rows'])
        values = frame[G.TARGET].to_numpy()
        if frame[G.TARGET].null_count() or values.dtype.kind not in 'iu' or not np.isfinite(values).all() or values.min() < 0:
            raise G.GuardError(f'{name}: predictions must be finite non-negative integers')
    c = candidate.sort(G.ID)
    ids = c[G.ID].to_numpy().astype(np.int64)
    pred = c[G.TARGET].to_numpy().astype(np.int64)
    served_ids = np.asarray(served['ids'], dtype=np.float64)
    if np.any(served_ids != np.floor(served_ids)):
        raise G.GuardError('served IDs must be exact integers')
    pos = np.searchsorted(ids, served_ids.astype(np.int64))
    if pos.max() >= len(ids) or not np.array_equal(ids[pos], served_ids.astype(np.int64)):
        raise G.GuardError('served E156 IDs are missing from the candidate')
    if not np.array_equal(pred[pos], np.asarray(served['values'], dtype=np.int64)):
        raise G.GuardError('the 42 fresh E156 values were not preserved exactly')
    r = reference.sort(G.ID)[G.TARGET].to_numpy().astype(np.int64)
    delta = (pred - r).astype(float)
    changed = int(np.count_nonzero(delta))
    rms = float(np.sqrt(np.mean(delta**2)))
    max_abs = float(np.abs(delta).max())
    if (changed != EXPECTED['changed'] or abs(rms-EXPECTED['rms_vs_v12075']) > 1e-6
            or max_abs != EXPECTED['max_abs_diff']):
        raise G.GuardError('difference against v12075 differs from the registered dry run')
    gram = [float(np.sqrt(max(REFERENCE_RMSE**2 - 2*REFERENCE_RMSE*rms + rms**2, 0.0))),
            float(np.sqrt(REFERENCE_RMSE**2 + 2*REFERENCE_RMSE*rms + rms**2))]
    return dict(rows=EXPECTED['rows'], served_rows=EXPECTED['served_rows'], changed=changed, rms_vs_v12075=rms,
                max_abs_diff=max_abs, size_bytes=path.stat().st_size, sha256=CANDIDATE_SHA,
                gram_range=[round(gram[0], 4), round(gram[1], 4)],
                uncorrelated=round(float(np.sqrt(REFERENCE_RMSE**2 + rms**2)), 4), forecast=FORECAST)


def preflight(root, proof_path, commit, source_sha):
    proof = json.loads(proof_path.read_text())
    registration(proof, commit, source_sha)
    G.verify_hashes(root, {SELF: source_sha})
    return validate(root)


def upload(root, proof_path, commit, source_sha):
    if root != Path('/root/prc-2026') or Path(sys.prefix).resolve() != root/'.venv-upload':
        raise G.GuardError('canonical server root and existing .venv-upload required')
    require_window()
    checks = preflight(root, proof_path, commit, source_sha)
    from e185_single_put_upload import OnePutHTTP
    from minio import Minio
    if G.sha256(Path(inspect.getfile(OnePutHTTP)).resolve()) != SOURCES['src/e185_single_put_upload.py']:
        raise G.GuardError('imported single PUT transport differs')
    path = root/CANDIDATE
    E.load_upload_credentials(root)
    http = OnePutHTTP()
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
                   secret_key=os.environ['PRC_SECRET_KEY'], secure=True, http_client=http)
    G.DAY = DAY  # This process only; shared guard source remains frozen.

    def inventory():
        return G.fetch_leaderboard(), E.bucket_inventory(root)

    def timeout(_signum, _frame):
        raise TimeoutError('single PUT expired')

    def put():
        require_window()
        G.verify_hashes(root, {SELF: source_sha, CANDIDATE: CANDIDATE_SHA, **SOURCES, **RECEIPTS})
        previous = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo', path.name, str(path), num_parallel_uploads=1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    print(f'TRANSPORT {PROTOCOL}; registration {commit}; source {source_sha}', flush=True)
    print(json.dumps(dict(preflight=checks), sort_keys=True), flush=True)
    started = now()
    try:
        used = G.single_attempt(root/'data/submission_upload_ledger.jsonl', path.name, CANDIDATE_SHA,
                                now=now, inventory=inventory, upload=put)
    finally:
        http.clear()
    print(json.dumps(dict(uploaded=True, version=VERSION, filename=path.name, sha256=CANDIDATE_SHA,
                          started_utc=started.isoformat(), finished_utc=now().isoformat(),
                          slots_used=used, network_put_attempts=1), sort_keys=True), flush=True)
    print(f'UPLOADED v{VERSION}; accounted {used}/5 UTC-day slots', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inspect', 'seal', 'preflight', 'upload'])
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--registration-commit')
    p.add_argument('--source-sha256')
    p.add_argument('--registration-proof', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    try:
        if a.action == 'inspect':
            print(json.dumps(dict(protocol=PROTOCOL, version=VERSION, registered_transport=False,
                                  network_used=False, checks=validate(root)), sort_keys=True))
        elif a.action == 'seal':
            if not re.fullmatch('[a-f0-9]{40}', a.registration_commit or ''):
                raise G.GuardError('full registration commit required')
            proof = dict(protocol=PROTOCOL, registration=a.registration_commit,
                         transport=export_commit(root, a.registration_commit, [SELF, 'HYPOTHESES_LOG.md']),
                         scientific=export_commit(root, SCIENTIFIC_REGISTRATION, [BUILDER, 'HYPOTHESES_LOG.md']))
            registration(proof, a.registration_commit, a.source_sha256)
            print(json.dumps(proof, sort_keys=True))
        elif a.registration_proof is None:
            p.error('--registration-proof is required')
        elif a.action == 'preflight':
            print(json.dumps(dict(protocol=PROTOCOL, version=VERSION, registration=a.registration_commit,
                                  network_used=False,
                                  checks=preflight(root, a.registration_proof, a.registration_commit, a.source_sha256)),
                             sort_keys=True))
        else:
            upload(root, a.registration_proof, a.registration_commit, a.source_sha256)
    except Exception as exc:
        print('STOP: '+(str(exc) if isinstance(exc, G.GuardError) else f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
