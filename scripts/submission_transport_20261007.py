"""Oct 7 transport only for the three unchanged, previously registered recipes.

inspect is local and network-free. seal exports verifiable Git object proofs of
the scientific and transport registrations. preflight/upload verify those proofs
without requiring a Git checkout on the canonical upload host. No fitting occurs.
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
import today_submissions_20261006 as T
G, E = T.G, T.E

PROTOCOL = 'fixed-E190-E188-remaining-three-transport-20261007-v1'
SELF = 'scripts/submission_transport_20261007.py'
DAY = '2026-10-07'
CUTOFF = datetime(2026, 10, 8, tzinfo=timezone.utc)
VERSIONS = (12070, 12071, 12074)
BUILD_REGISTRATION = 'cd3598e15a25db9bd6a401d38b4ad9bee6c2e862'
BUILDER = 'scripts/today_submissions_20261006.py'
SOURCES = {
    BUILDER: 'b0f7a9002760d9ca859b4b8780d08d80a98c75334b273bd996a7fa98311cb7fa',
    'src/submission_guard.py': 'e2815dbbc2e16edbbe6140de3fbbb6cd4fcd937c7277bcf508ce4837595d0c57',
    'src/e185_submission.py': 'b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c',
    'src/e185_single_put_upload.py': 'd7cc42d6ff5a5b16a3eb8be5858308596feb3268d00b022cfae4d1339c8354a8',
}
VALIDATION_INPUTS = {
    'data/prc-2026-datasets/submitting.parquet': '0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b',
    'submissions/gentle-igloo_v12061.parquet': '8d3b589213240bd6960984d6b2a46c6716617262c635ac2dc02d35b50b876015',
    'submissions/gentle-igloo_v291.parquet': 'e98595f2483919199de1032de7d2f6249275f94154a145657702e6298cd5592b',
    'submissions/gentle-igloo_v361.parquet': '1ccce805fa0877a8c6490c92cb441a5967e6aa5e5bc34eea330a8fc71a3c3291',
}
# (complete parquet SHA256, exact scientific build receipt SHA256).
ARTIFACTS = {
    12070: ('be753aeb9f752359faad3e403e63a3b7e57dbad1defd5a076cb8e7571b21575b',
            '69f3b8a707a5fac79912c3768f97b122e89ad5373575b548a37a651f7ce680a0'),
    12071: ('976b0b8617db6fd5dfeb63d42b87915356eec3f72e99972f81a6c4c9172d8a11',
            '2b9a0474020be541583aef430985704278b8fdb6bc1dce86e9c6ed2cd8a77db5'),
    12074: ('e5c78ab061492b69fe72d4a46d2e286f7a2f6b3129ebf2ae4a34731bb662374b',
            '225ea29ed31ebaf50400126b1073432b444bf0b570771800fca599e6ba168afc'),
}


def now():
    return datetime.now(timezone.utc)


def require_window():
    if now().date().isoformat() != DAY or (CUTOFF-now()).total_seconds() < 120:
        raise G.GuardError('Oct 7 UTC only, with at least 120 seconds before cutoff')


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
    original = verify_commit(proof['scientific'], BUILD_REGISTRATION, {BUILDER: SOURCES[BUILDER], 'HYPOTHESES_LOG.md': None})
    if PROTOCOL.encode() not in current['HYPOTHESES_LOG.md'] or T.PROTOCOL.encode() not in original['HYPOTHESES_LOG.md']:
        raise G.GuardError('code and protocol are not registered together in journal')
    if G.sha256(Path(__file__).resolve()) != source_sha:
        raise G.GuardError('active transport differs from registration')


def inspect_artifacts(root):
    G.verify_hashes(root, SOURCES)
    for path, module in ((BUILDER, T), ('src/submission_guard.py', G), ('src/e185_submission.py', E)):
        if G.sha256(Path(module.__file__).resolve()) != SOURCES[path]:
            raise G.GuardError('imported runtime differs: '+path)
    G.verify_hashes(root, VALIDATION_INPUTS)
    result = {}
    for version in VERSIONS:
        path = root/f'submissions/gentle-igloo_v{version}.parquet'
        rp = path.with_suffix('.manifest.json')
        expected, receipt_sha = ARTIFACTS[version]
        G.verify_hashes(root, {str(path.relative_to(root)): expected, str(rp.relative_to(root)): receipt_sha})
        receipt = json.loads(rp.read_text())
        fixed = dict(protocol=T.PROTOCOL, version=version, registration=BUILD_REGISTRATION,
                     recipe=T.RECIPES[version], forecast=T.FORECASTS[version],
                     reference=T.BEST, reference_sha256=T.BEST_SHA, sha256=expected,
                     full_chain_cv_waived_by_user=True, quality_validated_on_full_chain=False)
        if any(receipt.get(k) != v for k, v in fixed.items()):
            raise G.GuardError('scientific recipe/receipt drift')
        checks = T.validate(root, path)
        if checks != receipt['validation'] or not 0 < path.stat().st_size <= 5*1024**2:
            raise G.GuardError('complete output checks or single PUT size differ')
        result[str(version)] = dict(sha256=expected, receipt_sha256=receipt_sha, validation=checks)
    return result


def preflight(root, proof_path, commit, source_sha):
    proof = json.loads(proof_path.read_text())
    registration(proof, commit, source_sha)
    G.verify_hashes(root, {SELF: source_sha})
    return inspect_artifacts(root)


def upload(root, proof_path, commit, source_sha, version):
    if root != Path('/root/prc-2026') or Path(sys.prefix).resolve() != root/'.venv-upload':
        raise G.GuardError('canonical server root and existing .venv-upload required')
    if version not in VERSIONS:
        raise G.GuardError('only the three unchanged remaining versions are authorized')
    require_window()
    preflight(root, proof_path, commit, source_sha)
    from e185_single_put_upload import OnePutHTTP
    from minio import Minio
    if G.sha256(Path(inspect.getfile(OnePutHTTP)).resolve()) != SOURCES['src/e185_single_put_upload.py']:
        raise G.GuardError('imported single PUT transport differs')
    path = root/f'submissions/gentle-igloo_v{version}.parquet'
    rp = path.with_suffix('.manifest.json')
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
        G.verify_hashes(root, {SELF: source_sha, str(path.relative_to(root)): ARTIFACTS[version][0],
                              str(rp.relative_to(root)): ARTIFACTS[version][1], **SOURCES})
        previous = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo', path.name, str(path), num_parallel_uploads=1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    print(f'TRANSPORT {PROTOCOL}; registration {commit}; source {source_sha}', flush=True)
    try:
        used = G.single_attempt(root/'data/submission_upload_ledger.jsonl', path.name, ARTIFACTS[version][0],
                                now=now, inventory=inventory, upload=put)
    finally:
        http.clear()
    print(f'UPLOADED v{version}; accounted {used}/5 UTC-day slots', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inspect', 'seal', 'preflight', 'upload'])
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--registration-commit')
    p.add_argument('--source-sha256')
    p.add_argument('--registration-proof', type=Path)
    p.add_argument('--version', type=int, choices=VERSIONS)
    a = p.parse_args()
    root = a.root.resolve()
    try:
        if a.action == 'inspect':
            print(json.dumps(dict(protocol=PROTOCOL, registered_transport=False, network_used=False,
                                  artifacts=inspect_artifacts(root)), sort_keys=True))
        elif a.action == 'seal':
            if not re.fullmatch('[a-f0-9]{40}', a.registration_commit or ''):
                raise G.GuardError('full registration commit required')
            proof = dict(protocol=PROTOCOL, registration=a.registration_commit,
                         transport=export_commit(root, a.registration_commit, [SELF, 'HYPOTHESES_LOG.md']),
                         scientific=export_commit(root, BUILD_REGISTRATION, [BUILDER, 'HYPOTHESES_LOG.md']))
            registration(proof, a.registration_commit, a.source_sha256)
            print(json.dumps(proof, sort_keys=True))
        elif a.registration_proof is None:
            p.error('--registration-proof is required')
        elif a.action == 'preflight':
            print(json.dumps(dict(protocol=PROTOCOL, registration=a.registration_commit, network_used=False,
                                  artifacts=preflight(root, a.registration_proof, a.registration_commit, a.source_sha256)), sort_keys=True))
        else:
            upload(root, a.registration_proof, a.registration_commit, a.source_sha256, a.version)
    except Exception as exc:
        print('STOP: '+(str(exc) if isinstance(exc, G.GuardError) else type(exc).__name__), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
