"""Oct 9 transport of v12079: exactly one PUT of a single-change measurement on top of v12077.

v12079 = v12077 with the 42 frozen TabPFN tail rows replaced by the fresh ten-seed E156 values of the
08.10 canary (e156_served.json, same 42 IDs); E81 and every other stage unchanged (journal №433).
Built by scripts/v12078_v12079_builds_20261009.py --variant freshe156. inspect is local and network-free. seal exports
verifiable Git object proofs of the registration commit (this script + journal). preflight/
upload verify those proofs without a Git checkout on the canonical upload host. No fitting,
no weight choice, no retry. Modelled on scripts/submission_transport_20261008.py (№390).
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
import signal
import sys

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import polars as pl
import submission_guard as G
import e185_submission as E
import submission_transport_20261008 as C   # frozen canary helpers: export_commit / verify_commit

PROTOCOL = 'v12079-single-PUT-20261009-v1'
SELF = 'scripts/submission_transport_v12079_20261009.py'
DAY = '2026-10-09'
CUTOFF = datetime(2026, 10, 10, tzinfo=timezone.utc)
VERSION = 12079
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
CANDIDATE_SHA = 'd3111a7f1978e838db3c31490dba2ef923b003ee59f67d4cc815334f4660bb7e'
SCIENTIFIC_PROTOCOL = 'v12079-v12077-chain-fresh-E156-tenseed-tail-rows-zero-fit-20261009-v1'
BUILDER = 'scripts/v12078_v12079_builds_20261009.py'
SOURCES = {
    BUILDER: '4a3962360fb99aaedae051d7582f5c93c2caf1bd1dd7bc74f804388a3be930ea',
    'scripts/v12076_nn10seed_build_20261008.py': '3fc1e93fa1f4cbb9b6ec7cf2043dd9e9905e129ca24f9b039e3668ae947496f9',
    'src/submission_guard.py': 'e2815dbbc2e16edbbe6140de3fbbb6cd4fcd937c7277bcf508ce4837595d0c57',
    'src/e185_submission.py': 'b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c',
    'src/e185_single_put_upload.py': 'd7cc42d6ff5a5b16a3eb8be5858308596feb3268d00b022cfae4d1339c8354a8',
    'scripts/submission_transport_20261008.py': '79808a22e127d21709b50ab50576e023057afef61bae766e775e8fc1b26790f0',
}
REFERENCE = 'submissions/gentle-igloo_v12077.parquet'   # the parent: the difference is pinned against it
REFERENCE_RMSE = 230.4812                           # v12077 board score (journal №427)
SERVED = 'analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'
INPUTS = {
    'data/prc-2026-datasets/submitting.parquet': '0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b',
    REFERENCE: 'f06c66bfa3388b17f5b6ba2362fa45a7da1be8fd5ae2d37de47cfe0e6bf2ba44',
    'submissions/gentle-igloo_v12075.parquet': '90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2',
    SERVED: '80f5f261dd4d7be9ba6a15b951803d5025cefa42de3634dc6908e629d8797358',
}
RECEIPT = f'submissions/gentle-igloo_v{VERSION}.manifest.json'          # build report (copy of complete.json)
MEMBER_RECEIPT = 'data/e203_rank_seeds3to9_20261008/models/complete.json'  # Home ten-seed member receipt
RECEIPTS = {
    RECEIPT: '87cd69f98884a449dfbd80f93842815d32b64dc070f02f9162f945a9e2ca7dae',
    MEMBER_RECEIPT: 'e5910d44bbbcd100f9b7b9d37a72df2f856645c0047f9e2f512b90e4a6e40d97',
}
# Pinned from the build report `comparison['v12077']` block (difference against the parent v12077).
EXPECTED = dict(rows=344841, changed=42, rms_vs_parent=7.874374494231528, max_abs_diff=2058.0)
FORECAST = dict(reference=REFERENCE_RMSE, central=231.17, corridor=[230.83, 231.48],
                note='journal 433, written before the upload: expert corridor relative to v12077, not a confidence interval')


def now():
    return datetime.now(timezone.utc)


def require_window():
    if now().date().isoformat() != DAY or (CUTOFF-now()).total_seconds() < 120:
        raise G.GuardError('Oct 8 UTC only, with at least 120 seconds before cutoff')


def registration(proof, commit, source_sha):
    if not re.fullmatch('[a-f0-9]{40}', commit or '') or not re.fullmatch('[a-f0-9]{64}', source_sha or ''):
        raise G.GuardError('full transport registration and source SHA256 required')
    if proof.get('protocol') != PROTOCOL or proof.get('registration') != commit:
        raise G.GuardError('transport registration proof differs')
    current = C.verify_commit(proof['transport'], commit,
                              {SELF: source_sha, BUILDER: SOURCES[BUILDER], 'HYPOTHESES_LOG.md': None})
    journal = current['HYPOTHESES_LOG.md']
    if (PROTOCOL.encode() not in journal or SCIENTIFIC_PROTOCOL.encode() not in journal
            or f'v{VERSION}'.encode() not in journal):
        raise G.GuardError('code, protocol and version are not registered together in the journal')
    if G.sha256(Path(__file__).resolve()) != source_sha:
        raise G.GuardError('active transport differs from registration')
    if G.sha256(Path(C.__file__).resolve()) != SOURCES['scripts/submission_transport_20261008.py']:
        raise G.GuardError('imported canary helper differs')


def load_json(root, name):
    return json.loads((root/name).read_text())


def validate(root):
    """Local, network-free checks of the unchanged candidate against pinned receipts."""
    G.verify_hashes(root, {**SOURCES, **INPUTS, **RECEIPTS, CANDIDATE: CANDIDATE_SHA})
    for path, module in (('src/submission_guard.py', G), ('src/e185_submission.py', E),
                         ('scripts/submission_transport_20261008.py', C)):
        if G.sha256(Path(module.__file__).resolve()) != SOURCES[path]:
            raise G.GuardError('imported runtime differs: '+path)
    path = root/CANDIDATE
    if not 0 < path.stat().st_size <= 5*1024**2:
        raise G.GuardError('candidate must fit a single S3 PUT')
    receipt, member = load_json(root, RECEIPT), load_json(root, MEMBER_RECEIPT)
    fixed = dict(sha256=CANDIDATE_SHA, protocol=SCIENTIFIC_PROTOCOL, version=VERSION, candidate=CANDIDATE,
                 fits_performed=0, rows=EXPECTED['rows'], protected_rows=42, uploaded=False,
                 cached_tail_models=11, cached_e188_models=5, variant='freshe156',
                 e81_blend='present-44:0.1214,60:0.0270', tail_rows='fresh-E156-served')
    if any(receipt.get(k) != v for k, v in fixed.items()):
        raise G.GuardError('build report differs from the registered build')
    if (receipt.get('parent_sha256') != INPUTS[REFERENCE] or receipt.get('parent') != REFERENCE
            or not all(receipt['premix_parity']['v12077'].values())
            or receipt.get('weights', {}).get('nn') != '0.35'
            or receipt.get('weights', {}).get('anchors') != {'raw': '0.028888888888888857', 'lobt': '0.08666666666666667',
                                                            'sched': '0.08666666666666667', 'eobt': '0.08666666666666667', 'aobt': '0.08666666666666667'}):
        raise G.GuardError('build report does not describe the registered single change on the v12077 parent')
    if (receipt['member']['receipt_sha256'] != RECEIPTS[MEMBER_RECEIPT]
            or receipt['member']['member_sha256'] != member['files'].get('prediction_10seed.npz')
            or member.get('all_seeds') != list(range(10)) or member.get('uploaded') is not False
            or not (member.get('bins_equal_archived') and member.get('archived_member_exact')
                    and member.get('control_model0_replay_exact'))
            or any(f.get('checkpoint_replay_exact') is not True for f in member['fits'])):
        raise G.GuardError('ten-seed member receipt differs from the one the build consumed')
    if not receipt['premix_parity']['v940100']['mvt_id'] or not receipt['premix_parity']['v940100']['proba']:
        raise G.GuardError('premix parity failed in the build report')
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
    r = reference.sort(G.ID)
    pred = c[G.TARGET].to_numpy().astype(np.int64)
    ref = r[G.TARGET].to_numpy().astype(np.int64)
    ids = c[G.ID].to_numpy().astype(np.int64)
    base = pl.read_parquet(root/'submissions/gentle-igloo_v291.parquet').sort(G.ID)[G.TARGET].to_numpy().astype(np.int64)
    donor = pl.read_parquet(root/'submissions/gentle-igloo_v361.parquet').sort(G.ID)[G.TARGET].to_numpy().astype(np.int64)
    protected = base != donor
    served = load_json(root, SERVED)
    sids = np.asarray(served['ids'], dtype=np.float64)
    if (served.get('rows') != 42 or len(sids) != 42 or np.any(sids != np.floor(sids))
            or set(sids.astype(np.int64).tolist()) != set(ids[protected].tolist())):
        raise G.GuardError('served E156 IDs differ from the 42 frozen donor IDs')
    pos = np.searchsorted(ids, sids.astype(np.int64))
    if (protected.sum() != 42 or not np.array_equal(pred[pos], np.asarray(served['values'], dtype=np.int64))
            or not np.array_equal(ref[protected], donor[protected])):
        raise G.GuardError('the 42 fresh E156 values were not preserved exactly (or the parent lost its frozen donors)')
    delta = (pred - ref).astype(float)
    changed = int(np.count_nonzero(delta))
    rms = float(np.sqrt(np.mean(delta**2)))
    max_abs = float(np.abs(delta).max())
    if (changed != EXPECTED['changed'] or abs(rms-EXPECTED['rms_vs_parent']) > 1e-6
            or max_abs != EXPECTED['max_abs_diff']):
        raise G.GuardError('difference against v12077 differs from the registered build')
    if not 0.5 <= rms <= 30:
        raise G.GuardError('difference against v12077 outside the pre-registered stop rule [0.5; 30] s')
    gram = [float(np.sqrt(max(REFERENCE_RMSE**2 - 2*REFERENCE_RMSE*rms + rms**2, 0.0))),
            float(np.sqrt(REFERENCE_RMSE**2 + 2*REFERENCE_RMSE*rms + rms**2))]
    return dict(rows=EXPECTED['rows'], protected_rows=42, changed=changed, rms_vs_v12077=rms,
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
                         transport=C.export_commit(root, a.registration_commit, [SELF, BUILDER, 'HYPOTHESES_LOG.md']))
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
