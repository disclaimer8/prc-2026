"""Seal and send the fixed E203 candidate once through the canonical team ledger."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

PROTOCOL = 'E203-v12075-fixed-candidate-single-PUT-20261007-v1'
SELF = 'scripts/e203_single_put_20261007.py'
SPEC = 'analysis/e203_upload_20261007/spec.json'
DAY = '2026-10-07'
VERSION = 12075
FINAL = 'submissions/gentle-igloo_v12075.parquet'
RECEIPT = 'submissions/gentle-igloo_v12075.manifest.json'
CANDIDATE = 'data/e203_candidate_20261007/candidate.parquet'
MODEL_REG = '6e95e3a65936a3401e41499bb340f2d54a1a2121'
HELPERS = ('src/submission_guard.py', 'src/e185_submission.py', 'src/e185_single_put_upload.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc)


def load(root, args):
    if not re.fullmatch('[a-f0-9]{40}', args.registration):
        raise ValueError('full registration required')
    if sha(Path(__file__)) != args.source_sha256 or sha(root / SPEC) != args.spec_sha256:
        raise ValueError('transport source/spec drift')
    spec = read(root / SPEC)
    if spec['protocol'] != PROTOCOL or spec['version'] != VERSION or spec['day'] != DAY:
        raise ValueError('wrong fixed transport specification')
    for name in HELPERS:
        if sha(root / name) != spec['sources'][name]:
            raise ValueError('transport helper drift: ' + name)
    os.environ.setdefault('POLARS_MAX_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    sys.path.insert(0, str(root / 'src'))
    import submission_guard as G
    import e185_submission as E
    G.DAY = DAY  # Only this new process; the shared helper file stays unchanged.
    return spec, G, E


def inspect_candidate(root, spec, G, E):
    import polars as pl
    G.verify_hashes(root, spec['inputs'])
    gate = read(root / 'data/e203_rank_inputs_20261007/gate_authorization.json')
    extension = read(root / 'data/e202_independent_review_20261007/evaluation.json')
    manifest = read(root / 'data/e203_candidate_20261007/manifest.json')
    control = read(root / 'data/e203_chain_20261007/control_verified.json')
    result = gate['result']
    if (gate['registration_commit'] != MODEL_REG or not gate['current_chain_gate_pass']
        or not gate['extension_pass'] or not result['control_exact']
        or not result['gate_pass'] or len(result['folds']) != 6
        or not extension['context_capacity_six_pair_pass']
        or not extension['local_order_six_pair_pass']):
        raise ValueError('registered quality gates are incomplete')
    if sum(f['metrics']['decided']['delta_sse'] < 0 for f in result['folds']) < 4:
        raise ValueError('fewer than four improved folds')
    if (result['metrics']['body']['delta_sse'] >= 0
        or sum(f['metrics']['decided']['trim3_sse'] for f in result['folds']) >= 0
        or sum(f['metrics']['decided']['trim20_sse'] for f in result['folds']) >= 0):
        raise ValueError('body or registered trimmed gate failed')
    if (manifest['registration_commit'] != MODEL_REG
        or manifest['sha256'] != spec['inputs'][CANDIDATE]
        or manifest['control_verified_sha256'] != sha(root / 'data/e203_chain_20261007/control_verified.json')
        or manifest['current_chain_gate_sha256'] != gate['current_chain_result_sha256']
        or manifest['rank_receipt']['rows'] != 344841
        or [f['seed'] for f in manifest['rank_receipt']['fits']] != [0, 1, 2]
        or control['control_rows_exact'] is not True
        or control['reference_sha256'] != spec['inputs'][spec['reference']]):
        raise ValueError('complete candidate/control lineage differs')
    table = pl.read_parquet(root / CANDIDATE)
    checks = G.validate_submission(table, pl.read_parquet(root / E.TEMPLATE),
        pl.read_parquet(root / spec['reference']),
        pl.read_parquet(root / 'submissions/gentle-igloo_v291.parquet'),
        pl.read_parquet(root / 'submissions/gentle-igloo_v361.parquet'))
    if table[G.TARGET].min() < 0 or checks != manifest['validation']:
        raise ValueError('candidate validation differs')
    if not 0 < (root / CANDIDATE).stat().st_size <= 5 * 1024**2:
        raise ValueError('candidate does not fit one S3 PUT')
    return checks


def preflight(root, args, spec, G, E):
    checks = inspect_candidate(root, spec, G, E)
    receipt = read(root / RECEIPT)
    expected = dict(protocol=PROTOCOL, version=VERSION, day=DAY,
        registration=args.registration, source_sha256=args.source_sha256,
        spec_sha256=args.spec_sha256, candidate_sha256=spec['inputs'][CANDIDATE],
        forecast=spec['forecast'], validation=checks, accepted_for_upload=True)
    if receipt != expected or sha(root / FINAL) != expected['candidate_sha256']:
        raise ValueError('sealed submission or receipt drift')
    return checks


def run(root, args):
    spec, G, E = load(root, args)
    if args.action == 'seal':
        for name in (SELF, SPEC):
            registered = subprocess.check_output(['git', 'show', args.registration + ':' + name], cwd=root)
            if hashlib.sha256(registered).hexdigest() != sha(root / name):
                raise ValueError('local transport was not registered')
        if PROTOCOL.encode() not in subprocess.check_output(['git', 'show', args.registration + ':HYPOTHESES_LOG.md'], cwd=root):
            raise ValueError('journal registration missing')
        checks = inspect_candidate(root, spec, G, E)
        if (root / FINAL).exists() or (root / RECEIPT).exists():
            raise ValueError('new immutable submission path required')
        os.link(root / CANDIDATE, root / FINAL)
        E.write_exclusive_json(root / RECEIPT, dict(protocol=PROTOCOL, version=VERSION,
            day=DAY, registration=args.registration, source_sha256=args.source_sha256,
            spec_sha256=args.spec_sha256, candidate_sha256=spec['inputs'][CANDIDATE],
            forecast=spec['forecast'], validation=checks, accepted_for_upload=True))
    checks = preflight(root, args, spec, G, E)
    if args.action != 'upload':
        print(json.dumps(dict(status='PREFLIGHT_PASS', checks=checks, receipt_sha256=sha(root / RECEIPT))))
        return
    if root != Path('/root/prc-2026'):
        raise ValueError('canonical Hetzner ledger is required')
    G.require_window(now())
    from minio import Minio
    from e185_single_put_upload import OnePutHTTP
    E.load_upload_credentials(root)
    http = OnePutHTTP()
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
        secret_key=os.environ['PRC_SECRET_KEY'], secure=True, http_client=http)
    pinned = {FINAL:sha(root / FINAL), RECEIPT:sha(root / RECEIPT), SELF:args.source_sha256, SPEC:args.spec_sha256}
    def put():
        G.require_window(now())
        if (datetime(2026, 10, 8, tzinfo=timezone.utc) - now()).total_seconds() < 120:
            raise ValueError('too close to UTC cutoff')
        G.verify_hashes(root, {**spec['sources'], **spec['inputs'], **pinned})
        def timeout(_signum, _frame):
            raise TimeoutError('single PUT timed out; no retry')
        previous = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo', Path(FINAL).name,
                str(root / FINAL), num_parallel_uploads=1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    try:
        used = G.single_attempt(root / 'data/submission_upload_ledger.jsonl', Path(FINAL).name,
            pinned[FINAL], now=now, inventory=lambda:(G.fetch_leaderboard(), E.bucket_inventory(root)), upload=put)
    finally:
        http.clear()
    print(json.dumps(dict(status='UPLOADED', version=VERSION, slots_used=used,
        sha256=pinned[FINAL], at=now().isoformat())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal', 'preflight', 'upload'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--registration', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args)
    except Exception as exc:
        print('STOP: ' + type(exc).__name__ + '; no automatic retry', file=sys.stderr)
        sys.exit(2)
