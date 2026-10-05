"""Registered transport amendment: immutable E185 build, at most one HTTP PUT.

The original build manifest and source inventory remain unchanged. This extra
entry point is pinned separately by --source-sha256 before any network action.
It reuses the same preflight, quota check, lock and durable intent ledger.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import sys
from datetime import datetime, timezone

import certifi
from minio import Minio
import urllib3

import e185_submission as E


class OnePutHTTP(urllib3.PoolManager):
    def __init__(self):
        super().__init__(retries=False, cert_reqs='CERT_REQUIRED', ca_certs=certifi.where(),
                         timeout=urllib3.Timeout(connect=10, read=60, total=90))
        self.put_started = False

    def urlopen(self, method, url, *args, **kwargs):
        if method.upper() == 'PUT':
            if self.put_started:
                raise RuntimeError('a PUT was already attempted; network retry forbidden')
            self.put_started = True
        kwargs['retries'] = False
        kwargs['redirect'] = False
        return super().urlopen(method, url, *args, **kwargs)


def upload(root, source_sha256, registration_commit):
    if (not re.fullmatch(r'[0-9a-f]{40}', registration_commit)
            or E.sha256(Path(__file__).resolve()) != source_sha256):
        raise E.GuardError('transport source does not match its registration')
    E.require_window(E.utcnow())
    manifest = E.preflight(root)
    path = root / E.FINAL
    if not 0 < path.stat().st_size <= 5 * 1024 * 1024:
        raise E.GuardError('candidate must fit a single S3 PUT')
    E.load_upload_credentials(root)
    http = OnePutHTTP()
    client = Minio('s3.opensky-network.org', access_key=os.environ['PRC_ACCESS_KEY'],
                   secret_key=os.environ['PRC_SECRET_KEY'], secure=True, http_client=http)

    def inventory():
        return E.fetch_leaderboard(), E.bucket_inventory(root)

    def timeout(_signum, _frame):
        raise TimeoutError('bounded single PUT expired')

    def put():
        E.require_window(E.utcnow())
        if (datetime(2026, 10, 6, tzinfo=timezone.utc) - E.utcnow()).total_seconds() < 120:
            raise E.GuardError('too close to UTC day end')
        E.verify_hashes(root, {E.FINAL: manifest['outputs'][E.FINAL]})
        previous = signal.signal(signal.SIGALRM, timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo', path.name, str(path), num_parallel_uploads=1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    print(f'Transport registration {registration_commit}; source {source_sha256}', flush=True)
    try:
        used = E.single_attempt(root/'data/submission_upload_ledger.jsonl', path.name,
                                manifest['outputs'][E.FINAL], now=E.utcnow,
                                inventory=inventory, upload=put)
    finally:
        http.clear()
    print(f'UPLOADED v{E.VERSION}; accounted {used}/5 UTC-day slots', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--registration-commit', required=True)
    args = parser.parse_args()
    try:
        upload(args.root.resolve(), args.source_sha256, args.registration_commit)
    except Exception as exc:
        print(f'STOP: {exc}' if isinstance(exc, E.GuardError) else f'STOP: {type(exc).__name__}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
