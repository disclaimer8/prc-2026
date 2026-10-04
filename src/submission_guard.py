"""Fail-closed artifact checks and one-attempt UTC upload accounting.

No credentials are logged. Network writes belong to the supplied upload callback.
The durable intent is recorded before it is called, including on interruptions.
All automated team uploaders must share one ledger/lock; external uploads must
not run concurrently with this controller.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request

import numpy as np
import polars as pl

TEAM = 'gentle-igloo'
DAY = '2026-10-05'
ROWS = 344841
BASE = 'https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard'
ID, TARGET = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'


class GuardError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(root, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise GuardError('empty hash manifest')
    for name, digest in hashes.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not re.fullmatch('[a-f0-9]{64}', digest):
            raise GuardError('invalid manifest path/hash')
        if not (Path(root)/path).is_file() or sha256(Path(root)/path) != digest:
            raise GuardError(f'hash mismatch: {name}')


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
        if not isinstance(parsed, datetime) or parsed.tzinfo is None:
            raise ValueError('timezone required')
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError) as exc:
        raise GuardError('invalid aware UTC timestamp') from exc


def require_window(now):
    if timestamp(now).date().isoformat() != DAY:
        raise GuardError(f'upload allowed only during {DAY} UTC')


def exact_ids(ids, expected, rows):
    a, b = np.asarray(ids), np.asarray(expected)
    if a.ndim != 1 or len(a) != rows or b.ndim != 1 or len(b) != rows:
        raise GuardError('ID row count mismatch')
    for values in (a, b):
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
            raise GuardError('IDs require finite exact integers')
        if values.dtype.kind == 'f' and (np.any(np.abs(values) > 2**53)
                                         or np.any(values != np.floor(values))):
            raise GuardError('Float IDs must be exact integers in the safe range')
        if values.dtype.kind == 'u' and np.any(values > np.iinfo(np.int64).max):
            raise GuardError('IDs exceed signed integer range')
    # The official template uses Float64 IDs. Validate first; never round them.
    a, b = a.astype(np.int64), b.astype(np.int64)
    if len(np.unique(a)) != rows or len(np.unique(b)) != rows or not np.array_equal(np.sort(a), np.sort(b)):
        raise GuardError('IDs are duplicated or differ from template')


def validate_rank(path, log_path, ids, expected_rows=ROWS):
    try:
        with np.load(path, allow_pickle=False) as z:
            exact_ids(z['mvt_id'], ids, expected_rows)
            seeds, reg = z['per_seed'], z['reg']
            if seeds.shape != (5, expected_rows) or reg.shape != (expected_rows,):
                raise GuardError('ranking requires all five seeds')
            if seeds.dtype.kind not in 'iuf' or reg.dtype.kind not in 'iuf' or not np.isfinite(seeds).all() or not np.isfinite(reg).all():
                raise GuardError('ranking predictions must be finite real numbers')
            if not np.allclose(reg, seeds.mean(axis=0), rtol=0, atol=1e-8):
                raise GuardError('ranking mean differs from five seeds')
        lines = Path(log_path).read_text().splitlines()
        completed = [int(m.group(1)) for line in lines if (m := re.match(r'^seed (\d+) done\b', line))]
        if completed != list(range(5)) or not lines or lines[-1].strip() != 'done':
            raise GuardError('ranking log does not prove five seeds and done')
    except (KeyError, OSError, ValueError) as exc:
        if isinstance(exc, GuardError):
            raise
        raise GuardError('ranking artifact/log is unreadable or incomplete') from exc
    return {'rows': expected_rows, 'seeds': 5, 'sha256': sha256(path), 'log_sha256': sha256(log_path)}


def validate_submission(candidate, template, reference, tab_base, tab_donor, expected_rows=ROWS):
    for name, frame in [('candidate', candidate), ('reference', reference), ('tab_base', tab_base), ('tab_donor', tab_donor)]:
        if frame.schema != template.schema or frame.columns != template.columns:
            raise GuardError(f'{name}: schema differs from template')
        exact_ids(frame[ID].to_numpy(), template[ID].to_numpy(), expected_rows)
        values = frame[TARGET].to_numpy()
        if frame[TARGET].null_count() or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
            raise GuardError(f'{name}: predictions must be finite')
    c, r, b, d = [x.sort(ID)[TARGET].to_numpy().astype(float) for x in (candidate, reference, tab_base, tab_donor)]
    protected = b != d
    if protected.sum() != 42 or not np.array_equal(c[protected], d[protected]):
        raise GuardError('the 42 TabPFN donor rows were not preserved exactly')
    delta = c - r
    changed = int(np.count_nonzero(delta))
    rms = float(np.sqrt(np.mean(delta**2)))
    if changed == 0 or rms > 400:
        raise GuardError('unchanged or implausible submission vs reference')
    return {'rows': expected_rows, 'protected_rows': 42, 'changed': changed, 'rms_vs_v1139': rms}


def http_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_leaderboard(fetch_page=http_json):
    items, cursor, seen_cursors, seen_ids = [], None, set(), set()
    for _ in range(10000):
        page = fetch_page(BASE + ('?cursor=' + urllib.parse.quote(cursor, safe='') if cursor else ''))
        if not isinstance(page, dict) or not isinstance(page.get('items'), list) or 'nextCursor' not in page:
            raise GuardError('malformed leaderboard page')
        for item in page['items']:
            if not isinstance(item, dict) or not isinstance(item.get('submissionId'), str) or item['submissionId'] in seen_ids:
                raise GuardError('malformed or overlapping leaderboard pages')
            seen_ids.add(item['submissionId'])
            items.append(item)
        cursor = page['nextCursor']
        if cursor is None or cursor == '':
            return items
        if not isinstance(cursor, str) or cursor in seen_cursors or not page['items']:
            raise GuardError('incomplete or cyclic leaderboard pagination')
        seen_cursors.add(cursor)
    raise GuardError('leaderboard pagination limit reached')


def read_ledger(path):
    if not Path(path).exists():
        return []
    try:
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
        for record in records:
            if not isinstance(record, dict) or record.get('event') not in {'intent', 'uploaded', 'ambiguous'}:
                raise ValueError('invalid event')
            if not re.fullmatch(r'gentle-igloo_v\d+\.parquet', record['filename']):
                raise ValueError('invalid filename')
            timestamp(record['at'])
        return records
    except (ValueError, KeyError, TypeError) as exc:
        raise GuardError('malformed upload ledger; manual reconciliation required') from exc


def check_quota(items, objects, ledger, now, filename):
    require_window(now)
    scored_names, scored_ids, bucket_names, intended_names = set(), set(), set(), set()
    all_names, states = set(), {}
    for item in items:
        if not isinstance(item, dict):
            raise GuardError('malformed leaderboard item')
        if item.get('teamName') != TEAM:
            continue
        try:
            name, ident = item['filename'], item['submissionId']
            when = timestamp(item['processedAt'])
        except KeyError as exc:
            raise GuardError('incomplete team leaderboard item') from exc
        all_names.add(name)
        if when.date().isoformat() == DAY:
            scored_names.add(name)
            scored_ids.add(ident)
    for obj in objects:
        try:
            name, when = obj['filename'], timestamp(obj['last_modified'])
        except (KeyError, TypeError) as exc:
            raise GuardError('incomplete bucket inventory') from exc
        all_names.add(name)
        if when.date().isoformat() == DAY:
            bucket_names.add(name)
    for record in ledger:
        name = record['filename']
        all_names.add(name)
        states[name] = record['event']
        if timestamp(record['at']).date().isoformat() == DAY and record['event'] == 'intent':
            intended_names.add(name)
    if filename in all_names:
        raise GuardError('version already uploaded or attempted; no retry')
    if any(states[name] != 'uploaded' for name in intended_names):
        raise GuardError('ambiguous earlier attempt; manual reconciliation required')
    used = len(scored_ids) + len((bucket_names | intended_names) - scored_names)
    if used >= 5:
        raise GuardError('daily upload limit reached')
    return used


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GuardError('another submission controller holds the lock') from exc
        yield
    finally:
        os.close(fd)


def append_event(path, event):
    with open(path, 'a', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def single_attempt(ledger, filename, digest, *, now, inventory, upload):
    ledger = Path(ledger)
    with exclusive_lock(ledger.with_suffix(ledger.suffix + '.lock')):
        require_window(now())
        records = read_ledger(ledger)
        items, objects = inventory()  # Any network failure occurs before intent/write.
        current = now()
        used = check_quota(items, objects, records, current, filename)
        common = {'filename': filename, 'sha256': digest, 'at': timestamp(current).isoformat(), 'used_before': used}
        append_event(ledger, {**common, 'event': 'intent'})
        try:
            require_window(now())
            upload()
        except BaseException as exc:
            append_event(ledger, {**common, 'event': 'ambiguous', 'error_type': type(exc).__name__})
            raise GuardError('upload outcome ambiguous; intent retained; automatic retry forbidden') from None
        append_event(ledger, {**common, 'event': 'uploaded'})
    return used + 1
