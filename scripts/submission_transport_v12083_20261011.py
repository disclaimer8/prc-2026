"""Oct 11 transport of v12083: exactly one PUT of a single-change measurement on top of v12082 (journal №460).

v12083 = v12082 with the E218 PRU level shift removed on LIRF rows only: every LIRF row (ADEP_mvt = LIRF in the board
ranking) takes its v12080 value (the same row of the 09.10 final 76c32426, i.e. before the E218 stage), every other row
keeps v12082 (the Jan/Jul rows of the E218 final bbae2cc5). On fixed labels v12083 - v12082 is an exact board measurement
of the LIRF cells of E218 on Jan/Jul. Built by scripts/v12083_lirf_e218_exclusion_build_20261011.py build. inspect is local
and network-free. seal exports verifiable Git object proofs of the registration commit (this script + builder + journal).
preflight/upload verify those proofs without a Git checkout on the canonical upload host. No fitting, no weight choice,
no retry. Modelled on scripts/submission_transport_v12081_20261011.py (№455).
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

PROTOCOL = 'v12083-single-PUT-20261011-v1'
SELF = 'scripts/submission_transport_v12083_20261011.py'
DAY = '2026-10-11'
CUTOFF = datetime(2026, 10, 12, tzinfo=timezone.utc)
VERSION = 12083
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
CANDIDATE_SHA = 'a834327581cec4b3cff90666ebb808b27c8511449b9be902086443f2a123df1c'
SCIENTIFIC_PROTOCOL = 'v12083-v12082-minus-E218-on-LIRF-rows-zero-fit-20261011-v1'
BUILDER = 'scripts/v12083_lirf_e218_exclusion_build_20261011.py'
SOURCES = {
    BUILDER: '6fd11a6b6e5c396c6281e80fd725ff1d9b485311b789f1e7af0304721653302e',
    'src/submission_guard.py': 'e2815dbbc2e16edbbe6140de3fbbb6cd4fcd937c7277bcf508ce4837595d0c57',
    'src/e185_submission.py': 'b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c',
    'src/e185_single_put_upload.py': 'd7cc42d6ff5a5b16a3eb8be5858308596feb3268d00b022cfae4d1339c8354a8',
    'scripts/submission_transport_20261008.py': '79808a22e127d21709b50ab50576e023057afef61bae766e775e8fc1b26790f0',
}
REFERENCE = 'submissions/gentle-igloo_v12082.parquet'   # the parent: the difference is pinned against it
REFERENCE_RMSE = 231.1445                           # v12082 board score (journal №459)
LIRF_SOURCE = 'submissions/gentle-igloo_v12080.parquet'   # the same rows before the E218 stage (Jan/Jul rows of 76c32426)
RANKING = 'data/prc-2026-datasets/ranking.parquet'
EXCLUDE = 'LIRF'
INPUTS = {
    'data/prc-2026-datasets/submitting.parquet': '0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b',
    REFERENCE: 'e87b79789ab342c6ad4ca6c64900f159ac7affd242eb284717ff321c122f5298',
    LIRF_SOURCE: 'ae1680cbab9b56a7748e7eac97afdf6359bbd38ad57545a763cdfa7868064b37',
    RANKING: '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
}
RECEIPT = f'submissions/gentle-igloo_v{VERSION}.manifest.json'          # build report
RECEIPTS = {RECEIPT: '2bda5dd3a5f59a7bc1a70404457439b5a20869599b2f021aaf1b00a7b81ec23f'}
# Pinned from the build report `change` block (difference against the parent v12082).
EXPECTED = dict(rows=344841, changed=25017, rms_vs_parent=2.7485427830832387, max_abs_diff=11.0, lirf_rows=26899)
# Per airport-month integer shift of v12083 - v12082 and the number of rows it moved (all other cells: 0 rows moved).
CELLS = {'LIRF-1': (-9, 10549), 'LIRF-7': (-11, 14468)}
FORECAST = dict(reference=REFERENCE_RMSE, point=231.1533, corridor=[231.130, 231.174],
                readings={'lirf_neutral_nonLIRF_beta_hat': 231.1407, 'uniform_beta_0p516_v12082': 231.1659,
                          'uniform_beta_0p562_v12081': 231.1683},
                decision={'rebuild_final_with_e218_excluding_lirf_if_le': 231.1475, 'final_stays_bbae2cc5_if_gt': 231.1475},
                note='journal 460, written before the upload: analysis/v12083_upload_20261011/forecast.json')


def now():
    return datetime.now(timezone.utc)


def require_window():
    if now().date().isoformat() != DAY or (CUTOFF-now()).total_seconds() < 120:
        raise G.GuardError('Oct 11 UTC only, with at least 120 seconds before cutoff')


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
    receipt = load_json(root, RECEIPT)
    fixed = dict(sha256=CANDIDATE_SHA, protocol=SCIENTIFIC_PROTOCOL, version=VERSION, candidate=CANDIDATE, builder=BUILDER,
                 builder_sha256=SOURCES[BUILDER], fits_performed=0, labels_read=False, rows=EXPECTED['rows'], uploaded=False,
                 excluded_airport=EXCLUDE, parent=REFERENCE, parent_sha256=INPUTS[REFERENCE],
                 lirf_source=LIRF_SOURCE, lirf_source_sha256=INPUTS[LIRF_SOURCE])
    if any(receipt.get(k) != v for k, v in fixed.items()):
        raise G.GuardError('build report differs from the registered build')
    change = receipt.get('change', {})
    if (change.get('changed') != EXPECTED['changed'] or change.get('lirf_rows') != EXPECTED['lirf_rows']
            or change.get('cells') != {k: list(v) for k, v in CELLS.items()}
            or change.get('lirf_rows_equal_v12080') is not True or change.get('other_rows_equal_v12082') is not True
            or receipt.get('final_hybrid', {}).get('janjul_equals_candidate') is not True):
        raise G.GuardError('build report does not describe the registered single change')
    template = pl.read_parquet(root/E.TEMPLATE)
    candidate = pl.read_parquet(path)
    reference = pl.read_parquet(root/REFERENCE)
    source = pl.read_parquet(root/LIRF_SOURCE)
    for name, frame in (('candidate', candidate), ('reference', reference), ('lirf source', source)):
        if frame.schema != template.schema or frame.columns != template.columns:
            raise G.GuardError(f'{name}: schema differs from template')
        G.exact_ids(frame[G.ID].to_numpy(), template[G.ID].to_numpy(), EXPECTED['rows'])
        values = frame[G.TARGET].to_numpy()
        if frame[G.TARGET].null_count() or values.dtype.kind not in 'iu' or not np.isfinite(values).all() or values.min() < 0:
            raise G.GuardError(f'{name}: predictions must be finite non-negative integers')
    if not candidate[G.ID].equals(reference[G.ID]) or not source[G.ID].equals(reference[G.ID]):
        raise G.GuardError('row order differs from v12082')
    pred = candidate[G.TARGET].to_numpy().astype(np.int64)
    ref = reference[G.TARGET].to_numpy().astype(np.int64)
    src = source[G.TARGET].to_numpy().astype(np.int64)
    delta = (pred - ref).astype(float)
    changed = int(np.count_nonzero(delta))
    rms = float(np.sqrt(np.mean(delta**2)))
    max_abs = float(np.abs(delta).max())
    if (changed != EXPECTED['changed'] or abs(rms-EXPECTED['rms_vs_parent']) > 1e-9
            or max_abs != EXPECTED['max_abs_diff']):
        raise G.GuardError('difference against v12082 differs from the registered build')
    if not 0.5 <= rms <= 30:
        raise G.GuardError('difference against v12082 outside the pre-registered stop rule [0.5; 30] s')
    # independent of the builder: LIRF rows are the v12080 rows, every other row is the v12082 row; moved cells as pinned
    rank = (pl.scan_parquet(root/RANKING).filter(pl.col('PHASE_mvt') == 'DEP')
            .select(pl.col(G.ID).cast(candidate.schema[G.ID]), 'ADEP_mvt', pl.col('MVT_TIME_UTC_mvt').dt.month().alias('month')).collect())
    x = candidate.select(G.ID).with_columns(d=pl.Series(delta)).join(rank, on=G.ID, how='left', maintain_order='left')
    if x['ADEP_mvt'].null_count() or x.height != EXPECTED['rows']:
        raise G.GuardError('ranking does not cover the candidate')
    lirf = (x['ADEP_mvt'] == EXCLUDE).to_numpy()
    if int(lirf.sum()) != EXPECTED['lirf_rows'] or not np.array_equal(pred[lirf], src[lirf]) or not np.array_equal(pred[~lirf], ref[~lirf]):
        raise G.GuardError('LIRF rows must equal v12080 and every other row v12082')
    got = {}
    for row in (x.filter(pl.col('d') != 0).group_by('ADEP_mvt', 'month')
                .agg(dmin=pl.col('d').min(), dmax=pl.col('d').max(), n=pl.len()).iter_rows(named=True)):
        if row['dmin'] != row['dmax']:
            raise G.GuardError('rows of one airport-month moved by different amounts')
        got[f"{row['ADEP_mvt']}-{row['month']}"] = (int(row['dmin']), int(row['n']))
    if got != CELLS:
        raise G.GuardError('moved cells/shifts differ from minus the LIRF cells of the E218 stage')
    gram = [float(np.sqrt(max(REFERENCE_RMSE**2 - 2*REFERENCE_RMSE*rms + rms**2, 0.0))),
            float(np.sqrt(REFERENCE_RMSE**2 + 2*REFERENCE_RMSE*rms + rms**2))]
    return dict(rows=EXPECTED['rows'], changed=changed, rms_vs_v12082=rms, lirf_rows=int(lirf.sum()),
                max_abs_diff=max_abs, size_bytes=path.stat().st_size, sha256=CANDIDATE_SHA, cells_moved=len(got),
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
