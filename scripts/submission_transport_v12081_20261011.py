"""Oct 11 transport of v12081: exactly one PUT of a single-change measurement on top of v12077.

v12081 = v12077 + the E218 PRU level stage (journal №451/452, patch 0613fdd9) after `half`, exactly as the patch adds
it to the final assembly: per airport x month without de-icing, every row of the PRU population moves by
c = -0.5 (gap - emu_hat); the 42 frozen TabPFN rows are never moved (journal №455). The v12077 chain was rebuilt
byte-identically (f06c66bf) before the stage. Built by scripts/v12081_e218_build_20261011.py build. inspect is local
and network-free. seal exports verifiable Git object proofs of the registration commit (this script + builder +
journal). preflight/upload verify those proofs without a Git checkout on the canonical upload host. No fitting,
no weight choice, no retry. Modelled on scripts/submission_transport_v12078_20261009.py (№431).
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

PROTOCOL = 'v12081-single-PUT-20261011-v1'
SELF = 'scripts/submission_transport_v12081_20261011.py'
DAY = '2026-10-11'
CUTOFF = datetime(2026, 10, 12, tzinfo=timezone.utc)
VERSION = 12081
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
CANDIDATE_SHA = '719622263e3e66fb39ad67d3b4255dc7317f9a6449385b136de5d8d80dd42f4c'
SCIENTIFIC_PROTOCOL = 'v12081-v12077-chain-plus-E218-PRU-level-stage-zero-fit-20261011-v1'
BUILDER = 'scripts/v12081_e218_build_20261011.py'
SOURCES = {
    BUILDER: 'fa5b2bcbba7da2b79e619ec220f59f498fb0c6e62e3f7c559012b630283f7522',
    'scripts/v12076_nn10seed_build_20261008.py': '3fc1e93fa1f4cbb9b6ec7cf2043dd9e9905e129ca24f9b039e3668ae947496f9',
    'src/submission_guard.py': 'e2815dbbc2e16edbbe6140de3fbbb6cd4fcd937c7277bcf508ce4837595d0c57',
    'src/e185_submission.py': 'b1031e156dcd5ab65f808def63664b71eadfa99574d92112cd18b80c9dfef46c',
    'src/e185_single_put_upload.py': 'd7cc42d6ff5a5b16a3eb8be5858308596feb3268d00b022cfae4d1339c8354a8',
    'scripts/submission_transport_20261008.py': '79808a22e127d21709b50ab50576e023057afef61bae766e775e8fc1b26790f0',
}
REFERENCE = 'submissions/gentle-igloo_v12077.parquet'   # the parent: the difference is pinned against it
REFERENCE_RMSE = 230.4812                           # v12077 board score (journal №427)
PATCH = 'analysis/e218_pru_level_recheck_20261010/final_tail_e218_pru_level_stage.patch'
RANKING = 'data/prc-2026-datasets/ranking.parquet'
INPUTS = {
    'data/prc-2026-datasets/submitting.parquet': '0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b',
    REFERENCE: 'f06c66bfa3388b17f5b6ba2362fa45a7da1be8fd5ae2d37de47cfe0e6bf2ba44',
    'submissions/gentle-igloo_v12075.parquet': '90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2',
    PATCH: '0613fdd9bab1aad457c12ed466c5ef05ecadc8e21ddda0e599287118544b4615',
    'data/taxi_out_additional_time_2025.csv': '4aedf19520d3f84bebc418b56835762c907b714a9c19e751bb8658f9e62ce598',
    'data/taxi_out_additional_time_2026.csv': 'b72040e0987e6c7d5706279d645ef9f814a6506165c83f89d8509022154a1076',
    RANKING: '9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
}
RECEIPT = f'submissions/gentle-igloo_v{VERSION}.manifest.json'          # build report (copy of complete.json)
MEMBER_RECEIPT = 'data/e203_rank_seeds3to9_20261008/models/complete.json'  # Home ten-seed member receipt
RECEIPTS = {
    RECEIPT: '4c4988857d24b29fbf94aca482c8b334d3a82cb8f9bc35fad8b978fc0e783749',
    MEMBER_RECEIPT: 'e5910d44bbbcd100f9b7b9d37a72df2f856645c0047f9e2f512b90e4a6e40d97',
}
# Pinned from the build report `comparison['v12077']` block (difference against the parent v12077).
EXPECTED = dict(rows=344841, changed=215682, rms_vs_parent=3.5317197083835183, max_abs_diff=8.0)
# Per airport-month integer shift rint(c) of the E218 stage and the number of rows it moved (all other cells: 0 rows moved).
CELLS = {'EDDF-7': (1, 17534), 'EDDM-7': (-3, 13062), 'EGLL-1': (-1, 19005), 'EGLL-7': (-2, 20198), 'EHAM-1': (2, 14084),
         'EHAM-7': (-3, 20350), 'LEBL-7': (-4, 16766), 'LEMD-7': (-7, 16959), 'LFPG-7': (-5, 17492), 'LIRF-1': (8, 10549),
         'LIRF-7': (8, 14468), 'LSZH-7': (-3, 11948), 'LTFM-7': (-5, 23267)}
FORECAST = dict(reference=REFERENCE_RMSE, central=230.4079, corridor=[230.3935, 230.4224],
                scenarios={'lirf_gap_is_artifact': 230.4479, 'beta_0_gap_is_noise': 230.5080},
                decision={'rebuild_final_with_e218_if_le': 230.4512, 'e218_out_if_gt': 230.4812},
                note='journal 455, written before the upload: E218 board model recomputed on v12077, corridor = beta +- 2 se')


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
    receipt, member = load_json(root, RECEIPT), load_json(root, MEMBER_RECEIPT)
    fixed = dict(sha256=CANDIDATE_SHA, protocol=SCIENTIFIC_PROTOCOL, version=VERSION, candidate=CANDIDATE,
                 fits_performed=0, rows=EXPECTED['rows'], protected_rows=42, uploaded=False,
                 cached_tail_models=11, cached_e188_models=5,
                 e81_blend='present-44:0.1214,60:0.0270', tail_rows='frozen-v361-donors')
    if any(receipt.get(k) != v for k, v in fixed.items()):
        raise G.GuardError('build report differs from the registered build')
    if (receipt.get('parent_sha256') != INPUTS[REFERENCE] or receipt.get('parent') != REFERENCE
            or receipt.get('rebuild_v12077') != {'sha256': INPUTS[REFERENCE], 'byte_identical': True}
            or not all(receipt['premix_parity']['v12077'].values())
            or receipt.get('weights', {}).get('nn') != '0.35'
            or receipt.get('weights', {}).get('anchors') != {'raw': '0.028888888888888857', 'lobt': '0.08666666666666667',
                                                            'sched': '0.08666666666666667', 'eobt': '0.08666666666666667', 'aobt': '0.08666666666666667'}):
        raise G.GuardError('build report does not describe the registered single change on a byte-identical v12077 rebuild')
    stage = receipt.get('e218_stage', {})
    if (stage.get('k') != 0.5 or stage.get('xs_max') != 0.005 or stage.get('patch') != PATCH or stage.get('patch_sha256') != INPUTS[PATCH]
            or stage.get('ranking') != RANKING or stage.get('ranking_sha256') != INPUTS[RANKING]
            or stage.get('pru_sha256') != {'2025': INPUTS['data/taxi_out_additional_time_2025.csv'], '2026': INPUTS['data/taxi_out_additional_time_2026.csv']}
            or receipt.get('change', {}).get('protected_unchanged') is not True):
        raise G.GuardError('E218 stage in the build report differs from the registered rule')
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
    if not candidate[G.ID].equals(reference[G.ID]):
        raise G.GuardError('row order differs from v12077')
    c = candidate.sort(G.ID)
    r = reference.sort(G.ID)
    pred = c[G.TARGET].to_numpy().astype(np.int64)
    ref = r[G.TARGET].to_numpy().astype(np.int64)
    base = pl.read_parquet(root/'submissions/gentle-igloo_v291.parquet').sort(G.ID)[G.TARGET].to_numpy().astype(np.int64)
    donor = pl.read_parquet(root/'submissions/gentle-igloo_v361.parquet').sort(G.ID)[G.TARGET].to_numpy().astype(np.int64)
    protected = base != donor
    if protected.sum() != 42 or not np.array_equal(pred[protected], donor[protected]) or not np.array_equal(ref[protected], donor[protected]):
        raise G.GuardError('the 42 frozen TabPFN donor rows were not preserved exactly')
    delta = (pred - ref).astype(float)
    changed = int(np.count_nonzero(delta))
    rms = float(np.sqrt(np.mean(delta**2)))
    max_abs = float(np.abs(delta).max())
    if (changed != EXPECTED['changed'] or abs(rms-EXPECTED['rms_vs_parent']) > 1e-6
            or max_abs != EXPECTED['max_abs_diff']):
        raise G.GuardError('difference against v12077 differs from the registered build')
    if not 0.5 <= rms <= 30:
        raise G.GuardError('difference against v12077 outside the pre-registered stop rule [0.5; 30] s')
    # independent of the builder: every moved row belongs to a pinned airport-month cell and moved by that cell's shift
    rank = (pl.scan_parquet(root/RANKING).filter(pl.col('PHASE_mvt') == 'DEP')
            .select(pl.col(G.ID).cast(c.schema[G.ID]), 'ADEP_mvt', pl.col('MVT_TIME_UTC_mvt').dt.month().alias('month')).collect())
    x = c.select(G.ID).with_columns(d=pl.Series(delta)).join(rank, on=G.ID, how='left', maintain_order='left')
    if x['ADEP_mvt'].null_count():
        raise G.GuardError('ranking does not cover the candidate')
    got = {}
    for row in (x.filter(pl.col('d') != 0).group_by('ADEP_mvt', 'month')
                .agg(dmin=pl.col('d').min(), dmax=pl.col('d').max(), n=pl.len()).iter_rows(named=True)):
        if row['dmin'] != row['dmax']:
            raise G.GuardError('rows of one airport-month moved by different amounts')
        got[f"{row['ADEP_mvt']}-{row['month']}"] = (int(row['dmin']), int(row['n']))
    if got != CELLS:
        raise G.GuardError('moved cells/shifts differ from the registered E218 stage')
    gram = [float(np.sqrt(max(REFERENCE_RMSE**2 - 2*REFERENCE_RMSE*rms + rms**2, 0.0))),
            float(np.sqrt(REFERENCE_RMSE**2 + 2*REFERENCE_RMSE*rms + rms**2))]
    return dict(rows=EXPECTED['rows'], protected_rows=42, changed=changed, rms_vs_v12077=rms,
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
