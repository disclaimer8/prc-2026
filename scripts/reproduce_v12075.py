"""Independent, zero-fit saved-component replay of v12075; never uploads.

The reference prediction is opened only after the fresh complete output exists.
Original registered sources, inventories, models and component predictions remain
unchanged. This does not claim retraining from raw data or independent nested CV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import reproduce_best as R

PROTOCOL = 'v12075-independent-saved-component-replay-zero-fit-v1'
SELF = 'scripts/reproduce_v12075.py'
LAUNCHER = 'scripts/reproduce_v12075_launch_20261007.py'
AREA = 'analysis/reproduce_v12075_20261007'
SPEC = AREA + '/inventory.json'
DOC = AREA + '/PROTOCOL.md'
OUTPUT = 'data/reproduced_v12075'
CACHE = 'data/e203_chain_20261007/models'
RANK = 'data/e203_rank_20261007'
MEMBER = RANK + '/prediction.npz'
BEST = 'submissions/gentle-igloo_v12075.parquet'
BEST_SHA = '90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2'
ORACLES = (BEST, 'data/e203_candidate_20261007/candidate.parquet',
           'data/e203_chain_20261007/submissions/gentle-igloo_v940115.parquet')
E203_REG = '6e95e3a65936a3401e41499bb340f2d54a1a2121'
E203_PROTOCOL = 'E203-ordered-trajectory-fixed025-joint-preE81-sixpair-gated-fullrank-fulltail-v1'
REGISTERED_FILES = (SELF, LAUNCHER, SPEC, DOC, 'tests/test_reproduce_v12075.py')
ID, Y = R.ID, R.Y


def verify_inventory(root, inventory, oracles=ORACLES):
    """Check the closed input set; even an alias cannot make an oracle an input."""
    root = root.resolve()
    forbidden = [root / n for n in oracles]
    for name, expected in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('input escapes project root: ' + name)
        path = root / relative
        if any(path == p or (path.exists() and p.exists() and path.samefile(p)) for p in forbidden):
            raise ValueError('reference predictions cannot be replay inputs: ' + name)
        if not path.is_file() or R.sha(path) != expected:
            raise ValueError('missing or changed replay input: ' + name)


def registration(root, commit):
    if not re.fullmatch('[a-f0-9]{40}', commit or ''):
        raise ValueError('full reviewed registration commit required')
    for name in REGISTERED_FILES:
        frozen = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)
        if hashlib.sha256(frozen).hexdigest() != R.sha(root / name):
            raise ValueError('registered replay source/document changed: ' + name)
    journal = subprocess.check_output(['git', 'show', commit + ':HYPOTHESES_LOG.md'], cwd=root)
    if PROTOCOL.encode() not in journal:
        raise ValueError('replay protocol not registered')


def verify_e188_globs(root, original_inputs):
    # E188's registered apply resolves sensor globs in the original root.
    # All other stage globs are confined to individually pinned private links.
    for folder, pattern in (('data/adsblol_kt1', 'offblock2_2026-*.parquet'),
                            ('data/adsblol', 'visit_2026-*.parquet')):
        actual = {str(p.relative_to(root)) for p in (root / folder).glob(pattern)}
        expected = {n for n in original_inputs if Path(n).parent == Path(folder) and Path(n).match(pattern)}
        if not expected or actual != expected:
            raise ValueError('E188 live sensor inventory differs: ' + folder)


def seal(root):
    """Preparation only: preserve the original inventory and add replay dependencies."""
    setup = R.read(root / R.ARCHIVE / 'setup.json')
    old = R.read(root / R.SPEC)
    candidate = R.read(root / 'data/e203_candidate_20261007/manifest.json')
    rank = R.read(root / RANK / 'complete.json')
    if (candidate['protocol'] != E203_PROTOCOL or candidate['registration_commit'] != E203_REG
        or candidate['sha256'] != BEST_SHA or candidate['rank_receipt'] != rank
        or rank['rows'] != 344841 or [f['seed'] for f in rank['fits']] != [0, 1, 2]
        or any(len(f['losses']) != 8 for f in rank['fits'])):
        raise ValueError('E203 saved member/candidate lineage differs')
    if candidate['rank_complete_sha256'] != R.sha(root / RANK / 'complete.json'):
        raise ValueError('rank receipt changed')
    files = dict(setup['inputs'])
    files.update({R.ARCHIVE + '/' + n: h for n, h in setup['source_after'].items()})
    files.update({n: h for n, h in old['pinned_files'].items() if n != R.BEST})
    files.update({RANK + '/' + n: h for n, h in rank['files'].items()})
    for name in (R.SELF, R.SPEC, 'scripts/reproduce_best_queue_20261007.py',
                 RANK + '/complete.json', 'data/e203_candidate_20261007/manifest.json',
                 'data/e203_chain_20261007/control_verified.json',
                 'scripts/e203_trajectory_production.py'):
        files[name] = R.sha(root / name)
    if files['data/e203_chain_20261007/control_verified.json'] != candidate['control_verified_sha256']:
        raise ValueError('control proof changed')
    model_files = R.verify_cache(root / CACHE, old, setup)
    files.update({CACHE + '/' + n: h for n, h in model_files.items()})
    visit = R.read(root / R.E188 / 'fit_manifest.json')
    files.update({R.E188 + '/' + n: h for n, h in visit['model_sha256'].items()})
    verify_inventory(root, files)
    value = dict(protocol=PROTOCOL, reference=BEST, reference_sha256=BEST_SHA,
        scientific_registration=E203_REG, member=MEMBER, stage_cache=CACHE,
        files=files, original_inputs=setup['inputs'], original_sources=setup['source_after'],
        source_archive=R.ARCHIVE, stage_features=old['stage_features'], fits_authorized=0,
        output=OUTPUT, version_base=990200, original_input_count=len(setup['inputs']),
        original_source_count=len(setup['source_after']))
    R.save(root / SPEC, value)
    print(json.dumps(dict(sealed_files=len(files), oracle_read=False, fits=0)), flush=True)


def check(root):
    spec = R.read(root / SPEC)
    if (spec['protocol'] != PROTOCOL or spec['reference'] != BEST
        or spec['reference_sha256'] != BEST_SHA or spec['fits_authorized'] != 0
        or spec['member'] != MEMBER or spec['stage_cache'] != CACHE):
        raise ValueError('unexpected replay specification')
    verify_inventory(root, spec['files'])
    setup = R.read(root / R.ARCHIVE / 'setup.json')
    if setup['inputs'] != spec['original_inputs'] or setup['source_after'] != spec['original_sources']:
        raise ValueError('original inventory changed')
    verify_e188_globs(root, spec['original_inputs'])
    old = R.read(root / R.SPEC)
    R.verify_cache(root / CACHE, old, setup)
    return spec


def replay_stage(root, output, job_path):
    job = R.read(job_path)
    if job.get('mode') != 'replay':
        raise ValueError('only replay is authorized; zero new fits')
    if job['name'] not in ('e141', 'e64', 'lszh') or Path(job['cache']).resolve() != (root / CACHE / job['name']).resolve():
        raise ValueError('unexpected replay stage cache')
    R.stage_job(root, output, job_path)
    R.save(job_path.with_suffix('.complete.json'), dict(stage=job['name'],
        cached_calls=1 if job['name'] == 'lszh' else 5, fits_performed=0,
        job_sha256=R.sha(job_path), unix=time.time()))


def compare_final(actual_path, reference_path, reference_sha, rows=344841):
    """The only reference-prediction read in this runner is this final comparison."""
    import numpy as np
    import polars as pl
    actual = pl.read_parquet(actual_path)
    if (actual.columns != [ID, Y] or actual.height != rows or actual[ID].n_unique() != rows
        or actual.null_count().to_numpy().any() or not np.isfinite(actual[Y].to_numpy()).all()
        or actual[Y].min() < 0):
        raise ValueError('invalid complete replay output')
    digest = R.sha(actual_path)
    # Freeze the generated output before the first read of the oracle.
    if digest != reference_sha:
        raise ValueError('replay bytes differ from the pinned v12075 reference')
    expected = pl.read_parquet(reference_path)
    if (R.sha(reference_path) != reference_sha or actual.schema != expected.schema
        or not actual.sort(ID).equals(expected.sort(ID))):
        raise ValueError('replay schema/IDs/predictions differ from reference')
    return dict(rows=rows, exact_predictions=True, byte_identical=True,
        output_sha256=digest, reference_sha256=reference_sha, differing_rows=0, max_abs_difference=0)


def reproduce(root, output, commit):
    registration(root, commit)
    if output.resolve() != (root / OUTPUT).resolve() or output.exists():
        raise ValueError('fresh fixed replay output required; no retry')
    if shutil.disk_usage(root).free < 4 * 1024**3:
        raise ValueError('replay needs 4 GiB free')
    spec = check(root)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'data', 'submissions', 'logs', 'jobs'):
        (output / name).mkdir()
    R.save(output / 'started.json', dict(protocol=PROTOCOL, registration=commit,
        unix=time.time(), source_sha256=R.sha(root / SELF), inventory_sha256=R.sha(root / SPEC),
        fits_authorized=0, oracle_read=False))
    try:
        for name in sorted(set(spec['original_inputs']) | {MEMBER}):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((root / name).resolve())
        for name in spec['original_sources']:
            shutil.copy2(root / R.ARCHIVE / name, output / name)
        (output / 'submissions/cat_reg_rank.npz').symlink_to(root / 'submissions/cat_fin1b_seeds15.npz')
        b = 990200
        sub = lambda v: output / f'submissions/gentle-igloo_v{v}.parquet'

        def call(args, extra=None, stage=None):
            if shutil.disk_usage(root).free < 2500 * 1024**2:
                raise ValueError('replay disk floor violated')
            i = len(list((output / 'logs').glob('*.log')))
            if stage:
                job = output / 'jobs' / f'{i:02d}.json'
                R.save(job, dict(mode='replay', name=stage, args=args, cache=str(root / CACHE / stage),
                    worker_sha256=R.sha(root / R.SELF), spec_sha256=R.sha(root / R.SPEC),
                    source_sha256=spec['original_sources']['src/' + args[0]], features=spec['stage_features'][stage]))
                command = [sys.executable, '-u', str(root / SELF), '_stage', '--root', str(root), '--output', str(output), '--job', str(job)]
            else:
                command = [sys.executable, '-u', 'src/' + args[0], *map(str, args[1:])]
            print('STAGE', i, stage or args[0], flush=True)
            with (output / 'logs' / f'{i:02d}_{Path(args[0]).stem}.log').open('x') as log:
                subprocess.run(command, cwd=output, env={**R.environment(), **(extra or {})},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

        anchors = ','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m, w in
            [('raw', '0.0333333333333333'), ('lobt', '.10'), ('sched', '.10'), ('eobt', '.10'), ('aobt', '.10')])
        call(['blend.py', 870, b, .65], {'FEATURES': 'data/features_fin1b.parquet',
            'LGB_SEEDS': 'submissions/lgb_reg_seeds_v870.npz', 'E130_MEMBERS': anchors + ',' + MEMBER + ':0.25'})
        call(['e185_apply.py', b, b+1], {'E185_W': str(.35*2/3), 'E185_RANK': 'data/e185_motion_rank.npz'})
        call(['e167_nm_window_rule.py', b+1, b+2])
        import polars as pl
        donor, previous, base = [pl.read_parquet(p) for p in (output / 'submissions/gentle-igloo_v361.parquet',
            output / 'submissions/gentle-igloo_v291.parquet', sub(b+2))]
        protected = donor.join(previous, on=ID, suffix='_old').filter(pl.col(Y) != pl.col(Y + '_old')).select(ID)
        if protected.height != 42:
            raise ValueError('expected exactly 42 TabPFN donor rows')
        values = donor.join(protected, on=ID).select(ID, pl.col(Y).alias('tab'))
        base.join(values, on=ID, how='left').with_columns(pl.coalesce('tab', Y).alias(Y)).select(base.columns).write_parquet(sub(b+3))
        common = {'STAGE_SEEDS': '42,1,2,3,4', 'ADSB_DIR': 'data/adsblol_kt1', 'PARTS': f'submissions/parts_v{b+2}.npz'}
        call(['e141_apply.py', b+3, b+4], {**common, 'STAGE_KEEP_ALL': '1',
            'DROP_FEATS': 'n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts',
            'WALK_DIR': 'data/adsblol_sw_kt1', 'E36_BASE_DIR': 'data/cv_preds_e116prod'}, 'e141')
        call(['e64_apply.py', b+4, b+5], {**common, 'E64_KEEP_ALL': '1', 'DROP_FEATS': 'vi_n_pts,vi_prev_pts,vi_alt_first'}, 'e64')
        with (output / 'logs/e188_apply.log').open('x') as log:
            subprocess.run([sys.executable, str(root / 'scripts/e188_rank_20261006.py'), 'apply', '--registration-commit', R.E188_REG,
                '--models', str(root / R.E188), '--pre-e64', str(sub(b+4)), '--post-e64', str(sub(b+5)),
                '--parts', str(output / f'submissions/parts_v{b+2}.npz'), '--output', str(sub(b+6))],
                cwd=root, env=R.environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        call(['e152_stage_gate.py', b+3, b+6, b+2, b+10], {'E152_GATE': '0.95'})
        call(['blend_submissions.py', b+11], {'BLEND_BASE': str(b+10), 'BLEND_PARTS': '44:0.1214,60:0.0270'})
        call(['lobt_edge_lift.py', b+11, b+12], {'LOBT_CLIP': '1'})
        call(['protect_tabpfn_rows.py', b+12, b+3, b+13])
        call(['e159b_apply.py', b+13, b+14], {'E159_APS': 'LSZH', 'E159_MONTHS': 'all',
            'E159_BASE': 'data/cv_preds_e150c_e64_keep', 'E159_SHIFT': '0'}, 'lszh')
        call(['blend_submissions.py', b+15], {'BLEND_BASE': str(b+13), 'BLEND_PARTS': f'{b+14}:0.5'})
        final = output / 'reproduced_v12075.parquet'
        shutil.copy2(sub(b+15), final)
        stage_receipts = [R.read(p) for p in (output / 'jobs').glob('*.complete.json')]
        if sorted((r['stage'], r['cached_calls'], r['fits_performed']) for r in stage_receipts) != [('e141', 5, 0), ('e64', 5, 0), ('lszh', 1, 0)]:
            raise ValueError('incomplete zero-fit stage replay evidence')
        actual = pl.read_parquet(final)
        if not actual.join(protected, on=ID).sort(ID).equals(donor.join(protected, on=ID).sort(ID)):
            raise ValueError('protected TabPFN values changed')
        R.save(output / 'assembled_before_oracle.json', dict(sha256=R.sha(final), unix=time.time(),
            rows=actual.height, protected_rows=42, oracle_read=False, fits_performed=0))
        comparison = compare_final(final, root / BEST, BEST_SHA)
        check(root)
        registration(root, commit)
        report = dict(protocol=PROTOCOL, registration=commit, **comparison, protected_rows=42,
            fits_performed=0, cached_tail_models=11, cached_e188_models=5, archived_member_predictions=True,
            raw_data_retrained=False, full_tail_oof_validated=False, uploaded=False, unix=time.time(),
            inventory_sha256=R.sha(root / SPEC), assembled_receipt_sha256=R.sha(output / 'assembled_before_oracle.json'))
        R.save(output / 'complete.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        R.save(output / 'failure.json', dict(error=repr(exc), no_retry=True, partials_preserved=True, unix=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['seal', 'check', 'replay', '_stage'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    p.add_argument('--registration')
    p.add_argument('--job', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'seal':
        seal(root)
    elif a.action == 'check':
        spec = check(root)
        print(json.dumps(dict(protocol=PROTOCOL, files=len(spec['files']), verified=True, oracle_read=False, fits=0)))
    elif a.action == '_stage':
        replay_stage(root, a.output.resolve(), a.job.resolve())
    else:
        reproduce(root, (a.output or root / OUTPUT).resolve(), a.registration)


if __name__ == '__main__':
    main()
