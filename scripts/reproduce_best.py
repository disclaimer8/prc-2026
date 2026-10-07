"""Recompute v12074 from sealed archived components, with no submission action.

check: verify the archive without fitting. capture: rebuild eleven fixed tail
models once. replay: load those models and reject every unexpected training call.
This is archived-component reproduction, not a raw-data training pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

PROTOCOL = 'v12074-archived-component-reproduction-v1'
SELF = 'scripts/reproduce_best.py'
SPEC = 'docs/reproduction_manifest.json'
ARCHIVE = 'data/today_chain_20261006'
BEST = 'submissions/gentle-igloo_v12074.parquet'
MEMBER = 'data/e190_rank_20261006/complete/prediction.npz'
E188 = 'data/e188_rank_20261006_v2'
E188_REG = 'd45cb738fb7671cf289ae8a400a30b5b46dfbb3d'
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def verify_files(root, inventory):
    for name, expected in inventory.items():
        if not (root / name).is_file() or sha(root / name) != expected:
            raise ValueError('missing or changed archived file: ' + name)


def check(root):
    spec = read(root / SPEC)
    if spec['protocol'] != PROTOCOL:
        raise ValueError('unexpected reproduction manifest')
    verify_files(root, spec['pinned_files'])
    setup = read(root / ARCHIVE / 'setup.json')
    if setup['registration'] != spec['assembly_registration']:
        raise ValueError('original assembly registration differs')
    verify_files(root, setup['inputs'])
    verify_files(root / ARCHIVE, setup['source_after'])
    neural = read(root / 'data/e190_rank_20261006/complete/manifest.json')
    verify_files(root / 'data/e190_rank_20261006/complete', neural['files'])
    visit = read(root / E188 / 'fit_manifest.json')
    verify_files(root / E188, visit['model_sha256'])
    if sha(root / MEMBER) != spec['member_sha256']:
        raise ValueError('E190 ordinary member differs')
    return spec, setup


def verify_cache(cache, spec, setup):
    """Accept this tool's cache or the registered E203 exact-control cache."""
    if (cache / 'verified.json').is_file():
        proof = read(cache / 'verified.json')
        if proof['protocol'] != PROTOCOL or not proof['exact_predictions']:
            raise ValueError('cache lacks exact reproduction proof')
        verify_files(cache, proof['files'])
    else:
        proof = read(cache.parent / 'control_verified.json')
        if (proof['protocol'] != 'E203-ordered-trajectory-fixed025-joint-preE81-sixpair-gated-fullrank-fulltail-v1'
            or proof['registration_commit'] != '6e95e3a65936a3401e41499bb340f2d54a1a2121'
            or proof['control_rows_exact'] is not True or set(proof['models']) != {'e141', 'e64', 'lszh'}):
            raise ValueError('not the registered E203 exact-control cache')
        verify_files(cache, {name + '/complete.json':h for name,h in proof['models'].items()})
        control = cache.parent / 'submissions/gentle-igloo_v930115.parquet'
        if sha(control) != proof['control_sha256']:
            raise ValueError('E203 exact-control artifact changed')
    if proof['reference_sha256'] != spec['pinned_files'][BEST]:
        raise ValueError('cache was verified against another reference')
    files = {}
    for name, source in [('e141', 'e141_apply.py'), ('e64', 'e64_apply.py'), ('lszh', 'e159b_apply.py')]:
        record = read(cache / name / 'complete.json')
        if (record['source_sha256'] != setup['source_after']['src/' + source]
            or len(record['calls']) != (1 if name == 'lszh' else 5)):
            raise ValueError('model inventory/source differs')
        files[name + '/complete.json'] = sha(cache / name / 'complete.json')
        files.update({name + f'/model_{i}.txt':item['sha256'] for i,item in enumerate(record['calls'])})
    verify_files(cache, files)
    return files


def environment():
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
    env.update(OMP_NUM_THREADS='6', OMP_THREAD_LIMIT='6', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', POLARS_MAX_THREADS='2', OMP_WAIT_POLICY='PASSIVE',
        GOMP_SPINCOUNT='0', PYTHONUNBUFFERED='1',
        POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR='load_as_storage',
        FEATURES='data/features_e106.parquet', PROXY_SCALE='1', CLF_COST_WEIGHT='1',
        CLF_COST_MIN_SPREAD='3600',
        DROP_PREFIX='day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config')
    return env


def fingerprint(value):
    if value is None:
        return None
    import numpy as np
    x = np.ascontiguousarray(value)
    if x.dtype.hasobject:
        raise ValueError('object-valued training input')
    return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())


def stage_job(root, work, job_path):
    """Fresh LGB-only process: intercept only the original stage's train calls."""
    if 'torch' in sys.modules:
        raise ValueError('LightGBM stage must run without Torch')
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    job = read(job_path)
    if sha(root / SELF) != job['worker_sha256'] or sha(root / SPEC) != job['spec_sha256']:
        raise ValueError('reproduction source changed during execution')
    source = work / 'src' / job['args'][0]
    if sha(source) != job['source_sha256']:
        raise ValueError('immutable stage source differs')
    cache = Path(job['cache'])
    if job['mode'] == 'capture':
        cache.mkdir(parents=True, exist_ok=False)
        stored = None
    else:
        stored = read(cache / 'complete.json')
        if stored['source_sha256'] != job['source_sha256']:
            raise ValueError('cached model source differs')
    original, calls = lgb.train, []

    def wrapped(params, dataset, num_boost_round=100, **kwargs):
        if kwargs:
            raise ValueError('unexpected training-call options')
        caller = inspect.currentframe().f_back
        features = caller.f_locals.get('feats', caller.f_globals.get('F' if job['name'] == 'lszh' else 'FEATS'))
        if list(features or []) != job['features'] or dataset.data.shape[1] != len(features):
            raise ValueError('stage feature names/order differ')
        signature = dict(params=params, rounds=num_boost_round, data=fingerprint(dataset.data),
            label=fingerprint(dataset.label), weight=fingerprint(dataset.weight),
            categorical=dataset.categorical_feature, feature_names=list(features),
            dataset_feature_names=dataset.feature_name, lightgbm=lgb.__version__,
            numpy=np.__version__, polars=pl.__version__, python=sys.version)
        model_path = cache / f'model_{len(calls)}.txt'
        if job['mode'] == 'capture':
            model = original(params, dataset, num_boost_round)
            model.save_model(str(model_path))
            item = dict(signature=signature, sha256=sha(model_path))
        else:
            if len(calls) >= len(stored['calls']):
                raise ValueError('unexpected extra training call')
            item = stored['calls'][len(calls)]
            if signature != item['signature'] or sha(model_path) != item['sha256']:
                raise ValueError('cached training matrix/labels/recipe/model differs')
            model = lgb.Booster(model_file=str(model_path))
        if model.current_iteration() != num_boost_round:
            raise ValueError('incomplete stage model')
        calls.append(item)
        return model

    lgb.train = wrapped
    try:
        os.chdir(work)
        sys.path.insert(0, str(work / 'src'))
        sys.argv = [str(source), *map(str, job['args'][1:])]
        runpy.run_path(str(source), run_name='__main__')
    finally:
        lgb.train = original
    if len(calls) != (1 if job['name'] == 'lszh' else 5):
        raise ValueError('stage call count differs')
    if stored is None:
        save(cache / 'complete.json', dict(source_sha256=job['source_sha256'], calls=calls))
    elif calls != stored['calls']:
        raise ValueError('cached call inventory differs')


def reproduce(root, output, mode, model_cache, registration):
    if mode == 'capture':
        if not registration:
            raise ValueError('capture requires reviewed registration; replay never fits')
        for name in (SELF, SPEC, 'docs/reproduction.md'):
            frozen = subprocess.check_output(['git', 'show', registration + ':' + name], cwd=root)
            if hashlib.sha256(frozen).hexdigest() != sha(root / name):
                raise ValueError('capture code/documentation differs from registration')
        journal = subprocess.check_output(['git', 'show', registration + ':HYPOTHESES_LOG.md'], cwd=root)
        if PROTOCOL.encode() not in journal:
            raise ValueError('capture protocol is not registered')
    spec, setup = check(root)
    if mode == 'replay':
        cache_inventory = verify_cache(model_cache, spec, setup)
    elif model_cache.exists():
        raise ValueError('capture requires a fresh model-cache path')
    if shutil.disk_usage(root).free < 4*1024**3:
        raise ValueError('reproduction requires4GiB free disk at startup')
    if output.exists() or output.resolve() in (root, (root / ARCHIVE).resolve()):
        raise ValueError('fresh isolated output directory required')
    output.mkdir(parents=True, exist_ok=False)
    for directory in ('src', 'submissions', 'logs', 'jobs'):
        (output / directory).mkdir()
    worker_sha, spec_sha = sha(root / SELF), sha(root / SPEC)
    save(output / 'started.json', dict(protocol=PROTOCOL, mode=mode, unix=time.time(),
        worker_sha256=worker_sha, spec_sha256=spec_sha, registration=registration,
        scope='archived feature tables, member predictions and donors; fixed production tail'))
    try:
        for name in sorted(set(setup['inputs']) | {MEMBER}):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((root / name).resolve())
        for name in setup['source_after']:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / ARCHIVE / name, target)
        (output / 'submissions/cat_reg_rank.npz').symlink_to((root / 'submissions/cat_fin1b_seeds15.npz').resolve())

        def sub(v):
            return output / f'submissions/gentle-igloo_v{v}.parquet'

        def call(args, extra=None, name=None):
            if shutil.disk_usage(output).free < 2500*1024**2:
                raise ValueError('stage requires2.5GiB disk headroom')
            index = len(list((output / 'logs').glob('*.log')))
            if name:
                job = output / 'jobs' / f'{index:02d}.json'
                save(job, dict(mode=mode, name=name, args=args, cache=str(model_cache / name),
                    worker_sha256=worker_sha, spec_sha256=spec_sha,
                    source_sha256=setup['source_after']['src/' + args[0]], features=spec['stage_features'][name]))
                command = [sys.executable, str(root / SELF), '_stage', '--root', str(root), '--output', str(output), '--job', str(job)]
            else:
                command = [sys.executable, '-u', 'src/' + args[0], *map(str, args[1:])]
            print('STAGE', index, name or args[0], flush=True)
            with (output / 'logs' / f'{index:02d}_{Path(args[0]).stem}.log').open('x') as log:
                subprocess.run(command, cwd=output, env={**environment(), **(extra or {})}, stdout=log, stderr=subprocess.STDOUT, check=True)

        anchors = ','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m, w in
            [('raw', '0.0333333333333333'), ('lobt', '.10'), ('sched', '.10'), ('eobt', '.10'), ('aobt', '.10')])
        b = 990100
        call(['blend.py', 870, b, .65], {'FEATURES': 'data/features_fin1b.parquet',
             'LGB_SEEDS': 'submissions/lgb_reg_seeds_v870.npz', 'E130_MEMBERS': anchors + ',' + MEMBER + ':0.25'})
        call(['e185_apply.py', b, b+1], {'E185_W': str(.35*2/3), 'E185_RANK': 'data/e185_motion_rank.npz'})
        call(['e167_nm_window_rule.py', b+1, b+2])
        import polars as pl
        donor, previous, base = [pl.read_parquet(p) for p in (output / 'submissions/gentle-igloo_v361.parquet',
            output / 'submissions/gentle-igloo_v291.parquet', sub(b+2))]
        ids = donor.join(previous, on=ID, suffix='_old').filter(pl.col(Y) != pl.col(Y + '_old')).select(ID)
        if ids.height != 42:
            raise ValueError('expected exactly42 TabPFN donor rows')
        values = donor.join(ids, on=ID).select(ID, pl.col(Y).alias('tab'))
        base.join(values, on=ID, how='left').with_columns(pl.coalesce('tab', Y).alias(Y)).select(base.columns).write_parquet(sub(b+3))
        common = {'STAGE_SEEDS':'42,1,2,3,4', 'ADSB_DIR':'data/adsblol_kt1', 'PARTS':f'submissions/parts_v{b+2}.npz'}
        call(['e141_apply.py', b+3, b+4], {**common, 'STAGE_KEEP_ALL':'1',
             'DROP_FEATS':'n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts',
             'WALK_DIR':'data/adsblol_sw_kt1', 'E36_BASE_DIR':'data/cv_preds_e116prod'}, 'e141')
        call(['e64_apply.py', b+4, b+5], {**common, 'E64_KEEP_ALL':'1', 'DROP_FEATS':'vi_n_pts,vi_prev_pts,vi_alt_first'}, 'e64')
        with (output / 'logs/e188_apply.log').open('x') as log:
            subprocess.run([sys.executable, str(root / 'scripts/e188_rank_20261006.py'), 'apply', '--registration-commit', E188_REG,
                '--models', str(root / E188), '--pre-e64', str(sub(b+4)), '--post-e64', str(sub(b+5)),
                '--parts', str(output / f'submissions/parts_v{b+2}.npz'), '--output', str(sub(b+6))],
                cwd=root, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
        call(['e152_stage_gate.py', b+3, b+6, b+2, b+10], {'E152_GATE':'0.95'})
        call(['blend_submissions.py', b+11], {'BLEND_BASE':str(b+10), 'BLEND_PARTS':'44:0.1214,60:0.0270'})
        call(['lobt_edge_lift.py', b+11, b+12], {'LOBT_CLIP':'1'})
        call(['protect_tabpfn_rows.py', b+12, b+3, b+13])
        call(['e159b_apply.py', b+13, b+14], {'E159_APS':'LSZH', 'E159_MONTHS':'all', 'E159_BASE':'data/cv_preds_e150c_e64_keep', 'E159_SHIFT':'0'}, 'lszh')
        call(['blend_submissions.py', b+15], {'BLEND_BASE':str(b+13), 'BLEND_PARTS':f'{b+14}:0.5'})
        # The best submission is read only as a final oracle, never as a prediction input.
        actual, expected = [pl.read_parquet(p).sort(ID) for p in (sub(b+15), root / BEST)]
        if actual.height != 344841 or actual[ID].n_unique() != 344841 or actual.schema != expected.schema or not actual.equals(expected):
            raise ValueError('recomputed predictions do not exactly match all344841 v12074 rows')
        check(root)
        if mode == 'replay':
            verify_files(model_cache, cache_inventory)
        if sha(root / SELF) != worker_sha or sha(root / SPEC) != spec_sha:
            raise ValueError('reproduction code changed during run')
        shutil.copy2(sub(b+15), output / 'reproduced_v12074.parquet')
        report = dict(protocol=PROTOCOL, mode=mode, reference_sha256=spec['pinned_files'][BEST],
            output_sha256=sha(output / 'reproduced_v12074.parquet'), rows=344841, exact_predictions=True,
            byte_identical=sha(output / 'reproduced_v12074.parquet') == spec['pinned_files'][BEST],
            stage_models=11, fits_performed=11 if mode == 'capture' else 0,
            member_predictions_archived=True, raw_data_retrained=False, uploaded=False,
            worker_sha256=worker_sha, spec_sha256=spec_sha)
        if mode == 'capture':
            save(model_cache / 'verified.json', {**report, 'files':{str(p.relative_to(model_cache)):sha(p)
                for p in sorted(model_cache.rglob('*')) if p.is_file()}})
        save(output / 'complete.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        save(output / 'failure.json', dict(error=repr(exc), no_retry=True, partials_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'capture', 'replay', '_stage'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-cache', type=Path)
    parser.add_argument('--registration')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'check':
        spec, setup = check(root)
        print(json.dumps(dict(protocol=PROTOCOL, archived_inputs=len(setup['inputs']),
            frozen_sources=len(setup['source_after']), hashes_verified=True, real_fits=0)))
    elif args.action == '_stage':
        stage_job(root, args.output.resolve(), args.job.resolve())
    else:
        if args.output is None or args.model_cache is None:
            parser.error('capture/replay need --output and --model-cache')
        reproduce(root, args.output.resolve(), args.action, args.model_cache.resolve(), args.registration)


if __name__ == '__main__':
    main()
