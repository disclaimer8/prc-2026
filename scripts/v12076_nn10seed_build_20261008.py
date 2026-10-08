"""v12076: the v12075 chain with the E203 NN member averaged over ten seeds instead of three.

Zero-fit saved-component replay of v12075 (scripts/reproduce_v12075_v2.py, journal №372/374)
with exactly one input substituted: the 0.25-weight NN member
`data/e203_rank_20261007/prediction.npz` (seeds 0..2) is replaced by
`data/e203_rank_seeds3to9_20261008/models/prediction_10seed.npz` (seeds 0..9, same IDs, order
and dtype). E81 donors, the 42 frozen TabPFN rows, every weight and every cached stage model
stay exactly as in the replay; the frozen production sources are the archived copies. The
member receipt (bins equal, archived member exact, model_0 replay exact, every new checkpoint
replay exact) is required. The result is compared with v12075 only after it is assembled.
Never uploads.

    .venv/bin/python scripts/v12076_nn10seed_build_20261008.py check
    .venv/bin/python scripts/v12076_nn10seed_build_20261008.py build
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
sys.path.insert(0, str(HERE))
import numpy as np
import polars as pl
import reproduce_best as R
import reproduce_v12075_v2 as V
import submission_guard as G

PROTOCOL = 'v12076-v12075-chain-E203-nn-member-tenseed-mean-zero-fit-20261008-v1'
SELF = 'scripts/v12076_nn10seed_build_20261008.py'
VERSION = 12076
SEEDS_PROTOCOL = 'E203-ordered-rank-member-seeds3to9-tenseed-mean-v12075-chain-20261008-v1'
MEMBER_DIR = 'data/e203_rank_seeds3to9_20261008/models'
MEMBER10 = MEMBER_DIR + '/prediction_10seed.npz'
MEMBER_RECEIPT = MEMBER_DIR + '/complete.json'
OUTPUT = 'data/v12076_nn10seed_20261008'
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
MANIFEST = f'submissions/gentle-igloo_v{VERSION}.manifest.json'
TEMPLATE = 'data/prc-2026-datasets/submitting.parquet'
ID, Y = R.ID, R.Y


def verify_member(root):
    """The ten-seed member: receipt, controls, IDs/order/dtype against the archived three-seed member."""
    receipt = R.read(root / MEMBER_RECEIPT)
    if (receipt['protocol'] != SEEDS_PROTOCOL or receipt['registration_commit'] != V.E203_REG
            or receipt['all_seeds'] != list(range(10)) or receipt['seeds'] != list(range(3, 10))
            or receipt['rows'] != 344841 or receipt['uploaded'] is not False
            or not (receipt['bins_equal_archived'] and receipt['archived_member_exact'] and receipt['control_model0_replay_exact'])
            or [f['seed'] for f in receipt['fits']] != list(range(3, 10))
            or any(len(f['losses']) != 8 or not np.isfinite(f['losses']).all() or f['checkpoint_replay_exact'] is not True
                   for f in receipt['fits'])):
        raise ValueError('ten-seed member receipt does not record the registered recipe and controls')
    archived = R.read(root / V.RANK / 'complete.json')
    if (receipt['archived_rank_complete_sha256'] != R.sha(root / V.RANK / 'complete.json')
            or receipt['archived_prediction_sha256'] != archived['files']['prediction.npz']
            or receipt['runtime'] != archived['runtime']):
        raise ValueError('ten-seed member was not fitted against the archived E203 ranking member/runtime')
    for name, digest in receipt['files'].items():
        if R.sha(root / MEMBER_DIR / name) != digest:
            raise ValueError('ten-seed member file changed: ' + name)
    if 'prediction_10seed.npz' not in receipt['files'] or set(f'seed_{i}.npy' for i in range(3, 10)) - set(receipt['files']):
        raise ValueError('ten-seed member inventory incomplete')
    with np.load(root / MEMBER10, allow_pickle=False) as new, np.load(root / V.MEMBER, allow_pickle=False) as old:
        if set(new.files) != {'mvt_id', 'reg', 'reg2'} or set(old.files) != {'mvt_id', 'reg', 'reg2'}:
            raise ValueError('member fields differ')
        if not np.array_equal(new['mvt_id'], old['mvt_id']) or new['mvt_id'].dtype != old['mvt_id'].dtype:
            raise ValueError('member IDs/order differ from the archived member')
        for k in ('reg', 'reg2'):
            if new[k].shape != old[k].shape or new[k].dtype != old[k].dtype or not np.isfinite(new[k]).all():
                raise ValueError('member values invalid: ' + k)
        if not np.array_equal(new['reg'], new['reg2']) or not np.array_equal(old['reg'], old['reg2']):
            raise ValueError('reg/reg2 differ')
        delta = new['reg'] - old['reg']
        G.exact_ids(new['mvt_id'], pl.read_parquet(root / TEMPLATE)[ID].to_numpy(), 344841)
    return dict(receipt_sha256=R.sha(root / MEMBER_RECEIPT), member_sha256=R.sha(root / MEMBER10),
                archived_member_sha256=R.sha(root / V.MEMBER), rows=344841,
                rms_10_vs_3_sec=float(np.sqrt(np.mean(delta ** 2))), mean_shift_sec=float(delta.mean()),
                max_abs_shift_sec=float(np.abs(delta).max()), pairwise_seed_rms_sec=receipt['statistics']['pairwise_seed_rms_sec'])


def check(root):
    spec = V.check(root)       # the whole v12075 replay inventory and the cached stage models, unchanged
    member = verify_member(root)
    return spec, member


def compare_parts(actual_path, reference_path):
    """Premix parity: fields that do not depend on the NN member must be exact; the others are reported."""
    report = {}
    with np.load(actual_path, allow_pickle=False) as a, np.load(reference_path, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('premixture parts fields differ')
        for name in a.files:
            x, y = a[name], b[name]
            if x.dtype != y.dtype or x.shape != y.shape:
                raise ValueError('premixture parts dtype/shape differ: ' + name)
            report[name] = bool(np.array_equal(x, y, equal_nan=True))
    if not report['mvt_id'] or not report['proba']:
        raise ValueError('member-independent premix fields differ (IDs/regime probabilities)')
    return report


def build(root, output):
    if output.resolve() != (root / OUTPUT).resolve() or output.exists():
        raise ValueError('fresh fixed output required; no retry')
    if (root / CANDIDATE).exists() or (root / MANIFEST).exists():
        raise ValueError('candidate version already exists; choose another version')
    if shutil.disk_usage(root).free < 4 * 1024 ** 3:
        raise ValueError('build needs 4 GiB free')
    spec, member = check(root)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'data', 'submissions', 'logs', 'jobs'):
        (output / name).mkdir()
    R.save(output / 'started.json', dict(protocol=PROTOCOL, unix=time.time(), source_sha256=R.sha(root / SELF),
                                          replay_source_sha256=R.sha(root / V.SELF), inventory_sha256=R.sha(root / V.SPEC),
                                          member=MEMBER10, member_sha256=member['member_sha256'], fits_authorized=0))
    try:
        for name in sorted(set(spec['original_inputs']) | {MEMBER10}):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((root / name).resolve())
        for name in spec['original_sources']:
            shutil.copy2(root / R.ARCHIVE / name, output / name)
        (output / 'submissions/cat_reg_rank.npz').symlink_to(root / 'submissions/cat_fin1b_seeds15.npz')
        b = 990200
        sub = lambda v: output / f'submissions/gentle-igloo_v{v}.parquet'

        def call(args, extra=None, stage=None):
            if shutil.disk_usage(root).free < 2500 * 1024 ** 2:
                raise ValueError('build disk floor violated')
            i = len(list((output / 'logs').glob('*.log')))
            if stage:
                job = output / 'jobs' / f'{i:02d}.json'
                R.save(job, dict(mode='replay', name=stage, args=args, cache=str(root / V.CACHE / stage),
                                 worker_sha256=R.sha(root / R.SELF), spec_sha256=R.sha(root / R.SPEC),
                                 source_sha256=spec['original_sources']['src/' + args[0]], features=spec['stage_features'][stage]))
                command = [sys.executable, '-u', str(root / V.SELF), '_stage', '--root', str(root), '--output', str(output), '--job', str(job)]
            elif args[0] == 'blend.py':
                command = [sys.executable, '-u', str(root / V.LOWMEM), '--work', str(output), *map(str, args[1:])]
            else:
                command = [sys.executable, '-u', 'src/' + args[0], *map(str, args[1:])]
            print('STAGE', i, stage or args[0], flush=True)
            with (output / 'logs' / f'{i:02d}_{Path(args[0]).stem}.log').open('x') as log:
                subprocess.run(command, cwd=output, env={**R.environment(), **(extra or {})},
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

        anchors = ','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m, w in
                           [('raw', '0.0333333333333333'), ('lobt', '.10'), ('sched', '.10'), ('eobt', '.10'), ('aobt', '.10')])
        call(['blend.py', 870, b, .65], {'FEATURES': 'data/features_fin1b.parquet',
                                         'LGB_SEEDS': 'submissions/lgb_reg_seeds_v870.npz', 'E130_MEMBERS': anchors + ',' + MEMBER10 + ':0.25'})
        V.verify_inventory(root, spec['comparison_only'])
        parity = compare_parts(output / f'submissions/parts_v{b}.npz', root / 'data/e203_chain_20261007/submissions/parts_v940100.npz')
        R.save(output / 'premix_parity.json', dict(fields_exact=parity, rows=344841, fits_performed=0,
                                                   note='mvt_id/proba must be exact; member-dependent fields differ by design'))
        print('PREMIX PARITY', json.dumps(parity), flush=True)
        call(['e185_apply.py', b, b + 1], {'E185_W': str(.35 * 2 / 3), 'E185_RANK': 'data/e185_motion_rank.npz'})
        call(['e167_nm_window_rule.py', b + 1, b + 2])
        donor, previous, base = [pl.read_parquet(p) for p in (output / 'submissions/gentle-igloo_v361.parquet',
                                                              output / 'submissions/gentle-igloo_v291.parquet', sub(b + 2))]
        protected = donor.join(previous, on=ID, suffix='_old').filter(pl.col(Y) != pl.col(Y + '_old')).select(ID)
        if protected.height != 42:
            raise ValueError('expected exactly 42 TabPFN donor rows')
        values = donor.join(protected, on=ID).select(ID, pl.col(Y).alias('tab'))
        base.join(values, on=ID, how='left').with_columns(pl.coalesce('tab', Y).alias(Y)).select(base.columns).write_parquet(sub(b + 3))
        common = {'STAGE_SEEDS': '42,1,2,3,4', 'ADSB_DIR': 'data/adsblol_kt1', 'PARTS': f'submissions/parts_v{b + 2}.npz'}
        call(['e141_apply.py', b + 3, b + 4], {**common, 'STAGE_KEEP_ALL': '1',
                                                'DROP_FEATS': 'n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts',
                                                'WALK_DIR': 'data/adsblol_sw_kt1', 'E36_BASE_DIR': 'data/cv_preds_e116prod'}, 'e141')
        call(['e64_apply.py', b + 4, b + 5], {**common, 'E64_KEEP_ALL': '1', 'DROP_FEATS': 'vi_n_pts,vi_prev_pts,vi_alt_first'}, 'e64')
        with (output / 'logs/e188_apply.log').open('x') as log:
            subprocess.run([sys.executable, str(root / 'scripts/e188_rank_20261006.py'), 'apply', '--registration-commit', R.E188_REG,
                            '--models', str(root / R.E188), '--pre-e64', str(sub(b + 4)), '--post-e64', str(sub(b + 5)),
                            '--parts', str(output / f'submissions/parts_v{b + 2}.npz'), '--output', str(sub(b + 6))],
                           cwd=root, env=R.environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        call(['e152_stage_gate.py', b + 3, b + 6, b + 2, b + 10], {'E152_GATE': '0.95'})
        call(['blend_submissions.py', b + 11], {'BLEND_BASE': str(b + 10), 'BLEND_PARTS': '44:0.1214,60:0.0270'})
        call(['lobt_edge_lift.py', b + 11, b + 12], {'LOBT_CLIP': '1'})
        call(['protect_tabpfn_rows.py', b + 12, b + 3, b + 13])
        call(['e159b_apply.py', b + 13, b + 14], {'E159_APS': 'LSZH', 'E159_MONTHS': 'all',
                                                   'E159_BASE': 'data/cv_preds_e150c_e64_keep', 'E159_SHIFT': '0'}, 'lszh')
        call(['blend_submissions.py', b + 15], {'BLEND_BASE': str(b + 13), 'BLEND_PARTS': f'{b + 14}:0.5'})
        final = output / f'gentle-igloo_v{VERSION}.parquet'
        shutil.copy2(sub(b + 15), final)
        stage_receipts = [R.read(p) for p in (output / 'jobs').glob('*.complete.json')]
        if sorted((r['stage'], r['cached_calls'], r['fits_performed']) for r in stage_receipts) != [('e141', 5, 0), ('e64', 5, 0), ('lszh', 1, 0)]:
            raise ValueError('incomplete zero-fit stage replay evidence')
        candidate = pl.read_parquet(final)
        if not candidate.join(protected, on=ID).sort(ID).equals(donor.join(protected, on=ID).sort(ID)):
            raise ValueError('protected TabPFN values changed')
        R.save(output / 'assembled_before_reference.json', dict(sha256=R.sha(final), unix=time.time(), rows=candidate.height,
                                                                 protected_rows=42, reference_read=False, fits_performed=0))
        # Only now is v12075 opened, for the comparison and the guard.
        template = pl.read_parquet(root / TEMPLATE)
        reference = pl.read_parquet(root / V.BEST)
        if R.sha(root / V.BEST) != V.BEST_SHA:
            raise ValueError('reference v12075 changed')
        checks = G.validate_submission(candidate, template, reference, pl.read_parquet(root / 'submissions/gentle-igloo_v291.parquet'),
                                       pl.read_parquet(root / 'submissions/gentle-igloo_v361.parquet'))
        if (candidate.schema != template.schema or candidate.columns != template.columns or candidate.height != 344841
                or candidate[ID].n_unique() != 344841 or candidate[Y].null_count() or candidate[Y].dtype != pl.Int32
                or candidate[Y].min() < 0):
            raise ValueError('invalid candidate schema/IDs/values')
        if not candidate[ID].equals(reference[ID]):
            raise ValueError('row order differs from v12075')
        c = candidate[Y].to_numpy().astype(np.int64)
        r = reference[Y].to_numpy().astype(np.int64)
        delta = (c - r).astype(float)
        comparison = dict(rows=344841, changed_rows=int(np.count_nonzero(delta)), rms_vs_v12075=float(np.sqrt(np.mean(delta ** 2))),
                          mean_shift_vs_v12075=float(delta.mean()), max_abs_diff=float(np.abs(delta).max()),
                          reference_sha256=V.BEST_SHA, changed_fraction=float(np.count_nonzero(delta) / 344841))
        if comparison['changed_rows'] == 0:
            raise ValueError('candidate equals v12075; nothing to upload')
        tmp = root / (CANDIDATE + '.tmp')
        if tmp.exists():
            raise ValueError('temporary candidate exists')
        shutil.copy2(final, tmp)
        os.link(tmp, root / CANDIDATE)      # FileExistsError if someone else took the version meanwhile
        tmp.unlink()
        digest = R.sha(root / CANDIDATE)
        if digest != R.sha(final):
            raise ValueError('candidate copy differs')
        report = dict(protocol=PROTOCOL, version=VERSION, candidate=CANDIDATE, sha256=digest, size_bytes=(root / CANDIDATE).stat().st_size,
                      rows=344841, protected_rows=42, fits_performed=0, cached_tail_models=11, cached_e188_models=5,
                      replay_protocol=V.PROTOCOL, replay_inventory_sha256=R.sha(root / V.SPEC), member=member,
                      premix_parity=parity, validation=checks, comparison=comparison, recipe=(
                          'v12075 replay (reproduce_v12075_v2) with the 0.25-weight E203 NN member = mean of seeds 0..9 '
                          'instead of seeds 0..2; same E81 donors, 42 frozen TabPFN rows, weights and cached stage models'),
                      uploaded=False, accepted_for_submission=False, unix=time.time())
        R.save(output / 'complete.json', report)
        with (root / MANIFEST).open('x') as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(dict(version=VERSION, sha256=digest, **comparison)), flush=True)
    except BaseException as exc:
        R.save(output / 'failure.json', dict(error=repr(exc), no_retry=True, partials_preserved=True, unix=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'build'])
    p.add_argument('--root', type=Path, default=HERE.parent)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'check':
        spec, member = check(root)
        print(json.dumps(dict(protocol=PROTOCOL, inventory_files=len(spec['files']), member=member, fits=0), indent=1))
    else:
        build(root, root / OUTPUT)


if __name__ == '__main__':
    main()
