"""v12078 / v12079: two single-change measurements on top of v12077 (journal №431/433).

Both are zero-fit saved-component replays identical to scripts/v12077_nn035_build_20261008.py
(v12075 chain, ten-seed E203 NN member, E212 final weights) with exactly one change each:

  noe81      v12078 = v12077 without the E81 whole-file blend (no v44/v60 donors): the stage
             `blend_submissions.py` with BLEND_PARTS 44:0.1214,60:0.0270 is skipped and
             `lobt_edge_lift.py` reads the E152 output directly (as the final assembly does).
             The 42 frozen TabPFN rows stay.
  freshe156  v12079 = v12077 with the 42 frozen TabPFN tail rows (v361 vs v291) replaced by the
             fresh ten-seed E156 ensemble values the 08.10 canary computed on the same Jan/Jul
             inputs (analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json). The
             served ID set must equal the 42 donor IDs; the values enter where the donor values
             entered (b+2 -> b+3) and are restored by the unchanged `protect_tabpfn_rows.py` from
             that file, exactly as the frozen donors were. E81 stays.

Everything else (premix weights, E185/E167/E141/E64/E188/E152/LOBT/LSZH stages, cached stage
models, the ten-seed member) is byte-identical to the v12077 build. v12075/v12076/v12077 are
opened only after assembly. Never uploads.

    .venv/bin/python scripts/v12078_v12079_builds_20261009.py check
    .venv/bin/python scripts/v12078_v12079_builds_20261009.py build --variant noe81
    .venv/bin/python scripts/v12078_v12079_builds_20261009.py build --variant freshe156
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
import v12076_nn10seed_build_20261008 as B76   # verify_member, compare_parts (frozen by SHA in the v12076 transport)

SELF = 'scripts/v12078_v12079_builds_20261009.py'
VARIANTS = {
    'noe81': dict(version=12078, protocol='v12078-v12077-chain-without-E81-whole-file-blend-zero-fit-20261009-v1',
                  output='data/v12078_noe81_20261009'),
    'freshe156': dict(version=12079, protocol='v12079-v12077-chain-fresh-E156-tenseed-tail-rows-zero-fit-20261009-v1',
                      output='data/v12079_freshe156_20261009'),
}
PARENT = 'submissions/gentle-igloo_v12077.parquet'
PARENT_SHA = 'f06c66bfa3388b17f5b6ba2362fa45a7da1be8fd5ae2d37de47cfe0e6bf2ba44'
PARENT_REPORT = 'submissions/gentle-igloo_v12077.manifest.json'
PARENT_PROTOCOL = 'v12077-v12076-chain-E212-weights-nn035-anchors0867-zero-fit-20261008-v1'
PARENT_RMSE = 230.4812
ANCHORS = [('raw', '0.028888888888888857'), ('lobt', '0.08666666666666667'), ('sched', '0.08666666666666667'),
           ('eobt', '0.08666666666666667'), ('aobt', '0.08666666666666667')]
NN_W = '0.35'
E81_PARTS = {44: 0.1214, 60: 0.0270}
E81_SCORES = {44: 256.3401, 60: 252.1044}           # board scores pinned in the frozen blend_submissions.py
SERVED = 'analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'
SERVED_PROTOCOL = 'final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1'
MEMBER10 = B76.MEMBER10
TEMPLATE = B76.TEMPLATE
ID, Y = R.ID, R.Y


def served_rows(root):
    s = R.read(root / SERVED)
    ids = np.asarray(s['ids'], dtype=np.float64)
    values = np.asarray(s['values'])
    if (s.get('protocol') != SERVED_PROTOCOL or s.get('rows') != 42 or len(ids) != 42 or len(values) != 42
            or len(np.unique(ids)) != 42 or np.any(ids != np.floor(ids))
            or any(float(v) != int(v) for v in s['values']) or values.min() < 0):
        raise ValueError('served E156 rows are not the registered 42 integer-valued Jan/Jul rows')
    return s, pl.DataFrame({ID: ids, 'fresh': values.astype(np.int64)})


def check(root):
    spec = V.check(root)
    member = B76.verify_member(root)
    parent_report = R.read(root / PARENT_REPORT)
    if (R.sha(root / PARENT) != PARENT_SHA or parent_report['sha256'] != PARENT_SHA or parent_report['protocol'] != PARENT_PROTOCOL
            or parent_report['weights']['nn'] != NN_W or parent_report['weights']['anchors'] != dict(ANCHORS)
            or parent_report['member']['member_sha256'] != member['member_sha256']):
        raise ValueError('parent v12077 or its report differs from the registered build')
    s, fresh = served_rows(root)
    donor = pl.read_parquet(root / 'submissions/gentle-igloo_v361.parquet')
    previous = pl.read_parquet(root / 'submissions/gentle-igloo_v291.parquet')
    protected = donor.join(previous, on=ID, suffix='_old').filter(pl.col(Y) != pl.col(Y + '_old')).select(ID)
    if protected.height != 42 or set(protected[ID].to_list()) != set(fresh[ID].to_list()):
        raise ValueError('served E156 IDs differ from the 42 frozen donor IDs')
    frozen = donor.join(fresh, on=ID).select(ID, Y, 'fresh')
    delta = (frozen['fresh'] - frozen[Y]).to_numpy().astype(float)
    served = dict(served_sha256=R.sha(root / SERVED), ids_sha256=s['ids_sha256'], rows=42,
                  rms_fresh_vs_frozen=float(np.sqrt(np.mean(delta ** 2))), mean_fresh_minus_frozen=float(delta.mean()),
                  max_abs=float(np.abs(delta).max()))
    return spec, member, served


def e81_identity(root, candidate):
    """Gram identity solved for the E81-free base: v12077 ~ wX*X + w44*v44 + w60*v60 with X ~ v12078."""
    files = {v: pl.read_parquet(root / f'submissions/gentle-igloo_v{v}.parquet') for v in E81_PARTS}
    parent = pl.read_parquet(root / PARENT)
    x = candidate.sort(ID)[Y].to_numpy().astype(float)
    p = parent.sort(ID)[Y].to_numpy().astype(float)
    parts = {v: candidate.sort(ID).select(ID).join(f, on=ID, how='left').sort(ID)[Y].to_numpy().astype(float) for v, f in files.items()}
    if any(not np.isfinite(a).all() for a in parts.values()):
        raise ValueError('E81 donors do not cover the template')
    wx = 1.0 - sum(E81_PARTS.values())
    blend = np.rint(wx * x + sum(w * parts[v] for v, w in E81_PARTS.items()))
    d = lambda a, b: float(np.mean((a - b) ** 2))
    s2 = {v: E81_SCORES[v] ** 2 for v in E81_PARTS}
    w44, w60 = E81_PARTS[44], E81_PARTS[60]
    num = (PARENT_RMSE ** 2 - w44 ** 2 * s2[44] - w60 ** 2 * s2[60]
           - wx * w44 * (s2[44] - d(x, parts[44])) - wx * w60 * (s2[60] - d(x, parts[60]))
           - w44 * w60 * (s2[44] + s2[60] - d(parts[44], parts[60])))
    sx2 = num / wx
    return dict(weights=dict(base=wx, **{f'v{v}': w for v, w in E81_PARTS.items()}), donor_scores=E81_SCORES, parent_rmse=PARENT_RMSE,
                rms_exact_blend_vs_parent=float(np.sqrt(d(blend, p))), rows_blend_differs_from_parent=int(np.count_nonzero(blend != p)),
                implied_rmse_without_e81=float(np.sqrt(sx2)) if sx2 > 0 else None,
                implied_delta_vs_parent=float(np.sqrt(sx2) - PARENT_RMSE) if sx2 > 0 else None,
                note='identity exact for the unrounded blend of three fixed files; v12077 differs from that blend by the tail '
                     'stages applied after E81 (LOBT lift, protect, LSZH half), measured by rms_exact_blend_vs_parent')


def build(root, variant):
    cfg = VARIANTS[variant]
    version, protocol = cfg['version'], cfg['protocol']
    output = root / cfg['output']
    candidate_rel = f'submissions/gentle-igloo_v{version}.parquet'
    manifest_rel = f'submissions/gentle-igloo_v{version}.manifest.json'
    if output.exists():
        raise ValueError('fresh fixed output required; no retry')
    if (root / candidate_rel).exists() or (root / manifest_rel).exists():
        raise ValueError('candidate version already exists; choose another version')
    if shutil.disk_usage(root).free < 4 * 1024 ** 3:
        raise ValueError('build needs 4 GiB free')
    spec, member, served = check(root)
    _, fresh = served_rows(root)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'data', 'submissions', 'logs', 'jobs'):
        (output / name).mkdir()
    R.save(output / 'started.json', dict(protocol=protocol, variant=variant, unix=time.time(), source_sha256=R.sha(root / SELF),
                                          replay_source_sha256=R.sha(root / V.SELF), inventory_sha256=R.sha(root / V.SPEC),
                                          member=MEMBER10, member_sha256=member['member_sha256'], parent=PARENT, parent_sha256=PARENT_SHA,
                                          fits_authorized=0))
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

        anchors = ','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m, w in ANCHORS)
        call(['blend.py', 870, b, .65], {'FEATURES': 'data/features_fin1b.parquet',
                                         'LGB_SEEDS': 'submissions/lgb_reg_seeds_v870.npz', 'E130_MEMBERS': anchors + ',' + MEMBER10 + ':' + NN_W})
        V.verify_inventory(root, spec['comparison_only'])
        parity = B76.compare_parts(output / f'submissions/parts_v{b}.npz', root / 'data/e203_chain_20261007/submissions/parts_v940100.npz')
        parent_parts = root / 'data/v12077_nn035_20261008/submissions/parts_v990200.npz'
        parity_parent = B76.compare_parts(output / f'submissions/parts_v{b}.npz', parent_parts) if parent_parts.is_file() else None
        if parity_parent is not None and not all(parity_parent.values()):
            raise ValueError('premix differs from the v12077 build premix')
        R.save(output / 'premix_parity.json', dict(fields_exact_vs_v940100=parity, fields_exact_vs_v12077_premix=parity_parent, rows=344841, fits_performed=0))
        print('PREMIX PARITY', json.dumps(dict(v940100=parity, v12077=parity_parent)), flush=True)
        call(['e185_apply.py', b, b + 1], {'E185_W': str(.35 * 2 / 3), 'E185_RANK': 'data/e185_motion_rank.npz'})
        call(['e167_nm_window_rule.py', b + 1, b + 2])
        donor, previous, base = [pl.read_parquet(p) for p in (output / 'submissions/gentle-igloo_v361.parquet',
                                                              output / 'submissions/gentle-igloo_v291.parquet', sub(b + 2))]
        protected = donor.join(previous, on=ID, suffix='_old').filter(pl.col(Y) != pl.col(Y + '_old')).select(ID)
        if protected.height != 42:
            raise ValueError('expected exactly 42 TabPFN donor rows')
        if variant == 'freshe156':
            if set(protected[ID].to_list()) != set(fresh[ID].to_list()):
                raise ValueError('served E156 IDs differ from the 42 frozen donor IDs')
            values = fresh.join(protected, on=ID).select(ID, pl.col('fresh').cast(base.schema[Y]).alias('tab'))
            tail_values = fresh.rename({'fresh': Y}).with_columns(pl.col(Y).cast(base.schema[Y]))
        else:
            values = donor.join(protected, on=ID).select(ID, pl.col(Y).alias('tab'))
            tail_values = donor.join(protected, on=ID).select(ID, Y)
        if values.height != 42:
            raise ValueError('tail values incomplete')
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
        if variant == 'noe81':
            lobt_in = b + 10          # the E81 whole-file blend (b+11) is skipped
        else:
            call(['blend_submissions.py', b + 11], {'BLEND_BASE': str(b + 10), 'BLEND_PARTS': '44:0.1214,60:0.0270'})
            lobt_in = b + 11
        call(['lobt_edge_lift.py', lobt_in, b + 12], {'LOBT_CLIP': '1'})
        call(['protect_tabpfn_rows.py', b + 12, b + 3, b + 13])
        call(['e159b_apply.py', b + 13, b + 14], {'E159_APS': 'LSZH', 'E159_MONTHS': 'all',
                                                   'E159_BASE': 'data/cv_preds_e150c_e64_keep', 'E159_SHIFT': '0'}, 'lszh')
        call(['blend_submissions.py', b + 15], {'BLEND_BASE': str(b + 13), 'BLEND_PARTS': f'{b + 14}:0.5'})
        final = output / f'gentle-igloo_v{version}.parquet'
        shutil.copy2(sub(b + 15), final)
        stage_receipts = [R.read(p) for p in (output / 'jobs').glob('*.complete.json')]
        if sorted((r['stage'], r['cached_calls'], r['fits_performed']) for r in stage_receipts) != [('e141', 5, 0), ('e64', 5, 0), ('lszh', 1, 0)]:
            raise ValueError('incomplete zero-fit stage replay evidence')
        candidate = pl.read_parquet(final)
        kept = candidate.join(protected, on=ID).sort(ID)
        expected_tail = tail_values.join(protected, on=ID).sort(ID).select(candidate.columns)
        if kept.height != 42 or not kept.equals(expected_tail):
            raise ValueError('the 42 tail rows were not preserved exactly')
        R.save(output / 'assembled_before_reference.json', dict(sha256=R.sha(final), unix=time.time(), rows=candidate.height,
                                                                 protected_rows=42, reference_read=False, fits_performed=0))
        template = pl.read_parquet(root / TEMPLATE)
        if (candidate.schema != template.schema or candidate.columns != template.columns or candidate.height != 344841
                or candidate[ID].n_unique() != 344841 or candidate[Y].null_count() or candidate[Y].dtype != pl.Int32
                or candidate[Y].min() < 0):
            raise ValueError('invalid candidate schema/IDs/values')
        G.exact_ids(candidate[ID].to_numpy(), template[ID].to_numpy(), 344841)
        c = candidate[Y].to_numpy().astype(np.int64)
        comparisons = {}
        for label, path, sha in (('v12077', PARENT, PARENT_SHA),
                                 ('v12076', 'submissions/gentle-igloo_v12076.parquet', 'c22de523cd392fb035f9223c1ac2222173de0967ef5a8909ee9641876828e1b0'),
                                 ('v12075', V.BEST, V.BEST_SHA)):
            ref = pl.read_parquet(root / path)
            if sha is not None and R.sha(root / path) != sha:
                raise ValueError('reference changed: ' + label)
            if not candidate[ID].equals(ref[ID]):
                raise ValueError('row order differs from ' + label)
            delta = (c - ref[Y].to_numpy().astype(np.int64)).astype(float)
            comparisons[label] = dict(rows=344841, changed_rows=int(np.count_nonzero(delta)), rms=float(np.sqrt(np.mean(delta ** 2))),
                                      mean_shift=float(delta.mean()), max_abs_diff=float(np.abs(delta).max()), reference_sha256=R.sha(root / path))
        if comparisons['v12077']['changed_rows'] == 0:
            raise ValueError('candidate equals v12077; nothing to upload')
        identity = e81_identity(root, candidate) if variant == 'noe81' else None
        tmp = root / (candidate_rel + '.tmp')
        if tmp.exists():
            raise ValueError('temporary candidate exists')
        shutil.copy2(final, tmp)
        os.link(tmp, root / candidate_rel)
        tmp.unlink()
        digest = R.sha(root / candidate_rel)
        if digest != R.sha(final):
            raise ValueError('candidate copy differs')
        report = dict(protocol=protocol, variant=variant, version=version, candidate=candidate_rel, sha256=digest,
                      size_bytes=(root / candidate_rel).stat().st_size, rows=344841, protected_rows=42,
                      tail_rows='fresh-E156-served' if variant == 'freshe156' else 'frozen-v361-donors',
                      e81_blend='absent' if variant == 'noe81' else 'present-44:0.1214,60:0.0270',
                      fits_performed=0, cached_tail_models=11, cached_e188_models=5, replay_protocol=V.PROTOCOL,
                      replay_inventory_sha256=R.sha(root / V.SPEC), member=member, served=served,
                      parent=PARENT, parent_sha256=PARENT_SHA, parent_protocol=PARENT_PROTOCOL, parent_report_sha256=R.sha(root / PARENT_REPORT),
                      weights=dict(anchors=dict(ANCHORS), nn=NN_W, cat_inside_trees='0.65'), premix_parity=dict(v940100=parity, v12077=parity_parent),
                      comparison=comparisons, e81_identity=identity, uploaded=False, accepted_for_submission=False, unix=time.time())
        R.save(output / 'complete.json', report)
        with (root / manifest_rel).open('x') as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(dict(version=version, sha256=digest, comparison=comparisons, e81_identity=identity)), flush=True)
    except BaseException as exc:
        R.save(output / 'failure.json', dict(error=repr(exc), no_retry=True, partials_preserved=True, unix=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'build'])
    p.add_argument('--variant', choices=sorted(VARIANTS))
    p.add_argument('--root', type=Path, default=HERE.parent)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'check':
        spec, member, served = check(root)
        print(json.dumps(dict(inventory_files=len(spec['files']), member=member, served=served, fits=0), indent=1))
    else:
        if not a.variant:
            p.error('--variant is required')
        build(root, a.variant)


if __name__ == '__main__':
    main()
