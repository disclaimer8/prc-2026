"""v12081: v12077 + the E218 PRU level stage (journal №451/452) as the single change (journal №455).

Zero-fit saved-component replay identical to scripts/v12078_v12079_builds_20261009.py / the v12077 build
(v12075 chain, ten-seed E203 NN member, E212 final weights, E81 donors, 42 frozen TabPFN rows, cached stage
models). The rebuilt `half` output (b+15) must be byte-identical to v12077 (SHA f06c66bf) before anything else
happens. Then exactly one post-mixture stage is added, as the E218 patch adds it to the final assembly after
`half` (b+15 -> b+16): `pru_level_shift` with k = 0.5, xs_max = 0.005, the 2025 training labels, the ranking of the
board (data/prc-2026-datasets/ranking.parquet, Jan/Jul 2026), the two SHA-pinned PRU files. The protected rows are the
42 frozen TabPFN rows of this chain (the role the 62 E156 rows play in the final): they are never moved.

The stage code is not retyped: `pru_read`, `pru_ok`, `pru_reference`, `pru_level_shift` and the stage's `dep` reader
are taken verbatim from the post-image of the SHA-pinned patch
analysis/e218_pru_level_recheck_20261010/final_tail_e218_pru_level_stage.patch (0613fdd9). `parity` proves the
extraction: on the final 76c32426 (read only) with the final workspace inputs it must reproduce the E218 dry-run
data/e218_pru_level_20261010/dryrun_final_plus_e218_pru_level.parquet (bbae2cc5) on all 670 790 rows.

    .venv/bin/python scripts/v12081_e218_build_20261011.py check
    .venv/bin/python scripts/v12081_e218_build_20261011.py parity     # read-only, final 76c32426 -> E218 dry-run
    .venv/bin/python scripts/v12081_e218_build_20261011.py forecast   # stage on the pinned v12077; board-model forecast
    .venv/bin/python scripts/v12081_e218_build_20261011.py build
Never uploads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
sys.path.insert(0, str(HERE))
import numpy as np
import polars as pl
import reproduce_best as R
import reproduce_v12075_v2 as V
import submission_guard as G
import v12076_nn10seed_build_20261008 as B76   # verify_member, compare_parts (frozen by SHA in the v12076 transport)

SELF = 'scripts/v12081_e218_build_20261011.py'
VERSION = 12081
PROTOCOL = 'v12081-v12077-chain-plus-E218-PRU-level-stage-zero-fit-20261011-v1'
OUTPUT = 'data/v12081_e218_20261011'
ANALYSIS = 'analysis/v12081_upload_20261011'
CANDIDATE = f'submissions/gentle-igloo_v{VERSION}.parquet'
MANIFEST = f'submissions/gentle-igloo_v{VERSION}.manifest.json'
PARENT = 'submissions/gentle-igloo_v12077.parquet'
PARENT_SHA = 'f06c66bfa3388b17f5b6ba2362fa45a7da1be8fd5ae2d37de47cfe0e6bf2ba44'
PARENT_REPORT = 'submissions/gentle-igloo_v12077.manifest.json'
PARENT_REPORT_SHA = '368798d257f5766bdf58a17c2f041b8a553fde7b09c272eacbb37715717eca5e'
PARENT_PROTOCOL = 'v12077-v12076-chain-E212-weights-nn035-anchors0867-zero-fit-20261008-v1'
PARENT_RMSE = 230.4812
ANCHORS = [('raw', '0.028888888888888857'), ('lobt', '0.08666666666666667'), ('sched', '0.08666666666666667'),
           ('eobt', '0.08666666666666667'), ('aobt', '0.08666666666666667')]
NN_W = '0.35'
PATCH = 'analysis/e218_pru_level_recheck_20261010/final_tail_e218_pru_level_stage.patch'
PATCH_SHA = '0613fdd9bab1aad457c12ed466c5ef05ecadc8e21ddda0e599287118544b4615'
EVALUATE = 'analysis/e218_pru_level_recheck_20261010/evaluate.json'
EVALUATE_SHA = '14f656e4'                     # prefix recorded in journal №452
PRU = {2025: ('data/taxi_out_additional_time_2025.csv', '4aedf19520d3f84bebc418b56835762c907b714a9c19e751bb8658f9e62ce598'),
       2026: ('data/taxi_out_additional_time_2026.csv', 'b72040e0987e6c7d5706279d645ef9f814a6506165c83f89d8509022154a1076')}
DATASETS = 'data/prc-2026-datasets'
RANKING = DATASETS + '/ranking.parquet'       # board ranking, Jan/Jul 2026 (the rows v12077 predicts)
TEMPLATE = B76.TEMPLATE
# parity inputs (read only): the final, its workspace ranking and E156 served IDs, the E218 dry-run
FINAL = 'data/final_tail_20261009/work/gentle-igloo_final.parquet'
FINAL_SHA = '76c32426d750040690ff68bacd6ecbf3a7aa1e63340df5b9271a324b7064d554'
FINAL_RANKING = 'data/final_tail_20261009/work/data/prc-2026-datasets/ranking.parquet'
FINAL_SERVED = 'data/final_tail_20261009/work/e156/run_0/served.json'
DRYRUN = 'data/e218_pru_level_20261010/dryrun_final_plus_e218_pru_level.parquet'
DRYRUN_SHA = 'bbae2cc544989a0faf68abb85dbb4a9aab5ea38eec98d01fa9282c332b4234a1'
K, XS_MAX = 0.5, 0.005
N_BOARD = 344841
ID, Y = R.ID, R.Y


# ------------------------------------------------------------------------------------------------ the pinned stage code
def stage_code(root):
    """Namespace with the E218 stage functions, executed from the post-image of the SHA-pinned patch (no retyping)."""
    if R.sha(root / PATCH) != PATCH_SHA:
        raise ValueError('E218 patch changed')
    lines = (root / PATCH).read_text().splitlines()
    hunks, cur = [], None
    for line in lines:
        if line.startswith('@@'):
            cur = []
            hunks.append(cur)
        elif cur is not None and not line.startswith(('---', '+++')):
            if line.startswith('+') or line.startswith(' '):
                cur.append(line[1:])
            elif line == '':
                cur.append('')
    # first hunk: the constants block from `PRU_LEVEL =` up to `PRU_GRP, PRU_KEY = ...`
    h0 = hunks[0]
    i0 = next(i for i, ln in enumerate(h0) if ln.startswith('PRU_LEVEL ='))
    i1 = next(i for i, ln in enumerate(h0) if ln.startswith('PRU_GRP, PRU_KEY ='))
    consts = '\n'.join(h0[i0:i1 + 1])
    body = next(h for h in hunks if any(ln.startswith('def pru_level_shift(') for ln in h))
    j0 = next(i for i, ln in enumerate(body) if ln.startswith('def pru_read('))
    j1 = next(i for i, ln in enumerate(body) if ln.startswith('def pru_level_stage('))
    funcs = '\n'.join(body[j0:j1])
    k0 = next(i for i, ln in enumerate(body) if ln.startswith("    cols = [ID, 'PHASE_mvt'"))
    k1 = next(i for i, ln in enumerate(body) if ln.startswith("        return f.rename({Y: 'y'})"))
    reader = textwrap.dedent('\n'.join(body[k0:k1 + 1]))
    ns = dict(os=os, pl=pl, np=np, Path=Path, ID=ID, Y=Y, sha=R.sha)
    exec(compile(consts + '\n\n' + funcs + '\n\n' + reader + '\n', str(root / PATCH) + ':post-image', 'exec'), ns)
    if ns['PRU_K'] != K or ns['PRU_XS_MAX'] != XS_MAX or {y: tuple(v) for y, v in ns['PRU_CSV'].items()} != PRU:
        raise ValueError('patch constants differ from the registered rule')
    return ns


def stage_inputs(root, ns, ranking_path, id_dtype):
    """Exactly what pru_level_stage reads: the 2025 training DEP rows with labels, the ranking DEP rows, both PRU files."""
    train = pl.concat([ns['dep'](p, True) for p in sorted((root / DATASETS).glob('training_*.parquet'))])
    rank = ns['dep'](ranking_path, False).with_columns(pl.col(ID).cast(id_dtype))
    pru25, pru26 = (ns['pru_read'](root, y) for y in (2025, 2026))
    return train, rank, pru25, pru26


def frozen_tabpfn_ids(root):
    donor = pl.read_parquet(root / 'submissions/gentle-igloo_v361.parquet')
    previous = pl.read_parquet(root / 'submissions/gentle-igloo_v291.parquet')
    protected = donor.join(previous, on=ID, suffix='_old').filter(pl.col(Y) != pl.col(Y + '_old')).select(ID)
    if protected.height != 42:
        raise ValueError('expected exactly 42 TabPFN donor rows')
    return protected, donor


def apply_stage(root, ns, pred, ranking_path, protected_ids):
    """The E218 stage on `pred` (columns ID, Y), then the template/parent row order restored (as finalize does)."""
    train, rank, pru25, pru26 = stage_inputs(root, ns, ranking_path, pred.schema[ID])
    out, cells = ns['pru_level_shift'](train, rank, pred, pru25, pru26, np.asarray(protected_ids, dtype=np.float64), k=K, xs_max=XS_MAX)
    ordered = pred.select(ID).join(out, on=ID, how='left', maintain_order='left').with_columns(pl.col(Y).cast(pred.schema[Y]))
    if ordered.height != pred.height or ordered[Y].null_count() or not ordered[ID].equals(pred[ID]):
        raise ValueError('stage output does not cover the input rows in order')
    return ordered, cells, rank


def describe(root, ns, pred, new, cells, rank, protected_ids, beta, se, ref_rmse):
    """Changed rows, RMS of the change, per-cell table and the E218 board model (same formula as `board` of the E218 script)."""
    p = pred.rename({Y: 'old'}).with_columns(pl.col('old').cast(pl.Float64))
    x = p.join(new.rename({Y: 'new'}).with_columns(pl.col('new').cast(pl.Float64)), on=ID, maintain_order='left')
    x = x.join(rank.select(ID, 'ADEP_mvt', 'month', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt'), on=ID, how='left', maintain_order='left')
    if x['ADEP_mvt'].null_count():
        raise ValueError('ranking does not cover the file')
    train = pl.concat([ns['dep'](q, True) for q in sorted((root / DATASETS).glob('training_*.parquet'))])
    ref = ns['pru_reference'](train)
    x = x.join(ref, on=ns['PRU_GRP'], how='left', maintain_order='left').with_columns(
        inP=ns['pru_ok']('old') & pl.col('valid').fill_null(False) & ~pl.col(ID).is_in(list(np.asarray(protected_ids, dtype=np.float64))),
        d=pl.col('new') - pl.col('old'))
    x = x.join(cells.select(ns['PRU_KEY'] + ['c', 'eligible']), on=ns['PRU_KEY'], how='left', maintain_order='left')
    changed = x.filter(pl.col('d') != 0)
    if changed.filter(~pl.col('inP') | ~pl.col('eligible').fill_null(False)).height:
        raise ValueError('a row outside P of an eligible cell changed')
    if changed.filter((pl.col('d') - pl.col('c')).abs() > 0.5 + 1e-9).height:
        raise ValueError('a row moved by more than |c| + rounding')
    prot = x.filter(pl.col(ID).is_in(list(np.asarray(protected_ids, dtype=np.float64))))
    if prot.height != len(protected_ids) or prot.filter(pl.col('d') != 0).height:
        raise ValueError('a protected row changed')
    d = x['d'].to_numpy()
    pc = x.filter('inP').group_by(ns['PRU_KEY']).agg(nP=pl.len(), mean_shift=pl.col('d').mean(), sum_d=pl.col('d').sum(),
                                                       sum_d2=(pl.col('d') ** 2).sum(), changed=(pl.col('d') != 0).sum())
    cells = cells.join(pc, on=ns['PRU_KEY'], how='left').with_columns(
        pl.col('nP').fill_null(0), pl.col('changed').fill_null(0), pl.col('sum_d').fill_null(0.0), pl.col('sum_d2').fill_null(0.0),
        x_bias=pl.col('ind') - pl.col('pru_add_s') - pl.col('emu_hat').fill_null(0.0)).sort(ns['PRU_KEY'])
    cg = cells.filter('eligible')
    xg, cc, nP = cg['x_bias'].to_numpy(), cg['c'].to_numpy(), cg['nP'].to_numpy().astype(float)
    sd, sd2 = cg['sum_d'].to_numpy(), cg['sum_d2'].to_numpy()
    lirf = (cg['ADEP_mvt'] == 'LIRF').to_numpy()
    n = pred.height
    scen = {'beta_hat': beta, 'beta_lo_2se': max(beta - 2 * se, 0.0), 'beta_hi_2se': beta + 2 * se, 'beta_1': 1.0, 'beta_0_gap_is_noise': 0.0}
    est = {}
    for name, bb in list(scen.items()) + [('lirf_gap_is_artifact', None)]:
        b = np.where(lirf, 0.0, beta * xg) if bb is None else bb * xg
        model = float((nP * (cc ** 2 + 2 * cc * b)).sum())                       # E218 `board` formula (cell c, no rounding)
        exact = float((sd2 + 2 * sd * b).sum())                                  # same bias model with the actual rounded row changes
        est[name] = dict(beta=bb, dsse_model=model, dmse_model=model / n, drmse_model=float(np.sqrt(ref_rmse ** 2 + model / n) - ref_rmse),
                         dsse_rounded=exact, drmse_rounded=float(np.sqrt(ref_rmse ** 2 + exact / n) - ref_rmse))
    lirf_share = float((nP * (cc ** 2 + 2 * cc * beta * xg))[lirf].sum() / (nP * (cc ** 2 + 2 * cc * beta * xg)).sum())
    return dict(rows=n, changed_rows=int(np.count_nonzero(d)), rms_change=float(np.sqrt(np.mean(d ** 2))), mean_change=float(d.mean()),
                max_abs_change=float(np.abs(d).max()), protected_rows=int(len(protected_ids)), protected_unchanged=True,
                cells_total=int(cells.height), cells_eligible=int(cg.height), rows_in_P_eligible=int(nP.sum()),
                reference_rmse=ref_rmse, bias_model=dict(beta=beta, se_jackknife_airport=se, source=EVALUATE), estimate=est,
                lirf_share_of_beta_hat=lirf_share), cells


def bias_model(root):
    ev = R.read(root / EVALUATE)
    if not R.sha(root / EVALUATE).startswith(EVALUATE_SHA) or not ev['verdict_primary']['passes']:
        raise ValueError('E218 evaluate.json differs from the verdict №452')
    return ev['bias_model_2025']['beta'], ev['bias_model_2025']['beta_se_jackknife_airport']


# ------------------------------------------------------------------------------------------------ actions
def check(root):
    spec = V.check(root)
    member = B76.verify_member(root)
    report = R.read(root / PARENT_REPORT)
    if (R.sha(root / PARENT) != PARENT_SHA or R.sha(root / PARENT_REPORT) != PARENT_REPORT_SHA or report['sha256'] != PARENT_SHA
            or report['protocol'] != PARENT_PROTOCOL or report['weights']['nn'] != NN_W or report['weights']['anchors'] != dict(ANCHORS)
            or report['member']['member_sha256'] != member['member_sha256']):
        raise ValueError('parent v12077 or its report differs from the registered build')
    for y, (path, digest) in PRU.items():
        if R.sha(root / path) != digest:
            raise ValueError('PRU file changed: ' + path)
    ns = stage_code(root)
    return spec, member, ns


def parity(root):
    """Read only: the extracted stage on the final 76c32426 must equal the E218 dry-run on every row."""
    ns = stage_code(root)
    if R.sha(root / FINAL) != FINAL_SHA or R.sha(root / DRYRUN) != DRYRUN_SHA:
        raise ValueError('final or E218 dry-run changed')
    served = R.read(root / FINAL_SERVED)
    final = pl.read_parquet(root / FINAL)
    new, cells, _ = apply_stage(root, ns, final, root / FINAL_RANKING, served['ids'])
    dry = pl.read_parquet(root / DRYRUN)
    same = new.equals(dry)
    changed = int((new[Y].to_numpy() != final[Y].to_numpy()).sum())
    out = dict(final_sha256=FINAL_SHA, dryrun_sha256=DRYRUN_SHA, rows=final.height, equal_to_dryrun=bool(same), rows_changed=changed,
               protected_rows=len(served['ids']), cells_eligible=int(cells['eligible'].sum()), patch_sha256=PATCH_SHA, unix=time.time())
    if R.sha(root / FINAL) != FINAL_SHA:
        raise ValueError('final changed during parity')
    if not same or changed != 415697:
        raise ValueError('extracted stage does not reproduce the E218 dry-run: ' + json.dumps(out))
    return out


def forecast(root):
    """The stage applied to the pinned v12077 file (no rebuild) and the E218 board model recomputed on v12077."""
    ns = stage_code(root)
    if R.sha(root / PARENT) != PARENT_SHA:
        raise ValueError('v12077 changed')
    protected, _ = frozen_tabpfn_ids(root)
    parent = pl.read_parquet(root / PARENT)
    new, cells, rank = apply_stage(root, ns, parent, root / RANKING, protected[ID].to_list())
    beta, se = bias_model(root)
    summary, table = describe(root, ns, parent, new, cells, rank, protected[ID].to_list(), beta, se, PARENT_RMSE)
    summary.update(ranking=RANKING, ranking_sha256=R.sha(root / RANKING), parent_sha256=PARENT_SHA, patch_sha256=PATCH_SHA,
                   expected_values_sha256=hashlib.sha256(new[Y].to_numpy().astype('<i4').tobytes()).hexdigest())
    return summary, table, new


def build(root):
    output = root / OUTPUT
    if output.exists():
        raise ValueError('fresh fixed output required; no retry')
    if (root / CANDIDATE).exists() or (root / MANIFEST).exists():
        raise ValueError('candidate version already exists; choose another version')
    if shutil.disk_usage(root).free < 4 * 1024 ** 3:
        raise ValueError('build needs 4 GiB free')
    spec, member, ns = check(root)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('src', 'data', 'submissions', 'logs', 'jobs'):
        (output / name).mkdir()
    R.save(output / 'started.json', dict(protocol=PROTOCOL, unix=time.time(), source_sha256=R.sha(root / SELF),
                                          replay_source_sha256=R.sha(root / V.SELF), inventory_sha256=R.sha(root / V.SPEC),
                                          member=B76.MEMBER10, member_sha256=member['member_sha256'], parent=PARENT, parent_sha256=PARENT_SHA,
                                          patch_sha256=PATCH_SHA, fits_authorized=0))
    try:
        for name in sorted(set(spec['original_inputs']) | {B76.MEMBER10}):
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

        # ---- the v12077 chain, unchanged (scripts/v12077_nn035_build_20261008.py / v12078_v12079_builds_20261009.py)
        anchors = ','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m, w in ANCHORS)
        call(['blend.py', 870, b, .65], {'FEATURES': 'data/features_fin1b.parquet',
                                         'LGB_SEEDS': 'submissions/lgb_reg_seeds_v870.npz', 'E130_MEMBERS': anchors + ',' + B76.MEMBER10 + ':' + NN_W})
        V.verify_inventory(root, spec['comparison_only'])
        parity_parts = B76.compare_parts(output / f'submissions/parts_v{b}.npz', root / 'data/e203_chain_20261007/submissions/parts_v940100.npz')
        parent_parts = root / 'data/v12077_nn035_20261008/submissions/parts_v990200.npz'
        parity_parent = B76.compare_parts(output / f'submissions/parts_v{b}.npz', parent_parts) if parent_parts.is_file() else None
        if parity_parent is not None and not all(parity_parent.values()):
            raise ValueError('premix differs from the v12077 build premix')
        R.save(output / 'premix_parity.json', dict(fields_exact_vs_v940100=parity_parts, fields_exact_vs_v12077_premix=parity_parent, rows=N_BOARD, fits_performed=0))
        print('PREMIX PARITY', json.dumps(dict(v940100=parity_parts, v12077=parity_parent)), flush=True)
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
        stage_receipts = [R.read(p) for p in (output / 'jobs').glob('*.complete.json')]
        if sorted((r['stage'], r['cached_calls'], r['fits_performed']) for r in stage_receipts) != [('e141', 5, 0), ('e64', 5, 0), ('lszh', 1, 0)]:
            raise ValueError('incomplete zero-fit stage replay evidence')
        # ---- gate 1: the rebuilt chain output is v12077, byte for byte
        rebuilt_sha = R.sha(sub(b + 15))
        R.save(output / 'rebuild_v12077.json', dict(path=str(sub(b + 15).relative_to(root)), sha256=rebuilt_sha, expected=PARENT_SHA,
                                                    byte_identical=rebuilt_sha == PARENT_SHA, unix=time.time()))
        print('REBUILD v12077', rebuilt_sha, rebuilt_sha == PARENT_SHA, flush=True)
        if rebuilt_sha != PARENT_SHA:
            raise ValueError('rebuilt v12077 is not byte-identical; stop before the E218 stage')
        # ---- the single change: E218 PRU level stage b+15 -> b+16 (post-image of the pinned patch)
        src = pl.read_parquet(sub(b + 15))
        new, cells, rank = apply_stage(root, ns, src, root / RANKING, protected[ID].to_list())
        new.write_parquet(sub(b + 16))
        final = output / f'gentle-igloo_v{VERSION}.parquet'
        shutil.copy2(sub(b + 16), final)
        candidate = pl.read_parquet(final)
        beta, se = bias_model(root)
        summary, table = describe(root, ns, src, candidate, cells, rank, protected[ID].to_list(), beta, se, PARENT_RMSE)
        table.write_csv(output / 'cells_v12077_e218.csv', float_precision=4)
        R.save(output / 'assembled_before_reference.json', dict(sha256=R.sha(final), unix=time.time(), rows=candidate.height,
                                                                 protected_rows=42, reference_read=False, fits_performed=0))
        template = pl.read_parquet(root / TEMPLATE)
        if (candidate.schema != template.schema or candidate.columns != template.columns or candidate.height != N_BOARD
                or candidate[ID].n_unique() != N_BOARD or candidate[Y].null_count() or candidate[Y].dtype != pl.Int32
                or candidate[Y].min() < 0):
            raise ValueError('invalid candidate schema/IDs/values')
        G.exact_ids(candidate[ID].to_numpy(), template[ID].to_numpy(), N_BOARD)
        c = candidate[Y].to_numpy().astype(np.int64)
        comparisons = {}
        for label, path, digest in (('v12077', PARENT, PARENT_SHA),
                                    ('v12076', 'submissions/gentle-igloo_v12076.parquet', 'c22de523cd392fb035f9223c1ac2222173de0967ef5a8909ee9641876828e1b0'),
                                    ('v12075', V.BEST, V.BEST_SHA)):
            ref = pl.read_parquet(root / path)
            if R.sha(root / path) != digest:
                raise ValueError('reference changed: ' + label)
            if not candidate[ID].equals(ref[ID]):
                raise ValueError('row order differs from ' + label)
            delta = (c - ref[Y].to_numpy().astype(np.int64)).astype(float)
            comparisons[label] = dict(rows=N_BOARD, changed_rows=int(np.count_nonzero(delta)), rms=float(np.sqrt(np.mean(delta ** 2))),
                                      mean_shift=float(delta.mean()), max_abs_diff=float(np.abs(delta).max()), reference_sha256=digest)
        if comparisons['v12077']['changed_rows'] != summary['changed_rows'] or comparisons['v12077']['changed_rows'] == 0:
            raise ValueError('difference against v12077 inconsistent')
        tmp = root / (CANDIDATE + '.tmp')
        if tmp.exists():
            raise ValueError('temporary candidate exists')
        shutil.copy2(final, tmp)
        os.link(tmp, root / CANDIDATE)
        tmp.unlink()
        digest = R.sha(root / CANDIDATE)
        if digest != R.sha(final):
            raise ValueError('candidate copy differs')
        report = dict(protocol=PROTOCOL, version=VERSION, candidate=CANDIDATE, sha256=digest, size_bytes=(root / CANDIDATE).stat().st_size,
                      rows=N_BOARD, protected_rows=42, tail_rows='frozen-v361-donors', e81_blend='present-44:0.1214,60:0.0270',
                      e218_stage=dict(k=K, xs_max=XS_MAX, patch=PATCH, patch_sha256=PATCH_SHA, after='half (b+15 -> b+16)',
                                      ranking=RANKING, ranking_sha256=R.sha(root / RANKING), pru_sha256={str(y): v[1] for y, v in PRU.items()},
                                      protected='42 frozen TabPFN rows (v361 vs v291)', cells=table.select(
                                          ['ADEP_mvt', 'month', 'ind', 'pru_add_s', 'xs', 'emu_hat', 'eligible', 'c', 'nP', 'changed']).to_dicts()),
                      rebuild_v12077=dict(sha256=rebuilt_sha, byte_identical=True), change=summary,
                      fits_performed=0, cached_tail_models=11, cached_e188_models=5, replay_protocol=V.PROTOCOL,
                      replay_inventory_sha256=R.sha(root / V.SPEC), member=member,
                      parent=PARENT, parent_sha256=PARENT_SHA, parent_protocol=PARENT_PROTOCOL, parent_report_sha256=PARENT_REPORT_SHA,
                      weights=dict(anchors=dict(ANCHORS), nn=NN_W, cat_inside_trees='0.65'),
                      premix_parity=dict(v940100=parity_parts, v12077=parity_parent),
                      comparison=comparisons, uploaded=False, accepted_for_submission=False, unix=time.time())
        R.save(output / 'complete.json', report)
        with (root / MANIFEST).open('x') as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(dict(version=VERSION, sha256=digest, comparison=comparisons,
                              change={k: v for k, v in summary.items() if k != 'estimate'},
                              estimate={k: round(v['drmse_model'], 4) for k, v in summary['estimate'].items()})), flush=True)
    except BaseException as exc:
        R.save(output / 'failure.json', dict(error=repr(exc), no_retry=True, partials_preserved=True, unix=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'parity', 'forecast', 'build'])
    p.add_argument('--root', type=Path, default=HERE.parent)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'check':
        spec, member, ns = check(root)
        print(json.dumps(dict(protocol=PROTOCOL, inventory_files=len(spec['files']), member=member,
                              stage_functions=sorted(k for k in ns if k.startswith('pru_') or k == 'dep'), fits=0), indent=1))
    elif a.action == 'parity':
        out = parity(root)
        (root / ANALYSIS).mkdir(parents=True, exist_ok=True)
        R.save(root / ANALYSIS / 'stage_parity_final_dryrun.json', out)
        print(json.dumps(out, indent=1))
    elif a.action == 'forecast':
        summary, table, _ = forecast(root)
        (root / ANALYSIS).mkdir(parents=True, exist_ok=True)
        table.write_csv(root / ANALYSIS / 'cells_v12077_e218.csv', float_precision=4)
        R.save(root / ANALYSIS / 'forecast.json', summary)
        print(json.dumps(summary, indent=1))
    else:
        build(root)


if __name__ == '__main__':
    main()
