#!/usr/bin/env python3
"""Registered descriptive audit of FIN1b OOF regime residual dependence.

Only `preflight` is permitted before the protocol/source/input receipt is committed.
`run --registration-commit SHA` verifies that immutable registration and all inputs.
No model fitting, model inference, ranking data, or submission is performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from types import SimpleNamespace

for _name, _value in {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'POLARS_MAX_THREADS': '2',
                     'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage'}.items():
    os.environ[_name] = _value

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT_REL = 'analysis/regime_dependence_20261012'
PROTOCOL = 'regime-dependence-FIN1b-conditioned-day-hour-v1-20261012'
EXPECTED_N = 2_085_047
FOLDS = [(i, i + 6) for i in range(1, 7)]
AIRPORTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LTFM', 'LSZH']
EXCLUDED = {'EDDF', 'EHAM', 'LSZH'}
P_EDGES = np.array([.01, .05, .20, .50, .80, .95, .99])
MIN_N, MIN_DAYS = 20, 4
PERMUTATIONS, BOOTSTRAPS, SEED = 199, 999, 20261012
STAT_NAMES = ['airport_day', 'airport_hour', 'same_day_different_hours']
SOURCE_REL = 'scripts/regime_dependence_audit_20261012.py'
TEST_REL = 'tests/test_regime_dependence_audit_20261012.py'
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'


def exact_ids(ids, context):
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.isfinite(ids).all() or np.any(np.abs(ids) >= 2**53) or np.any(ids != np.rint(ids)):
        raise ValueError(f'{context}: IDs must be finite exact integral values below 2**53')
    ids = ids.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'{context}: duplicate IDs')
    return ids


def validate_fold_names(names):
    expected = {f'fold_{a}_{b}.npz' for a, b in FOLDS}
    if len(names) != 6 or set(names) != expected:
        raise ValueError('exactly the six registered fold files are required')


def alignment_order(source_ids, target_ids):
    source_ids = exact_ids(source_ids, 'source')
    target_ids = exact_ids(target_ids, 'target')
    order = np.argsort(source_ids)
    if len(source_ids) != len(target_ids) or not np.array_equal(source_ids[order], np.sort(target_ids)):
        raise ValueError('source/target ID set differs')
    return order[np.searchsorted(source_ids[order], target_ids)]


def check_probabilities(p):
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all():
        raise ValueError('probabilities must be finite N by 6')
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1., atol=2e-6, rtol=0):
        raise ValueError('probabilities violate [0,1] simplex')


def dense_keys(*columns):
    return np.unique(np.column_stack(columns), axis=0, return_inverse=True)


def eligible_strata(strata, day, min_n=MIN_N, min_days=MIN_DAYS):
    n = np.bincount(strata)
    pairs = np.unique(np.column_stack([strata, day]), axis=0)
    ndays = np.bincount(pairs[:, 0], minlength=len(n))
    return ((n >= min_n) & (ndays >= min_days))[strata]


def pair_components(x, day, hour, nday):
    """Exact oriented distinct-pair products; no quadratic pair enumeration."""
    n = np.bincount(day, minlength=nday).astype(np.float64)
    sums = np.bincount(day, weights=x, minlength=nday)
    squares = np.bincount(day, weights=x*x, minlength=nday)
    hn = np.bincount(hour, minlength=nday*24).astype(np.float64)
    hs = np.bincount(hour, weights=x, minlength=nday*24)
    hq = np.bincount(hour, weights=x*x, minlength=nday*24)
    day_num = sums*sums-squares
    hour_num = (hs*hs-hq).reshape(nday, 24).sum(axis=1)
    day_pairs = n*(n-1)
    hour_pairs = (hn*(hn-1)).reshape(nday, 24).sum(axis=1)
    return {'numerator': np.column_stack([day_num, hour_num, day_num-hour_num]),
            'pairs': np.column_stack([day_pairs, hour_pairs, day_pairs-hour_pairs]),
            'n': n, 'q': squares, 'sum': sums}


def finite_null(x, strata, day, hour, nday):
    """Expected pair numerator under independent uniform within-stratum shuffles.

    x must have zero mean within each stratum. Distinct sampled residuals from s
    have expected product -sum(x_s**2)/(n_s*(n_s-1)), not zero.
    """
    n = np.bincount(strata)
    ss = np.bincount(strata, weights=x*x)
    means = np.bincount(strata, weights=x)/np.maximum(n, 1)
    if np.max(np.abs(means), initial=0.) > 1e-10:
        raise ValueError('finite-null input must be centred within strata')
    factor = np.divide(-ss, n*(n-1), out=np.zeros_like(ss), where=n > 1)
    def cluster_expected(cluster, size):
        keys, counts = np.unique(strata.astype(np.int64)*size+cluster, return_counts=True)
        sid, cid = keys//size, keys % size
        return np.bincount(cid, weights=counts*(counts-1)*factor[sid], minlength=size)
    d = cluster_expected(day, nday)
    h = cluster_expected(hour, nday*24).reshape(nday, 24).sum(axis=1)
    return np.column_stack([d, h, d-h])


def shuffle_strata(x, strata, rng, destination=None):
    """Uniform permutation of whole residual values, independently in each s."""
    if destination is None:
        destination = np.argsort(strata, kind='stable')
    source = np.lexsort((rng.random(len(x)), strata))
    shuffled = np.empty_like(x)
    shuffled[destination] = x[source]
    return shuffled


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    with path.open('x') as fh:
        json.dump(obj, fh, indent=2, allow_nan=False)
        fh.write('\n')


def resources(start):
    # Darwin ru_maxrss is bytes; Linux reports KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname != 'Darwin':
        rss *= 1024
    if rss > 6*1024**3:
        raise RuntimeError('6 GiB resident-memory audit limit exceeded')
    if time.monotonic()-start > 900:
        raise RuntimeError('900 s audit runtime limit exceeded')
    return {'elapsed_seconds': time.monotonic()-start, 'peak_rss_bytes': rss}


def input_paths(root):
    folds = sorted((root/'data/cv_preds_fin1b_hz').glob('fold_*.npz'))
    validate_fold_names([x.name for x in folds])
    cov = sorted((root/'data/adsblol_kt1').glob('offblock2_2025-*.parquet'))
    expected = [f'offblock2_{str(x)}.parquet' for x in np.arange(np.datetime64('2025-01-01'), np.datetime64('2025-12-31'))]
    if [x.name for x in cov] != expected:
        raise ValueError('coverage must contain exact 364 registered days; 31 December absent')
    return [root/'data/features_fin1b.parquet', root/'src/train.py', *folds, *cov]


def load_inputs(root):
    """Alignment/integrity only: no regime labels or residuals computed here."""
    cols = [ID, Y, 'ADEP_mvt', 'PHASE_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt', 'split',
            'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_lobt', 'd_mvt_eobt']
    df = pl.scan_parquet(root/'data/features_fin1b.parquet').filter(pl.col('split') == 'train').select(cols).collect()
    ids = exact_ids(df[ID].to_numpy(), 'FIN1b train')
    if df.height != EXPECTED_N or df['PHASE_mvt'].unique().to_list() != ['DEP']:
        raise ValueError('unexpected 2025 training DEP population')
    if df['MVT_TIME_UTC_mvt'].null_count() or df['MVT_TIME_UTC_mvt'].dt.year().unique().to_list() != [2025]:
        raise ValueError('2025 timestamp alignment failed')
    if df[Y].null_count() or not np.isfinite(df[Y].to_numpy()).all():
        raise ValueError('invalid training target')
    if sorted(df['ADEP_mvt'].unique().to_list()) != sorted(AIRPORTS):
        raise ValueError('airport population differs')
    paths = input_paths(root)
    fold_rows, oid, oy, op, omonth = [], [], [], [], []
    for a, b in FOLDS:
        path = root/f'data/cv_preds_fin1b_hz/fold_{a}_{b}.npz'
        with np.load(path, allow_pickle=False) as z:
            fold_ids = exact_ids(z['mvt_id'], path.name)
            p = z['proba'].astype(np.float64)
            check_probabilities(p)
            if len(p) != len(fold_ids) or len(z['y']) != len(fold_ids):
                raise ValueError('fold array lengths differ')
            oid.append(fold_ids); oy.append(z['y'].copy()); op.append(1-p[:, 0])
            omonth.append(np.full(len(p), a, np.int8))
            fold_rows.append({'file': str(path.relative_to(root)), 'rows': len(p)})
    oid = exact_ids(np.concatenate(oid), 'all six folds')
    order = alignment_order(oid, ids)
    y = np.concatenate(oy)[order]
    if not np.array_equal(y, df[Y].to_numpy()):
        raise ValueError('OOF y does not exactly match feature-table y by ID')
    month = df['MVT_TIME_UTC_mvt'].dt.month().to_numpy()
    if not np.array_equal((month-1) % 6+1, np.concatenate(omonth)[order]):
        raise ValueError('OOF fold/month assignment differs')
    coverage = pl.concat([pl.read_parquet(f, columns=[ID, 'coverage']) for f in paths if f.name.startswith('offblock2_')])
    cov_ids = exact_ids(coverage[ID].to_numpy(), 'all coverage')
    c = coverage['coverage'].to_numpy()
    if not np.isfinite(c).all() or not np.isin(c, [-1, 0, 1, 2, 3]).all():
        raise ValueError('invalid recorded coverage values')
    if not np.isin(cov_ids, ids).all():
        raise ValueError('coverage IDs outside exact 2025 population')
    coverage = coverage.with_columns(pl.col(ID).cast(pl.Int64), pl.col('coverage').cast(pl.Int8))
    df = df.with_columns(pl.col(ID).cast(pl.Int64)).join(coverage, on=ID, how='left', validate='1:1', maintain_order='left')
    if not np.array_equal(df[ID].to_numpy(), ids):
        raise ValueError('coverage join changed ID order')
    missing = df['coverage'].is_null().to_numpy()
    # Missing file/row remains separate (-1); observed coverage 0/1 share class 0.
    df = df.with_columns(pl.col('coverage').fill_null(-1))
    checks = {'rows': df.height, 'folds': fold_rows, 'coverage_rows': len(cov_ids),
              'coverage_join_missing_rows': int(missing.sum()), 'coverage_files': 364,
              'coverage_missing_calendar_day': '2025-12-31',
              'id_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
              'unique_exact_ids': True, 'six_month_pairs': True, 'exact_y_alignment': True,
              'probability_simplex_tolerance': 2e-6, 'year': 2025}
    return df, np.concatenate(op)[order], checks


def existing_regime_labels(df, train_path):
    """Execute ONLY the existing pure regime_label AST with frozen recipe globals."""
    tree = ast.parse(train_path.read_text())
    fn = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'regime_label']
    if len(fn) != 1:
        raise ValueError('exactly one existing regime_label function required')
    ns = {'np': np, 'pl': pl, 'TARGET': Y, 'TOL': 6, 'DAY': 86400.,
          'CLASSES': ['regular', 'sched', 'iobt', 'lobt', 'eobt', 'dayshift'],
          'CANDIDATES': dict(zip(['sched', 'iobt', 'lobt', 'eobt'],
                                ['d_mvt_sched', 'd_mvt_iobt', 'd_mvt_lobt', 'd_mvt_eobt'])),
          'PLANNED_REGIME_EXCLUDED': EXCLUDED,
          'os': SimpleNamespace(environ={'SCHED_TOL_LONG': '0'})}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(train_path), 'exec'), ns)
    labels = ns['regime_label'](df)
    if not np.isin(labels, np.arange(6)).all():
        raise ValueError('existing regime labels invalid')
    return labels


def prepare_population(df, p, labels):
    ap = np.array([AIRPORTS.index(x) for x in df['ADEP_mvt'].to_list()], dtype=np.int16)
    dt = df['MVT_TIME_UTC_mvt']
    date = (dt.dt.epoch('s').to_numpy()//86400).astype(np.int64)
    month = dt.dt.month().to_numpy().astype(np.int16)
    hour = dt.dt.hour().to_numpy().astype(np.int16)
    weekend = (dt.dt.weekday().to_numpy() >= 6).astype(np.int16)
    nm = df['FLIGHT_ID_mvt'].is_not_null().to_numpy().astype(np.int16)
    cov = df['coverage'].to_numpy().astype(np.int16)
    cov = np.where((cov == 0) | (cov == 1), 0, cov)
    pbin = np.searchsorted(P_EDGES, p, side='right')
    _, strata = dense_keys(ap, month, nm, cov, pbin, hour//4, weekend)
    eligible = eligible_strata(strata, date)
    primary = ~np.isin(ap, [AIRPORTS.index(x) for x in EXCLUDED])
    keep = eligible & primary
    support = {'all_rows': len(p), 'seven_airport_rows': int(primary.sum()),
               'primary_eligible_rows': int(keep.sum()), 'stratum_ineligible_rows_primary': int((primary & ~eligible).sum()),
               'excluded_airports_reason': 'planned classes structurally disabled in regime_label',
               'per_airport': {a: {'rows': int((ap == j).sum()), 'eligible_rows': int(((ap == j) & eligible).sum())}
                               for j, a in enumerate(AIRPORTS)}}
    if keep.sum() == 0:
        raise ValueError('no eligible primary rows')
    _, s = np.unique(strata[keep], return_inverse=True)
    z = (labels[keep] > 0).astype(np.float64)-p[keep]
    n = np.bincount(s)
    u = z-(np.bincount(s, weights=z)/n)[s]
    day_keys, day = dense_keys(ap[keep], date[keep], month[keep])
    hours = day*24+hour[keep]
    support['eligible_strata'] = len(n)
    support['primary_airport_days'] = len(day_keys)
    support['label_rule'] = 'exact src/train.py regime_label; tol6; sched_tol_long0; dayshift first'
    return u, s, day, hours, day_keys, support


def summarise(comp, null, select):
    n = float(comp['n'][select].sum()); q = float(comp['q'][select].sum())
    numerator = comp['numerator'][select].sum(axis=0)
    pairs = comp['pairs'][select].sum(axis=0)
    expected = null[select].sum(axis=0)
    def div(a, b):
        return float(a/b) if b > 0 else None
    values = {}
    for j, name in enumerate(STAT_NAMES):
        values[name] = {'ordered_pairs': int(pairs[j]), 'pair_covariance': div(numerator[j], pairs[j]),
                        'permutation_null_covariance': div(expected[j], pairs[j]),
                        'excess_covariance': div(numerator[j]-expected[j], pairs[j]),
                        'variance_normalised_excess': div((numerator[j]-expected[j])*n, pairs[j]*q)}
    values['n'] = int(n); values['airport_days'] = int(select.sum())
    values['residual_second_moment'] = div(q, n)
    values['mean_residual_cluster_iid_variance_ratio'] = div(q+numerator[0], q)
    values['same_ratio_finite_shuffle_null'] = div(q+expected[0], q)
    return values


def bootstrap_primary(comp, null, keys):
    """Resample complete UTC dates jointly across airports, separately per month.

    Inference is conditional on fitted OOF models and fixed stratum centring;
    centring is NOT re-estimated. No independent error/RMSE variance is implied.
    """
    dates, inverse = np.unique(keys[:, 1], return_inverse=True)
    date_month = np.array([keys[inverse == i, 2][0] for i in range(len(dates))])
    fields = np.column_stack([comp['numerator'], null, comp['pairs'], comp['q'], comp['n']])
    daily = np.zeros((len(dates), fields.shape[1]))
    np.add.at(daily, inverse, fields)
    rng = np.random.default_rng(SEED+1)
    samples = []
    for _ in range(BOOTSTRAPS):
        chosen = np.concatenate([rng.choice(np.flatnonzero(date_month == m), size=(date_month == m).sum(), replace=True)
                                 for m in range(1, 13) if np.any(date_month == m)])
        v = daily[chosen].sum(axis=0)
        samples.append(np.r_[(v[:3]-v[3:6])/v[6:9], (v[9]+v[0])/v[9]])
    q = np.quantile(np.asarray(samples), [.025, .975], axis=0)
    return {'replicates': BOOTSTRAPS, 'seed': SEED+1, 'unit': 'UTC date jointly across airports, within month',
            'conditional_on_fixed_stratum_means_and_OOF': True,
            'excess_covariance_ci95': {name: q[:, j].tolist() for j, name in enumerate(STAT_NAMES)},
            'mean_residual_cluster_iid_variance_ratio_ci95': q[:, 3].tolist()}


def analyse(df, p, root, start):
    labels = existing_regime_labels(df, root/'src/train.py')
    u, strata, day, hour, keys, support = prepare_population(df, p, labels)
    nday = len(keys)
    comp = pair_components(u, day, hour, nday)
    null = finite_null(u, strata, day, hour, nday)
    populations = {'primary_seven': np.ones(nday, bool), 'half_1': keys[:, 2] <= 6, 'half_2': keys[:, 2] >= 7}
    for j, a in enumerate(AIRPORTS):
        if a not in EXCLUDED:
            populations[f'airport_{a}'] = keys[:, 0] == j
            populations[f'leave_out_{a}'] = keys[:, 0] != j
    measured = {name: summarise(comp, null, take) for name, take in populations.items()}
    # Structural-zero airports are documented in support only.
    # Primary/halves/LOAO share the same immutable retained rows.
    rng = np.random.default_rng(SEED)
    destination = np.argsort(strata, kind='stable')
    observed = {name: np.array([measured[name][s]['excess_covariance'] for s in STAT_NAMES]) for name in populations}
    exceed = {name: np.zeros(3, int) for name in populations}
    perm_values = []
    for b in range(PERMUTATIONS):
        xp = shuffle_strata(u, strata, rng, destination)
        cp = pair_components(xp, day, hour, nday)
        for name, take in populations.items():
            sums = cp['numerator'][take].sum(axis=0)-null[take].sum(axis=0)
            stat = sums/comp['pairs'][take].sum(axis=0)
            exceed[name] += np.abs(stat) >= np.abs(observed[name])-1e-15
            if name == 'primary_seven':
                perm_values.append(stat.tolist())
        if (b+1) % 20 == 0:
            print(json.dumps({'phase': 'permutation', 'done': b+1, 'total': PERMUTATIONS, **resources(start)}), flush=True)
    for name in measured:
        for j, s in enumerate(STAT_NAMES):
            measured[name][s]['permutation_p_two_sided'] = float((exceed[name][j]+1)/(PERMUTATIONS+1))
    uncertainty = bootstrap_primary(comp, null, keys)
    primary = measured['primary_seven']['airport_day']
    nominal = (primary['excess_covariance'] > 0 and primary['permutation_p_two_sided'] <= .05
               and uncertainty['excess_covariance_ci95']['airport_day'][0] > 0)
    halves = all(measured[k]['airport_day']['excess_covariance'] > 0 for k in ['half_1', 'half_2'])
    loao = all(v['airport_day']['excess_covariance'] > 0 for k, v in measured.items() if k.startswith('leave_out_'))
    return {'protocol': PROTOCOL, 'support': support, 'statistics': measured, 'bootstrap': uncertainty,
            'permutations': {'count': PERMUTATIONS, 'seed': SEED, 'primary_excess_covariances': perm_values,
                             'null': 'uniform whole-residual permutation within immutable strata',
                             'secondary_pvalues_unadjusted_descriptive': True},
            'interpretation_gate': {'positive_primary_with_ci_and_nominal_p': bool(nominal),
                                    'positive_both_halves': bool(halves), 'positive_every_leave_airport_out': bool(loao),
                                    'distributed_temporal_association': bool(nominal and halves and loao)},
            'limitations': ['Previously explored, selection-informed 2025 OOF; not an independent confirmatory test.',
                            'Residual exchangeability within coarse strata is an assumption; unmeasured composition remains.',
                            'Common monthly model misspecification or temporal omitted features can explain association.',
                            'No physical/recording mechanism, causal effect, Bayes floor, RMSE standard error or score gain established.',
                            'Cluster/iid ratio concerns a mean binary regime residual under independent-cluster assumptions only.',
                            'Bootstrap is conditional on OOF fits/stratum means; serial dependence across dates is not modelled.',
                            'Secondary and subgroup p-values are descriptive and not multiplicity-corrected.']}


def verify_registration(root, commit, manifest):
    if not commit or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('run requires an explicit full registration commit SHA')
    for rel in [SOURCE_REL, TEST_REL, OUT_REL+'/PROTOCOL.md', OUT_REL+'/preflight.json']:
        saved = subprocess.check_output(['git', 'show', f'{commit}:{rel}'], cwd=root)
        if saved != (root/rel).read_bytes():
            raise ValueError(f'registered source/receipt differs: {rel}')
    log = subprocess.check_output(['git', 'show', f'{commit}:HYPOTHESES_LOG.md'], cwd=root).decode()
    if not any(line.startswith('| 463 |') and PROTOCOL in line for line in log.splitlines()):
        raise ValueError('registration journal row 463 lacks the exact protocol string')
    for item in manifest['inputs']:
        f = root/item['path']
        if f.stat().st_size != item['bytes'] or sha256(f) != item['sha256']:
            raise ValueError(f'registered input changed: {item["path"]}')


def render_report(result):
    primary = result['statistics']['primary_seven']
    lines = ['# Regime residual dependence audit', '', f'Protocol: `{PROTOCOL}`.', '',
             'Retrospective analysis on previously explored 2025 OOF; no model changes, fitting, inference or submissions.', '',
             '| Population | n | excess day covariance | normalised excess | day permutation p | mean-residual cluster/iid ratio |',
             '|---|---:|---:|---:|---:|---:|']
    for name, v in result['statistics'].items():
        d = v['airport_day']
        lines.append(f'| {name} | {v["n"]} | {d["excess_covariance"]:.8g} | {d["variance_normalised_excess"]:.6g} | '
                     f'{d["permutation_p_two_sided"]:.4f} | {v["mean_residual_cluster_iid_variance_ratio"]:.6g} |')
    lines += ['', '## Primary temporal scales', '',
              '| Pair type | observed covariance | exact finite-shuffle null | excess | date-bootstrap 95% interval |',
              '|---|---:|---:|---:|---|']
    for s in STAT_NAMES:
        v = primary[s]; ci = result['bootstrap']['excess_covariance_ci95'][s]
        lines.append(f'| {s} | {v["pair_covariance"]:.8g} | {v["permutation_null_covariance"]:.8g} | '
                     f'{v["excess_covariance"]:.8g} | [{ci[0]:.8g}, {ci[1]:.8g}] |')
    lines += ['', 'Gate: `'+json.dumps(result['interpretation_gate'])+'`.', '', '## Limits', '']
    lines += ['- '+x for x in result['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'run'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--registration-commit')
    args = parser.parse_args()
    start = time.monotonic()
    def expired(*_):
        raise TimeoutError('900 second hard wall-clock limit')
    signal.signal(signal.SIGALRM, expired); signal.alarm(900)
    os.nice(5)
    out = args.root/OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out/'preflight.json'
    if args.action == 'run':
        if (out/'result.json').exists() or (out/'RESULTS.md').exists():
            raise FileExistsError('immutable outcome output already exists')
        manifest = json.loads(receipt_path.read_text())
        verify_registration(args.root, args.registration_commit, manifest)
    df, p, checks = load_inputs(args.root)
    if args.action == 'preflight':
        paths = input_paths(args.root)
        manifest = {'protocol': PROTOCOL, 'action': 'preflight_only_no_residuals_or_dependence', 'checks': checks,
                    'inputs': [{'path': str(f.relative_to(args.root)), 'bytes': f.stat().st_size, 'sha256': sha256(f)} for f in paths],
                    'versions': {'numpy': np.__version__, 'polars': pl.__version__}, **resources(start)}
        save_json(receipt_path, manifest)
        print(json.dumps({'preflight': 'PASS', 'checks': checks, 'receipt': str(receipt_path), **resources(start)}))
        return
    if checks != manifest['checks']:
        raise ValueError('preflight checks changed')
    result = analyse(df, p, args.root, start)
    result.update({'registration_commit': args.registration_commit, 'checks': checks,
                   'input_receipt_sha256': sha256(receipt_path), 'versions': manifest['versions'], **resources(start)})
    save_json(out/'result.json', result)
    with (out/'RESULTS.md').open('x') as fh:
        fh.write(render_report(result))
    print(json.dumps({'result': str(out/'RESULTS.md'), 'gate': result['interpretation_gate'], **resources(start)}))


if __name__ == '__main__':
    main()
