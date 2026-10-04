"""Preregistered E187: fixed-grid all-SSE versus decided-body weight selection.

Only reads existing predictions. Run AFTER journal+code preregistration commit:
  .venv/bin/python src/e187_weight_objective.py --repo-root /path/prc-2026 --output /fresh/e187/run
Exit 0: valid passing gate; 1: valid rejected experiment; 2: invalid execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import zipfile

# Set before importing numerical libraries; this script never trains or imports train.py.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import numpy as np
import polars as pl

MEMBERS = ('raw', 'lobt', 'sched', 'eobt', 'aobt', 'mlpa')
BOX = ((0., .1), (.05, .2), (.05, .15), (.05, .15), (.1, .2), (.15, .35))
FOLDS = tuple(f'fold_{k}_{k+6}.npz' for k in range(1, 7))
CONTROL = 'data/cv_preds_fin1b_hz'
REFERENCE = 'data/cv_preds_e172_raw_lobt_sched_eobt_aobt_mlpa_f1b'
FEATURES = 'data/features_fin1b.parquet'
FIXED_WEIGHTS = np.array([.033, .10, .067, .083, .10, .30])
DAY = 86400.
LOBT_WINDOW = 3606.
REFERENCE_TOLERANCE = 1e-6
LB_DENOMINATOR = 6 * 2 * 252.43 * 344841


class InvalidInput(ValueError):
    """Inputs/protocol do not support a valid experiment."""


def validate_real_array(values, where, allow_nan=False):
    """Reject malformed dtypes before calling numpy ufuncs or comparisons."""
    if not isinstance(values, np.ndarray) or values.dtype.kind not in 'iuf':
        raise InvalidInput(f'{where}: expected a real numeric array')
    valid = not np.isinf(values).any() if allow_nan else np.isfinite(values).all()
    if not valid:
        raise InvalidInput(f'{where}: nonfinite values')


def fixed_grid():
    g = np.round(np.arange(0., 1.01, .05), 2)
    axes = [g[(g >= lo-1e-9) & (g <= hi+1e-9)] for lo, hi in BOX]
    return np.array([(0.,)*6] + [w for w in itertools.product(*axes) if sum(w) <= 1.+1e-9], dtype=np.float64)


def clip_prediction(expected, lobt, window=LOBT_WINDOW):
    return np.maximum(np.where(np.isnan(lobt), expected, np.clip(expected, lobt-window, lobt+window)), 0.)


def mix(proba, values, reg, lobt, window=LOBT_WINDOW):
    values = values.astype(np.float64).copy()
    values[:, 0], values[:, -1] = reg, reg + DAY
    return clip_prediction(np.sum(proba.astype(np.float64) * np.nan_to_num(values), axis=1), lobt, window)


def build_prediction(z, members, lobt, weights, window=LOBT_WINDOW):
    weights = np.asarray(weights, dtype=np.float64)
    reg = (1.-weights.sum())*z['reg'] + members @ weights
    out = dict(z)
    out['pred'] = z['pred'] + mix(z['proba'], z['values'], reg, lobt, window) - mix(z['proba'], z['values'], z['reg'], lobt, window)
    out['reg'] = reg
    out['values'] = z['values'].astype(np.float64).copy()
    out['values'][:, 0], out['values'][:, -1] = reg, reg + DAY
    if not np.isfinite(out['pred']).all():
        raise InvalidInput('nonfinite rebuilt prediction')
    return out


def score_grid(z, members, lobt, body_mask, grid, row_chunk=8192, grid_chunk=64, window=LOBT_WINDOW):
    """Evaluate the exact piecewise clipped mixture, bounded row×grid memory.

    The unclipped mixture is affine in reg; all clipping remains per-row/per-weight.
    Reordering the float64 sum differs from scalar E172 only at rounding precision.
    Saved-reference maxabs parity is mandatory before reporting experimental metrics.
    """
    if row_chunk < 1 or grid_chunk < 1:
        raise InvalidInput('chunk sizes must be positive')
    all_sse = np.zeros(len(grid), dtype=np.float64)
    body_sse = np.zeros(len(grid), dtype=np.float64)
    for start in range(0, len(z['reg']), row_chunk):
        sl = slice(start, start+row_chunk)
        reg = z['reg'][sl].astype(np.float64)
        p = z['proba'][sl].astype(np.float64)
        v = z['values'][sl].astype(np.float64).copy()
        v[:, 0], v[:, -1] = reg, reg+DAY
        baseline_raw = np.sum(p*np.nan_to_num(v), axis=1)
        baseline = clip_prediction(baseline_raw, lobt[sl], window)
        slope = p[:, 0]+p[:, -1]
        # Preserve E172's stored prediction delta in the same addition/subtraction order.
        for j in range(0, len(grid), grid_chunk):
            w = grid[j:j+grid_chunk]
            newreg = reg[:, None]*(1.-w.sum(axis=1))[None, :] + members[sl] @ w.T
            expected = baseline_raw[:, None]+slope[:, None]*(newreg-reg[:, None])
            new = z['pred'][sl, None]+clip_prediction(expected, lobt[sl, None], window)-baseline[:, None]
            errors = (new-z['y'][sl, None])**2
            if not np.isfinite(errors).all():
                raise InvalidInput('nonfinite grid errors')
            all_sse[j:j+len(w)] += errors.sum(axis=0, dtype=np.float64)
            body_sse[j:j+len(w)] += errors[body_mask[sl]].sum(axis=0, dtype=np.float64)
    return all_sse, body_sse


def select_indices(scores):
    validate_real_array(scores, 'selection scores')
    if scores.ndim != 2 or scores.shape[0] != 6 or not scores.shape[1]:
        raise InvalidInput('selection requires six finite fold scores')
    # Explicit other-fold summation avoids held-out-label influence through cancellation.
    return np.array([np.argmin(np.sum(scores[np.arange(6) != k], axis=0, dtype=np.float64)) for k in range(6)])


def integer_ids(ids, where):
    validate_real_array(ids, where+' IDs')
    if ids.ndim != 1:
        raise InvalidInput(f'{where}: IDs must be finite numeric vector')
    if np.any(ids != np.floor(ids)) or np.any(np.abs(ids) >= 2**53):
        raise InvalidInput(f'{where}: IDs must be exactly representable integers')
    ids = ids.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise InvalidInput(f'{where}: duplicate IDs')
    return ids


def read_npz(path):
    try:
        loaded = np.load(path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise InvalidInput(f'{path}: expected an NPZ archive')
        with loaded as f:
            if len(set(f.files)) != len(f.files):
                raise InvalidInput(f'{path}: duplicate NPZ array names')
            return {k: f[k] for k in f.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InvalidInput(f'{path}: malformed or unreadable npz') from exc


def load_features(path):
    columns = ['MVT_ID_mvt', 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'd_mvt_sched', 'd_mvt_lobt', 'TAXITIME_SEC_mvt']
    try:
        df = pl.read_parquet(path, columns=columns).filter(pl.col('split') == 'train')
    except Exception as exc:
        raise InvalidInput(f'{path}: missing or malformed feature metadata') from exc
    integer_ids(df['MVT_ID_mvt'].to_numpy(), str(path))
    df = df.with_columns(pl.col('MVT_ID_mvt').cast(pl.Int64))
    y = df['TAXITIME_SEC_mvt'].to_numpy()
    month = df['month'].to_numpy()
    validate_real_array(y, 'feature targets')
    validate_real_array(month, 'feature months')
    validate_real_array(df['d_mvt_sched'].to_numpy(), 'feature schedule delay', allow_nan=True)
    validate_real_array(df['d_mvt_lobt'].to_numpy(), 'feature LOBT delay', allow_nan=True)
    if not len(df) or not np.array_equal(np.unique(month), np.arange(1, 13)):
        raise InvalidInput('feature targets/months invalid')
    return df


def feature_masks(meta):
    tail_population = (meta['airport'] == 'LIRF') & meta['nm_missing']
    delay = meta['delay']
    protected = tail_population & (delay > 14400.) & (delay <= 108000.)
    return dict(decided=~(tail_population & (delay >= 14400.)),
                strict=~(tail_population & (delay > 0.)), protected=protected)


def align(z, ids, where):
    if 'mvt_id' not in z:
        raise InvalidInput(f'{where}: no IDs')
    source = integer_ids(z['mvt_id'], where)
    if not np.array_equal(np.sort(source), np.sort(ids)):
        raise InvalidInput(f'{where}: mismatched ID sets')
    order = np.argsort(source)
    idx = order[np.searchsorted(source[order], ids)]
    result = {}
    for key, value in z.items():
        if value.ndim and value.shape[0] == len(source):
            result[key] = value[idx]
        else:
            result[key] = value
    return result


def validate_prediction(z, ids, meta, where):
    n = len(ids)
    for key in ('pred', 'reg', 'y'):
        if key not in z or z[key].shape != (n,):
            raise InvalidInput(f'{where}: invalid {key}')
        validate_real_array(z[key], f'{where}: {key}')
    for key in ('values', 'proba'):
        if key not in z or z[key].shape != (n, 6):
            raise InvalidInput(f'{where}: invalid {key} shape')
        validate_real_array(z[key], f'{where}: {key}', allow_nan=key == 'values')
    if np.any(z['proba'] < 0) or not np.allclose(z['proba'].sum(axis=1), 1., atol=1e-5, rtol=0):
        raise InvalidInput(f'{where}: probabilities invalid')
    if not np.array_equal(z['y'], meta['y']):
        raise InvalidInput(f'{where}: feature target mismatch')
    if 'month' in z:
        validate_real_array(z['month'], f'{where}: month')
        if not np.array_equal(z['month'], meta['month']):
            raise InvalidInput(f'{where}: month mismatch')


def load_fold(root, name, features):
    z = read_npz(root/CONTROL/name)
    if 'mvt_id' not in z:
        raise InvalidInput(f'{name}: missing IDs')
    ids = integer_ids(z['mvt_id'], name)
    df = pl.DataFrame({'MVT_ID_mvt': ids}).join(features, on='MVT_ID_mvt', how='left', maintain_order='left')
    meta = dict(y=df['TAXITIME_SEC_mvt'].to_numpy(), month=df['month'].to_numpy(),
                airport=df['ADEP_mvt'].to_numpy(), nm_missing=df['FLIGHT_ID_mvt'].is_null().to_numpy(),
                delay=df['d_mvt_sched'].to_numpy(), lobt=df['d_mvt_lobt'].to_numpy())
    a, b = map(int, name[5:-4].split('_'))
    if not np.isin(meta['month'], [a, b]).all():
        raise InvalidInput(f'{name}: missing metadata or incorrect fold month')
    validate_prediction(z, ids, meta, name)
    cols = []
    for member in MEMBERS:
        where = root/f'data/cv_e130{member}'/name
        m = align(read_npz(where), ids, str(where))
        if 'reg2' not in m or m['reg2'].shape != (len(ids),):
            raise InvalidInput(f'{where}: invalid reg2')
        validate_real_array(m['reg2'], f'{where}: reg2')
        for key, expected in [('y', meta['y']), ('month', meta['month'])]:
            if key in m:
                validate_real_array(m[key], f'{where}: {key}')
                if not np.array_equal(m[key], expected):
                    raise InvalidInput(f'{where}: mismatched {key}')
        cols.append(m['reg2'])
    ref = align(read_npz(root/REFERENCE/name), ids, f'reference {name}')
    validate_prediction(ref, ids, meta, f'reference {name}')
    return z, np.column_stack(cols), meta


def validate_inputs(root, features):
    for directory in [CONTROL, REFERENCE] + [f'data/cv_e130{m}' for m in MEMBERS]:
        names = {p.name for p in (root/directory).glob('fold_*.npz')}
        if names != set(FOLDS):
            raise InvalidInput(f'{directory}: expected exactly six folds; missing={set(FOLDS)-names}, extras={names-set(FOLDS)}')
    ids = []
    for name in FOLDS:
        z, _, _ = load_fold(root, name, features)
        ids.append(z['mvt_id'])
    observed = integer_ids(np.concatenate(ids), 'all folds')
    expected = integer_ids(features['MVT_ID_mvt'].to_numpy(), 'train features')
    if not np.array_equal(np.sort(observed), np.sort(expected)):
        raise InvalidInput('sixfold coverage differs from training features')
    return FOLDS


def compare_fold(control, candidate, y, meta):
    delta = (candidate-y)**2-(control-y)**2
    masks = feature_masks(meta)
    tail = y > 7200.
    selectors = dict(all=np.ones(len(y), bool), decided=masks['decided'], strict=masks['strict'],
                     body=y <= 7200., decided_body=masks['decided'] & (y <= 7200.), tail=tail,
                     protected=masks['protected'], nonprotected=~masks['protected'],
                     protected_tail=tail & masks['protected'], nonprotected_tail=tail & ~masks['protected'])
    for month in np.unique(meta['month']):
        selectors[f'month_{int(month)}'] = meta['month'] == month
    for airport in np.unique(meta['airport']):
        selectors[f'airport_{airport}'] = meta['airport'] == airport
    selectors['nm_missing'] = meta['nm_missing']
    selectors['nm_present'] = ~meta['nm_missing']
    sums = {k: float(delta[m].sum(dtype=np.float64)) for k, m in selectors.items()}
    s = np.sort(delta)
    # Identical convention to fold_metrics, including tiny synthetic fixtures.
    sums['trimmed'] = float(delta.sum()-s[:3].sum()-s[-3:].sum())
    return dict(sse=sums, seconds={k: v/LB_DENOMINATOR for k, v in sums.items()},
                counts={k: int(m.sum()) for k, m in selectors.items()})


def aggregate(folds):
    sums = {}
    counts = {}
    for fold in folds:
        for key, val in fold['sse'].items():
            sums[key] = sums.get(key, 0.)+val
        for key, val in fold['counts'].items():
            counts[key] = counts.get(key, 0)+val
    lower = sum(fold['sse']['decided'] < 0 for fold in folds)
    flags = dict(six_complete=len(folds) == 6, decided=sums['decided'] < 0,
                 four_of_six=lower >= 4, trimmed=sums['trimmed'] < 0, body=sums['body'] < 0)
    return dict(sse=sums, seconds={k: v/LB_DENOMINATOR for k, v in sums.items()}, counts=counts,
                lower_decided=lower, gate_flags=flags, gate_pass=all(flags.values()))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def train_constants(path):
    """Read literal numerical constants without importing heavy training libraries."""
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('DAY', 'LOBT_WINDOW'):
                    values[target.id] = float(ast.literal_eval(node.value))
    if set(values) != {'DAY', 'LOBT_WINDOW'} or values['DAY'] != DAY:
        raise InvalidInput('train.py mixture constants missing or DAY changed')
    if values['LOBT_WINDOW'] <= 0:
        raise InvalidInput('invalid LOBT_WINDOW')
    return values


def write_npz_exclusive(path, values):
    with path.open('xb') as f:
        np.savez(f, **values)


def run(repo_root, output, row_chunk=8192, grid_chunk=64, progress=True):
    root, output = Path(repo_root).resolve(), Path(output).resolve()
    if row_chunk < 1 or grid_chunk < 1:
        raise InvalidInput('chunk sizes must be positive')
    # Never place new artifacts inside an immutable source directory.
    for source in [CONTROL, REFERENCE] + [f'data/cv_e130{m}' for m in MEMBERS]:
        if output.is_relative_to(root/source):
            raise InvalidInput('output lies inside immutable source directory')
    output.mkdir(parents=True, exist_ok=False)
    lock = output/'.lock'
    with lock.open('x') as f:
        f.write(f'pid={os.getpid()}\n')
    try:
        constants_path = root/'src/train.py'
        constants = train_constants(constants_path)
        window = constants['LOBT_WINDOW']
        features_path = root/FEATURES
        features = load_features(features_path)
        validate_inputs(root, features)
        sources = [features_path, constants_path, Path(__file__).resolve(), root/'src/e172_members_cv.py', root/'src/fold_metrics.py']
        for directory in [CONTROL, REFERENCE]+[f'data/cv_e130{m}' for m in MEMBERS]:
            sources.extend(root/directory/name for name in FOLDS)
        source_hashes = {str(p): sha256(p) for p in sources}
        grid = fixed_grid()
        scores_all, scores_body = [], []
        started = time.monotonic()
        for i, name in enumerate(FOLDS):
            z, members, meta = load_fold(root, name, features)
            body = feature_masks(meta)['decided'] & (z['y'] <= 7200.)
            all_sse, body_sse = score_grid(z, members, meta['lobt'], body, grid, row_chunk, grid_chunk, window)
            scores_all.append(all_sse); scores_body.append(body_sse)
            if progress:
                print(f'scored {i+1}/6 {name}: rows={len(z["y"])} grid={len(grid)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        indices_all = select_indices(np.array(scores_all))
        indices_body = select_indices(np.array(scores_body))
        for directory in ('all_selected', 'body_selected', 'fixed_production_pre_tail'):
            (output/directory).mkdir()
        fold_results, primary, fixed_vs_all, fixed_vs_base, parity = [], [], [], [], []
        for i, name in enumerate(FOLDS):
            z, members, meta = load_fold(root, name, features)
            all_selected = build_prediction(z, members, meta['lobt'], grid[indices_all[i]], window)
            body_selected = build_prediction(z, members, meta['lobt'], grid[indices_body[i]], window)
            fixed = build_prediction(z, members, meta['lobt'], FIXED_WEIGHTS, window)
            ref = align(read_npz(root/REFERENCE/name), integer_ids(z['mvt_id'], name), name)
            maxabs = float(np.max(np.abs(all_selected['pred']-ref['pred'])))
            reg_maxabs = float(np.max(np.abs(all_selected['reg']-ref['reg'])))
            # Recover saved weights as evidence independent of predictions; allow only numerically
            # indistinguishable rank-deficient ambiguity, which is recorded explicitly.
            design = members-z['reg'][:, None]
            inferred, _, rank, _ = np.linalg.lstsq(design, ref['reg']-z['reg'], rcond=None)
            weight_maxabs = float(np.max(np.abs(inferred-grid[indices_all[i]])))
            weights_match = bool(rank < 6 or weight_maxabs <= REFERENCE_TOLERANCE)
            if maxabs > REFERENCE_TOLERANCE or reg_maxabs > REFERENCE_TOLERANCE or not weights_match:
                raise InvalidInput(f'{name}: saved-reference parity failed: pred={maxabs:.9g}, reg={reg_maxabs:.9g}, weight={weight_maxabs:.9g}')
            parity.append(dict(fold=name, maxabs_pred_seconds=maxabs, maxabs_reg_seconds=reg_maxabs,
                               inferred_reference_weights=inferred.tolist(), design_rank=int(rank),
                               selected_weights_match=weights_match,
                               ambiguity='rank-deficient reference; exact reg/pred parity required' if rank < 6 else None))
            fold_results.append(dict(fold=name, all_index=int(indices_all[i]), body_index=int(indices_body[i]),
                                     all_weights=grid[indices_all[i]].tolist(), body_weights=grid[indices_body[i]].tolist()))
            for directory, values in [('all_selected', all_selected), ('body_selected', body_selected), ('fixed_production_pre_tail', fixed)]:
                values = dict(values, month=meta['month'], complete=np.array(True), objective=np.array(directory))
                write_npz_exclusive(output/directory/name, values)
            if progress:
                print(f'built {i+1}/6 {name}: all={grid[indices_all[i]].tolist()} body={grid[indices_body[i]].tolist()}', flush=True)
        # All six saved-reference checks must pass before calculating experimental metrics.
        for i, name in enumerate(FOLDS):
            z, _, meta = load_fold(root, name, features)
            all_selected = read_npz(output/'all_selected'/name)
            body_selected = read_npz(output/'body_selected'/name)
            fixed = read_npz(output/'fixed_production_pre_tail'/name)
            for directory, values in [('all_selected', all_selected), ('body_selected', body_selected), ('fixed_production_pre_tail', fixed)]:
                validate_prediction(values, integer_ids(z['mvt_id'], name), meta, directory+'/'+name)
                if not np.array_equal(values['mvt_id'], z['mvt_id']):
                    raise InvalidInput(f'{directory}/{name}: saved output IDs not aligned')
                if not np.array_equal(values['values'][:, 0], values['reg']) or not np.array_equal(values['values'][:, -1], values['reg']+DAY):
                    raise InvalidInput(f'{directory}/{name}: unsynchronized candidate values')
            p = compare_fold(all_selected['pred'], body_selected['pred'], z['y'], meta)
            f = compare_fold(all_selected['pred'], fixed['pred'], z['y'], meta)
            fb = compare_fold(z['pred'], fixed['pred'], z['y'], meta)
            primary.append(p); fixed_vs_all.append(f); fixed_vs_base.append(fb)
            fold_results[i].update(primary=p, fixed_vs_all=f, fixed_vs_fin1b=fb)
        # Detect concurrent source changes before any completed manifest is published.
        if any(sha256(Path(p)) != digest for p, digest in source_hashes.items()):
            raise InvalidInput('source changed during execution')
        result = dict(complete=True, experiment='E187', fold_results=fold_results, primary=aggregate(primary),
                      secondary=dict(label='fixed production weights on same FIN1b/member OOF; pre-tail, not v1139; CatBoost/seeds/stages absent',
                                     weights=FIXED_WEIGHTS.tolist(), vs_all_selected=aggregate(fixed_vs_all), vs_fin1b=aggregate(fixed_vs_base)),
                      saved_reference_parity=parity, provenance=dict(repo_root=str(root), output=str(output), source_sha256=source_hashes),
                      config=dict(members=list(MEMBERS), box=BOX, step=.05, total_weight_max=1., grid_size=len(grid),
                                  fold_names=list(FOLDS), selection='other five folds only; first original itertools index wins ties',
                                  objectives=['all SSE', 'decided AND y<=7200 SSE'],
                                  decided_mask='NOT (ADEP=LIRF AND FLIGHT_ID IS NULL AND D>=14400)',
                                  exact_protected_mask='ADEP=LIRF AND FLIGHT_ID IS NULL AND 14400<D<=108000',
                                  source_values_nan='nonregular NaNs use E172 np.nan_to_num; regular/day overwritten',
                                  constants=constants, reference_maxabs_tolerance_seconds=REFERENCE_TOLERANCE,
                                  lb_denominator=LB_DENOMINATOR, row_chunk=row_chunk, grid_chunk=grid_chunk,
                                  threads=2, sums='float64', runtime_seconds=time.monotonic()-started))
        write_npz_exclusive(output/'selection_scores.npz', dict(grid=grid, all_sse=np.array(scores_all), body_sse=np.array(scores_body),
                            selected_all=indices_all, selected_body=indices_body))
        verdict = 'PASS' if result['primary']['gate_pass'] else 'REJECT'
        with (output/'report.md').open('x') as f:
            f.write(f'E187 {verdict}\n\nBody-selected minus all-selected: '+json.dumps(result['primary']['seconds'], sort_keys=True)+'\n\n'+
                    f'Decided lower folds: {result["primary"]["lower_decided"]}/6. Gate: {result["primary"]["gate_flags"]}.\n\n'+
                    'Fixed production comparison is pre-tail OOF, not v1139. All selected outputs passed saved-reference parity.\n')
        result['provenance']['output_sha256'] = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file() and p != lock}
        # Manifest is the last, atomic completion marker; partial runs retain no marker.
        text = json.dumps(result, indent=2, allow_nan=False)+'\n'
        pending = output/'.manifest.pending'
        with pending.open('x') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.link(pending, output/'manifest.json')
        pending.unlink()
        return result
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--row-chunk', type=int, default=8192)
    parser.add_argument('--grid-chunk', type=int, default=64)
    args = parser.parse_args(argv)
    try:
        result = run(args.repo_root, args.output, args.row_chunk, args.grid_chunk)
    except (InvalidInput, OSError, ValueError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(dict(primary=result['primary'], output=str(args.output)), indent=2))
    return 0 if result['primary']['gate_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
