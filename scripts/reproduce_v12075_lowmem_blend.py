"""Execute frozen blend.py without materializing features it discards.

No arithmetic/model change: select the original classifier features plus the
original mixture inputs, skip excluded encoding columns, and derive category
levels one column at a time instead of building an unused training matrix.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import resource
import runpy
import sys

import reproduce_best as R


def categories_only(frame, categorical, cols):
    return {c: sorted(frame[c].to_pandas().dropna().unique().tolist())
            for c in categorical if c in cols}


def require_unused_encodings(cols, drop_prefix, regime_rates):
    dropped = tuple(p for p in drop_prefix.split(',') if p)
    if regime_rates or not all(any(prefix.startswith(p) for p in dropped) for prefix in ('te_', 'ter_', 'tq_')):
        raise ValueError('all target encodings must be excluded and regime rates disabled')
    if any(c.startswith(('te_', 'ter_', 'tq_', 'rr_')) for c in cols):
        raise ValueError('model consumes an encoding; cannot use this low-memory path')


def compare_parts(actual_path, reference_path):
    import numpy as np
    with np.load(actual_path, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as reference:
        if set(actual.files) != set(reference.files):
            raise ValueError('premixture parts fields differ')
        for name in actual.files:
            a, b = actual[name], reference[name]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                raise ValueError('premixture parts differ: ' + name)


def run(work, base, version, weight):
    import lightgbm as lgb
    import polars as pl
    root = Path(__file__).resolve().parents[1]
    setup = R.read(root / R.ARCHIVE / 'setup.json')
    for name in ('blend.py', 'train.py', 'target_encoding.py', 'features.py'):
        if R.sha(work / 'src' / name) != setup['source_after']['src/' + name]:
            raise ValueError('frozen blend dependency changed: ' + name)
    os.chdir(work)
    sys.path.insert(0, str(work / 'src'))
    import train as T
    import target_encoding as E
    full_schema = pl.scan_parquet(T.FEATURES_PATH).collect_schema()
    cols = T.feature_columns(pl.DataFrame(schema=full_schema))
    require_unused_encodings(cols, os.environ.get('DROP_PREFIX', ''), os.environ.get('REGIME_RATES') == '1')
    extra = [R.ID, R.Y, 'split', 'ADEP_mvt', 'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_lobt', 'd_mvt_eobt', 'd_mvt_aobt']
    selected = [c for c in full_schema if c in set(cols + extra)]
    original_read, original_to_pandas, original_encodings, original_train = pl.read_parquet, T.to_pandas, E.add_encodings, lgb.train
    calls = {'features': 0, 'categories': 0, 'ranking_matrix': 0, 'skipped_encodings': 0}
    category_snapshot = {}

    def projected_read(source, *args, **kwargs):
        if Path(source).resolve() == Path(T.FEATURES_PATH).resolve():
            if args or kwargs or calls['features']:
                raise ValueError('unexpected feature-table read')
            calls['features'] += 1
            print('LOWMEM projected feature read', len(selected), 'of', len(full_schema), flush=True)
            return original_read(source, columns=selected)
        return original_read(source, *args, **kwargs)

    def unused_encodings(fit, pred, fit_labels_regular, fit_labels=None):
        calls['skipped_encodings'] += 1
        if calls['skipped_encodings'] != 1 or T.feature_columns(fit) != cols:
            raise ValueError('unexpected target encoding call/feature order')
        print('LOWMEM excluded encoding materialization skipped', flush=True)
        return fit, pred

    def narrow_pandas(frame, names, categories):
        if names != cols:
            raise ValueError('classifier feature order changed')
        if categories is None:
            if calls['categories']:
                raise ValueError('unexpected second training-category request')
            calls['categories'] += 1
            category_snapshot.update(categories_only(frame, T.CATEGORICAL, names))
            print('LOWMEM exact categories derived', len(category_snapshot), flush=True)
            return None, dict(category_snapshot)
        calls['ranking_matrix'] += 1
        return original_to_pandas(frame, names, categories)

    def no_fit(*_args, **_kwargs):
        raise ValueError('replay cannot train a model')

    pl.read_parquet, T.to_pandas, E.add_encodings, lgb.train = projected_read, narrow_pandas, unused_encodings, no_fit
    try:
        sys.argv = [str(work / 'src/blend.py'), str(base), str(version), str(weight)]
        scope = runpy.run_path(str(work / 'src/blend.py'), run_name='__main__')
        if calls != {'features': 1, 'categories': 1, 'ranking_matrix': 1, 'skipped_encodings': 1}:
            raise ValueError('unexpected low-memory call inventory')
        model_categories = [category_snapshot[c] for c in cols if c in category_snapshot]
        if scope['clf'].pandas_categorical != model_categories:
            raise ValueError('derived categories differ from the saved classifier')
        R.save(work / 'lowmem_blend.json', dict(calls=calls, selected_columns=selected,
            original_column_count=len(full_schema), model_feature_names=scope['clf'].feature_name(),
            categories_match_saved_classifier=True, fits_performed=0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            original_blend_sha256=R.sha(work / 'src/blend.py'), helper_sha256=R.sha(__file__)))
    finally:
        pl.read_parquet, T.to_pandas, E.add_encodings, lgb.train = original_read, original_to_pandas, original_encodings, original_train


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('base', type=int)
    p.add_argument('version', type=int)
    p.add_argument('weight', type=float)
    a = p.parse_args()
    run(a.work.resolve(), a.base, a.version, a.weight)
