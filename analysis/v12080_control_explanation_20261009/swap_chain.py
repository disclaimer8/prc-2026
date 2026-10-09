"""v12080 control explanation (journal 446): one-component-at-a-time swaps through the frozen final chain, zero fits.

Each variant rebuilds the premix of the final workspace with ONE group of Jan/Jul inputs taken from the v12077/v12078-era
chain (old members / old classifier / old candidates / old motion member / old 2026 daily ADS-B products), then replays
stages 1-11 of `scripts/final_tail_assembly_20261008.py` (same frozen sources, same cached stage models, 0 fits) in a
private scratch workspace. Feb/Jun inputs are never changed; the 62 served E156 rows are injected with the final stage-3
values (the tail lottery is analysed separately). Nothing under data/final_tail_20261009/work* or models/ is written:
the scratch workspace holds per-file symlinks to the resolved inputs and its own copies of src/ and sealed.json.

    PYTHONDONTWRITEBYTECODE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage OMP_NUM_THREADS=4 \
      .venv/bin/python analysis/v12080_control_explanation_20261009/swap_chain.py --runs <scratch dir> [variants...]
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OMP_THREAD_LIMIT'] = '4'
import numpy as np
import polars as pl

ROOT = Path('/Users/denyskolomiiets/prc-2026')
WORK = ROOT / 'data/final_tail_20261009/work'
OLD = ROOT / 'data/v12078_noe81_20261009'
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
spec = importlib.util.spec_from_file_location('fta', ROOT / 'scripts/final_tail_assembly_20261008.py')
fta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fta)
fta.environment = (lambda f: (lambda threads=4: f(4)))(fta.environment)  # <= 4 threads in every child stage
B = fta.BASE_VERSION
ANCH = ['raw', 'lobt', 'sched', 'eobt', 'aobt']
TAB = ['cat', 'lgb', *ANCH]


def table(path, key='reg'):
    z = np.load(path)
    return z['mvt_id'], z[key]


def align(ids, mid, v, missing=None):
    o = np.argsort(mid)
    pos = np.clip(np.searchsorted(mid[o], ids), 0, len(mid) - 1)
    hit = mid[o][pos] == ids
    if missing is None and not hit.all():
        raise ValueError('ids missing')
    out = np.where(hit, v[o][pos], np.nan if missing is None else missing)
    return out, hit


def raw_proba(features, clf_path, state_path, src):
    import lightgbm as lgb
    sys.path.insert(0, str(src))
    import features as F
    import train as T
    clf = lgb.Booster(model_file=str(clf_path))
    state = json.loads(Path(state_path).read_text()) if state_path else None
    schema = pl.scan_parquet(features).collect_schema()
    model_cols = [c for c in schema if c.replace(' ', '_') in set(clf.feature_name())]
    extra = [ID, 'split', *T.CANDIDATES.values()]
    frame = pl.read_parquet(features, columns=[c for c in schema if c in set(model_cols + extra)]).filter(pl.col('split') == 'rank')
    xr, _, _ = fta.classifier_matrix(frame, clf, F.CATEGORICAL, state)
    p = clf.predict(xr, num_threads=4)
    cands = np.column_stack([frame[c].to_numpy().astype(np.float64) for c in T.CANDIDATES.values()])
    return frame[ID].to_numpy().astype(np.float64), p, cands, frame['d_mvt_lobt'].to_numpy().astype(np.float64)


def load_inputs(cache):
    if Path(cache).exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    src = WORK / 'src'
    fid, fp, fc, fd = raw_proba(WORK / 'data/features_fin1b.parquet', WORK / 'submissions/classifier.txt', WORK / 'submissions/classifier_state.json', src)
    oid, op, oc, od = raw_proba(ROOT / 'data/features_fin1b.parquet', ROOT / 'submissions/clf_v870.txt', None, src)
    out = dict(ids=fid, p_new=fp, cand_new=fc, d_new=fd)
    jj = np.isin(fid, oid)
    out['jj'] = jj
    if jj.sum() != len(oid):
        raise ValueError('old ranking rows are not a subset of the final rows')
    idx = np.argsort(oid)[np.searchsorted(np.sort(oid), fid[jj])]
    out['p_old'] = fp.copy(); out['p_old'][jj] = op[idx]
    out['cand_old'] = fc.copy(); out['cand_old'][jj] = oc[idx]
    out['d_old'] = fd.copy(); out['d_old'][jj] = od[idx]
    new = dict(cat=WORK / 'submissions/cat_rank.npz', lgb=WORK / 'submissions/lgb_rank.npz', nn=WORK / 'submissions/nn_rank.npz',
               motion=WORK / 'data/e185_motion_rank.npz', **{k: WORK / f'submissions/e130_{k}f_rank.npz' for k in ANCH})
    old = dict(cat=OLD / 'submissions/cat_fin1b_seeds15.npz', lgb=OLD / 'submissions/lgb_reg_seeds_v870.npz',
               nn=OLD / 'data/e203_rank_seeds3to9_20261008/models/prediction_10seed.npz', motion=OLD / 'data/e185_motion_rank.npz',
               **{k: OLD / f'submissions/e130_{k}f_rank.npz' for k in ANCH})
    for k in new:
        out[k + '_new'] = align(fid, *table(new[k]))[0]
        o, hit = align(fid, *table(old[k]), missing=0.0)
        if not (hit == jj).all():
            raise ValueError('old member does not cover exactly the Jan/Jul rows: ' + k)
        out[k + '_old'] = np.where(jj, o, out[k + '_new'])
    np.savez(cache, **out)
    return out


def cross_inputs(inp, cache):
    """Classifier model vs view: old clf_v870 on the final FIN1b view, final classifier on the old view (Jan/Jul rows)."""
    if Path(cache).exists():
        z = np.load(cache)
        inp.update({k: z[k] for k in z.files})
        return inp
    fid, jj, src = inp['ids'], inp['jj'], WORK / 'src'
    i1, p1, _, _ = raw_proba(WORK / 'data/features_fin1b.parquet', ROOT / 'submissions/clf_v870.txt', None, src)
    i2, p2, _, _ = raw_proba(ROOT / 'data/features_fin1b.parquet', WORK / 'submissions/classifier.txt', WORK / 'submissions/classifier_state.json', src)
    if not np.array_equal(i1, fid):
        raise ValueError('row order')
    out = dict(p_model=np.where(jj[:, None], p1, inp['p_new']))
    idx = np.argsort(i2)[np.searchsorted(np.sort(i2), fid[jj])]
    out['p_view'] = inp['p_new'].copy(); out['p_view'][jj] = p2[idx]
    np.savez(cache, **out)
    inp.update(out)
    return inp


def premix_for(inp, swap):
    """swap: set of groups taken from the old chain on Jan/Jul rows: tab, nn, proba, cand;
    proba_model = old classifier on the final view, proba_view = final classifier on the old view."""
    jj = inp['jj']
    pick = lambda k, grp: np.where(jj, inp[k + '_old'], inp[k + '_new']) if grp in swap else inp[k + '_new']
    w = dict(zip(ANCH, [w for _, w in fta.WEIGHTS['final'][0]]))
    members = [(w[k], pick(k, 'tab')) for k in ANCH] + [(fta.WEIGHTS['final'][1], pick('nn', 'nn'))]
    proba = np.where(jj[:, None], inp['p_old'], inp['p_new']) if 'proba' in swap else inp['p_new']
    if 'proba_model' in swap:
        proba = inp['p_model']
    if 'proba_view' in swap:
        proba = inp['p_view']
    cand = np.where(jj[:, None], inp['cand_old'], inp['cand_new']) if 'cand' in swap else inp['cand_new']
    return fta.premix_arrays(proba, pick('cat', 'tab'), pick('lgb', 'tab'), members, list(cand.T), inp['d_new'], fta.CAT_W)


def mirror(src_dir, dst_dir, override=None):
    """Per-file symlinks to resolved targets (never into WORK)."""
    for p in sorted(Path(src_dir).rglob('*')):
        rel = p.relative_to(src_dir)
        dst = Path(dst_dir) / rel
        if p.is_dir() and not p.is_symlink():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        target = (override or {}).get(str(rel), p.resolve())
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(target)


def old_products():
    """Jan/Jul 2026 daily products of the v12075-era chain (old MVT) for the seven families the stages read."""
    o = {}
    for folder, product in fta.PRODUCTS:
        for p in sorted((ROOT / 'data' / folder).glob(f'{product}_2026-0[17]-*.parquet')):
            o[f'{folder}/{p.name}'] = p.resolve()
    return o


def run_chain(runs, name, parts, inp, motion_old=False, products_old=False):
    S = Path(runs) / name
    if S.exists():
        shutil.rmtree(S)
    for d in ('src', 'submissions', 'logs', 'jobs', 'receipts', 'e156/run_0'):
        (S / d).mkdir(parents=True)
    for p in (WORK / 'src').glob('*.py'):
        shutil.copyfile(p, S / 'src' / p.name)
    shutil.copyfile(WORK / 'sealed.json', S / 'sealed.json')
    shutil.copyfile(WORK / 'e156/run_0/served.json', S / 'e156/run_0/served.json')
    override = old_products() if products_old else {}
    if products_old and len(override) != 7 * 62:
        raise ValueError(f'old product inventory {len(override)} != 434')
    mirror(WORK / 'data', S / 'data', override)
    if motion_old:
        (S / 'data/e185_motion_rank.npz').unlink()
        np.savez(S / 'data/e185_motion_rank.npz', mvt_id=inp['ids'], reg=inp['motion_old'])
    np.savez(S / f'submissions/parts_v{B}.npz', mvt_id=inp['ids'], reg=parts['reg'], proba=parts['proba'], values=parts['values'], pred=parts['pred'])
    template = pl.read_parquet(S / 'data/prc-2026-datasets/submitting.parquet')
    sub = template.select(ID).join(pl.DataFrame({ID: inp['ids'], Y: parts['pred']}), on=ID, how='left')
    sub.with_columns(pl.col(Y).round(0).cast(template.schema[Y])).write_parquet(S / f'submissions/gentle-igloo_v{B}.parquet')
    plan = json.loads((S / 'sealed.json').read_text())['plan']
    t0 = time.time()
    for index, stage in enumerate(plan):
        if stage['name'] == 'premix':
            continue
        if stage['name'] == 'e156':  # inject the final stage-3 values of the 62 served rows (no TabPFN run)
            served = json.loads((S / 'e156/run_0/served.json').read_text())
            keys = pl.DataFrame({ID: np.array(served['ids'], dtype=np.float64)})
            fin3 = pl.read_parquet(WORK / f'submissions/gentle-igloo_v{B + 3}.parquet').join(keys, on=ID, how='inner').rename({Y: 'tab'})
            prev = pl.read_parquet(S / f'submissions/gentle-igloo_v{B + 2}.parquet')
            prev.join(fin3, on=ID, how='left').with_columns(pl.coalesce('tab', Y).cast(prev.schema[Y]).alias(Y)).select(prev.columns).write_parquet(
                S / f'submissions/gentle-igloo_v{B + 3}.parquet')
        elif stage['kind'] == 'script':
            fta.script_stage(S, index, stage)
        elif stage['kind'] == 'cached':
            r = fta.cached_stage(S, index, stage, 'replay')
            if r['fits_performed']:
                raise ValueError('fit performed')
        elif stage['kind'] == 'e188':
            fta.e188_stage(S, index, stage)
        elif stage['name'] == 'protect':
            fta.protect_stage(S, index, stage)
    versions = {'premix': 0, 'e185': 1, 'e167': 2, 'e156': 3, 'e141': 4, 'e64': 5, 'e188': 6, 'e152': 10, 'lobt': 12, 'protect': 13, 'lszh': 14, 'half': 15}
    out = {}
    for k, o in versions.items():
        d = pl.read_parquet(S / f'submissions/gentle-igloo_v{B + o}.parquet')
        out[k] = d.sort(ID)
    return out, time.time() - t0


VARIANTS = {
    'identity': dict(swap=set()),
    'tab': dict(swap={'tab'}),
    'nn': dict(swap={'nn'}),
    'proba': dict(swap={'proba'}),
    'cand': dict(swap={'cand'}),
    'motion': dict(swap=set(), motion_old=True),
    'products': dict(swap=set(), products_old=True),
    'proba_model': dict(swap={'proba_model'}),
    'proba_view': dict(swap={'proba_view'}),
    'allpremix': dict(swap={'tab', 'nn', 'proba', 'cand'}, motion_old=True),
    'allold': dict(swap={'tab', 'nn', 'proba', 'cand'}, motion_old=True, products_old=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('variants', nargs='*', default=list(VARIANTS))
    a = ap.parse_args()
    runs = Path(a.runs)
    runs.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(runs / 'inputs.npz')
    if any(n.startswith('proba_') for n in a.variants):
        inp = cross_inputs(inp, runs / 'inputs_cross.npz')
    jan_jul_ids = inp['ids'][inp['jj']]
    # premix reproduction checks
    P_F = np.load(WORK / f'submissions/parts_v{B}.npz')
    pf = premix_for(inp, set())
    for k in ('reg', 'proba', 'values', 'pred'):
        if not np.array_equal(pf[k], P_F[k], equal_nan=True):
            raise ValueError('final premix not reproduced: ' + k)
    P_O = np.load(OLD / 'submissions/parts_v990200.npz')
    j = inp['jj']
    po = fta.premix_arrays(inp['p_old'][j], inp['cat_old'][j], inp['lgb_old'][j],
                           [(w, inp[k + '_old'][j]) for k, w in zip(ANCH, [w for _, w in fta.WEIGHTS['final'][0]])] +
                           [(fta.WEIGHTS['final'][1], inp['nn_old'][j])], list(inp['cand_old'][j].T), inp['d_old'][j], fta.CAT_W)
    o_idx = np.argsort(P_O['mvt_id'])[np.searchsorted(np.sort(P_O['mvt_id']), jan_jul_ids)]
    checks = {k: float(np.nanmax(np.abs(np.nan_to_num(po[k]) - np.nan_to_num(P_O[k][o_idx])))) for k in ('reg', 'proba', 'values', 'pred')}
    print('old premix reproduction max|diff|', checks, flush=True)
    results = {}
    for name in a.variants:
        v = VARIANTS[name]
        parts = premix_for(inp, v['swap'])
        stages, secs = run_chain(runs, name, parts, inp, v.get('motion_old', False), v.get('products_old', False))
        arr = {k: d.filter(pl.col(ID).is_in(jan_jul_ids))[Y].cast(pl.Float64).to_numpy() for k, d in stages.items()}
        ids_sorted = stages['half'].filter(pl.col(ID).is_in(jan_jul_ids))[ID].to_numpy()
        np.savez(runs / f'{name}_janjul.npz', ids=ids_sorted, **arr)
        results[name] = dict(seconds=secs)
        print('VARIANT', name, 'done', round(secs, 1), 's', flush=True)
        if not a.keep:
            shutil.rmtree(runs / name)
    prev = json.loads((runs / 'variants.json').read_text()) if (runs / 'variants.json').exists() else {}
    results = {**prev.get('results', {}), **results}
    (runs / 'variants.json').write_text(json.dumps(dict(results=results, premix_old_reproduction=checks), indent=1))


if __name__ == '__main__':
    main()
