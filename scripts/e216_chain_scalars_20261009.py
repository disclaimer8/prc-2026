"""E216: are the remaining global scalars of the pre-E81 chain mis-set at chain level?

Registered in HYPOTHESES_LOG №442. E212 (№407a/410/425) found that the NN member weight had never been tuned
at chain level (0.25 -> 0.35, lower decided SSE in all twelve 2025 months). This script checks the other global
scalars the same way, ONE AT A TIME from the current final configuration (NN 10 seeds at 0.35, five anchors
scaled by 0.65/0.75, tree pair = 1 - sum), with the E212 machinery (scripts/e212_chain_weights_20261008.py:
cached FIN1b probabilities, frozen E141/E64/E188 stage models, E167 rules, E203 feasibility exclusion):

  (a) cat_w      - CatBoost share inside the tree pair          {0.55, 0.60, [0.65], 0.70, 0.75}
  (b) anchor_mult- LOBT/sched/EOBT/AOBT anchor weights together {0.6, 0.8, [1.0], 1.2, 1.4} x 0.08666...; tree pair keeps 1 - sum
  (c) raw_mult   - raw member weight                            {0.5, [1], 1.5, 2} x 0.02888...; tree pair keeps 1 - sum
  (d) e185_mult  - E185 motion weight on LOBT-less rows         {0.5, [1], 1.5} x (0.35*2/3)
  (e) e152_gate  - E152 stage gate threshold                    {0.90, [0.95], 0.98}
[current value in brackets]. The E152 variants are evaluated on the current chain run (the gate is applied after the
stage models, so no extra stage inference is needed); every other variant is a full chain recomputation.

Control: the current configuration must reproduce the E212 saved m10 chain at w = 0.35
(data/e212_chain_20261008/fold_*/chain_m10.npz, grid index of 0.35) bit-exactly on every row of every pair.

Decision rule (registered before any outcome was computed): a value may replace the current one only if its
chain-level decided SSE is lower than the current value's in ALL twelve months of 2025 AND in both half-years
(1-6, 7-12) AND pooled decided / body (y <= 7200) / trim20 (decided rows, -20/-20 per pair) deltas are all < 0.
If several values of the same scalar qualify, the one closest to the current value is taken. At most ONE scalar
changes overall: the qualifying one with the largest pooled decided gain. Otherwise no change (confirmation).

Outputs (Mac, small): data/e216_chain_20261009/fold_*/chain_e216.npz (int32 deltas vs `ordered`, feasibility
flags) and analysis/e216_chain_scalars_20261009/{result.json,table.md}. No fitting, no uploads, no edits to
scripts/final_tail_assembly_20261008.py.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
for k, v in {'POLARS_MAX_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE',
             'GOMP_SPINCOUNT': '0', 'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage'}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_pre_e81_comparison as A
import e212_chain_weights_20261008 as W

PROTOCOL = 'E216-chain-global-scalars-one-at-a-time-allmonths-v1'
OUT = 'data/e216_chain_20261009'
RESULT = 'analysis/e216_chain_scalars_20261009/result.json'
TABLE = 'analysis/e216_chain_scalars_20261009/table.md'
NN_W = 0.35                      # final NN weight (№425)
J35 = W.GRID.index(NN_W)
SCALE = (1 - NN_W) / 0.75        # 0.65/0.75: E212 proportional scheme (anchors and tree pair scaled together)
CAT_W = .65                      # final_tail_assembly CAT_W
E185_W = .35 * 2 / 3             # final_tail_assembly E185_W (and A.upstream literal)
E152_GATE = .95                  # final_tail_assembly E152_GATE (and A.stage_predictions literal)
CURRENT = dict(cat_w=CAT_W, anchor_mult=1.0, raw_mult=1.0, e185_mult=1.0, e152_gate=E152_GATE)
SCALARS = {'cat_w': (0.55, 0.60, 0.65, 0.70, 0.75), 'anchor_mult': (0.6, 0.8, 1.0, 1.2, 1.4), 'raw_mult': (0.5, 1.0, 1.5, 2.0),
           'e185_mult': (0.5, 1.0, 1.5), 'e152_gate': (0.90, 0.95, 0.98)}
GATES = SCALARS['e152_gate']
HALVES = {'h1': list(range(1, 7)), 'h2': list(range(7, 13))}


def variants():
    """Ordered list of (name, scalar, value, params). 'current' first; E152 variants share the current chain run."""
    out = [('current', None, None, dict(CURRENT))]
    for s, grid in SCALARS.items():
        for v in grid:
            if v == CURRENT[s]: continue
            out.append((f'{s}={v:g}', s, v, {**CURRENT, s: v}))
    return out


VARIANTS = variants()
RUNS = [x for x in VARIANTS if x[1] != 'e152_gate']   # full chain recomputations (14 per pair)


def regular(F, cat_w, anchor_mult, raw_mult):
    """Final-configuration premix with the three premix scalars. At the current values bit-identical to
    W.regular_at(0.35, r0, anchors, m10): r0 is formed by the same expression, the multipliers are exact *1.0,
    and the tree weight is the E212 tree weight minus an exactly-zero mass difference."""
    r0 = cat_w * F['cat'] + (1 - cat_w) * F['lgb']
    base_w = [x * SCALE for x in A.WEIGHTS[:-1]]                     # raw, lobt, sched, eobt, aobt at the final configuration
    new_w = [base_w[0] * raw_mult, *(x * anchor_mult for x in base_w[1:])]
    tree_w = ((1 - sum(A.WEIGHTS)) * SCALE) - (sum(new_w) - sum(base_w))
    if tree_w <= 0: raise ValueError('tree pair weight not positive')
    return tree_w * r0 + sum(wk * g for wk, g in zip([*new_w, NN_W], [*F['anchors'], F['mem']['m10']]))


def upstream(z, r, motion, f, ia, ib, excluded, e185_w):
    """A.upstream with the E185 motion weight as a parameter (literal .35*2/3 in A)."""
    lobt = f['d_mvt_lobt'].to_numpy(); v = z['values'].astype(float).copy()
    v[:, 0] = r; v[:, -1] = r + 86400
    changes = np.any(A.feasibility(v, lobt) != A.feasibility(z['values'], lobt), axis=1) & ~excluded
    if changes.any(): raise ValueError(f'Feasibility exclusion failed on {changes.sum()} scored rows')
    p = z['proba'].astype(float).copy()
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=2e-7): raise ValueError('invalid cached probabilities')
    p = np.where(np.isnan(v), 0, p); p = p / p.sum(axis=1, keepdims=True)
    base = np.rint(A.mixture(p, v, lobt))
    subset = (f['LOBT_flt'].is_null() & (f['ADEP_mvt'] != 'LIRF')).to_numpy(); w = e185_w
    r = r.copy(); r[subset] = (1 - w) * r[subset] + w * motion[subset]
    v[:, 0] = r; v[:, -1] = r + 86400
    base[subset] = np.rint(A.mixture(p, v, lobt)[subset])
    p[ia, 1] = 0; p[ia, 5] = 0
    rest = np.clip(1 - p[ib, 1], 1e-9, None); p[ib] *= ((1 - .9) / rest)[:, None]; p[ib, 1] = .9
    p = np.where(np.isnan(v), 0, p); p /= p.sum(axis=1, keepdims=True)
    changed = ia | ib; base[changed] = np.rint(A.mixture(p, v, lobt)[changed])
    return base, p


def stage_pregate(root, fold, d, base, p, spec, threads):
    """A.stage_predictions up to (not including) the E152 gate; returns the augmented arm before gating."""
    if 'torch' in sys.modules: raise RuntimeError('LGB-only fresh process required')
    import lightgbm as lgb
    features = {s['name']: s['features'] for s in spec['stages']}
    d = d.with_columns(pl.Series('base', base), pl.Series('p_nonreg', 1 - p[:, 0])).with_columns(
        base_vs_first_move=pl.col('base') - pl.col('mvt_first_move'), base_vs_first=pl.col('base') - pl.col('mvt_first'))
    covered = (d['coverage'].to_numpy() >= 2); eligible = ((d['coverage'] < 2) & (d['vi_has_prev'] == 1)).fill_null(False).to_numpy()
    target = eligible & (d['ADEP_mvt'].to_numpy() == 'LIRF') & (d['has_nm'].to_numpy() == 1)
    path = root / A.MODELS / fold
    def predict(name, mask, data):
        x = data.filter(pl.Series(mask)).select(features[name]).to_numpy()
        if np.isinf(x).any(): raise ValueError('infinite stage input')
        out = []
        for seed in A.SEEDS:
            m = lgb.Booster(model_file=str(path / name / f'seed{seed}.txt'))
            if m.num_feature() != len(features[name]) or m.current_iteration() != (800 if name == 'e141' else 400): raise ValueError('stage model dimensions')
            out.append(m.predict(x, num_threads=threads))
        return out
    e141 = base.copy()
    e141[covered] = np.rint(np.mean([np.clip(base[covered] + c, 0, None) for c in predict('e141', covered, d)], axis=0))
    d = d.with_columns(pl.Series('base', e141))
    ordinary = e141.copy(); ordinary[eligible] = np.rint(np.clip(e141[eligible] + np.mean(predict('e64', eligible, d), axis=0), 0, None))
    augmented = ordinary.copy(); augmented[target] = np.rint(np.clip(e141[target] + np.mean(predict('e188', target, d), axis=0), 0, None))
    if not np.array_equal(ordinary[~target], augmented[~target]): raise ValueError('E188 population escaped')
    return augmented


def chain(root, spec, F, ra, rb, P, gates, threads):
    """One chain run at premix/E185 params P; returns {gate: int64 prediction} for each requested E152 gate and the flags."""
    r = regular(F, P['cat_w'], P['anchor_mult'], P['raw_mult'])
    v = F['z']['values'].astype(float).copy(); v[:, 0] = r; v[:, -1] = r + 86400
    unstable = np.any(A.feasibility(v, F['lobt']) != A.feasibility(F['z']['values'], F['lobt']), axis=1)
    excluded = F['excluded'] | unstable
    base, p = upstream(F['z'], r, F['motion'], F['f'], np.isin(F['ids'], ra), np.isin(F['ids'], rb), excluded, E185_W * P['e185_mult'])
    pre = stage_pregate(root, F['name'], F['d'], base, p, spec, threads)
    conf = p[:, 1:5].max(axis=1); out = {}
    for g in gates:
        aug = pre.copy(); m = conf >= g; aug[m] = base[m]
        aug[excluded] = np.rint(F['z']['pred'][excluded])
        if not np.isfinite(aug).all() or (aug < 0).any() or not np.array_equal(aug, np.rint(aug)): raise ValueError('invalid stage output')
        out[g] = aug.astype(np.int64)
    return out, unstable, int((conf >= E152_GATE).sum())


def load_fold(root, chain_df, feature, natural, fold):
    F = W.load_fold(root, chain_df, feature, natural, fold, ('m3', 'm10'))
    name = F['name']; ids = F['ids']; y = F['y']; mo = F['f']['month'].to_numpy()
    F['cat'] = A.aligned(root / 'data/cat_extra175_20261006/merged' / (name + '.npz'), ids, y, mo)['reg2']
    F['lgb'] = A.aligned(root / 'data/lgb_fin1b_oof_20261005' / name / 'complete/prediction.npz', ids, y, mo)['reg2']
    if not np.array_equal(.65 * F['cat'] + (1 - .65) * F['lgb'], F['r0']): raise ValueError('cat/lgb pair does not rebuild r0 ' + name)
    if not np.array_equal(regular(F, **{k: CURRENT[k] for k in ('cat_w', 'anchor_mult', 'raw_mult')}), W.regular_at(NN_W, F['r0'], F['anchors'], F['mem']['m10'])):
        raise ValueError('premix arithmetic not bit-identical to E212 at the current configuration ' + name)
    with np.load(root / W.OUT / name / 'chain_m10.npz') as z:
        if not np.array_equal(z['mvt_id'], ids) or z['grid'][J35] != NN_W or not np.array_equal(z['excluded_e203'], F['excluded']): raise ValueError('E212 chain file alignment ' + name)
        F['e212_delta'] = z['delta'][J35].copy(); F['e212_unstable'] = z['unstable'][J35].copy()
    return F


def compute(root, folds, threads):
    out = root / OUT; out.mkdir(parents=True, exist_ok=True)
    chain_df = pl.read_parquet(root / W.CHAIN)
    if chain_df.height != 2085047: raise ValueError('chain population')
    spec = A.read(root / A.BUNDLE / 'manifest.json')['spec']; natural = pl.read_parquet(root / A.BUNDLE / 'natural_features.parquet')
    cols = [A.ID, A.Y, 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'LOBT_flt', 'd_mvt_lobt', 'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_eobt']
    feature = pl.read_parquet(root / 'data/features_fin1b.parquet', columns=cols).filter(pl.col('split') == 'train').sort(A.ID)
    ra, rb = A.rules(root)
    names = [x[0] for x in VARIANTS]
    for fold in folds:
        name = f'fold_{fold}_{fold + 6}'; fd = out / name
        if (fd / 'chain_e216.npz').exists(): print('DONE ALREADY', name, flush=True); continue
        F = load_fold(root, chain_df, feature, natural, fold)
        delta = np.zeros((len(VARIANTS), len(F['ids'])), np.int32); unstable = np.zeros((len(VARIANTS), len(F['ids'])), bool); secs = {}; gated = {}
        for vname, scalar, value, P in RUNS:
            gates = GATES if vname == 'current' else (E152_GATE,)
            t = time.monotonic(); preds, u, n_gated = chain(root, spec, F, ra, rb, P, gates, threads); secs[vname] = time.monotonic() - t; gated[vname] = n_gated
            for g, aug in preds.items():
                vn = vname if g == E152_GATE else f'e152_gate={g:g}'; j = names.index(vn)
                dd = aug - F['ordered']
                if np.abs(dd).max() >= 2 ** 31: raise ValueError('delta overflow')
                delta[j] = dd; unstable[j] = u
                if vn == 'current':
                    if not np.array_equal(delta[j], F['e212_delta']) or not np.array_equal(u, F['e212_unstable']):
                        raise ValueError(f'CONTROL FAILED {name}: current configuration differs from the E212 saved m10 delta at 0.35 on {int((delta[j] != F["e212_delta"]).sum())} rows')
                    print('CONTROL OK', name, 'E212 m10 delta at 0.35 reproduced on all', len(aug), 'rows', flush=True)
                print(name, vn, 'rms(delta vs ordered) %.2f s' % float(np.sqrt(np.mean(dd[~F['excluded']].astype(float) ** 2))),
                      'rms(vs current) %.2f s' % float(np.sqrt(np.mean((delta[j] - delta[0])[~F['excluded']].astype(float) ** 2))),
                      'unstable', int(u.sum()), 'gated', n_gated, '%.1f s' % secs[vname], flush=True)
        fd.mkdir(exist_ok=True); tmp = fd / '.chain_e216.partial.npz'
        np.savez_compressed(tmp, mvt_id=F['ids'], names=np.array(names), delta=delta, unstable=unstable, excluded_e203=F['excluded'])
        tmp.rename(fd / 'chain_e216.npz')
        A.save(fd / 'chain_e216.json', dict(protocol=PROTOCOL, fold=fold, variants=[dict(name=n, scalar=s, value=v, params=P) for n, s, v, P in VARIANTS],
                                            threads=threads, seconds=secs, rows_gated_at_095=gated, control_exact=True,
                                            m10_source=f'{W.SEEDS10}/{name}/ordered/prediction_10seed.npz', m10_sha256=A.sha(root / W.SEEDS10 / name / 'ordered' / 'prediction_10seed.npz'),
                                            e212_chain_m10_sha256=A.sha(root / W.OUT / name / 'chain_m10.npz'), chain_sha256=A.sha(root / W.CHAIN), self_sha256=A.sha(Path(__file__))))
        print('COMPLETE', name, 'total %.0f s' % sum(secs.values()), flush=True)
        del F
    print('ALL REQUESTED FOLDS COMPLETE', flush=True)


def trim(d, k=20):
    s = np.sort(d); return float(s[k:-k].sum())


def evaluate(root):
    chain_df = pl.read_parquet(root / W.CHAIN).sort(A.ID)
    names = [x[0] for x in VARIANTS]; folds = []
    for fold in W.FOLDS:
        name = f'fold_{fold}_{fold + 6}'; old = chain_df.filter(pl.col('month').is_in([fold, fold + 6]))
        ids = old[A.ID].to_numpy(); y = old[A.Y].to_numpy().astype(np.float64); ordered = old['ordered'].to_numpy().astype(np.float64)
        ok = old['evaluation_mask'].to_numpy(); airport = old['ADEP_mvt'].to_numpy(); nm = old['has_nm'].to_numpy(); D = old['D'].to_numpy(); month = old['month'].to_numpy()
        with np.load(root / OUT / name / 'chain_e216.npz') as z:
            if not np.array_equal(z['mvt_id'], ids) or list(z['names']) != names or not np.array_equal(z['excluded_e203'], ~ok): raise ValueError('chain file alignment ' + name)
            delta = z['delta'].astype(np.float64); unstable_any = z['unstable'].any(axis=0)
        with np.load(root / W.OUT / name / 'chain_m10.npz') as z:
            if not np.array_equal(z['mvt_id'], ids) or not np.array_equal(z['delta'][J35], delta[0].astype(np.int32)): raise ValueError('control not exact in saved file ' + name)
        common = ok & ~unstable_any
        masks = dict(included=common, body=common & (y <= 7200), decided=common & ~((airport == 'LIRF') & ~nm & (D >= 14400)))
        se = (ordered[None, :] + delta - y[None, :]) ** 2
        folds.append(dict(fold=fold, name=name, y=y, se=se, masks=masks, month=month, n_rows=len(ids), e203_included=int(ok.sum()),
                          extra_excluded_feasibility=int((ok & unstable_any).sum())))
    res = dict(protocol=PROTOCOL, nn_w=NN_W, current=CURRENT, scalars={k: list(v) for k, v in SCALARS.items()}, control_exact=True,
               populations={F['name']: dict(rows=F['n_rows'], e203_included=F['e203_included'], extra_excluded_feasibility=F['extra_excluded_feasibility'],
                                             **{k: int(v.sum()) for k, v in F['masks'].items()}) for F in folds})
    nd = {m: sum(int(F['masks'][m].sum()) for F in folds) for m in ('decided', 'body')}
    base_month = {mo: 0.0 for mo in range(1, 13)}; n_month = {mo: 0 for mo in range(1, 13)}
    for F in folds:
        for mo in (F['fold'], F['fold'] + 6):
            sel = F['masks']['decided'] & (F['month'] == mo); base_month[mo] = float(F['se'][0][sel].sum()); n_month[mo] = int(sel.sum())
    base_dec = sum(base_month.values()); base_body = sum(float(F['se'][0][F['masks']['body']].sum()) for F in folds)
    res['current_rmse'] = dict(decided=float(np.sqrt(base_dec / nd['decided'])), body=float(np.sqrt(base_body / nd['body'])),
                               per_month={str(mo): float(np.sqrt(base_month[mo] / n_month[mo])) for mo in range(1, 13)}, n_decided=nd['decided'], n_body=nd['body'],
                               n_month={str(mo): n_month[mo] for mo in range(1, 13)})
    out = {}
    for j, (vname, scalar, value, P) in enumerate(VARIANTS):
        if j == 0: continue
        months = {}; per_fold = []; dec = body = t20 = 0.0
        for F in folds:
            d = F['se'][j] - F['se'][0]; m = F['masks']
            dd = float(d[m['decided']].sum()); db = float(d[m['body']].sum()); tt = trim(d[m['decided']])
            dec += dd; body += db; t20 += tt
            per_fold.append(dict(fold=F['fold'], decided_delta_sse=dd, body_delta_sse=db, trim20_sse=tt, included_delta_sse=float(d[m['included']].sum())))
            for mo in (F['fold'], F['fold'] + 6):
                sel = m['decided'] & (F['month'] == mo); ds = float(d[sel].sum()); b = base_month[mo]; n = n_month[mo]
                months[str(mo)] = dict(n_decided=n, delta_sse=ds, delta_rmse=float(np.sqrt((b + ds) / n) - np.sqrt(b / n)))
        halves = {}
        for h, ms in HALVES.items():
            ds = sum(months[str(mo)]['delta_sse'] for mo in ms); b = sum(base_month[mo] for mo in ms); n = sum(n_month[mo] for mo in ms)
            halves[h] = dict(n_decided=n, delta_sse=ds, delta_rmse=float(np.sqrt((b + ds) / n) - np.sqrt(b / n)))
        months_lower = sum(v['delta_sse'] < 0 for v in months.values()); halves_lower = sum(v['delta_sse'] < 0 for v in halves.values())
        gate = dict(months_lower=months_lower, all_12_months=months_lower == 12, both_halves=halves_lower == 2, pooled_decided_lt0=dec < 0, pooled_body_lt0=body < 0, pooled_trim20_lt0=t20 < 0,
                    pairs_lower=sum(x['decided_delta_sse'] < 0 for x in per_fold))
        gate['qualifies'] = gate['all_12_months'] and gate['both_halves'] and gate['pooled_decided_lt0'] and gate['pooled_body_lt0'] and gate['pooled_trim20_lt0']
        out[vname] = dict(scalar=scalar, value=value, params=P, pooled_decided_delta_sse=dec, pooled_decided_delta_rmse=float(np.sqrt((base_dec + dec) / nd['decided']) - np.sqrt(base_dec / nd['decided'])),
                          pooled_body_delta_sse=body, pooled_body_delta_rmse=float(np.sqrt((base_body + body) / nd['body']) - np.sqrt(base_body / nd['body'])), pooled_trim20_sse=t20,
                          per_fold=per_fold, per_month=months, half_years=halves, gate=gate)
    res['variants'] = out
    # decision: per scalar the qualifying value closest to the current one; overall at most one scalar (largest pooled decided gain)
    decision = {}
    for s in SCALARS:
        q = [(v, out[n]) for n, (_, sc, v, _) in [(x[0], x) for x in VARIANTS[1:]] if sc == s and out[n]['gate']['qualifies']]
        if q:
            v, r = min(q, key=lambda t: (abs(t[0] - CURRENT[s]), t[0]))
            decision[s] = dict(qualifying_values=[t[0] for t in q], chosen=v, pooled_decided_delta_sse=r['pooled_decided_delta_sse'], pooled_decided_delta_rmse=r['pooled_decided_delta_rmse'])
        else: decision[s] = dict(qualifying_values=[], chosen=None)
    cands = [(s, d) for s, d in decision.items() if d['chosen'] is not None]
    if cands:
        s, d = min(cands, key=lambda t: t[1]['pooled_decided_delta_sse'])
        res['verdict'] = dict(change=True, scalar=s, value=d['chosen'], pooled_decided_delta_rmse=d['pooled_decided_delta_rmse'], per_scalar=decision)
    else: res['verdict'] = dict(change=False, scalar=None, value=None, per_scalar=decision)
    p = root / RESULT; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, indent=1)); print('WROTE', p, flush=True)
    # markdown table: per scalar x per month decided dRMSE vs current (s), plus halves and pooled columns
    lines = ['| вариант | ' + ' | '.join(str(m) for m in range(1, 13)) + ' | 1–6 | 7–12 | pooled decided | body | trim20 | мес.<0 | пар<0 | квалиф. |', '|' + '---|' * 22]
    for n, r in out.items():
        g = r['gate']
        lines.append(f'| {n} | ' + ' | '.join('%+.3f' % r['per_month'][str(m)]['delta_rmse'] for m in range(1, 13)) + ' | %+.3f | %+.3f | %+.4f | %+.4f | %+.2e | %d/12 | %d/6 | %s |' % (
            r['half_years']['h1']['delta_rmse'], r['half_years']['h2']['delta_rmse'], r['pooled_decided_delta_rmse'], r['pooled_body_delta_rmse'], r['pooled_trim20_sse'],
            g['months_lower'], g['pairs_lower'], 'да' if g['qualifies'] else 'нет'))
    (root / TABLE).write_text('\n'.join(lines) + '\n'); print('\n'.join(lines), flush=True)
    print('populations', json.dumps(res['populations']), flush=True)
    print('current decided RMSE %.4f body %.4f' % (res['current_rmse']['decided'], res['current_rmse']['body']), flush=True)
    print('VERDICT', json.dumps(res['verdict']), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['compute', 'evaluate'])
    p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--folds', default='1,2,3,4,5,6'); p.add_argument('--threads', type=int, default=4); a = p.parse_args()
    if not 1 <= a.threads <= 4: raise ValueError('1..4 threads on the Mac (registered)')
    if a.action == 'compute': compute(a.root.resolve(), [int(x) for x in a.folds.split(',')], a.threads)
    else: evaluate(a.root.resolve())


if __name__ == '__main__':
    main()
