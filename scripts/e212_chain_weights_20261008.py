"""E212 stage 2: exact chain recomputation at each NN-member weight with the real 10-seed member.

Registered in HYPOTHESES_LOG №407a (stage 2b). For every 2025 month-pair fold and every grid
weight w in 0.05..0.60 (step 0.05) the production pre-E81 chain of E203 (`cv_subset`:
A.upstream + A.stage_predictions, same cached FIN1b probabilities, same E141/E64/E188 stage
models, same E167 rules, same feasibility exclusion) is recomputed with the NN member at
weight w, the remaining members keeping their proportions:
    regular(w) = ((1-w)/0.75) * [ (1-sum(W))*(.65*cat+.35*lgb) + sum_k W_k*anchor_k ] + w*NN
which at w=0.25 is bit-identical to the E203 arithmetic (scale factor 1.0).
Member variants: `m3` = the original 3-seed mean (data/e197|e202_trajectory_20261007), `m10` =
the 10-seed mean produced on Home by scripts/e212_extra_seeds_20261008.py (prediction_10seed.npz,
whose `member3` must equal the original member exactly).

Control: w=0.25 with m3 must reproduce the `ordered` column of
data/e203_pre_e81_20261007/predictions.parquet on every row.

Decision (action `evaluate`): leave-one-pair-out on chain-level decided SSE with m10 - for each
held-out pair the weight w_k minimising the pooled decided SSE of the other five pairs; the
held-out pair is scored at w_k against w=0.25 (same member). Gate exactly as registered:
decided dSSE<0 in >=4/6 pairs, pooled body (y<=7200) dSSE<0, pooled trim20 (decided rows,
-20/-20 per pair) <0, pairs 2+8 and 6+12 not worse (<=0). Stop rule: |pooled decided dRMSE|<0.02 s
-> null. The same evaluation is repeated with m3 for comparison only.

Outputs (Mac, small): data/e212_chain_20261008/fold_*/chain_{m3,m10}.npz (int32 deltas vs
`ordered`, feasibility flags) and analysis/e212_nn_weight_10seed_20261008/chain_result.json.
No fitting, no uploads, no edits to the final assembly.
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
import e195_pre_e81_comparison as A  # aligned(), rules(), feasibility(), upstream(), stage_predictions(), WEIGHTS, ANCHORS, ID, Y

PROTOCOL = 'E212-nn-weight-10seed-analytic-lopo-decided-grid005-v1'
STAGE = 'stage2-exact-chain'
CHAIN = 'data/e203_pre_e81_20261007/predictions.parquet'
SEEDS10 = 'data/e212_seeds3to9_20261008'
OUT = 'data/e212_chain_20261008'
RESULT = 'analysis/e212_nn_weight_10seed_20261008/chain_result.json'
GRID = tuple(k / 20 for k in range(1, 13))  # 0.05 .. 0.60, exact doubles; 0.25 == A.WEIGHTS[-1]
W25 = GRID.index(0.25)
MEMBERS = ('m3', 'm10')
FOLDS = (1, 2, 3, 4, 5, 6)


def traj_dir(fold):
    return Path('data/e197_trajectory_20261007' if fold in (1, 4) else 'data/e202_trajectory_20261007') / f'fold_{fold}_{fold + 6}' / 'ordered'


def regular_at(w, r0, anchors, member):
    """E203 arithmetic with the NN weight w; at w=0.25 bit-identical to cv_subset (scale == 1.0)."""
    scale = (1 - w) / 0.75
    weights = [*(x * scale for x in A.WEIGHTS[:-1]), w]
    return ((1 - sum(A.WEIGHTS)) * scale) * r0 + sum(wk * g for wk, g in zip(weights, [*anchors, member]))


def load_fold(root, chain, feature, natural, fold, members):
    name = f'fold_{fold}_{fold + 6}'
    f = feature.filter(pl.col('month').is_in([fold, fold + 6]))
    ids = f[A.ID].to_numpy(); y = f[A.Y].to_numpy(); mo = f['month'].to_numpy()
    old = chain.filter(pl.col('month').is_in([fold, fold + 6])).sort(A.ID)
    if any(not np.array_equal(old[k].to_numpy(), f[k].to_numpy()) for k in (A.ID, A.Y, 'month')): raise ValueError('E203 population changed')
    d = f.select(A.ID).join(natural, on=A.ID, how='left', validate='1:1', maintain_order='left')
    if d[A.Y].null_count() or not np.array_equal(d[A.Y].to_numpy(), y): raise ValueError('natural feature alignment')
    z = A.aligned(root / 'data/cv_preds_fin1b_hz' / (name + '.npz'), ids, y)
    cat = A.aligned(root / 'data/cat_extra175_20261006/merged' / (name + '.npz'), ids, y, mo)['reg2']
    lgb = A.aligned(root / 'data/lgb_fin1b_oof_20261005' / name / 'complete/prediction.npz', ids, y, mo)['reg2']
    anchors = [A.aligned(root / 'data' / ('cv_e130' + x) / (name + '.npz'), ids)['reg2'] for x in A.ANCHORS]
    motion = A.aligned(root / 'data/cv_e130motion' / (name + '.npz'), ids)['reg2']
    m3 = A.aligned(root / traj_dir(fold) / 'prediction.npz', ids, y, mo)['member']
    mem = {'m3': m3}
    if 'm10' in members:
        p10 = root / SEEDS10 / name / 'ordered' / 'prediction_10seed.npz'
        z10 = A.aligned(p10, ids, y, mo)
        if not np.array_equal(z10['member3'], m3): raise ValueError('10-seed file member3 differs from the original member ' + name)
        if z10['per_seed_sec'].shape[0] != 10 or not np.isfinite(z10['member10']).all(): raise ValueError('10-seed file incomplete ' + name)
        c = json.loads((p10.parent / 'complete.json').read_text())
        if c['fold'] != fold or c['arm'] != 'ordered' or sorted(c['seeds']) != list(range(3, 10)) or not c['original_member_exact']: raise ValueError('10-seed receipt ' + name)
        mem['m10'] = z10['member10'].astype(np.float64) if z10['member10'].dtype != np.float64 else z10['member10']
    if not all(np.isfinite(v).all() for v in [cat, lgb, *anchors, motion, *mem.values()]): raise ValueError('nonfinite member')
    r0 = .65 * cat + (1 - .65) * lgb
    # bit-identity check of the general formula against the E203 expression at w = 0.25
    e203 = (1 - sum(A.WEIGHTS)) * r0 + sum(w * g for w, g in zip(A.WEIGHTS, [*anchors, m3]))
    if not np.array_equal(regular_at(0.25, r0, anchors, m3), e203): raise ValueError('weight arithmetic not bit-identical at 0.25')
    return dict(name=name, fold=fold, f=f, d=d, ids=ids, y=y, z=z, r0=r0, anchors=anchors, motion=motion, mem=mem,
                ordered=old['ordered'].to_numpy(), excluded=~old['evaluation_mask'].to_numpy(), lobt=f['d_mvt_lobt'].to_numpy())


def chain(root, spec, F, ra, rb, w, member, threads):
    r = regular_at(w, F['r0'], F['anchors'], F['mem'][member])
    v = F['z']['values'].astype(float).copy(); v[:, 0] = r; v[:, -1] = r + 86400
    unstable = np.any(A.feasibility(v, F['lobt']) != A.feasibility(F['z']['values'], F['lobt']), axis=1)
    excluded = F['excluded'] | unstable
    base, p = A.upstream(F['z'], r, F['motion'], F['f'], np.isin(F['ids'], ra), np.isin(F['ids'], rb), excluded)
    _, aug = A.stage_predictions(root, F['name'], F['d'], base, p, spec, threads)
    aug[excluded] = np.rint(F['z']['pred'][excluded])
    if not np.isfinite(aug).all() or (aug < 0).any() or not np.array_equal(aug, np.rint(aug)): raise ValueError('invalid stage output')
    return aug.astype(np.int64), unstable


def compute(root, folds, members, threads):
    out = root / OUT; out.mkdir(parents=True, exist_ok=True)
    chain_df = pl.read_parquet(root / CHAIN)
    if chain_df.height != 2085047: raise ValueError('chain population')
    spec = A.read(root / A.BUNDLE / 'manifest.json')['spec']; natural = pl.read_parquet(root / A.BUNDLE / 'natural_features.parquet')
    cols = [A.ID, A.Y, 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'LOBT_flt', 'd_mvt_lobt', 'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_eobt']
    feature = pl.read_parquet(root / 'data/features_fin1b.parquet', columns=cols).filter(pl.col('split') == 'train').sort(A.ID)
    ra, rb = A.rules(root)
    for fold in folds:
        todo = [m for m in members if not (out / f'fold_{fold}_{fold + 6}' / f'chain_{m}.npz').exists()]
        if not todo: print('DONE ALREADY', fold, members, flush=True); continue
        F = load_fold(root, chain_df, feature, natural, fold, todo); fd = out / F['name']; fd.mkdir(exist_ok=True)
        for m in todo:
            delta = np.zeros((len(GRID), len(F['ids'])), np.int32); unstable = np.zeros((len(GRID), len(F['ids'])), bool); secs = []
            for j, w in enumerate(GRID):
                t = time.monotonic(); aug, u = chain(root, spec, F, ra, rb, w, m, threads); secs.append(time.monotonic() - t)
                dd = aug - F['ordered']
                if np.abs(dd).max() >= 2 ** 31: raise ValueError('delta overflow')
                delta[j] = dd; unstable[j] = u
                if m == 'm3' and j == W25:
                    if not np.array_equal(aug, F['ordered']): raise ValueError(f'CONTROL FAILED {F["name"]}: {int((aug != F["ordered"]).sum())} rows differ from `ordered`')
                    print('CONTROL OK', F['name'], 'ordered reproduced on all', len(aug), 'rows', flush=True)
                print(F['name'], m, f'w={w:.2f}', 'rms(delta) %.2f s' % float(np.sqrt(np.mean(dd[~F['excluded']].astype(float) ** 2))),
                      'unstable', int(u.sum()), '%.1f s' % secs[-1], flush=True)
            tmp = fd / f'.chain_{m}.partial.npz'
            np.savez_compressed(tmp, mvt_id=F['ids'], grid=np.array(GRID), delta=delta, unstable=unstable, excluded_e203=F['excluded'])
            tmp.rename(fd / f'chain_{m}.npz')
            A.save(fd / f'chain_{m}.json', dict(protocol=PROTOCOL, stage=STAGE, fold=fold, member=m, grid=list(GRID), threads=threads, seconds=secs,
                                                 control_exact=(m == 'm3'), member_source=str(traj_dir(fold) / 'prediction.npz') if m == 'm3' else f'{SEEDS10}/{F["name"]}/ordered/prediction_10seed.npz',
                                                 member_sha256=A.sha(root / traj_dir(fold) / 'prediction.npz') if m == 'm3' else A.sha(root / SEEDS10 / F['name'] / 'ordered' / 'prediction_10seed.npz'),
                                                 chain_sha256=A.sha(root / CHAIN), self_sha256=A.sha(Path(__file__))))
            print('COMPLETE', F['name'], m, 'total %.0f s' % sum(secs), flush=True)
        del F
    print('ALL REQUESTED FOLDS COMPLETE', flush=True)


def trims(d, ks=(3, 20)):
    s = np.sort(d); return {f'trim{k}_sse': float(s[k:-k].sum()) for k in ks}


def evaluate(root, members):
    chain_df = pl.read_parquet(root / CHAIN).sort(A.ID)
    folds = []
    for fold in FOLDS:
        name = f'fold_{fold}_{fold + 6}'; fd = root / OUT / name
        old = chain_df.filter(pl.col('month').is_in([fold, fold + 6]))
        ids = old[A.ID].to_numpy(); y = old[A.Y].to_numpy().astype(np.float64); ordered = old['ordered'].to_numpy().astype(np.float64)
        ok = old['evaluation_mask'].to_numpy(); airport = old['ADEP_mvt'].to_numpy(); nm = old['has_nm'].to_numpy(); D = old['D'].to_numpy()
        pred = {}; unstable_any = np.zeros(len(ids), bool)
        for m in members:
            with np.load(fd / f'chain_{m}.npz') as z:
                if not np.array_equal(z['mvt_id'], ids) or not np.array_equal(z['grid'], np.array(GRID)) or not np.array_equal(z['excluded_e203'], ~ok): raise ValueError('chain file alignment ' + name)
                pred[m] = ordered[None, :] + z['delta'].astype(np.float64); unstable_any |= z['unstable'].any(axis=0)
        if 'm3' in pred and np.any(pred['m3'][W25] != ordered): raise ValueError('control not exact in saved file ' + name)
        common = ok & ~unstable_any
        masks = dict(included=common, body=common & (y <= 7200), decided=common & ~((airport == 'LIRF') & ~nm & (D >= 14400)))
        se = {m: (pred[m] - y[None, :]) ** 2 for m in pred}
        folds.append(dict(fold=fold, name=name, y=y, se=se, masks=masks, n_rows=len(ids), e203_included=int(ok.sum()),
                          extra_excluded_feasibility=int((ok & unstable_any).sum())))
    res = dict(protocol=PROTOCOL, stage=STAGE, grid=list(GRID), members=list(members), primary_member='m10', control_exact=True,
               populations={F['name']: dict(rows=F['n_rows'], e203_included=F['e203_included'], extra_excluded_feasibility=F['extra_excluded_feasibility'],
                                             **{k: int(v.sum()) for k, v in F['masks'].items()}) for F in folds})
    sse = {m: {k: np.array([[F['se'][m][j][F['masks'][k]].sum() for j in range(len(GRID))] for F in folds]) for k in ('included', 'body', 'decided')} for m in members}
    n = {k: np.array([int(F['masks'][k].sum()) for F in folds]) for k in ('included', 'body', 'decided')}
    for m in members:
        pooled = sse[m]['decided'].sum(axis=0); jstar = int(np.argmin(pooled))
        r = dict(global_w_star=GRID[jstar],
                 rmse_decided_at_w={f'{w:.2f}': float(np.sqrt(pooled[j] / n['decided'].sum())) for j, w in enumerate(GRID)},
                 rmse_body_at_w={f'{w:.2f}': float(np.sqrt(sse[m]['body'].sum(axis=0)[j] / n['body'].sum())) for j, w in enumerate(GRID)},
                 delta_rmse_wstar_vs_025_in_sample=float(np.sqrt(pooled[jstar] / n['decided'].sum()) - np.sqrt(pooled[W25] / n['decided'].sum())),
                 per_fold_w_star={F['name']: GRID[int(np.argmin(sse[m]['decided'][i]))] for i, F in enumerate(folds)},
                 per_fold_rmse_decided_at_025={F['name']: float(np.sqrt(sse[m]['decided'][i, W25] / n['decided'][i])) for i, F in enumerate(folds)})
        # leave-one-pair-out (registered rule), scored against w = 0.25 of the same member
        lopo = []; pooled_delta = dict(decided=0.0, body=0.0, trim20=0.0, trim3=0.0)
        for i, F in enumerate(folds):
            train = [k for k in range(len(folds)) if k != i]
            jw = int(np.argmin(sse[m]['decided'][train].sum(axis=0))); wk = GRID[jw]
            d = F['se'][m][jw] - F['se'][m][W25]
            dec = F['masks']['decided']; body = F['masks']['body']; t = trims(d[dec])
            dd = float(d[dec].sum()); db = float(d[body].sum()); base = float(F['se'][m][W25][dec].sum()); nd = int(dec.sum())
            lopo.append(dict(fold=F['fold'], name=F['name'], w_chosen=wk, n_decided=nd, decided_delta_sse=dd,
                             decided_delta_rmse=float(np.sqrt((base + dd) / nd) - np.sqrt(base / nd)), body_delta_sse=db,
                             included_delta_sse=float(d[F['masks']['included']].sum()), **t))
            pooled_delta['decided'] += dd; pooled_delta['body'] += db; pooled_delta['trim20'] += t['trim20_sse']; pooled_delta['trim3'] += t['trim3_sse']
        lower = sum(1 for x in lopo if x['decided_delta_sse'] < 0)
        f2 = next(x for x in lopo if x['fold'] == 2); f6 = next(x for x in lopo if x['fold'] == 6)
        base_all = float(sse[m]['decided'][:, W25].sum()); nd_all = int(n['decided'].sum())
        pooled_drmse = float(np.sqrt((base_all + pooled_delta['decided']) / nd_all) - np.sqrt(base_all / nd_all))
        gate = dict(decided_lower=lower, decided_lower_ge4=lower >= 4, pooled_body_lt0=pooled_delta['body'] < 0, pooled_trim20_lt0=pooled_delta['trim20'] < 0,
                    fold2_not_worse=f2['decided_delta_sse'] <= 0, fold6_not_worse=f6['decided_delta_sse'] <= 0)
        gate['pass'] = all(v for k, v in gate.items() if k != 'decided_lower')
        r.update(lopo=lopo, pooled_lopo_delta_sse=pooled_delta, pooled_lopo_decided_delta_rmse=pooled_drmse, null_by_threshold=abs(pooled_drmse) < 0.02,
                 gate=gate, lopo_weights_unanimous=len({x['w_chosen'] for x in lopo}) == 1)
        res[m] = r
    # cross-member comparisons at fixed weights (what production would actually see)
    if 'm3' in members and 'm10' in members:
        cmp = {}
        for label, jm10 in [('m10_025_vs_m3_025', W25)] + [(f'm10_{w:.2f}_vs_m3_025', j) for j, w in enumerate(GRID) if j != W25]:
            per = []
            for F in folds:
                d = F['se']['m10'][jm10] - F['se']['m3'][W25]; dec = F['masks']['decided']
                per.append(dict(fold=F['fold'], decided_delta_sse=float(d[dec].sum()), body_delta_sse=float(d[F['masks']['body']].sum()), **trims(d[dec])))
            tot = sum(x['decided_delta_sse'] for x in per); base = float(sse['m3']['decided'][:, W25].sum()); nd_all = int(n['decided'].sum())
            cmp[label] = dict(per_fold=per, pooled_decided_delta_sse=tot, pooled_decided_delta_rmse=float(np.sqrt((base + tot) / nd_all) - np.sqrt(base / nd_all)),
                              pooled_body_delta_sse=sum(x['body_delta_sse'] for x in per), pooled_trim20_sse=sum(x['trim20_sse'] for x in per),
                              decided_lower=sum(x['decided_delta_sse'] < 0 for x in per))
        # LOPO-chosen m10 weight per held-out pair against the production chain (m3 at 0.25)
        per = []
        for i, F in enumerate(folds):
            jw = GRID.index(res['m10']['lopo'][i]['w_chosen']); d = F['se']['m10'][jw] - F['se']['m3'][W25]; dec = F['masks']['decided']
            per.append(dict(fold=F['fold'], w_chosen=GRID[jw], decided_delta_sse=float(d[dec].sum()), body_delta_sse=float(d[F['masks']['body']].sum()), **trims(d[dec])))
        tot = sum(x['decided_delta_sse'] for x in per); base = float(sse['m3']['decided'][:, W25].sum()); nd_all = int(n['decided'].sum())
        cmp['m10_lopo_vs_production_m3_025'] = dict(per_fold=per, pooled_decided_delta_sse=tot, pooled_decided_delta_rmse=float(np.sqrt((base + tot) / nd_all) - np.sqrt(base / nd_all)),
                                                    pooled_body_delta_sse=sum(x['body_delta_sse'] for x in per), pooled_trim20_sse=sum(x['trim20_sse'] for x in per),
                                                    decided_lower=sum(x['decided_delta_sse'] < 0 for x in per))
        res['comparisons'] = cmp
    # supplementary, NOT part of the registered gate: forward-in-time legs on the same saved chain deltas
    # (half-year 1-6 -> 7-12 and expanding window months 1..k-1 -> k, k = 4..12); weight chosen on decided SSE of the
    # training months, scored on the held-out months against w = 0.25 of the same member.
    month = {F['name']: chain_df.filter(pl.col('month').is_in([F['fold'], F['fold'] + 6]))['month'].to_numpy() for F in folds}
    def leg(m, train_months, test_months):
        tr = np.zeros(len(GRID)); te = np.zeros(len(GRID)); n_te = 0
        for F in folds:
            dec = F['masks']['decided']; mo = month[F['name']]
            a = dec & np.isin(mo, train_months); b = dec & np.isin(mo, test_months); n_te += int(b.sum())
            tr += F['se'][m][:, a].sum(axis=1); te += F['se'][m][:, b].sum(axis=1)
        jw = int(np.argmin(tr))
        return dict(w_chosen=GRID[jw], n_decided_test=n_te, test_delta_sse=float(te[jw] - te[W25]),
                    test_delta_rmse=float(np.sqrt(te[jw] / n_te) - np.sqrt(te[W25] / n_te)), test_w_star=GRID[int(np.argmin(te))])
    res['supplementary_forward_legs'] = {m: dict(half_year_1to6_to_7to12=leg(m, list(range(1, 7)), list(range(7, 13))),
                                                 expanding={f'month_{k}': leg(m, list(range(1, k)), [k]) for k in range(4, 13)}) for m in members}
    res['verdict'] = dict(primary='m10 LOPO (registered rule)', gate_pass=res['m10']['gate']['pass'] if 'm10' in members else None,
                          null_by_threshold=res['m10']['null_by_threshold'] if 'm10' in members else None)
    p = root / RESULT; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1)); print('WROTE', p, flush=True)
    for m in members:
        r = res[m]
        print(m, 'w*', r['global_w_star'], 'per-fold', r['per_fold_w_star'], 'LOPO w', [x['w_chosen'] for x in r['lopo']],
              'pooled decided dRMSE %.4f s' % r['pooled_lopo_decided_delta_rmse'], 'gate', r['gate'], 'null', r['null_by_threshold'], flush=True)
    if 'comparisons' in res:
        for k in ('m10_025_vs_m3_025', 'm10_lopo_vs_production_m3_025'):
            c = res['comparisons'][k]; print(k, 'pooled decided dRMSE %.4f s' % c['pooled_decided_delta_rmse'], 'lower', c['decided_lower'], '/ 6', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['compute', 'evaluate'])
    p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--folds', default='1,2,3,4,5,6'); p.add_argument('--members', default='m3,m10')
    p.add_argument('--threads', type=int, default=4); a = p.parse_args()
    if not 1 <= a.threads <= 4: raise ValueError('1..4 threads on the Mac (registered)')
    members = tuple(m for m in MEMBERS if m in a.members.split(','))
    if a.action == 'compute': compute(a.root.resolve(), [int(x) for x in a.folds.split(',')], members, a.threads)
    else: evaluate(a.root.resolve(), members)


if __name__ == '__main__':
    main()
