"""E215: exact chain recomputation of the 12-epoch ordered member against the 8-epoch member at the
final NN weight 0.35 (anchors scaled 0.65/0.75). Registered in HYPOTHESES_LOG №428.

Per 2025 month-pair fold the production pre-E81 chain of E203 (`cv_subset`, via the E212 machinery
`e212_chain_weights_20261008.load_fold/chain`) is recomputed with
    regular(0.35) = (0.65/0.75) * [(1-sum(W))*(.65*cat+.35*lgb) + sum_k W_k*anchor_k] + 0.35*NN
for two members: `m3` = the original 8-epoch 3-seed member (E197 folds 1/4, E202 folds 2/3/5/6) and
`m12` = the 12-epoch 3-seed member (data/e215_epochs12_20261008, Home,
scripts/e215_epochs12_folds_20261008.py; its `member3_8ep` must equal the original member exactly).
Control: the `m3` chain at 0.35 must reproduce the E212 saved delta (data/e212_chain_20261008,
grid 0.35) on every row; at 0.25 the E212 machinery itself reproduced `ordered` exactly.

Gate (all must hold, decided rows = E203 evaluation mask minus the LIRF/no-NM/D>=14400 rows,
minus rows whose feasibility changes between the two members):
  decided dSSE<0 in >=4/6 pairs; pooled decided, body (y<=7200) and trim20 (decided rows, -20/-20
  per pair) <0; pairs 2+8 and 6+12 not worse (<=0); per-month dSSE<0 in >=10 of 12 months and both
  half-years (1-6, 7-12) <0.
Outputs (Mac, small): data/e215_chain_20261008/fold_*/chain_e215.npz (int32 deltas vs `ordered`)
and analysis/e215_nn_epochs_20261008/result.json. No fitting, no uploads, no edits to the final assembly.
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

PROTOCOL = 'E215-ordered-member-epochs12-vs-8-3seed-6pair-chain-w035-v1'
W_FINAL = 0.35
J35 = W.GRID.index(W_FINAL)
EP12 = 'data/e215_epochs12_20261008'
OUT = 'data/e215_chain_20261008'
RESULT = 'analysis/e215_nn_epochs_20261008/result.json'
MEMBERS = ('m3', 'm12')
HALVES = {'h1': list(range(1, 7)), 'h2': list(range(7, 13))}


def load_m12(root, F):
    p = root / EP12 / F['name'] / 'ordered' / 'prediction_12ep.npz'
    z = A.aligned(p, F['ids'], F['y'], F['f']['month'].to_numpy())
    if not np.array_equal(z['member3_8ep'], F['mem']['m3']): raise ValueError('12-epoch file member3_8ep differs from the original member ' + F['name'])
    if z['per_seed_sec'].shape[0] != 3 or not np.isfinite(z['member3_12ep']).all(): raise ValueError('12-epoch file incomplete ' + F['name'])
    c = json.loads((p.parent / 'complete.json').read_text())
    if (c['protocol'] != PROTOCOL or c['fold'] != F['fold'] or c['arm'] != 'ordered' or c['seeds'] != [0, 1, 2] or c['epochs'] != 12
            or not c['original_member_exact'] or not c['bins_equal_original'] or any(len(f['losses']) != 12 for f in c['fits'])): raise ValueError('12-epoch receipt ' + F['name'])
    if c['files']['prediction_12ep.npz'] != A.sha(p): raise ValueError('12-epoch prediction SHA differs from its receipt ' + F['name'])
    m = z['member3_12ep']
    return (m.astype(np.float64) if m.dtype != np.float64 else m), c, A.sha(p)


def compute(root, folds, threads):
    out = root / OUT; out.mkdir(parents=True, exist_ok=True)
    chain_df = pl.read_parquet(root / W.CHAIN)
    if chain_df.height != 2085047: raise ValueError('chain population')
    spec = A.read(root / A.BUNDLE / 'manifest.json')['spec']; natural = pl.read_parquet(root / A.BUNDLE / 'natural_features.parquet')
    cols = [A.ID, A.Y, 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'LOBT_flt', 'd_mvt_lobt', 'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_eobt']
    feature = pl.read_parquet(root / 'data/features_fin1b.parquet', columns=cols).filter(pl.col('split') == 'train').sort(A.ID)
    ra, rb = A.rules(root)
    for fold in folds:
        name = f'fold_{fold}_{fold + 6}'; fd = out / name
        if (fd / 'chain_e215.npz').exists(): print('DONE ALREADY', name, flush=True); continue
        F = W.load_fold(root, chain_df, feature, natural, fold, ('m3',)); F['mem']['m12'], receipt, sha12 = load_m12(root, F)
        with np.load(root / W.OUT / name / 'chain_m3.npz') as z:
            if not np.array_equal(z['mvt_id'], F['ids']) or z['grid'][J35] != W_FINAL: raise ValueError('E212 chain file alignment ' + name)
            e212_delta, e212_unstable = z['delta'][J35].copy(), z['unstable'][J35].copy()
        delta = {}; unstable = {}; secs = {}
        for m in MEMBERS:
            t = time.monotonic(); aug, u = W.chain(root, spec, F, ra, rb, W_FINAL, m, threads); secs[m] = time.monotonic() - t
            dd = aug - F['ordered']
            if np.abs(dd).max() >= 2 ** 31: raise ValueError('delta overflow')
            delta[m] = dd.astype(np.int32); unstable[m] = u
            if m == 'm3':
                if not np.array_equal(delta[m], e212_delta) or not np.array_equal(u, e212_unstable): raise ValueError(f'CONTROL FAILED {name}: m3 chain at 0.35 differs from the E212 saved delta')
                print('CONTROL OK', name, 'E212 m3 delta at 0.35 reproduced on all', len(aug), 'rows', flush=True)
            print(name, m, f'w={W_FINAL:.2f}', 'rms(delta vs ordered) %.2f s' % float(np.sqrt(np.mean(dd[~F['excluded']].astype(float) ** 2))),
                  'unstable', int(u.sum()), '%.1f s' % secs[m], flush=True)
        fd.mkdir(exist_ok=True); tmp = fd / '.chain_e215.partial.npz'
        np.savez_compressed(tmp, mvt_id=F['ids'], w=np.array(W_FINAL), delta_m3=delta['m3'], delta_m12=delta['m12'], unstable_m3=unstable['m3'], unstable_m12=unstable['m12'],
                            excluded_e203=F['excluded'], member_m3_sec=F['mem']['m3'].astype(np.float32), member_m12_sec=F['mem']['m12'].astype(np.float32))
        tmp.rename(fd / 'chain_e215.npz')
        A.save(fd / 'chain_e215.json', dict(protocol=PROTOCOL, fold=fold, w=W_FINAL, threads=threads, seconds=secs, control_exact=True,
                                            m3_source=str(W.traj_dir(fold) / 'prediction.npz'), m3_sha256=A.sha(root / W.traj_dir(fold) / 'prediction.npz'),
                                            m12_source=f'{EP12}/{name}/ordered/prediction_12ep.npz', m12_sha256=sha12, m12_receipt_rms_8ep_vs_12ep_sec=receipt['rms_8ep_vs_12ep_sec'],
                                            e212_chain_m3_sha256=A.sha(root / W.OUT / name / 'chain_m3.npz'), chain_sha256=A.sha(root / W.CHAIN), self_sha256=A.sha(Path(__file__))))
        print('COMPLETE', name, flush=True)
        del F
    print('ALL REQUESTED FOLDS COMPLETE', flush=True)


def evaluate(root):
    chain_df = pl.read_parquet(root / W.CHAIN).sort(A.ID)
    folds = []
    for fold in W.FOLDS:
        name = f'fold_{fold}_{fold + 6}'; fd = root / OUT / name
        old = chain_df.filter(pl.col('month').is_in([fold, fold + 6]))
        ids = old[A.ID].to_numpy(); y = old[A.Y].to_numpy().astype(np.float64); ordered = old['ordered'].to_numpy().astype(np.float64)
        ok = old['evaluation_mask'].to_numpy(); airport = old['ADEP_mvt'].to_numpy(); nm = old['has_nm'].to_numpy(); D = old['D'].to_numpy(); month = old['month'].to_numpy()
        with np.load(fd / 'chain_e215.npz') as z:
            if not np.array_equal(z['mvt_id'], ids) or float(z['w']) != W_FINAL or not np.array_equal(z['excluded_e203'], ~ok): raise ValueError('chain file alignment ' + name)
            pred = {m: ordered + z[f'delta_{m}'].astype(np.float64) for m in MEMBERS}; unstable = z['unstable_m3'] | z['unstable_m12']
            member = {m: z[f'member_{m}_sec'].astype(np.float64) for m in MEMBERS}
        common = ok & ~unstable
        masks = dict(included=common, body=common & (y <= 7200), decided=common & ~((airport == 'LIRF') & ~nm & (D >= 14400)))
        se = {m: (pred[m] - y) ** 2 for m in MEMBERS}; mse = {m: (member[m] - y) ** 2 for m in MEMBERS}
        folds.append(dict(fold=fold, name=name, y=y, se=se, mse=mse, masks=masks, month=month, n_rows=len(ids), e203_included=int(ok.sum()),
                          extra_excluded_feasibility=int((ok & unstable).sum()), member_rms_diff_sec=float(np.sqrt(np.mean((member['m12'] - member['m3']) ** 2)))))
    res = dict(protocol=PROTOCOL, w=W_FINAL, members=list(MEMBERS), comparison='m12 (12 epochs, seeds 0-2) minus m3 (8 epochs, seeds 0-2), both at NN weight 0.35',
               populations={F['name']: dict(rows=F['n_rows'], e203_included=F['e203_included'], extra_excluded_feasibility=F['extra_excluded_feasibility'],
                                             **{k: int(v.sum()) for k, v in F['masks'].items()}) for F in folds})
    per = []; pooled = dict(decided=0.0, body=0.0, trim20=0.0, trim3=0.0, included=0.0); base_dec = 0.0; n_dec = 0
    for F in folds:
        d = F['se']['m12'] - F['se']['m3']; dec = F['masks']['decided']; body = F['masks']['body']; t = W.trims(d[dec])
        dd = float(d[dec].sum()); db = float(d[body].sum()); base = float(F['se']['m3'][dec].sum()); nd = int(dec.sum())
        dm = F['mse']['m12'] - F['mse']['m3']
        per.append(dict(fold=F['fold'], name=F['name'], n_decided=nd, decided_delta_sse=dd, decided_delta_rmse=float(np.sqrt((base + dd) / nd) - np.sqrt(base / nd)),
                        rmse_decided_m3=float(np.sqrt(base / nd)), body_delta_sse=db, included_delta_sse=float(d[F['masks']['included']].sum()), **t,
                        member_rmse_decided_m3=float(np.sqrt(F['mse']['m3'][dec].mean())), member_rmse_decided_m12=float(np.sqrt(F['mse']['m12'][dec].mean())),
                        member_delta_rmse_decided=float(np.sqrt(F['mse']['m12'][dec].mean()) - np.sqrt(F['mse']['m3'][dec].mean())), member_rms_diff_sec=F['member_rms_diff_sec'],
                        member_delta_sse_decided=float(dm[dec].sum())))
        pooled['decided'] += dd; pooled['body'] += db; pooled['trim20'] += t['trim20_sse']; pooled['trim3'] += t['trim3_sse']; pooled['included'] += float(d[F['masks']['included']].sum())
        base_dec += base; n_dec += nd
    pooled_drmse = float(np.sqrt((base_dec + pooled['decided']) / n_dec) - np.sqrt(base_dec / n_dec))
    months = {}
    for mo in range(1, 13):
        F = next(F for F in folds if mo in (F['fold'], F['fold'] + 6)); sel = F['masks']['decided'] & (F['month'] == mo)
        d = (F['se']['m12'] - F['se']['m3'])[sel]; base = float(F['se']['m3'][sel].sum()); n = int(sel.sum())
        months[str(mo)] = dict(n_decided=n, delta_sse=float(d.sum()), delta_rmse=float(np.sqrt((base + d.sum()) / n) - np.sqrt(base / n)), rmse_m3=float(np.sqrt(base / n)),
                               member_delta_sse=float((F['mse']['m12'] - F['mse']['m3'])[sel].sum()))
    halves = {}
    for h, ms in HALVES.items():
        d = sum(months[str(m)]['delta_sse'] for m in ms); base = sum(months[str(m)]['rmse_m3'] ** 2 * months[str(m)]['n_decided'] for m in ms); n = sum(months[str(m)]['n_decided'] for m in ms)
        halves[h] = dict(months=ms, n_decided=n, delta_sse=d, delta_rmse=float(np.sqrt((base + d) / n) - np.sqrt(base / n)))
    lower = sum(1 for x in per if x['decided_delta_sse'] < 0); months_lower = sum(1 for v in months.values() if v['delta_sse'] < 0)
    f2 = next(x for x in per if x['fold'] == 2); f6 = next(x for x in per if x['fold'] == 6)
    gate = dict(decided_lower=lower, decided_lower_ge4=lower >= 4, pooled_decided_lt0=pooled['decided'] < 0, pooled_body_lt0=pooled['body'] < 0, pooled_trim20_lt0=pooled['trim20'] < 0,
                fold2_not_worse=f2['decided_delta_sse'] <= 0, fold6_not_worse=f6['decided_delta_sse'] <= 0, months_lower=months_lower, months_lower_ge10=months_lower >= 10,
                h1_lt0=halves['h1']['delta_sse'] < 0, h2_lt0=halves['h2']['delta_sse'] < 0)
    gate['pass'] = all(v for k, v in gate.items() if k not in ('decided_lower', 'months_lower'))
    member_total = dict(delta_sse_decided=sum(x['member_delta_sse_decided'] for x in per),
                        rms_diff_sec_mean=float(np.mean([x['member_rms_diff_sec'] for x in per])))
    res.update(per_fold=per, pooled_delta_sse=pooled, pooled_decided_delta_rmse=pooled_drmse, n_decided=n_dec, rmse_decided_m3_pooled=float(np.sqrt(base_dec / n_dec)),
               per_month=months, half_years=halves, member_only=member_total, gate=gate, verdict=dict(gate_pass=gate['pass'], pooled_decided_delta_rmse=pooled_drmse))
    p = root / RESULT; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res, indent=1)); print('WROTE', p, flush=True)
    print('pooled decided dRMSE %.4f s' % pooled_drmse, 'pairs lower', lower, '/6', 'months lower', months_lower, '/12', 'halves', {h: round(v['delta_rmse'], 4) for h, v in halves.items()},
          'body', pooled['body'], 'trim20', pooled['trim20'], flush=True)
    for x in per: print(x['name'], 'decided dSSE %.3e dRMSE %+.4f' % (x['decided_delta_sse'], x['decided_delta_rmse']), 'member dRMSE %+.3f' % x['member_delta_rmse_decided'], 'rms diff %.1f' % x['member_rms_diff_sec'], flush=True)
    for mo, v in months.items(): print('month', mo, 'dSSE %.3e dRMSE %+.4f' % (v['delta_sse'], v['delta_rmse']), flush=True)
    print('GATE', gate, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['compute', 'evaluate'])
    p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--folds', default='1,2,3,4,5,6'); p.add_argument('--threads', type=int, default=4); a = p.parse_args()
    if not 1 <= a.threads <= 4: raise ValueError('1..4 threads on the Mac (registered)')
    if a.action == 'compute': compute(a.root.resolve(), [int(x) for x in a.folds.split(',')], a.threads)
    else: evaluate(a.root.resolve())


if __name__ == '__main__':
    main()
