"""E212: optimal weight of the ordered-trajectory NN member (E203) for a 10-seed mean.

Analytic stage on 2025 out-of-fold rows only. No fitting, no uploads.

Chain control: data/e203_pre_e81_20261007/predictions.parquet (column `ordered` =
production chain pre-E81 with the 3-seed ordered member at weight 0.25, `joint` = the
same chain with the PLE member at 0.25). Production arithmetic (e195/e203):
    regular = (1-sum(W))*(.65*cat+.35*lgb) + sum(W[k]*anchor_k) + 0.25*NN,  W=[1/30,.1,.1,.1,.1]
so a weight w for the NN member means regular(w) = (1-w)*B + w*NN with
    B = (regular(0.25) - 0.25*NN3)/0.75   (the other members, proportionally).
Downstream stages (E141/E64/LSZH) are approximated as an additive shift around premix:
    chain(w, NN_n) ~= ordered + g*[(w-0.25)*(NN_n - B)],   g = per-row pass-through.
Seed noise: NN_n = mu + eps_n, Var(eps_n) = s^2/n, s^2 = unbiased per-row seed variance
from the 3 seeds (ddof=1, in seconds^2 via sd of the fold). With m3 the observed 3-seed
mean, D_hat(w) = (y - ordered) - g*(w-0.25)*(m3 - B) = D(w) - g*w*eps_3, hence
    E[(y - chain(w, NN_n))^2] = D_hat(w)^2 + (g*w)^2 * s^2 * (1/n - 1/3).
Pre-registered rule (journal line): for each held-out pair k, choose w_k on the grid
0.05..0.60 (step 0.05) minimising the pooled decided-row expected SSE at n=10 over the
other five pairs (masked pass-through variant is primary); gate on the held-out pair.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
for k, v in {'POLARS_MAX_THREADS': '4', 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2'}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_pre_e81_comparison as A  # aligned(), WEIGHTS, ANCHORS, ID, Y

PROTOCOL = 'E212-nn-weight-10seed-analytic-lopo-decided-grid005-v1'
CHAIN = 'data/e203_pre_e81_20261007/predictions.parquet'
GRID = np.round(np.arange(0.05, 0.6001, 0.05), 2)
NS = (3, 10, 10 ** 9)  # 10**9 stands for n = infinity
OUT = Path('analysis/e212_nn_weight_10seed_20261008')


def traj_dir(fold):
    base = 'data/e197_trajectory_20261007' if fold in (1, 4) else 'data/e202_trajectory_20261007'
    return Path(base) / f'fold_{fold}_{fold + 6}' / 'ordered'


def load_fold(chain, fold):
    name = f'fold_{fold}_{fold + 6}'
    f = chain.filter(pl.col('month').is_in([fold, fold + 6])).sort(A.ID)
    ids = f[A.ID].to_numpy(); y = f[A.Y].to_numpy().astype(float); mo = f['month'].to_numpy()
    cat = A.aligned(f'data/cat_extra175_20261006/merged/{name}.npz', ids, y.astype(np.int32), mo)['reg2']
    lgb = A.aligned(f'data/lgb_fin1b_oof_20261005/{name}/complete/prediction.npz', ids, y.astype(np.int32), mo)['reg2']
    anchors = [A.aligned(f'data/cv_e130{x}/{name}.npz', ids)['reg2'] for x in A.ANCHORS]
    ple = A.aligned(f'data/e190_numemb_20261005/{name}/prediction.npz', ids, y.astype(np.int32), mo)['member']
    td = traj_dir(fold)
    z = A.aligned(td / 'prediction.npz', ids, y.astype(np.int32), mo)
    nn3 = z['member']
    # per-seed arrays are in the producer's row order (prediction.npz mvt_id); re-align identically
    with np.load(td / 'prediction.npz') as zz: src = zz['mvt_id']
    s = np.argsort(src); ix = s[np.searchsorted(src[s], ids)]
    if not np.array_equal(src[ix], ids): raise ValueError('seed alignment')
    seeds = np.stack([np.load(td / f'seed_{i}.npy')[ix].astype(np.float64) for i in (0, 1, 2)])
    qb = f'data/mlp_split_20261005/inputs/{name}'
    sd = float(json.load(open(f'{qb}/manifest.json'))['sd'])
    base_p = np.load(f'{qb}/base_p.npy')[ix]
    recon = seeds.mean(axis=0) * sd + base_p
    if not np.allclose(recon, nn3, atol=1e-6): raise ValueError(f'member reconstruction failed fold {fold}')
    w = A.WEIGHTS[:-1]
    regular = (1 - sum(A.WEIGHTS)) * (.65 * cat + .35 * lgb) + sum(wk * g for wk, g in zip(w, anchors))
    reg25 = regular + 0.25 * nn3
    regple = regular + 0.25 * ple
    B = regular / 0.75
    s2 = (seeds * sd).var(axis=0, ddof=1)  # unbiased per-row seed variance, s^2
    ordered = f['ordered'].to_numpy(); joint = f['joint'].to_numpy()
    ok = f['evaluation_mask'].to_numpy()
    airport = f['ADEP_mvt'].to_numpy(); nm = f['has_nm'].to_numpy(); D = f['D'].to_numpy()
    masks = dict(included=ok, body=ok & (y <= 7200), decided=ok & ~((airport == 'LIRF') & ~nm & (D >= 14400)))
    # pass-through of a premix change through the stages: compare ordered-joint with the premix difference
    dpre = reg25 - regple; dch = ordered - joint
    big = np.abs(dpre) >= 2.0
    override = big & (np.abs(dch) < 0.5)
    slope = float((dch[ok & big] * dpre[ok & big]).sum() / (dpre[ok & big] ** 2).sum())
    g_masked = np.where(override, 0.0, 1.0)
    return dict(fold=fold, name=name, y=y, ordered=ordered, m3=nn3, B=B, s2=s2, masks=masks, g=g_masked,
                diag=dict(rows=int(len(ids)), premix_vs_ordered_rms=float(np.sqrt(np.mean((reg25 - ordered)[ok] ** 2))),
                          premix_change_rms=float(np.sqrt(np.mean(dpre[ok] ** 2))),
                          chain_change_rms=float(np.sqrt(np.mean(dch[ok] ** 2))),
                          passthrough_slope=slope, override_share=float(override[ok].mean()),
                          seed_rms_sec=float(np.sqrt(s2[ok].mean())), nn_minus_B_rms=float(np.sqrt(np.mean((nn3 - B)[ok] ** 2))),
                          sd=sd))


def per_row_terms(F, w, n, passthrough):
    """Expected per-row squared error at weight w with an n-seed mean (vectorised)."""
    g = F['g'] if passthrough == 'masked' else np.ones_like(F['g'])
    Dh = (F['y'] - F['ordered']) - g * (w - 0.25) * (F['m3'] - F['B'])
    return Dh ** 2 + (g * w) ** 2 * F['s2'] * (1.0 / n - 1.0 / 3.0)


def sse_table(folds, passthrough):
    """folds x grid x n -> dict mask -> array[fold, grid, n] of expected SSE."""
    out = {m: np.zeros((len(folds), len(GRID), len(NS))) for m in ('included', 'body', 'decided')}
    for i, F in enumerate(folds):
        for j, w in enumerate(GRID):
            for k, n in enumerate(NS):
                t = per_row_terms(F, w, n, passthrough)
                for m in out: out[m][i, j, k] = float(t[F['masks'][m]].sum())
    return out


def evaluate(folds, passthrough):
    T = sse_table(folds, passthrough)
    i25 = int(np.where(np.isclose(GRID, 0.25))[0][0]); k10 = NS.index(10)
    cnt = {m: np.array([int(F['masks'][m].sum()) for F in folds]) for m in T}
    pooled = {m: T[m].sum(axis=0) for m in T}
    glob = {f'n{"inf" if n == 10 ** 9 else n}': dict(
        w_star=float(GRID[int(np.argmin(pooled['decided'][:, k]))]),
        rmse_decided_at_w={f'{w:.2f}': float(np.sqrt(pooled['decided'][j, k] / cnt['decided'].sum())) for j, w in enumerate(GRID)},
        delta_rmse_wstar_vs_025=float(np.sqrt(pooled['decided'][:, k].min() / cnt['decided'].sum()) - np.sqrt(pooled['decided'][i25, k] / cnt['decided'].sum())))
        for k, n in enumerate(NS)}
    # leave-one-pair-out at n=10 (pre-registered rule)
    lopo = []; sum_body = 0.0; sum_trim20 = 0.0; sum_decided = 0.0
    for i, F in enumerate(folds):
        train = [j for j in range(len(folds)) if j != i]
        jw = int(np.argmin(T['decided'][train, :, k10].sum(axis=0))); wk = float(GRID[jw])
        d = per_row_terms(F, wk, 10, passthrough) - per_row_terms(F, 0.25, 10, passthrough)
        dec = F['masks']['decided']; body = F['masks']['body']
        ds = np.sort(d[dec]); trim20 = float(ds[20:-20].sum())
        dd = float(d[dec].sum()); db = float(d[body].sum())
        n_dec = int(dec.sum()); base = T['decided'][i, i25, k10]
        lopo.append(dict(fold=F['fold'], w_chosen=wk, decided_delta_sse=dd, decided_delta_rmse=float(np.sqrt((base + dd) / n_dec) - np.sqrt(base / n_dec)),
                         body_delta_sse=db, trim20_delta_sse=trim20, n_decided=n_dec))
        sum_body += db; sum_trim20 += trim20; sum_decided += dd
    lower = sum(1 for r in lopo if r['decided_delta_sse'] < 0)
    f28 = next(r for r in lopo if r['fold'] == 2); f612 = next(r for r in lopo if r['fold'] == 6)
    n_dec_all = int(cnt['decided'].sum()); base_all = float(pooled['decided'][i25, k10])
    pooled_delta_rmse = float(np.sqrt((base_all + sum_decided) / n_dec_all) - np.sqrt(base_all / n_dec_all))
    gate = dict(decided_lower=lower, decided_lower_ge4=lower >= 4, pooled_body_lt0=sum_body < 0, pooled_trim20_lt0=sum_trim20 < 0,
                fold2_not_worse=f28['decided_delta_sse'] <= 0, fold6_not_worse=f612['decided_delta_sse'] <= 0)
    gate['pass'] = all(v for k, v in gate.items() if k != 'decided_lower')
    return dict(passthrough=passthrough, global_fit=glob, lopo=lopo, pooled_decided_delta_sse=sum_decided,
                pooled_decided_delta_rmse=pooled_delta_rmse, pooled_body_delta_sse=sum_body, pooled_trim20_delta_sse=sum_trim20,
                null_by_threshold=abs(pooled_delta_rmse) < 0.02, gate=gate,
                per_fold_wstar_n10={F['name']: float(GRID[int(np.argmin(T['decided'][i, :, k10]))]) for i, F in enumerate(folds)},
                per_fold_wstar_n3={F['name']: float(GRID[int(np.argmin(T['decided'][i, :, 0]))]) for i, F in enumerate(folds)})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=OUT); a = p.parse_args()
    chain = pl.read_parquet(CHAIN)
    if chain.height != 2085047: raise ValueError('chain population')
    folds = [load_fold(chain, f) for f in (1, 2, 3, 4, 5, 6)]
    for F in folds: print(F['name'], json.dumps(F['diag']), flush=True)
    res = dict(protocol=PROTOCOL, chain=CHAIN, grid=[float(w) for w in GRID], n_values=['3', '10', 'inf'],
               primary='masked', diagnostics={F['name']: F['diag'] for F in folds},
               masked=evaluate(folds, 'masked'), full=evaluate(folds, 'full'))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'result.json').write_text(json.dumps(res, indent=1))
    for v in ('masked', 'full'):
        r = res[v]
        print(v, 'w*', {k: x['w_star'] for k, x in r['global_fit'].items()}, 'LOPO pooled dRMSE(n=10) %.4f s' % r['pooled_decided_delta_rmse'],
              'gate', r['gate'], 'null', r['null_by_threshold'], flush=True)


if __name__ == '__main__':
    main()
