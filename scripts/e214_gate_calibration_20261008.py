"""E214: the (A) exact-SSE regime-probability calibration decided at chain level (journal №421).

Question. Journal №419 (A) closed the 11-direction premix calibration by its registered stop rule on all
rows, but a POST-HOC subpopulation (serviced rows with the saved (proba, values) reproducing pred) kept a
forward-stable −0.13…−0.19 s premix budget. Instead of arguing about populations, apply the calibration to
the regime probabilities of EVERY row at the premix and let the unchanged production chain decide.

Chain. The E203 pre-E81 chain exactly as replayed by E212 (scripts/e212_chain_weights_20261008.py, helpers of
scripts/e195_pre_e81_comparison.py): regular = (1-sum W)*(.65 cat + .35 lgb) + sum W_k anchor_k + .25 NN(3 seeds,
ordered), cached FIN1b probabilities (data/cv_preds_fin1b_hz), premix mixture with the LOBT ±3606 clip, E185
motion blend, E167 NM rules, E141 / E64 / E188 frozen stage models (data/e195_stage_models_20261007), E152 gate
0.95, 126 TabPFN rows protected (fallback to the saved prediction, excluded from every metric). Ends before E81
(not in the final) / LOBT lift / LSZH half, like E212. Control: theta = 0 must reproduce the `ordered` column of
data/e203_pre_e81_20261007/predictions.parquet on every row (max abs difference reported).

Calibration (identical to (A) except for where the premix is read). Basis J = 11: g = p0 (v0 - m) x
{4 disagreement buckets <600 / 600-1800 / 1800-3600 / >=3600 s, NM missing, LIRF} and h_k = p0 p_k (v_k - v0),
k = 1..5, with p = cached probabilities masked on infeasible candidates and renormalised, v = candidates with
v0 = the E203 regular after the E185 motion blend (the regular the chain actually mixes), m = sum_k p_k v_k
(the pure mixture, so the identity holds on all rows by construction; the LOBT clip / E167 are applied by the
chain afterwards). Ridge lambda = (240/0.1)^2 = 5.76e6, closed form theta = (Q + lambda I)^-1 c on the premix of
the TRAINING months only. Application: p' = clip(p + B theta, 0, 1) over feasible candidates, renormalised
(the chain needs a valid simplex; the number of clipped rows is reported). p' feeds everything downstream:
premix, E167, the p_nonreg stage feature, the E152 gate.

Fit populations (registered). PRIMARY = the chain's decided rows (evaluation mask and not (LIRF and no NM and
D >= 14400)) — the population the production chain serves and is judged on; it still contains 1 360 LIRF rows
without NM and D < 14400 (10.4 % of premix SSE). Secondary, reported with the same gate for information only:
`serviced` (decided and not (LIRF and no NM) — the (A) post-hoc choice) and `all` (every evaluated row, the (A)
main mask). Whatever the fit population, theta is applied to ALL rows and the chain is scored on its registered
populations.

Evaluation splits (theta refitted per split on its training months): leave-one-pair-out (train the other ten
months, score the held-out pair k & k+6); half-year forward (train months 1-6, score months 7-12); expanding
forward (train 1..k-1, score month k, k = 4..12). One chain replay per (population, split, fold): 1 control +
3 x (1 + 1 + 1|2) = 10 or 13 replays per fold, ~22 s each at 4 threads (E212 timing).

Gate (all must hold, primary population, chain-level SSE against `ordered`): LOPO decided SSE lower in >= 4/6
folds; pooled decided, body (y <= 7200) and trim20 (decided rows, 20 largest and 20 smallest per-row deltas
removed per fold) < 0; folds 2+8 and 6+12 not worse (<= 0); half-year forward pooled decided < 0; expanding
forward pooled decided < 0 with >= 6/9 months lower. Reported: chain dRMSE (s) on included = full served
population, decided and body; pass-through = chain dSSE / premix dSSE on the same rows (premix dSSE both exact
linear theta^T Q theta - 2 c^T theta and the actual clipped mixture).

Mac only, .venv/bin/python, <= 4 threads, read-only on data/ inputs. Outputs: data/e214_chain_20261008/
fold_*/replays.npz (int32 deltas vs `ordered`), analysis/e214_gate_calibration_20261008/{result.json,run.log}.
No fitting of any model, no uploads, no edits to other agents' files.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for k, v in {'POLARS_MAX_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '4',
             'OMP_WAIT_POLICY': 'PASSIVE', 'GOMP_SPINCOUNT': '0', 'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage'}.items():
    os.environ.setdefault(k, v)
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_pre_e81_comparison as A  # aligned(), rules(), feasibility(), mixture(), stage_predictions(), WEIGHTS, ANCHORS, ID, Y

PROTOCOL = 'E214-gate-calibration-chain-decided-v1'
CHAIN = 'data/e203_pre_e81_20261007/predictions.parquet'
OUT = 'data/e214_chain_20261008'
RESULT_DIR = 'analysis/e214_gate_calibration_20261008'
LAMBDA = (240.0 / 0.1) ** 2
BUCKETS = np.array([600.0, 1800.0, 3600.0])
J = 11
COLS = ['g*bucket<600', 'g*bucket600-1800', 'g*bucket1800-3600', 'g*bucket>=3600', 'g*nm_missing', 'g*LIRF',
        'h_schedule', 'h_IOBT', 'h_LOBT', 'h_EOBT', 'h_dayshift']
MONTHS = list(range(1, 13))
POPS = ('decided', 'serviced', 'all')
PRIMARY = 'decided'
FOLDS = (1, 2, 3, 4, 5, 6)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def traj_dir(fold):  # as in scripts/e212_chain_weights_20261008.py
    return Path('data/e197_trajectory_20261007' if fold in (1, 4) else 'data/e202_trajectory_20261007') / f'fold_{fold}_{fold + 6}' / 'ordered'


def e203_regular(r0, anchors, member):  # E203 / E195 arithmetic, bit-identical to the chain's `ordered` input
    return (1 - sum(A.WEIGHTS)) * r0 + sum(w * g for w, g in zip(A.WEIGHTS, [*anchors, member]))


def load_fold(root, chain, feature, fold, natural=None):
    name = f'fold_{fold}_{fold + 6}'
    f = feature.filter(pl.col('month').is_in([fold, fold + 6]))
    ids = f[A.ID].to_numpy(); y = f[A.Y].to_numpy(); mo = f['month'].to_numpy()
    old = chain.filter(pl.col('month').is_in([fold, fold + 6])).sort(A.ID)
    if any(not np.array_equal(old[k].to_numpy(), f[k].to_numpy()) for k in (A.ID, A.Y, 'month')):
        raise ValueError('E203 population changed')
    z = A.aligned(root / 'data/cv_preds_fin1b_hz' / (name + '.npz'), ids, y)
    cat = A.aligned(root / 'data/cat_extra175_20261006/merged' / (name + '.npz'), ids, y, mo)['reg2']
    lgb = A.aligned(root / 'data/lgb_fin1b_oof_20261005' / name / 'complete/prediction.npz', ids, y, mo)['reg2']
    anchors = [A.aligned(root / 'data' / ('cv_e130' + x) / (name + '.npz'), ids)['reg2'] for x in A.ANCHORS]
    motion = A.aligned(root / 'data/cv_e130motion' / (name + '.npz'), ids)['reg2']
    m3 = A.aligned(root / traj_dir(fold) / 'prediction.npz', ids, y, mo)['member']
    if not all(np.isfinite(v).all() for v in [cat, lgb, *anchors, motion, m3]):
        raise ValueError('nonfinite member')
    r = e203_regular(.65 * cat + (1 - .65) * lgb, anchors, m3)
    F = dict(name=name, fold=fold, f=f, ids=ids, y=y, month=mo, z=z, r=r, motion=motion,
             ordered=old['ordered'].to_numpy().astype(np.int64), excluded=~old['evaluation_mask'].to_numpy(),
             lobt=f['d_mvt_lobt'].to_numpy(), lirf=(f['ADEP_mvt'] == 'LIRF').to_numpy(),
             nm_missing=f['FLIGHT_ID_mvt'].is_null().to_numpy(), D=f['d_mvt_sched'].to_numpy())
    if natural is not None:
        d = f.select(A.ID).join(natural, on=A.ID, how='left', validate='1:1', maintain_order='left')
        if d[A.Y].null_count() or not np.array_equal(d[A.Y].to_numpy(), y):
            raise ValueError('natural feature alignment')
        F['d'] = d
    return F


def premix_state(F):
    """p (masked, renormalised), v after the E185 motion blend, m = sum p v: the mixture the chain computes."""
    z = F['z']; r = F['r']; f = F['f']
    v = z['values'].astype(float).copy(); v[:, 0] = r; v[:, -1] = r + 86400
    if np.any(A.feasibility(v, F['lobt']) != A.feasibility(z['values'], F['lobt'])):
        raise ValueError('feasibility changed by the E203 regular (unexpected: E212 found none)')
    p = z['proba'].astype(float).copy()
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=2e-7):
        raise ValueError('invalid cached probabilities')
    p = np.where(np.isnan(v), 0, p); p = p / p.sum(axis=1, keepdims=True)
    subset = (f['LOBT_flt'].is_null() & (f['ADEP_mvt'] != 'LIRF')).to_numpy(); w = .35 * 2 / 3
    r2 = r.copy(); r2[subset] = (1 - w) * r[subset] + w * F['motion'][subset]
    v[:, 0] = r2; v[:, -1] = r2 + 86400
    feas = ~np.isnan(v)
    vf = np.where(feas, v, 0.0)
    m = np.sum(p * vf, axis=1)
    return dict(p=p, v=vf, feas=feas, m=m)


def basis(F, S):
    p, vf, feas, m = S['p'], S['v'], S['feas'], S['m']
    p0 = p[:, 0]; v0 = vf[:, 0]
    g = p0 * (v0 - m)
    pn = 1.0 - p0
    dis = np.sum(p[:, 1:] * np.abs(vf[:, 1:] - v0[:, None]) * feas[:, 1:], axis=1)
    dis = np.where(pn > 1e-9, dis / np.maximum(pn, 1e-9), 0.0)
    b = np.digitize(dis, BUCKETS)
    D = np.zeros((len(m), J))
    for j in range(4):
        D[:, j] = g * (b == j)
    D[:, 4] = g * F['nm_missing']
    D[:, 5] = g * F['lirf']
    for k in range(1, 6):
        D[:, 5 + k] = p0 * p[:, k] * np.where(feas[:, k], vf[:, k] - v0, 0.0)
    e = F['y'].astype(np.float64) - m
    S.update(p0=p0, b=b, D=D, e=e)
    return D, e


def delta_p(theta, F, S):
    p, p0, feas, b = S['p'], S['p0'], S['feas'], S['b']
    s = np.zeros(len(p0))
    for j in range(4):
        s += theta[j] * (b == j)
    s += theta[4] * F['nm_missing'] + theta[5] * F['lirf']
    dp = -(s * p0)[:, None] * p
    dp[:, 0] += s * p0
    for k in range(1, 6):
        t = theta[5 + k] * p0 * p[:, k] * feas[:, k]
        dp[:, k] += t
        dp[:, 0] -= t
    return dp


def populations(F):
    inc = ~F['excluded']
    tail = F['lirf'] & F['nm_missing']
    decided = inc & ~(tail & (F['D'] >= 14400))
    return dict(decided=decided, serviced=decided & ~tail, all=inc, included=inc, body=inc & (F['y'] <= 7200))


def month_stats(D, e, month, masks):
    st = {}
    for name in POPS:
        st[name] = {}
        for mo in MONTHS:
            sel = (month == mo) & masks[name]
            Dm = D[sel]; em = e[sel]
            st[name][mo] = dict(Q=Dm.T @ Dm, c=Dm.T @ em, sse=float(em @ em), n=int(sel.sum()))
    return st


def agg(st, months):
    return (sum(st[m]['Q'] for m in months), sum(st[m]['c'] for m in months),
            sum(st[m]['sse'] for m in months), sum(st[m]['n'] for m in months))


def solve(Q, c, lam=LAMBDA):
    return np.linalg.solve(Q + lam * np.eye(len(c)), c)


def upstream_calibrated(z, r, motion, f, ia, ib, excluded, dp):
    """scripts/e195_pre_e81_comparison.upstream with one inserted step: p <- clip(p + dp) renormalised, right
    after the cached probabilities are masked and renormalised and before the first mixture. dp=None reproduces
    A.upstream exactly (asserted by the control)."""
    lobt = f['d_mvt_lobt'].to_numpy(); v = z['values'].astype(float).copy()
    v[:, 0] = r; v[:, -1] = r + 86400
    changes = np.any(A.feasibility(v, lobt) != A.feasibility(z['values'], lobt), axis=1) & ~excluded
    if changes.any(): raise ValueError(f'Feasibility exclusion failed on {changes.sum()} scored rows')
    p = z['proba'].astype(float).copy()
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=2e-7): raise ValueError('invalid cached probabilities')
    p = np.where(np.isnan(v), 0, p); p = p / p.sum(axis=1, keepdims=True)
    clipped = 0; floored = 0
    if dp is not None:
        q = p + dp
        clipped = int(np.any((q < 0) | (q > 1), axis=1).sum())
        q = np.clip(q, 0.0, 1.0); q = np.where(np.isnan(v), 0, q)
        # Technical repair (journal №422): the classifier never outputs p0 == 0, and the E167 rule a below zeroes the
        # schedule/day-shift classes, so a row whose regular probability was clipped to 0 would become empty (NaN).
        # Floor the regular probability at 1e-9 before renormalising; the number of floored rows is reported.
        floored = int((q[:, 0] < 1e-9).sum()); q[:, 0] = np.maximum(q[:, 0], 1e-9)
        q = q / q.sum(axis=1, keepdims=True)
        if not np.isfinite(q).all(): raise ValueError('calibrated probabilities not finite')
        p = q
    base = np.rint(A.mixture(p, v, lobt))
    subset = (f['LOBT_flt'].is_null() & (f['ADEP_mvt'] != 'LIRF')).to_numpy(); w = .35 * 2 / 3
    r = r.copy(); r[subset] = (1 - w) * r[subset] + w * motion[subset]
    v[:, 0] = r; v[:, -1] = r + 86400
    base[subset] = np.rint(A.mixture(p, v, lobt)[subset])
    p[ia, 1] = 0; p[ia, 5] = 0
    rest = np.clip(1 - p[ib, 1], 1e-9, None); p[ib] *= ((1 - .9) / rest)[:, None]; p[ib, 1] = .9
    p = np.where(np.isnan(v), 0, p); p /= p.sum(axis=1, keepdims=True)
    changed = ia | ib; base[changed] = np.rint(A.mixture(p, v, lobt)[changed])
    return base, p, (clipped, floored)


def chain(root, spec, F, ra, rb, dp, threads):
    base, p, clipped = upstream_calibrated(F['z'], F['r'], F['motion'], F['f'], np.isin(F['ids'], ra), np.isin(F['ids'], rb), F['excluded'], dp)
    _, aug = A.stage_predictions(root, F['name'], F['d'], base, p, spec, threads)
    aug[F['excluded']] = np.rint(F['z']['pred'][F['excluded']])
    if not np.isfinite(aug).all() or (aug < 0).any() or not np.array_equal(aug, np.rint(aug)): raise ValueError('invalid stage output')
    return aug.astype(np.int64), clipped


def legs_for_fold(fold):
    """(population-independent) list of (leg, key, train_months, eval_months) for the fold's replays."""
    k = fold; legs = [('lopo', f'pair{k}', [m for m in MONTHS if m not in (k, k + 6)], [k, k + 6]),
                      ('half', 'half', [1, 2, 3, 4, 5, 6], [k + 6])]
    for mo in (k, k + 6):
        if mo >= 4:
            legs.append(('expanding', f'm{mo}', list(range(1, mo)), [mo]))
    return legs


def compute(root, threads, folds):
    out = root / OUT; out.mkdir(parents=True, exist_ok=True)
    res_dir = root / RESULT_DIR; res_dir.mkdir(parents=True, exist_ok=True)
    chain_df = pl.read_parquet(root / CHAIN)
    if chain_df.height != 2085047: raise ValueError('chain population')
    cols = [A.ID, A.Y, 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'LOBT_flt', 'd_mvt_lobt', 'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_eobt']
    feature = pl.read_parquet(root / 'data/features_fin1b.parquet', columns=cols).filter(pl.col('split') == 'train').sort(A.ID)
    # ---- phase 1: premix basis and monthly statistics for every fold (small, kept in memory)
    log('phase 1: premix basis')
    stats = {name: {} for name in POPS}; premix = {}
    for fold in FOLDS:
        F = load_fold(root, chain_df, feature, fold)
        S = premix_state(F); D, e = basis(F, S); masks = populations(F)
        st = month_stats(D, e, F['month'], masks)
        for name in POPS:
            for mo in (fold, fold + 6):  # only this fold's months (the other ten entries of st are empty)
                stats[name][mo] = st[name][mo]
        premix[fold] = dict(p=S['p'], p0=S['p0'], feas=S['feas'], b=S['b'], v=S['v'], m=S['m'], D=D, e=e,
                            lirf=F['lirf'], nm_missing=F['nm_missing'], ids=F['ids'])
        log(F['name'], 'rows', len(F['ids']), 'premix rmse(all) %.2f' % np.sqrt(np.mean(e ** 2)),
            'bucket counts', [int((S['b'] == j).sum()) for j in range(4)])
        del F
    for name in POPS:
        if sorted(stats[name]) != MONTHS or any(stats[name][mo]['n'] <= 0 for mo in MONTHS): raise ValueError('monthly statistics incomplete: ' + name)
    log('monthly n (decided):', {mo: stats['decided'][mo]['n'] for mo in MONTHS})
    thetas = {}
    for name in POPS:
        for fold in FOLDS:
            for leg, key, train, ev in legs_for_fold(fold):
                if (name, key) not in thetas:
                    Q, c, _, _ = agg(stats[name], train)
                    thetas[(name, key)] = solve(Q, c)
    (res_dir / 'thetas.json').write_text(json.dumps({f'{n}|{k}': t.tolist() for (n, k), t in thetas.items()}, indent=1))
    log('phase 1 done: %d thetas' % len(thetas))
    # ---- phase 2: chain replays
    spec = A.read(root / A.BUNDLE / 'manifest.json')['spec']
    natural = pl.read_parquet(root / A.BUNDLE / 'natural_features.parquet')
    ra, rb = A.rules(root)
    for fold in folds:
        name = f'fold_{fold}_{fold + 6}'; fd = out / name; fd.mkdir(exist_ok=True)
        if (fd / 'replays.npz').exists():
            log('DONE ALREADY', name); continue
        F = load_fold(root, chain_df, feature, fold, natural)
        P = premix[fold]
        if not np.array_equal(P['ids'], F['ids']): raise ValueError('phase-1/2 id mismatch')
        t0 = time.monotonic()
        ctrl, clipped = chain(root, spec, F, ra, rb, None, threads)
        maxdiff = int(np.abs(ctrl - F['ordered']).max())
        if maxdiff != 0: raise ValueError(f'CONTROL FAILED {name}: max abs diff {maxdiff}')
        log('CONTROL OK', name, 'ordered reproduced on all', len(ctrl), 'rows, max abs diff 0, %.1f s' % (time.monotonic() - t0))
        keys = []; deltas = []; dm_premix = []; clip_counts = []; floor_counts = []; secs = []; cache = {}; reused = []
        for pop in POPS:
            for leg, key, train, ev in legs_for_fold(fold):
                th = thetas[(pop, key)]
                dp = delta_p(th, F, P)
                t0 = time.monotonic()
                sig = th.tobytes()
                if sig in cache:  # identical theta (e.g. `all` == `decided` up to one row) -> identical replay, reused verbatim
                    aug, (clipped, floored) = cache[sig]; reused.append(f'{pop}|{leg}|{key}')
                else:
                    aug, (clipped, floored) = chain(root, spec, F, ra, rb, dp, threads); cache[sig] = (aug, (clipped, floored))
                secs.append(time.monotonic() - t0)
                dd = aug - F['ordered']
                if np.abs(dd).max() >= 2 ** 31: raise ValueError('delta overflow')
                # actual premix change (clipped simplex, same floor), unclipped by LOBT: the quantity theta was fitted on
                q = np.clip(P['p'] + dp, 0, 1) * P['feas']; q[:, 0] = np.maximum(q[:, 0], 1e-9); q = q / q.sum(1, keepdims=True)
                dm = np.sum(q * P['v'], axis=1) - P['m']
                keys.append(f'{pop}|{leg}|{key}'); deltas.append(dd.astype(np.int32)); dm_premix.append(dm.astype(np.float32)); clip_counts.append(clipped); floor_counts.append(floored)
                ev_rows = np.isin(F['month'], ev) & ~F['excluded']
                log(name, pop, leg, key, 'rms(chain delta) %.2f s' % float(np.sqrt(np.mean(dd[ev_rows].astype(float) ** 2))),
                    'rms(premix dm) %.2f s' % float(np.sqrt(np.mean(dm[ev_rows] ** 2))), 'clipped', clipped, 'p0 floored', floored, '%.1f s' % secs[-1])
        tmp = fd / '.replays.partial.npz'
        np.savez_compressed(tmp, mvt_id=F['ids'], month=F['month'], keys=np.array(keys), delta=np.stack(deltas), dm_premix=np.stack(dm_premix),
                            clipped=np.array(clip_counts), floored=np.array(floor_counts), excluded=F['excluded'], control_max_abs_diff=np.array(maxdiff))
        tmp.rename(fd / 'replays.npz')
        A.save(fd / 'replays.json', dict(protocol=PROTOCOL, fold=fold, keys=keys, seconds=secs, threads=threads, control_max_abs_diff=maxdiff, reused_identical_theta=reused,
                                         chain_sha256=A.sha(root / CHAIN), self_sha256=A.sha(Path(__file__))))
        log('COMPLETE', name, 'replays', len(keys), 'total %.0f s' % sum(secs))
        del F
    log('ALL REQUESTED FOLDS COMPLETE')


def trim20(d):
    s = np.sort(d); return float(s[20:-20].sum())


def drmse(sse0, dsse, n):
    return float(np.sqrt(max(sse0 + dsse, 0.0) / n) - np.sqrt(sse0 / n))


def evaluate(root):
    chain_df = pl.read_parquet(root / CHAIN).sort(A.ID)
    res_dir = root / RESULT_DIR
    thetas = json.loads((res_dir / 'thetas.json').read_text())
    folds = {}
    for fold in FOLDS:
        name = f'fold_{fold}_{fold + 6}'
        old = chain_df.filter(pl.col('month').is_in([fold, fold + 6]))
        ids = old[A.ID].to_numpy(); y = old[A.Y].to_numpy().astype(np.float64); ordered = old['ordered'].to_numpy().astype(np.float64)
        ok = old['evaluation_mask'].to_numpy(); ap = old['ADEP_mvt'].to_numpy(); nm = old['has_nm'].to_numpy(); D = old['D'].to_numpy(); mo = old['month'].to_numpy()
        with np.load(root / OUT / name / 'replays.npz') as z:
            if not np.array_equal(z['mvt_id'], ids) or not np.array_equal(z['excluded'], ~ok) or int(z['control_max_abs_diff']) != 0: raise ValueError('replay alignment ' + name)
            keys = list(z['keys']); delta = z['delta'].astype(np.float64); dm = z['dm_premix'].astype(np.float64); clipped = z['clipped']; floored = z['floored']
        tail = (ap == 'LIRF') & ~nm
        masks = dict(included=ok, decided=ok & ~(tail & (D >= 14400)), body=ok & (y <= 7200))
        se0 = (ordered - y) ** 2
        folds[fold] = dict(name=name, y=y, mo=mo, masks=masks, se0=se0, keys=keys, delta=delta, dm=dm, clipped=clipped, floored=floored, ordered=ordered)
    res = dict(protocol=PROTOCOL, primary_population=PRIMARY, lambda_ridge=LAMBDA, columns=COLS, control_exact=True,
               populations={f['name']: {k: int(v.sum()) for k, v in f['masks'].items()} for f in folds.values()}, legs={})
    for pop in POPS:
        R = {}
        # ---- LOPO
        units = []; pooled = dict(decided=0.0, body=0.0, included=0.0, trim20=0.0, premix_decided=0.0, premix_linear_decided=0.0)
        base = dict(decided=0.0, body=0.0, included=0.0); nn = dict(decided=0, body=0, included=0)
        for fold in FOLDS:
            F = folds[fold]; key = f'{pop}|lopo|pair{fold}'; j = F['keys'].index(key)
            pred = F['ordered'] + F['delta'][j]; d = (pred - F['y']) ** 2 - F['se0']
            u = dict(fold=fold, w_key=key, clipped_rows=int(F['clipped'][j]), p0_floored_rows=int(F['floored'][j]))
            for m in ('decided', 'body', 'included'):
                mk = F['masks'][m]; s0 = float(F['se0'][mk].sum()); ds = float(d[mk].sum()); n = int(mk.sum())
                u[m] = dict(n=n, dsse=ds, drmse=drmse(s0, ds, n)); pooled[m] += ds; base[m] += s0; nn[m] += n
            u['trim20_sse'] = trim20(d[F['masks']['decided']]); pooled['trim20'] += u['trim20_sse']
            units.append(u)
        lower = sum(1 for u in units if u['decided']['dsse'] < 0)
        f2 = next(u for u in units if u['fold'] == 2); f6 = next(u for u in units if u['fold'] == 6)
        R['lopo'] = dict(units=units, decided_lower=lower,
                         pooled={m: dict(n=nn[m], dsse=pooled[m], drmse=drmse(base[m], pooled[m], nn[m])) for m in ('decided', 'body', 'included')},
                         pooled_trim20_sse=pooled['trim20'],
                         gate=dict(decided_lower_ge4=lower >= 4, pooled_decided_lt0=pooled['decided'] < 0, pooled_body_lt0=pooled['body'] < 0,
                                   pooled_trim20_lt0=pooled['trim20'] < 0, fold2_not_worse=f2['decided']['dsse'] <= 0, fold6_not_worse=f6['decided']['dsse'] <= 0))
        # ---- forward legs: per month units
        for leg, months in (('half', list(range(7, 13))), ('expanding', list(range(4, 13)))):
            units = []; pooled = dict(decided=0.0, body=0.0, included=0.0); base = dict(decided=0.0, body=0.0, included=0.0); nn = dict(decided=0, body=0, included=0)
            for mo in months:
                fold = (mo - 1) % 6 + 1; F = folds[fold]; key = f'{pop}|{leg}|' + ('half' if leg == 'half' else f'm{mo}'); j = F['keys'].index(key)
                pred = F['ordered'] + F['delta'][j]; d = (pred - F['y']) ** 2 - F['se0']; inm = F['mo'] == mo
                u = dict(month=mo, w_key=key, clipped_rows=int(F['clipped'][j]), p0_floored_rows=int(F['floored'][j]))
                for m in ('decided', 'body', 'included'):
                    mk = F['masks'][m] & inm; s0 = float(F['se0'][mk].sum()); ds = float(d[mk].sum()); n = int(mk.sum())
                    u[m] = dict(n=n, dsse=ds, drmse=drmse(s0, ds, n)); pooled[m] += ds; base[m] += s0; nn[m] += n
                units.append(u)
            lower = sum(1 for u in units if u['decided']['dsse'] < 0)
            R[leg] = dict(units=units, decided_lower=lower, units_total=len(units),
                          pooled={m: dict(n=nn[m], dsse=pooled[m], drmse=drmse(base[m], pooled[m], nn[m])) for m in ('decided', 'body', 'included')})
        R['half']['gate'] = dict(pooled_decided_lt0=R['half']['pooled']['decided']['dsse'] < 0)
        R['expanding']['gate'] = dict(pooled_decided_lt0=R['expanding']['pooled']['decided']['dsse'] < 0, months_lower_ge6=R['expanding']['decided_lower'] >= 6)
        g = {**{f'lopo_{k}': v for k, v in R['lopo']['gate'].items()}, **{f'half_{k}': v for k, v in R['half']['gate'].items()},
             **{f'expanding_{k}': v for k, v in R['expanding']['gate'].items()}}
        g['pass'] = all(g.values())
        R['gate'] = g
        R['theta'] = {k.split('|', 1)[1]: dict(zip(COLS, v)) for k, v in thetas.items() if k.startswith(pop + '|')}
        res['legs'][pop] = R
    return res, folds


def evaluate_with_premix(root):
    """evaluate() plus the premix-level SSE change on the same decided rows (needs phase-1 state: recomputed cheaply)."""
    res, folds = evaluate(root)
    chain_df = pl.read_parquet(root / CHAIN)
    cols = [A.ID, A.Y, 'split', 'month', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'LOBT_flt', 'd_mvt_lobt', 'd_mvt_sched', 'd_mvt_iobt', 'd_mvt_eobt']
    feature = pl.read_parquet(root / 'data/features_fin1b.parquet', columns=cols).filter(pl.col('split') == 'train').sort(A.ID)
    pt = {pop: {leg: dict(chain_dsse=0.0, premix_dsse_actual=0.0, premix_dsse_linear=0.0, n=0, sse0_chain=0.0, sse0_premix=0.0) for leg in ('lopo', 'half', 'expanding')} for pop in POPS}
    thetas = json.loads((root / RESULT_DIR / 'thetas.json').read_text())
    for fold in FOLDS:
        Fd = load_fold(root, chain_df, feature, fold); S = premix_state(Fd); D, e = basis(Fd, S); F = folds[fold]
        if not np.array_equal(Fd['ids'], chain_df.filter(pl.col('month').is_in([fold, fold + 6])).sort(A.ID)[A.ID].to_numpy()): raise ValueError('id order')
        for pop in POPS:
            for key in F['keys']:
                if not key.startswith(pop + '|'): continue
                leg = key.split('|')[1]; j = F['keys'].index(key)
                if leg == 'lopo': ev = np.ones(len(F['y']), bool)
                elif leg == 'half': ev = F['mo'] >= 7
                else: ev = F['mo'] == int(key.rsplit('m', 1)[1])
                mk = F['masks']['decided'] & ev
                th = np.array(thetas[pop + '|' + key.split('|')[2]]); dm = F['dm'][j]
                m1 = S['m'] + dm
                pre_act = float(((F['y'] - m1) ** 2 - e ** 2)[mk].sum())
                Dm = D[mk]; em = e[mk]
                pre_lin = float(th @ (Dm.T @ Dm) @ th - 2 * (Dm.T @ em) @ th)
                pred = F['ordered'] + F['delta'][j]
                ch = float(((pred - F['y']) ** 2 - F['se0'])[mk].sum())
                P = pt[pop][leg]; P['chain_dsse'] += ch; P['premix_dsse_actual'] += pre_act; P['premix_dsse_linear'] += pre_lin; P['n'] += int(mk.sum())
                P['sse0_chain'] += float(F['se0'][mk].sum()); P['sse0_premix'] += float((e ** 2)[mk].sum())
    for pop in POPS:
        for leg, P in pt[pop].items():
            P['chain_drmse'] = drmse(P['sse0_chain'], P['chain_dsse'], P['n']); P['premix_drmse_actual'] = drmse(P['sse0_premix'], P['premix_dsse_actual'], P['n'])
            P['premix_drmse_linear'] = drmse(P['sse0_premix'], P['premix_dsse_linear'], P['n'])
            P['pass_through_sse'] = P['chain_dsse'] / P['premix_dsse_actual'] if P['premix_dsse_actual'] else None
            P['pass_through_rmse'] = P['chain_drmse'] / P['premix_drmse_actual'] if P['premix_drmse_actual'] else None
    res['pass_through_decided'] = pt
    res['verdict'] = dict(primary=PRIMARY, gate_pass=res['legs'][PRIMARY]['gate']['pass'],
                          secondary_gate_pass={p: res['legs'][p]['gate']['pass'] for p in POPS if p != PRIMARY})
    p = root / RESULT_DIR / 'result.json'; p.write_text(json.dumps(res, indent=1, default=float))
    for pop in POPS:
        R = res['legs'][pop]
        print(pop, 'LOPO decided %+.4f s (%d/6) body %+.4f included %+.4f trim20 %.3e | half %+.4f (%d/6) | expanding %+.4f (%d/9) | gate %s' % (
            R['lopo']['pooled']['decided']['drmse'], R['lopo']['decided_lower'], R['lopo']['pooled']['body']['drmse'], R['lopo']['pooled']['included']['drmse'],
            R['lopo']['pooled_trim20_sse'], R['half']['pooled']['decided']['drmse'], R['half']['decided_lower'], R['expanding']['pooled']['decided']['drmse'],
            R['expanding']['decided_lower'], R['gate']['pass']), flush=True)
        for leg in ('lopo', 'half', 'expanding'):
            P = pt[pop][leg]; print('   ', leg, 'premix actual %+.4f linear %+.4f chain %+.4f pass-through(sse) %s' % (
                P['premix_drmse_actual'], P['premix_drmse_linear'], P['chain_drmse'], '%.3f' % P['pass_through_sse'] if P['pass_through_sse'] is not None else 'n/a'), flush=True)
    print('WROTE', p, 'verdict', res['verdict'], flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['compute', 'evaluate'])
    p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--threads', type=int, default=4); p.add_argument('--folds', default='1,2,3,4,5,6')
    a = p.parse_args()
    if not 1 <= a.threads <= 4: raise ValueError('1..4 threads on the Mac (registered)')
    if a.action == 'compute': compute(a.root.resolve(), a.threads, [int(x) for x in a.folds.split(',')])
    else: evaluate_with_premix(a.root.resolve())


if __name__ == '__main__':
    main()
