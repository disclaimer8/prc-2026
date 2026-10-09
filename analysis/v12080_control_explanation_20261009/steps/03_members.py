"""Step 03_members of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
ID = 'MVT_ID_mvt'
FW = 'data/final_tail_20261009/work/submissions/'
O8 = 'data/v12078_noe81_20261009/submissions/'
P_O = np.load(O8 + 'parts_v990200.npz')
ids = P_O['mvt_id']
def al(mid, v):
    o = np.argsort(mid); pos = np.searchsorted(mid[o], ids); assert (mid[o][pos] == ids).all(); return v[..., o][..., pos]
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
W = {'cat': 0.2744444444444445 * 0.65, 'lgb': 0.2744444444444445 * 0.35, 'raw': 0.028888888888888857, 'lobt': 0.08666666666666667,
     'sched': 0.08666666666666667, 'eobt': 0.08666666666666667, 'aobt': 0.08666666666666667, 'nn': 0.35}
old = {'cat': O8 + 'cat_fin1b_seeds15.npz', 'lgb': O8 + 'lgb_reg_seeds_v870.npz', 'nn': 'data/v12078_noe81_20261009/data/e203_rank_seeds3to9_20261008/models/prediction_10seed.npz',
       'motion': 'data/v12078_noe81_20261009/data/e185_motion_rank.npz', **{k: O8 + f'e130_{k}f_rank.npz' for k in ('raw', 'lobt', 'sched', 'eobt', 'aobt')}}
new = {'cat': FW + 'cat_rank.npz', 'lgb': FW + 'lgb_rank.npz', 'nn': FW + 'nn_rank.npz', 'motion': 'data/final_tail_20261009/work/data/e185_motion_rank.npz',
       **{k: FW + f'e130_{k}f_rank.npz' for k in ('raw', 'lobt', 'sched', 'eobt', 'aobt')}}
out = {}
M = {}
for k in ['cat', 'lgb', 'raw', 'lobt', 'sched', 'eobt', 'aobt', 'nn', 'motion']:
    zo, zn = np.load(old[k]), np.load(new[k])
    mo = al(zo['mvt_id'], zo['reg']); mn = al(zn['mvt_id'], zn['reg'])
    M[k] = (mo, mn)
    d = mn - mo
    rec = dict(rms_new_minus_old=rms(d), mean_shift=float(d.mean()), weight_in_regular=W.get(k), premix_rms_if_weighted=rms(d) * W[k] if k in W else None)
    if 'per_seed' in zo.files and 'per_seed' in zn.files:
        po = al(zo['mvt_id'], zo['per_seed']); pn = al(zn['mvt_id'], zn['per_seed'])
        no, nn_ = po.shape[0], pn.shape[0]
        vo = po.var(axis=0, ddof=1).mean(); vn = pn.var(axis=0, ddof=1).mean()
        noise = np.sqrt(vo / no + vn / nn_)
        rec.update(seeds_old=no, seeds_new=nn_, seed_sd_old=float(np.sqrt(vo)), seed_sd_new=float(np.sqrt(vn)), expected_noise_rms=float(noise),
                   systematic_rms=float(np.sqrt(max(rms(d) ** 2 - noise ** 2, 0))))
        # same-seed-index pairs
        rec['same_index_seed_rms'] = [rms(pn[i] - po[i]) for i in range(min(no, nn_))][:5]
    out[k] = rec
    print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in rec.items()}))
# NN old per-seed sd
sd_old = 328.68674019085205
seeds = [np.load(f'data/e203_rank_20261007/seed_{s}.npy') for s in range(3)] + [np.load(f'data/e203_rank_seeds3to9_20261008/models/seed_{s}.npy') for s in range(3, 10)]
S = np.stack(seeds).astype(np.float64) * sd_old
v = S.var(axis=0, ddof=1).mean()
print('NN old seed sd (s)', np.sqrt(v), 'noise of 10-mean', np.sqrt(v / 10))
st = json.load(open('analysis/final_ordered_trajectory_20261008/execution_10seed/statistics.json'))
pr = st['pairwise_seed_rms_sec_mean']
print('NN final pairwise rms (4 months)', pr, '-> seed sd', pr / np.sqrt(2), 'noise of 10-mean', pr / np.sqrt(2) / np.sqrt(10))
exp = np.sqrt(v / 10 + (pr / np.sqrt(2)) ** 2 / 10)
print('NN expected noise rms of 10 vs 10', exp, 'observed', out['nn']['rms_new_minus_old'], 'systematic', np.sqrt(max(out['nn']['rms_new_minus_old'] ** 2 - exp ** 2, 0)))
out['nn'].update(seed_sd_old=float(np.sqrt(v)), seed_sd_new_4months=float(pr / np.sqrt(2)), expected_noise_rms=float(exp),
                 systematic_rms=float(np.sqrt(max(out['nn']['rms_new_minus_old'] ** 2 - exp ** 2, 0))))
# 3-seed final vs 3 seed old
z3 = np.load('analysis/final_ordered_trajectory_20261008/execution_10seed/prediction_3seed.npz'); z3o = np.load('data/e203_rank_20261007/prediction.npz')
print('NN 3seed final vs 3seed old rms', rms(al(z3['mvt_id'], z3['reg']) - al(z3o['mvt_id'], z3o['reg'])))
# regular candidate total
r_o = P_O['reg']; P_F = np.load(FW + 'parts_v970100.npz'); r_n = al(P_F['mvt_id'], P_F['reg'])
print('regular candidate rms new-old', rms(r_n - r_o), 'sum of weighted parts', {k: round(W[k] * out[k]['rms_new_minus_old'], 2) for k in W})
json.dump(out, open(os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/members.json', 'w'), indent=1)
