"""Step 09_attribution of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""Attribution of F-H to swapped components (variants of swap_chain.py)."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
RUNS = SP + 'runs/'
st = np.load(SP + 'stages.npz'); ids = st['ids']; N = len(ids)
meta = np.load(SP + 'meta.npz')
order = np.argsort(ids)
F = st['F_half']; H = st['H_half']; D = F - H
def load(name):
    z = np.load(RUNS + f'{name}_janjul.npz')
    assert np.array_equal(z['ids'], ids[order])
    out = {}
    for k in z.files:
        if k == 'ids':
            continue
        a = np.empty(N); a[order] = z[k]; out[k] = a
    return out
V = {n: load(n) for n in ['identity', 'tab', 'nn', 'proba', 'proba_model', 'proba_view', 'cand', 'motion', 'products', 'allpremix', 'allold'] if os.path.exists(RUNS + f'{n}_janjul.npz')}
assert np.array_equal(V['identity']['half'], F), 'identity variant does not reproduce v12080'
tail = meta['tail42']
top = np.zeros(N, bool); top[np.argsort(-np.abs(D))[:345]] = True
lirf_nonm = (meta['ap'] == 'LIRF') & meta['nonm']
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
a0, b_ = json.load(open(SP + 'sigma.json'))['proxy'].values()
p = np.load('data/v12078_noe81_20261009/submissions/parts_v990202.npz'); idx = np.argsort(p['mvt_id'])[np.searchsorted(np.sort(p['mvt_id']), ids)]
mv = np.sum(p['proba'][idx] * (np.nan_to_num(p['values'][idx]) - p['pred'][idx][:, None]) ** 2, axis=1)
v = a0 + b_ * mv
rows = []
def describe(name, d):
    r = dict(component=name, rms=rms(d), changed=int((d != 0).sum()), mean=float(d.mean()),
             share_top345=float((d[top] ** 2).sum() / max((d ** 2).sum(), 1e-9)), share_lirf_nonm=float((d[lirf_nonm] ** 2).sum() / max((d ** 2).sum(), 1e-9)),
             rms_premix=None, proj_on_D=float(np.dot(d, D) / np.dot(D, D)), corr_with_D=float(np.corrcoef(d, D)[0, 1]) if d.std() > 0 else 0.0,
             sigma_proxy=float(2 * np.sqrt(np.sum(d ** 2 * v)) / N), hard_bound_s=rms(d))
    rows.append(r)
    return r
for n, lab in (('tab', 'tabular members (cat, lgb, 5 anchors)'), ('nn', 'NN member'), ('proba', 'base classifier probabilities'), ('proba_model', '  of which: classifier model (old clf on final view)'), ('proba_view', '  of which: input view (final clf on old view)'),
               ('cand', 'planning candidates (input dithering in premix)'), ('motion', 'E185 motion member'), ('products', 'daily ADS-B products')):
    if n in V:
        d = F - V[n]['half']
        r = describe(lab, d)
        r['rms_premix'] = rms((V['identity']['premix'] - V[n]['premix'])[~tail])
        r['pass_through'] = float(np.dot(d[~tail], (V['identity']['premix'] - V[n]['premix'])[~tail]) / max(np.sum((V['identity']['premix'] - V[n]['premix'])[~tail] ** 2), 1e-9))
if 'allpremix' in V:
    sum1 = sum(F - V[n]['half'] for n in ('tab', 'nn', 'proba', 'cand', 'motion'))
    dall = F - V['allpremix']['half']
    describe('all premix inputs + motion together', dall)
    print('additivity premix: rms(sum of singles) %.3f vs rms(joint) %.3f, rms(interaction) %.3f' % (rms(sum1), rms(dall), rms(dall - sum1)))
if 'allold' in V:
    dao = F - V['allold']['half']
    describe('all of the above + old products', dao)
    resid = V['allold']['half'] - H
    resid[tail] = 0  # tail rows: final served values were injected in every variant
    describe('remainder: H-chain vs final chain on old inputs (ranking-file dither in stage features, E167 arrivals, OPDI/LSZH, d_lobt)', resid)
    print('remainder on tail rows (served values differ) rms', rms(np.where(tail, V['allold']['half'] - H, 0)))
    dtail = np.where(tail, D, 0)
    describe('tail 42 rows: final E156 vs canary E156 values', dtail)
df = pl.DataFrame(rows)
with pl.Config(tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=60):
    print(df)
json.dump(rows, open(SP + 'attribution.json', 'w'), indent=1)
