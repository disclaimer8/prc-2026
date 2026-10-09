"""Step 12_sigma_piecewise of the v12080 control explanation (journal 446): noise scale of the cross term with the residual
variance taken bin-wise from Jan+Jul 2025 OOF (E[r^2 | premix mixture variance]), per swapped component; plus the binomial
sigma of the LIRF no-NM group under the two-point label model. Intermediate files live in $V12080_SCRATCH."""
import os, json
import numpy as np
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
def mixvar(p, v, pred): return np.sum(p * (np.nan_to_num(v.astype(float)) - pred[:, None]) ** 2, axis=1)
z = np.load('data/cv_preds_e116prod/fold_1_7.npz')
mv25 = mixvar(z['proba'].astype(float), z['values'], z['pred']); r2 = (z['y'] - z['pred']) ** 2
edges = np.array([0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e12])
b = np.digitize(mv25, edges) - 1
tab = np.array([r2[b == k].mean() for k in range(len(edges) - 1)])
st = np.load(SP + 'stages.npz'); ids = st['ids']; N = len(ids); order = np.argsort(ids)
p = np.load('data/v12078_noe81_20261009/submissions/parts_v990202.npz'); idx = np.argsort(p['mvt_id'])[np.searchsorted(np.sort(p['mvt_id']), ids)]
mv = mixvar(p['proba'][idx], p['values'][idx], p['pred'][idx])
v = tab[np.clip(np.digitize(mv, edges) - 1, 0, len(tab) - 1)]
F = st['F_half']; D = F - st['H_half']
res = dict(sigma_FH=float(2 * np.sqrt(np.sum(D ** 2 * v)) / N), bins=dict(edges=edges.tolist(), E_r2=tab.tolist()))
for n in ['tab', 'nn', 'proba', 'proba_model', 'proba_view', 'cand', 'motion', 'products', 'allold']:
    zz = np.load(SP + f'runs/{n}_janjul.npz'); a = np.empty(N); a[order] = zz['half']
    res[n] = float(2 * np.sqrt(np.sum((F - a) ** 2 * v)) / N)
tail = np.isin(ids, np.array(json.load(open('analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'))['ids']))
res['tail42'] = float(2 * np.sqrt(np.sum(np.where(tail, D, 0) ** 2 * v)) / N)
meta = np.load(SP + 'meta.npz')
g = (meta['ap'] == 'LIRF') & meta['nonm'] & ~meta['tail42']
pf = np.load('data/final_tail_20261009/work/submissions/parts_v970102.npz'); jf = np.argsort(pf['mvt_id'])[np.searchsorted(np.sort(pf['mvt_id']), ids)]
dv = np.nan_to_num(p['values'][idx][g, 1]) - np.nan_to_num(p['values'][idx][g, 0])
q = (p['proba'][idx][g, 1] + pf['proba'][jf][g, 1]) / 2
res['lirf_nonm_binomial_sigma'] = float(2 * np.sqrt(np.sum(D[g] ** 2 * dv ** 2 * q * (1 - q))) / N)
print(json.dumps(res, indent=1))
json.dump(res, open(SP + 'sigma_piecewise.json', 'w'), indent=1)
