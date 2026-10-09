"""Step 07_top_rows of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""Who are the rows that carry F-H? Premix-level anatomy of the top rows by |D|."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
st = np.load(SP + 'stages.npz'); ids = st['ids']; N = len(ids)
meta = np.load(SP + 'meta.npz')
D = st['F_half'] - st['H_half']
def parts(path):
    p = np.load(path); idx = np.argsort(p['mvt_id'])[np.searchsorted(np.sort(p['mvt_id']), ids)]
    return {k: p[k][idx] for k in ('reg', 'proba', 'values', 'pred')}
PF = parts('data/final_tail_20261009/work/submissions/parts_v970102.npz')
PH = parts('data/v12078_noe81_20261009/submissions/parts_v990202.npz')
top = np.argsort(-np.abs(D))[:345]
m = np.zeros(N, bool); m[top] = True
ap, mon, nonm, grp = meta['ap'], meta['mon'], meta['nonm'], meta['grp']
print('top345: share of sumsq', (D[m] ** 2).sum() / (D ** 2).sum(), 'RMS D', np.sqrt(np.mean(D[m] ** 2)))
print('by airport-month', pl.DataFrame({'ap': ap[m], 'mon': mon[m]}).group_by(['ap', 'mon']).len().sort('len', descending=True).head(8))
print('noNM', nonm[m].sum(), 'group', dict(zip(*np.unique(grp[m], return_counts=True))))
# premix components on the top rows
dreg = PF['reg'] - PH['reg']
dprob = np.abs(PF['proba'] - PH['proba']).sum(axis=1) / 2  # total variation distance
dval = np.nanmax(np.abs(np.nan_to_num(PF['values'][:, 1:5]) - np.nan_to_num(PH['values'][:, 1:5])), axis=1)
maskF = (PF['proba'] == 0); maskH = (PH['proba'] == 0)
maskflip = (maskF != maskH).any(axis=1)
dpre = PF['pred'] - PH['pred']
for lab, sel in (('top345', m), ('rest', ~m & ~meta['tail42'])):
    print(lab, 'RMS dpremix(after E167) %.1f' % np.sqrt(np.mean(dpre[sel] ** 2)), 'RMS dreg %.1f' % np.sqrt(np.mean(dreg[sel] ** 2)),
          'mean TV(proba) %.3f' % dprob[sel].mean(), 'median TV %.3f' % np.median(dprob[sel]), 'LOBT-mask flip rows %d' % maskflip[sel].sum(),
          'max cand shift median %.1f' % np.median(dval[sel]))
# decomposition of premix change on top rows: proba part vs values part (first-order)
vF = np.nan_to_num(PF['values']); vH = np.nan_to_num(PH['values'])
part_p = ((PF['proba'] - PH['proba']) * (vF + vH) / 2).sum(axis=1)
part_v = (((PF['proba'] + PH['proba']) / 2) * (vF - vH)).sum(axis=1)
for lab, sel in (('top345', m), ('all non-tail', ~meta['tail42'])):
    print(lab, 'sumsq share from proba part %.3f, from values part %.3f (of mix change)' % (
        (part_p[sel] ** 2).sum() / ((part_p[sel] + part_v[sel]) ** 2).sum(), (part_v[sel] ** 2).sum() / ((part_p[sel] + part_v[sel]) ** 2).sum()))
# candidate spread on the top rows
spread = np.nanmax(np.where(PH['proba'] > 0.02, vH, np.nan), axis=1) - np.nanmin(np.where(PH['proba'] > 0.02, vH, np.nan), axis=1)
print('top345 candidate spread (p>0.02) median %.0f s, rest median %.0f s' % (np.nanmedian(spread[m]), np.nanmedian(spread[~m])))
# is D on the top rows driven by stages or premix? stage-level D at e167 vs final
D167 = st['F_e167'] - st['H_e167']
print('top345: corr(D_final, D_premix_e167) %.3f; RMS D_e167 %.1f; slope %.3f' % (np.corrcoef(D[m], D167[m])[0, 1], np.sqrt(np.mean(D167[m] ** 2)),
      np.sum(D[m] * D167[m]) / np.sum(D167[m] ** 2)))
rows = []
for i in top[:25]:
    rows.append(dict(id=int(ids[i]), ap=str(ap[i]), mon=int(mon[i]), noNM=bool(nonm[i]), grp=str(grp[i]), H=st['H_half'][i], F=st['F_half'][i], D=D[i],
                     pH=np.round(PH['proba'][i], 2).tolist(), pF=np.round(PF['proba'][i], 2).tolist(), vH=np.round(vH[i, :5]).tolist()))
for r in rows:
    print(r)
json.dump(dict(top_rows=rows), open(SP + 'top_rows.json', 'w'), indent=1, default=float)
