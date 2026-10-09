"""Step 06_sigma_cluster of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""Cluster-robust noise scale: residual = airport-day shared part (tau^2 from 2025 OOF) + row part (calibrated mixvar proxy)."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
ID = 'MVT_ID_mvt'
def mixvar(proba, values, pred):
    v = np.nan_to_num(values.astype(np.float64))
    return np.sum(proba * (v - pred[:, None]) ** 2, axis=1)
# 2025 OOF Jan+Jul: tau^2 of the airport-day (and airport-day-hour) mean residual
z = np.load('data/cv_preds_e116prod/fold_1_7.npz')
tr = pl.concat([pl.read_parquet(f, columns=[ID, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']) for f in
                ['data/prc-2026-datasets/training_2025-01-01_2025-02-01.parquet', 'data/prc-2026-datasets/training_2025-07-01_2025-08-01.parquet']]).filter(pl.col('PHASE_mvt') == 'DEP')
d = pl.DataFrame({ID: z['mvt_id'], 'r': (z['y'] - z['pred']).astype(float), 'mv': mixvar(z['proba'].astype(float), z['values'], z['pred'])}).join(tr, on=ID, how='inner')
print('OOF rows joined', d.height)
a0, b_ = json.load(open(SP + 'sigma.json'))['proxy'].values()
d = d.with_columns(v=a0 + b_ * pl.col('mv'), day=pl.col('MVT_TIME_UTC_mvt').dt.date(), hour=pl.col('MVT_TIME_UTC_mvt').dt.truncate('1h'))
res = {}
for key in (['ADEP_mvt', 'day'], ['ADEP_mvt', 'hour']):
    # trimmed residual (|r| <= 3600) for the shared part, to keep it from being one outlier
    g = d.filter(pl.col('r').abs() <= 3600).group_by(key).agg(m=pl.col('r').mean(), n=pl.len(), s2=pl.col('r').var())
    g = g.filter(pl.col('n') >= 5)
    # tau^2 = var(cluster means) - mean(s2/n)
    tau2 = float(g['m'].var() - (g['s2'] / g['n']).mean())
    res['_'.join(key)] = tau2
    print(key, 'clusters', g.height, 'tau', np.sqrt(max(tau2, 0)))
json.dump(res, open(SP + 'tau.json', 'w'))
# 2026 rows: cluster keys
st = np.load(SP + 'stages.npz'); ids = st['ids']; N = len(ids)
rk = pl.read_parquet('data/final_tail_20261009/work/data/prc-2026-datasets/ranking.parquet', columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
meta = pl.DataFrame({ID: ids}).join(rk, on=ID, how='left', maintain_order='left').with_columns(
    day=pl.col('MVT_TIME_UTC_mvt').dt.date(), hour=pl.col('MVT_TIME_UTC_mvt').dt.truncate('1h'))
p = np.load('data/v12078_noe81_20261009/submissions/parts_v990202.npz'); idx = np.argsort(p['mvt_id'])[np.searchsorted(np.sort(p['mvt_id']), ids)]
v = a0 + b_ * mixvar(p['proba'][idx], p['values'][idx], p['pred'][idx])
ref = pl.read_parquet('submissions/gentle-igloo_v12077.parquet')
def vec(path):
    return ref.select(ID).join(pl.read_parquet(path), on=ID, how='left')['TAXITIME_SEC_mvt'].to_numpy().astype(float)
def sigma(D, key, tau2):
    row = np.sum(D ** 2 * v)
    cl = meta.select(key).with_columns(D=pl.Series(D)).group_by(key).agg(s=pl.col('D').sum())['s'].to_numpy()
    return 2 * np.sqrt(row + tau2 * np.sum(cl ** 2)) / N, 2 * np.sqrt(row) / N
D_FH = st['F_half'] - st['H_half']
cases = {'F-H': (D_FH, 231.1805 ** 2 - (231.0488 ** 2 + 231.1770 ** 2 - 230.4812 ** 2), 0.0),
         'v12076-v12075': (vec('submissions/gentle-igloo_v12076.parquet') - vec('submissions/gentle-igloo_v12075.parquet'), 230.8183 ** 2 - 230.9387 ** 2, None)}
cases['v12076-v12075'] = (cases['v12076-v12075'][0], cases['v12076-v12075'][1], -float(np.mean(cases['v12076-v12075'][0] ** 2)))
out = {}
for name, (D, obs, expect) in cases.items():
    for key, tk in ((['ADEP_mvt', 'day'], 'ADEP_mvt_day'), (['ADEP_mvt', 'hour'], 'ADEP_mvt_hour')):
        s, s0 = sigma(D, key, res[tk])
        out[f'{name}|{tk}'] = dict(sigma_cluster=float(s), sigma_rows=float(s0), observed=float(obs), expected=float(expect), z=float((obs - expect) / s))
        print(name, tk, 'sigma rows %.1f cluster %.1f; observed %.1f expected %.1f z %.2f' % (s0, s, obs, expect, (obs - expect) / s))
json.dump(out, open(SP + 'sigma_cluster.json', 'w'), indent=1)
