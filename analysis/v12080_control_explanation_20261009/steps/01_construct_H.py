"""Step 01_construct_H of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
import os, json, hashlib
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
FW = 'data/final_tail_20261009/work/submissions/'
O8 = 'data/v12078_noe81_20261009/submissions/'
O9 = 'data/v12079_freshe156_20261009/submissions/'
O7 = 'data/v12077_nn035_20261008/submissions/'
ref = pl.read_parquet('submissions/gentle-igloo_v12077.parquet')
ids = ref[ID].to_numpy()

def vec(path):
    d = pl.read_parquet(path)
    j = ref.select(ID).join(d, on=ID, how='left')
    assert j[Y].null_count() == 0, path
    return j[Y].to_numpy().astype(np.float64)

S = {'v12077': 230.4812, 'v12078': 231.0488, 'v12079': 231.1770, 'v12080': 231.1805}
f = {k: vec(f'submissions/gentle-igloo_{k}.parquet') for k in S}
F = f['v12080']
tail42 = np.array(json.load(open('analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'))['ids'])
m42 = np.isin(ids, tail42)
print('42 found', m42.sum())
d79_77 = f['v12079'] != f['v12077']
print('v12079 vs v12077 differ rows', d79_77.sum(), 'subset of 42', (d79_77 & ~m42).sum())
print('v12077 vs v12078 on 42 rows equal', (f['v12077'][m42] == f['v12078'][m42]).all())
H = f['v12078'].copy(); H[m42] = f['v12079'][m42]
sH = np.sqrt(S['v12078']**2 + S['v12079']**2 - S['v12077']**2)
print('s(H) exact', sH, 'additive', S['v12077'] + (S['v12078'] - S['v12077']) + (S['v12079'] - S['v12077']))
# final served E156
srv = np.load('data/final_tail_20261009/work/e156/run_0/served_ids.npy')
ms = np.isin(ids, srv)
print('final served total', len(srv), 'on JanJul', ms.sum(), 'overlap with 42', (ms & m42).sum(), 'served not in 42', (ms & ~m42).sum(), '42 not served', (m42 & ~ms).sum())
D = F - H
N = len(F)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
print('RMS(F-H)', rms(D), 'mean', D.mean(), 'changed', (D != 0).sum())
dmse = S['v12080'] ** 2 - sH ** 2
print('dMSE F vs H', dmse, ' 2cov', dmse - rms(D) ** 2, 'corr', (dmse - rms(D) ** 2) / 2 / rms(D) / sH)
tail = ms | m42
print('RMS on tail rows part', rms(np.where(tail, D, 0)), 'non-tail part', rms(np.where(tail, 0, D)))
print('sum D^2 tail', (D[tail] ** 2).sum() / N, 'nontail', (D[~tail] ** 2).sum() / N)
for k in np.where(tail)[0]:
    print(int(ids[k]), 'in42' if m42[k] else '----', 'srv' if ms[k] else '---', 'v77', f['v12077'][k], 'v79/H', H[k], 'F', F[k], 'D', D[k])
