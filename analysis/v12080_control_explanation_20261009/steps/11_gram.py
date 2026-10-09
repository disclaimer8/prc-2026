"""Step 11_gram of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""Gram identities among the scored Jan/Jul files and the exact hypothetical H."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
ref = pl.read_parquet('submissions/gentle-igloo_v12077.parquet')
def vec(p):
    return ref.select(ID).join(pl.read_parquet(p), on=ID, how='left')[Y].to_numpy().astype(float)
S = {'v12075': 230.9387, 'v12076': 230.8183, 'v12077': 230.4812, 'v12078': 231.0488, 'v12079': 231.1770, 'v12080': 231.1805, 'v12090': 232.1933}
f = {k: vec(f'submissions/gentle-igloo_{k}.parquet') for k in S}
st = np.load(SP + 'stages.npz')
assert np.array_equal(st['ids'], ref[ID].to_numpy())
f['H'] = st['H_half']
S['H'] = float(np.sqrt(S['v12078'] ** 2 + S['v12079'] ** 2 - S['v12077'] ** 2))
keys = ['v12075', 'v12077', 'v12078', 'v12079', 'H', 'v12080', 'v12090']
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
tab = []
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        d = rms(f[a] - f[b])
        inner = (S[a] ** 2 + S[b] ** 2 - d ** 2) / 2
        mid = np.sqrt((S[a] ** 2 + S[b] ** 2) / 2 - d ** 2 / 4)
        dmse = S[a] ** 2 - S[b] ** 2
        tab.append(dict(a=a, b=b, rms_diff=round(d, 3), dmse_a_minus_b=round(dmse, 1), corr_change_vs_b_resid=round((dmse - d ** 2) / (2 * d * S[b]), 4),
                        resid_corr=round(inner / S[a] / S[b], 6), midpoint_score=round(float(mid), 4)))
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print(pl.DataFrame(tab))
print('s(H) exact', S['H'])
# pure re-draw null: E[2cov] = -RMS^2 -> corr_null = -RMS/(2 s_H)
d = rms(f['v12080'] - f['H'])
print('F-H: rms %.3f, corr observed %.4f, corr expected for an equal-variance re-draw %.4f' % (d, (S['v12080'] ** 2 - S['H'] ** 2 - d ** 2) / (2 * d * S['H']), -d / (2 * S['H'])))
json.dump(dict(scores=S, pairs=tab), open(SP + 'gram.json', 'w'), indent=1)
