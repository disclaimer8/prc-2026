"""Step 05_sigma_proxy of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""Noise scale of the cross term: label-free residual proxy (premix mixture variance), calibrated on Jan+Jul 2025 OOF."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
def mixvar(proba, values, pred):
    v = np.nan_to_num(values.astype(np.float64))
    return np.sum(proba * (v - pred[:, None]) ** 2, axis=1)
# --- calibration on 2025 OOF Jan+Jul (fold_1_7): E[(y - pred)^2] vs mixture variance
z = np.load('data/cv_preds_e116prod/fold_1_7.npz')
mv = mixvar(z['proba'].astype(np.float64), z['values'], z['pred'])
r2 = (z['y'] - z['pred']) ** 2
print('OOF Jan+Jul 2025 rows', len(r2), 'RMSE', np.sqrt(r2.mean()), 'mean mixvar', mv.mean())
bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e12]
cal = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (mv >= lo) & (mv < hi)
    if m.sum():
        cal.append(dict(lo=lo, hi=hi, n=int(m.sum()), mean_mixvar=float(mv[m].mean()), mean_r2=float(r2[m].mean())))
print(pl.DataFrame(cal))
# fit r2 ~ a + b*mixvar (by bin means) -> use as proxy
a0 = float(r2[mv < 1e3].mean())
b_ = float((r2[mv >= 1e3].mean() - a0) / mv[mv >= 1e3].mean())
print('proxy: r2 = %.0f + %.3f * mixvar' % (a0, b_))
# --- 2026: D = F - H, proxy residual variance from the H-chain parts after E167 and the final parts (both)
st = np.load(SP + 'stages.npz'); ids = st['ids']
D = st['F_half'] - st['H_half']; N = len(D)
def parts_on(path):
    p = np.load(path); idx = np.argsort(p['mvt_id'])[np.searchsorted(np.sort(p['mvt_id']), ids)]
    assert (p['mvt_id'][idx] == ids).all()
    return mixvar(p['proba'][idx], p['values'][idx], p['pred'][idx])
mvH = parts_on('data/v12078_noe81_20261009/submissions/parts_v990202.npz')
mvF = parts_on('data/final_tail_20261009/work/submissions/parts_v970102.npz')
meta = np.load(SP + 'meta.npz', allow_pickle=False)
e141 = meta['grp'] == 'E141'
dmse = 231.1805 ** 2 - (231.0488 ** 2 + 231.1770 ** 2 - 230.4812 ** 2)
out = {}
for lab, mvx in [('H', mvH), ('F', mvF), ('max', np.maximum(mvH, mvF))]:
    for e141_scale in (1.0, 0.5):
        v = a0 + b_ * mvx
        v = np.where(e141, v * e141_scale, v)
        sig = 2 * np.sqrt(np.sum(D ** 2 * v)) / N
        top = np.argsort(-np.abs(D))[:345]
        sig_top = 2 * np.sqrt(np.sum(D[top] ** 2 * v[top])) / N
        out[f'{lab}_e141x{e141_scale}'] = dict(sigma_dmse=float(sig), sigma_top345=float(sig_top), z=float(dmse / sig))
        print(lab, 'E141 scale', e141_scale, 'sigma(dMSE) %.1f s^2' % sig, 'top-345 rows %.1f' % sig_top, 'z = %.2f' % (dmse / sig))
# simple homoscedastic version
print('homoscedastic RMSE 231.7: sigma', 2 * np.sqrt(np.sum(D ** 2)) * 231.74 / N)
# --- the same proxy on the clean null calibration pair v12076 vs v12075 (nested seed-noise reduction)
ref = pl.read_parquet('submissions/gentle-igloo_v12077.parquet')
def vec(p):
    return ref.select('MVT_ID_mvt').join(pl.read_parquet(p), on='MVT_ID_mvt', how='left')['TAXITIME_SEC_mvt'].to_numpy().astype(float)
pairs = [('v12076', 'v12075', 230.8183, 230.9387, 'nested 10-vs-3 seeds: E = -RMS^2'),
         ('v12077', 'v12076', 230.4812, 230.8183, 'weight change (systematic)'),
         ('v12078', 'v12077', 231.0488, 230.4812, 'E81 removal (systematic)')]
cal_pairs = []
for a, b, sa, sb, note in pairs:
    d = vec(f'submissions/gentle-igloo_{a}.parquet') - vec(f'submissions/gentle-igloo_{b}.parquet')
    v = a0 + b_ * mvH
    sig = 2 * np.sqrt(np.sum(d ** 2 * v)) / N
    obs = sa ** 2 - sb ** 2
    cal_pairs.append(dict(pair=f'{a}-{b}', rms=float(np.sqrt(np.mean(d ** 2))), dmse=float(obs), sigma_proxy=float(sig), note=note))
    print(a, b, 'rms %.3f dMSE %.1f sigma_proxy %.1f' % (np.sqrt(np.mean(d ** 2)), obs, sig), note)
json.dump(dict(calibration=cal, proxy=dict(a=a0, b=b_), sigma=out, pairs=cal_pairs, dmse_F_minus_H=dmse), open(SP + 'sigma.json', 'w'), indent=1)
