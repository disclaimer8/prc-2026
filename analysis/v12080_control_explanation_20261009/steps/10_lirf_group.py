"""Step 10_lirf_group of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""LIRF no-NM rows: how large a group-level miscalibration of the 'block = schedule' share would make F-H worth -260 s^2?"""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
ID = 'MVT_ID_mvt'
st = np.load(SP + 'stages.npz'); ids = st['ids']; N = len(ids)
meta = np.load(SP + 'meta.npz')
F = st['F_half']; H = st['H_half']; D = F - H
def parts(path):
    p = np.load(path); idx = np.argsort(p['mvt_id'])[np.searchsorted(np.sort(p['mvt_id']), ids)]
    return {k: p[k][idx] for k in ('proba', 'values', 'pred')}
PH = parts('data/v12078_noe81_20261009/submissions/parts_v990202.npz')
PF = parts('data/final_tail_20261009/work/submissions/parts_v970102.npz')
dmse_target = 231.1805 ** 2 - (231.0488 ** 2 + 231.1770 ** 2 - 230.4812 ** 2)
out = {}
for gname, g in (('LIRF no-NM (non-tail)', (meta['ap'] == 'LIRF') & meta['nonm'] & ~meta['tail42']),
                 ('LIRF July no-NM (non-tail)', (meta['ap'] == 'LIRF') & meta['nonm'] & ~meta['tail42'] & (meta['mon'] == 7)),
                 ('all no-NM (non-tail)', meta['nonm'] & ~meta['tail42'])):
    vr = np.nan_to_num(PH['values'][g, 0]); vs = np.nan_to_num(PH['values'][g, 1])
    pH = PH['proba'][g, 1]; pF = PF['proba'][g, 1]
    f, h = F[g], H[g]
    def dmse(q):
        return float(np.sum((1 - q) * ((f - vr) ** 2 - (h - vr) ** 2) + q * ((f - vs) ** 2 - (h - vs) ** 2)) / N)
    res = dict(rows=int(g.sum()), sumsq_share=float((D[g] ** 2).sum() / (D ** 2).sum()), mean_D=float(D[g].mean()), frac_D_neg=float((D[g] < 0).mean()),
               mean_pH_sched=float(pH.mean()), mean_pF_sched=float(pF.mean()), dmse_if_q_eq_pH=dmse(pH), dmse_if_q_eq_pF=dmse(pF),
               dmse_if_q0=dmse(np.zeros_like(pH)), dmse_if_q1=dmse(np.ones_like(pH)))
    deltas = np.linspace(-0.5, 0.5, 2001)
    vals = np.array([dmse(np.clip(pH + dl, 0, 1)) for dl in deltas])
    k = np.argmin(np.abs(vals - dmse_target))
    res['delta_for_target'] = float(deltas[k]) if abs(vals[k] - dmse_target) < 5 else None
    res['dmse_at_delta'] = {f'{dl:+.2f}': float(vals[np.argmin(np.abs(deltas - dl))]) for dl in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)}
    out[gname] = res
    print(gname, json.dumps(res, indent=None)[:900])
# 2025 OOF calibration of the schedule share for LIRF no-NM rows, per month (premix-level OOF, e116prod)
months = []
for f in sorted(os.listdir('data/cv_preds_e116prod')):
    z = np.load('data/cv_preds_e116prod/' + f)
    months.append(pl.DataFrame({ID: z['mvt_id'], 'y': z['y'].astype(float), 'ps': z['proba'][:, 1].astype(float), 'vs': z['values'][:, 1].astype(float)}))
oof = pl.concat(months)
tr = pl.concat([pl.read_parquet(p, columns=[ID, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt', 'LOBT_flt']) for p in sorted(
    __import__('glob').glob('data/prc-2026-datasets/training_2025-*.parquet'))]).filter(pl.col('PHASE_mvt') == 'DEP')
d = oof.join(tr, on=ID, how='inner').filter((pl.col('ADEP_mvt') == 'LIRF') & pl.col('LOBT_flt').is_null() & pl.col('vs').is_not_nan())
d = d.with_columns(b=((pl.col('y') - pl.col('vs')).abs() <= 6).cast(pl.Float64), month=pl.col('MVT_TIME_UTC_mvt').dt.month())
cal = d.group_by('month').agg(n=pl.len(), mean_p=pl.col('ps').mean(), share=pl.col('b').mean()).with_columns(delta=pl.col('share') - pl.col('mean_p')).sort('month')
print(cal)
out['oof2025_LIRF_noNM_monthly'] = cal.to_dicts()
out['oof2025_delta_sd'] = float(cal['delta'].std())
json.dump(out, open(SP + 'lirf_group.json', 'w'), indent=1, default=float)
