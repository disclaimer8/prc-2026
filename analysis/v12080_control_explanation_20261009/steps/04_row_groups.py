"""Step 04_row_groups of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
import os, json, glob
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
z = np.load(SP + 'stages.npz')
ids = z['ids']
F = {k[2:]: z[k] for k in z.files if k.startswith('F_')}
H = {k[2:]: z[k] for k in z.files if k.startswith('H_')}
rk = pl.read_parquet('data/final_tail_20261009/work/data/prc-2026-datasets/ranking.parquet', columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'LOBT_flt', 'FLIGHT_ID_mvt', 'SCHED_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
meta = pl.DataFrame({ID: ids}).join(rk, on=ID, how='left', maintain_order='left')
assert meta['ADEP_mvt'].null_count() == 0
ap = meta['ADEP_mvt'].to_numpy(); mon = meta['MVT_TIME_UTC_mvt'].dt.month().to_numpy()
nonm = meta['LOBT_flt'].is_null().to_numpy(); noflt = meta['FLIGHT_ID_mvt'].is_null().to_numpy()
print('no LOBT', nonm.sum(), 'no FLIGHT_ID', noflt.sum(), 'both', (nonm & noflt).sum())
# coverage from products: final and old kt1 offblock2
def cov(pattern):
    fs = sorted(glob.glob(pattern))
    d = pl.concat([pl.read_parquet(f, columns=[ID, 'coverage']) for f in fs])
    return pl.DataFrame({ID: ids}).join(d.unique(ID), on=ID, how='left', maintain_order='left')['coverage'].fill_null(0).to_numpy()
cov_new = cov('data/final_adsb_derivatives_20261008/data/adsblol_kt1/offblock2_2026-0[17]-*.parquet')
cov_old = cov('data/adsblol_kt1/offblock2_2026-0[17]-*.parquet')
print('coverage>=2 new', (cov_new >= 2).sum(), 'old', (cov_old >= 2).sum(), 'both', ((cov_new >= 2) & (cov_old >= 2)).sum(), 'flip', ((cov_new >= 2) != (cov_old >= 2)).sum())
def visit(pattern):
    fs = sorted(glob.glob(pattern))
    d = pl.concat([pl.read_parquet(f, columns=[ID]) for f in fs]).unique(ID)
    return np.isin(ids, d[ID].to_numpy())
vis_new = visit('data/final_adsb_derivatives_20261008/data/adsblol/visit_2026-0[17]-*.parquet')
vis_old = visit('data/adsblol/visit_2026-0[17]-*.parquet')
print('visit new', vis_new.sum(), 'old', vis_old.sum(), 'flip', (vis_new != vis_old).sum())
served = {}
for st, prev in [('e141', 'e156'), ('e64', 'e141'), ('e188', 'e64'), ('lszh', 'protect'), ('e185', 'premix'), ('e167', 'e185'), ('e152', 'e188'), ('lobt', 'e152')]:
    served[st] = (F[st] != F[prev], H[st] != H[prev])
    a, b = served[st]
    print(f'{st:6s} changed F {a.sum():7d} H {b.sum():7d} both {(a & b).sum():7d} only F {(a & ~b).sum():6d} only H {(~a & b).sum():6d}')
tail42 = np.isin(ids, np.array(json.load(open('analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'))['ids']))
D = F['half'] - H['half']
N = len(D)
def rms(x): return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')
grp = np.full(N, 'other', dtype=object)
e141m = cov_new >= 2
e64m = served['e64'][0] | served['e64'][1]
grp[e64m] = 'E64'
grp[e141m] = 'E141'
grp[(ap == 'LIRF') & ~nonm & (served['e188'][0] | served['e188'][1])] = 'E188'
grp[tail42] = 'tail42'
rows = []
for g in ['E141', 'E64', 'E188', 'other', 'tail42']:
    m = grp == g
    rows.append(dict(group=g, n=int(m.sum()), rms_D=rms(D[m]), share_sumsq=float((D[m] ** 2).sum() / (D ** 2).sum()),
                     rms_premix_diff=rms((F['premix'] - H['premix'])[m]), mean_D=float(D[m].mean())))
print(pl.DataFrame(rows))
# NM
for lab, m in [('NM', ~nonm), ('noNM', nonm)]:
    print(lab, m.sum(), 'rms D', rms(D[m]), 'share', (D[m] ** 2).sum() / (D ** 2).sum())
# airport-month
am = []
for a in sorted(set(ap)):
    for mo in (1, 7):
        m = (ap == a) & (mon == mo)
        am.append(dict(ap=a, month=int(mo), n=int(m.sum()), rms_D=round(rms(D[m]), 2), share=round(float((D[m] ** 2).sum() / (D ** 2).sum()), 4),
                       e141_frac=round(float(e141m[m].mean()), 3), rms_premix=round(rms((F['premix'] - H['premix'])[m & ~tail42]), 2)))
print(pl.DataFrame(am).sort('share', descending=True))
# concentration
s = np.sort(D ** 2)[::-1]
for q in (0.001, 0.01, 0.1):
    k = int(q * N); print('top', q, 'rows share of sumsq', s[:k].sum() / s.sum())
print('max |D|', np.abs(D).max())
# big predictions
big = H['half'] > 7200
print('rows H>7200', big.sum(), 'rms D there', rms(D[big]), 'share', (D[big] ** 2).sum() / (D ** 2).sum())
np.savez(SP + 'meta.npz', ids=ids, ap=ap.astype(str), mon=mon, nonm=nonm, cov_new=cov_new, cov_old=cov_old, vis_new=vis_new, vis_old=vis_old, grp=grp.astype(str), tail42=tail42)
json.dump(dict(groups=rows, airport_month=am), open(SP + 'groups.json', 'w'), indent=1)
