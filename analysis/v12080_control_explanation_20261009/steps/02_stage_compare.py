"""Step 02_stage_compare of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
import os, json
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
FW = 'data/final_tail_20261009/work/submissions/'
O8 = 'data/v12078_noe81_20261009/submissions/'
O9 = 'data/v12079_freshe156_20261009/submissions/'
ref = pl.read_parquet('submissions/gentle-igloo_v12077.parquet')
ids = ref[ID].to_numpy()
N = len(ids)
def vec(path):
    d = pl.read_parquet(path)
    j = ref.select(ID).join(d, on=ID, how='left')
    assert j[Y].null_count() == 0, path
    return j[Y].to_numpy().astype(np.float64)
def rms(x): return float(np.sqrt(np.mean(x ** 2)))
stages = [('premix', 0), ('e185', 1), ('e167', 2), ('e156', 3), ('e141', 4), ('e64', 5), ('e188', 6), ('e152', 10), ('lobt', 12), ('protect', 13), ('lszh', 14), ('half', 15)]
Fs = {n: vec(f'{FW}gentle-igloo_v9701{o:02d}.parquet') for n, o in stages}
Os = {n: vec(f'{O8}gentle-igloo_v9902{o:02d}.parquet') for n, o in stages}
O9s = {n: vec(f'{O9}gentle-igloo_v9902{o:02d}.parquet') for n, o in [('e156', 3), ('e141', 4), ('e64', 5), ('e188', 6), ('e152', 10), ('lobt', 12), ('protect', 13), ('half', 15)]}
v78 = vec('submissions/gentle-igloo_v12078.parquet'); v80 = vec('submissions/gentle-igloo_v12080.parquet')
print('O half == v12078', (Os['half'] == v78).all(), ' F half == v12080', (Fs['half'] == v80).all())
tail42 = np.array(json.load(open('analysis/final_tail_assembly_20261008/dryrun_janjul/e156_served.json'))['ids'])
m42 = np.isin(ids, tail42)
# H-chain: old chain with the canary served values at e156 (v12079 chain without E81). Stage outputs of v12078 chain
# are identical to H-chain except at the 42 rows (the 42 rows do not feed other rows' stages).
Hs = {k: v.copy() for k, v in Os.items()}
for k in Hs:
    if stages[[n for n, _ in stages].index(k)][1] >= 3:
        pass
for k in ('e156', 'e141', 'e64', 'e188', 'e152', 'lobt'):
    Hs[k][m42] = O9s[k][m42]
Hs['protect'][m42] = O9s['protect'][m42]; Hs['lszh'][m42] = O9s['protect'][m42]; Hs['half'][m42] = O9s['half'][m42]
v79 = vec('submissions/gentle-igloo_v12079.parquet')
H = v78.copy(); H[m42] = v79[m42]
print('H == Hs[half]', (H == Hs['half']).all())
print('v12079 chain e156 on 42 == v12079 final on 42', (O9s['e156'][m42] == v79[m42]).all())
res = {}
prev = None
print(f"{'stage':8s} {'RMS(F-O)':>9s} {'mean':>7s} {'chg rows F':>10s} {'chg rows O':>10s} {'RMS stage F':>11s} {'RMS stage O':>11s}")
prevF = prevO = None
for n, o in stages:
    d = Fs[n] - Hs[n]
    rF = rms(Fs[n] - prevF) if prevF is not None else float('nan')
    rO = rms(Hs[n] - prevO) if prevO is not None else float('nan')
    cF = int((Fs[n] != prevF).sum()) if prevF is not None else 0
    cO = int((Hs[n] != prevO).sum()) if prevO is not None else 0
    print(f'{n:8s} {rms(d):9.3f} {d.mean():7.3f} {cF:10d} {cO:10d} {rF:11.3f} {rO:11.3f}')
    res[n] = dict(rms_F_minus_H=rms(d), mean=float(d.mean()), changed_rows_F=cF, changed_rows_H=cO, stage_rms_F=rF, stage_rms_H=rO)
    if n == 'lszh':
        continue
    prevF, prevO = Fs[n], Hs[n]
# half stage is relative to protect
np.savez(os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/stages.npz', ids=ids,
         **{'F_' + k: v for k, v in Fs.items()}, **{'H_' + k: v for k, v in Hs.items()})
json.dump(res, open(os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/stages.json', 'w'), indent=1)
