"""Step 08_classifier_2x2 of the v12080 control explanation (journal 446). Intermediate files live in $V12080_SCRATCH (default: the session scratchpad)."""
"""2x2: {old clf_v870, final classifier} x {old FIN1b view, final FIN1b view} on the Jan/Jul rows."""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, polars as pl
os.chdir('/Users/denyskolomiiets/prc-2026')
sys.path.insert(0, 'analysis/v12080_control_explanation_20261009')
import swap_chain as S
SP = os.environ.get('V12080_SCRATCH', '/private/tmp/claude-501/-Users-denyskolomiiets/237b9694-d169-4305-ab5c-7ff3144974ed/scratchpad') + '/'
W, R = S.WORK, S.ROOT
fin = (W / 'data/features_fin1b.parquet'); old = (R / 'data/features_fin1b.parquet')
cn = (W / 'submissions/classifier.txt', W / 'submissions/classifier_state.json'); co = (R / 'submissions/clf_v870.txt', None)
res = {}
for fl, fp in (('Fnew', fin), ('Fold', old)):
    for cl, (cp, stp) in (('Cnew', cn), ('Cold', co)):
        ids, p, cand, d = S.raw_proba(fp, cp, stp, W / 'src')
        res[(fl, cl)] = (ids, p)
        print(fl, cl, p.shape, flush=True)
ids0 = res[('Fold', 'Cold')][0]
def on(k):
    ids, p = res[k]
    idx = np.argsort(ids)[np.searchsorted(np.sort(ids), ids0)]
    assert (ids[idx] == ids0).all()
    return p[idx]
P = {k: on(k) for k in res}
meta = np.load(SP + 'meta.npz')
mids = meta['ids']; midx = np.argsort(ids0)[np.searchsorted(np.sort(ids0), mids)]
P = {k: v[midx] for k, v in P.items()}
lirf_nonm = (meta['ap'] == 'LIRF') & meta['nonm']
tv = lambda a, b: np.abs(a - b).sum(axis=1) / 2
out = {}
for name, a, b in (('classifier (old view)', ('Fold', 'Cnew'), ('Fold', 'Cold')), ('classifier (final view)', ('Fnew', 'Cnew'), ('Fnew', 'Cold')),
                   ('view (old clf)', ('Fnew', 'Cold'), ('Fold', 'Cold')), ('view (final clf)', ('Fnew', 'Cnew'), ('Fold', 'Cnew')),
                   ('total', ('Fnew', 'Cnew'), ('Fold', 'Cold'))):
    t = tv(P[a], P[b])
    dps = P[a][:, 1] - P[b][:, 1]
    out[name] = dict(mean_tv=float(t.mean()), p99_tv=float(np.quantile(t, .99)), mean_tv_lirf_nonm=float(t[lirf_nonm].mean()),
                     mean_dpsched_lirf_nonm=float(dps[lirf_nonm].mean()), frac_neg_dpsched_lirf_nonm=float((dps[lirf_nonm] < 0).mean()))
    print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}))
json.dump(out, open(SP + 'clf2x2.json', 'w'), indent=1)
np.savez(SP + 'clf2x2.npz', ids=mids, **{f'{a}_{b}': v for (a, b), v in P.items()})
