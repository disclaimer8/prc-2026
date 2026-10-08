import numpy as np, sys, json
out = sys.argv[1]
res = {}
for tag, d in [('cv', '/home/a1/prc-2026/data/e202_prepared_20261007/fold_6_12'), ('final', '/home/a1/prc-nnfinal-20261008/out/context')]:
    ids = np.load(d + '/query_ids.npy')
    pr = np.load(d + '/present.npy', mmap_mode='r')
    sup = np.load(d + '/support.npy') if tag == 'cv' else None
    n = len(ids); cnt = np.zeros((n, 2), np.uint8)
    for b in range(0, n, 200000):
        cnt[b:b+200000] = pr[b:b+200000].sum(axis=2)
    res[tag + '_ids'] = ids.astype(np.float64); res[tag + '_ntok'] = cnt
    if sup is not None: res[tag + '_support'] = sup
    print(tag, n, 'rows; stream0 none', int((cnt[:, 0] == 0).sum()), 'stream1 none', int((cnt[:, 1] == 0).sum()), 'both none', int((cnt.sum(1) == 0).sum()), flush=True)
np.savez_compressed(out, **res)
print('saved', out)
