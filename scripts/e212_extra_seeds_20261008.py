"""E212: seeds 3..9 of the ordered-trajectory member on the six 2025 month-pair folds (Home GPU).

Identical fold recipe to E197 (folds 1/4) and E202 (folds 2/3/5/6): same query bundles,
same prepared context tables, same `S.fit(..., 'ordered', seed)` with 8 epochs, batch 4096,
AdamW lr .002 / wd 1e-5, OneCycle .1; only the seed differs (3..9). Seeds 0..2 are not
refitted: their saved arrays must reproduce the original fold `member` exactly, and the
10-seed mean is written next to them for the Mac chain evaluation (journal №407).
No ranking, no submission.
"""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import e197_trajectory_encoder as S
E, M, C = S.E, S.M, S.C

PROTOCOL = 'E212-nn-weight-10seed-analytic-lopo-decided-grid005-v1'
SEEDS = (3, 4, 5, 6, 7, 8, 9)
OUT = 'data/e212_seeds3to9_20261008'


def places(fold):
    old = fold in (1, 4)
    return (Path('data/e197_inputs_20261007' if old else 'data/e202_inputs_20261007'),
            Path('data/e197_prepared_20261007' if old else 'data/e202_prepared_20261007'),
            Path('data/e197_trajectory_20261007' if old else 'data/e202_trajectory_20261007'))


def run(root, folds):
    S.fixed_env(); import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); guard = M.verify_cgroup()
    for fold in folds:
        name = f'fold_{fold}_{fold + 6}'; bundle, prepared, orig = (root / p for p in places(fold))
        od = root / OUT / name / 'ordered'; src = orig / name / 'ordered'
        if (od / 'complete.json').exists(): print('DONE ALREADY', name, flush=True); continue
        m = C.read_json(bundle / 'manifest.json'); fs = next(f for f in m['spec']['folds'] if f['fold'] == fold)
        qb = root / fs['query_bundle']; s = M.verify_bundle(qb, E.OLD_REGISTRATION)['spec']; arrays, _ = M.load_arrays(qb, s)
        tr, pr = np.load(bundle / name / 'train_indices.npy'), np.load(bundle / name / 'pred_indices.npy')
        if len(tr) != len(arrays['xnf']) or len(pr) != len(arrays['xnp']): raise ValueError('query/context shapes differ')
        cont, scalar, bins = E.numeric_bins(arrays['xnf']); N = E.trainer(qb / 'src')
        with np.load(src / 'bins.npz') as ob:
            if (not np.array_equal(ob['continuous'], cont) or not np.array_equal(ob['scalar'], scalar)
                    or any(not np.array_equal(ob[f'bin_{i}'], b) for i, b in enumerate(bins))): raise ValueError('PLE bins differ from the original fold')
        z = E.read(src / 'prediction.npz'); ci = arrays['control_indices']
        old = [np.load(src / f'seed_{i}.npy') for i in (0, 1, 2)]
        if not np.array_equal((np.mean(old, axis=0) * s['sd'] + arrays['base_p'])[ci], z['member']): raise ValueError('original 3-seed member not reproduced')
        tables = S.gpu_tables(prepared / name); od.mkdir(parents=True, exist_ok=True); fits = []
        for seed in SEEDS:
            if (od / f'seed_{seed}.npy').exists() and (od / f'fit_{seed}.json').exists(): print('SEED EXISTS', name, seed, flush=True); continue
            print('START', name, 'ordered seed', seed, flush=True)
            r = S.fit(arrays, s, N, cont, scalar, bins, tables, tr, pr, od, 'ordered', seed)
            C.write_json(od / f'fit_{seed}.json', r); fits.append(r)
        fits = [C.read_json(od / f'fit_{seed}.json') for seed in SEEDS]
        new = [np.load(od / f'seed_{i}.npy') for i in SEEDS]
        if any(a.shape != old[0].shape or not np.isfinite(a).all() for a in new): raise ValueError('incomplete seed arrays')
        allseeds = old + new
        per_seed = np.stack([(a.astype(np.float64) * s['sd'] + arrays['base_p'])[ci].astype(np.float32) for a in allseeds])
        member10 = (np.mean(allseeds, axis=0) * s['sd'] + arrays['base_p'])[ci]
        np.savez_compressed(od / 'prediction_10seed.npz', mvt_id=z['mvt_id'], y=z['y'], month=z['month'], member3=z['member'], member10=member10, per_seed_sec=per_seed)
        C.write_json(od / 'complete.json', dict(protocol=PROTOCOL, fold=fold, arm='ordered', seeds=list(SEEDS), sd=s['sd'], cgroup=guard, fits=fits,
                                                original_member_exact=True, rms_3_vs_10_sec=float(np.sqrt(np.mean((member10 - z['member']) ** 2))),
                                                files={p.name: C.sha(p) for p in od.iterdir() if p.is_file() and p.name != 'complete.json'}))
        print('COMPLETE', name, 'rms(3 vs 10) %.2f s' % float(np.sqrt(np.mean((member10 - z['member']) ** 2))), flush=True)
        del arrays, tables, per_seed; gc.collect(); torch.cuda.empty_cache()
    print('ALL FOLDS COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--folds', default='1,2,3,4,5,6')
    a = p.parse_args(); run(a.root.resolve(), [int(x) for x in a.folds.split(',')])
