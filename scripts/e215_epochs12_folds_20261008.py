"""E215: the ordered-trajectory member E203 refitted with 12 epochs instead of 8 on the six 2025
month-pair folds, seeds 0,1,2 (Home GPU). Registered in HYPOTHESES_LOG №428.

Identical fold recipe to E197 (folds 1/4) / E202 (folds 2/3/5/6) / E212 (seeds 3..9): same query
bundles, same prepared context tables, same frozen sibling sources, same PLE bins (checked against
the fold's bins.npz), same `S.fit(..., 'ordered', seed)` network, batch 4096, AdamW lr .002 /
wd 1e-5, OneCycleLR(max_lr .002, pct_start .1) — the ONLY change is the epoch count: the frozen
`S.fit` hard-codes 8 (`steps=8*(n//bs)`, `for ep in range(8)`), so `fit_epochs` below is the
frozen source with exactly those three literals replaced by EPOCHS (asserted, recorded in the
receipt). OneCycleLR therefore spans 12*(n//bs) steps: 10 % warm-up (1.2 epochs instead of 0.8),
then the same cosine anneal stretched over the remaining 10.8 epochs — the code's own behaviour.
Seeds 0..2 of the 8-epoch member are not refitted: their saved arrays must reproduce the original
fold `member` exactly; the 12-epoch 3-seed mean is written next to it for the Mac chain evaluation.
No ranking, no submission, no edits to frozen files.
"""
from __future__ import annotations
import argparse, gc, hashlib, inspect, json, time
from pathlib import Path
import numpy as np
import e197_trajectory_encoder as S
E, M, C = S.E, S.M, S.C

PROTOCOL = 'E215-ordered-member-epochs12-vs-8-3seed-6pair-chain-w035-v1'
SEEDS = (0, 1, 2)
EPOCHS = 12
OUT = 'data/e215_epochs12_20261008'
LITERALS = ('steps=8*(n//bs)', 'for ep in range(8):', "epoch{ep+1}/8")


def patched_fit():
    """The frozen S.fit with the epoch literals replaced; everything else byte-identical."""
    src = inspect.getsource(S.fit)
    for lit in LITERALS:
        if src.count(lit) != 1: raise ValueError('frozen S.fit changed: literal not unique ' + lit)
    new = (src.replace('def fit(', 'def fit_epochs(EPOCHS,').replace(LITERALS[0], 'steps=EPOCHS*(n//bs)')
              .replace(LITERALS[1], 'for ep in range(EPOCHS):').replace(LITERALS[2], 'epoch{ep+1}/{EPOCHS}'))
    if new.count('8') != src.count('8') - 3: raise ValueError('unexpected literal 8 in the frozen S.fit')
    ns = dict(S.__dict__); exec(compile(new, S.__file__ + '#fit_epochs', 'exec'), ns)
    return ns['fit_epochs'], dict(frozen_fit_sha256=hashlib.sha256(src.encode()).hexdigest(),
                                  patched_fit_sha256=hashlib.sha256(new.encode()).hexdigest(), patched_source=new)


def places(fold):
    old = fold in (1, 4)
    return (Path('data/e197_inputs_20261007' if old else 'data/e202_inputs_20261007'),
            Path('data/e197_prepared_20261007' if old else 'data/e202_prepared_20261007'),
            Path('data/e197_trajectory_20261007' if old else 'data/e202_trajectory_20261007'))


def run(root, folds):
    S.fixed_env(); import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); guard = M.verify_cgroup()
    fit_epochs, patch = patched_fit()
    siblings = {p.name: C.sha(p) for p in Path(__file__).resolve().parent.glob('*.py')}
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
        old = [np.load(src / f'seed_{i}.npy') for i in SEEDS]
        if not np.array_equal((np.mean(old, axis=0) * s['sd'] + arrays['base_p'])[ci], z['member']): raise ValueError('original 3-seed member not reproduced')
        tables = S.gpu_tables(prepared / name); od.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            if (od / f'seed_{seed}.npy').exists() and (od / f'fit_{seed}.json').exists(): print('SEED EXISTS', name, seed, flush=True); continue
            print('START', name, 'ordered seed', seed, 'epochs', EPOCHS, flush=True)
            r = fit_epochs(EPOCHS, arrays, s, N, cont, scalar, bins, tables, tr, pr, od, 'ordered', seed)
            if len(r['losses']) != EPOCHS: raise ValueError('epoch count')
            r['rms_vs_8epoch_same_seed_sec'] = float(np.sqrt(np.mean((np.load(od / f'seed_{seed}.npy').astype(np.float64) - old[seed]) ** 2)) * s['sd'])
            C.write_json(od / f'fit_{seed}.json', r)
            print('FIT', name, seed, '%.0f s' % r['seconds'], 'final mse %.6f' % r['losses'][-1], 'rms vs 8-epoch seed %.2f s' % r['rms_vs_8epoch_same_seed_sec'], flush=True)
        fits = [C.read_json(od / f'fit_{seed}.json') for seed in SEEDS]
        new = [np.load(od / f'seed_{i}.npy') for i in SEEDS]
        if any(a.shape != old[0].shape or not np.isfinite(a).all() for a in new): raise ValueError('incomplete seed arrays')
        per_seed = np.stack([(a.astype(np.float64) * s['sd'] + arrays['base_p'])[ci].astype(np.float32) for a in new])
        member12 = (np.mean(new, axis=0) * s['sd'] + arrays['base_p'])[ci]
        rms = float(np.sqrt(np.mean((member12 - z['member']) ** 2)))
        np.savez_compressed(od / 'prediction_12ep.npz', mvt_id=z['mvt_id'], y=z['y'], month=z['month'], member3_8ep=z['member'], member3_12ep=member12, per_seed_sec=per_seed)
        C.write_json(od / 'complete.json', dict(protocol=PROTOCOL, fold=fold, arm='ordered', seeds=list(SEEDS), epochs=EPOCHS, sd=s['sd'], cgroup=guard, fits=fits,
                                                original_member_exact=True, bins_equal_original=True, rms_8ep_vs_12ep_sec=rms,
                                                mean_shift_12_minus_8_sec=float(np.mean(member12 - z['member'])),
                                                patch={k: v for k, v in patch.items() if k != 'patched_source'}, patched_source=patch['patched_source'],
                                                siblings_sha256=siblings, original_prediction_sha256=C.sha(src / 'prediction.npz'),
                                                files={p.name: C.sha(p) for p in od.iterdir() if p.is_file() and p.name != 'complete.json'}))
        print('COMPLETE', name, 'rms(8ep vs 12ep) %.2f s' % rms, flush=True)
        del arrays, tables, per_seed; gc.collect(); torch.cuda.empty_cache()
    print('ALL FOLDS COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--folds', default='1,2,3,4,5,6')
    a = p.parse_args(); run(a.root.resolve(), [int(x) for x in a.folds.split(',')])
