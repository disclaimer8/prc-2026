"""E203 ranking member, seeds 3..9 on the Home GPU: the archived 07.10 recipe, only the seed differs.

The ten-seed mean replaces the three-seed mean as the 0.25-weight NN member of the
v12075 chain (journal №416, candidate v12076). Inputs are the frozen E203 ranking pack
`data/e203_rank_inputs_20261007` and the prepared context tables
`data/e203_rank_prepared_20261007`; the frozen helpers are imported from the pack's own
`scripts/` directory, whose SHA-256 is checked by `W.verify_rank_input`. The call is the
unchanged `S.fit(..., 'ordered', seed)`: width 512 / PLE 48 / d 12 / version B, 2x64 tokens,
batch 4096, AdamW lr .002 / wd 1e-5, OneCycle .1, 8 epochs.

Controls before any fit: PLE bins recomputed from the same numeric tensors must equal the
archived `bins.npz`; the archived `model_0.pt` must reproduce `seed_0.npy` exactly through the
inference path used here; the archived three-seed mean must reproduce `prediction.npz` exactly.
Every new checkpoint is reloaded and must reproduce its own seed array exactly. The runtime
must equal the one admitted for the archived fit. No blend, no ranking chain, no upload.

    .venv-numemb/bin/python -u data/e203_rank_seeds3to9_20261008/scripts/e203_rank_seeds3to9_20261008.py \
        --root /home/a1/prc-2026 [--check-only]
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROTOCOL = 'E203-ordered-rank-member-seeds3to9-tenseed-mean-v12075-chain-20261008-v1'
E203_REG = '6e95e3a65936a3401e41499bb340f2d54a1a2121'
PACK = 'data/e203_rank_inputs_20261007'
OUT = 'data/e203_rank_seeds3to9_20261008'
OLD_SEEDS = (0, 1, 2)
SEEDS = (3, 4, 5, 6, 7, 8, 9)


def modules(root):
    sys.path.insert(0, str(root / PACK / 'scripts'))
    import e203_trajectory_production as W
    return W


def infer(S, net, arrays, tables, pidx):
    """Identical to the prediction loop at the end of S.fit."""
    import torch
    net.eval()
    parts = []
    bs = 4096
    with torch.no_grad():
        for lo in range(0, len(pidx), bs):
            parts.append(net(torch.from_numpy(arrays['xnp'][lo:lo + bs]).to('cuda:0'),
                             torch.from_numpy(arrays['xcp'][lo:lo + bs]).to('cuda:0'),
                             *S.context_batch(tables, pidx[lo:lo + bs])).cpu().numpy())
    return np.concatenate(parts)


def replay(S, path, arrays, s, N, cont, scalar, bins, tables, pidx, expected):
    import torch
    net = S.network(N, s['card'], cont, scalar, bins, arrays['xnf'].shape[1], 'ordered').to('cuda:0')
    net.load_state_dict(torch.load(path, map_location='cuda:0', weights_only=True))
    pred = infer(S, net, arrays, tables, pidx)
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return bool(pred.shape == expected.shape and pred.dtype == expected.dtype and np.array_equal(pred, expected))


def member(seeds, sd, base_p):
    """Exactly the archived arithmetic: float32 seed mean times sd plus proxy."""
    return np.mean(seeds, axis=0) * sd + base_p


def run(root, check_only):
    W = modules(root)
    A, S, E, M, R = W.A, W.S, W.E, W.M, W.R
    S.fixed_env()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    guard = None if check_only else M.verify_cgroup()
    m = W.verify_rank_input(root, E203_REG)
    q = root / W.RANK_INPUT
    s = R.verify(q, W.RANK_REG)['spec']
    archived = W.verify_rank_output(root, E203_REG)
    runtime = E.package_runtime()
    if runtime != archived['runtime'] or runtime != s['admitted_runtime']:
        raise ValueError('runtime differs from the admitted/archived E203 ranking fit')
    prep = W.verify_prepared(root, E203_REG)
    arrays = {n: np.load(q / (n + '.npy'), allow_pickle=False) for n in R.ARRAYS}
    ids = np.load(q / 'mvt_id.npy')
    R.validate(arrays, ids, s)
    tr = np.load(root / W.PACK / 'train_indices.npy')
    pr = np.load(root / W.PACK / 'pred_indices.npy')
    if len(tr) != len(arrays['xnf']) or len(pr) != len(arrays['xnp']):
        raise ValueError('ranking context/query alignment')
    cont, scalar, bins = E.numeric_bins(arrays['xnf'])
    with np.load(root / W.RANK_OUT / 'bins.npz') as ob:
        n_bins = sum(1 for k in ob.files if k.startswith('bin_'))
        if (n_bins != len(bins) or not np.array_equal(ob['continuous'], cont) or not np.array_equal(ob['scalar'], scalar)
                or any(not np.array_equal(ob[f'bin_{i}'], b) for i, b in enumerate(bins))):
            raise ValueError('PLE bins differ from the archived E203 ranking fit')
    old = [np.load(root / W.RANK_OUT / f'seed_{i}.npy', allow_pickle=False) for i in OLD_SEEDS]
    z = E.read(root / W.RANK_OUT / 'prediction.npz')
    if not np.array_equal(z['mvt_id'], ids) or not np.array_equal(member(old, s['sd'], arrays['base_p']), z['reg']):
        raise ValueError('archived three-seed member not reproduced')
    print('CONTROLS PASS: pack/prepared/archived output verified, bins equal, 3-seed member exact', flush=True)
    if check_only:
        print('CHECK ONLY; no GPU, no fit', flush=True)
        return
    out = root / OUT / 'models'
    if (out / 'complete.json').exists():
        raise ValueError('complete output exists; no retry')
    out.mkdir(parents=True, exist_ok=True)
    N = E.trainer(q / 'src')
    tables = W.gpu_tables(root / W.PREPARED)
    started = time.time()
    control = replay(S, root / W.RANK_OUT / 'model_0.pt', arrays, s, N, cont, scalar, bins, tables, pr, old[0])
    if not control:
        raise ValueError('archived model_0.pt does not reproduce seed_0.npy through this inference path')
    print('CONTROL PASS: archived model_0.pt reproduces seed_0.npy exactly', flush=True)
    for seed in SEEDS:
        if (out / f'seed_{seed}.npy').exists() and (out / f'fit_{seed}.json').exists():
            print('SEED EXISTS', seed, flush=True)
            continue
        print('START ordered seed', seed, flush=True)
        r = S.fit(arrays, s, N, cont, scalar, bins, tables, tr, pr, out, 'ordered', seed)
        r['checkpoint_replay_exact'] = replay(S, out / f'model_{seed}.pt', arrays, s, N, cont, scalar, bins, tables, pr,
                                              np.load(out / f'seed_{seed}.npy', allow_pickle=False))
        if not r['checkpoint_replay_exact']:
            raise ValueError(f'checkpoint model_{seed}.pt does not reproduce seed_{seed}.npy')
        A.save(out / f'fit_{seed}.json', r)
    fits = [A.read(out / f'fit_{seed}.json') for seed in SEEDS]
    if [f['seed'] for f in fits] != list(SEEDS) or any(len(f['losses']) != 8 or not np.isfinite(f['losses']).all()
                                                       or not f['checkpoint_replay_exact'] for f in fits):
        raise ValueError('fit inventory incomplete')
    new = [np.load(out / f'seed_{i}.npy', allow_pickle=False) for i in SEEDS]
    if any(a.shape != old[0].shape or a.dtype != old[0].dtype or not np.isfinite(a).all() for a in new):
        raise ValueError('incomplete seed arrays')
    allseeds = old + new
    pred3 = member(old, s['sd'], arrays['base_p'])
    pred10 = member(allseeds, s['sd'], arrays['base_p'])
    if not np.array_equal(pred3, z['reg']) or pred10.shape != pred3.shape or pred10.dtype != pred3.dtype or not np.isfinite(pred10).all():
        raise ValueError('ten-seed member invalid')
    np.savez(out / 'prediction_10seed.npz', mvt_id=ids, reg=pred10, reg2=pred10)
    sec = np.stack([a.astype(np.float64) * s['sd'] for a in allseeds])
    pair = [float(np.sqrt(np.mean((sec[i] - sec[j]) ** 2))) for i in range(10) for j in range(i + 1, 10)]
    stats = dict(sd=s['sd'], rows=int(len(ids)), rms_3_vs_10_sec=float(np.sqrt(np.mean((pred10 - pred3) ** 2))),
                 mean_shift_10_minus_3_sec=float(np.mean(pred10 - pred3)), max_abs_10_minus_3_sec=float(np.abs(pred10 - pred3).max()),
                 pairwise_seed_rms_sec=dict(min=min(pair), mean=float(np.mean(pair)), max=max(pair)),
                 pairwise_seed_rms_std_units=dict(min=min(pair) / s['sd'], mean=float(np.mean(pair)) / s['sd'], max=max(pair) / s['sd']),
                 per_seed_mean_sec=[float(x) for x in sec.mean(axis=1)])
    A.save(out / 'statistics.json', stats)
    A.save(out / 'complete.json', dict(
        protocol=PROTOCOL, registration_commit=E203_REG, input_spec_sha256=m['spec_sha256'], rows=int(len(ids)),
        seeds=list(SEEDS), all_seeds=list(OLD_SEEDS + SEEDS), arm='ordered', epochs=8, runtime=runtime, cgroup=guard,
        fits=fits, seconds=time.time() - started,
        bins_equal_archived=True, archived_member_exact=True, control_model0_replay_exact=True,
        prepared_complete_sha256=A.sha(root / W.PREPARED / 'complete.json'),
        archived_rank_complete_sha256=A.sha(root / W.RANK_OUT / 'complete.json'),
        archived_prediction_sha256=A.sha(root / W.RANK_OUT / 'prediction.npz'),
        statistics=stats, uploaded=False,
        files={p.name: A.sha(p) for p in out.iterdir() if p.is_file() and p.name != 'complete.json'}))
    print('COMPLETE ORDERED RANKING SEEDS 3..9; rms(3 vs 10) %.2f s; mean shift %+.3f s' % (
        stats['rms_3_vs_10_sec'], stats['mean_shift_10_minus_3_sec']), flush=True)
    del tables
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    run(a.root.resolve(), a.check_only)


if __name__ == '__main__':
    main()
