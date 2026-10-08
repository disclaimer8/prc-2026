"""Inference-only token ablation of the saved E197/E202 ordered fold checkpoints (no fitting).
For each fold: predict held-out rows with the real trajectory context and with the context zeroed
(present=known=values=0, i.e. exactly how a natural no-token row enters the network)."""
import os, sys, json, time, gc
from pathlib import Path
import numpy as np
os.chdir('/home/a1/prc-2026'); sys.path.insert(0, '/home/a1/prc-2026/data/e202_inputs_20261007/scripts')
import torch
import e197_trajectory_encoder as S
E = S.E
OUT = Path('/home/a1/claude_cov_audit'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)

def load_bins(path):
    with np.load(path, allow_pickle=False) as z:
        cont = z['continuous'].tolist(); scalar = z['scalar'].tolist()
        return cont, scalar, [z[f'bin_{j}'] for j in range(len(cont))]

def predict(net, arrays, tables, pidx, masked):
    parts = []
    with torch.no_grad():
        for lo in range(0, len(pidx), 4096):
            ctx = S.context_batch(tables, pidx[lo:lo + 4096])
            if masked:
                ctx = tuple(torch.zeros_like(t) for t in ctx)
            parts.append(net(torch.from_numpy(arrays['xnp'][lo:lo + 4096]).to('cuda:0'),
                             torch.from_numpy(arrays['xcp'][lo:lo + 4096]).to('cuda:0'), *ctx).cpu().numpy())
    return np.concatenate(parts)

folds = [int(x) for x in sys.argv[1:]] or [6, 1, 2, 3, 4, 5]
for fold in folds:
    name = f'fold_{fold}_{fold + 6}'; t0 = time.monotonic()
    old = fold in (1, 4)
    qb = Path(f'data/mlp_split_20261005/inputs/{name}')
    prepared = Path('data/e197_prepared_20261007' if old else 'data/e202_prepared_20261007') / name
    od = Path('data/e197_trajectory_20261007' if old else 'data/e202_trajectory_20261007') / name / 'ordered'
    inp = Path('data/e197_inputs_20261007' if old else 'data/e202_inputs_20261007') / name
    spec = json.load(open(qb / 'manifest.json'))['spec']
    arrays = {n: np.load(qb / f'{n}.npy', allow_pickle=False) for n in ('xnp', 'xcp', 'base_p', 'control_indices')}
    pidx = np.load(inp / 'pred_indices.npy', allow_pickle=False)
    assert len(pidx) == len(arrays['xnp'])
    cont, scalar, knots = load_bins(od / 'bins.npz')
    N = E.trainer(qb / 'src')
    tables = S.gpu_tables(prepared)
    z = np.load(od / 'prediction.npz', allow_pickle=False)
    real, masked = [], []
    for seed in (0, 1, 2):
        net = S.network(N, spec['card'], cont, scalar, knots, arrays['xnp'].shape[1], 'ordered').to('cuda:0')
        net.load_state_dict(torch.load(od / f'model_{seed}.pt', map_location='cuda:0', weights_only=True)); net.eval()
        r = predict(net, arrays, tables, pidx, False); m = predict(net, arrays, tables, pidx, True)
        saved = np.load(od / f'seed_{seed}.npy', allow_pickle=False)
        print(name, 'seed', seed, 'replay max|diff| vs saved', float(np.abs(r - saved).max()), 'exact', bool(np.array_equal(r, saved)), flush=True)
        real.append(r); masked.append(m); del net; torch.cuda.empty_cache()
    ci = arrays['control_indices']; sd = spec['sd']
    member_real = (np.mean(real, axis=0) * sd + arrays['base_p'])[ci]
    member_masked = (np.mean(masked, axis=0) * sd + arrays['base_p'])[ci]
    print(name, 'member replay max|diff|', float(np.abs(member_real - z['member']).max()), 'rows', len(ci), flush=True)
    np.savez_compressed(OUT / f'ablation_{name}.npz', mvt_id=z['mvt_id'], y=z['y'], month=z['month'], member_real=member_real,
                        member_masked=member_masked, member_saved=z['member'],
                        masked_seeds=np.stack(masked)[:, ci].astype(np.float32), real_seeds=np.stack(real)[:, ci].astype(np.float32))
    del tables, arrays, z; gc.collect(); torch.cuda.empty_cache()
    print(name, 'done in', round(time.monotonic() - t0, 1), 's', flush=True)
print('ALL DONE', flush=True)
