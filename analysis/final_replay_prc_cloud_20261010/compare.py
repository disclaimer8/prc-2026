"""Journal 453/454: compare the prc-cloud replay of the final assembly with the Mac build, stage by stage.

Read-only on both sides. Receipts are compared field by field without the volatile keys (wall-clock
`unix`/`seconds`, and `job_sha256`, whose job JSON carries absolute paths). Arrays and stage outputs are
compared bitwise and, where they differ, by value.

    .venv/bin/python analysis/final_replay_prc_cloud_20261010/compare.py \
        --mac-analysis analysis/final_tail_20261009 --mac-work data/final_tail_20261009/work \
        --cloud data/final_replay_prc_cloud_20261010/work_prc_cloud_20261010 \
        --out analysis/final_replay_prc_cloud_20261010/comparison.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
VOLATILE = {'unix', 'seconds', 'second_run_seconds', 'job_sha256'}
FINAL_SHA = '76c32426d750040690ff68bacd6ecbf3a7aa1e63340df5b9271a324b7064d554'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def diff(a, b, path=''):
    """Leaf-level differences between two JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [dict(field=path, mac=a, cloud=b)]


def compare_receipts(mac_dir, cloud_dir):
    rows, first = [], None
    for p in sorted(Path(mac_dir).glob('*.json')):
        q = Path(cloud_dir) / p.name
        if not q.exists():
            rows.append(dict(receipt=p.name, cloud='absent'))
            first = first or p.name
            continue
        d = diff(strip(json.loads(p.read_text())), strip(json.loads(q.read_text())))
        rows.append(dict(receipt=p.name, identical_without_volatile=not d, differences=d))
        if d and first is None:
            first = p.name
    return rows, first


def frame_diff(a_path, b_path):
    a, b = pl.read_parquet(a_path).sort(ID), pl.read_parquet(b_path).sort(ID)
    if a.height != b.height or not np.array_equal(a[ID].to_numpy(), b[ID].to_numpy()):
        return dict(same_ids=False)
    d = b[Y].cast(pl.Float64).to_numpy() - a[Y].cast(pl.Float64).to_numpy()
    return dict(same_ids=True, rows_differing=int((d != 0).sum()), max_abs_diff=float(np.abs(d).max()))


def compare_outputs(mac_work, cloud_work):
    out = []
    for p in sorted((Path(cloud_work) / 'submissions').glob('gentle-igloo_v*.parquet')):
        q = Path(mac_work) / 'submissions' / p.name
        if not q.exists():
            out.append(dict(file=p.name, mac='absent'))
            continue
        same = sha(p) == sha(q)
        item = dict(file=p.name, sha256_cloud=sha(p), sha256_mac=sha(q), bitwise_identical=same)
        if not same:
            item.update(frame_diff(q, p))
        out.append(item)
    for p in sorted((Path(cloud_work) / 'submissions').glob('parts_v*.npz')):
        q = Path(mac_work) / 'submissions' / p.name
        item = dict(file=p.name, bitwise_identical=sha(p) == sha(q), arrays={})
        with np.load(p) as zc, np.load(q) as zm:
            for k in zm.files:
                a, b = zm[k], zc[k]
                eq = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
                r = dict(identical=bool(eq))
                if not eq and a.shape == b.shape:
                    dd = np.abs(np.nan_to_num(a.astype(np.float64)) - np.nan_to_num(b.astype(np.float64)))
                    r.update(elements_differing=int((dd != 0).sum()), max_abs_diff=float(dd.max()))
                item['arrays'][k] = r
        out.append(item)
    return out


def compare_e156(mac_work, cloud_work):
    res = {}
    mac_run = Path(mac_work) / 'e156/run_0'
    for run in sorted((Path(cloud_work) / 'e156').glob('run_*')):
        cap_m, cap_c = json.loads((mac_run / 'capture.json').read_text()), json.loads((run / 'capture.json').read_text())
        srv_m, srv_c = json.loads((mac_run / 'served.json').read_text()), json.loads((run / 'served.json').read_text())
        seeds = []
        for s in cap_m['seeds']:
            a, b = np.load(mac_run / f'seed_{s}_proba.npy'), np.load(run / f'seed_{s}_proba.npy')
            cm = next(c for c in cap_m['calls'] if c['seed'] == s)
            cc = next(c for c in cap_c['calls'] if c['seed'] == s)
            seeds.append(dict(seed=s, proba_bitwise_identical=bool(np.array_equal(a, b)), max_abs_proba_diff=float(np.abs(a - b).max()),
                              argmax_rows_differing=int((a.argmax(1) != b.argmax(1)).sum()),
                              fit_x_same=cm['fit_x'] == cc['fit_x'], fit_y_same=cm['fit_y'] == cc['fit_y'], query_x_same=cm['query_x'] == cc['query_x'],
                              classes_same=cm['classes'] == cc['classes'], n_estimators=[cm['n_estimators'], cc['n_estimators']]))
        am, ac = np.load(mac_run / 'averaged_proba.npy'), np.load(run / 'averaged_proba.npy')
        vm, vc = np.array(srv_m['values']), np.array(srv_c['values'])
        res[run.name] = dict(seeds=seeds, averaged_max_abs_diff=float(np.abs(am - ac).max()),
                             served_ids_identical=srv_m['ids'] == srv_c['ids'] and srv_m['ids_sha256'] == srv_c['ids_sha256'],
                             served_values_identical=bool(np.array_equal(vm, vc)), served_rows_differing=int((vm != vc).sum()),
                             served_max_abs_diff=float(np.abs(vm - vc).max()), base_values_identical=srv_m['base_values'] == srv_c['base_values'],
                             versions=dict(mac=dict(tabpfn=cap_m['tabpfn'], torch=cap_m['torch'], python=cap_m['python'], threads=cap_m['threads']),
                                           cloud=dict(tabpfn=cap_c['tabpfn'], torch=cap_c['torch'], python=cap_c['python'], threads=cap_c['threads'])))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mac-analysis', type=Path, required=True)
    p.add_argument('--mac-work', type=Path, required=True)
    p.add_argument('--cloud', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    receipts, first = compare_receipts(a.mac_analysis / 'receipts', a.cloud / 'receipts')
    complete = a.cloud / 'complete.json'
    failure = a.cloud / 'failure.json'
    final = dict(mac_sha256=json.loads((a.mac_analysis / 'complete.json').read_text())['output_sha256'], expected=FINAL_SHA)
    if complete.exists():
        c = json.loads(complete.read_text())
        final.update(cloud_sha256=c['output_sha256'], identical=c['output_sha256'] == FINAL_SHA, fits_performed=c['fits_performed'])
    result = dict(final=final, failure=json.loads(failure.read_text()) if failure.exists() else None, first_differing_receipt=first,
                  receipts=receipts, outputs=compare_outputs(a.mac_work, a.cloud), e156=compare_e156(a.mac_work, a.cloud))
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(final=final, failure=result['failure'], first_differing_receipt=first,
                          receipts=[(r['receipt'], r.get('identical_without_volatile', r.get('cloud'))) for r in receipts]), indent=1))


if __name__ == '__main__':
    main()
