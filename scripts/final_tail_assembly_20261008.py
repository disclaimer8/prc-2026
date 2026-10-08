"""Final uniform tail assembly: fresh members -> frozen premix -> E185/E167 -> fresh E156 ->
cached E141/E64 -> E188 -> E152 -> LOBT lift/clip -> protect all served E156 IDs -> LSZH half.

No E81, no 42 historical donors, no old Jan/Jul prediction splice, no upload. Frozen stage
sources are executed unchanged in child processes (two hash-bound row-count transformations
only). Cached LightGBM stage models are reused only on exact training-signature match;
refit is a separate explicit action. Mac: `.venv/bin/python`, <=6 OpenMP threads.

    PY=.venv/bin/python
    $PY scripts/final_tail_assembly_20261008.py make-contract --mode janjul --out data/<x>/contract.json
    $PY scripts/final_tail_assembly_20261008.py prepare --contract data/<x>/contract.json --work data/<x>/work
    $PY scripts/final_tail_assembly_20261008.py assemble --work data/<x>/work [--repeat-e156]
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('OMP_THREAD_LIMIT', '6')
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
import numpy as np
import polars as pl

ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
PROTOCOL = 'final-uniform-tail-E156fresh-noE81-fourmonths-20261008-v1'
SELF = 'scripts/final_tail_assembly_20261008.py'
ARCHIVE = 'data/today_chain_20261006'            # frozen stage source closure (setup.json source_after)
STAGE_CACHE = 'data/e203_chain_20261007/models'  # 11 cached E141/E64/LSZH models with training signatures
E188_SCRIPT = 'scripts/e188_rank_20261006.py'
E188_MODELS = 'data/e188_rank_20261006_v2'
E188_REG = 'd45cb738fb7671cf289ae8a400a30b5b46dfbb3d'
TABPFN_SHA = 'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
TABPFN_CKPT = 'data/tabpfn_v2_checkpoint/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'  # repo copy: macOS purged ~/Library/Caches on 08.10
DAY = 86400.0
CAT_W = .65
# Mixture weights by contract mode. janjul = the v12090 canary recipe (journal 389-391), kept for replay;
# final = neural member 0.35 with the five anchors scaled by 0.65/0.75, the tree pair keeps 1 - sum (E212,
# journal 407a/410/425: lower decided SSE than 0.25 in all twelve 2025 months, 3 and 10 seeds).
WEIGHTS = {
    'janjul': ([('raw', '0.0333333333333333'), ('lobt', '.10'), ('sched', '.10'), ('eobt', '.10'), ('aobt', '.10')], '0.25'),
    'final': ([('raw', '0.028888888888888857'), ('lobt', '0.08666666666666667'), ('sched', '0.08666666666666667'),
               ('eobt', '0.08666666666666667'), ('aobt', '0.08666666666666667')], '0.35'),
}
ANCHORS, NN_W = WEIGHTS['janjul']  # member names (and the canary weights)
E185_W = .35 * 2 / 3
STAGE_SEEDS = '42,1,2,3,4'
D7 = 'n_pts,sf_n_seg,vi_n_pts,vi_prev_pts,vi_alt_first,vm_n_pts,vm_prev_pts'
D3 = 'vi_n_pts,vi_prev_pts,vi_alt_first'
TABPFN_SEEDS = list(range(10))
EXPECTED_SEEDS = {'cat': 15, 'lgb': 5, 'raw': 5, 'lobt': 5, 'sched': 5, 'eobt': 5, 'aobt': 5, 'motion': 5, 'nn': 3}
PRODUCTS = [('adsblol_kt1', 'offblock2'), ('adsblol_kt1', 'offblock3'), ('adsblol_kt1', 'offblock5'),
            ('adsblol_sw_kt1', 'offblock2'), ('adsblol', 'visit'), ('adsblol', 'visit_m'), ('adsblol', 'head')]
HISTORICAL_PRODUCTS = PRODUCTS + [('adsblol', 'offblock2')]  # archived 2025 inventory also binds 3 kt offblock2
FROZEN_SOURCES = ['train.py', 'features.py', 'target_encoding.py', 'e185_apply.py', 'e167_nm_window_rule.py',
                  'lirf_tabpfn.py', 'lirf_table.py', 'e141_apply.py', 'e36_adsb_stand_geometry.py', 'e69_visit_merge.py',
                  'adsb_visit.py', 'geometry.py', 'e134_head_features.py', 'adsb_offblock_v2.py', 'adsb_offblock_v3.py',
                  'e64_apply.py', 'e64_visit_stage.py', 'fold_metrics.py', 'e152_stage_gate.py', 'lobt_edge_lift.py',
                  'blend_submissions.py', 'e159b_apply.py', 'e159_opdi_stage_cv.py']
CACHED = {'e141': ('e141_apply.py', 5, 'FEATS'), 'e64': ('e64_apply.py', 5, 'FEATS'), 'lszh': ('e159b_apply.py', 1, 'F')}
BASE_VERSION = 970100
FORBIDDEN = ('e81', 'blend_submissions.py:44', 'v44', 'v60', 'v291', 'v361', 'donor')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False)
        f.write('\n')


def ids_sha(ids):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, dtype='<f8')).tobytes()).hexdigest()


def fingerprint(value):
    if value is None:
        return None
    x = np.ascontiguousarray(value)
    if x.dtype.hasobject:
        raise ValueError('object-valued array')
    return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())


def environment(threads=6):
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
    env.update(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads), OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', POLARS_MAX_THREADS='2', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0',
               PYTHONUNBUFFERED='1', POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR='load_as_storage',
               FEATURES='data/features_e106.parquet', PROXY_SCALE='1', CLF_COST_WEIGHT='1', CLF_COST_MIN_SPREAD='3600',
               DROP_PREFIX='day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config', HF_HUB_OFFLINE='1')
    return env


# ----------------------------------------------------------------------------- stage plan

def stage_plan(mode='final'):
    """Literal uniform recipe. Versions are offsets from the private base version."""
    b = BASE_VERSION
    anchors, nn_w = WEIGHTS[mode]
    members = ','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m, w in anchors) + f',submissions/nn_rank.npz:{nn_w}'
    return [
        dict(name='premix', kind='native', out=b, weights=dict(cat=CAT_W, members=members)),
        dict(name='e185', kind='script', args=['e185_apply.py', b, b + 1], env={'E185_W': str(E185_W), 'E185_RANK': 'data/e185_motion_rank.npz'}),
        dict(name='e167', kind='script', args=['e167_nm_window_rule.py', b + 1, b + 2], env={}),
        dict(name='e156', kind='tabpfn', args=['lirf_tabpfn.py', b + 2, b + 3],
             env={'TABPFN_EXTRA': 'pfx', 'TABPFN_SEEDS': ','.join(map(str, TABPFN_SEEDS)), 'TABPFN_ALLOW_CPU_LARGE_DATASET': '1',
                  'PFX_FEATURES': 'data/features_e113.parquet'}),
        dict(name='e141', kind='cached', args=['e141_apply.py', b + 3, b + 4], cache='e141',
             env={'STAGE_SEEDS': STAGE_SEEDS, 'ADSB_DIR': 'data/adsblol_kt1', 'PARTS': f'submissions/parts_v{b + 2}.npz',
                  'STAGE_KEEP_ALL': '1', 'DROP_FEATS': D7, 'WALK_DIR': 'data/adsblol_sw_kt1', 'E36_BASE_DIR': 'data/cv_preds_e116prod'}),
        dict(name='e64', kind='cached', args=['e64_apply.py', b + 4, b + 5], cache='e64',
             env={'STAGE_SEEDS': STAGE_SEEDS, 'ADSB_DIR': 'data/adsblol_kt1', 'PARTS': f'submissions/parts_v{b + 2}.npz',
                  'E64_KEEP_ALL': '1', 'DROP_FEATS': D3}),
        dict(name='e188', kind='e188', pre=b + 4, post=b + 5, parts=b + 2, out=b + 6),
        dict(name='e152', kind='script', args=['e152_stage_gate.py', b + 3, b + 6, b + 2, b + 10], env={'E152_GATE': '0.95'}),
        dict(name='lobt', kind='script', args=['lobt_edge_lift.py', b + 10, b + 12], env={'LOBT_CLIP': '1'}),
        dict(name='protect', kind='native', final=b + 12, tab=b + 3, out=b + 13),
        dict(name='lszh', kind='cached', args=['e159b_apply.py', b + 13, b + 14], cache='lszh',
             env={'E159_APS': 'LSZH', 'E159_MONTHS': 'all', 'E159_BASE': 'data/cv_preds_e150c_e64_keep', 'E159_SHIFT': '0'}),
        dict(name='half', kind='script', args=['blend_submissions.py', b + 15], env={'BLEND_BASE': str(b + 13), 'BLEND_PARTS': f'{b + 14}:0.5'}),
    ]


def check_plan(plan, mode='final'):
    names = [s['name'] for s in plan]
    expected = ['premix', 'e185', 'e167', 'e156', 'e141', 'e64', 'e188', 'e152', 'lobt', 'protect', 'lszh', 'half']
    if names != expected:
        raise ValueError('stage order differs from the registered uniform recipe')
    text = json.dumps(plan).lower()
    for token in FORBIDDEN:
        if token in text:
            raise ValueError('forbidden historical component in plan: ' + token)
    premix = plan[0]['weights']
    members = [x.rsplit(':', 1) for x in premix['members'].split(',')]
    anchors, nn_w = WEIGHTS[mode]
    if premix['cat'] != CAT_W or [w for _, w in members] != [w for _, w in anchors] + [nn_w]:
        raise ValueError('mixture weights differ from the registered recipe')
    if sum(m[0].endswith('nn_rank.npz') for m in members) != 1:
        raise ValueError('the neural member enters exactly once')
    if plan[1]['env']['E185_W'] != str(.35 * 2 / 3) or plan[3]['env']['TABPFN_SEEDS'] != '0,1,2,3,4,5,6,7,8,9' \
            or plan[3]['env']['TABPFN_EXTRA'] != 'pfx' or plan[7]['env']['E152_GATE'] != '0.95' or plan[11]['env']['BLEND_PARTS'].endswith(':0.5') is False:
        raise ValueError('stage parameters differ from the registered recipe')
    if 'E81' in names or any(s.get('kind') == 'script' and s['args'][0] == 'blend_submissions.py' and '44' in s['env'].get('BLEND_PARTS', '') for s in plan):
        raise ValueError('E81 is excluded')
    return plan


# ----------------------------------------------------------------------------- contracts

def member_table(path, expected_seeds=None, require_per_seed=True):
    """Load one member npz; return ids and reg plus a receipt. Full seed sets only."""
    with np.load(path, allow_pickle=False) as z:
        if 'mvt_id' not in z.files or 'reg' not in z.files:
            raise ValueError('member lacks mvt_id/reg: ' + str(path))
        ids, reg = z['mvt_id'].astype(np.float64), z['reg'].astype(np.float64)
        receipt = dict(path=str(path), rows=int(len(ids)), sha256=sha(path), fields=list(z.files))
        if 'per_seed' in z.files:
            per = z['per_seed']
            receipt['seeds'] = int(per.shape[0])
            if expected_seeds is not None and per.shape[0] != expected_seeds:
                raise ValueError(f'{path}: {per.shape[0]} seeds, expected {expected_seeds}')
            if not np.allclose(per.mean(axis=0), reg, rtol=0, atol=1e-6):
                raise ValueError('member reg is not the mean of its seeds: ' + str(path))
        elif require_per_seed and expected_seeds is not None:
            raise ValueError('member has no per-seed evidence: ' + str(path))
        if 'complete' in z.files and not bool(z['complete']):
            raise ValueError('member marked incomplete: ' + str(path))
    if len(ids) != len(np.unique(ids)) or not np.isfinite(ids).all() or not np.isfinite(reg).all():
        raise ValueError('member IDs duplicate/nonfinite or predictions nonfinite: ' + str(path))
    return ids, reg, receipt


def check_completion(kind, comp, expected, npz_name, npz_sha):
    """The completion receipt must list exactly seeds 0..expected-1 and, if it lists the prediction file, its SHA."""
    if isinstance(comp.get('all_seeds'), list):  # NN seeds 3..9 run: its receipt lists the full 0..9 set
        listed = comp['all_seeds']
    else:
        listed = comp.get('seeds') if isinstance(comp.get('seeds'), list) else comp.get('fits', [])
    seeds = [x['seed'] if isinstance(x, dict) else int(x) for x in listed]
    if sorted(seeds) != list(range(expected)):
        raise ValueError(f'{kind}: completion receipt lists seeds {sorted(seeds)}')
    listed_sha = comp.get('files', {}).get(npz_name)
    if isinstance(listed_sha, str) and listed_sha != npz_sha:
        raise ValueError(f'{kind}: prediction SHA differs from its completion receipt')


def align(ids, member_ids, values):
    """Return values ordered like ids; every id must exist exactly once (blend.py dict semantics)."""
    order = np.argsort(member_ids, kind='stable')
    pos = np.searchsorted(member_ids[order], ids)
    if pos.max(initial=-1) >= len(order) or not np.array_equal(member_ids[order][np.clip(pos, 0, len(order) - 1)], ids):
        raise ValueError('member does not cover every prediction ID')
    return values[order][pos]


def relative(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('contract path must be relative and inside the repo: ' + name)
    return root / p


def admit(root, contract_path):
    """Verify the pinned contract: files, member seed sets, population, derivative days, caches."""
    root = Path(root).resolve()
    c = read(contract_path)
    if c.get('protocol') != PROTOCOL:
        raise ValueError('wrong contract protocol')
    if c['rows'] <= 0 or c['train_rows'] != 2085047:
        raise ValueError('population contract')
    for name, h in c['files'].items():
        p = relative(root, name)
        if not p.is_file() or sha(p) != h:
            raise ValueError('missing or changed pinned input: ' + name)
    for key in ('features_fin1b', 'features_e106', 'features_e113', 'ranking', 'template'):
        if c['inputs'][key] not in c['files']:
            raise ValueError('unpinned input ' + key)
    template = pl.read_parquet(relative(root, c['inputs']['template']))
    if template.columns != [ID, Y] or template.height != c['rows'] or template[ID].n_unique() != c['rows']:
        raise ValueError('template rows/columns differ from the contract')
    tpl_ids = template[ID].to_numpy().astype(np.float64)
    feat = pl.read_parquet(relative(root, c['inputs']['features_fin1b']), columns=[ID, 'split', Y])
    rank = feat.filter(pl.col('split') == 'rank')
    if rank.height != c['rows'] or rank[ID].n_unique() != c['rows'] or feat.filter(pl.col('split') == 'train').height != c['train_rows']:
        raise ValueError('feature split counts differ from the contract')
    if rank[Y].null_count() != c['rows']:
        raise ValueError('prediction rows carry labels')
    if not np.array_equal(np.sort(rank[ID].to_numpy().astype(np.float64)), np.sort(tpl_ids)):
        raise ValueError('feature prediction IDs differ from the template')
    rank_ids = rank[ID].to_numpy().astype(np.float64)
    if ids_sha(rank_ids) != c['rank_ids_sha256']:
        raise ValueError('ordered prediction IDs differ from the contract')
    members = {}
    for kind, spec in c['members'].items():
        if kind == 'base':
            continue
        if spec['npz'] not in c['files']:
            raise ValueError('unpinned member ' + kind)
        expected = c.get('nn_seeds', EXPECTED_SEEDS['nn']) if kind == 'nn' else EXPECTED_SEEDS[kind]
        ids, reg, receipt = member_table(relative(root, spec['npz']), expected, require_per_seed=kind != 'nn')
        align(rank_ids, ids, reg)
        if spec.get('complete'):
            receipt['complete_sha256'] = c['files'][spec['complete']]
            check_completion(kind, read(relative(root, spec['complete'])), expected, Path(spec['npz']).name, receipt['sha256'])
        members[kind] = receipt
    if set(members) != set(EXPECTED_SEEDS):
        raise ValueError('missing model family: ' + str(set(EXPECTED_SEEDS) - set(members)))
    base = c['members']['base']
    if base['classifier'] not in c['files'] or (base.get('state') and base['state'] not in c['files']):
        raise ValueError('unpinned classifier')
    if base.get('masked_proba_source'):
        raise ValueError('masked base probabilities must not be an input; the saved classifier is')
    deriv = c['derivatives']
    if deriv['ranking_sha256'] != c['files'][c['inputs']['ranking']]:
        raise ValueError('daily derivatives were produced for another ranking file')
    days = deriv['days']
    if len(days) != len(set(days)) or sorted(days) != days:
        raise ValueError('derivative day list must be sorted and unique')
    for folder, product in PRODUCTS:
        for day in days:
            name = f'{deriv["root"]}/{folder}/{product}_{day}.parquet'
            if name not in c['files']:
                raise ValueError('missing daily product: ' + name)
    for folder, product in HISTORICAL_PRODUCTS:
        n = sum(1 for k in c['files'] if re.fullmatch(rf'data/{folder}/{product}_2025-\d\d-\d\d\.parquet', k))
        if n != 364:
            raise ValueError(f'historical 2025 inventory of {folder}/{product} is {n}, expected 364')
    for stage, (_, calls, _) in CACHED.items():
        rec = relative(root, f'{STAGE_CACHE}/{stage}/complete.json')
        if f'{STAGE_CACHE}/{stage}/complete.json' not in c['files']:
            raise ValueError('unpinned stage cache ' + stage)
        r = read(rec)
        if len(r['calls']) != calls:
            raise ValueError('cached stage call inventory ' + stage)
        for i, item in enumerate(r['calls']):
            if c['files'].get(f'{STAGE_CACHE}/{stage}/model_{i}.txt') != item['sha256']:
                raise ValueError('cached stage model not pinned/changed ' + stage)
    m = read(relative(root, f'{E188_MODELS}/fit_manifest.json'))
    if m['registration_commit'] != E188_REG or not m['fit_complete'] or m['rounds'] != 400:
        raise ValueError('E188 fit manifest')
    for name, h in m['model_sha256'].items():
        if c['files'].get(f'{E188_MODELS}/{name}') != h:
            raise ValueError('E188 model not pinned/changed ' + name)
    if c['files'].get(E188_SCRIPT) != m['source_sha256'][E188_SCRIPT]:
        raise ValueError('E188 apply source differs from its fit registration')
    ckpt = checkpoint_path(root, c['tabpfn']['checkpoint'])
    if not ckpt.is_file() or sha(ckpt) != TABPFN_SHA or c['tabpfn']['sha256'] != TABPFN_SHA or c['files'].get(c['tabpfn']['checkpoint']) != TABPFN_SHA:
        raise ValueError('TabPFN V2 checkpoint missing, unpinned or changed')
    setup = read(relative(root, ARCHIVE + '/setup.json'))
    for name in FROZEN_SOURCES:
        key = f'{ARCHIVE}/src/{name}'
        if c['files'].get(key) != setup['source_after']['src/' + name]:
            raise ValueError('frozen source not pinned to the archived closure: ' + name)
    if not c['opdi']['files'] or any(k not in c['files'] for k in c['opdi']['files']):
        raise ValueError('OPDI event inventory not pinned')
    opdi_bases = [Path(k).name for k in c['opdi']['files']]
    if len(set(opdi_bases)) != len(opdi_bases) or not all(b.startswith('ev_') and b.endswith('.parquet') for b in opdi_bases):
        raise ValueError('OPDI event files must be unique ev_*.parquet basenames (they share <work>/data/opdi)')
    return c, dict(members=members, template_rows=int(template.height), rank_ids_sha256=ids_sha(rank_ids))


# ----------------------------------------------------------------------------- workspace

def checkpoint_path(root, name):
    p = Path(name).expanduser()
    return p if p.is_absolute() else relative(root, name)


def transform_e159b(source, rows):
    tree = ast.parse(source)
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 344841]
    if len(hits) != 1 or source.count('344841') != 1:
        raise ValueError('E159b fixed row-count assertion must occur exactly once')
    stmt = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert) and any(h is x for x in ast.walk(n) for h in hits))
    if not isinstance(stmt, ast.Assert):
        raise ValueError('row count is not inside the assertion')
    return source.replace('344841', str(int(rows)))


E188_BRIDGE = '''def validate_registration(root, commit):
    """Registration-receipt bridge: the original apply source and this transformed copy are hash-bound."""
    b = json.loads(os.environ["E188_BRIDGE"])
    if commit != b["registration"]:
        raise RuntimeError("E188 registration differs")
    if sha(b["original_source"]) != b["original_sha256"] or sha(Path(__file__).resolve()) != b["transformed_sha256"]:
        raise RuntimeError("E188 apply source/bridge changed")
    return {b["original_name"]: b["original_sha256"]}
'''


def transform_e188(source, rows):
    tree = ast.parse(source)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'ROWS']
    if len(assigns) != 1 or assigns[0].value.value != 344841 or source.count('ROWS = 344841') != 1:
        raise ValueError('E188 ROWS constant must be assigned exactly once')
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_registration']
    if len(funcs) != 1:
        raise ValueError('E188 validate_registration must be defined exactly once')
    lines = source.splitlines(keepends=True)
    f = funcs[0]
    out = ''.join(lines[:f.lineno - 1]) + E188_BRIDGE + ''.join(lines[f.end_lineno:])
    if out.count('"E188_BRIDGE"') != 1:
        raise ValueError('bridge insertion failed')
    return out.replace('ROWS = 344841', f'ROWS = {int(rows)}')


def link(target, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(Path(target).resolve())


def prepare(root, contract_path, work):
    """Immutable private workspace: links to pinned inputs, frozen sources, sealed receipt."""
    root, work = Path(root).resolve(), Path(work).resolve()
    if work.exists():
        raise FileExistsError('fresh workspace required; no overwrite/resume')
    if shutil.disk_usage(root).free < 4 * 1024 ** 3:
        raise ValueError('4 GiB disk floor at preparation')
    c, evidence = admit(root, contract_path)
    plan = check_plan(stage_plan(c.get('mode', 'final')), c.get('mode', 'final'))
    work.mkdir(parents=True)
    for d in ('src', 'submissions', 'logs', 'jobs', 'receipts', 'e156', 'data/prc-2026-datasets'):
        (work / d).mkdir(parents=True)
    # datasets: final ranking/template under the standard names the frozen stages read
    link(root / c['inputs']['ranking'], work / 'data/prc-2026-datasets/ranking.parquet')
    link(root / c['inputs']['template'], work / 'data/prc-2026-datasets/submitting.parquet')
    for name in c['files']:
        if re.fullmatch(r'data/prc-2026-datasets/training_2025-\d\d-\d\d_20\d\d-\d\d-\d\d\.parquet', name):
            link(root / name, work / name)
        elif re.fullmatch(r'data/(cv_preds_e116prod|cv_preds_nomonth|cv_preds_e150c_e64_keep)/fold_\d+_\d+\.npz', name):
            link(root / name, work / name)
        elif re.fullmatch(r'data/(adsblol|adsblol_kt1|adsblol_sw_kt1)/[a-z0-9_]+_2025-\d\d-\d\d\.parquet', name):
            link(root / name, work / name)
    for name in c['opdi']['files']:
        link(root / name, work / 'data/opdi' / Path(name).name)
    deriv = c['derivatives']
    for folder, product in PRODUCTS:
        for day in deriv['days']:
            link(root / f'{deriv["root"]}/{folder}/{product}_{day}.parquet', work / f'data/{folder}/{product}_{day}.parquet')
    for key, dest in (('features_fin1b', 'data/features_fin1b.parquet'), ('features_e106', 'data/features_e106.parquet'),
                      ('features_e113', 'data/features_e113.parquet')):
        link(root / c['inputs'][key], work / dest)
    for kind, spec in c['members'].items():
        if kind == 'base':
            link(root / spec['classifier'], work / 'submissions/classifier.txt')
            if spec.get('state'):
                link(root / spec['state'], work / 'submissions/classifier_state.json')
        elif kind == 'motion':
            link(root / spec['npz'], work / 'data/e185_motion_rank.npz')
        elif kind == 'nn':
            link(root / spec['npz'], work / 'submissions/nn_rank.npz')
        elif kind in ('cat', 'lgb'):
            link(root / spec['npz'], work / f'submissions/{kind}_rank.npz')
        else:
            link(root / spec['npz'], work / f'submissions/e130_{kind}f_rank.npz')
    sources = {}
    for name in FROZEN_SOURCES:
        src = root / ARCHIVE / 'src' / name
        text = src.read_text()
        before = sha(src)
        if name == 'e159b_apply.py':
            text = transform_e159b(text, c['rows'])
        (work / 'src' / name).write_text(text)
        sources[name] = dict(before=before, after=sha(work / 'src' / name), transformed=name == 'e159b_apply.py')
    e188_src = root / E188_SCRIPT
    (work / 'src/e188_apply_dynamic.py').write_text(transform_e188(e188_src.read_text(), c['rows']))
    sources['e188_apply_dynamic.py'] = dict(before=sha(e188_src), after=sha(work / 'src/e188_apply_dynamic.py'), transformed=True,
                                            original=E188_SCRIPT)
    save(work / 'sealed.json', dict(protocol=PROTOCOL, contract=str(Path(contract_path).resolve()), contract_sha256=sha(contract_path),
         adapter_sha256=sha(Path(__file__).resolve()), rows=c['rows'], base_version=BASE_VERSION, plan=plan, sources=sources,
         evidence=evidence, tabpfn_checkpoint=str(checkpoint_path(root, c['tabpfn']['checkpoint'])), output_name=c['output_name'],
         mode=c.get('mode', 'final'), created_unix=time.time(), python=sys.version, numpy=np.__version__, polars=pl.__version__))
    print('WORKSPACE READY', work, 'rows', c['rows'], flush=True)


# ----------------------------------------------------------------------------- premix

def classifier_matrix(frame, clf, categorical, state=None):
    """Exactly train.to_pandas semantics: model feature order, training category levels."""
    names = clf.feature_name()
    columns = []
    for n in names:
        match = [c for c in frame.columns if c.replace(' ', '_') == n]
        if len(match) != 1:
            raise ValueError('classifier feature absent from the feature table: ' + n)
        columns.append(match[0])
    cat_cols = [c for c in columns if c in categorical]
    if state is not None:
        if list(state['columns']) != columns:
            raise ValueError('saved classifier state columns differ from the model')
        levels = {c: state['categories'][c] for c in cat_cols}
    else:
        if len(clf.pandas_categorical or []) != len(cat_cols):
            raise ValueError('classifier category count differs from the categorical columns')
        levels = dict(zip(cat_cols, clf.pandas_categorical))
    if clf.pandas_categorical and [levels[c] for c in cat_cols] != list(clf.pandas_categorical):
        raise ValueError('category levels differ from those stored in the classifier')
    pdf = frame.select(columns).to_pandas()
    for c in cat_cols:
        pdf[c] = pdf[c].astype('category').cat.set_categories(levels[c])
    return pdf, columns, levels


def premix_arrays(proba_raw, r_cat, r_lgb, members, candidates, d_lobt, w_cat=CAT_W):
    """Frozen blend.py arithmetic on arrays. members = [(weight_literal, array)] in order raw,lobt,sched,eobt,aobt,nn."""
    LOBT_WINDOW = 3606
    r = w_cat * r_cat + (1 - w_cat) * r_lgb
    weights = [float(w) for w, _ in members]
    r = (1 - sum(weights)) * r + sum(w_ * g for w_, g in zip(weights, [g for _, g in members]))
    values = np.column_stack([r] + list(candidates) + [r + DAY])
    proba = np.where(np.isnan(values), 0.0, proba_raw)
    d = d_lobt[:, None]
    bad = ~np.isnan(d) & (np.abs(np.nan_to_num(values) - np.nan_to_num(d)) > LOBT_WINDOW)
    bad[:, 0] = False
    out = np.where(bad, 0.0, proba)
    s = out.sum(axis=1, keepdims=True)
    proba = np.where(s > 0, out / np.where(s > 0, s, 1), proba)
    proba = proba / proba.sum(axis=1, keepdims=True)
    mix = np.sum(proba * np.nan_to_num(values), axis=1)
    lo, hi = d_lobt - LOBT_WINDOW, d_lobt + LOBT_WINDOW
    pred = np.clip(np.where(np.isnan(d_lobt), mix, np.clip(mix, lo, hi)), 0, None)
    return dict(reg=r, proba=proba, values=values, pred=pred)


def premix(work):
    work = Path(work)
    sealed = read(work / 'sealed.json')
    stage = sealed['plan'][0]
    b = stage['out']
    import lightgbm as lgb
    sys.path.insert(0, str(work / 'src'))
    import features as F  # frozen closure
    import train as T
    clf = lgb.Booster(model_file=str(work / 'submissions/classifier.txt'))
    state = read(work / 'submissions/classifier_state.json') if (work / 'submissions/classifier_state.json').exists() else None
    schema = pl.scan_parquet(work / 'data/features_fin1b.parquet').collect_schema()
    model_cols = [c for c in schema if c.replace(' ', '_') in set(clf.feature_name())]
    extra = [ID, 'split', *T.CANDIDATES.values()]
    frame = pl.read_parquet(work / 'data/features_fin1b.parquet', columns=[c for c in schema if c in set(model_cols + extra)]).filter(pl.col('split') == 'rank')
    ids = frame[ID].to_numpy().astype(np.float64)
    if ids_sha(ids) != sealed['evidence']['rank_ids_sha256']:
        raise ValueError('prediction row order changed since admission')
    xr, columns, levels = classifier_matrix(frame, clf, F.CATEGORICAL, state)
    proba_raw = clf.predict(xr, num_threads=int(os.environ.get('OMP_NUM_THREADS', '6')))
    if proba_raw.shape != (frame.height, 6) or not np.isfinite(proba_raw).all():
        raise ValueError('classifier output shape/finite')
    del xr
    load = lambda name: member_table(work / 'submissions' / name, require_per_seed=False)
    cat_ids, cat_reg, _ = load('cat_rank.npz')
    lgb_ids, lgb_reg, _ = load('lgb_rank.npz')
    r_cat, r_lgb = align(ids, cat_ids, cat_reg), align(ids, lgb_ids, lgb_reg)
    members = []
    for item in stage['weights']['members'].split(','):
        path, w = item.rsplit(':', 1)
        m_ids, m_reg, _ = member_table(work / path, require_per_seed=False)
        members.append((w, align(ids, m_ids, m_reg)))
    cands = [frame[c].to_numpy().astype(np.float64) for c in T.CANDIDATES.values()]
    parts = premix_arrays(proba_raw, r_cat, r_lgb, members, cands, frame['d_mvt_lobt'].to_numpy().astype(np.float64), stage['weights']['cat'])
    out_npz = work / f'submissions/parts_v{b}.npz'
    if out_npz.exists():
        raise FileExistsError(out_npz)
    np.savez(out_npz, mvt_id=ids, reg=parts['reg'], proba=parts['proba'], values=parts['values'], pred=parts['pred'])
    template = pl.read_parquet(work / 'data/prc-2026-datasets/submitting.parquet')
    preds = pl.DataFrame({ID: frame[ID], Y: parts['pred']})
    sub = template.select(ID).join(preds, on=ID, how='left')
    if sub.height != template.height or sub[Y].null_count():
        raise ValueError('premix output incomplete')
    sub = sub.with_columns(pl.col(Y).round(0).cast(template.schema[Y]))
    sub.write_parquet(work / f'submissions/gentle-igloo_v{b}.parquet')
    save(work / 'receipts/00_premix.json', dict(stage='premix', rows=int(frame.height), classifier_sha256=sha(work / 'submissions/classifier.txt'),
         classifier_columns=len(columns), categorical=list(levels), raw_proba_sha256=fingerprint(proba_raw)['sha256'],
         proba_masked_from_raw_classifier=True, parts_sha256=sha(out_npz), output_sha256=sha(work / f'submissions/gentle-igloo_v{b}.parquet'),
         member_order=[m.split(':')[0] for m in stage['weights']['members'].split(',')], unix=time.time()))
    print('PREMIX', frame.height, 'rows; mean pred', float(parts['pred'].mean()), flush=True)


# ----------------------------------------------------------------------------- child stages

def sub(work, v):
    return Path(work) / f'submissions/gentle-igloo_v{v}.parquet'


def read_sub(path):
    d = pl.read_parquet(path)
    if d.columns != [ID, Y] or d[Y].null_count() or d[ID].n_unique() != d.height:
        raise ValueError('invalid stage output ' + str(path))
    return d


def run_child(work, index, name, command, extra_env, log_name):
    work = Path(work)
    if shutil.disk_usage(work).free < 2500 * 1024 ** 2:
        raise ValueError('disk floor 2.5 GiB violated before ' + name)
    log = work / 'logs' / f'{index:02d}_{log_name}.log'
    print('STAGE', index, name, flush=True)
    with log.open('x') as f:
        subprocess.run(command, cwd=work, env={**environment(), **extra_env}, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, check=True)


def change_receipt(work, name, index, src_v, out_v, extra=None):
    work = Path(work)
    a, b = read_sub(sub(work, src_v)).sort(ID), read_sub(sub(work, out_v)).sort(ID)
    if not np.array_equal(a[ID].to_numpy(), b[ID].to_numpy()):
        raise ValueError('stage changed the ID set/order: ' + name)
    d = b[Y].cast(pl.Float64).to_numpy() - a[Y].cast(pl.Float64).to_numpy()
    if (b[Y].cast(pl.Float64).to_numpy() < 0).any():
        raise ValueError('negative prediction after ' + name)
    r = dict(stage=name, input_version=src_v, output_version=out_v, input_sha256=sha(sub(work, src_v)), output_sha256=sha(sub(work, out_v)),
             rows=int(b.height), changed_rows=int((d != 0).sum()), rms_change=float(np.sqrt(np.mean(d ** 2))), max_abs_change=float(np.abs(d).max()),
             unix=time.time(), evidence=extra or {})
    save(work / 'receipts' / f'{index:02d}_{name}.json', r)
    return r


def script_stage(work, index, stage):
    args = stage['args']
    run_child(work, index, stage['name'], [sys.executable, '-u', 'src/' + args[0], *map(str, args[1:])], stage['env'], Path(args[0]).stem)


def cached_stage(work, index, stage, mode):
    work = Path(work)
    source, calls, feature_global = CACHED[stage['cache']]
    if mode not in ('replay', 'refit'):
        raise ValueError('cached stage mode')
    sealed = read(work / 'sealed.json')
    job = work / 'jobs' / f'{index:02d}_{stage["cache"]}.json'
    root = Path(__file__).resolve().parents[1]
    save(job, dict(mode=mode, name=stage['cache'], args=stage['args'], expected_calls=calls, feature_global=feature_global,
                   cache=str(root / STAGE_CACHE / stage['cache']), refit_cache=str(work / 'models' / stage['cache']),
                   source_sha256=sealed['sources'][source]['after'], adapter_sha256=sha(Path(__file__).resolve())))
    run_child(work, index, stage['name'] + '/' + mode, [sys.executable, '-u', str(Path(__file__).resolve()), '_stage', '--work', str(work), '--job', str(job)],
              stage['env'], stage['cache'] + '_' + mode)
    receipt = read(job.with_suffix('.complete.json'))
    if receipt['fits_performed'] != (0 if mode == 'replay' else calls):
        raise ValueError('stage fit inventory differs from the mode')
    return receipt


def stage_worker(work, job_path):
    """Fresh LightGBM-only process: run the frozen stage, intercept its lgb.train calls."""
    if 'torch' in sys.modules:
        raise ValueError('LightGBM stage must run without Torch')
    import inspect
    import lightgbm as lgb
    work, job = Path(work), read(job_path)
    source = work / 'src' / job['args'][0]
    if sha(source) != job['source_sha256'] or sha(Path(__file__).resolve()) != job['adapter_sha256']:
        raise ValueError('stage source/adapter changed during execution')
    cache = Path(job['cache'])
    stored = read(cache / 'complete.json')
    if len(stored['calls']) != job['expected_calls']:
        raise ValueError('cached call inventory differs')
    if job['mode'] == 'refit':
        refit = Path(job['refit_cache'])
        refit.mkdir(parents=True, exist_ok=False)
    original, calls = lgb.train, []

    def wrapped(params, dataset, num_boost_round=100, **kwargs):
        if kwargs:
            raise ValueError('unexpected training-call options')
        caller = inspect.currentframe().f_back
        features = caller.f_locals.get('feats', caller.f_globals.get(job['feature_global']))
        i = len(calls)
        if i >= job['expected_calls']:
            raise ValueError('unexpected extra training call')
        item = stored['calls'][i]
        signature = dict(params=params, rounds=num_boost_round, data=fingerprint(dataset.data), label=fingerprint(dataset.label),
                         weight=fingerprint(dataset.weight), categorical=dataset.categorical_feature, feature_names=list(features or []),
                         dataset_feature_names=dataset.feature_name, lightgbm=lgb.__version__, numpy=np.__version__, polars=pl.__version__,
                         python=sys.version)
        if list(features or []) != item['signature']['feature_names'] or dataset.data.shape[1] != len(item['signature']['feature_names']):
            raise ValueError('stage feature names/order differ from the cached training call')
        if job['mode'] == 'replay':
            if signature != item['signature']:
                diff = [k for k in signature if signature[k] != item['signature'].get(k)]
                raise ValueError('cached stage training signature differs in ' + ','.join(diff) + '; explicit fixed refit required')
            path = cache / f'model_{i}.txt'
            if sha(path) != item['sha256']:
                raise ValueError('cached stage model changed')
            model = lgb.Booster(model_file=str(path))
            calls.append(dict(signature=signature, sha256=item['sha256'], fitted=False))
        else:
            model = original(params, dataset, num_boost_round)
            path = Path(job['refit_cache']) / f'model_{i}.txt'
            model.save_model(str(path))
            calls.append(dict(signature=signature, sha256=sha(path), fitted=True,
                              cached_signature_match=signature == item['signature']))
        if model.current_iteration() != num_boost_round:
            raise ValueError('incomplete stage model')
        return model

    lgb.train = wrapped
    try:
        os.chdir(work)
        sys.path.insert(0, str(work / 'src'))
        sys.argv = [str(source), *map(str, job['args'][1:])]
        runpy.run_path(str(source), run_name='__main__')
    finally:
        lgb.train = original
    if len(calls) != job['expected_calls']:
        raise ValueError('stage call count differs')
    fits = sum(c['fitted'] for c in calls)
    if job['mode'] == 'refit':
        save(Path(job['refit_cache']) / 'complete.json', dict(protocol=PROTOCOL, source_sha256=job['source_sha256'], calls=calls, mode='refit'))
    save(Path(job_path).with_suffix('.complete.json'), dict(stage=job['name'], mode=job['mode'], cached_calls=len(calls) - fits,
         fits_performed=fits, models=[c['sha256'] for c in calls], job_sha256=sha(job_path), unix=time.time()))


# ----------------------------------------------------------------------------- E156 fresh TabPFN

def served_predicate(work):
    """LIRF, no NM record, delay bins of the tail table; ordered as lirf_tabpfn.py serves them."""
    sys.path.insert(0, str(Path(work) / 'src'))
    import lirf_table as L
    f = pl.read_parquet(Path(work) / 'data/features_e106.parquet', columns=[ID, 'split', 'ADEP_mvt', 'FLIGHT_ID_mvt', 'd_mvt_sched'])
    r = f.filter(pl.col('split') == 'rank').filter((pl.col('ADEP_mvt') == 'LIRF') & pl.col('FLIGHT_ID_mvt').is_null()).with_columns(
        b=pl.col('d_mvt_sched').cut(L.BINS, labels=L.LABELS)).filter(pl.col('b').is_in(L.OVERRIDE))
    return r[ID].to_numpy().astype(np.float64), r['b'].cast(pl.String).to_list()


def tabpfn_worker(work, job_path):
    """Fresh process: run frozen lirf_tabpfn.py with ten ordered seeds; capture per-seed evidence."""
    work, job = Path(work), read(job_path)
    import torch  # before lightgbm (libomp)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '6')))
    import tabpfn
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    ckpt = Path(job['checkpoint'])
    if sha(ckpt) != TABPFN_SHA:
        raise ValueError('TabPFN checkpoint differs')
    out = Path(job['capture'])
    out.mkdir(parents=True, exist_ok=False)
    seeds, calls = job['seeds'], []
    original = TabPFNClassifier.create_default_for_version.__func__

    def wrapped(cls, version, **kw):
        i = len(calls)
        if version != ModelVersion.V2 or kw.get('device') != 'cpu' or i >= len(seeds) or kw.get('random_state') != seeds[i]:
            raise ValueError(f'unexpected TabPFN call {i}: {version} {kw}')
        est = original(cls, version, model_path=str(ckpt), **kw)
        item = dict(seed=seeds[i], kwargs={k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kw.items()},
                    n_estimators=est.n_estimators, model_path=str(ckpt))
        fit0, pp0 = est.fit, est.predict_proba

        def fit(X, y, *a, **k):
            item['fit_x'], item['fit_y'] = fingerprint(np.asarray(X)), fingerprint(np.asarray(y))
            return fit0(X, y, *a, **k)

        def predict_proba(X, *a, **k):
            p = pp0(X, *a, **k)
            item['query_x'] = fingerprint(np.asarray(X))
            item['classes'] = [int(c) for c in est.classes_]
            item['proba'] = fingerprint(p)
            np.save(out / f'seed_{seeds[i]}_proba.npy', np.asarray(p))
            return p
        est.fit, est.predict_proba = fit, predict_proba
        calls.append(item)
        return est
    TabPFNClassifier.create_default_for_version = classmethod(wrapped)
    os.chdir(work)
    sys.argv = ['src/lirf_tabpfn.py', *map(str, job['args'][1:])]
    runpy.run_path(str(work / 'src/lirf_tabpfn.py'), run_name='__main__')
    if [c['seed'] for c in calls] != seeds or any('proba' not in c for c in calls):
        raise ValueError('TabPFN seed sequence incomplete')
    n = calls[0]['proba']['shape'][0]
    if any(c['proba']['shape'][0] != n or c['fit_x'] != calls[0]['fit_x'] or c['query_x'] != calls[0]['query_x'] for c in calls):
        raise ValueError('TabPFN seeds saw different arrays')
    ids, bins = served_predicate(work)
    if len(ids) != n:
        raise ValueError(f'served predicate {len(ids)} rows but TabPFN scored {n}')
    base, new = read_sub(sub(work, job['args'][1])), read_sub(sub(work, job['args'][2]))
    j = pl.DataFrame({ID: ids, 'b': bins}).join(new, on=ID, how='left', maintain_order='left').join(base.rename({Y: 'base'}), on=ID, how='left', maintain_order='left')
    if j[Y].null_count() or j['base'].null_count():
        raise ValueError('served IDs missing from stage outputs')
    other = new.join(pl.DataFrame({ID: ids}), on=ID, how='anti').sort(ID)
    prev = base.join(pl.DataFrame({ID: ids}), on=ID, how='anti').sort(ID)
    if not other.equals(prev):
        raise ValueError('TabPFN stage changed rows outside the served predicate')
    averaged = np.mean([np.load(out / f'seed_{s}_proba.npy') for s in seeds], axis=0)
    np.save(out / 'averaged_proba.npy', averaged)
    np.save(out / 'served_ids.npy', ids)
    save(out / 'served.json', dict(protocol=PROTOCOL, rows=int(n), ids=[float(x) for x in ids], bins=bins,
         values=[int(v) for v in j[Y].to_list()], base_values=[int(v) for v in j['base'].to_list()],
         unchanged_rounded=int((j[Y] == j['base']).sum()), ids_sha256=ids_sha(ids)))
    save(out / 'capture.json', dict(protocol=PROTOCOL, seeds=seeds, calls=calls, tabpfn=tabpfn.__version__, torch=torch.__version__,
         checkpoint=str(ckpt), checkpoint_sha256=TABPFN_SHA, offline=os.environ.get('HF_HUB_OFFLINE'), threads=torch.get_num_threads(),
         averaged_proba=fingerprint(averaged), python=sys.version, unix=time.time()))


def tabpfn_stage(work, index, stage, repeat=False):
    work = Path(work)
    sealed = read(work / 'sealed.json')
    runs = []
    for k, (base, new) in enumerate([(stage['args'][1], stage['args'][2])] + ([(stage['args'][1], 999903)] if repeat else [])):
        job = work / 'jobs' / f'{index:02d}_e156_{k}.json'
        save(job, dict(args=[stage['args'][0], base, new], seeds=TABPFN_SEEDS, checkpoint=sealed['tabpfn_checkpoint'], capture=str(work / 'e156' / f'run_{k}')))
        started = time.time()
        run_child(work, index, f'e156/run{k}', [sys.executable, '-u', str(Path(__file__).resolve()), '_tabpfn', '--work', str(work), '--job', str(job)],
                  {**stage['env'], 'TABPFN_MODEL_CACHE_DIR': str(Path(sealed['tabpfn_checkpoint']).parent)}, f'e156_run{k}')
        runs.append(dict(capture=read(work / 'e156' / f'run_{k}/capture.json'), served=read(work / 'e156' / f'run_{k}/served.json'), seconds=time.time() - started))
    receipt = dict(served_rows=runs[0]['served']['rows'], served_ids_sha256=runs[0]['served']['ids_sha256'], seeds=TABPFN_SEEDS,
                   seconds=runs[0]['seconds'], unchanged_rounded_rows=runs[0]['served']['unchanged_rounded'],
                   per_seed_proba=[c['proba']['sha256'] for c in runs[0]['capture']['calls']], n_estimators=runs[0]['capture']['calls'][0]['n_estimators'])
    if repeat:
        a = [np.load(work / 'e156/run_0' / f'seed_{s}_proba.npy') for s in TABPFN_SEEDS]
        b = [np.load(work / 'e156/run_1' / f'seed_{s}_proba.npy') for s in TABPFN_SEEDS]
        receipt['determinism'] = dict(max_abs_proba_diff_per_seed=[float(np.abs(x - y).max()) for x, y in zip(a, b)],
                                      bitwise_identical=all(np.array_equal(x, y) for x, y in zip(a, b)),
                                      rounded_values_identical=runs[0]['served']['values'] == runs[1]['served']['values'],
                                      rounded_rows_differing=int(sum(x != y for x, y in zip(runs[0]['served']['values'], runs[1]['served']['values']))),
                                      second_run_seconds=runs[1]['seconds'])
        sub(work, 999903).unlink()
    return receipt


# ----------------------------------------------------------------------------- E188, protect, finalize

def e188_stage(work, index, stage):
    work = Path(work)
    sealed = read(work / 'sealed.json')
    root = Path(__file__).resolve().parents[1]
    src = sealed['sources']['e188_apply_dynamic.py']
    bridge = dict(registration=E188_REG, original_source=str(root / E188_SCRIPT), original_sha256=src['before'], original_name=E188_SCRIPT,
                  transformed_sha256=src['after'])
    out = sub(work, stage['out'])
    run_child(work, index, 'e188', [sys.executable, '-u', 'src/e188_apply_dynamic.py', 'apply', '--root', str(work), '--registration-commit', E188_REG,
              '--models', str(root / E188_MODELS), '--pre-e64', str(sub(work, stage['pre'])), '--post-e64', str(sub(work, stage['post'])),
              '--parts', str(work / f'submissions/parts_v{stage["parts"]}.npz'), '--output', str(out)], {'E188_BRIDGE': json.dumps(bridge)}, 'e188_apply')
    r = read(out.with_suffix('.json'))
    if not r['offmask_unchanged'] or r['rows'] != sealed['rows'] or r['ranking_augmentation']:
        raise ValueError('E188 apply receipt')
    return dict(target_rows=r['target_rows'], changed_rows=r['changed_rows'], e188_receipt_sha256=sha(out.with_suffix('.json')))


def protect(final, tab, served_ids):
    """Restore every served E156 ID from the saved E156 output, including unchanged rounded values."""
    served = np.asarray(served_ids, dtype=np.float64)
    if len(served) == 0 or len(np.unique(served)) != len(served):
        raise ValueError('served IDs must be non-empty and unique')
    for name, d in (('final', final), ('tab', tab)):
        if d.columns != [ID, Y] or d[ID].n_unique() != d.height:
            raise ValueError('invalid frame ' + name)
    keys = pl.DataFrame({ID: served})
    values = tab.join(keys, on=ID, how='inner').rename({Y: 'tab'})
    if values.height != len(served) or final.join(keys, on=ID, how='inner').height != len(served):
        raise ValueError('served IDs missing from the final or tab frame')
    out = final.join(values, on=ID, how='left').with_columns(pl.coalesce('tab', Y).cast(final.schema[Y]).alias(Y)).select(final.columns)
    if out.height != final.height or out[Y].null_count():
        raise ValueError('protection changed the population')
    return out


def protect_stage(work, index, stage):
    work = Path(work)
    served = read(work / 'e156/run_0/served.json')
    ids = np.array(served['ids'], dtype=np.float64)
    final, tab = read_sub(sub(work, stage['final'])), read_sub(sub(work, stage['tab']))
    out = protect(final, tab, ids)
    out.write_parquet(sub(work, stage['out']))
    return dict(protected_rows=int(len(ids)))


def check_protected(work, version):
    served = read(Path(work) / 'e156/run_0/served.json')
    keys = pl.DataFrame({ID: np.array(served['ids'], dtype=np.float64)})
    final = read_sub(sub(work, version)).join(keys, on=ID, how='inner').sort(ID)
    tab = pl.DataFrame({ID: np.array(served['ids'], dtype=np.float64), Y: np.array(served['values'])}).sort(ID)
    if final.height != keys.height or not np.array_equal(final[Y].cast(pl.Int64).to_numpy(), tab[Y].cast(pl.Int64).to_numpy()):
        raise ValueError('protected E156 values did not survive the tail')


def scope_check(work, src_v, out_v, airport):
    """Rows changed by an airport-scoped stage must all belong to that airport (no OPDI match => unchanged)."""
    work = Path(work)
    a, b = read_sub(sub(work, src_v)), read_sub(sub(work, out_v))
    j = a.rename({Y: 'old'}).join(b, on=ID, how='inner').filter(pl.col('old') != pl.col(Y))
    rk = pl.read_parquet(work / 'data/prc-2026-datasets/ranking.parquet', columns=[ID, 'PHASE_mvt', 'ADEP_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
    outside = j.join(rk, on=ID, how='left').filter(pl.col('ADEP_mvt') != airport).height
    if outside:
        raise ValueError(f'{outside} rows outside {airport} changed by the airport-scoped stage')
    return int(outside)


def finalize(work, candidate_version):
    work = Path(work)
    sealed = read(work / 'sealed.json')
    template = pl.read_parquet(work / 'data/prc-2026-datasets/submitting.parquet')
    cand = read_sub(sub(work, candidate_version))
    if cand.height != template.height or cand.schema != template.schema or not np.array_equal(np.sort(cand[ID].to_numpy()), np.sort(template[ID].to_numpy())):
        raise ValueError('candidate schema/IDs differ from the template')
    v = cand[Y].cast(pl.Float64).to_numpy()
    if not np.isfinite(v).all() or (v < 0).any() or not np.array_equal(v, np.rint(v)):
        raise ValueError('candidate values must be finite nonnegative integers')
    ordered = template.select(ID).join(cand, on=ID, how='left', maintain_order='left')
    if ordered[Y].null_count():
        raise ValueError('template order join incomplete')
    check_protected(work, candidate_version)
    out = work / sealed['output_name']
    if out.exists():
        raise FileExistsError(out)
    ordered.select(template.columns).with_columns(pl.col(Y).cast(template.schema[Y])).write_parquet(out)
    return out


def assemble(work, repeat_e156=False, refit=()):
    work = Path(work).resolve()
    sealed = read(work / 'sealed.json')
    if (work / 'complete.json').exists() or (work / 'failure.json').exists() or list((work / 'receipts').iterdir()):
        raise FileExistsError('workspace already executed; no retry/resume')
    mode = sealed.get('mode', 'final')
    plan = check_plan(sealed['plan'], mode)
    if plan != stage_plan(mode):
        raise ValueError('sealed plan differs from the adapter plan')
    for s in refit:
        if s not in CACHED:
            raise ValueError('unknown refit stage ' + s)
    started = time.time()
    save(work / 'started.json', dict(protocol=PROTOCOL, unix=started, adapter_sha256=sha(Path(__file__).resolve()), refit=list(refit), repeat_e156=repeat_e156))
    receipts = []
    try:
        for index, stage in enumerate(plan):
            t0 = time.time()
            extra = {}
            if stage['kind'] == 'native' and stage['name'] == 'premix':
                premix(work)
                continue
            if stage['kind'] == 'script':
                script_stage(work, index, stage)
                src_v, out_v = (stage['args'][1], stage['args'][2]) if stage['name'] != 'half' else (int(stage['env']['BLEND_BASE']), stage['args'][1])
                if stage['name'] == 'e152':
                    src_v, out_v = stage['args'][2], stage['args'][4]
            elif stage['kind'] == 'tabpfn':
                extra = tabpfn_stage(work, index, stage, repeat=repeat_e156)
                src_v, out_v = stage['args'][1], stage['args'][2]
            elif stage['kind'] == 'cached':
                extra = cached_stage(work, index, stage, 'refit' if stage['cache'] in refit else 'replay')
                src_v, out_v = stage['args'][1], stage['args'][2]
            elif stage['kind'] == 'e188':
                extra = e188_stage(work, index, stage)
                src_v, out_v = stage['post'], stage['out']
            elif stage['kind'] == 'native' and stage['name'] == 'protect':
                extra = protect_stage(work, index, stage)
                src_v, out_v = stage['final'], stage['out']
            else:
                raise ValueError('unknown stage kind')
            extra['seconds'] = time.time() - t0
            if stage['name'] == 'lszh':
                extra['changed_rows_outside_LSZH'] = scope_check(work, src_v, out_v, 'LSZH')
            receipts.append(change_receipt(work, stage['name'], index, src_v, out_v, extra))
        final_version = plan[-1]['args'][1]
        out = finalize(work, final_version)
        result = dict(protocol=PROTOCOL, output=str(out), output_sha256=sha(out), rows=sealed['rows'], stages=[r['stage'] for r in receipts],
                      receipts={f'receipts/{p.name}': sha(p) for p in sorted((work / 'receipts').iterdir())},
                      served_e156_ids_sha256=read(work / 'e156/run_0/served.json')['ids_sha256'], e81_applied=False, donor_rows_copied=0,
                      fits_performed=sum(r['evidence'].get('fits_performed', 0) for r in receipts), seconds=time.time() - started, uploaded=False,
                      accepted_for_submission=False, unix=time.time())
        save(work / 'complete.json', result)
        print(json.dumps(result), flush=True)
        return out
    except BaseException as exc:
        save(work / 'failure.json', dict(error=repr(exc), no_retry=True, partials_preserved=True, unix=time.time()))
        raise


# ----------------------------------------------------------------------------- contract helpers

def pin(root, names):
    return {n: sha(Path(root) / n) for n in names}


def opdi_inventory(root, extra_dir=None):
    """OPDI event files the LSZH stage reads: `data/opdi/ev_*.parquet` plus, when given, an extended set
    (journal 398: Feb/Jun 2026 LSZH extracts). All are linked into `<work>/data/opdi/` by basename, which the frozen
    `e159_opdi_stage_cv.opdi_wide` globs as `ev_*.parquet`; basenames must therefore be unique and start with `ev_`."""
    root = Path(root).resolve()
    names = sorted(str(p.relative_to(root)) for p in (root / 'data/opdi').glob('ev_*.parquet'))
    if extra_dir:
        extra = sorted(str(p.relative_to(root)) for p in (root / extra_dir).glob('ev_*.parquet'))
        if not extra:
            raise ValueError('extended OPDI set has no ev_*.parquet: ' + str(extra_dir))
        names += extra
    bases = [Path(n).name for n in names]
    if len(set(bases)) != len(bases):
        raise ValueError('OPDI event basenames collide: ' + str(sorted(b for b in set(bases) if bases.count(b) > 1)))
    return names


def archive_inputs(root, patterns):
    setup = read(Path(root) / ARCHIVE / 'setup.json')
    return [n for n in setup['inputs'] if any(re.fullmatch(p, n) for p in patterns)]


def make_contract(root, mode, out, **kw):
    """Write the pinned input contract. janjul = v12075-era artifacts dry run (old ranking); final = fresh members."""
    root = Path(root).resolve()
    files = {}
    names = archive_inputs(root, [r'data/prc-2026-datasets/training_.*', r'data/cv_preds_(e116prod|nomonth|e150c_e64_keep)/fold_.*',
                                  r'data/(adsblol|adsblol_kt1|adsblol_sw_kt1)/[a-z0-9_]+_2025-.*'])
    names += [f'{ARCHIVE}/src/{n}' for n in FROZEN_SOURCES] + [E188_SCRIPT, f'{E188_MODELS}/fit_manifest.json']
    names += [f'{E188_MODELS}/{n}' for n in read(root / E188_MODELS / 'fit_manifest.json')['model_sha256']]
    for stage, (_, calls, _) in CACHED.items():
        names += [f'{STAGE_CACHE}/{stage}/complete.json'] + [f'{STAGE_CACHE}/{stage}/model_{i}.txt' for i in range(calls)]
    opdi = opdi_inventory(root, kw.get('opdi_extra'))   # default: the 44 held files only; --opdi-extra adds a pinned extended set
    names += opdi
    if mode == 'janjul':
        inputs = dict(features_fin1b='data/features_fin1b.parquet', features_e106='data/features_e106.parquet', features_e113='data/features_e113.parquet',
                      ranking='data/prc-2026-datasets/ranking.parquet', template='data/prc-2026-datasets/submitting.parquet')
        members = dict(cat=dict(npz='submissions/cat_fin1b_seeds15.npz'), lgb=dict(npz='submissions/lgb_reg_seeds_v870.npz'),
                       nn=dict(npz='data/e203_rank_20261007/prediction.npz', complete='data/e203_rank_20261007/complete.json'),
                       motion=dict(npz='data/e185_motion_rank.npz'), base=dict(classifier='submissions/clf_v870.txt', state=None),
                       **{k: dict(npz=f'submissions/e130_{k}f_rank.npz') for k, _ in ANCHORS})
        rank = pl.read_parquet(root / inputs['ranking'], columns=[ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
        days = sorted(rank['MVT_TIME_UTC_mvt'].dt.date().cast(pl.String).unique().to_list())
        deriv = dict(root='data', days=days, source='historical v12075-era daily products (old MVT); dry run only')
        output_name, rows = kw.get('output_name', 'gentle-igloo_v12090.parquet'), 344841
    elif mode == 'final':
        fw = kw['feature_work']
        inputs = dict(features_fin1b=f'{fw}/data/features_fin1b.parquet', features_e106=f'{fw}/data/features_e106.parquet',
                      features_e113=f'{fw}/data/features_e113.parquet', ranking=f'{fw}/data/prc-2026-datasets/ranking.parquet',
                      template=f'{fw}/data/prc-2026-datasets/submitting.parquet')
        tab = kw['tabular_root']
        members = {k: dict(npz=f'{tab}/{k}/merged/predictions.npz', complete=f'{tab}/{k}/merged/complete.json') for k in ('cat', 'lgb', 'motion', *[a for a, _ in ANCHORS])}
        members['nn'] = dict(npz=kw['nn_prediction'], complete=kw['nn_complete'])
        members['base'] = dict(classifier=f'{tab}/base/seed_0/classifier.txt', state=f'{tab}/base/seed_0/state.json', complete=f'{tab}/base/seed_0/complete.json')
        names.append(members['base']['complete'])
        dw = kw['derivatives_work']  # scripts/final_adsb_derivatives_20261008.py output root (complete.json + receipts/<day>.json)
        summary = read(root / dw / 'complete.json')
        if summary['status_counts']['input_missing'] or summary['days'] != 120 or summary['final_ranking']['sha256'] != sha(root / inputs['ranking']):
            raise ValueError('daily derivatives incomplete or produced for another ranking file')
        days = []
        for rp in sorted((root / dw / 'receipts').glob('*.json')):
            r = read(rp)
            if r['status'] not in ('processed', 'processed_no_matches') or r['protocol'] != summary['protocol']:
                raise ValueError('daily receipt not accepted: ' + rp.name)
            days.append(r['day'])
            names.append(str(rp.relative_to(root)))
        deriv = dict(root=f'{dw}/data', days=sorted(days), receipt=f'{dw}/complete.json', status_counts=summary['status_counts'])
        names.append(f'{dw}/complete.json')
        output_name, rows = 'gentle-igloo_final.parquet', 670790
    else:
        raise ValueError(mode)
    for folder, product in PRODUCTS:
        names += [f'{deriv["root"]}/{folder}/{product}_{d}.parquet' for d in deriv['days']]
    for spec in members.values():
        names += [v for k, v in spec.items() if k in ('npz', 'complete', 'classifier', 'state') and v]
    names += list(inputs.values()) + [TABPFN_CKPT]
    files = pin(root, sorted(set(names)))
    feat = pl.read_parquet(root / inputs['features_fin1b'], columns=[ID, 'split']).filter(pl.col('split') == 'rank')
    contract = dict(protocol=PROTOCOL, mode=mode, rows=rows, train_rows=2085047, inputs=inputs, members=members,
                    nn_seeds=3 if mode == 'janjul' else 10,
                    rank_ids_sha256=ids_sha(feat[ID].to_numpy()), derivatives=dict(**deriv, ranking_sha256=files[inputs['ranking']]),
                    opdi=dict(files={n: files[n] for n in opdi}, extra_dir=kw.get('opdi_extra'),
                              extra_files=[n for n in opdi if not n.startswith('data/opdi/')]),
                    tabpfn=dict(checkpoint=TABPFN_CKPT, sha256=TABPFN_SHA),
                    files=files, output_name=output_name, created_unix=time.time())
    save(out, contract)
    print('CONTRACT', out, len(files), 'pinned files', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['make-contract', 'prepare', 'assemble', '_stage', '_tabpfn', 'plan'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--contract', type=Path)
    p.add_argument('--work', type=Path)
    p.add_argument('--job', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--mode', choices=['janjul', 'final'])
    p.add_argument('--repeat-e156', action='store_true')
    p.add_argument('--refit', nargs='*', default=[], help='explicit fixed refit of cached stages (e141 e64 lszh)')
    p.add_argument('--feature-work')
    p.add_argument('--tabular-root')
    p.add_argument('--nn-prediction')
    p.add_argument('--nn-complete')
    p.add_argument('--derivatives-work', help='output root of scripts/final_adsb_derivatives_20261008.py')
    p.add_argument('--output-name')
    p.add_argument('--opdi-extra', help='directory (relative to root) with an extended OPDI event set, e.g. data/opdi_final_20261008 (journal 398)')
    a = p.parse_args()
    if a.action == 'plan':
        print(json.dumps(check_plan(stage_plan(a.mode or 'final'), a.mode or 'final'), indent=1))
    elif a.action == 'make-contract':
        make_contract(a.root, a.mode, a.out, **{k: v for k, v in vars(a).items() if v is not None and k in
                      ('feature_work', 'tabular_root', 'nn_prediction', 'nn_complete', 'derivatives_work', 'output_name', 'opdi_extra')})
    elif a.action == 'prepare':
        prepare(a.root, a.contract, a.work)
    elif a.action == 'assemble':
        assemble(a.work, repeat_e156=a.repeat_e156, refit=tuple(a.refit))
    elif a.action == '_stage':
        stage_worker(a.work, a.job)
    elif a.action == '_tabpfn':
        tabpfn_worker(a.work, a.job)


if __name__ == '__main__':
    main()
