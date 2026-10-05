"""Materialize unchanged Cat model inputs on Mac; qualify a low-memory fit.

Only resource layout changes. One fold/seed is compared exactly to Home.
Prepared target and proxy values are frozen; no automatic merge is allowed.
"""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import re
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT'):os.environ[key]='6'
os.environ['POLARS_MAX_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import polars as pl
import cat_fin1b_oof as C
import cat_runtime_parity as R

PROTOCOL='cat-fin1b-laptop-prepared-pools-exact-parity-v1'
PARAMETERS=dict(iterations=5600,learning_rate=.08,depth=8,loss_function='RMSE',thread_count=6,verbose=500,random_seed=0)


def residual(y,base):
    return np.clip(np.clip(y,0,7200)-base,-7200,7200)


def check_columns(cols,cats,expected_cols,expected_cats):
    if cols!=expected_cols or cats!=expected_cats:raise ValueError('model feature/category order differs')


def prepare(inp,bundle,registration):
    home,ids,reference=R.snapshot(inp)
    bundle.mkdir(parents=True,exist_ok=False);(bundle/'sources').mkdir()
    for name in C.SOURCES:shutil.copyfile(inp/'sources'/name,bundle/'sources'/name)
    # All scientific environment values are those of the original trainer.
    clean=C.child_environment(inp/'inputs'/R.NAME/'features.parquet',bundle)
    clean.update(CATBOOST_THREADS='6',OMP_NUM_THREADS='6',OMP_THREAD_LIMIT='6')
    os.environ.clear();os.environ.update(clean)
    if pl.__version__!=home['runtime']['packages']['polars'] or np.__version__!=home['runtime']['packages']['numpy']:
        raise ValueError('preparation NumPy/Polars versions differ')
    import torch  # before LightGBM on Mac
    sys.path.insert(0,str(bundle/'sources'))
    import train as T
    from target_encoding import add_encodings
    if T.REG_CLIP!=(0,7200):raise ValueError('target clipping differs')
    path=inp/'inputs'/R.NAME/'features.parquet'
    df=pl.read_parquet(path);train=df.filter(pl.col('split')=='train');rank=df.filter(pl.col('split')=='rank')
    lab=T.regime_label(train)
    train,rank=add_encodings(train,rank,lab==0,fit_labels=lab)
    cols=T.feature_columns(train);cats=[c for c in T.CATEGORICAL if c in cols]
    check_columns(cols,cats,home['columns']['features'],home['columns']['categoricals'])
    fit=train.filter(pl.Series(lab==0))
    base_f,base_p=T.proxy_base(fit,rank)
    y=residual(fit[T.TARGET].to_numpy(),base_f)
    if not np.array_equal(rank['MVT_ID_mvt'].to_numpy(),ids):raise ValueError('preprocessing changed heldout order')
    if not np.isfinite(y).all() or not np.isfinite(base_p).all():raise ValueError('nonfinite prepared target/proxy')
    for name,frame in [('train',fit),('rank',rank)]:
        out=bundle/f'{name}.parquet';frame.select(cols).write_parquet(out)
        if not frame.select(cols).equals(pl.read_parquet(out)):raise ValueError('prepared feature round trip differs')
    np.savez(bundle/'arrays.npz',train_ids=fit['MVT_ID_mvt'].to_numpy(),target=y,base_p=base_p,mvt_id=ids,reference=reference)
    R.snapshot(inp)
    files=[bundle/'train.parquet',bundle/'rank.parquet',bundle/'arrays.npz',*sorted((bundle/'sources').glob('*.py'))]
    P=dict(protocol=PROTOCOL,registration_commit=registration,home_lineage=R.HOME_LINEAGE,
        input_sha256=C.sha(path),home_runtime=home['runtime'],preparation_runtime=dict(python=sys.version,numpy=np.__version__,polars=pl.__version__),
        train_rows=fit.height,rank_rows=rank.height,columns=cols,categoricals=cats,
        files={str(p.relative_to(bundle)):C.sha(p) for p in files},
        helpers={Path(p).name:C.sha(p) for p in (C.__file__,R.__file__)},worker_sha256=C.sha(__file__),
        parameters=PARAMETERS,
        accepted_for_merge=False)
    C.write_json(bundle/'manifest.json',P)
    print(f'PREPARED exact columns/rows/targets/proxies: train={fit.height}, rank={rank.height}, columns={len(cols)}',flush=True)


def limits():
    cg=next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    p=Path('/sys/fs/cgroup')/cg.lstrip('/')
    quota,period=(p/'cpu.max').read_text().split()
    mem=(p/'memory.max').read_text().strip();swap=(p/'memory.swap.max').read_text().strip()
    if quota=='max' or int(quota)>6*int(period) or mem=='max' or int(mem)>12*1024**3 or swap!='0' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise ValueError('Laptop needs CPU<=600%, RAM<=12GiB, swap0 and nice>=10')
    return dict(cgroup=cg,cpu_max=[int(quota),int(period)],memory_max=int(mem),swap_max=0)


def run(bundle,out,registration):
    m=C.read_json(bundle/'manifest.json')
    if m['protocol']!=PROTOCOL or m['registration_commit']!=registration or m['worker_sha256']!=C.sha(__file__) or m['parameters']!=PARAMETERS:
        raise ValueError('protocol/source/parameter drift')
    for path in (C.__file__,R.__file__):
        if C.sha(path)!=m['helpers'][Path(path).name]:raise ValueError('helper drift')
    def verify():
        for name,h in m['files'].items():
            if C.sha(bundle/name)!=h:raise ValueError(f'prepared input drift {name}')
        C.verify_sources(bundle/'sources')
    verify();runtime=C.packages()
    if runtime['packages']!=m['home_runtime']['packages']:raise ValueError('scientific package versions differ')
    resource_limits=limits();out.mkdir(parents=True,exist_ok=False)
    os.chdir(out)
    C.write_json(out/'invocation.json',dict(runtime=runtime,limits=resource_limits,parameters=m['parameters'],input_manifest_sha256=C.sha(bundle/'manifest.json')))
    from catboost import Pool,CatBoostRegressor
    with np.load(bundle/'arrays.npz',allow_pickle=False) as z:arrays=dict(z)
    if len(arrays['target'])!=m['train_rows'] or len(arrays['mvt_id'])!=344419:raise ValueError('pool row count differs')
    def pool(name,y=None):
        df=pl.read_parquet(bundle/f'{name}.parquet')
        if df.columns!=m['columns']:raise ValueError('pool feature order differs')
        pdf=df.to_pandas();del df
        for c in m['categoricals']:pdf[c]=pdf[c].fillna('NA').astype(str)
        value=Pool(pdf,y,cat_features=m['categoricals']);del pdf;gc.collect()
        if value.get_feature_names()!=m['columns']:raise ValueError('CatBoost feature order differs')
        return value
    start=time.monotonic();train_pool=pool('train',arrays['target'])
    model=CatBoostRegressor(**m['parameters'])
    print(f'TRAINING seed0, {train_pool.num_row()} rows, 5600 iterations, six CPU threads',flush=True)
    model.fit(train_pool);del train_pool;gc.collect()
    reg=model.predict(pool('rank'))+arrays['base_p']
    np.savez(out/'prediction.npz',mvt_id=arrays['mvt_id'],reg=reg)
    model.save_model(str(out/'model.cbm'));verify()
    report=R.compare(arrays['mvt_id'],arrays['reference'],arrays['mvt_id'],reg)
    report.update(protocol=PROTOCOL,registration_commit=registration,seconds=time.monotonic()-start,runtime=runtime,
        prediction_sha256=C.sha(out/'prediction.npz'),model_sha256=C.sha(out/'model.cbm'),accepted_for_merge=False,
        limitation='One-fold resource qualification only; Mac prepared numerical values also require final prediction parity.')
    C.write_json(out/'report.json',report);print(report,flush=True)
    if not report['exact']:raise SystemExit('Qualification failed: predictions differ')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--input',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--registration-commit',required=True);a=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',a.registration_commit):p.error('full registration SHA required')
    if a.action=='prepare':prepare(a.input.resolve(),a.bundle.resolve(),a.registration_commit)
    else:run(a.bundle.resolve(),a.out.resolve(),a.registration_commit)


if __name__=='__main__':main()
