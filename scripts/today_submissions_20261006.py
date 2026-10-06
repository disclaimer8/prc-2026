"""Five fixed full files authorized on Oct 6, without waiting for full-chain CV.

Private assembly only; active sources/queues are never changed. Shared single-PUT
ledger is used on Hetzner with an explicit Oct 6 window. No score-dependent choice.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import e185_submission as E
import submission_guard as G

PROTOCOL='five-fixed-E190-E188-user-waived-full-CV-20261006-v1'
DAY='2026-10-06'
VERSIONS=(12070,12071,12072,12073,12074)
MODEL_REG='f40a465633cc32bff959c598d27d59f9bdc11cc3'
E188_REG='d45cb738fb7671cf289ae8a400a30b5b46dfbb3d'
BEST='submissions/gentle-igloo_v12061.parquet'
BEST_SHA='8d3b589213240bd6960984d6b2a46c6716617262c635ac2dc02d35b50b876015'
WORK='data/today_chain_20261006'
MEMBER='data/e190_rank_20261006/complete/prediction.npz'
RECIPES={12070:'E190 full production chain, original MLP weight .25, half LSZH',
         12071:'rounded whole-file .5*v12061 + .5*v12070',
         12072:'E188 natural LIRF NM stage through full tail, other rows v12061',
         12073:'rounded whole-file .5*v12061 + .5*v12072',
         12074:'E190 plus E188 at its own pre-E64 base, original full tail and half LSZH'}
FORECASTS={12070:[0.,-1.,1.],12071:[0.,-.5,.5],12072:[0.,-.5,.5],
           12073:[0.,-.25,.25],12074:[0.,-1.5,1.5]}
ID,Y=G.ID,G.TARGET

def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)

def now():return datetime.now(timezone.utc)

def check_registration(root,commit):
    if not re.fullmatch('[a-f0-9]{40}',commit):raise ValueError('registration SHA required')
    source=Path(__file__).resolve();rel=source.relative_to(root)
    old=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=root)
    if hashlib.sha256(old).hexdigest()!=G.sha256(source):raise ValueError('assembly code differs from registration')
    journal=subprocess.check_output(['git','show',f'{commit}:HYPOTHESES_LOG.md'],cwd=root).decode()
    if PROTOCOL not in journal:raise ValueError('five fixed candidates were not registered')

def env():
    x=E.clean_environment()
    x.update(OMP_THREAD_LIMIT='6',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
             POLARS_MAX_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    return x

def sub(work,v):return work/f'submissions/gentle-igloo_v{v}.parquet'

def stage(work,args,extra=None):
    index=len(list((work/'logs').glob('*.log')))
    print('STAGE',index,args,flush=True)
    with (work/f'logs/{index:02d}_{Path(args[0]).stem}.log').open('x') as f:
        subprocess.run([sys.executable,'-u','src/'+args[0],*map(str,args[1:])],cwd=work,
                       env={**env(),**(extra or {})},stdout=f,stderr=subprocess.STDOUT,check=True)

def prepare(root,commit):
    check_registration(root,commit);G.verify_hashes(root,{BEST:BEST_SHA})
    work=root/WORK
    parent=json.loads((root/'submissions/gentle-igloo_v12048.manifest.json').read_text())
    lszh=json.loads((root/'src/e159_fixed_lineage.json').read_text())
    inputs=set(parent['inputs'])|set(parent['outputs'])|set(lszh['inputs'])|{BEST,MEMBER,
        'submissions/gentle-igloo_v12060.parquet','submissions/gentle-igloo_v12048.manifest.json'}
    # A missing neural member can be linked now and must be complete before build.
    for name in inputs-{MEMBER}:
        if not (root/name).is_file():raise ValueError('missing '+name)
    frozen=root/'data/today_chain_source_20261006'
    G.verify_hashes(frozen,parent['scratch_sources'])
    E.prepare_workspace(root,work,sorted(inputs),[],[])
    for name in parent['scratch_sources']:
        dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(frozen/name,dest)
    # Historical import executes six unused CV fits. Keep only its original
    # constants/imports in this PRIVATE copy; the actual ranking fit is unchanged.
    p=work/'src/e64_visit_stage.py';s=p.read_text()
    if '\ncols = ' not in s:raise ValueError('unknown E64 source layout')
    p.write_text(s.split('\ncols = ',1)[0]+'\n')
    (work/'submissions/cat_reg_rank.npz').symlink_to((root/'submissions/cat_fin1b_seeds15.npz').resolve())
    save(work/'setup.json',dict(protocol=PROTOCOL,registration=commit,created=now().isoformat(),
         source_before=parent['scratch_sources'],source_after={n:G.sha256(work/n) for n in parent['scratch_sources']},
         inputs={n:G.sha256(root/n) for n in inputs if (root/n).is_file()},full_chain_cv_waived_by_user=True))
    print('PRIVATE WORKSPACE READY',flush=True)

def tails(work,pre,staged,parts,first):
    stage(work,['e152_stage_gate.py',pre,staged,parts,first],{'E152_GATE':'0.95'})
    stage(work,['blend_submissions.py',first+1],{'BLEND_BASE':str(first),'BLEND_PARTS':'44:0.1214,60:0.0270'})
    stage(work,['lobt_edge_lift.py',first+1,first+2],{'LOBT_CLIP':'1'})
    stage(work,['protect_tabpfn_rows.py',first+2,pre,first+3])
    stage(work,['e159b_apply.py',first+3,first+4],{'E159_APS':'LSZH','E159_MONTHS':'all',
          'E159_BASE':'data/cv_preds_e150c_e64_keep','E159_SHIFT':'0'})
    stage(work,['blend_submissions.py',first+5],{'BLEND_BASE':str(first+3),'BLEND_PARTS':f'{first+4}:0.5'})
    return first+5

def validate(root,path):
    read=lambda p:pl.read_parquet(root/p)
    candidate=pl.read_parquet(path)
    result=G.validate_submission(candidate,read(E.TEMPLATE),read(BEST),
        read('submissions/gentle-igloo_v291.parquet'),read('submissions/gentle-igloo_v361.parquet'))
    if candidate[Y].min()<0:raise ValueError('negative prediction')
    return result

def publish(root,path,version,commit,evidence):
    if version not in VERSIONS:raise ValueError('unregistered version')
    final=root/f'submissions/gentle-igloo_v{version}.parquet'
    if final.exists():raise ValueError('refusing final overwrite')
    checks=validate(root,path)
    os.link(path,final)
    receipt=dict(protocol=PROTOCOL,version=version,registration=commit,created=now().isoformat(),
        recipe=RECIPES[version],forecast=FORECASTS[version],reference=BEST,reference_sha256=BEST_SHA,
        full_chain_cv_waived_by_user=True,quality_validated_on_full_chain=False,
        sha256=G.sha256(final),validation=checks,evidence=evidence)
    save(final.with_suffix('.manifest.json'),receipt)
    print('READY',version,checks,'SHA',receipt['sha256'],flush=True)

def build_e190(root,commit):
    check_registration(root,commit);work=root/WORK
    setup=json.loads((work/'setup.json').read_text())
    if setup['registration']!=commit:raise ValueError('workspace registration differs')
    G.verify_hashes(work,setup['source_after'])
    sys.path.insert(0,str(root/'scripts'));import e190_rank_20261006 as N
    neural=N.verify_output(root/'data/e190_rank_inputs_20261006',root/'data/e190_rank_20261006/complete',MODEL_REG)
    members=','.join(f'submissions/e130_{m}f_rank.npz:{w}' for m,w in
                    [('raw','0.0333333333333333'),('lobt','.10'),('sched','.10'),('eobt','.10'),('aobt','.10')])
    baseenv={'FEATURES':'data/features_fin1b.parquet','LGB_SEEDS':'submissions/lgb_reg_seeds_v870.npz'}
    # Rebuild only the cached pre-stage control to verify classifier/member routing.
    stage(work,['blend.py',870,900000,.65],{**baseenv,'E130_MEMBERS':members+',submissions/e130_mlpa_rank.npz:0.25'})
    a=pl.read_parquet(sub(work,900000)).sort(ID);b=pl.read_parquet(root/'submissions/gentle-igloo_v12020.parquet').sort(ID)
    if not a.equals(b):raise ValueError('unchanged pre-stage control failed exact reproduction')
    stage(work,['blend.py',870,900100,.65],{**baseenv,'E130_MEMBERS':members+','+MEMBER+':0.25'})
    stage(work,['e185_apply.py',900100,900101],{'E185_W':str(.35*2/3),'E185_RANK':E.RANK})
    stage(work,['e167_nm_window_rule.py',900101,900102])
    E.tabpfn_rows(work,900102,900103)
    common={'STAGE_SEEDS':'42,1,2,3,4','ADSB_DIR':'data/adsblol_kt1','PARTS':'submissions/parts_v900102.npz'}
    stage(work,['e141_apply.py',900103,900104],{**common,'STAGE_KEEP_ALL':'1','DROP_FEATS':E.D7,
          'WALK_DIR':'data/adsblol_sw_kt1','E36_BASE_DIR':'data/cv_preds_e116prod'})
    stage(work,['e64_apply.py',900104,900105],{**common,'E64_KEEP_ALL':'1','DROP_FEATS':E.D3})
    final=tails(work,900103,900105,900102,900110)
    evidence={'member_sha256':G.sha256(root/MEMBER),'member_manifest':neural,'setup_sha256':G.sha256(work/'setup.json')}
    publish(root,sub(work,final),12070,commit,evidence)
    # Fixed entire-file shrinkage, defined before any leaderboard result.
    stage(work,['blend_submissions.py',900116],{'BLEND_BASE':'12061','BLEND_PARTS':f'{final}:0.5'})
    publish(root,sub(work,900116),12071,commit,evidence)

def build_joint(root,commit):
    check_registration(root,commit);work=root/WORK
    output=sub(work,900200)
    subprocess.run([sys.executable,str(root/'scripts/e188_rank_20261006.py'),'apply',
       '--registration-commit',E188_REG,'--models',str(root/'data/e188_rank_20261006_v2'),
       '--pre-e64',str(sub(work,900104)),'--post-e64',str(sub(work,900105)),
       '--parts',str(work/'submissions/parts_v900102.npz'),'--output',str(output)],cwd=root,env=env(),check=True)
    final=tails(work,900103,900200,900102,900210)
    publish(root,sub(work,final),12074,commit,{'stage_sha256':G.sha256(output),'e188_registration':E188_REG,
         'member_sha256':G.sha256(root/MEMBER),'setup_sha256':G.sha256(work/'setup.json')})

def upload(root,version,receipt_sha):
    if root.resolve()!=Path('/root/prc-2026'):raise ValueError('canonical server ledger required')
    if version not in VERSIONS:raise ValueError('unexpected candidate')
    if now().date().isoformat()!=DAY:raise ValueError('Oct 6 window only')
    path=root/f'submissions/gentle-igloo_v{version}.parquet';rp=path.with_suffix('.manifest.json')
    if G.sha256(rp)!=receipt_sha:raise ValueError('pinned receipt differs')
    r=json.loads(rp.read_text())
    if (r['protocol']!=PROTOCOL or r['version']!=version or r['recipe']!=RECIPES[version]
        or r['forecast']!=FORECASTS[version] or r['reference_sha256']!=BEST_SHA
        or G.sha256(path)!=r['sha256']):raise ValueError('candidate receipt drift')
    G.verify_hashes(root,{BEST:BEST_SHA});checks=validate(root,path)
    if checks!=r['validation'] or not 0<path.stat().st_size<=5*1024**2:raise ValueError('output checks/size differ')
    from minio import Minio
    from e185_single_put_upload import OnePutHTTP
    E.load_upload_credentials(root);http=OnePutHTTP()
    client=Minio('s3.opensky-network.org',access_key=os.environ['PRC_ACCESS_KEY'],
        secret_key=os.environ['PRC_SECRET_KEY'],secure=True,http_client=http)
    # Existing reusable quota/ledger algorithm; only this new process's day changes.
    G.DAY=DAY
    def inventory():return G.fetch_leaderboard(),E.bucket_inventory(root)
    def timeout(_a,_b):raise TimeoutError('single PUT timeout')
    def put():
        if (datetime(2026,10,7,tzinfo=timezone.utc)-now()).total_seconds()<120:raise ValueError('too near UTC cutoff')
        G.verify_hashes(root,{str(path.relative_to(root)):r['sha256'],str(rp.relative_to(root)):receipt_sha})
        old=signal.signal(signal.SIGALRM,timeout)
        try:
            signal.alarm(90);client.fput_object('prc-2026-gentle-igloo',path.name,str(path),num_parallel_uploads=1)
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
    try:
        used=G.single_attempt(root/'data/submission_upload_ledger.jsonl',path.name,r['sha256'],now=now,inventory=inventory,upload=put)
    finally:http.clear()
    print('UPLOADED',version,'slots',used,'of 5',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','e190','joint','upload'])
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--registration-commit')
    p.add_argument('--version',type=int);p.add_argument('--receipt-sha256');a=p.parse_args();root=a.root.resolve()
    if a.action=='upload':upload(root,a.version,a.receipt_sha256)
    else:{'prepare':prepare,'e190':build_e190,'joint':build_joint}[a.action](root,a.registration_commit)

if __name__=='__main__':main()
