"""Fixed Oct5 LSZH full and 50:50 whole-file submissions; two-phase pinned receipt.

Build and full preflight run where original archives exist. Server transport
preflight verifies the externally pinned receipt, exact registered inventory,
small required source/input files and every output/semantic check again.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import e185_submission as E
import numpy as np
import polars as pl
from submission_guard import GuardError, ID, TARGET, ROWS, exact_ids, exclusive_lock, sha256, verify_hashes

PROTOCOL = 'E159-fixed-LSZH-full-and-half-2026-10-05'
LINEAGE = 'src/e159_fixed_lineage.json'
LINEAGE_SHA = '2c607b8be32293924e044a1f3be978fb31aac5e0f8e3b3ecc99ee81a2971e7a3'
SELF = 'src/e159_fixed_submission.py'
PARENT = 'submissions/gentle-igloo_v12048.parquet'
PARENT_SHA = '0c72511bf5be05ecb9d68a72245e244feeb6088b92372732bb3aae0dc7dddbb1'
VERSIONS = (12060, 12061)
OUTPUTS = [f'submissions/gentle-igloo_v{v}.parquet' for v in VERSIONS]
MANIFEST = 'submissions/e159_fixed_v12060_12061.manifest.json'
BUILD = 'data/e159_fixed_v12060_build'
FORECAST = {'12060': {'delta': -.10, 'interval': [-.30,.15]},
            '12061': {'delta': -.05, 'interval': [-.20,.10]},
            'reference': PARENT, 'reference_score': 232.2411,
            'scope': 'expert forecasts; half is fixed shrinkage, not independently accepted gain'}
VALIDATION_INPUTS = {PARENT, 'submissions/gentle-igloo_v12048.manifest.json', E.TEMPLATE, 'data/prc-2026-datasets/ranking.parquet',
                     'data/features_e106.parquet', 'submissions/gentle-igloo_v291.parquet',
                     'submissions/gentle-igloo_v361.parquet'}
STAGES = [(['e159b_apply.py','12048','12060'],
           {'E159_APS':'LSZH','E159_MONTHS':'all','E159_BASE':'data/cv_preds_e150c_e64_keep','E159_SHIFT':'0'}),
          (['blend_submissions.py','12061'],{'BLEND_BASE':'12048','BLEND_PARTS':'12060:0.5'})]
# LSZH is the reference airport: airport_offsets(['LSZH']) computes x-x=0.
# SHIFT=0 is the same exact shift without reading unrelated ADS-B archives.


def validate_lineage(value):
    try:
        inputs = value['inputs']; sources = value['sources']
        if (value['protocol'] != PROTOCOL or inputs[PARENT] != PARENT_SHA
                or len(inputs) != 69 or len(sources) != 6
                or not VALIDATION_INPUTS <= set(inputs)
                or sum(p.startswith('data/opdi/ev_') for p in inputs) != 44
                or sum(p.startswith('data/prc-2026-datasets/training_') for p in inputs) != 12
                or sum(p.startswith('data/cv_preds_e150c_e64_keep/fold_') for p in inputs) != 6):
            raise GuardError('registered input lineage differs')
        for mapping in (inputs,sources):
            for path,digest in mapping.items():
                if Path(path).is_absolute() or '..' in Path(path).parts or not re.fullmatch('[0-9a-f]{64}',digest):
                    raise GuardError('unsafe lineage path/hash')
    except (KeyError,TypeError,AttributeError) as exc:
        raise GuardError('incomplete lineage') from exc
    return value


def lineage(root):
    verify_hashes(root,{LINEAGE:LINEAGE_SHA})
    return validate_lineage(json.loads((root/LINEAGE).read_text()))


def sources(line, source_sha):
    if not re.fullmatch('[0-9a-f]{64}',source_sha): raise GuardError('registered builder SHA required')
    return {**line['sources'], LINEAGE:LINEAGE_SHA, SELF:source_sha}


def verify_code(root, source_sha):
    if sha256(Path(__file__).resolve()) != source_sha:
        raise GuardError('running builder differs from reviewed source SHA')
    line=lineage(root)
    verify_hashes(root,sources(line,source_sha))
    return line


def verify_registration(root, commit, line, source_sha):
    if not re.fullmatch('[0-9a-f]{40}',commit): raise GuardError('full registration commit required')
    for name,digest in sources(line,source_sha).items():
        result=subprocess.run(['git','show',f'{commit}:{name}'],cwd=root,capture_output=True,check=True)
        if hashlib.sha256(result.stdout).hexdigest()!=digest:
            raise GuardError(f'code not present unchanged in registration commit: {name}')
    journal=subprocess.run(['git','show',f'{commit}:HYPOTHESES_LOG.md'],cwd=root,capture_output=True,check=True).stdout.decode()
    if not any(PROTOCOL in row and '12060' in row and '12061' in row for row in journal.splitlines()):
        raise GuardError('both fixed candidates require one journal registration before fit')


def verify_inventory(root,line):
    for pattern in ('data/opdi/ev_*.parquet','data/prc-2026-datasets/training_*.parquet',
                    'data/cv_preds_e150c_e64_keep/fold_*.npz'):
        actual={str(p.relative_to(root)) for p in root.glob(pattern)}
        expected={p for p in line['inputs'] if Path(p).match(pattern)}
        if actual!=expected: raise GuardError(f'archive inventory changed: {pattern}')
    verify_hashes(root,line['inputs'])


def context_for(root, ids):
    rank=pl.read_parquet(root/'data/prc-2026-datasets/ranking.parquet',columns=[ID,'ADEP_mvt','PHASE_mvt']).filter(pl.col('PHASE_mvt')=='DEP')
    exact_ids(rank[ID].to_numpy(),ids,ROWS)
    feats=pl.read_parquet(root/'data/features_e106.parquet',columns=[ID,'split','d_mvt_lobt']).filter(pl.col('split')=='rank')
    exact_ids(feats[ID].to_numpy(),ids,ROWS)
    return rank.select(ID,'ADEP_mvt').join(feats.select(ID,'d_mvt_lobt'),on=ID,how='inner')


def validate_pair(full,half,parent,template,tab_base,tab_donor,context,rows=ROWS):
    exact_ids(context[ID].to_numpy(),template[ID].to_numpy(),rows)
    context=context.sort(ID)
    base=parent.sort(ID)[TARGET].to_numpy().astype(float)
    full_values=full.sort(ID)[TARGET].to_numpy().astype(float)
    metrics={}
    for version,frame in zip(VERSIONS,(full,half)):
        result=E.validate_submission(frame,template,parent,tab_base,tab_donor,expected_rows=rows)
        values=frame.sort(ID)[TARGET].to_numpy().astype(float)
        changed=values!=base
        if np.any(changed & (context['ADEP_mvt'].to_numpy()!='LSZH')):
            raise GuardError('changes outside LSZH are forbidden')
        d=context['d_mvt_lobt'].to_numpy()
        known=np.isfinite(d) & changed
        if (np.any(values[changed]<0) or np.any(values[known]<np.maximum(0,d[known]-3606)-.500001)
                or np.any(values[known]>d[known]+3606+.500001)):
            raise GuardError('changed prediction violates LOBT window or nonnegative floor')
        result['rms_vs_v12048']=result.pop('rms_vs_v1139')
        metrics[str(version)]=result
    if not np.array_equal(half.sort(ID)[TARGET].to_numpy(),np.rint(.5*base+.5*full_values)):
        raise GuardError('half file differs from exact fixed whole-file rounded mean')
    if np.array_equal(full_values,half.sort(ID)[TARGET].to_numpy()):
        raise GuardError('two candidates are identical')
    return metrics


def validate_outputs(root):
    read=lambda p:pl.read_parquet(root/p)
    template=read(E.TEMPLATE)
    return validate_pair(read(OUTPUTS[0]),read(OUTPUTS[1]),read(PARENT),template,
                         read('submissions/gentle-igloo_v291.parquet'),read('submissions/gentle-igloo_v361.parquet'),
                         context_for(root,template[ID].to_numpy()))


def reserve(root,commit):
    for p in (root/'submissions').iterdir():
        if any(int(n) in VERSIONS for n in re.findall(r'_v(\d+)(?:\D|$)',p.name)):
            raise GuardError(f'version already occupied: {p.name}')
    E.write_exclusive_json(root/'submissions/e159_fixed_v12060_12061.reservation.json',
                          {'protocol':PROTOCOL,'versions':list(VERSIONS),'registration_commit':commit,'forecast':FORECAST})


def stage_environment():
    env={k:os.environ[k] for k in ('PATH','HOME','LANG','LC_ALL','TMPDIR') if k in os.environ}
    env.update(OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4',POLARS_MAX_THREADS='2',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
               POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR='load_as_storage')
    return env


def build(root,commit,source_sha):
    line=verify_code(root,source_sha)
    verify_registration(root,commit,line,source_sha)
    with exclusive_lock(root/'data/e159_fixed_build.lock'):
        verify_inventory(root,line)
        reserve(root,commit)
        build_root=root/BUILD
        snapshot=sources(line,source_sha)
        E.prepare_workspace(root,build_root,list(line['inputs']),[],list(snapshot))
        verify_hashes(build_root,snapshot)
        verify_hashes(build_root,line['inputs'])
        for n,(args,extra) in enumerate(STAGES):
            with (build_root/f'logs/{n}_{args[0]}.log').open('x') as log:
                subprocess.run(['nice','-n','10',sys.executable,'src/'+args[0],*args[1:]],cwd=build_root,
                               env={**stage_environment(),**extra},stdout=log,stderr=subprocess.STDOUT,check=True)
        checks=validate_outputs(build_root)
        verify_inventory(root,line)
        verify_hashes(root,snapshot); verify_hashes(build_root,snapshot)
        actual={str(p.relative_to(build_root)) for p in (build_root/'submissions').iterdir() if not p.is_symlink()}
        if actual!=set(OUTPUTS): raise GuardError('unexpected scratch outputs')
        receipt={'protocol':PROTOCOL,'versions':list(VERSIONS),'registration_commit':commit,'forecast':FORECAST,
                 'created_utc':E.utcnow().isoformat(),'inputs':line['inputs'],'sources':snapshot,
                 'outputs':{p:sha256(build_root/p) for p in OUTPUTS},'validation':checks,
                 'stages':STAGES,'environment':stage_environment(),'build_directory':BUILD,
                 'full_input_verification':True}
        E.write_exclusive_json(build_root/'receipt.json',receipt)
        for p in OUTPUTS: os.link(build_root/p,root/p)
        os.link(build_root/'receipt.json',root/MANIFEST)
        print(f'BUILT both fixed candidates; receipt SHA {sha256(root/MANIFEST)}',flush=True)
    return receipt


def preflight(root,commit,source_sha,receipt_sha,full_inputs=False):
    line=verify_code(root,source_sha)
    if not re.fullmatch('[0-9a-f]{40}',commit): raise GuardError('registration commit required')
    verify_hashes(root,{MANIFEST:receipt_sha})
    receipt=json.loads((root/MANIFEST).read_text())
    fixed={'protocol':PROTOCOL,'versions':list(VERSIONS),'registration_commit':commit,'forecast':FORECAST,
           'inputs':line['inputs'],'sources':sources(line,source_sha),'stages':json.loads(json.dumps(STAGES)),
           'build_directory':BUILD,'full_input_verification':True}
    if any(receipt.get(k)!=v for k,v in fixed.items()) or set(receipt.get('outputs',{}))!=set(OUTPUTS):
        raise GuardError('receipt differs from fixed registration')
    if full_inputs:
        verify_registration(root,commit,line,source_sha)
        verify_inventory(root,line)
    else:
        verify_hashes(root,{p:line['inputs'][p] for p in VALIDATION_INPUTS})
    verify_hashes(root,receipt['outputs'])
    validation=validate_outputs(root)
    if validation!=receipt.get('validation'): raise GuardError('semantic checks changed since build')
    return receipt


def upload(root,commit,source_sha,receipt_sha,version):
    from e185_single_put_upload import OnePutHTTP
    from minio import Minio
    if version not in VERSIONS: raise GuardError('unregistered submission version')
    E.require_window(E.utcnow())
    receipt=preflight(root,commit,source_sha,receipt_sha)
    name=f'submissions/gentle-igloo_v{version}.parquet'; path=root/name
    if not 0<path.stat().st_size<=5*1024*1024: raise GuardError('one PUT size limit exceeded')
    if root.resolve()!=Path('/root/prc-2026'): raise GuardError('upload must use canonical Hetzner root/shared ledger')
    E.load_upload_credentials(root)
    http=OnePutHTTP()
    client=Minio('s3.opensky-network.org',access_key=os.environ['PRC_ACCESS_KEY'],secret_key=os.environ['PRC_SECRET_KEY'],secure=True,http_client=http)
    def inventory(): return E.fetch_leaderboard(),E.bucket_inventory(root)
    def timeout(_s,_f): raise TimeoutError('bounded single PUT expired')
    def put():
        E.require_window(E.utcnow())
        if (datetime(2026,10,6,tzinfo=timezone.utc)-E.utcnow()).total_seconds()<120: raise GuardError('too close to day end')
        verify_hashes(root,{name:receipt['outputs'][name],MANIFEST:receipt_sha})
        previous=signal.signal(signal.SIGALRM,timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo',path.name,str(path),num_parallel_uploads=1)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM,previous)
    try:
        used=E.single_attempt(root/'data/submission_upload_ledger.jsonl',path.name,receipt['outputs'][name],
                              now=E.utcnow,inventory=inventory,upload=put)
    finally: http.clear()
    print(f'UPLOADED v{version}; accounted {used}/5 UTC-day slots',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','preflight','upload'])
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--registration-commit',required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--receipt-sha256')
    p.add_argument('--full-inputs',action='store_true')
    p.add_argument('--version',type=int,choices=VERSIONS)
    a=p.parse_args(); root=a.root.resolve()
    try:
        if a.action=='build': build(root,a.registration_commit,a.source_sha256)
        else:
            if not a.receipt_sha256: raise GuardError('externally pinned receipt SHA required')
            if a.action=='preflight':
                preflight(root,a.registration_commit,a.source_sha256,a.receipt_sha256,a.full_inputs)
                print('PREFLIGHT PASS full inputs' if a.full_inputs else 'PREFLIGHT PASS pinned receipt and semantic outputs')
            else: upload(root,a.registration_commit,a.source_sha256,a.receipt_sha256,a.version)
    except Exception as exc:
        print(f'STOP: {exc}' if isinstance(exc,GuardError) else f'STOP: {type(exc).__name__}; inspect local logs',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__': raise SystemExit(main())
