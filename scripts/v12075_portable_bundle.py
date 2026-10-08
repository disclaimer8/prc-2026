"""Private allowlisted v12075 package; no downloads, training or publication.

Only seal is metadata-only. Other actions require a reviewed registration and
the exact manifest SHA. The qualified external Mac runtime is not packaged.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import types

PROTOCOL='v12075-private-allowlist-package-minimal-git-proof-zero-fit-v1'
AREA='analysis/reproduction_bundle_20261008'
SELF='scripts/v12075_portable_bundle.py'
TEST='tests/test_v12075_portable_bundle.py'
DOC=AREA+'/PROTOCOL.md'
SPEC=AREA+'/manifest.json'
ORIGINAL='analysis/reproduction_bundle_20261007/dependency_manifest.json'
CLOSURE='analysis/reproduction_bundle_20261007/closure.json'
REPLAY_REG='70aff7199731c381945c45754ee70c7e3a964aa5'
BEST_SHA='90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2'
HEADER='BUNDLE-MANIFEST.json'
CHUNK=8*1024**2
FLOOR=4*1024**3


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):h.update(block)
    return h.hexdigest()


def relative(name):
    p=PurePosixPath(name)
    if not name or any(ord(c)<32 or ord(c)==127 for c in name) or p.is_absolute() or '\\' in name or p.as_posix()!=name or any(x in ('','..','.') for x in name.split('/')) or p.parts[0]=='.git':
        raise ValueError('unsafe member path: '+name)
    return Path(*p.parts)


def safe_file(root,name):
    root=Path(root).resolve();p=root/relative(name)
    for part in (p,*p.parents):
        if part==root:break
        if part.is_symlink():raise ValueError('linked input: '+name)
    if not p.is_file() or not stat.S_ISREG(p.stat().st_mode) or p.resolve()!=p:raise ValueError('not a contained regular file: '+name)
    return p


def verify_file(root,record):
    p=safe_file(root,record['path'])
    if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('size/hash mismatch: '+record['path'])
    return p


def members(spec):
    result={}
    for r in [*spec['files'],*spec['objects']]:
        relative(r['path'])
        if r['path']==HEADER or r['path'] in result or not re.fullmatch('[0-9a-f]{64}',r['sha256']) or not isinstance(r['bytes'],int) or r['bytes']<0:
            raise ValueError('duplicate/invalid specification member')
        result[r['path']]=r
    if sum(r['bytes'] for r in result.values())!=spec['total_bytes']:raise ValueError('specification byte total differs')
    return result


def git(root,args,raw=None):
    # No user/system Git configuration, hooks, remotes or credentials copied.
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_SYSTEM=os.devnull)
    return subprocess.check_output(['git','-C',str(root),*args],input=raw,env=env,stderr=subprocess.PIPE)


def git_object(root,kind,oid):
    if kind not in ('commit','tree','blob') or not re.fullmatch('[0-9a-f]{40}',oid):raise ValueError('invalid Git proof object')
    return git(root,['cat-file',kind,oid])


def object_closure(root,proofs):
    found={}
    def add(kind,oid):
        raw=git_object(root,kind,oid)
        found[oid]=dict(oid=oid,kind=kind,path='__bundle__/objects/'+oid,bytes=len(raw),sha256=digest(raw))
        return raw
    for proof in proofs:
        parts=relative(proof['path']).parts;commit=add('commit',proof['commit'])
        tree=commit.splitlines()[0].decode().removeprefix('tree ')
        for i,part in enumerate(parts):
            add('tree',tree)
            entries=git(root,['ls-tree','-z',tree,'--',part]).split(b'\0')
            entries=[e for e in entries if e]
            if len(entries)!=1:raise ValueError('missing/ambiguous historical path')
            details,name=entries[0].split(b'\t',1);mode,kind,oid=details.decode().split()
            if name.decode()!=part:raise ValueError('historical path differs')
            if i<len(parts)-1:
                if kind!='tree':raise ValueError('historical path not directory')
                tree=oid
            else:
                if kind!='blob' or mode not in ('100644','100755'):raise ValueError('historical source is not regular')
                raw=add('blob',oid)
                if digest(raw)!=proof['blob_sha256']:raise ValueError('historical proof SHA differs')
                if proof.get('git_blob_oid',oid)!=oid:raise ValueError('historical blob OID differs')
    return sorted(found.values(),key=lambda r:r['oid'])


def restore_git(root,spec):
    root=Path(root)
    if (root/'.git').exists():raise ValueError('fresh Git proof repository required')
    git(root,['init','-q'])
    for r in spec['objects']:
        raw=verify_file(root,r).read_bytes()
        actual=git(root,['hash-object','-w','-t',r['kind'],'--stdin'],raw).decode().strip()
        if actual!=r['oid']:raise ValueError('restored Git object OID differs')
    for proof in spec['proofs']:
        if digest(git(root,['show',proof['commit']+':'+proof['path']]))!=proof['blob_sha256']:raise ValueError('restored git show differs')


def seal(root):
    import reproduce_v12075_v2 as V
    V.registration(root,REPLAY_REG)
    old=read(root/ORIGINAL);closure=read(root/CLOSURE);files=[];replay=read(root/V.SPEC)
    expected=dict(prediction_inputs=replay['files'],runner_prerequisites={name:sha(root/name) for name in V.REGISTERED_FILES},
        comparison_only={**replay['comparison_only'],V.BEST:V.BEST_SHA})
    if old['inventory_sha256']!=sha(root/V.SPEC):raise ValueError('original inventory provenance differs')
    for group,entries in expected.items():
        if len(old[group])!=len(entries) or {r['path']:r['sha256'] for r in old[group]}!=entries:raise ValueError('legacy dependency group differs: '+group)
    wanted={(REPLAY_REG,p) for p in (*V.REGISTERED_FILES,'HYPOTHESES_LOG.md')}|{(V.R.E188_REG,'scripts/e188_rank_20261006.py')}
    if len(closure['required_git_objects'])!=len(wanted) or {(p['commit'],p['path']) for p in closure['required_git_objects']}!=wanted:
        raise ValueError('unapproved Git proof path')
    for group in ('prediction_inputs','comparison_only','runner_prerequisites'):
        for item in old[group]:
            p=safe_file(root,item['path'])
            if p.stat().st_size!=item['bytes']:raise ValueError('inventory size changed')
            files.append({**item,'group':group})
    for name in (SELF,TEST,DOC,'LICENSE','pyproject.toml','uv.lock'):
        p=safe_file(root,name);files.append(dict(path=name,bytes=p.stat().st_size,sha256=sha(p),group='package_sources'))
    objects=object_closure(root,closure['required_git_objects'])
    spec=dict(protocol=PROTOCOL,original_inventory_sha256=old['inventory_sha256'],files=files,objects=objects,
        proofs=closure['required_git_objects'],total_bytes=sum(r['bytes'] for r in files+objects),
        best_sha256=BEST_SHA,replay_registration=REPLAY_REG,fits_authorized=0,
        runtime=dict(python='3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]',numpy='2.5.3',polars='1.44.2',lightgbm='4.7.0',platform='qualified macOS arm64; external existing interpreter'),
        scope='fresh materialized source/data checkout; partial exact Git object proof, not full history or clean-machine installation')
    members(spec);save(root/SPEC,spec)
    print(json.dumps(dict(sealed=True,files=len(files),git_objects=len(objects),bytes=spec['total_bytes'],manifest_sha256=sha(root/SPEC),metadata_only=True)))


def registration(root,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('full packaging registration required')
    for name in (SELF,TEST,DOC,SPEC):
        if digest(git(root,['show',commit+':'+name]))!=sha(root/name):raise ValueError('registered package source/spec differs: '+name)
    if PROTOCOL.encode() not in git(root,['show',commit+':HYPOTHESES_LOG.md']):raise ValueError('package protocol not registered')


def copy_checked(source,destination,record):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src,destination.open('xb') as out:
        h=hashlib.sha256();size=0
        for block in iter(lambda:src.read(CHUNK),b''):out.write(block);h.update(block);size+=len(block)
    if size!=record['bytes'] or h.hexdigest()!=record['sha256']:raise ValueError('source changed while copying: '+record['path'])


def bootstrap(root,spec,out,commit):
    registration(root,commit);out.mkdir(exist_ok=False)
    save(out/HEADER,spec)
    for r in spec['files']:
        if r['group']=='package_sources':copy_checked(verify_file(root,r),out/r['path'],r)
    for r in spec['objects']:
        raw=git_object(root,r['kind'],r['oid'])
        if len(raw)!=r['bytes'] or digest(raw)!=r['sha256']:raise ValueError('Git export drift')
        p=out/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
    save(out/'BOOTSTRAP.json',dict(protocol=PROTOCOL,registration=commit,manifest_sha256=digest(canonical(spec))))


def stage(root,spec,out,commit):
    receipt=read(out/'BOOTSTRAP.json')
    if receipt!=dict(protocol=PROTOCOL,registration=commit,manifest_sha256=digest(canonical(spec))):raise ValueError('bootstrap lineage differs')
    missing=[]
    for r in spec['files']:
        if (out/r['path']).exists():verify_file(out,r)
        elif (root/r['path']).exists():copy_checked(verify_file(root,r),out/r['path'],r)
        else:missing.append(r['path'])
    for r in spec['objects']:verify_file(out,r)
    save(out/'MISSING.json',dict(paths=missing,bytes=sum(r['bytes'] for r in spec['files'] if r['path'] in set(missing))))
    with (out/'MISSING.txt').open('x') as f:f.write(''.join(p+'\n' for p in missing))
    print(json.dumps(dict(missing=len(missing),staged=len(spec['files'])-len(missing))))


def archive(root,spec,out):
    inventory=members(spec)
    for record in inventory.values():verify_file(root,record)
    with Path(out).open('xb') as stream,tarfile.open(fileobj=stream,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        raw=canonical(spec);header=tarfile.TarInfo(HEADER);header.size=len(raw);header.mode=0o644;tar.addfile(header,io.BytesIO(raw))
        for name,r in sorted(inventory.items()):
            info=tarfile.TarInfo(name);info.size=r['bytes'];info.mode=0o644
            with safe_file(root,name).open('rb') as source:tar.addfile(info,source)
    with Path(out).open('rb') as stream:proof=check_stream(stream,spec)
    return {**proof,'archive_bytes':Path(out).stat().st_size,'archive_sha256':sha(out)}


def check_stream(stream,spec,destination=None,disk_floor=FLOOR):
    inventory=members(spec);seen=set();destination=Path(destination) if destination is not None else None
    if destination is not None:
        if destination.exists():raise ValueError('fresh materialized checkout required')
        if shutil.disk_usage(destination.parent).free<spec['total_bytes']+disk_floor:raise ValueError('insufficient extraction disk reserve')
        destination.mkdir()
    with tarfile.open(fileobj=stream,mode='r|*',ignore_zeros=True) as tar:
        for number,entry in enumerate(tar):
            relative(entry.name)
            if not entry.isfile() or entry.sparse is not None or entry.name in seen:raise ValueError('nonregular/sparse/duplicate archive member')
            seen.add(entry.name)
            if number==0:
                if entry.name!=HEADER or entry.size!=len(canonical(spec)):raise ValueError('manifest must be first and exact')
                raw=tar.extractfile(entry).read()
                if raw!=canonical(spec):raise ValueError('untrusted archive specification')
                if destination is not None:
                    with (destination/HEADER).open('xb') as f:f.write(raw)
                continue
            if entry.name not in inventory:raise ValueError('undeclared archive member')
            r=inventory[entry.name]
            if entry.size!=r['bytes']:raise ValueError('archive size differs')
            out=None
            if destination is not None:
                if shutil.disk_usage(destination).free<entry.size+disk_floor:raise ValueError('4GiB extraction reserve violated')
                path=destination/relative(entry.name);path.parent.mkdir(parents=True,exist_ok=True);out=path.open('xb')
            h=hashlib.sha256();size=0
            try:
                source=tar.extractfile(entry)
                for block in iter(lambda:source.read(CHUNK),b''):
                    h.update(block);size+=len(block)
                    if out is not None:out.write(block)
            finally:
                if out is not None:out.close()
            if size!=r['bytes'] or h.hexdigest()!=r['sha256']:raise ValueError('archive member hash differs')
    if seen!={HEADER,*inventory}:raise ValueError('archive member inventory incomplete')
    return dict(passed=True,payload_files=len(inventory),payload_bytes=spec['total_bytes'],manifest_sha256=digest(canonical(spec)))


def runtime_check(root,spec):
    import platform
    import numpy
    import polars
    import lightgbm
    actual=dict(python=sys.version,numpy=numpy.__version__,polars=polars.__version__,lightgbm=lightgbm.__version__)
    if sys.platform!='darwin' or platform.machine()!='arm64' or actual!={k:spec['runtime'][k] for k in actual}:
        raise ValueError('existing qualified external Mac runtime required')
    for path in ('/usr/bin/caffeinate','/usr/bin/memory_pressure','/usr/bin/vm_stat'):
        if not Path(path).is_file():raise ValueError('Mac supervisor tool missing')
    if shutil.disk_usage(root).free<5*1024**3:raise ValueError('need5GiB before replay, including4GiB reserve')
    import reproduce_v12075_v2 as V
    V.registration(root,REPLAY_REG)
    return dict(passed=True,runtime=actual,scope='qualified external interpreter; not clean-machine installation')


def verify_materialized(root,spec,commit):
    materialized=read(root/'MATERIALIZED.json')
    if (not materialized['passed'] or materialized['manifest_sha256']!=digest(canonical(spec)) or materialized['registration']!=commit
        or materialized['fits']!=0 or materialized['scope']!=spec['scope']):raise ValueError('materialization proof differs')
    for r in members(spec).values():verify_file(root,r)


def stricter_limits(resources,elapsed,disk_free):
    import reproduce_v12075_v2_launch_20261007 as Q
    Q.enforce_limits(resources,elapsed,disk_free)
    if disk_free<FLOOR:raise RuntimeError('portable replay disk below4GiB; stop owned process group without retry')


def replay(root,spec,commit):
    verify_materialized(root,spec,commit);runtime_check(root,spec)
    import reproduce_v12075_v2_launch_20261007 as Q
    # Reuse the exact reviewed supervisor, tightening only its disk floor.
    # Neither the original module globals nor its frozen file are modified.
    supervised=types.FunctionType(Q.supervise.__code__,{**Q.supervise.__globals__,'enforce_limits':stricter_limits},'supervise')
    supervised(root,REPLAY_REG)
    save(root/'PORTABLE_REPLAY_VERIFIED.json',verify_replay(root,spec,commit))


def verify_replay(root,spec,commit):
    verify_materialized(root,spec,commit)
    folder=root/'data/reproduced_v12075_v2';proof=read(folder/'complete.json')
    supervisor=read(root/'data/reproduce_v12075_v2_launch_20261007/complete.json')
    if (proof['registration']!=REPLAY_REG or proof['fits_performed']!=0 or proof['rows']!=344841 or proof['protected_rows']!=42
        or not proof['exact_predictions'] or not proof['byte_identical'] or proof['output_sha256']!=BEST_SHA
        or supervisor['replay_receipt_sha256']!=sha(folder/'complete.json') or supervisor['fits_performed']!=0
        or sha(folder/'reproduced_v12075.parquet')!=BEST_SHA or sha(root/'submissions/gentle-igloo_v12075.parquet')!=BEST_SHA):
        raise ValueError('fresh checkout exact zero-fit replay proof differs')
    return dict(passed=True,output_sha256=BEST_SHA,rows=344841,protected_rows=42,fits=0,uploaded=False,
        materialized_receipt_sha256=sha(root/'MATERIALIZED.json'),replay_receipt_sha256=sha(folder/'complete.json'),supervisor_receipt_sha256=sha(root/'data/reproduce_v12075_v2_launch_20261007/complete.json'))


def home_guard():
    if sys.platform!='linux':raise ValueError('bulk stage/build is reserved for bounded Home Linux unit')
    relative=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    folder=Path('/sys/fs/cgroup')/relative.lstrip('/')
    mem,swap=[(folder/n).read_text().strip() for n in ('memory.max','memory.swap.max')]
    quota,period=(folder/'cpu.max').read_text().split()
    if mem=='max' or int(mem)>1024**3 or swap!='0' or quota=='max' or int(quota)>int(period):
        raise ValueError('Home bulk stage/build requires<=1CPU/1GiB/swap0')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['seal','bootstrap','stage','build','check','materialize','runtime-check','replay','verify-replay'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--manifest',type=Path)
    p.add_argument('--manifest-sha256');p.add_argument('--registration');p.add_argument('--out',type=Path);p.add_argument('--archive')
    a=p.parse_args();root=a.root.resolve()
    if a.action=='seal':seal(root);return
    manifest=(a.manifest or root/SPEC).resolve()
    if not re.fullmatch('[0-9a-f]{40}',a.registration or '') or not re.fullmatch('[0-9a-f]{64}',a.manifest_sha256 or '') or sha(manifest)!=a.manifest_sha256:
        raise ValueError('reviewed registration and trusted manifest SHA required')
    spec=read(manifest)
    if spec['protocol']!=PROTOCOL or spec['fits_authorized']!=0 or spec['best_sha256']!=BEST_SHA:raise ValueError('package specification differs')
    members(spec)
    if a.action=='bootstrap':bootstrap(root,spec,a.out,a.registration)
    elif a.action=='stage':home_guard();stage(root,spec,a.out,a.registration)
    elif a.action=='build':
        home_guard()
        if read(root/'BOOTSTRAP.json')['registration']!=a.registration:raise ValueError('build registration differs')
        if shutil.disk_usage(a.out.parent).free<spec['total_bytes']+FLOOR:raise ValueError('insufficient archive disk reserve')
        save(Path(str(a.out)+'.verified.json'),archive(root,spec,a.out))
    elif a.action in ('check','materialize'):
        if a.action=='materialize':registration(root,a.registration)
        stream=sys.stdin.buffer if a.archive=='-' else Path(a.archive).open('rb')
        try:result=check_stream(stream,spec,a.out if a.action=='materialize' else None)
        finally:
            if stream is not sys.stdin.buffer:stream.close()
        if a.action=='materialize':
            restore_git(a.out,spec);save(a.out/'MATERIALIZED.json',{**result,'registration':a.registration,'fits':0,'scope':spec['scope']})
        print(json.dumps(result))
    elif a.action=='runtime-check':print(json.dumps(runtime_check(root,spec)))
    elif a.action=='replay':replay(root,spec,a.registration)
    else:save(root/'PORTABLE_REPLAY_VERIFIED.json',verify_replay(root,spec,a.registration))


if __name__=='__main__':main()
