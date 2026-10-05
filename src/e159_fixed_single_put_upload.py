"""Registered transport-only amendment for the unchanged LSZH pair.

The trusted Mac build receipt proves original model inputs and build sources.
This uploader verifies that exact receipt, its complete old source map, all
locally-used runtime code, both output hashes and all submission semantics.
Unused OPDI training source on the upload host is deliberately not imported or
executed. No build recipe, archive, prediction or existing module is changed.
"""
import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
import signal
import sys

import e159_fixed_submission as B
from submission_guard import GuardError, sha256, verify_hashes

PROTOCOL = 'E159-fixed-LSZH-transport-amendment-2026-10-05'
SELF = 'src/e159_fixed_single_put_upload.py'
BUILDER_SHA = 'fd3fbfe244b159281b1dbbbd0d2784be3cbc3d7e45b9543b3ee33027791a622a'
BUILD_REGISTRATION = '6ac022966e22baeaec2a4f921c7a7f6e8223475a'
RECEIPT_SHA = '2958dae1ed7e8cab6ae5aa31463463a129443bf9b808ec4d6b81148a91e326f8'
RUNTIME = ('src/e185_submission.py', 'src/submission_guard.py', 'src/e185_single_put_upload.py')


def verify_runtime(root):
    # Do not call B.verify_code/preflight: they audit build-time-only scripts
    # that need not be installed identically on the transport host.
    if sha256(Path(B.__file__).resolve()) != BUILDER_SHA:
        raise GuardError('running immutable builder module changed')
    verify_hashes(root,{B.SELF:BUILDER_SHA,B.LINEAGE:B.LINEAGE_SHA})
    line=B.lineage(root)
    verify_hashes(root,{name:line['sources'][name] for name in RUNTIME})
    for name,module in [('src/e185_submission.py',B.E),('src/submission_guard.py',sys.modules['submission_guard'])]:
        if sha256(Path(module.__file__).resolve()) != line['sources'][name]:
            raise GuardError('imported runtime module differs from pinned source')
    return line


def read_receipt(root,line):
    verify_hashes(root,{B.MANIFEST:RECEIPT_SHA})
    receipt=json.loads((root/B.MANIFEST).read_text())
    fixed={'protocol':B.PROTOCOL,'versions':list(B.VERSIONS),'registration_commit':BUILD_REGISTRATION,
           'forecast':B.FORECAST,'inputs':line['inputs'],'sources':B.sources(line,BUILDER_SHA),
           'stages':json.loads(json.dumps(B.STAGES)),'build_directory':B.BUILD,'full_input_verification':True}
    if any(receipt.get(key)!=value for key,value in fixed.items()) or set(receipt.get('outputs',{}))!=set(B.OUTPUTS):
        raise GuardError('trusted receipt differs from unchanged build registration')
    return receipt


def verify_artifacts(root,line,receipt):
    verify_hashes(root,{name:line['inputs'][name] for name in B.VALIDATION_INPUTS})
    verify_hashes(root,receipt['outputs'])


def preflight(root,source_sha,transport_registration,receipt_sha=RECEIPT_SHA):
    if (not re.fullmatch('[0-9a-f]{40}',transport_registration)
            or not re.fullmatch('[0-9a-f]{64}',source_sha)
            or receipt_sha!=RECEIPT_SHA
            or sha256(Path(__file__).resolve())!=source_sha):
        raise GuardError('transport source/registration or trusted receipt SHA differs')
    verify_hashes(root,{SELF:source_sha})
    line=verify_runtime(root)
    receipt=read_receipt(root,line)
    verify_artifacts(root,line,receipt)
    if B.validate_outputs(root)!=receipt.get('validation'):
        raise GuardError('submission semantics differ from verified Mac build')
    return receipt


def upload(root,source_sha,transport_registration,version,receipt_sha=RECEIPT_SHA):
    if root.resolve()!=Path('/root/prc-2026'):
        raise GuardError('use canonical Hetzner root and shared submission ledger')
    if version not in B.VERSIONS: raise GuardError('unregistered LSZH version')
    B.E.require_window(B.E.utcnow())
    receipt=preflight(root,source_sha,transport_registration,receipt_sha)
    from e185_single_put_upload import OnePutHTTP
    from minio import Minio
    if sha256(Path(inspect.getfile(OnePutHTTP)).resolve())!=receipt['sources']['src/e185_single_put_upload.py']:
        raise GuardError('imported OnePutHTTP source differs')
    name=f'submissions/gentle-igloo_v{version}.parquet';path=root/name
    if not 0<path.stat().st_size<=5*1024*1024: raise GuardError('candidate exceeds single PUT limit')
    B.E.load_upload_credentials(root)
    http=OnePutHTTP()
    client=Minio('s3.opensky-network.org',access_key=os.environ['PRC_ACCESS_KEY'],
                 secret_key=os.environ['PRC_SECRET_KEY'],secure=True,http_client=http)
    def inventory(): return B.E.fetch_leaderboard(),B.E.bucket_inventory(root)
    def timeout(_signum,_frame): raise TimeoutError('bounded single PUT expired')
    def put():
        B.E.require_window(B.E.utcnow())
        if (datetime(2026,10,6,tzinfo=timezone.utc)-B.E.utcnow()).total_seconds()<120:
            raise GuardError('too close to UTC day end')
        verify_hashes(root,{**receipt['outputs'],B.MANIFEST:RECEIPT_SHA,SELF:source_sha})
        previous=signal.signal(signal.SIGALRM,timeout)
        try:
            signal.alarm(90)
            client.fput_object('prc-2026-gentle-igloo',path.name,str(path),num_parallel_uploads=1)
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    print(f'Transport {PROTOCOL}; registration {transport_registration}; source {source_sha}',flush=True)
    try:
        used=B.E.single_attempt(root/'data/submission_upload_ledger.jsonl',path.name,receipt['outputs'][name],
                                now=B.E.utcnow,inventory=inventory,upload=put)
    finally: http.clear()
    print(f'UPLOADED v{version}; accounted {used}/5 UTC-day slots',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','upload'])
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--transport-registration',required=True)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--version',type=int,choices=B.VERSIONS)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='preflight':
            preflight(root,args.source_sha256,args.transport_registration,args.receipt_sha256)
            print('PREFLIGHT PASS unchanged pair, trusted receipt and pinned runtime')
        else: upload(root,args.source_sha256,args.transport_registration,args.version,args.receipt_sha256)
    except Exception as exc:
        print(f'STOP: {exc}' if isinstance(exc,GuardError) else f'STOP: {type(exc).__name__}',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
