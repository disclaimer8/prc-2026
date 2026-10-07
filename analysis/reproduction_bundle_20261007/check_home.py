"""Read-only bounded file/runtime check; does not build or extract archives."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import platform
import sys
import time

root=Path(__file__).resolve().parents[2]
area=Path(__file__).resolve().parent
manifest=json.loads((area/'dependency_manifest.json').read_text())
result={'root':str(root),'platform':platform.platform(),'python':sys.version,'groups':{}}
started=time.monotonic()
for group in ('prediction_inputs','comparison_only','runner_prerequisites'):
    checked=[];missing=[];changed=[];unsafe=[]
    for row in manifest[group]:
        name=PurePosixPath(row['path']);path=root/row['path']
        if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] not in ('analysis','data','docs','scripts','submissions','tests'):
            unsafe.append(row['path']);continue
        if not path.is_file():missing.append(row);continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):unsafe.append(row['path']);continue
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        record={**row,'actual_bytes':path.stat().st_size,'actual_sha256':actual}
        (checked if actual==row['sha256'] and record['actual_bytes']==row['bytes'] else changed).append(record)
    result['groups'][group]={'matching_count':len(checked),'matching_bytes':sum(r['bytes'] for r in checked),'missing':missing,'changed':changed,'unsafe':unsafe}
result['seconds']=time.monotonic()-started
with (area/'home_check.json').open('x') as output:json.dump(result,output,indent=2)
print(json.dumps({group:{key:value for key,value in report.items() if not isinstance(value,list)}|{'missing':len(report['missing']),'changed':len(report['changed']),'unsafe':len(report['unsafe'])} for group,report in result['groups'].items()}))
