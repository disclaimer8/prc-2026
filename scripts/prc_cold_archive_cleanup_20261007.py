"""Remove only the user-authorized old Mac tables after verified remote archival."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / 'analysis/disk_cleanup_20261007'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def main():
    receipt = AREA / 'removed.json'
    if receipt.exists():
        raise RuntimeError('Cleanup already recorded; do not rerun')
    manifest_path = AREA / 'archive_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    remote = json.loads((AREA / 'archive_verified.json').read_text())
    items = manifest['items']
    assert remote['verified'] and remote['manifest_sha256'] == sha(manifest_path)
    assert remote['verified_paths'] == [r['path'] for r in items]
    assert len(items) == remote['files'] == 21
    protected = set()
    for name in ('data/today_chain_20261006/setup.json',
                 'analysis/e203_input_receipt_20261007.json',
                 'docs/reproduction_manifest.json',
                 'analysis/reproduce_v12075_20261007/inventory.json',
                 'analysis/night_cb_20261007/input_receipt.json'):
        protected.update(strings(json.loads((ROOT / name).read_text())))
    # Complete preflight before removing any file. Current-source arrays stay local.
    for row in items:
        path = ROOT / row['path']
        assert path.parent == ROOT / 'data' and path.name.startswith('features')
        assert path.suffix == '.parquet' and not path.is_symlink()
        assert row['path'] not in protected
        stat = path.stat()
        assert (stat.st_ino, stat.st_mtime_ns, stat.st_size) == (row['inode'], row['mtime_ns'], row['bytes'])
        assert sha(path) == row['sha256']
    before = shutil.disk_usage(ROOT).free
    removed = []
    for row in items:
        path = ROOT / row['path']
        assert path.stat().st_ino == row['inode'] and path.stat().st_mtime_ns == row['mtime_ns']
        path.unlink()
        removed.append(row['path'])
        with (AREA / 'removal_events.jsonl').open('a') as out:
            out.write(json.dumps({'path': row['path'], 'unix': time.time()}) + '\n')
    result = dict(removed=removed, count=len(removed), logical_bytes=sum(r['bytes'] for r in items),
                  archive_host=manifest['archive_host'], archive_root=manifest['archive_root'],
                  remote_verified=True, free_before=before, free_after=shutil.disk_usage(ROOT).free,
                  unix=time.time(), current_recipe_inputs_preserved=True)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
