"""Verify every file pinned by a final-tail contract under a root (stdlib only).
usage: python verify_pins.py CONTRACT ROOT OUT_JSON"""
import hashlib, json, os, sys, time
contract, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(contract))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()
res = {'ok': [], 'missing': [], 'mismatch': {}}
t0 = time.time(); nbytes = 0
for name, h in sorted(c['files'].items()):
    p = os.path.join(root, name)
    if not os.path.isfile(p):
        res['missing'].append(name); continue
    nbytes += os.path.getsize(p)
    got = sha(p)
    if got == h: res['ok'].append(name)
    else: res['mismatch'][name] = {'expected': h, 'got': got}
summary = dict(contract=os.path.abspath(contract), contract_sha256=sha(contract), root=os.path.abspath(root), pinned=len(c['files']),
               ok=len(res['ok']), missing=len(res['missing']), mismatch=len(res['mismatch']), bytes_hashed=nbytes,
               seconds=round(time.time() - t0, 1), unix=time.time())
json.dump(dict(summary=summary, missing=res['missing'], mismatch=res['mismatch']), open(out, 'w'), indent=1, sort_keys=True)
print(json.dumps(summary))
