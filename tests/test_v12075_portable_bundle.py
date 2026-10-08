import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def module():return importlib.import_module('v12075_portable_bundle')


class PortableBundleTests(unittest.TestCase):
    def test_replay_guard_only_tightens_disk_floor_and_keeps_frozen_guard(self):
        p=module();import reproduce_v12075_v2_launch_20261007 as q
        original=q.enforce_limits
        p.stricter_limits({'rss_bytes':0},0,4*1024**3)
        with self.assertRaises(RuntimeError):p.stricter_limits({'rss_bytes':0},0,4*1024**3-1)
        with self.assertRaises(RuntimeError):p.stricter_limits({'rss_bytes':17*1024**3},0,10*1024**3)
        with self.assertRaises(RuntimeError):p.stricter_limits({'rss_bytes':0},4*3600+1,10*1024**3)
        self.assertIs(q.enforce_limits,original)
    def test_paths_reject_escape_and_git_overwrite(self):
        p=module()
        for name in ('/tmp/x','../x','a/../b','a//b','./x','a\\b','.git/config','a\nb',''):
            with self.subTest(name=name),self.assertRaises(ValueError):p.relative(name)
        self.assertEqual(p.relative('data/model.txt'),Path('data/model.txt'))

    @staticmethod
    def fixture():
        data=b'saved model\n';record=dict(path='data/model.txt',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),group='prediction_inputs')
        return dict(protocol='synthetic',files=[record],objects=[],proofs=[],total_bytes=len(data)),data

    def test_stream_archive_exact_and_extract_has_only_declared_bytes(self):
        p=module();spec,data=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stage=root/'stage';(stage/'data').mkdir(parents=True);(stage/'data/model.txt').write_bytes(data)
            archive=root/'a.tar';p.archive(stage,spec,archive)
            with archive.open('rb') as f:self.assertEqual(p.check_stream(f,spec)['payload_files'],1)
            dest=root/'fresh'
            with archive.open('rb') as f:p.check_stream(f,spec,dest,disk_floor=0)
            self.assertEqual((dest/'data/model.txt').read_bytes(),data)
            self.assertEqual(sorted(str(x.relative_to(dest)) for x in dest.rglob('*') if x.is_file()),['BUNDLE-MANIFEST.json','data/model.txt'])

    def test_stream_rejects_unsafe_extra_duplicate_link_size_hash_and_missing(self):
        p=module();spec,data=self.fixture()
        for kind in ('traversal','extra','duplicate','symlink','hardlink','device','wrong_size','wrong_hash','missing'):
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w') as t:
                raw=p.canonical(spec);entry=tarfile.TarInfo('BUNDLE-MANIFEST.json');entry.size=len(raw);t.addfile(entry,io.BytesIO(raw))
                if kind=='missing':pass
                else:
                    e=tarfile.TarInfo('../escape' if kind=='traversal' else 'extra' if kind=='extra' else 'data/model.txt')
                    payload=data
                    if kind in ('symlink','hardlink'):e.type=tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE;e.linkname='/tmp/elsewhere';payload=b''
                    elif kind=='device':e.type=tarfile.CHRTYPE;payload=b''
                    elif kind=='wrong_size':payload=data+b'x'
                    elif kind=='wrong_hash':payload=b'X'+data[1:]
                    e.size=len(payload);t.addfile(e,io.BytesIO(payload))
                    if kind=='duplicate':t.addfile(e,io.BytesIO(payload))
            out.seek(0)
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.check_stream(out,spec)

    def test_minimal_git_proof_restores_exact_git_show_without_unrelated_blob(self):
        p=module()
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);repo=base/'origin';repo.mkdir()
            subprocess.run(['git','init','-q',str(repo)],check=True)
            (repo/'keep').mkdir();(repo/'keep/source.py').write_text('print(42)\n');(repo/'unrelated.txt').write_text('not exported\n')
            subprocess.run(['git','-C',str(repo),'add','keep/source.py','unrelated.txt'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
            parent=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            (repo/'unrelated.txt').write_text('new unrelated content\n')
            subprocess.run(['git','-C',str(repo),'add','unrelated.txt'],check=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','child'],check=True)
            commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            blob=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD:unrelated.txt'],text=True).strip()
            proof=dict(commit=commit,path='keep/source.py',blob_sha256=hashlib.sha256((repo/'keep/source.py').read_bytes()).hexdigest())
            objects=p.object_closure(repo,[proof]);self.assertNotIn(blob,{x['oid'] for x in objects})
            self.assertNotIn(parent,{x['oid'] for x in objects})
            fresh=base/'fresh';fresh.mkdir()
            for item in objects:
                dest=fresh/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.git_object(repo,item['kind'],item['oid']))
            p.restore_git(fresh,dict(objects=objects,proofs=[proof]))
            self.assertEqual(subprocess.check_output(['git','-C',str(fresh),'show',commit+':keep/source.py']),b'print(42)\n')
            self.assertNotEqual(subprocess.run(['git','-C',str(fresh),'cat-file','-e',blob],capture_output=True).returncode,0)
            self.assertNotEqual(subprocess.run(['git','-C',str(fresh),'cat-file','-e',parent],capture_output=True).returncode,0)
            self.assertFalse((fresh/'.git/config').read_text().find('remote "')>=0)

    def test_input_links_and_mutated_files_are_rejected(self):
        p=module();spec,data=self.fixture();r=spec['files'][0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data').mkdir();(root/'data/model.txt').write_bytes(data)
            p.verify_file(root,r)
            (root/'data/model.txt').write_bytes(data+b'x')
            with self.assertRaises(ValueError):p.verify_file(root,r)
            (root/'data/model.txt').unlink();(root/'outside').write_bytes(data);(root/'data/model.txt').symlink_to(root/'outside')
            with self.assertRaises(ValueError):p.verify_file(root,r)

    def test_concatenated_archive_does_not_hide_extra_members(self):
        p=module();spec,data=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data').mkdir();(root/'data/model.txt').write_bytes(data)
            out=root/'a.tar';p.archive(root,spec,out)
            extra=io.BytesIO()
            with tarfile.open(fileobj=extra,mode='w') as t:
                entry=tarfile.TarInfo('undeclared');entry.size=1;t.addfile(entry,io.BytesIO(b'x'))
            with out.open('ab') as f:f.write(extra.getvalue())
            with out.open('rb') as f,self.assertRaises(ValueError):p.check_stream(f,spec)


if __name__=='__main__':unittest.main()
