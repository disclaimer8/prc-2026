import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

class FixedTransportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('e159_fixed_single_put_upload'),'transport amendment missing')
        import e159_fixed_single_put_upload as T
        self.T=T
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); (self.root/'src').mkdir()
        self.project=Path(__file__).resolve().parents[1]
        for name in [T.B.SELF,T.B.LINEAGE,*T.RUNTIME]: shutil.copy2(self.project/name,self.root/name)
    def test_unused_stage_source_may_differ_without_changing_receipt_inventory(self):
        (self.root/'src/e159_opdi_stage_cv.py').write_text('different server-only build script\n')
        line=self.T.verify_runtime(self.root)
        self.assertEqual(line['sources']['src/e159_opdi_stage_cv.py'],'6919aae6e7b1642d80b1470ec31a09f5a6edf8c922d0cb308af543b54146c348')
    def test_runtime_or_builder_tampering_rejected(self):
        for name in [self.T.B.SELF,*self.T.RUNTIME]:
            with self.subTest(name=name):
                path=self.root/name; original=path.read_bytes(); path.write_bytes(original+b'\n# changed')
                with self.assertRaises(self.T.GuardError): self.T.verify_runtime(self.root)
                path.write_bytes(original)
    def test_lineage_tampering_rejected(self):
        (self.root/self.T.B.LINEAGE).write_text('{}')
        with self.assertRaises(self.T.GuardError):self.T.verify_runtime(self.root)
    def test_receipt_tampering_rejected_before_loading_artifacts(self):
        path=self.root/self.T.B.MANIFEST;path.parent.mkdir();path.write_text('{}')
        with self.assertRaises(self.T.GuardError):self.T.read_receipt(self.root,self.T.verify_runtime(self.root))
    def make_artifacts(self):
        inputs={};outputs={}
        for group,names in [(inputs,self.T.B.VALIDATION_INPUTS),(outputs,self.T.B.OUTPUTS)]:
            for name in names:
                p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
                group[name]=self.T.sha256(p)
        return {'inputs':inputs},{'outputs':outputs}
    def test_input_tampering_rejected(self):
        line,receipt=self.make_artifacts()
        self.T.verify_artifacts(self.root,line,receipt)
        (self.root/self.T.B.PARENT).write_bytes(b'wrong parent')
        with self.assertRaises(self.T.GuardError):self.T.verify_artifacts(self.root,line,receipt)
    def test_either_output_tampering_rejected(self):
        line,receipt=self.make_artifacts()
        for name in self.T.B.OUTPUTS:
            with self.subTest(name=name):
                p=self.root/name;original=p.read_bytes();p.write_bytes(b'wrong output')
                with self.assertRaises(self.T.GuardError):self.T.verify_artifacts(self.root,line,receipt)
                p.write_bytes(original)

if __name__=='__main__':unittest.main()
