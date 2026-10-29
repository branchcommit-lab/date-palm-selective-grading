from pathlib import Path
import unittest,subprocess,sys,tempfile,json,shutil,hashlib,ast
ROOT=Path(__file__).resolve().parent.parent

class ReleaseTests(unittest.TestCase):
    def call(self,*args,root=ROOT):
        return subprocess.run([sys.executable,str(root/'run.py'),*args],cwd=tempfile.gettempdir(),text=True,capture_output=True)
    def test_fresh_c3_evidence(self):
        sys.path.insert(0,str(ROOT/'src/methods'))
        from c3_fresh_audit import audit
        r=audit(ROOT)
        self.assertEqual(r['models'],33)
        self.assertEqual(r['evaluation_rows'],12954)
    def test_c3_missing_inputs_fail_closed(self):
        p=subprocess.run([sys.executable,str(ROOT/'src/methods/c3_controller.py')],text=True,capture_output=True)
        self.assertEqual(p.returncode,3,p.stderr)
        self.assertFalse(json.loads(p.stdout)['inference_executed'])
    def test_safe_default_inference(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.call('inference','--output',d)
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads((Path(d)/'INFERENCE_EVIDENCE_STATUS.json').read_text())
            self.assertFalse(report['new_model_inference_executed'])
            self.assertEqual(report['supplementary']['metric_checks'],168)
            self.assertEqual(report['certificate_checks']['C3_inventory_entries'],33)
    def test_missing_execute_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.call('inference','--action','execute','--output',d)
            self.assertEqual(p.returncode,3,p.stderr)
            report=json.loads((Path(d)/'INFERENCE_EXECUTION_STATUS.json').read_text())
            self.assertEqual(report['status'],'REQUIRES_INPUTS')
            self.assertTrue(report['missing_inputs'])
            self.assertFalse(report['new_model_inference_executed'])
    def test_retrain_does_not_train(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.call('retrain','--output',d)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads((Path(d)/'RETRAIN_STATUS.json').read_text())['training_executed'])
    def test_rejects_corrupt_probability_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'release';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('outputs','results','validation','__pycache__'))
            p=root/'data/tables/candidate_probabilities_calibration.csv';p.write_bytes(p.read_bytes()+b'\n')
            q=self.call('inference','--output',str(Path(d)/'output'),root=root)
            self.assertEqual(q.returncode,2)
    def test_rejects_missing_C3_inventory_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'release';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('outputs','results','validation','__pycache__'))
            p=root/'data/verification/C3_CHECKPOINT_INVENTORY.json';o=json.loads(p.read_text());o['checkpoints'].pop();p.write_text(json.dumps(o))
            q=self.call('inference','--output',str(Path(d)/'output'),root=root)
            self.assertEqual(q.returncode,2)
    def test_python_and_original_notebook_cells_parse(self):
        import nbformat
        for p in (ROOT/'src').rglob('*.py'):ast.parse(p.read_text())
        for p in (ROOT/'src/methods').glob('*.ipynb'):
            n=nbformat.read(p,as_version=4);nbformat.validate(n)
            for c in n.cells:
                if c.cell_type=='code':ast.parse(c.source)
    def test_immutable_file_manifest(self):
        p=subprocess.run([sys.executable,str(ROOT/'src/methods/verify_package.py'),'--root',str(ROOT)],text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
