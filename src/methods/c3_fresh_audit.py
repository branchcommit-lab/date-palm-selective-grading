from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
def audit(root):
 root=Path(root);base=root/'data/verification/c3_fresh';report=json.loads((base/'C3_FRESH_INFERENCE_REPORT.json').read_text())
 assert report['status']=='PASS_ALL_33_AUTHENTIC_CHECKPOINT_TO_PREDICTION_REPLAYS'
 assert len(report['runs'])==33 and report['evaluation_rows']==12954
 assert report['inference_executed'] and not report['training_executed'] and not report['thresholds_refitted']
 inv={r['run']:r for r in json.loads((root/'data/verification/C3_CHECKPOINT_INVENTORY.json').read_text())['checkpoints']}
 for r in report['runs']:
  assert inv[r['run']]['sha256']==r['checkpoint_sha256']
  assert r['calibration_bitwise_equal_original_cache'] and r['evaluation_bitwise_equal_original_cache'] and r['pass']
  assert all(r[k]==0 for k in ['probabilities_outside_tolerance','grade_mismatches','referral_mismatches'])
  old=next((root/'data/raw_records/c3').rglob(r['run']+'.csv'));assert hashlib.sha256(old.read_bytes()).hexdigest()==r['source_predictions_sha256']
  old=pd.read_csv(old,dtype={'source_id':str});new=pd.read_csv(base/r['run']/'FRESH_HELDOUT_PREDICTIONS.csv',dtype={'source_id':str})
  assert old.source_id.tolist()==new.source_id.tolist() and len(new)==r['evaluation_rows']
  pc=['prob_'+c for c in r['classes']];assert np.allclose(old[pc],new[pc],atol=1e-5,rtol=0)
  assert old.predicted_label.tolist()==new.predicted_label.tolist()
  assert old.referred.astype(str).str.lower().tolist()==new.referred.astype(str).str.lower().tolist()
  cp=np.load(base/r['run']/'FRESH_CALIBRATION.npz',allow_pickle=False)['probabilities'];ep=np.load(base/r['run']/'FRESH_EVALUATION.npz',allow_pickle=False)['probabilities']
  assert len(cp)==r['calibration_rows']
  assert hashlib.sha256(cp.tobytes()).hexdigest()==r['calibration_probability_bytes_sha256']==r['original_calibration_probability_bytes_sha256']
  assert hashlib.sha256(ep.tobytes()).hexdigest()==r['evaluation_probability_bytes_sha256']==r['original_evaluation_probability_bytes_sha256']
 return {'status':'PASS_RECORDED_ALL_33_FRESH_CHECKPOINT_REPLAYS','models':33,'evaluation_rows':12954,'calibration_rows':report['calibration_rows'],'grade_mismatches':0,'referral_mismatches':0,'current_audit_launches_models':False}
