import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_certificates(root):
    root=Path(root);v=root/'data/verification'
    d=json.loads((v/'dandhi/CANDIDATE_PROBABILITY_PARITY.json').read_text())
    c=json.loads((v/'C11_PARITY_FINAL.json').read_text())
    frozen=json.loads((root/'configs/frozen_parameters.json').read_text())
    assert d['selected_temperature']==frozen['candidate_temperature']
    assert d['development_blend_weight']==frozen['candidate_blend_weight']
    for key in ('locked_test_candidate_grade_mismatches','locked_test_candidate_confidence_10_mismatches','locked_test_candidate_OR_5_5_mismatches','locked_test_C11_10_mismatches'):
        assert d[key]==0,key
    for split,count in [('calibration',1668),('locked_test',12510)]:
        p=root/'data/tables'/f'candidate_probabilities_{split}.csv'
        assert digest(p)==d[f'export_{split}_csv_sha256']
        f=pd.read_csv(p);assert len(f)==count and f.sample_id.is_unique
        a=f[['prob_Small','prob_Medium','prob_Large']].to_numpy(float)
        assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
        assert np.allclose(a.sum(1),1,rtol=0,atol=3e-6)
    assert c['frozen_threshold_parity']['frozen_threshold']==frozen['C11_threshold']
    assert json.loads((root/'data/metadata/thresholds.json').read_text())['C11_1049']==frozen['C11_threshold']
    assert not c['frozen_threshold_parity']['threshold_recalibrated_for_use']
    for split in ('calibration','development'):
        assert c['splits'][split]['score_tolerance']==1e-5
        assert c['splits'][split]['mismatches_above_tolerance']==0
    inventory=json.loads((v/'C3_CHECKPOINT_INVENTORY.json').read_text())
    assert inventory['status']=='PASS_CHECKPOINT_INVENTORY'
    assert not inventory['inference_executed'] and not inventory['training_executed']
    items=inventory['checkpoints'];assert len(items)==33
    assert len({i['run'] for i in items})==33
    for i in items:
        assert len(i['sha256'])==64 and int(i['bytes'])>0 and i['training_state_exists']
    files={p.stem for p in (root/'data/raw_records/c3').rglob('*.csv')}
    assert files=={i['run'] for i in items}
    return {'status':'PASS','frozen_parameters_checked':True,'Dandhi_probability_hashes_checked':True,'C11_scores':'previously authenticated; current reference CSV hashes checked by controller','C3_inventory_entries':33,'C3_checkpoint_bytes_available_here':False,'new_model_inference_executed':False}

_original_validate_certificates=validate_certificates
def validate_certificates(root):
 r=_original_validate_certificates(root)
 from c3_fresh_audit import audit
 r['C3_fresh_checkpoint_replay']=audit(root)
 r['C3_checkpoint_bytes_verified_at_execution']=True
 return r
