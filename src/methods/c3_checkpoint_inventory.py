#!/usr/bin/env python3
"""Read-only inventory for 33 model checkpoints and their individual evaluation predictions.

This never loads/checkpoints with pickle or Torch, infers images, or trains models.
It reports the files needed for a full optional raw-to-C3 inference replay.
"""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd
FAMILIES={
 'four_view':('foreground_background_probes_v1','C3_FOUR_VIEW_ALL_12_RUNS.csv'),
 'size_holdout':('class_novelty_probes_v1','C3_CLASS_HOLDOUT_ALL_9_RUNS.csv'),
 'cultivar_holdout':('independent_training_probes_resumable_v3','C3_CULTIVAR_HOLDOUT_ALL_12_RUNS.csv'),
}
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def execute(study,experiment_root,out):
    study=Path(study); root=Path(experiment_root);out=Path(out)
    runs=[]
    for family,(folder,table) in FAMILIES.items():
        records=pd.read_csv(study/'data/tables'/table)
        for r in records.itertuples(index=False):
            choices=[root/folder/r.run,root/'additional_evidence_20261024'/folder/r.run]
            match=next((x for x in choices if x.is_dir()),None)
            expected=study/'data/raw_records/c3'/family/(r.run+'.csv')
            row={'family':family,'run':r.run,'expected_observations':int(r.n),
                 'found_run_folder':bool(match),'checkpoint_present':False,
                 'result_present':False,'predictions_equal_packaged_records':False}
            if match:
                row['directory']=str(match)
                model=match/'BEST_MODEL.pt';result=match/'RESULT.json'; preds=match/'HELDOUT_PREDICTIONS.csv'
                row['checkpoint_present']=model.is_file()
                if model.is_file():
                    row['checkpoint_sha256']=sha(model);row['checkpoint_bytes']=model.stat().st_size
                row['result_present']=result.is_file()
                if result.is_file():
                    meta=json.loads(result.read_text())
                    row['result_completed']=meta.get('status')=='COMPLETED'
                    row['run_fingerprint']=meta.get('fingerprint')
                    row['recorded_evaluation_n']=meta.get('n_evaluation')
                row['saved_predictions_present']=preds.is_file()
                if preds.is_file():
                    row['saved_prediction_sha256']=sha(preds)
                    row['predictions_equal_packaged_records']=sha(preds)==sha(expected)
            runs.append(row)
    success=sum(bool(r['checkpoint_present']) for r in runs)
    matched=sum(bool(r['predictions_equal_packaged_records']) for r in runs)
    completed=sum(bool(r.get('result_completed')) for r in runs)
    status={'status':'CHECKPOINTS_PRESENT_NOT_INFERRED' if success==33 and completed==33 and matched==33 else 'INCOMPLETE_CHECKPOINT_INVENTORY',
            'runs':len(runs),'checkpoint_files_found':success,'completed_result_files':completed,
            'exact_saved_prediction_matches':matched,'training_executed':False,'inference_executed':False,
            'notes':'Checkpoint bytes are SHA-256 hashed and inventoried without deserialization. Matching records do not prove newly executed checkpoint inference.',
            'records':runs}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(status,indent=2)+'\n')
    return status
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--experiment-root',required=True,type=Path)
    ap.add_argument('--output',type=Path,default=Path('outputs/inference/C3_CHECKPOINT_INVENTORY.json'))
    a=ap.parse_args()
    r=execute(a.study,a.experiment_root,a.output)
    print(json.dumps({k:r[k] for k in ('status','runs','checkpoint_files_found','completed_result_files','exact_saved_prediction_matches')},indent=2))
