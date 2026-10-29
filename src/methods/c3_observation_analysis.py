#!/usr/bin/env python3
"""Regenerate all three C3 evaluation families from individual prediction records.

No model training, checkpoint deserialization, optimization, or test selection occurs.
The source-level run records are scientific observations, not independently rerun inference.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

FAMILIES={
 'four_view':('C3_FOUR_VIEW_ALL_12_RUNS.csv',12),
 'cultivar_holdout':('C3_CULTIVAR_HOLDOUT_ALL_12_RUNS.csv',12),
 'size_holdout':('C3_CLASS_HOLDOUT_ALL_9_RUNS.csv',9),
}
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compute(root:Path, output:Path, rtol=0, atol=1e-10):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics=['n','referral_fraction','accuracy','macro_f1','n_known','n_unknown',
             'unknown_AUROC','unknown_AUPRC','known_accuracy','unknown_referral']
    rows=[];checks=[];artifacts=[];pairs=[]
    for family,(summary_file,expected_runs) in FAMILIES.items():
        summary=pd.read_csv(root/'tables'/summary_file)
        if len(summary)!=expected_runs or summary.run.duplicated().any():
            raise AssertionError('Incomplete or ambiguous design for '+family)
        for row in summary.itertuples(index=False):
            path=root/'raw_records'/'c3'/family/(row.run+'.csv')
            if not path.is_file():raise FileNotFoundError(path)
            frame=pd.read_csv(path,dtype={'source_id':str,'source_group':str})
            if len(frame)!=row.n or frame.source_id.isna().any() or frame.source_id.duplicated().any():
                raise AssertionError('Observation identity/size violation '+row.run)
            if not {'native_size_label','predicted_label','referred','max_probability','uncertainty'}<=set(frame):
                raise AssertionError('Missing required prediction columns in '+row.run)
            if not frame.split.eq('development').all():
                raise AssertionError('Non-development source in '+row.run)
            if not frame.referred.isin([True,False]).all():
                raise AssertionError('Malformed referral flags in '+row.run)
            y=frame.native_size_label.astype(str).to_numpy()
            yh=frame.predicted_label.astype(str).to_numpy()
            known=np.isin(y,['Small','Medium','Large'])
            if not known.all(): raise AssertionError('Invalid grading labels in '+row.run)
            result={'analysis_family':family,'experiment':row.experiment,'run':row.run,
                    'condition':row.condition,'seed':int(row.seed),'n':len(frame),
                    'referral_fraction':float(frame.referred.mean())}
            if family=='size_holdout':
                if 'is_withheld_class' not in frame:raise AssertionError('No unknown-class indicators')
                unknown=frame.is_withheld_class.to_numpy(dtype=bool)
                if unknown.sum()==0 or (~unknown).sum()==0:raise AssertionError('Missing known or unknown cases')
                u=frame.uncertainty.to_numpy(dtype=float)
                result.update(n_known=int((~unknown).sum()),n_unknown=int(unknown.sum()),
                              unknown_AUROC=float(roc_auc_score(unknown,u)),
                              unknown_AUPRC=float(average_precision_score(unknown,u)),
                              known_accuracy=float(np.mean(y[~unknown]==yh[~unknown])),
                              unknown_referral=float(frame.referred.to_numpy(dtype=bool)[unknown].mean()))
            else:
                result.update(accuracy=float(np.mean(y==yh)),
                              macro_f1=float(f1_score(y,yh,average='macro')))
            for metric in metrics:
                if metric not in result:continue
                val=getattr(row,metric)
                okay=bool(np.isclose(result[metric],val,rtol=rtol,atol=atol,equal_nan=True))
                checks.append({'run':row.run,'metric':metric,'match':okay,
                               'recomputed':result[metric],'reference':float(val)})
                if not okay:raise AssertionError(f'{row.run}: {metric} mismatch {result[metric]} vs {val}')
            rows.append(result)
            artifacts.append({'family':family,'run':row.run,'sha256':sha256(path),'records':len(frame)})
            pairs.append(frame.assign(analysis_family=family,run=row.run))
    d=pd.DataFrame(rows)
    d.to_csv(output/'C3_recomputed_from_individual_predictions.csv',index=False,float_format='%.16g')
    status={'status':'PASS','runs':len(d),'individual_predictions':int(sum(x['records'] for x in artifacts)),
            'numeric_checks':len(checks),'failed_checks':sum(not x['match'] for x in checks),
            'inference_executed':False,'training_executed':False,
            'interpretation':'Run-level metrics recalculated from independent recorded predictions; no model checkpoint replay performed',
            'inputs':artifacts}
    (output/'C3_OBSERVATION_PARITY.json').write_text(json.dumps(status,indent=2)+'\n')
    return status,d
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    report,frame=compute(a.data,a.output)
    print(json.dumps({k:report[k] for k in ('status','runs','individual_predictions','numeric_checks','failed_checks')},indent=2))
