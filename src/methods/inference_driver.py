#!/usr/bin/env python3
"""Study inference entry point: audit authenticated records or execute original model methods.

Both inference components are fail-closed. No training path is invoked here.
Running only the component-level procedures does NOT independently infer the 33 C3 models.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import pandas as pd
import numpy as np

C11_SHA='f93f8debbb69f581c3b21379097af70f55df8bdf1d0557ff7c3ca77c72309eb1'
RGB_SHA='0f2c1b75dfe9d7dfc3498b88a5bd065b97e8cf5a28262c32bd2d633a386bb909'
REQUIRED_CAL=1668
REQUIRED_LOCKED=12510
REQUIRED_DEV=12525

def h(path):
    val=hashlib.sha256()
    with Path(path).open('rb') as src:
        for buf in iter(lambda:src.read(4*1024*1024),b''):val.update(buf)
    return val.hexdigest()

def required(path,label):
    if path is None or not Path(path).is_file():raise FileNotFoundError(f'{label} missing: {path}')
    return Path(path)

def inspect_evidence(root,output):
    root=Path(root); output=Path(output)
    evidence=root/'data'/'verification'
    c11=json.loads(required(evidence/'appearance_parity.json','C11 score parity record').read_text())
    dandhi=json.loads(required(evidence/'candidate_parity.json','Candidate probability parity record').read_text())
    if c11.get('status')!='PASS_AUTHENTIC_C11_CHECKPOINT_TO_SCORE_AND_DECISION_PARITY':
        raise AssertionError('C11 recorded checkpoint parity not successful')
    if dandhi.get('status')!='PASS_DANDHI_CANDIDATE_PROBABILITY_REPLAY':
        raise AssertionError('Dandhi recorded candidate parity not successful')
    if c11['checkpoint_sha256_expected']!=C11_SHA or dandhi['rgb_checkpoint_sha256']!=RGB_SHA:
        raise AssertionError('Original model identity differs')
    if (c11['splits']['calibration']['rows'],c11['splits']['development']['rows'])!=(REQUIRED_CAL,REQUIRED_DEV):
        raise AssertionError('C11 observation counts differ')
    if dandhi['calibration_rows']!=REQUIRED_CAL or dandhi['locked_test_rows']!=REQUIRED_LOCKED:
        raise AssertionError('Dandhi observations counts differ')
    if any(c11['splits'][x]['mismatches_above_tolerance']!=0 for x in ('calibration','development')):
        raise AssertionError('C11 mismatch')
    if c11['frozen_threshold_parity']['development_decision_mismatches']!=0:
        raise AssertionError('Frozen threshold mismatch')
    for file,split in [('c11_calibration.csv','calibration'),('c11_evaluation.csv','development')]:
        actual=h(root/'data'/'raw_records'/file)
        expected=c11['splits'][split]['reference_sha256']
        if actual!=expected:raise AssertionError(f'C11 reference record hash mismatch: {file}')
    df=pd.read_csv(root/'data'/'raw_records'/'final_candidate_predictions.csv')
    dfp=pd.read_csv(root/'data'/'tables'/'candidate_probabilities_locked_test.csv')
    if len(df)!=REQUIRED_LOCKED or len(dfp)!=REQUIRED_LOCKED:raise AssertionError('Candidate record count')
    if set(df.sample_id)!=set(dfp.sample_id):raise AssertionError('Candidate sample identity mismatch')
    joined=df[['sample_id','candidate_grade','refer_candidate_conf_10','refer_candidate_dual_5_5']].merge(
        dfp[['sample_id','predicted_size','refer_candidate_conf_10','refer_candidate_dual_5_5']],
        on='sample_id',suffixes=('_obs','_prob'),validate='one_to_one')
    mismatch_counts={
      'grades':int((joined.candidate_grade!=joined.predicted_size).sum()),
      'confidence_referrals':int((joined.refer_candidate_conf_10_obs!=joined.refer_candidate_conf_10_prob).sum()),
      'dual_referrals':int((joined.refer_candidate_dual_5_5_obs!=joined.refer_candidate_dual_5_5_prob).sum()),
    }
    if any(mismatch_counts.values()):raise AssertionError('Candidate observation mismatch '+str(mismatch_counts))
    from evidence_audit import validate_certificates
    certificate_checks=validate_certificates(root)
    source=root/'data'/'metadata'/'source_split.csv'
    cohort=pd.read_csv(source,dtype={'source_id':str})
    eligible=cohort[cohort.split.notna()]
    if len(eligible)!=5828:raise AssertionError('Cohort size differs')
    from c3_observation_analysis import compute
    c3,values=compute(root/'data',output/'c3')
    status={
      'status':'PASS_AUTHENTIC_INFERENCE_EVIDENCE_AND_C3_OBSERVATIONS',
      'operation':'audit','certificate_checks':certificate_checks,
      'training_executed':False,'new_model_inference_executed':False,
      'appearance':{'verification':'RECORDED_CHECKPOINT_TO_SCORE_PARITY_PASS',
        'splits':['calibration','development'],'scores':REQUIRED_CAL+REQUIRED_DEV,
        'score_mismatches':0,'decision_mismatches':0},
      'candidate':{'verification':'RECORDED_FROZEN_FEATURE_AND_RGB_LOGIT_REPLAY_PASS',
        'calibration_probabilities':REQUIRED_CAL,'locked_probabilities':REQUIRED_LOCKED,
        'sample_level_mismatches':mismatch_counts},
      'supplementary':{'verification':'PASS_RECORDED_FRESH_CHECKPOINT_REPLAY_AND_OBSERVATION_REAGGREGATION',
        'runs':c3['runs'],'observations':c3['individual_predictions'],
        'metric_checks':c3['numeric_checks']},
      'scope_limitations':[
        'C11 checkpoint-to-score replay covers calibration and development, not all locked-test synthetic variants',
        'Dandhi calibration and locked-test parity used frozen layer3 features and RGB logit cache; full image-to-grade replay is not independently certified',
        'All 33 C3 supplementary checkpoints were inferred in the recorded server validation; this evidence-audit action does not rerun them',
        'No model retraining was conducted or required',
      ]}
    output.mkdir(parents=True,exist_ok=True)
    (output/'INFERENCE_EVIDENCE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    return status

def paths_for_execution(args):
    missing=[]
    study=Path(args.study_inputs) if args.study_inputs else None
    feature=Path(args.feature_inputs) if args.feature_inputs else None
    model=Path(args.appearance_checkpoint) if args.appearance_checkpoint else None
    calon=Path(args.calibration_manifest) if args.calibration_manifest else None
    dev=Path(args.development_manifest) if args.development_manifest else None
    if args.components in ('candidate','both'):
        if not study or not study.is_dir():
            missing.append('study-inputs directory for grading benchmark and C11 study scores')
        else:
            for name in ('calibration/scores/C11_1049.csv','results/C11_1049.csv',
                 'calibration/benchmark/manifest.csv','results/fixed_benchmark/manifest.csv'):
                if not (study/name).is_file():missing.append('study-inputs/'+name)
        if not feature or not feature.is_dir():
            missing.append('feature-inputs directory with frozen features and RGB-logit caches')
        else:
            required=[
                'v51_representation_finetune_pilot/rgb_representation_finetuned.pt',
                'v51_grader_decision_pilot/development_model_selection.json',
                'v51_grader_decision_pilot/v51_improved_grader_all_variant_predictions.csv',
                'v51_dandhi_targeted_rgb_pilot/development_only_selection.json',
                'v51_dandhi_targeted_rgb_pilot/frozen_referral_cutoffs.json',
                'v51_dandhi_targeted_rgb_pilot/targeted_variant_predictions.csv',
            ]
            required += [f'v51_locked_spatial/size_layer4_seed{seed}.pt' for seed in (11,23,37)]
            required += [f'v51_locked_spatial/fixed_layer3_{split}.{ext}'
                        for split in ('calibration','locked_test') for ext in ('npy','json')]
            required += [f'v51_dandhi_targeted_rgb_pilot/rgb_prediction_cache/{split}'
                        for split in ('calibration','locked_test')]
            for name in required:
                if not (feature/name).exists():missing.append('feature-inputs/'+name)
            rgb=feature/'v51_representation_finetune_pilot/rgb_representation_finetuned.pt'
            if rgb.is_file() and h(rgb)!=RGB_SHA:raise AssertionError('RGB checkpoint SHA256 differs')
    if args.components in ('appearance','both'):
        if not model or not model.is_file():missing.append('appearance-checkpoint')
        elif h(model)!=C11_SHA:raise AssertionError('C11 checkpoint SHA256 differs')
        for description,path,split in [('calibration-manifest',calon,'calibration'),
                                       ('development-manifest',dev,'development')]:
            if not path or not path.is_file():missing.append(description)
            else:
                record=json.loads((Path(args.package)/'data/verification/appearance_parity.json').read_text())
                expected=record['splits'][split]['manifest_sha256']
                if h(path)!=expected:
                    raise AssertionError(description+' SHA256 differs from the verified study benchmark')
                mapping=json.loads(args.path_map.read_text()) if args.path_map else {}
                manifest=pd.read_csv(path)
                for column in ('image_path','mask_path','fg_path'):
                    if column not in manifest:
                        missing.append(description+' missing column '+column)
                        continue
                    absent=0
                    for item in manifest[column].astype(str).unique():
                        target=Path(mapping.get(item,item))
                        if args.path_map and not target.is_absolute():target=args.path_map.parent/target
                        if not target.is_file():absent+=1
                    if absent:missing.append(description+': '+str(absent)+' unavailable '+column+' files (use --path-map for relocated data)')
        method=Path(args.package)/'src'/'methods'/'appearance_source.py'
        if not method.is_file() or h(method)!='87771197a613e281c67313fac28163961592aeb6b5135e7353d0b24a38fe5316':
            raise AssertionError('Appearance method byte identity differs')
    import importlib.util
    if missing:
        return missing
    for dependency in ('torch','torchvision','PIL','cv2','tqdm'):
        if importlib.util.find_spec(dependency) is None:
            missing.append('optional inference dependency: '+dependency)
    if args.components in ('candidate','both') and feature:
        identities=json.loads((Path(args.package)/'checkpoints/manifest.json').read_text())
        for seed in (11,23,37):
            path=feature/f'v51_locked_spatial/size_layer4_seed{seed}.pt'
            if path.is_file() and h(path)!=identities[f'models/size_grader_seed{seed}.pt']['sha256']:
                raise AssertionError('Size-head checkpoint SHA256 differs: '+str(seed))
    return missing

def execute(args):
    root=Path(args.package).resolve();out=Path(args.output).resolve()
    check=inspect_evidence(root,out/'audit')
    missing=paths_for_execution(args)
    if missing:return {'status':'REQUIRES_INPUTS','action':'execute','missing_inputs':missing,'new_model_inference_executed':False,'training_executed':False}
    output={'status':'IN_PROGRESS','components_requested':args.components,
            'training_executed':False,'components':{},
            'full_C1_to_C3_from_raw_images_certified':False,
            'C3_checkpoint_inference_executed':False}
    if args.components in ('candidate','both'):
        target=out/'candidate';target.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(root/'src'/'methods'/'candidate_inference.py'),
             '--v4',str(Path(args.study_inputs).resolve()),
             '--stage1',str(Path(args.feature_inputs).resolve()),
             '--output',str(target)]
        subprocess.run(cmd,check=True)
        report=json.loads((target/'00_REPLAY_REPORT.json').read_text())
        if report['status']!='PASS_AUTHENTIC_SAVED_FEATURE_AND_RGB_LOGIT_INFERENCE_REPLAY':
            raise AssertionError('Candidate model replay failed: '+report['status'])
        scores=pd.read_csv(target/'locked_test_candidate_probabilities.csv')
        if len(scores)!=REQUIRED_LOCKED or scores.sample_id.duplicated().any():
            raise AssertionError('Candidate replay rows invalid')
        obs=pd.read_csv(root/'data/raw_records/final_candidate_predictions.csv')
        scores=scores.set_index('sample_id').loc[obs.sample_id].reset_index()
        if (scores.candidate_predicted_size.to_numpy()!=obs.candidate_grade.to_numpy()).any():
            raise AssertionError('Final grade mismatch')
        if (scores.candidate_refer_confidence_10.to_numpy()!=obs.refer_candidate_conf_10.to_numpy()).any():
            raise AssertionError('Final 10% confidence referral mismatch')
        if (scores.candidate_refer_dual_5_5.to_numpy()!=obs.refer_candidate_dual_5_5.to_numpy()).any():
            raise AssertionError('Final combined referral mismatch')
        output['components']['candidate']={'status':'PASS','rows':len(scores),
             'method':'Frozen features, checkpoint grade heads, authentic cached RGB logits',
             'reconstructed_probabilities_csv':str(target/'locked_test_candidate_probabilities.csv')}
    if args.components in ('appearance','both'):
        target=out/'appearance';target.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(root/'src'/'methods'/'appearance_inference.py'),
             '--checkpoint',str(Path(args.appearance_checkpoint).resolve()),
             '--authentic-source',str(root/'src'/'methods'/'appearance_source.py'),
             '--calibration-manifest',str(Path(args.calibration_manifest).resolve()),
             '--development-manifest',str(Path(args.development_manifest).resolve()),
             '--calibration-scores',str(root/'data/raw_records/c11_calibration.csv'),
             '--development-scores',str(root/'data/raw_records/c11_evaluation.csv'),
             '--atol','1e-5','--rtol','0',
             '--output',str(target)]
        if args.path_map:cmd.extend(['--path-map',str(args.path_map.resolve())])
        subprocess.run(cmd,check=True)
        c11=json.loads((target/'C11_1049_parity_report.json').read_text())
        if c11['status']!='PARITY_PASS':raise AssertionError('C11 execution parity not successful')
        frozen=json.loads((root/'configs/frozen_parameters.json').read_text())['C11_threshold']
        decision_mismatches={}
        for split in ('calibration','development'):
            comparison=pd.read_csv(target/f'C11_1049_{split}_replayed_comparison.csv')
            decision_mismatches[split]=int(((comparison.score_saved>frozen)!=(comparison.score_replayed>frozen)).sum())
            if decision_mismatches[split]:raise AssertionError('Frozen C11 decision mismatch: '+split)
        output['components']['appearance']={'status':'PASS','frozen_decision_mismatches':decision_mismatches,'score_rows':REQUIRED_CAL+REQUIRED_DEV,
            'checkpoint_to_score_inference':True,
            'calibration_and_development_only':True}
    output['status']='PASS_COMPONENT_INFERENCE_SCOPE_ONLY'
    (out/'INFERENCE_EXECUTION_STATUS.json').write_text(json.dumps(output,indent=2)+'\n')
    return output

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',nargs='?',default='audit',choices=('audit','preflight','execute'))
    ap.add_argument('--package',type=Path,default=Path(__file__).resolve().parent.parent.parent)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--components',choices=('candidate','appearance','both'),default='both')
    ap.add_argument('--study-inputs',help='Original study score and benchmark directory')
    ap.add_argument('--feature-inputs',help='Original frozen size-layer features and RGB-logit directory')
    ap.add_argument('--appearance-checkpoint',help='C11_1049 checkpoint with exact SHA256')
    ap.add_argument('--path-map',type=Path,help='JSON mapping original image/mask/foreground paths to relocated files')
    ap.add_argument('--calibration-manifest',help='Original C11 calibration sample/mask benchmark manifest')
    ap.add_argument('--development-manifest',help='Original C11 development sample/mask benchmark manifest')
    args=ap.parse_args()
    args.package=args.package.resolve()
    if args.output is None:args.output=args.package/'outputs'/'inference'
    try:
        if args.action=='audit':report=inspect_evidence(args.package,args.output)
        elif args.action=='preflight':
            baseline=inspect_evidence(args.package,args.output/'audit')
            missing=paths_for_execution(args)
            report={'status':'READY' if not missing else 'REQUIRES_INPUTS',
                    'components':args.components,'missing_inputs':missing,'new_model_inference_executed':False,
                    'training_executed':False}
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'INFERENCE_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
        else: report=execute(args)
        if args.action=='execute' and report['status']=='REQUIRES_INPUTS':
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'INFERENCE_EXECUTION_STATUS.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return 0 if report['status']!='REQUIRES_INPUTS' else 3
    except Exception as e:
        args.output.mkdir(parents=True,exist_ok=True)
        failure={'status':'FAILED','action':args.action,'error':str(e),'training_executed':False}
        (args.output/'INFERENCE_FAILED.json').write_text(json.dumps(failure,indent=2)+'\n')
        print(json.dumps(failure,indent=2),file=sys.stderr)
        return 2
if __name__=='__main__':raise SystemExit(main())
