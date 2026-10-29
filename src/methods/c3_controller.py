from pathlib import Path
import argparse,json,os,subprocess,sys
FOLDERS={'four_view':'foreground_background_probes_v1','cultivar_holdout':'independent_training_probes_resumable_v3','size_class_holdout':'class_novelty_probes_v1'}
def main():
 p=argparse.ArgumentParser(description='Explicit optional C3 inference; default is input preflight. No training.')
 p.add_argument('action',nargs='?',default='preflight',choices=['preflight','execute']);p.add_argument('--stage-root',type=Path);p.add_argument('--study-root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 root=Path(__file__).resolve().parents[2];output=(a.output or root/'outputs/c3_fresh').resolve();missing=[]
 for path,label in [(a.stage_root,'stage-root'),(a.study_root,'study-root')]:
  if path is None or not path.is_dir():missing.append(label)
 inv=json.loads((root/'data/verification/C3_CHECKPOINT_INVENTORY.json').read_text())['checkpoints']
 if not missing:
  for r in inv:
   d=a.stage_root/'additional_evidence_20261024'/FOLDERS[r['group']]/r['run']
   for name in ['BEST_MODEL.pt','RESULT.json','HELDOUT_PREDICTIONS.csv']:
    if not (d/name).is_file():missing.append(r['run']+'/'+name)
  for base,rel in [(a.study_root,'data/source_split.csv'),(a.stage_root,'verified_v4_image_path_map.csv'),(a.stage_root,'additional_evidence_20261024/foreground_background_probes_v1/MASK_QC_ALL_IMAGES.csv'),(a.stage_root,'additional_evidence_20261024/class_novelty_probes_v1/image_tensor_cache/canonical_RGB_224_uint8.npy')]:
   if not (base/rel).is_file():missing.append(rel)
  for view in ['background','foreground','full','shape_only']:
   if not (a.stage_root/'additional_evidence_20261024/foreground_background_probes_v1/uint8_view_tensors'/(view+'.npy')).is_file():missing.append(view+' image tensor cache')
 if missing:
  print(json.dumps({'status':'REQUIRES_INPUTS','missing_inputs':missing,'training_executed':False,'inference_executed':False},indent=2));return 3
 if a.action=='preflight':print(json.dumps({'status':'INPUT_PATHS_PRESENT','inference_executed':False,'training_executed':False}));return 0
 output.mkdir(parents=True,exist_ok=False);env=os.environ.copy();env.update(DATEPALM_PACKAGE_ROOT=str(root),DATEPALM_STAGE_ROOT=str(a.stage_root.resolve()),DATEPALM_STUDY_ROOT=str(a.study_root.resolve()))
 allruns=[]
 for family in FOLDERS:
  env['DATEPALM_C3_OUTPUT']=str(output/family)
  with (output/(family+'.log')).open('w') as log:r=subprocess.run([sys.executable,str(Path(__file__).with_name('c3_replay_protocol.py')),family],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:print(json.dumps({'status':'C3_EXECUTION_FAILED','family':family,'diagnostic_log':family+'.log','training_executed':False}));return 2
  q=json.loads((output/family/'C3_SOURCE_FUNCTION_REPLAY.json').read_text());allruns+=q['runs']
 success=len(allruns)==33 and all(r['pass'] and r['calibration_bitwise_equal_original_cache'] and r['evaluation_bitwise_equal_original_cache'] for r in allruns)
 report={'status':'PASS_ALL_33' if success else 'COMPLETED_WITH_PARITY_DIFFERENCES','runs':allruns,'inference_executed':True,'training_executed':False,'thresholds_refitted':False}
 (output/'C3_FRESH_INFERENCE_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'runs':len(allruns),'training_executed':False}));return 0 if success else 2
if __name__=='__main__':raise SystemExit(main())
