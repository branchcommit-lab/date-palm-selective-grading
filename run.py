from pathlib import Path
import argparse,os,subprocess,sys,json
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Reproduce recorded results or validate component inference evidence.')
    p.add_argument('mode',nargs='?',default='results',choices=['results','inference','retrain'])
    p.add_argument('--action',default='audit',choices=['audit','preflight','execute'])
    p.add_argument('--output',type=Path)
    a,extra=p.parse_known_args();out=(a.output or ROOT/'outputs'/a.mode).resolve()
    path_flags={'--study-inputs','--feature-inputs','--appearance-checkpoint','--calibration-manifest','--development-manifest','--path-map'}
    for i in range(len(extra)-1):
        if extra[i] in path_flags:extra[i+1]=str(Path(extra[i+1]).resolve())
    env=os.environ.copy();env['DATEPALM_PACKAGE_ROOT']=str(ROOT);env['DATEPALM_OUTPUT']=str(out);env['MPLBACKEND']='Agg'
    if a.mode=='results':
        if extra:p.error('Unknown results arguments: '+' '.join(extra))
        cmd=[sys.executable,str(ROOT/'src/results_workflow.py')]
    elif a.mode=='inference':
        cmd=[sys.executable,str(ROOT/'src/methods/inference_driver.py'),a.action,'--package',str(ROOT),'--output',str(out)]+extra
    else:
        if extra:p.error('Retraining is documented in docs/RETRAINING.md; no training arguments are accepted by this safe controller.')
        out.mkdir(parents=True,exist_ok=True)
        report={'status':'NOT_EXECUTED_OPTIONAL_ORIGINAL_METHODS_AVAILABLE','training_executed':False,'instructions':'docs/RETRAINING.md','method_registry':'src/methods/manifest.json'}
        (out/'RETRAIN_STATUS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return 0
    completed=subprocess.run(cmd,env=env,cwd=str(ROOT))
    if completed.returncode:return completed.returncode
    if a.mode=='inference' and a.action in ('audit','execute'):
        env['DATEPALM_ANALYSIS_MODE']='inference'
        env['DATEPALM_ANALYSIS_ACTION']=a.action
        env['DATEPALM_INFERENCE_STATUS']=str(out/('INFERENCE_EVIDENCE_STATUS.json' if a.action=='audit' else 'INFERENCE_EXECUTION_STATUS.json'))
        env['DATEPALM_FRESH_INFERENCE_OUTPUT']=str(out)
        return subprocess.run([sys.executable,str(ROOT/'src/results_workflow.py')],env=env,cwd=str(ROOT)).returncode
    return 0
if __name__=='__main__':raise SystemExit(main())
