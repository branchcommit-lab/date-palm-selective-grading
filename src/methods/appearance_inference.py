#!/usr/bin/env python3
"""Read-only C11 checkpoint-to-score audit using extracted original method definitions.

No model training, calibration, threshold optimisation, test-set model selection or
mutation of original benchmark/checkpoint/score files. Only generated audit outputs
are created under --output. This is not a full raw-image dataset release.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

EXPECTED_CKPT_SHA256 = 'f93f8debbb69f581c3b21379097af70f55df8bdf1d0557ff7c3ca77c72309eb1'
EXPECTED_METHOD_SHA256 = '87771197a613e281c67313fac28163961592aeb6b5135e7353d0b24a38fe5316'
EXPECTED_COUNTS = {'calibration': 1668, 'development': 12525}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for chunk in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def checked_file(path: Path, kind: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'{kind} missing: {path}')
    return path

def get_samples(name, manifest_path, saved_path, pd):
    manifest_path, saved_path = checked_file(manifest_path,'benchmark manifest'),checked_file(saved_path,'saved C11 scores')
    manifest = pd.read_csv(manifest_path, dtype={'sample_id':str,'source_id':str,'source_group':str})
    saved = pd.read_csv(saved_path, dtype={'sample_id':str,'source_id':str,'source_group':str})
    expected=EXPECTED_COUNTS[name]
    if len(manifest)!=expected or len(saved)!=expected:
        raise ValueError(f'{name} population wrong: manifest={len(manifest)}, saved={len(saved)}, expected={expected}')
    for title, df, columns in [
        ('benchmark',manifest,{'sample_id','image_path','mask_path','fg_path','image_sha256','mask_sha256'}),
        ('saved-score',saved,{'sample_id','score','image_sha256','mask_sha256'})]:
        missing=columns-set(df.columns)
        if missing:raise ValueError(f'{name} {title} missing fields: {sorted(missing)}')
        ids=df['sample_id']
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f'{name} {title} has missing/duplicate sample_id')
    if set(manifest.sample_id)!=set(saved.sample_id):
        miss=set(manifest.sample_id)-set(saved.sample_id)
        extra=set(saved.sample_id)-set(manifest.sample_id)
        raise ValueError(f'{name} sample population mismatch: missing {len(miss)}, extra {len(extra)}')
    keyed=saved.set_index('sample_id').loc[manifest.sample_id]
    for col in ('image_sha256','mask_sha256'):
        if not (manifest[col].astype(str).to_numpy()==keyed[col].astype(str).to_numpy()).all():
            raise ValueError(f'{name} authentic saved-score versus benchmark {col} mismatch')
    if not pd.to_numeric(saved['score'],errors='coerce').notna().all():
        raise ValueError(f'{name} saved scores contain non-numeric/missing values')
    if 'source_id' in manifest and 'source_id' in keyed:
        if not (manifest['source_id'].astype(str).to_numpy()==keyed['source_id'].astype(str).to_numpy()).all():
            raise ValueError(f'{name} source_id mismatch')
    return manifest,saved

def verify_images(samples, name):
    missing, wrong=[],[]
    fg_paths=set()
    for i, row in enumerate(samples.itertuples(index=False)):
        for pcol,hcol in [('image_path','image_sha256'),('mask_path','mask_sha256')]:
            path=Path(getattr(row,pcol)); expected=str(getattr(row,hcol)).lower()
            if not path.is_file():missing.append((row.sample_id,pcol,str(path)))
            elif sha256(path)!=expected:wrong.append((row.sample_id,pcol,str(path)))
        fg_paths.add(str(row.fg_path))
        if (i+1)%1000==0:print(f'{name} verified {i+1}/{len(samples)} benchmark images and masks',flush=True)
    fg_missing=[p for p in sorted(fg_paths) if not Path(p).is_file()]
    if missing or wrong or fg_missing:
        raise RuntimeError(f'{name} integrity FAILED: {len(missing)} missing, {len(wrong)} wrong SHA-256, '
                           f'{len(fg_missing)} missing foreground paths; first examples: '
                           f'{(missing+wrong+[("foreground",p) for p in fg_missing])[:8]}')
    return len(samples),len(fg_paths)

def install_exact_method(source_path):
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18, ResNet18_Weights
    from PIL import Image, ImageOps
    from tqdm.auto import tqdm
    source=checked_file(source_path,'authenticated original method source')
    if sha256(source)!=EXPECTED_METHOD_SHA256:
        raise RuntimeError('Original method definitions do not match the authenticated transfer SHA256')
    names=dict(__name__='original_c11_method_namespace',np=np,torch=torch,nn=nn,F=F,cv2=cv2,
               DataLoader=DataLoader,resnet18=resnet18,ResNet18_Weights=ResNet18_Weights,
               Image=Image,ImageOps=ImageOps,tqdm=tqdm,Path=Path,
               SHAPE=(256,256),V4_PIN_MEMORY=False,live_update=lambda *a,**k:None)
    exec(compile(source.read_bytes(),str(source),'exec'),names)
    if not callable(names.get('matched_predict')):raise RuntimeError('Original matched_predict not available')
    return names['matched_predict'],torch,np

def replay(name, samples, saved, matched_predict, checkpoint, device, batch_size, np, pd, output, atol, rtol):
    results=matched_predict(str(checkpoint),samples,device,batch_size=batch_size,keep_maps=False)
    frame=pd.DataFrame([{'sample_id':p['sample_id'],'score_replayed':p['score']} for p in results])
    if len(frame)!=len(samples) or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise RuntimeError(f'{name} C11 replay emitted invalid ID population')
    expected=saved[['sample_id','score']].rename(columns={'score':'score_saved'})
    joined=expected.merge(frame,on='sample_id',validate='one_to_one',indicator=True)
    if len(joined)!=len(samples) or not (joined['_merge']=='both').all():
        raise RuntimeError(f'{name} unmatched replay sample population')
    joined=joined.drop(columns=['_merge'])
    a=joined.score_saved.to_numpy(dtype=np.float64)
    b=joined.score_replayed.to_numpy(dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise RuntimeError(f'{name} non-finite score values')
    d=np.abs(a-b)
    close=np.isclose(a,b,atol=atol,rtol=rtol)
    joined['absolute_difference']=d
    joined['within_tolerance']=close
    out=output/f'C11_1049_{name}_replayed_comparison.csv'
    joined.to_csv(out,index=False)
    summary={'population':name,'count':len(joined),'saved_scores_sha256':None,
             'replay_comparison_sha256':sha256(out),'max_absolute_difference':float(d.max()),
             'median_absolute_difference':float(np.median(d)),
             'mean_absolute_difference':float(d.mean()),
             'num_outside_tolerance':int((~close).sum()),
             'atol':atol,'rtol':rtol,
             'continuous_score_parity':'PASS' if close.all() else 'FAIL',
             'note':'PASS certifies agreement under explicit numeric tolerance, not external validation or retraining.'}
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key in ('checkpoint','calibration-manifest','development-manifest','calibration-scores','development-scores','output'):
        ap.add_argument('--'+key,required=True,type=Path)
    ap.add_argument('--authentic-source',type=Path,default=Path(__file__).with_name('C11_AUTHENTIC_METHOD_SOURCES.py'))
    ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    ap.add_argument('--batch-size',default=24,type=int)
    ap.add_argument('--atol',type=float,default=1e-8,help='Predeclared absolute numerical tolerance; not a score threshold')
    ap.add_argument('--rtol',type=float,default=1e-4,help='Predeclared relative numerical tolerance; not a score threshold')
    ap.add_argument('--path-map',type=Path,help='JSON mapping original image/mask/foreground paths to relocated files; relative targets resolve from the JSON directory')
    ap.add_argument('--preflight-only',action='store_true',help='Validate inputs, hashes, populations; do not execute model')
    args=ap.parse_args()
    if args.batch_size<1 or args.atol<0 or args.rtol<0:ap.error('batch size and tolerances must be non-negative (batch >=1)')
    args.output.mkdir(parents=True,exist_ok=True)
    report={'audit':'C11_1049_ORIGINAL_INFERENCE_PARITY','utc':datetime.now(timezone.utc).isoformat(),
            'inference_executed':False,'training_executed':False,'external_validation_performed':False,
            'checkpoint_path':str(args.checkpoint),'authentic_method_source':str(args.authentic_source),
            'checkpoint_expected_sha256':EXPECTED_CKPT_SHA256,
            'source_expected_sha256':EXPECTED_METHOD_SHA256,
            'status':'PREFLIGHT_NOT_COMPLETE','populations':{}}
    output=args.output/'C11_1049_parity_report.json'
    def write_report():
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('Report:',output,flush=True)
    try:
        checked_file(args.checkpoint,'checkpoint')
        ckh=sha256(args.checkpoint)
        report['checkpoint_observed_sha256']=ckh
        if ckh!=EXPECTED_CKPT_SHA256:raise RuntimeError('C11 checkpoint SHA mismatch; refusing deserialization')
        checked_file(args.authentic_source,'original method code')
        if sha256(args.authentic_source)!=EXPECTED_METHOD_SHA256:raise RuntimeError('Extracted source SHA mismatch')
        import pandas as pd
        items={
          'calibration':(args.calibration_manifest,args.calibration_scores),
          'development':(args.development_manifest,args.development_scores),
        }
        dataset={}
        for name,(manifest,reference) in items.items():
            samples,saved=get_samples(name,manifest,reference,pd)
            if args.path_map:
                relocation=json.loads(args.path_map.read_text())
                for column in ('image_path','mask_path','fg_path'):
                    def relocate(value):
                        target=Path(relocation.get(str(value),str(value)))
                        return str(target if target.is_absolute() else (args.path_map.parent/target).resolve())
                    samples[column]=samples[column].map(relocate)
            n,fg=verify_images(samples,name)
            report['populations'][name]={'samples':n,'foreground_paths':fg,
                'manifest_path':str(manifest),'manifest_sha256':sha256(manifest),
                'scores_path':str(reference),'saved_scores_sha256':sha256(reference),
                'image_and_mask_hashes_verified':True}
            dataset[name]=(samples,saved)
        report['status']='PREFLIGHT_PASS'
        write_report()
        if args.preflight_only:
            print('PREFLIGHT PASS. No inference executed.')
            return 0
        matched_predict,torch,np=install_exact_method(args.authentic_source)
        device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        report['device']=device
        report['torch_version']=torch.__version__
        report['inference_executed']=True
        report['status']='INFERENCE_STARTED'
        write_report()
        for name,(samples,saved) in dataset.items():
            print('Running original C11 inference:',name,len(samples),'observations',flush=True)
            summary=replay(name,samples,saved,matched_predict,args.checkpoint,device,
                           args.batch_size,np,pd,args.output,args.atol,args.rtol)
            summary['saved_scores_sha256']=report['populations'][name]['saved_scores_sha256']
            report['populations'][name].update(summary)
            write_report()
        report['status']='PARITY_PASS' if all(report['populations'][n]['continuous_score_parity']=='PASS' for n in dataset) else 'PARITY_FAIL'
        write_report()
        return 0 if report['status']=='PARITY_PASS' else 2
    except Exception as exc:
        report['status']='FAILED'
        report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc()
        write_report()
        print(report['error'],file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
