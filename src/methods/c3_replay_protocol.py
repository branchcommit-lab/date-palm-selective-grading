"""Inference-only C3 replay. Original artifacts are read-only; no optimizer or threshold fitting."""
from pathlib import Path
import ast, json, hashlib, time, platform, shutil, zipfile, os
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset

PACKAGE=Path(os.environ['DATEPALM_PACKAGE_ROOT']).resolve()
BASE=PACKAGE/'src/methods'
HOME=Path.home()
STAGE=Path(os.environ['DATEPALM_STAGE_ROOT']).resolve()
V4=Path(os.environ['DATEPALM_STUDY_ROOT']).resolve()
ROOT=STAGE/'additional_evidence_20261024'
VIEW=ROOT/'foreground_background_probes_v1'
SIZE=ROOT/'class_novelty_probes_v1'
import sys
OUT=Path(os.environ['DATEPALM_C3_OUTPUT']); OUT.mkdir(exist_ok=False)
TOL=1e-5
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
def definitions(name,names,ns):
 nb=json.loads((BASE/(name+'.ipynb')).read_text())
 code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
 tree=ast.parse(code); selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in selected}==set(names)
 exec(compile(ast.Module(body=selected,type_ignores=[]),name+'.ipynb','exec'),ns)
 return hashlib.sha256(code.encode()).hexdigest()

assert torch.cuda.is_available(),'CUDA unavailable; no automatic CPU fallback'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
report={'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),'training_executed':False,'thresholds_refitted':False,'inference_executed':True,'absolute_tolerance':TOL,'torch':torch.__version__,'torchvision':torchvision.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'batch_size':{'four_view':128,'cultivar_holdout':32,'size_class_holdout':128},'cuda_matmul_tf32':{'four_view':True,'cultivar_holdout':False,'size_class_holdout':True},'cudnn_tf32':True,'cudnn_benchmark':True,'autocast':False,'runs':[]}
assert sha(V4/'data/source_split.csv')=='bcd80e4b8ec20ffaddfc124d742b665ebf6f6347658e4c6fc750d9b92908c0f1'
M=pd.read_csv(V4/'data/source_split.csv',dtype=str,keep_default_na=False)
F=pd.read_csv(STAGE/'verified_v4_image_path_map.csv',dtype=str,keep_default_na=False)
M=M.loc[M.eligibility.eq('eligible')].copy()
D=M.loc[M.split.isin(['train','development'])].merge(F[['source_id','file_sha256','verified_image_path']],on='source_id',validate='one_to_one',suffixes=('','_map'))
assert len(D)==4438 and D.file_sha256.eq(D.file_sha256_map).all()
dev=D[D.split.eq('development')]
image_records=[]
for r in dev.itertuples():
 actual=sha(r.verified_image_path); assert actual==r.file_sha256,r.source_id
 image_records.append({'source_id':r.source_id,'sha256':actual,'bytes':Path(r.verified_image_path).stat().st_size})
save(OUT/'IMAGE_BYTE_VERIFICATION.json',image_records)
QA=pd.read_csv(VIEW/'MASK_QC_ALL_IMAGES.csv',dtype=str,keep_default_na=False)
DV=D.loc[~D.source_group.isin(set(QA.loc[QA.status.ne('OK'),'source_group']))].copy()
assert len(DV)==4402
PRE=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
common={'np':np,'pd':pd,'torch':torch,'models':models,'nn':nn,'Image':Image,'ImageOps':ImageOps,'PRE':PRE,'Dataset':Dataset,'hashlib':hashlib,'SIZES':['Large','Medium','Small']}
cv=dict(common); definitions('cultivar_holdout',['stable_half','make_fold','image_tensor','Fruits','new_model'],cv)
sz=dict(common); definitions('size_holdout',['bucket','make_fold','Fruits'],sz)
sz['IMAGE_INDEX_BY_PATH']={str(r.verified_image_path):i for i,r in enumerate(D.itertuples(index=False))}
sz['IMAGE_ARRAY']=np.load(SIZE/'image_tensor_cache/canonical_RGB_224_uint8.npy',mmap_mode='r')
vw=dict(common); definitions('visual_cues',['stable_bucket','image_tensor','Fruits'],vw)
vw['VIEW_INDEX_BY_PATH']={str(r.verified_image_path):i for i,r in enumerate(DV.itertuples(index=False))}
vw['VIEW_ARRAYS']={v:np.load(VIEW/'uint8_view_tensors'/(v+'.npy'),mmap_mode='r') for v in ['background','foreground','full','shape_only']}
vw['MASK_LOOKUP']={str(r.verified_image_path):VIEW/'source_masks'/(hashlib.sha256(str(r.source_id).encode()).hexdigest()+'.png') for r in DV.itertuples(index=False)}
# Re-decode every development photo used by C3, including all four original masked views.
pixel_checks=[]
for r in dev.itertuples():
 raw=cv['image_tensor'](r.verified_image_path)
 cached,_=sz['Fruits'](pd.DataFrame([r._asdict()]),['Large','Medium','Small'])[0]
 assert torch.equal(raw,cached),('RGB input cache mismatch',r.source_id)
 pixel_checks.append({'source_id':r.source_id,'view':'RGB','exact':True})
 if str(r.verified_image_path) in vw['VIEW_INDEX_BY_PATH']:
  for view in vw['VIEW_ARRAYS']:
   vw['CURRENT_VIEW']=view
   rendered=vw['image_tensor'](r.verified_image_path)
   cached,_=vw['Fruits'](pd.DataFrame([r._asdict()]),['Large','Medium','Small'])[0]
   assert torch.equal(rendered,cached),('View input cache mismatch',r.source_id,view)
   pixel_checks.append({'source_id':r.source_id,'view':view,'exact':True})
save(OUT/'PREPROCESSING_PARITY.json',pixel_checks)
print('INPUTS_VERIFIED',len(image_records),'photos',len(pixel_checks),'exact tensor comparisons',flush=True)
def infer(model,frame,classes,ns):
 ds=ns['Fruits'](frame,classes);probs=[]
 model.eval()
 with torch.no_grad():
  for lo in range(0,len(ds),BATCH):
   x=torch.stack([ds[i][0] for i in range(lo,min(len(ds),lo+BATCH))]).to('cuda')
   probs.append(torch.softmax(model(x),dim=1).cpu().numpy())
 return np.concatenate(probs)


print('READY_DIAGNOSTIC',flush=True)

import os,sys,random
from tqdm import tqdm
family=sys.argv[1]
items=[r for r in json.loads((PACKAGE/'data/verification/C3_CHECKPOINT_INVENTORY.json').read_text())['checkpoints'] if r['group']==family]
for record in items:
 record['checkpoint']=str(ROOT/{'four_view':'foreground_background_probes_v1','cultivar_holdout':'independent_training_probes_resumable_v3','size_class_holdout':'class_novelty_probes_v1'}[family]/record['run']/'BEST_MODEL.pt')
report.update(original_model_factory_used=True,original_inference_function_used=True,group=family,batch_size=32 if family=='cultivar_holdout' else 128,cudnn_benchmark=family!='cultivar_holdout',cuda_matmul_tf32=family!='cultivar_holdout',cudnn_tf32=True)
for item in items:
 p=Path(item['checkpoint']);assert sha(p)==item['sha256'];meta=json.loads((p.parent/'RESULT.json').read_text());ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['fingerprint']==meta['fingerprint'];classes=ck['classes']
 name={'four_view':'visual_cues','cultivar_holdout':'cultivar_holdout','size_class_holdout':'size_holdout'}[family]
 ns={'four_view':vw,'cultivar_holdout':cv,'size_class_holdout':sz}[family]
 ns.update(os=os,tqdm=tqdm,INFERENCE_BATCH_SIZE=32 if family=='cultivar_holdout' else 128,MONITOR_EVERY_N_BATCHES=1,run_status=lambda *a,**k:None,utc_now=lambda:'fresh')
 definitions(name,['cached_inference'],ns)
 if family=='four_view':
  vw['CURRENT_VIEW']=meta['excluded_group'];known=DV[DV.split.eq('development')].copy();known['bucket']=known.source_group.map(vw['stable_bucket']);parts=(DV[DV.split.eq('train')],known[known.bucket.le(2)],known[known.bucket.eq(3)],known[known.bucket.ge(4)])
 elif family=='cultivar_holdout':parts=cv['make_fold'](D,meta['excluded_group'],'variety')
 else:parts=sz['make_fold'](D,meta['excluded_group'])[0]
 for f,k in zip(parts,['n_fit','n_selection','n_calibration','n_evaluation']):assert len(f)==meta[k]
 archived=pd.read_csv(p.parent/'HELDOUT_PREDICTIONS.csv',dtype={'source_id':str});assert archived.source_id.tolist()==parts[3].source_id.tolist()
 torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=family!='cultivar_holdout';torch.backends.cuda.matmul.allow_tf32=family!='cultivar_holdout'
 if family!='four_view':random.seed(int(ck['seed']));np.random.seed(int(ck['seed']));torch.manual_seed(int(ck['seed']));torch.cuda.manual_seed_all(int(ck['seed']))
 initial=models.resnet18(weights=None).state_dict();model=cv['new_model'](classes,initial);model.load_state_dict(ck['model'],strict=True);model.to('cuda').eval()
 dest=OUT/item['run']
 if family=='four_view':
  ep,ey=ns['cached_inference'](model,parts[3],classes,torch.device('cuda'),dest,'fresh_evaluation',sha(p));cp,cy=ns['cached_inference'](model,parts[2],classes,torch.device('cuda'),dest,'fresh_calibration',sha(p))
 else:
  cp,cy=ns['cached_inference'](model,parts[2],classes,torch.device('cuda'),dest,'fresh_calibration',sha(p));ep,ey=ns['cached_inference'](model,parts[3],classes,torch.device('cuda'),dest,'fresh_evaluation',sha(p))
 fresh=parts[3][['source_id','source_group','variety','native_size_label','split']].reset_index(drop=True).copy();fresh['predicted_label']=[classes[j] for j in ep.argmax(1)];fresh['max_probability']=ep.max(1);fresh['uncertainty']=1-ep.max(1);fresh['referred']=fresh.uncertainty>meta['calibration_threshold_90pct']
 for j,c in enumerate(classes):fresh['prob_'+c]=ep[:,j]
 fresh.to_csv(dest/'FRESH_HELDOUT_PREDICTIONS.csv',index=False)
 np.savez_compressed(dest/'FRESH_CALIBRATION.npz',probabilities=cp,source_ids=parts[2].source_id.to_numpy(dtype=str),classes=np.array(classes))
 oldcal=np.concatenate([np.load(f)['probabilities'] for f in sorted((p.parent/'inference_cache/final_calibration').glob('batch_*.npz'))]);oldev=np.concatenate([np.load(f)['probabilities'] for f in sorted((p.parent/'inference_cache/final_evaluation').glob('batch_*.npz'))])
 delta=np.abs(ep.astype(float)-archived[['prob_'+c for c in classes]].to_numpy(float));r={'run':item['run'],'group':family,'checkpoint_sha256':sha(p),'fingerprint':ck['fingerprint'],'classes':classes,'evaluation_rows':len(ep),'calibration_rows':len(cp),'frozen_threshold':meta['calibration_threshold_90pct'],'maximum_probability_absolute_error':float(delta.max()),'probabilities_outside_tolerance':int((delta>1e-5).sum()),'grade_mismatches':int((fresh.predicted_label!=archived.predicted_label).sum()),'referral_mismatches':int((fresh.referred.to_numpy()!=archived.referred.astype(str).str.lower().eq('true').to_numpy()).sum()),'calibration_max_error_vs_original_cache':float(np.abs(cp.astype(float)-oldcal.astype(float)).max()),'evaluation_bitwise_equal_original_cache':bool(np.array_equal(ep,oldev)),'calibration_bitwise_equal_original_cache':bool(np.array_equal(cp,oldcal))}
 r['pass']=all(r[k]==0 for k in ['probabilities_outside_tolerance','grade_mismatches','referral_mismatches']);report['runs'].append(r);save(dest/'PARITY.json',r);save(OUT/'C3_SOURCE_FUNCTION_REPLAY.json',report);print('SOURCE_REPLAY',json.dumps(r),flush=True)
 del model;torch.cuda.empty_cache()
report['status']='PASS' if all(r['pass'] for r in report['runs']) else 'MISMATCH';save(OUT/'C3_SOURCE_FUNCTION_REPLAY.json',report);print('FAMILY_COMPLETE',family,report['status'],len(items),flush=True)
