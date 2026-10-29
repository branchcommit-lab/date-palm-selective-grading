#!/usr/bin/env python3
"""Read-only calibrated candidate probability computation using authentic study methods.

Requires original study checkpoint, feature-cache and benchmark files. It does not train
or fit a classifier, change the frozen models, or infer anything from test labels.

Algorithm, selection rules, and transforms follow the archived
Dandhi model-selection protocol (notebook cells 4 and 10).

Writes only to --output. A parity failure creates a diagnostic report and an
explicit FAILED flag; results are never silently presented as verified.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = np.array(['Small','Medium','Large'])
SEEDS = (11,23,37)
CHECKPOINT_SHA = '0f2c1b75dfe9d7dfc3498b88a5bd065b97e8cf5a28262c32bd2d633a386bb909'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def require(value,message):
    if not bool(value):
        raise RuntimeError(message)


def load_json(p):
    require(p.is_file(),f'MISSING: {p}')
    return json.loads(p.read_text(encoding='utf-8'))


def text_csv(p):
    require(p.is_file(),f'MISSING: {p}')
    return pd.read_csv(p,dtype=str,keep_default_na=False,low_memory=False)


def row_digest(frame,columns):
    h=hashlib.sha256()
    for values in frame[list(columns)].itertuples(index=False,name=None):
        h.update(json.dumps([str(v) for v in values],separators=(',',':')).encode()+b'\n')
    return h.hexdigest()


def row_fingerprint(rows):
    keys=rows[['sample_id','image_sha256']].rename(columns={'sample_id':'id','image_sha256':'sha'})
    return row_digest(keys,['id','sha'])


def bool_saved(frame,col):
    require(col in frame,f'Missing archived decision column {col}')
    x=frame[col].str.strip().str.lower()
    require(x.isin(['true','false','1','0']).all(),f'Invalid boolean in {col}')
    return x.isin(['true','1']).to_numpy(dtype=bool)


def cached_rgb(folder,part,frame,ckpt_sha):
    cache=folder/'rgb_prediction_cache'/part
    require(cache.is_dir(),f'RGB cache missing: {cache}')
    mask=frame.source_variety.eq('Dandhi').to_numpy(dtype=bool)
    subset=frame.loc[mask].reset_index(drop=True)
    collected=[]
    for i in range(0,len(subset),128):
        stop=min(i+128,len(subset))
        name=f'block_{i:06d}_{stop:06d}.npz'
        path=cache/name
        require(path.is_file(),f'RGB cache block missing: {path}')
        with np.load(path,allow_pickle=False) as saved:
            require(set(saved.files)=={'ids','sha','ckpt_sha','logits'},f'Unexpected keys in {path}')
            require(str(saved['ckpt_sha'])==ckpt_sha,f'Checkpoint identity mismatch: {path}')
            require(np.array_equal(saved['ids'].astype(str),subset.sample_id.iloc[i:stop].to_numpy()),
                    f'RGB sample order mismatch: {path}')
            require(np.array_equal(saved['sha'].astype(str),subset.image_sha256.iloc[i:stop].to_numpy()),
                    f'RGB image hash mismatch: {path}')
            logits=saved['logits'].astype(np.float32)
            require(logits.shape==(stop-i,3) and np.isfinite(logits).all(),
                    f'Invalid cached RGB logits: {path}')
            collected.append(logits)
    result=np.concatenate(collected,axis=0)
    require(result.shape==(int(mask.sum()),3),f'Incorrect RGB logits count for {part}')
    return result,mask


def empirical_rank(clean_cal_risk,all_risk):
    baseline=np.sort(np.asarray(clean_cal_risk,dtype=np.float64))
    scores=np.asarray(all_risk,dtype=np.float64)
    require(len(baseline)==556 and np.isfinite(baseline).all() and np.isfinite(scores).all(),
            'Empirical calibration risk invalid')
    return np.searchsorted(baseline,scores,side='right')/len(baseline)


def calibrated_referral(cal_risk,test_risk,cal_clean,budget):
    reference=np.asarray(cal_risk,dtype=float)[cal_clean]
    ranks_cal=empirical_rank(reference,cal_risk)
    ranks_test=empirical_rank(reference,test_risk)
    threshold=np.quantile(ranks_cal[cal_clean],1-budget,method='higher')
    return ranks_test>threshold


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v4',type=str,required=True,help='Study benchmark and C11 score root')
    ap.add_argument('--stage1',type=str,required=True,help='Frozen grading and RGB feature-cache root')
    ap.add_argument('--output',type=str,required=True,help='Separate output directory')
    ap.add_argument('--batch',type=int,default=96)
    args=ap.parse_args()
    v4=Path(args.v4).resolve();stage=Path(args.stage1).resolve();out=Path(args.output).resolve()
    spatial=stage/'v51_locked_spatial';pilot=stage/'v51_grader_decision_pilot';dandhi=stage/'v51_dandhi_targeted_rgb_pilot'
    require(out!=v4 and out!=stage and not any(parent in out.parents for parent in (v4,stage,spatial,pilot,dandhi)),
            'Output must be outside all recorded experiment folders')
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'IN_PROGRESS','origin':'Authentic frozen-feature / checkpoint candidate inference',
            'v4':str(v4),'stage1':str(stage),'checks':{},'input_sha256':{},'failures':[]}
    try:
        import torch
        from torch import nn
        from torchvision import models
        from scipy.special import softmax
        require(args.batch>0,'Batch must be positive')
        torch.set_num_threads(min(8,max(1,os.cpu_count() or 1)))
        selection=load_json(dandhi/'development_only_selection.json')
        grader_choice=load_json(pilot/'development_model_selection.json')
        alpha=float(selection['alpha']);T=float(grader_choice['temperature']);winner=str(grader_choice['winner'])
        require(alpha==0.5,'Recorded selected Dandhi alpha differs from 0.5; stop and investigate')
        require(winner=='equal_weight_ensemble' and T==1.4756966894758048,
                f'Recorded grader winner/temperature unexpected: {winner}, {T}')
        require(selection['selected_checkpoint_sha256']==CHECKPOINT_SHA,'Dandhi selection checkpoint hash differs')
        report.update(alpha=alpha,temperature=T,selected_grader=winner)
        test=text_csv(v4/'results/C11_1049.csv')
        cal=text_csv(v4/'calibration/scores/C11_1049.csv')
        benchmark_test=text_csv(v4/'results/fixed_benchmark/manifest.csv')
        benchmark_cal=text_csv(v4/'calibration/benchmark/manifest.csv')
        existing=text_csv(pilot/'v51_improved_grader_all_variant_predictions.csv')
        archival=text_csv(dandhi/'targeted_variant_predictions.csv')
        require(len(cal)==len(benchmark_cal)==1668 and len(test)==len(benchmark_test)==len(existing)==len(archival)==12510,
                'Original calibration/locked-test row counts changed')
        require(all(x.sample_id.is_unique for x in (cal,test,existing,archival)),
                'Duplicate recorded IDs')
        for part,frame,benchmark in (('calibration',cal,benchmark_cal),('locked_test',test,benchmark_test)):
            require(set(frame.sample_id)==set(benchmark.sample_id),f'{part} benchmark IDs mismatch')
            bench=benchmark.set_index('sample_id').loc[frame.sample_id]
            for col in ('source_id','source_group','source_variety','kind','label','image_sha256'):
                require(np.array_equal(frame[col].to_numpy(),bench[col].to_numpy()),
                        f'{part} benchmark {col} changed')
            require(frame.groupby('source_id').size().eq(3 if part=='calibration' else 15).all(),
                    f'{part} source sampling mismatch')
        existing=existing.set_index('sample_id').loc[test.sample_id].reset_index()
        archival=archival.set_index('sample_id').loc[test.sample_id].reset_index()
        for c in ('source_id','source_group','kind','label','source_size','source_variety'):
            require(np.array_equal(test[c].to_numpy(),existing[c].to_numpy()),f'Prior predictions mismatch {c}')
            require(np.array_equal(test[c].to_numpy(),archival[c].to_numpy()),f'Archived candidate mismatch {c}')
        ckpt=stage/'v51_representation_finetune_pilot/rgb_representation_finetuned.pt'
        require(ckpt.is_file(),f'Original RGB checkpoint missing: {ckpt}')
        report['input_sha256'][str(ckpt)]=digest(ckpt)
        require(report['input_sha256'][str(ckpt)]==CHECKPOINT_SHA,'RGB checkpoint file hash mismatch')
        rgb_cal,mask_cal=cached_rgb(dandhi,'calibration',cal,CHECKPOINT_SHA)
        rgb_test,mask_test=cached_rgb(dandhi,'locked_test',test,CHECKPOINT_SHA)
        report['checks']['cached_rgb_calibration_rows']=len(rgb_cal)
        report['checks']['cached_rgb_test_rows']=len(rgb_test)

        class SizeHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer4=models.resnet18(weights=None).layer4
                self.fc=nn.Linear(512,3)
            def forward(self,x):
                return self.fc(self.layer4(x).mean((-2,-1)))

        models_by_seed={}
        for seed in SEEDS:
            path=spatial/f'size_layer4_seed{seed}.pt'
            require(path.is_file(),f'Original size head missing: {path}')
            checksum=digest(path)
            report['input_sha256'][str(path)]=checksum
            payload=torch.load(path,map_location='cpu',weights_only=True)
            require(int(payload['config']['seed'])==seed,f'Incorrect seed checkpoint {path}')
            net=SizeHead()
            net.load_state_dict(payload['model_state'],strict=True)
            net.eval()
            models_by_seed[seed]=net

        @torch.inference_mode()
        def head_logits(net,features):
            output=[]
            for i in range(0,len(features),args.batch):
                batch=torch.from_numpy(np.asarray(features[i:i+args.batch],dtype=np.float32).copy())
                output.append(net(batch).detach().cpu().numpy())
            return np.concatenate(output,axis=0)

        def frozen_features(part,frame):
            data_path=spatial/f'fixed_layer3_{part}.npy'
            meta_path=spatial/f'fixed_layer3_{part}.json'
            require(data_path.is_file() and meta_path.is_file(),f'Frozen features missing for {part}')
            meta=load_json(meta_path)
            require(meta['row_sha256']==row_fingerprint(frame),f'Frozen feature row/hash identity differs for {part}')
            array=np.load(data_path,mmap_mode='r',allow_pickle=False)
            require(array.shape==(len(frame),256,14,14) and array.dtype==np.float16,
                    f'Unexpected frozen features shape/dtype for {part}')
            return array

        def replay_baseline(part,frame):
            features=frozen_features(part,frame)
            raw=np.stack([softmax(head_logits(models_by_seed[seed],features),axis=1) for seed in SEEDS])
            # Exact recorded two-stage operation: mean the seed probabilities,
            # then apply temperature to the logarithm of their ensemble mean.
            mean=raw.mean(axis=0)
            return softmax(np.log(np.clip(mean,1e-12,1))/T,axis=1)

        p_base_cal=replay_baseline('calibration',cal)
        p_base_test=replay_baseline('locked_test',test)
        err=float(np.max(np.abs(p_base_test.max(axis=1)-pd.to_numeric(existing.max_probability).to_numpy())))
        n_pred_baseline=int((LABELS[p_base_test.argmax(1)]!=existing.predicted_size.to_numpy()).sum())
        report['checks'].update(baseline_max_probability_error=err,baseline_grade_mismatches=n_pred_baseline)
        require(err<0.003 and n_pred_baseline==0,
                f'Baseline model parity FAIL: max_conf_delta={err}, grade_mismatch={n_pred_baseline}')
        p_candidate_cal=p_base_cal.copy()
        p_candidate_test=p_base_test.copy()
        p_candidate_cal[mask_cal]=(1-alpha)*p_base_cal[mask_cal]+alpha*softmax(rgb_cal/T,axis=1)
        p_candidate_test[mask_test]=(1-alpha)*p_base_test[mask_test]+alpha*softmax(rgb_test/T,axis=1)
        require(np.array_equal(p_candidate_test[~mask_test],p_base_test[~mask_test]),
                'Non-Dandhi locked probabilities changed')
        require(np.array_equal(p_candidate_cal[~mask_cal],p_base_cal[~mask_cal]),
                'Non-Dandhi calibration probabilities changed')
        for x in (p_candidate_cal,p_candidate_test):
            require(np.isfinite(x).all() and np.allclose(x.sum(1),1,atol=1e-5),'Candidate probabilities invalid')
        n_candidate=int((LABELS[p_candidate_test.argmax(1)]!=archival.candidate_grade.to_numpy()).sum())
        report['checks']['candidate_grade_mismatches']=n_candidate
        clean_cal=cal.kind.eq('original').to_numpy()
        frozen=load_json(dandhi/'frozen_referral_cutoffs.json')
        require(frozen.get('authenticated') is True,'Frozen referral cutoffs must carry authenticated provenance')
        computed={}
        for budget in (0.05,0.10):
            pct=int(round(100*budget))
            require(frozen.get('calibration_probability_sha256')=='10f5e52486cfe4d2ec7f2037beb5b622af08006f9fe29d4f7aff931aa5586b95','Cutoff calibration identity differs')
            cutoff=float(frozen['empirical_rank_cutoffs'][str(pct)])
            require(0<=cutoff<=1,'Invalid frozen rank cutoff')
            ranks=empirical_rank((1-p_candidate_cal.max(1))[clean_cal],1-p_candidate_test.max(1))
            computed[pct]=ranks>cutoff
        c11_5=bool_saved(existing,'refer_C11_5')
        comparisons={
            'refer_candidate_conf_10':computed[10],
            'refer_candidate_dual_5_5':computed[5]|c11_5,
        }
        for col,expected in comparisons.items():
            observed=bool_saved(archival,col)
            report['checks'][f'{col}_mismatches']=int((observed!=expected).sum())
        report['checks']['n_locked_samples']=len(test)
        report['checks']['n_locked_originals']=int(test.kind.eq('original').sum())
        report['checks']['n_candidate_Dandhi']=int(mask_test.sum())
        report['checks']['n_calibration_Dandhi']=int(mask_cal.sum())
        parity_ok=(n_candidate==0 and all(report['checks'][f'{col}_mismatches']==0 for col in comparisons))

        def save_rows(part,frame,base,candidate,mask):
            out_frame=frame[['sample_id','source_id','source_group','kind','label','source_size','source_variety']].copy()
            out_frame['targeted_Dandhi']=mask
            for name,p in (('baseline',base),('candidate',candidate)):
                for i,label in enumerate(LABELS):
                    out_frame[f'{name}_p_{label}']=p[:,i]
                out_frame[f'{name}_max_probability']=p.max(1)
                out_frame[f'{name}_predicted_size']=LABELS[p.argmax(1)]
                out_frame[f'{name}_confidence_risk']=1-p.max(1)
            if part=='locked_test':
                out_frame['candidate_refer_confidence_5']=computed[5]
                out_frame['candidate_refer_confidence_10']=computed[10]
                out_frame['candidate_refer_dual_5_5']=computed[5]|c11_5
            path=out/f'{part}_candidate_probabilities.csv'
            out_frame.to_csv(path,index=False,float_format='%.10g')
            return path
        out_paths=[save_rows('calibration',cal,p_base_cal,p_candidate_cal,mask_cal),
                   save_rows('locked_test',test,p_base_test,p_candidate_test,mask_test)]
        report['outputs']={str(p):{'sha256':digest(p),'rows':int(len(cal) if 'calibration' in p.name else len(test))}
                           for p in out_paths}
        report['status']='PASS_AUTHENTIC_SAVED_FEATURE_AND_RGB_LOGIT_INFERENCE_REPLAY' if parity_ok else 'FAILED_PARITY'
        if not parity_ok:report['failures'].append('Candidate grades or decisions differ from recorded saved records')
    except Exception:
        report['status']='FAILED_EXECUTION'
        report['failures'].append(traceback.format_exc())
    report_path=out/'00_REPLAY_REPORT.json'
    report_path.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('RESULT:',report['status'])
    print('REPORT:',report_path)
    print('CHECKS:',json.dumps(report['checks'],indent=2))
    if report['status']!='PASS_AUTHENTIC_SAVED_FEATURE_AND_RGB_LOGIT_INFERENCE_REPLAY':
        print('ERROR DETAILS:',*report['failures'],sep='\n',file=sys.stderr)
        raise SystemExit(2)
    print('Recovered candidate probability vectors and historic decisions match.')
    print('No training or raw-image inference performed; original saved layer3 features/RGB logits used.')

if __name__=='__main__':
    main()
