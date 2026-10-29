
# Analysis section 1
from pathlib import Path
import os, json, hashlib, sys, time, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from IPython.display import display

ROOT=Path(os.environ['DATEPALM_PACKAGE_ROOT']).resolve()
DATA=ROOT/'data'
METHODS=ROOT/'src'/'methods'
OUT=Path(os.environ.get('DATEPALM_OUTPUT',ROOT/'outputs')).resolve()
MODE=os.getenv('DATEPALM_ANALYSIS_MODE','results')
INFERENCE_STATUS=json.loads(Path(os.environ['DATEPALM_INFERENCE_STATUS']).read_text()) if os.environ.get('DATEPALM_INFERENCE_STATUS') else None
assert MODE in ('results','inference','retrain'),MODE
for p in ('tables','figures','reports'):
    (OUT/p).mkdir(parents=True,exist_ok=True)
assert DATA.is_dir() and METHODS.is_dir(), 'Open the notebook in the extracted study directory'
REPORT={'mode':MODE,'status':'IN_PROGRESS','generated':{},'checks':{},'limitations':[]}
if MODE=='inference' and INFERENCE_STATUS:
    REPORT['inference']={'action':os.getenv('DATEPALM_ANALYSIS_ACTION','audit'),'status':INFERENCE_STATUS['status'],'full_raw_image_to_C1_C3_certified':False}
def check(test,message):
    REPORT['checks'][message]='PASS' if bool(test) else 'FAIL'
    if not test:raise AssertionError(message)
def hashfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(rel,**k):
    p=DATA/rel
    check(p.is_file(),'Input exists: '+rel)
    return pd.read_csv(p,**k)
def emit(df,name):
    p=OUT/'tables'/name
    df.to_csv(p,index=False,float_format='%.16g')
    REPORT['generated']['tables/'+name]=hashfile(p)
    return p
def figure(name):
    p=OUT/'figures'/name
    plt.savefig(p,dpi=180,bbox_inches='tight')
    plt.close()
    REPORT['generated']['figures/'+name]=hashfile(p)
def finish(status):
    REPORT['status']=status
    path=OUT/'reports'/'EXPERIMENT_STATUS.json'
    path.write_text(json.dumps(REPORT,indent=2,default=str))
    return path
print('Study mode:',MODE,'| Study directory:',ROOT)


# Analysis section 3
source=read('metadata/source_split.csv',dtype={'source_id':'string','source_group':'string','file_sha256':'string'})
eligible=source.loc[source['split'].notna()].copy()
splits={'train':3603,'development':835,'calibration':556,'locked_test':834}
check(len(source)==5836 and len(eligible)==5828,'Study population and exclusions')
check(eligible.split.value_counts().to_dict()==splits,'Exact partition sizes')
check(eligible.source_id.is_unique,'Unique source identifiers')
check(eligible.file_sha256.str.fullmatch('[0-9a-f]{64}').all(),'Source file hash format')
for i,a in enumerate(splits):
    for b in list(splits)[i+1:]:
        a_set=set(eligible.loc[eligible.split.eq(a),'source_group'].astype(str))
        b_set=set(eligible.loc[eligible.split.eq(b),'source_group'].astype(str))
        check(not a_set.intersection(b_set),f'Source group separation {a} / {b}')
display(eligible.groupby(['split','native_size_label']).size().rename('n').reset_index())


# Analysis section 5
method_manifest=json.loads((METHODS/'manifest.json').read_text())
for name,meta in method_manifest.items():
    p=METHODS/meta['file']
    check(p.is_file() and hashfile(p)==meta['sha256'],'Method identity: '+name)
print('Authenticated method components:',', '.join(method_manifest))


# Analysis section 9
pred=read('raw_records/final_candidate_predictions.csv')
orig=read('raw_records/original_ensemble_predictions.csv')
# In full component execution, the locked-test probabilities and grades below
# come from the newly executed authentic candidate checkpoint/feature replay.
# In audit or results mode, the supplied authenticated observation records are used.
if MODE=='inference' and INFERENCE_STATUS and INFERENCE_STATUS.get('components',{}).get('candidate',{}).get('status')=='PASS':
    executed=pd.read_csv(Path(os.environ.get('DATEPALM_FRESH_INFERENCE_OUTPUT',str(OUT)))/'candidate'/'locked_test_candidate_probabilities.csv')
    check(len(executed)==len(pred) and executed.sample_id.is_unique,
          'Executed candidate observation count and uniqueness')
    aligned=executed.set_index('sample_id').loc[pred.sample_id]
    updated=pred.copy()
    updated['candidate_grade']=aligned.candidate_predicted_size.to_numpy()
    updated['candidate_error']=updated.candidate_grade.ne(updated.source_size).to_numpy()
    updated['refer_candidate_conf_10']=aligned.candidate_refer_confidence_10.to_numpy(dtype=bool)
    updated['refer_candidate_dual_5_5']=aligned.candidate_refer_dual_5_5.to_numpy(dtype=bool)
    check(updated.candidate_grade.eq(pred.candidate_grade).all(),
          'Executed candidate grade parity')
    check(updated.refer_candidate_conf_10.eq(pred.refer_candidate_conf_10).all(),
          'Executed candidate confidence referral parity')
    check(updated.refer_candidate_dual_5_5.eq(pred.refer_candidate_dual_5_5).all(),
          'Executed candidate combined referral parity')
    pred=updated
    print('C1/C2 use newly computed candidate grades and referral decisions.')

check(len(pred)==12510 and pred.sample_id.is_unique,'Evaluation observation identities')
check(pred.source_id.nunique()==834,'Evaluation source size')
check(pred.groupby('source_id').size().eq(15).all(),'15 observations per source')
check(set(pred.sample_id)==set(orig.sample_id),'Baseline and candidate alignment')
check(set(pred.loc[pred.kind.eq('original'),'source_id'].astype(str))==
      set(eligible.loc[eligible.split.eq('locked_test'),'source_id'].astype(str)),
      'Grading evaluation aligns to frozen source partition')
POLICIES={'confidence_10':'refer_candidate_conf_10',
          'appearance_10':'refer_C11_10',
          'combined_5_plus_5':'refer_candidate_dual_5_5'}
mask_orig=pred.kind.eq('original').to_numpy()
mask_ben=pred.kind.astype(str).str.startswith('benign').to_numpy()
mask_alt=~(mask_orig|mask_ben)
check((mask_orig.sum(),mask_ben.sum(),mask_alt.sum())==(834,1668,10008),
      'Original, benign and altered population sizes')
rows=[]
for name,col in POLICIES.items():
    flags=pred[col].astype(bool).to_numpy()
    accepted=mask_orig&~flags
    wrong=accepted&pred.candidate_error.astype(bool).to_numpy()
    rows.append(dict(policy=name,original_n=int(mask_orig.sum()),
       original_ref_n=int((flags&mask_orig).sum()),
       original_ref_fraction=float(flags[mask_orig].mean()),
       benign_n=int(mask_ben.sum()),
       benign_ref_n=int((flags&mask_ben).sum()),
       benign_ref_fraction=float(flags[mask_ben].mean()),
       altered_n=int(mask_alt.sum()),
       altered_ref_n=int((flags&mask_alt).sum()),
       altered_ref_fraction=float(flags[mask_alt].mean()),
       accepted_originals=int(accepted.sum()),
       wrong_accepted_originals=int(wrong.sum()),
       accepted_error_fraction=float(wrong.sum()/accepted.sum())))
c1=pd.DataFrame(rows)
check(c1.set_index('policy').loc['confidence_10','altered_ref_n']==1103,'Confidence referral count')
check(c1.set_index('policy').loc['appearance_10','altered_ref_n']==4471,'Appearance referral count')
check(c1.set_index('policy').loc['combined_5_plus_5','altered_ref_n']==4263,'Combined referral count')
emit(c1,'C1_operating_policy_comparison.csv')
display(c1)


# Analysis section 10
group_ids=np.sort(pred.source_id.astype(str).unique())
codes=pd.Categorical(pred.source_id.astype(str),categories=group_ids).codes
N=len(group_ids)
a=pred[POLICIES['combined_5_plus_5']].astype(bool).to_numpy()
b=pred[POLICIES['confidence_10']].astype(bool).to_numpy()
error=pred.candidate_error.astype(bool).to_numpy()
def count_by_group(x):
    return np.bincount(codes,weights=x.astype(float),minlength=N)
components={
 'altered':(count_by_group(a&mask_alt),count_by_group(b&mask_alt)),
 'original':(count_by_group(a&mask_orig),count_by_group(b&mask_orig)),
 'benign':(count_by_group(a&mask_ben),count_by_group(b&mask_ben)),
 'wrong':(count_by_group(~a&mask_orig&error),count_by_group(~b&mask_orig&error)),
 'accepted':(count_by_group(~a&mask_orig),count_by_group(~b&mask_orig))}
rng=np.random.default_rng(20261025)
bootstrap=[]
for j in range(2000):
    take=rng.integers(0,N,size=N)
    aa=lambda key:components[key][0][take].sum()
    bb=lambda key:components[key][1][take].sum()
    bootstrap.append({
     'altered_referral':(aa('altered')-bb('altered'))/(12*N),
     'original_referral':(aa('original')-bb('original'))/N,
     'benign_referral':(aa('benign')-bb('benign'))/(2*N),
     'accepted_error':aa('wrong')/aa('accepted')-bb('wrong')/bb('accepted')})
boot=pd.DataFrame(bootstrap)
point={'altered_referral':(components['altered'][0].sum()-components['altered'][1].sum())/(12*N),
       'original_referral':(components['original'][0].sum()-components['original'][1].sum())/N,
       'benign_referral':(components['benign'][0].sum()-components['benign'][1].sum())/(2*N),
       'accepted_error':components['wrong'][0].sum()/components['accepted'][0].sum()-
                        components['wrong'][1].sum()/components['accepted'][1].sum()}
intervals=pd.DataFrame([dict(endpoint=k,combined_minus_confidence=point[k],
                       ci95_low=boot[k].quantile(.025),ci95_high=boot[k].quantile(.975),
                       replicates=2000,seed=20261025) for k in point])
emit(intervals,'C1_paired_source_bootstrap.csv')
display(intervals)


# Analysis section 12
clean=pred.loc[mask_orig].copy()
old_good=clean.baseline_grade.eq(clean.source_size)
new_good=clean.candidate_grade.eq(clean.source_size)
check((int(old_good.sum()),int(new_good.sum()))==(703,707),'Paired clean-image totals')
c2_rows=[]
for cultivar,sub in [('Overall',clean)]+list(clean.groupby('source_variety',sort=True)):
    old=sub.baseline_grade.eq(sub.source_size)
    new=sub.candidate_grade.eq(sub.source_size)
    c2_rows.append(dict(variety=cultivar,n=len(sub),
     baseline_correct=int(old.sum()),candidate_correct=int(new.sum()),
     baseline_accuracy=float(old.mean()),candidate_accuracy=float(new.mean()),
     corrected=int((~old&new).sum()),regressed=int((old&~new).sum()),
     net_gain=int(new.sum()-old.sum())))
c2=pd.DataFrame(c2_rows)
corrected=int((~old_good&new_good).sum())
regressed=int((old_good&~new_good).sum())
pvalue=float(binomtest(min(corrected,regressed),corrected+regressed,0.5,alternative='two-sided').pvalue)
check((corrected,regressed)==(12,8),'Paired improvement and regression counts')
check(abs(pvalue-0.5034446716308594)<1e-12,'Exact paired McNemar test')
emit(c2,'C2_cultivar_grading.csv')
display(c2)
print('McNemar exact p:',pvalue)


# Analysis section 14
stress=read('raw_records/stress_predictions.csv')
check(stress.source_id.nunique()==835,'Photographic stress source count')
check(stress.condition.nunique()==7,'Seven stress conditions')
check(stress.groupby(['source_id','model']).size().eq(7).all(),
      'Complete paired stress design')
stress_group=(stress.groupby(['variety','condition','model'],as_index=False)
             .agg(n=('correct','size'),accuracy=('correct','mean')))
emit(stress_group,'C3_stress_by_cultivar.csv')
wide=stress_group.pivot_table(index=['variety','condition'],columns='model',values='accuracy').reset_index()
wide['candidate_minus_baseline']=wide['Dandhi_targeted']-wide['baseline']
emit(wide,'C3_stress_paired_differences.csv')
sys.path.insert(0,str(METHODS))
from c3_observation_analysis import compute as reconstruct_c3
c3_parity,runs=reconstruct_c3(DATA,OUT/'tables')
check(c3_parity['runs']==33,'All 33 supplementary evaluations represented')
check(c3_parity['individual_predictions']==12954,'All 12,954 individual C3 predictions represented')
check(c3_parity['numeric_checks']==168 and c3_parity['failed_checks']==0,
      'All C3 run-level numerical estimates reproduced from predictions')
emit(runs,'C3_all_run_level_results.csv')
REPORT['generated']['tables/C3_recomputed_from_individual_predictions.csv']=hashfile(OUT/'tables'/'C3_recomputed_from_individual_predictions.csv')
REPORT['generated']['tables/C3_OBSERVATION_PARITY.json']=hashfile(OUT/'tables'/'C3_OBSERVATION_PARITY.json')
display(runs.groupby('analysis_family').agg(runs=('run','size'),conditions=('condition','nunique')))
print('C3 individual predictions:',c3_parity['individual_predictions'],
      '| numeric metric checks:',c3_parity['numeric_checks'])


# Analysis section 16
plt.figure(figsize=(9,5))
x=np.arange(3);width=.25
for j,(col,label) in enumerate([('original_ref_fraction','Original'),
                                  ('benign_ref_fraction','Benign'),
                                  ('altered_ref_fraction','Altered')]):
    plt.bar(x+(j-1)*width,c1[col],width,label=label)
plt.xticks(x,c1.policy,rotation=13)
plt.ylabel('Fraction referred');plt.title('C1: Referral under fixed inspection policies')
plt.legend();plt.tight_layout();figure('C1_referral_tradeoff.png')
plt.figure(figsize=(9,5))
sub=c2.loc[c2.variety.ne('Overall')].sort_values('variety')
x=np.arange(len(sub))
plt.bar(x-.2,sub.baseline_accuracy,.4,label='Baseline')
plt.bar(x+.2,sub.candidate_accuracy,.4,label='Cultivar-conditional')
plt.xticks(x,sub.variety);plt.ylabel('Original-image grading accuracy')
plt.ylim(0,1);plt.legend();plt.tight_layout();figure('C2_cultivar_accuracy.png')
pv=wide.pivot(index='variety',columns='condition',values='candidate_minus_baseline')
plt.figure(figsize=(10,4))
plt.imshow(pv.to_numpy(),aspect='auto',cmap='coolwarm',vmin=-.20,vmax=.20)
plt.colorbar(label='Candidate minus baseline accuracy')
plt.xticks(np.arange(len(pv.columns)),pv.columns,rotation=40,ha='right')
plt.yticks(np.arange(len(pv.index)),pv.index)
plt.title('C3: Grading change under imaging stress')
plt.tight_layout();figure('C3_stress_delta.png')
print('Figures regenerated from observation-level records.')


# Analysis section 18
calprob=read('tables/candidate_probabilities_calibration.csv')
testprob=read('tables/candidate_probabilities_locked_test.csv')
check(len(calprob)==1668 and len(testprob)==12510,'Candidate probability populations')
check(testprob.sample_id.is_unique and calprob.sample_id.is_unique,'Unique probability identifiers')
check(set(testprob.sample_id)==set(pred.sample_id),'Candidate probability alignment')
for label,frame in [('calibration',calprob),('evaluation',testprob)]:
    m=frame[['prob_Small','prob_Medium','prob_Large']].to_numpy(dtype=float)
    check(np.isfinite(m).all() and (m>=-1e-7).all(),'Valid probability values: '+label)
    check(np.allclose(m.sum(axis=1),1,rtol=0,atol=3e-6),'Probability normalization: '+label)
pmerge=pred[['sample_id','candidate_grade','refer_candidate_conf_10']].merge(
    testprob[['sample_id','predicted_size','refer_candidate_conf_10','max_probability']],
    on='sample_id',validate='one_to_one',suffixes=('_record','_prob'))
check(pmerge.candidate_grade.eq(pmerge.predicted_size).all(),'Grade parity from continuous probabilities')
check(pmerge.refer_candidate_conf_10_record.eq(pmerge.refer_candidate_conf_10_prob).all(),
      'Confidence-referral parity')
sub=pmerge.merge(pred[['sample_id','kind','candidate_error']],on='sample_id',validate='one_to_one')
sub=sub[sub.kind.eq('original')].copy()
curve=[]
for q in np.linspace(0,0.50,51):
    # Analysis only: threshold based on evaluation ranking is descriptive,
    # not a new deployment calibration or independently selected policy.
    k=int(np.floor(q*len(sub)))
    order=np.argsort(sub.max_probability.to_numpy(),kind='stable')
    flags=np.zeros(len(sub),dtype=bool)
    flags[order[:k]]=True
    accepted=~flags
    curve.append(dict(descriptive_fraction_referred=k/len(sub),
       accepted_n=int(accepted.sum()),
       wrong_accepted=int(sub.candidate_error.astype(bool).to_numpy()[accepted].sum()),
       accepted_error_rate=float(sub.candidate_error.astype(bool).to_numpy()[accepted].mean())))
curve=pd.DataFrame(curve)
emit(curve,'C1_descriptive_candidate_confidence_risk_coverage.csv')
display(curve.iloc[::10])


# Analysis section 20
# Inspection of the source identity registry is independent of execution mode.
print('Authentic study components packaged:',len(method_manifest))
if MODE=='inference':
    print('Inference action:',REPORT['inference']['action'])
    print('Component scope:',REPORT['inference']['status'])
    print('C3 metrics are generated from 33 observation-level records; '
          'the 33 separate checkpoint inferences are not re-executed here.')
else:
    print('No new model inference executed. All results regenerate from authenticated observations.')

# Analysis section 22
import zipfile
if MODE in ('results','inference'):
    REPORT['limitations']+= [
      'Within-collection source-group-disjoint evaluation is not independent external validation.',
      'Synthetic alterations do not directly establish performance on naturally occurring defects.',
      'The C2 overall improvement has a nonsignificant paired McNemar test.',
      'All 33 C3 run-level metrics are reconstructed from individual predictions; their checkpoint inference is not rerun.',
      'The combined policy increases synthetic referral but may increase error among automatically accepted originals.',
      'Results-mode execution establishes analytical regeneration, not complete training replay.'
    ]
status=('PASS_RESULTS_MODE' if MODE=='results' else
        ('PASS_INFERENCE_EVIDENCE_AUDIT_PLUS_RESULTS' if REPORT.get('inference',{}).get('action')=='audit' else
         'PASS_COMPONENT_INFERENCE_PLUS_RECORDED_C3_RESULTS'))
status_path=finish(status)
archive=OUT/'DatePalm_Study_Results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ('tables','figures','reports'):
        for p in sorted((OUT/folder).rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
with zipfile.ZipFile(archive) as z:
    check(z.testzip() is None,'Publication output ZIP integrity')
finish(status)
print('Mode:',MODE,'| checks:',len(REPORT['checks']),'| outputs:',len(REPORT['generated']))
print('Reports:',status_path)
print('Publication output:',archive)
