#!/usr/bin/env python3
"""Regenerate all seven manuscript illustrations and seven principal tables.

Inputs are the frozen observation-level experimental records in data/raw_records,
not photographs, newly inferred model outputs, or re-trained checkpoints.
Run: python publication/rebuild_manuscript_assets.py [--output DIR]
All figures are generated afresh by this script; visual styling may differ from
historical manuscript figure binaries. No external/network resources required.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'publication'/'generated'
POLICIES=[('Confidence-only','refer_candidate_conf_10','10/0'),('Combined','refer_candidate_dual_5_5','5/5'),('Appearance-only','refer_C11_10','0/10')]
BLUE='#426B86';TEAL='#4C8580';PLUM='#806C86';GREY='#536473';LIGHT='#EFF3F6'
COLORS=[BLUE,TEAL,PLUM]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})

def savefig(fig,folder,name):
    fig.savefig(folder/name,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def pct(x):return 100*np.asarray(x,dtype=float)
def delong_se(known,unknown):
    # Empirical two-sample DeLong / placement-value variance of Mann–Whitney AUROC
    hit=(unknown[:,None]>known[None,:]).astype(float)+.5*(unknown[:,None]==known[None,:])
    return float(np.sqrt(np.var(hit.mean(axis=1),ddof=1)/len(unknown)+
                         np.var(hit.mean(axis=0),ddof=1)/len(known)))

def main(output):
  output=Path(output); figs=output/'figures';tables=output/'tables';figs.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
  pred=pd.read_csv(ROOT/'data/raw_records/final_candidate_predictions.csv',dtype={'source_id':str})
  original=pred.kind.eq('original').to_numpy(); benign=pred.kind.str.startswith('benign').to_numpy(); altered=~(original|benign)
  err=pred.candidate_error.to_numpy(dtype=bool)
  ids=np.sort(pred.source_id.unique());N=len(ids)
  codes=pd.Categorical(pred.source_id,categories=ids).codes
  def group(x):return np.bincount(codes,weights=np.asarray(x,dtype=float),minlength=N)
  rng=np.random.default_rng(20261025); draws=rng.integers(0,N,size=(2000,N))
  comps={}; desc=[]; arm_cis={}
  for title,col,budget in POLICIES:
    r=pred[col].to_numpy(dtype=bool); accepted=original&~r
    c={z:group(r&m) for z,m in [('original',original),('benign',benign),('altered',altered)]}
    c['wrong']=group(accepted&err);c['accepted']=group(accepted);comps[title]=c
    nref={z:int((r&m).sum()) for z,m in [('original',original),('benign',benign),('altered',altered)]}
    nref['full']=int(r.sum())
    wrongacc=int((accepted&err).sum());wrref=int((original&r&err).sum());orref=nref['original']
    desc.append(dict(policy=title,nominal_budget=budget,original_referred=orref,original_referral_pct=100*orref/N,
      benign_referred=nref['benign'],benign_referral_pct=100*nref['benign']/int(benign.sum()),
      altered_referred=nref['altered'],altered_referral_pct=100*nref['altered']/int(altered.sum()),
      full_referred=nref['full'],full_referral_pct=100*nref['full']/len(pred),
      accepted_originals=int(accepted.sum()),wrong_accepted=wrongacc,
      accepted_original_error_pct=100*wrongacc/accepted.sum(),
      error_capture=wrref,queue_precision_pct=100*wrref/orref))
    arm_cis[title]={}
    for key,den in [('original',N),('benign',int(benign.sum())),('altered',int(altered.sum()))]:
      rates=c[key][draws].sum(axis=1)/den
      arm_cis[title][key]=100*np.quantile(rates,[.025,.975])
    rates=c['wrong'][draws].sum(axis=1)/c['accepted'][draws].sum(axis=1)
    arm_cis[title]['cer']=100*np.quantile(rates,[.025,.975])
  c1=pd.DataFrame(desc)
  assert len(pred)==12510 and len(ids)==834 and sum(benign)==1668 and sum(altered)==10008
  assert c1.altered_referred.tolist()==[1103,4263,4471]
  assert c1.error_capture.tolist()==[36,23,15]
  assert c1.wrong_accepted.tolist()==[91,104,112]
  c1.to_csv(tables/'Table_3_C1_operating_policy_outcomes.csv',index=False)
  # Table 2: all three paired bootstrap contrasts; source resampling honors 15 views/source
  contrasts=[]
  for a,b in [('Combined','Confidence-only'),('Appearance-only','Confidence-only'),('Appearance-only','Combined')]:
    ca,cb=comps[a],comps[b]
    for field,den in [('accepted_error',None),('altered',int(altered.sum())),('benign',int(benign.sum())),('original',N)]:
      if field=='accepted_error':
        p=ca['wrong'].sum()/ca['accepted'].sum()-cb['wrong'].sum()/cb['accepted'].sum()
        v=ca['wrong'][draws].sum(axis=1)/ca['accepted'][draws].sum(axis=1)-cb['wrong'][draws].sum(axis=1)/cb['accepted'][draws].sum(axis=1)
      else:
        p=(ca[field].sum()-cb[field].sum())/den
        v=(ca[field][draws].sum(axis=1)-cb[field][draws].sum(axis=1))/den
      lo,hi=np.quantile(v,[.025,.975])
      contrasts.append(dict(endpoint=field,contrast=f'{a} − {b}',difference_pp=p*100,ci95_low_pp=lo*100,ci95_high_pp=hi*100))
  pd.DataFrame(contrasts).to_csv(tables/'Table_2_C1_all_paired_bootstrap_contrasts.csv',index=False)
  pairs=[];original_frame=pred[original]
  wrong=original_frame.candidate_error.astype(bool)
  for a,b in [('Confidence-only','Appearance-only'),('Confidence-only','Combined'),('Combined','Appearance-only')]:
    fa=original_frame[[x[1] for x in POLICIES if x[0]==a][0]].astype(bool)
    fb=original_frame[[x[1] for x in POLICIES if x[0]==b][0]].astype(bool)
    k=int((wrong&fa&~fb).sum());l=int((wrong&~fa&fb).sum())
    pairs.append(dict(first=a,second=b,only_first=k,only_second=l,exact_p=binomtest(min(k,l),k+l,.5).pvalue))
  pd.DataFrame(pairs).to_csv(tables/'Supplementary_Table_S2_C1_paired_error_capture.csv',index=False)
  # S1: every held-out family/severity cell, enabling independent assessment of appearance referral
  altered_records=pred[altered].copy()
  s1=[]
  for (family,severity),z in altered_records.groupby(['family','severity'],sort=True):
    entry={'family':family,'severity':severity,'n':len(z)}
    for title,col,_ in POLICIES:
      label=title.lower().replace('-','_').replace(' ','_')
      entry[f'{label}_referred']=int(z[col].astype(bool).sum())
      entry[f'{label}_referral_pct']=100*float(z[col].astype(bool).mean())
    s1.append(entry)
  s1=pd.DataFrame(s1)
  assert len(s1)==12 and s1.n.eq(834).all()
  s1.to_csv(tables/'Supplementary_Table_S1_C1_alteration_families.csv',index=False)
  # Table 1 population inventory from registry
  source=pd.read_csv(ROOT/'data/metadata/source_split.csv')
  sc=source.split.value_counts().to_dict()
  assert len(source)==5836 and int(source.split.notna().sum())==5828 and sc=={'train':3603,'development':835,'calibration':556,'locked_test':834}
  pd.DataFrame([('Original collection',5836,'Four cultivars, three size labels'),('Eligible images',5828,'Source-group-separated'),('Training',3603,'Baseline/appearance training'),('Development',835,'Model choice and acquisition stress'),('Calibration',556,'Referral thresholds'),('Locked-test originals',834,'C1/C2 original endpoints'),('Constructed locked test',12510,'834 × (1 original + 2 benign + 12 altered)'),('Supplementary C3 runs',33,'12 four-view + 12 cultivar + 9 size holdout')],columns=['population','n','use']).to_csv(tables/'Table_1_study_populations.csv',index=False)
  # Table 4 cultivar-specific paired grade changes
  c2=[]
  for variety,sub in [('Overall',original_frame)]+list(original_frame.groupby('source_variety',sort=True)):
    old=sub.baseline_grade.eq(sub.source_size);new=sub.candidate_grade.eq(sub.source_size)
    c2.append(dict(cultivar=variety,n=len(sub),baseline_correct=int(old.sum()),candidate_correct=int(new.sum()),
       baseline_accuracy_pct=100*old.mean(),candidate_accuracy_pct=100*new.mean(),
       corrected=int((~old&new).sum()),regressed=int((old&~new).sum())))
  c2=pd.DataFrame(c2);assert (c2.iloc[0].corrected,c2.iloc[0].regressed)==(12,8)
  c2.to_csv(tables/'Table_4_C2_cultivar_conditional_grading.csv',index=False)
  # C3 Table 5: reconstruct AUROC SE from raw prediction scores
  size=[]
  for cond in ['Large','Medium','Small']:
    for seed in [11,23,37]:
      f=pd.read_csv(ROOT/f'data/raw_records/c3/size_holdout/CLASS_HOLDOUT_{cond}_seed{seed}.csv')
      unk=f.is_withheld_class.to_numpy(dtype=bool);u=f.uncertainty.to_numpy(dtype=float)
      auc=roc_auc_score(unk,u);se=delong_se(u[~unk],u[unk]);known=(f.native_size_label==f.predicted_label).to_numpy()
      size.append(dict(excluded_class=cond,seed=seed,known_n=int((~unk).sum()),unknown_n=int(unk.sum()),
                       known_accuracy_pct=100*known[~unk].mean(),unknown_auroc=auc,delong_se=se,
                       unknown_referral_pct=100*f.referred.to_numpy(dtype=bool)[unk].mean()))
  size=pd.DataFrame(size);size.to_csv(tables/'Table_5_C3_size_class_holdout.csv',index=False)
  # Table 6 stress by condition, paired within Dandhi
  stress=pd.read_csv(ROOT/'data/raw_records/stress_predictions.csv')
  stressmap={'clean':'Clean','contrast_0_7':'Contrast reduction','blur_1_2':'Blur','low_res_96':'Low resolution',
             'bright_1_3':'Brightness increase','dark_0_7':'Darkening','jpeg_35':'JPEG compression'}
  stress_rows=[]
  for key,label in stressmap.items():
    s=stress[(stress.variety=='Dandhi')&(stress.condition==key)]
    w=s.pivot(index='source_id',columns='model',values='correct')
    b=w.baseline.to_numpy(dtype=bool);c=w.Dandhi_targeted.to_numpy(dtype=bool)
    corrected=int((~b&c).sum());regressed=int((b&~c).sum())
    pval=binomtest(min(corrected,regressed),corrected+regressed,.5).pvalue
    stress_rows.append(dict(condition=label,n=len(w),accuracy_change_pp=100*(c.mean()-b.mean()),corrected=corrected,regressed=regressed,exact_p=pval))
  stress_df=pd.DataFrame(stress_rows);stress_df.to_csv(tables/'Table_6_Dandhi_photographic_stress.csv',index=False)
  # Table 7 LOVO
  lovo=pd.read_csv(ROOT/'data/tables/C3_CULTIVAR_HOLDOUT_ALL_12_RUNS.csv')
  context=dict(zip(c2.cultivar,c2.baseline_accuracy_pct))
  lovo_rows=[]
  for var in ['Aseel','Dandhi','Karblian','Kupro']:
    d=lovo[lovo.condition.eq(var)]
    lovo_rows.append(dict(cultivar=var,withheld_cultivar_accuracy_mean_pct=100*d.accuracy.mean(),
      reference_within_collection_accuracy_pct=context[var],withheld_evaluation_n=int(d.n.iloc[0]),
      seed_min_pct=100*d.accuracy.min(),seed_max_pct=100*d.accuracy.max()))
  lovo_df=pd.DataFrame(lovo_rows);lovo_df.to_csv(tables/'Table_7_C3_cultivar_holdout.csv',index=False)
  lovo_df.to_csv(tables/'Supplementary_Table_S3_LOVO_seed_ranges.csv',index=False)
  # Figure 1 scientific schematic, no externally sourced images or misleading 4-cultivar x severity statement
  fig,ax=plt.subplots(figsize=(13.5,5.6));ax.set(xlim=(0,13.5),ylim=(0,5.6));ax.axis('off')
  def panel(x,y,w,h,heading,lines,fill=LIGHT):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13,rounding_size=.12',facecolor=fill,edgecolor='#A7B6C0',lw=1))
    ax.text(x+.18,y+h-.38,heading,fontsize=10.6,fontweight='bold',color='#253C4C',va='top')
    ax.text(x+.18,y+h-.94,'\n'.join(lines),fontsize=8.7,color='#283E4A',va='top',linespacing=1.55)
  def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1.3,color=BLUE))
  panel(.18,3.36,4.1,1.95,'Source collection and splits',['5,836 images → 5,828 eligible','3,603 train · 835 development','556 calibration · 834 locked test'])
  panel(4.69,3.36,4.1,1.95,'Locked-test construction',['834 source-group-separated originals','1 original + 2 benign + 12 altered/source','4 held-out families × 3 severities','12,510 observations'])
  panel(9.2,3.36,4.1,1.95,'Nominal policy allocation',['Confidence (10/0) · Combined (5/5)','Appearance (0/10)','Original referral ≈ 10% in all policies'])
  arrow(4.29,4.33,4.67,4.33);arrow(8.8,4.33,9.18,4.33)
  panel(.18,.72,4.1,1.95,'C1  Selective referral',['Confidence vs appearance novelty','Accepted-original errors','Which misgraded originals are referred'],fill='#EDF4F6')
  panel(4.69,.72,4.1,1.95,'C2  Cultivar-targeted refinement',['Dandhi-specific RGB fine-tuning','0.5 probability blend (Dandhi only)','Corrections and regressions'],fill='#EFF4F0')
  panel(9.2,.72,4.1,1.95,'C3  Reliability under change',['4 visual representations · 7 imaging states','Leave-one-cultivar-out','Excluded-size-class recognition'],fill='#F3F1F5')
  ax.plot([6.75,6.75],[3.36,3.02],color=BLUE,lw=1.3)
  ax.plot([2.22,11.25],[3.02,3.02],color=BLUE,lw=1.3)
  for x in [2.22,6.75,11.25]:arrow(x,3.02,x,2.68)
  ax.text(6.75,.2,'Evaluation outcomes: accepted-set error · review-queue composition · subgroup accuracy · OOD discrimination',ha='center',color=GREY,fontsize=9)
  savefig(fig,figs,'Figure_1_study_design.png')
  # Figure 2 six-panel comparison, source-cluster uncertainty intervals for 4 panels
  titles=['Held-out synthetic referral','Accepted-original error','Misgraded originals referred','Benign referral','Original-image referral','Constructed-mixture referral']
  cols=['altered_referral_pct','accepted_original_error_pct','error_capture','benign_referral_pct','original_referral_pct','full_referral_pct']
  fig,axes=plt.subplots(2,3,figsize=(12,6.8));x=np.arange(3)
  for i,ax in enumerate(axes.flat):
    v=c1[cols[i]].to_numpy();ax.bar(x,v,color=COLORS,width=.63)
    if i in [0,1,3,4]:
      kind={0:'altered',1:'cer',3:'benign',4:'original'}[i]
      lows=np.array([arm_cis[n][kind][0] for n,_,_ in POLICIES]);highs=np.array([arm_cis[n][kind][1] for n,_,_ in POLICIES]);
      ax.errorbar(x,v,yerr=np.vstack([v-lows,highs-v]),color='#202B35',fmt='none',capsize=3,lw=1.0)
    for j,y in enumerate(v):ax.annotate(f'{y:.1f}'+('' if i==2 else '%'),(j,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(x,['Confidence','Combined','Appearance']);ax.set_title('abcdef'[i]+') '+titles[i],fontweight='bold',loc='left',fontsize=9)
    ax.set_ylim(0,max(v)*1.3+(.5 if i==2 else 0));ax.set_ylabel('Count' if i==2 else 'Percent (%)');ax.grid(axis='y',alpha=.13)
  fig.suptitle('Referral policies under a common nominal 10% original-image calibration budget',fontsize=12,y=1.01)
  fig.tight_layout();savefig(fig,figs,'Figure_2_policy_comparison.png')
  # Figure 3 paired changes
  fig,ax=plt.subplots(figsize=(7,3.4));v=np.array([12,-8,4]);ax.bar(range(3),v,color=[BLUE,'#AC7770',TEAL],width=.55)
  ax.axhline(0,color='#697E8B',lw=.9)
  for i,a in enumerate(v):ax.text(i,a+(0.45 if a>=0 else -.75),str(a),ha='center',va='bottom' if a>=0 else 'top',fontsize=10)
  ax.set_xticks(range(3),['Corrected','Regressed','Net gain']);ax.set_ylabel('Change in correctly graded originals (count)')
  ax.set_title('Dandhi-targeted refinement: paired prediction changes');fig.tight_layout();savefig(fig,figs,'Figure_3_dandhi_changes.png')
  # Figure 4 four-view accuracy by seed
  four=pd.read_csv(ROOT/'data/tables/C3_FOUR_VIEW_ALL_12_RUNS.csv')
  fig,ax=plt.subplots(figsize=(7.5,4.2))
  for label,keyword,color in [('Full image','full',BLUE),('Background-preserving','background',TEAL),('Foreground-only','foreground','#AE8A4E'),('Shape-only','only',PLUM)]:
    a=four[four.condition.eq(keyword)].sort_values('seed')
    ax.plot(a.seed,pct(a.accuracy),marker='o',lw=2,label=label,color=color)
  ax.set_xticks([11,23,37]);ax.set_xlabel('Training seed');ax.set_ylabel('Grading accuracy (%)');ax.set_ylim(55,86);ax.grid(alpha=.15);ax.legend(ncol=2,frameon=False,loc='lower left')
  ax.set_title('Four separately trained visual-representation models');fig.tight_layout();savefig(fig,figs,'Figure_4_four_views.png')
  # Figure 5 known accuracy vs unknown AUROC
  fig,ax=plt.subplots(figsize=(7,4.4))
  for cond,color,marker in [('Small',BLUE,'o'),('Medium',TEAL,'s'),('Large',PLUM,'^')]:
    d=size[size.excluded_class.eq(cond)]
    ax.scatter(d.known_accuracy_pct,d.unknown_auroc,color=color,s=56,marker=marker,label=f'Excluded {cond}')
    for _,row in d.iterrows():ax.annotate(str(int(row.seed)),(row.known_accuracy_pct,row.unknown_auroc),xytext=(3,3),textcoords='offset points',fontsize=7)
  ax.axhline(.5,color='#8395A1',ls='--',lw=.9);ax.set_xlabel('Known-class accuracy (%)');ax.set_ylabel('Excluded-class AUROC');ax.set_ylim(.49,.81)
  ax.grid(alpha=.13);ax.legend(frameon=False);fig.tight_layout();savefig(fig,figs,'Figure_5_class_holdout.png')
  # Figure 6 full cultivar-by-condition matrix
  g=stress.groupby(['variety','condition','model'],as_index=False).correct.mean()
  w=g.pivot_table(index=['variety','condition'],columns='model',values='correct').reset_index()
  w['delta']=100*(w.Dandhi_targeted-w.baseline)
  conditions=list(stressmap);varieties=['Aseel','Dandhi','Karblian','Kupro']
  mat=np.array([[w[(w.variety==v)&(w.condition==condition)].delta.iloc[0] for condition in conditions] for v in varieties])
  fig,ax=plt.subplots(figsize=(10,3.7));im=ax.imshow(mat,cmap='RdBu',vmin=-4,vmax=4,aspect='auto')
  for i in range(4):
    for j in range(7):ax.text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='#1A2835')
  ax.set_xticks(np.arange(7),['Clean','Blur','Bright','Contrast','Dark','JPEG','Low-res'],rotation=22)
  ax.set_yticks(np.arange(4),varieties);ax.set_title('Dandhi-targeted candidate minus baseline under acquisition stress')
  fig.colorbar(im,ax=ax,shrink=.8,label='Accuracy difference (percentage points)');fig.tight_layout();savefig(fig,figs,'Figure_6_stress.png')
  # Figure 7 LOVO contextual comparison
  fig,ax=plt.subplots(figsize=(7.6,4.4));x=np.arange(4);width=.34
  ax.bar(x-width/2,lovo_df.withheld_cultivar_accuracy_mean_pct,width,color=BLUE,label='Excluded-cultivar model')
  ax.bar(x+width/2,lovo_df.reference_within_collection_accuracy_pct,width,color='#AFB8C2',label='Within-collection reference (different models)')
  ax.set_xticks(x,lovo_df.cultivar);ax.set_ylim(0,108);ax.set_ylabel('Accuracy (%)');ax.legend(frameon=False)
  ax.set_title('Excluded-cultivar evaluation and contextual reference');fig.tight_layout();savefig(fig,figs,'Figure_7_cultivar_holdout.png')
  # Ensure paper-level numerical quantities are generated correctly
  assert abs(float(stress_df.loc[stress_df.condition.eq('Contrast reduction'),'accuracy_change_pp'].iloc[0])-3.211009)<.02
  assert abs(float(size.loc[(size.excluded_class=='Large')&(size.seed==11),'unknown_auroc'].iloc[0])-.5525)<.001
  manifest={'status':'PASS','input_population':len(pred),'sources':len(ids),
    'figures':[p.name for p in sorted(figs.glob('*.png'))],
    'tables':[p.name for p in sorted(tables.glob('*.csv'))],
    'replay_scope':'Figures and tables derived from frozen observation-level predictions. No new model inference, raw-image processing or retraining performed.',
    'bootstrap':{'replicates':2000,'seed':20261025}}
  assert len(manifest['figures'])==7 and len(manifest['tables'])==10
  (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps(manifest,indent=2))
if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(DEFAULT));args=parser.parse_args();main(args.output)
