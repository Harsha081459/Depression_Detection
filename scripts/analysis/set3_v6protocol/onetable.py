import json, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
BASE='/mnt/user-data/uploads/AIX PE'; SEED=42
rows=json.loads(open(f'{BASE}/private/cohort_v1/split.json').read())['rows']
dev=[r for r in rows if r['split']=='train']
ids=[str(r['user_id']) for r in dev]; y=np.array([int(r['label']) for r in dev])
pilot=np.array([bool(r['pilot_exposed']) for r in dev])
np_ids=[u for u,p in zip(ids,pilot) if not p]
lab=pd.Series(y,index=ids)
parts=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED).split(np.zeros(len(y)),y))
def ravg(df): return df.assign(r=df.groupby(['repeat','fold']).score.rank(pct=True)).groupby('user_id').r.mean()
text=json.loads(open(f'{BASE}/private/cohort_v6/folds/text_scores_day28.json').read())
vr=[]
for pos,(tr,va) in enumerate(parts):
    s=np.mean([pd.Series(text[pos][m]['valid']).rank(pct=True).to_numpy() for m in ('word','char','nb_word')],axis=0)
    vr+=[{'repeat':pos//5,'fold':pos%5,'user_id':ids[i],'label':int(y[i]),'score':float(v)} for i,v in zip(va,s)]
p12=pd.read_parquet('sets12_v6_preds.parquet'); p3=pd.read_parquet('v6protocol_preds.parquet')
M={'v6 text ensemble':ravg(pd.DataFrame(vr)),
   'Set 3, full':ravg(p3[p3.key=='masked|28|M1_clocks_emo|xgb']),
   'Set 1, LLM signals':ravg(p12[p12.key=='set1_llm_signals|28|logistic']),
   'Set 1 + Set 2':ravg(p12[p12.key=='set1_plus_set2|28|xgb']),
   'Set 3, artefact-free':ravg(p3[p3.key=='masked|28|M1r_rhythm_emo_nolen|xgb']),
   'Set 2, writing style':ravg(p12[p12.key=='set2_writing_style|28|xgb']),
   'Count only':ravg(p12[p12.key=='count_only|28|logistic'])}
def pboot(yy,a,b,boot=2000):
    rng=np.random.default_rng(SEED); ip=np.where(yy==1)[0]; ineg=np.where(yy==0)[0]; d=[]
    for _ in range(boot):
        t=np.concatenate([rng.choice(ip,len(ip)),rng.choice(ineg,len(ineg))])
        d.append(roc_auc_score(yy[t],a[t])-roc_auc_score(yy[t],b[t]))
    return float(roc_auc_score(yy,a)-roc_auc_score(yy,b)),[float(np.quantile(d,.025)),float(np.quantile(d,.975))]
yv=lab.loc[np_ids].to_numpy(); cnt=M['Count only'].reindex(np_ids).to_numpy()
v6s=M['v6 text ensemble'].reindex(np_ids).to_numpy()
out={}
print('%-24s %6s | %-22s | %-22s'%('model','NP','vs count-only','vs v6'))
for k,v in M.items():
    a=v.reindex(np_ids).to_numpy(); auc=roc_auc_score(yv,a)
    dc,cc=pboot(yv,a,cnt); dv,cv=pboot(yv,a,v6s)
    out[k]={'auroc_np':float(auc),'vs_count':{'delta':dc,'ci95':cc},'vs_v6':{'delta':dv,'ci95':cv}}
    print('%-24s %.3f | %+.3f [%+.3f,%+.3f] | %+.3f [%+.3f,%+.3f]'%(k,auc,dc,cc[0],cc[1],dv,cv[0],cv[1]))
json.dump(out,open('onetable_day28_nonpilot.json','w'),indent=1)
