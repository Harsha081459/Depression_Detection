"""Sets 1 and 2 under the v6 split. Masked arm, days 14/28/84."""
import json, sys, time, numpy as np, pandas as pd
from datetime import timedelta
BASE='/mnt/user-data/uploads/AIX PE'
sys.path.insert(0,f'{BASE}/scripts/analysis')
import final_indirect_pipeline as fip
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
SEED=42; DAYS=(14,28,84); CGRID=(0.01,0.1,1.0,10.0)
SIG=['self_blame','helplessness','isolation','family_strain','emptiness']
STYLE=['polarity_flip','polarity_jump','distress_burst','absolutist']
t0=time.time()

rows=json.loads(open(f'{BASE}/private/cohort_v1/split.json').read())['rows']
allu=pd.DataFrame([{'user_id':str(r['user_id']),'label':int(r['label']),
                    'dev':r['split']=='train','pilot':bool(r['pilot_exposed'])} for r in rows])
ids=allu.loc[allu.dev,'user_id'].tolist(); y=allu.loc[allu.dev,'label'].to_numpy()
pilot=pd.Series(allu.loc[allu.dev,'pilot'].to_numpy(),index=ids)
tids=allu.loc[~allu.dev,'user_id'].tolist(); ty=allu.loc[~allu.dev,'label'].to_numpy()
every=allu.user_id.tolist()

# ---- set 2: style features from text, masked arm ----
text=pd.read_csv(f'{BASE}/data/processed/erisk/task2_target_text.csv')
text['user_id']=text['user_id'].astype(str)
data=fip._prepare(text)
print('prepared %.0fs'%(time.time()-t0),flush=True)
grouped={u:g.to_dict('records') for u,g in data.groupby('user_id',sort=False)}
set2={}
for day in DAYS:
    recs=[]
    for u in every:
        posts=grouped.get(u,[])
        if posts:
            origin=posts[0]['ts'].to_pydatetime(); end=origin+timedelta(days=day)
            prefix=[p for p in posts if p['ts'].to_pydatetime()<=end]
            kept=[p for p in prefix if not p['disclosure']]
        else:
            kept=[]
        agg=fip.aggregate(kept,None)
        recs.append({'user_id':u,**{k:agg[k] for k in STYLE},
                     'log_n_posts':float(np.log1p(len(kept))),'has_posts':int(len(kept)>0)})
    set2[day]=pd.DataFrame(recs).set_index('user_id')
    print('set2 day',day,'%.0fs'%(time.time()-t0),flush=True)

# validate against features.csv (4w == day 28, 12w == day 84)
fc=pd.read_csv(f'{BASE}/results/final_indirect/features.csv')
fc['user_id']=fc['user_id'].astype(str)
for day,win in ((28,'4w'),(84,'12w')):
    ref=fc[(fc.window==win)&(fc.arm=='masked')].set_index('user_id')
    for c in STYLE:
        d=np.abs(set2[day][c].reindex(ref.index).to_numpy()-ref[c].to_numpy())
        print(f'  validate day{day} {c}: maxdiff {np.nanmax(d):.2e}',flush=True)

# ---- set 1: LLM signal window means ----
sc=pd.read_parquet(f'{BASE}/private/cohort_v1/scores/part-00000000.parquet')
sc['user_id']=sc['user_id'].astype(str)
org=pd.read_parquet(f'{BASE}/private/cohort_v1/origins.parquet'); org['user_id']=org.user_id.astype(str)
sc['date']=pd.to_datetime(sc['date'],utc=True)
origin=pd.to_datetime(sc.user_id.map(org.set_index('user_id').origin),utc=True)
sc['delta']=(sc['date']-origin).dt.total_seconds()/86400
set1={}
for day in DAYS:
    w=sc[sc.delta<=day]
    m=w.groupby('user_id')[SIG].mean()
    n=w.groupby('user_id').size().rename('n')
    f=m.reindex(every).fillna(0.0)
    f['log_n_posts']=np.log1p(n.reindex(every).fillna(0).to_numpy())
    f['has_posts']=(~n.reindex(every).isna()).astype(int).to_numpy()
    set1[day]=f
    print('set1 day',day,'users with posts',int(f.has_posts.sum()),flush=True)

FEATS={'set1_llm_signals':(set1,SIG+['log_n_posts','has_posts']),
       'set2_writing_style':(set2,STYLE+['log_n_posts','has_posts']),
       'count_only':(set1,['log_n_posts'])}
def frame(day):
    return pd.concat([set1[day].drop(columns=['log_n_posts','has_posts']),set2[day]],axis=1)
set12={d:frame(d) for d in DAYS}
FEATS['set1_plus_set2']=(set12,SIG+STYLE+['log_n_posts','has_posts'])

def fit_logistic(X,yy):
    best,bc=-1,1.0
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    for C in CGRID:
        s=[]
        for itr,iva in inner.split(X,yy):
            m=Pipeline([('s',StandardScaler()),('c',LogisticRegression(C=C,max_iter=2000))]).fit(X[itr],yy[itr])
            s.append(roc_auc_score(yy[iva],m.predict_proba(X[iva])[:,1]))
        if np.mean(s)>best: best,bc=np.mean(s),C
    return Pipeline([('s',StandardScaler()),('c',LogisticRegression(C=bc,max_iter=2000))]).fit(X,yy)
def fit_xgb(X,yy):
    p=float(yy.sum()); n=float(len(yy)-p)
    return XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=3,min_child_weight=20,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,tree_method='hist',eval_metric='logloss',
        n_jobs=2,random_state=SEED,scale_pos_weight=n/max(p,1)).fit(X,yy)
def evaluate(scores,pb):
    s=scores.copy(); s['score']=s.groupby(['repeat','fold']).score.rank(pct=True)
    a=[];n=[];aa=[];na=[]
    for _,g in s.groupby('repeat'):
        npm=~g.user_id.map(pb).to_numpy(bool)
        a.append(roc_auc_score(g.label,g.score)); n.append(roc_auc_score(g.label[npm],g.score[npm]))
        aa.append(average_precision_score(g.label,g.score)); na.append(average_precision_score(g.label[npm],g.score[npm]))
    return dict(auroc_all=float(np.mean(a)),sd_all=float(np.std(a,ddof=1)),
                auroc_np=float(np.mean(n)),sd_np=float(np.std(n,ddof=1)),
                auprc_all=float(np.mean(aa)),auprc_np=float(np.mean(na)),
                np_by_repeat=[round(v,4) for v in n])

parts=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED).split(np.zeros(len(y)),y))
res=[]; preds={}
for name,(tbl,cols) in FEATS.items():
    for day in DAYS:
        X=tbl[day].reindex(ids)[cols].to_numpy(float)
        for kind,fitter in (('logistic',fit_logistic),('xgb',fit_xgb)):
            rr=[]
            for pos,(tr,va) in enumerate(parts):
                m=fitter(X[tr],y[tr]); p=m.predict_proba(X[va])[:,1]
                rr+=[{'repeat':pos//5,'fold':pos%5,'user_id':ids[i],'label':int(y[i]),'score':float(v)} for i,v in zip(va,p)]
            scdf=pd.DataFrame(rr); r=evaluate(scdf,pilot)
            preds[f'{name}|{day}|{kind}']=scdf
            res.append(dict(feature_set=name,day=day,model=kind,n_features=len(cols),**r))
            print('%-34s d%-3d %-8s all %.3f±%.3f | NP %.3f±%.3f  [%.0fs]'%(
                name,day,kind,r['auroc_all'],r['sd_all'],r['auroc_np'],r['sd_np'],time.time()-t0),flush=True)

# lockbox (already spent)
lb={}
for name,(tbl,cols) in FEATS.items():
    for day in DAYS:
        Xtr=tbl[day].reindex(ids)[cols].to_numpy(float); Xte=tbl[day].reindex(tids)[cols].to_numpy(float)
        m=fit_xgb(Xtr,y); pt=m.predict_proba(Xte)[:,1]
        rng=np.random.default_rng(SEED); ip=np.where(ty==1)[0]; ineg=np.where(ty==0)[0]
        bs=[roc_auc_score(ty[np.concatenate([rng.choice(ip,len(ip)),rng.choice(ineg,len(ineg))])],
            pt[np.concatenate([rng.choice(ip,len(ip)),rng.choice(ineg,len(ineg))])]) for _ in range(0)]
        lb[f'{name}|{day}']={'auroc':float(roc_auc_score(ty,pt)),'auprc':float(average_precision_score(ty,pt))}
pd.DataFrame(res).to_csv('sets12_v6_results.csv',index=False)
pd.concat([v.assign(key=k) for k,v in preds.items()]).to_parquet('sets12_v6_preds.parquet',index=False)
json.dump(lb,open('sets12_lockbox_spent.json','w'),indent=1)
print('done %.0fs'%(time.time()-t0))
