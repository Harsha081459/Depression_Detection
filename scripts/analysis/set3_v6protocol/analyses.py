"""Topic/artefact checks, paired bootstrap vs v6, rank fusion, lockbox."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
BASE='/mnt/user-data/uploads/AIX PE'; SEED=42
EMO=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","positive","negative"]
CLOCKS=["log_posts_per_week_so_far","night_frac_so_far","weekend_frac_so_far","comment_frac_so_far",
        "log_median_gap_so_far","gap_cv_so_far","burst_1h_frac_so_far","hour_entropy_so_far",
        "log_rate_change_so_far","log_n_last7d","log_n_last28d","log_recent_rate_ratio"]
RHYTHM=[f for f in CLOCKS if f!="comment_frac_so_far"]
SETS={"M1_clocks_emo":CLOCKS+[f"cum_emo_{e}" for e in EMO]+["mean_log_tokens_so_far"],
      "M1r_rhythm_emo_nolen":RHYTHM+[f"cum_emo_{e}" for e in EMO]}

def add_transforms(df):
    df=df.copy()
    df["log_posts_per_week_so_far"]=np.log1p(df["posts_per_week_so_far"])
    df["log_median_gap_so_far"]=np.log1p(df["median_gap_so_far"])
    df["log_rate_change_so_far"]=np.log(df["rate_change_so_far"])
    df["log_n_last7d"]=np.log1p(df["n_last7d"]); df["log_n_last28d"]=np.log1p(df["n_last28d"])
    df["log_recent_rate_ratio"]=np.log(df["recent_rate_ratio"].clip(lower=1e-3))
    return df
def day_rows(pre,day,ids):
    sub=pre[pre["days_since_first"]<=day]
    out=sub.groupby("user_id",sort=False).tail(1).set_index("user_id").reindex(ids)
    return out.fillna(0.0)
def fit_xgb(X,y):
    pos=float(y.sum()); neg=float(len(y)-pos)
    return XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=3,min_child_weight=20,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,tree_method="hist",eval_metric="logloss",
        n_jobs=2,random_state=SEED,scale_pos_weight=neg/max(pos,1)).fit(X,y)
def adjusted_auroc(score,label,strata):
    num=den=0.0; used=0
    for _,idx in strata.groupby(strata).groups.items():
        yy,ss=label.loc[idx],score.loc[idx]
        p,n=int(yy.sum()),int((1-yy).sum())
        if p and n: num+=roc_auc_score(yy,ss)*p*n; den+=p*n; used+=1
    return (num/den if den else float('nan')),used
def matched_auroc(score,label,T,k=5,boot=2000):
    pos=label.index[label==1]; ctl=label.index[label==0]
    A=T.loc[pos].to_numpy(); B=T.loc[ctl].to_numpy()
    na=np.linalg.norm(A,axis=1,keepdims=True); nb=np.linalg.norm(B,axis=1,keepdims=True)
    sim=(A/np.where(na>0,na,1))@(B/np.where(nb>0,nb,1)).T
    nn=np.argsort(-sim,axis=1)[:,:k]
    sp=score.loc[pos].to_numpy(); sc=score.loc[ctl].to_numpy()
    wins=np.array([[(sp[i]>sc[j])+0.5*(sp[i]==sc[j]) for j in nn[i]] for i in range(len(pos))])
    rng=np.random.default_rng(SEED)
    bs=[wins[rng.integers(0,len(pos),len(pos))].mean() for _ in range(boot)]
    return float(wins.mean()),[float(np.quantile(bs,0.025)),float(np.quantile(bs,0.975))]
def paired_boot(y,sa,sb,boot=2000):
    rng=np.random.default_rng(SEED); y=np.asarray(y)
    ip=np.where(y==1)[0]; ineg=np.where(y==0)[0]; d=[]
    for _ in range(boot):
        t=np.concatenate([rng.choice(ip,len(ip)),rng.choice(ineg,len(ineg))])
        d.append(roc_auc_score(y[t],sa[t])-roc_auc_score(y[t],sb[t]))
    return float(roc_auc_score(y,sa)-roc_auc_score(y,sb)),[float(np.quantile(d,.025)),float(np.quantile(d,.975))]

rows=json.loads(open(f'{BASE}/private/cohort_v1/split.json').read())['rows']
dev=[r for r in rows if r['split']=='train']; test=[r for r in rows if r['split']=='test']
ids=[str(r['user_id']) for r in dev]; y=np.array([int(r['label']) for r in dev])
pilot=np.array([bool(r['pilot_exposed']) for r in dev])
np_ids=[u for u,p in zip(ids,pilot) if not p]
tids=[str(r['user_id']) for r in test]; ty=np.array([int(r['label']) for r in test])
parts=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED).split(np.zeros(len(y)),y))
lab=pd.Series(y,index=ids)
T=pd.read_parquet('topics_k20.parquet'); strata=pd.read_parquet('topic_strata.parquet')['stratum']
preds=pd.read_parquet('v6protocol_preds.parquet')
out={}

# v6 ensemble OOF (day 28) per user, repeat-averaged
text=json.loads(open(f'{BASE}/private/cohort_v6/folds/text_scores_day28.json').read())
v6rows=[]
for pos,(tr,va) in enumerate(parts):
    r=[pd.Series(text[pos][m]['valid']).rank(pct=True).to_numpy() for m in ('word','char','nb_word')]
    s=np.mean(r,axis=0)
    v6rows+=[{'repeat':pos//5,'fold':pos%5,'user_id':ids[i],'label':int(y[i]),'score':float(v)} for i,v in zip(va,s)]
v6=pd.DataFrame(v6rows)
def ravg(df): return df.assign(r=df.groupby(['repeat','fold']).score.rank(pct=True)).groupby('user_id').r.mean()
v6u=ravg(v6)

# --- A: topic-adjusted / matched, non-pilot dev 175 ---
cmp_sets={'v6_text_ensemble':v6u}
for s in ('M1_clocks_emo','M1r_rhythm_emo_nolen'):
    cmp_sets[f'set3_{s}']=ravg(preds[preds.key==f'masked|28|{s}|xgb'])
res={}
for name,sc in cmp_sets.items():
    sc=sc.reindex(np_ids); l=lab.loc[np_ids]
    adj,used=adjusted_auroc(sc,l,strata.loc[np_ids])
    mt,ci=matched_auroc(sc,l,T.loc[np_ids])
    res[name]={'unadjusted':float(roc_auc_score(l,sc)),'topic_adjusted':adj,'strata_used':used,
               'topic_matched':mt,'topic_matched_ci':ci}
    print(f"{name:28s} unadj {res[name]['unadjusted']:.3f}  adj {adj:.3f}  matched {mt:.3f} {['%.3f'%c for c in ci]}",flush=True)
out['A_topic_non_pilot_dev_175']=res

# --- B: artefact strata (non-pilot dev), set3 artefact-free ---
pre=add_transforms(pd.read_parquet('prefix_masked.parquet')); pre['user_id']=pre['user_id'].astype(str)
R=day_rows(pre,28,ids)
art=R[['url_frac_so_far','nonascii_frac_so_far','t']].rename(columns={'t':'n_posts'})
s3=cmp_sets['set3_M1r_rhythm_emo_nolen']
strat={}
for col in ('url_frac_so_far','nonascii_frac_so_far','n_posts'):
    v=art.loc[np_ids,col]; med=v.median()
    lo=v.index[v<=med]; hi=v.index[v>med]
    d={}
    for nm,uu in (('low',lo),('high',hi)):
        yy=lab.loc[uu]
        d[nm]={'n':int(len(uu)),'pos':int(yy.sum()),
               'set3':float(roc_auc_score(yy,s3.loc[uu])) if yy.nunique()==2 else None,
               'v6':float(roc_auc_score(yy,v6u.loc[uu])) if yy.nunique()==2 else None}
    strat[col]=d
zero=art.loc[np_ids].index[art.loc[np_ids,'url_frac_so_far']==0]
strat['users_with_zero_url']={'n':int(len(zero)),'pos':int(lab.loc[zero].sum()),
    'set3':float(roc_auc_score(lab.loc[zero],s3.loc[zero])),'v6':float(roc_auc_score(lab.loc[zero],v6u.loc[zero]))}
out['B_artefact_strata_non_pilot_dev']=strat
print('artefact strata done',flush=True)

# --- C: paired bootstrap set3 vs v6, non-pilot dev ---
pb={}
for s in ('M1_clocks_emo','M1r_rhythm_emo_nolen'):
    a=cmp_sets[f'set3_{s}'].reindex(np_ids).to_numpy(); b=v6u.reindex(np_ids).to_numpy()
    est,ci=paired_boot(lab.loc[np_ids].to_numpy(),a,b)
    pb[f'set3_{s}_minus_v6']={'delta_auroc':est,'ci95':ci}
    print(f'paired set3 {s} - v6: {est:+.3f} {["%.3f"%c for c in ci]}',flush=True)
out['C_paired_bootstrap_vs_v6_non_pilot_dev']=pb

# --- D: rank fusion ---
fus={}
for s in ('M1_clocks_emo','M1r_rhythm_emo_nolen'):
    sp=preds[preds.key==f'masked|28|{s}|xgb'][['repeat','fold','user_id','label','score']]
    j=v6.merge(sp,on=['repeat','fold','user_id','label'],suffixes=('_v6','_s3'))
    j['r_v6']=j.groupby(['repeat','fold']).score_v6.rank(pct=True)
    j['r_s3']=j.groupby(['repeat','fold']).score_s3.rank(pct=True)
    j['score']=(j.r_v6+j.r_s3)/2
    per=[np.mean([roc_auc_score(g.label[~g.user_id.isin(set(ids)-set(np_ids))],
                                g['score'][~g.user_id.isin(set(ids)-set(np_ids))])
         for _,g in j[j.repeat==rp].groupby('fold')]) for rp in range(5)]
    fu=ravg(j[['repeat','fold','user_id','label','score']])
    a=fu.reindex(np_ids).to_numpy(); b=v6u.reindex(np_ids).to_numpy()
    est,ci=paired_boot(lab.loc[np_ids].to_numpy(),a,b)
    fus[f'fusion_v6_plus_{s}']={'auroc_np':float(roc_auc_score(lab.loc[np_ids],a)),
        'auroc_all':float(roc_auc_score(lab,fu.reindex(ids))),
        'delta_vs_v6':est,'ci95':ci}
    print(f'fusion v6+{s}: NP {fus[f"fusion_v6_plus_{s}"]["auroc_np"]:.3f}  delta {est:+.3f} {["%.3f"%c for c in ci]}',flush=True)
out['D_rank_fusion_non_pilot_dev']=fus

# --- E: lockbox (already spent) ---
lb={}
for arm in ('masked',):
    p=add_transforms(pd.read_parquet(f'prefix_{arm}.parquet')); p['user_id']=p['user_id'].astype(str)
    for day in (14,28,84):
        Rtr=day_rows(p,day,ids); Rte=day_rows(p,day,tids)
        for s,feats in SETS.items():
            m=fit_xgb(Rtr[feats].to_numpy(float),y)
            pt=m.predict_proba(Rte[feats].to_numpy(float))[:,1]
            rng=np.random.default_rng(SEED); ip=np.where(ty==1)[0]; ineg=np.where(ty==0)[0]; bs=[]
            for _ in range(2000):
                t=np.concatenate([rng.choice(ip,len(ip)),rng.choice(ineg,len(ineg))])
                bs.append(roc_auc_score(ty[t],pt[t]))
            lb[f'{arm}|{day}|{s}']={'auroc':float(roc_auc_score(ty,pt)),
                'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
                'auprc':float(average_precision_score(ty,pt))}
            print(f'lockbox {arm} day{day} {s}: {lb[f"{arm}|{day}|{s}"]["auroc"]:.3f}',flush=True)
out['E_lockbox_ALREADY_SPENT']=lb
json.dump(out,open('analyses.json','w'),indent=1)
print('WROTE analyses.json')
