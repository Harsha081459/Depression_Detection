"""Point 6: early-flag analysis on the artefact-free masked features, dev 418."""
import json, sys, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
BASE='/mnt/user-data/uploads/AIX PE'
sys.path.insert(0,f'{BASE}/scripts/analysis')
from erisk_early_metrics import Decisions, apply_policy, summarize, sweep_policy
SEED=42
EMO=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","positive","negative"]
CLOCKS=["log_posts_per_week_so_far","night_frac_so_far","weekend_frac_so_far","comment_frac_so_far",
        "log_median_gap_so_far","gap_cv_so_far","burst_1h_frac_so_far","hour_entropy_so_far",
        "log_rate_change_so_far","log_n_last7d","log_n_last28d","log_recent_rate_ratio"]
RHYTHM=[f for f in CLOCKS if f!="comment_frac_so_far"]
SETS={"M1r_rhythm_emo_nolen":RHYTHM+[f"cum_emo_{e}" for e in EMO],
      "M1_clocks_emo":CLOCKS+[f"cum_emo_{e}" for e in EMO]+["mean_log_tokens_so_far"]}
TAUS=np.round(np.arange(0.30,0.951,0.025),3); T_MINS=(1,3,5,10)
OBJ=("erde5","erde50","f_latency"); K_WINDOWS=(5,10,25,50,100,200); DAYS={"14d":14,"28d":28,"84d":84}

def add_transforms(df):
    df=df.copy()
    df["log_posts_per_week_so_far"]=np.log1p(df["posts_per_week_so_far"])
    df["log_median_gap_so_far"]=np.log1p(df["median_gap_so_far"])
    df["log_rate_change_so_far"]=np.log(df["rate_change_so_far"])
    df["log_n_last7d"]=np.log1p(df["n_last7d"]); df["log_n_last28d"]=np.log1p(df["n_last28d"])
    df["log_recent_rate_ratio"]=np.log(df["recent_rate_ratio"].clip(lower=1e-3))
    return df
def row_weights(rows):
    n_per=rows.groupby("user_id")["t"].transform("size").to_numpy(float)
    u=rows.drop_duplicates("user_id"); nu=len(u); npos=int(u["label"].sum())
    cf=np.where(rows["label"].to_numpy()==1, nu/(2*npos), nu/(2*(nu-npos)))
    w=cf/n_per; return w*(len(w)/w.sum())
def fit(X,y,w):
    return XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=3,min_child_weight=20,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,tree_method="hist",eval_metric="logloss",
        n_jobs=2,random_state=SEED).fit(X,y,sample_weight=w)

t0=time.time()
rows=json.loads(open(f'{BASE}/private/cohort_v1/split.json').read())['rows']
dev=[r for r in rows if r['split']=='train']
ids=[str(r['user_id']) for r in dev]
users=pd.DataFrame({'user_id':ids,'label':[int(r['label']) for r in dev],
                    'pilot':[bool(r['pilot_exposed']) for r in dev]})
np_ids=set(users.loc[~users.pilot,'user_id'])
pre=add_transforms(pd.read_parquet('prefix_masked.parquet'))
pre['user_id']=pre['user_id'].astype(str)
df=pre[pre.user_id.isin(set(ids))].reset_index(drop=True)
prev=float(users.label.mean())
print('rows',len(df),'users',df.user_id.nunique(),'prev %.3f'%prev,flush=True)

uidx={u:i for i,u in enumerate(ids)}
rup=df.user_id.map(uidx).to_numpy()
yu=users.label.to_numpy()
outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
out={}
for sname,feats in SETS.items():
    oof=df[['user_id','label','t','days_since_first']].copy()
    tuned={o:[] for o in OBJ}
    for fold,(tr_u,te_u) in enumerate(outer.split(np.zeros(len(yu)),yu),1):
        trm=np.isin(rup,tr_u); tem=~trm
        train=df[trm]; test=df[tem]
        w=row_weights(train); ytr=train['label'].to_numpy()
        prev_tr=float(users.iloc[tr_u]['label'].mean())
        m=fit(train[feats].to_numpy(float),ytr,w)
        pte=m.predict_proba(test[feats].to_numpy(float))[:,1]
        oof.loc[tem,'p']=pte
        inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED+fold)
        iu=users.iloc[tr_u].reset_index(drop=True)
        ipos=train['user_id'].map({u:i for i,u in enumerate(iu['user_id'])}).to_numpy()
        pin=np.zeros(len(train))
        for itr,ite in inner.split(np.zeros(len(iu)),iu['label'].to_numpy()):
            mi=np.isin(ipos,itr); sub=train[mi]
            md=fit(sub[feats].to_numpy(float),sub['label'].to_numpy(),row_weights(sub))
            pin[~mi]=md.predict_proba(train[~mi][feats].to_numpy(float))[:,1]
        istream=pd.DataFrame({'user_id':train['user_id'].to_numpy(),'t':train['t'].to_numpy(),
                              'label':ytr,'p':pin})
        best,_=sweep_policy(istream,TAUS,T_MINS,OBJ,c_fp=prev_tr)
        tstream=pd.DataFrame({'user_id':test['user_id'].to_numpy(),'t':test['t'].to_numpy(),
                              'label':test['label'].to_numpy(),'p':pte})
        for o,(tau,tmin) in best.items():
            tuned[o].append(apply_policy(tstream,tau,tmin))
        print(f'{sname} fold {fold}/5 {time.time()-t0:.0f}s',flush=True)
    pooled={o:Decisions(user_id=np.concatenate([d.user_id for d in v]),
                        y_true=np.concatenate([d.y_true for d in v]),
                        flagged=np.concatenate([d.flagged for d in v]),
                        delay=np.concatenate([d.delay for d in v]),
                        n_writings=np.concatenate([d.n_writings for d in v])) for o,v in tuned.items()}
    ent={'early':{o:summarize(d,c_fp=prev) for o,d in pooled.items()}}
    # non-pilot restriction of the ERDE50 policy
    d=pooled['erde50']; keep=np.isin(d.user_id,list(np_ids))
    ent['early_non_pilot']=summarize(Decisions(d.user_id[keep],d.y_true[keep],d.flagged[keep],
                                               d.delay[keep],d.n_writings[keep]),
                                     c_fp=float(users.loc[~users.pilot,'label'].mean()))
    # window AUROCs
    wins={}
    for nm,dd in DAYS.items():
        r=oof[oof.days_since_first<=dd].groupby('user_id',sort=True).tail(1).set_index('user_id')
        wins[nm]=r
    for k in K_WINDOWS:
        r=oof[oof.t<=k].groupby('user_id',sort=True).tail(1).set_index('user_id')
        wins[f'k{k}']=r
    wm={}
    for nm,r in wins.items():
        npm=r.index.isin(np_ids)
        wm[nm]={'auroc_all':float(roc_auc_score(r.label,r.p)),
                'auroc_np':float(roc_auc_score(r.label[npm],r.p[npm])),
                'auprc_all':float(average_precision_score(r.label,r.p)),
                'median_writings_seen':float(r.t.median()),'n_users':int(len(r))}
    ent['windows']=wm
    out[sname]=ent
    print(sname,'ERDE50 %.3f  P %.2f R %.2f  delay %s | k5 NP %.3f  28d NP %.3f'%(
        ent['early']['erde50']['erde50'],ent['early']['erde50']['precision'],
        ent['early']['erde50']['recall'],ent['early']['erde50']['median_tp_delay'],
        wm['k5']['auroc_np'],wm['28d']['auroc_np']),flush=True)

nw=df.groupby('user_id')['t'].max().reindex(ids).to_numpy()
out['baselines']={
 'never_flag':summarize(Decisions(np.array(ids),yu,np.zeros(len(ids),int),nw,nw),prev),
 'flag_all_k1':summarize(Decisions(np.array(ids),yu,np.ones(len(ids),int),np.ones(len(ids),int),nw),prev)}
json.dump(out,open('early_flag.json','w'),indent=1,default=float)
print('WROTE early_flag.json %.0fs'%(time.time()-t0))
