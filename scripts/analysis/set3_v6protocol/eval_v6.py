"""Set 3 under the v6 protocol: dev 418, 5x5 RepeatedStratifiedKFold seed 42,
non-pilot AUROC on the 175, days 14/28/84, three disclosure arms."""
import json, time, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

BASE='/mnt/user-data/uploads/AIX PE'; OUT='.'
SEED=42; DAYS=(14,28,84); EMOTIONS=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","positive","negative"]
CLOCKS=["log_posts_per_week_so_far","night_frac_so_far","weekend_frac_so_far","comment_frac_so_far",
        "log_median_gap_so_far","gap_cv_so_far","burst_1h_frac_so_far","hour_entropy_so_far",
        "log_rate_change_so_far","log_n_last7d","log_n_last28d","log_recent_rate_ratio"]
EMO=[f"cum_emo_{e}" for e in EMOTIONS]+["mean_log_tokens_so_far"]
EMO_NC=[f"cum_emonc_{e}" for e in EMOTIONS]+["mean_log_tokens_so_far"]
RHYTHM=[f for f in CLOCKS if f!="comment_frac_so_far"]
EMO_NOLEN=[f"cum_emo_{e}" for e in EMOTIONS]
EMONC_NOLEN=[f"cum_emonc_{e}" for e in EMOTIONS]
SETS={"M1_clocks_emo":CLOCKS+EMO,
      "M1r_rhythm_emo_nolen":RHYTHM+EMO_NOLEN,
      "M1nc_clocks_emo_noclinical":CLOCKS+EMO_NC,
      "M1rnc_rhythm_emonc_nolen":RHYTHM+EMONC_NOLEN,
      "M0_clocks":CLOCKS,"M0r_rhythm_only":RHYTHM,
      "E_emo_only":EMO,"Er_emo_only_nolen":EMO_NOLEN}
MAIN=("M1_clocks_emo","M1r_rhythm_emo_nolen","M1nc_clocks_emo_noclinical","M1rnc_rhythm_emonc_nolen")
CGRID=(0.01,0.1,1.0,10.0)

def add_transforms(df):
    df=df.copy()
    df["log_posts_per_week_so_far"]=np.log1p(df["posts_per_week_so_far"])
    df["log_median_gap_so_far"]=np.log1p(df["median_gap_so_far"])
    df["log_rate_change_so_far"]=np.log(df["rate_change_so_far"])
    df["log_n_last7d"]=np.log1p(df["n_last7d"])
    df["log_n_last28d"]=np.log1p(df["n_last28d"])
    df["log_recent_rate_ratio"]=np.log(df["recent_rate_ratio"].clip(lower=1e-3))
    return df

def day_rows(prefix, day, ids):
    sub=prefix[prefix["days_since_first"]<=day]
    last=sub.groupby("user_id",sort=False).tail(1).set_index("user_id")
    out=last.reindex(ids)
    out["has_posts"]=(~out["t"].isna()).astype(int)
    return out.fillna(0.0)

def fit_logistic(X,y):
    best,bc=-1,1.0
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    for C in CGRID:
        s=[]
        for itr,iva in inner.split(X,y):
            m=Pipeline([("s",StandardScaler()),("c",LogisticRegression(C=C,max_iter=2000))]).fit(X[itr],y[itr])
            s.append(roc_auc_score(y[iva],m.predict_proba(X[iva])[:,1]))
        if np.mean(s)>best: best,bc=np.mean(s),C
    return Pipeline([("s",StandardScaler()),("c",LogisticRegression(C=bc,max_iter=2000))]).fit(X,y),bc

def fit_xgb(X,y):
    pos=float(y.sum()); neg=float(len(y)-pos)
    m=XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=3,min_child_weight=20,
        subsample=0.8,colsample_bytree=0.8,reg_lambda=2.0,tree_method="hist",
        eval_metric="logloss",n_jobs=2,random_state=SEED,scale_pos_weight=neg/max(pos,1))
    m.fit(X,y); return m,None

def evaluate(scores,pilot_by_user):
    s=scores.copy()
    s["score"]=s.groupby(["repeat","fold"]).score.rank(pct=True)
    a=[];n=[];aa=[];na=[]
    for _,g in s.groupby("repeat"):
        npm=~g.user_id.map(pilot_by_user).to_numpy(bool)
        a.append(roc_auc_score(g.label,g.score)); n.append(roc_auc_score(g.label[npm],g.score[npm]))
        aa.append(average_precision_score(g.label,g.score)); na.append(average_precision_score(g.label[npm],g.score[npm]))
    return dict(auroc_all=float(np.mean(a)),sd_all=float(np.std(a,ddof=1)),
                auroc_np=float(np.mean(n)),sd_np=float(np.std(n,ddof=1)),
                auprc_all=float(np.mean(aa)),auprc_np=float(np.mean(na)))

def main():
    t0=time.time()
    rows=json.loads(open(f'{BASE}/private/cohort_v1/split.json').read())["rows"]
    dev=[r for r in rows if r["split"]=="train"]
    ids=[str(r["user_id"]) for r in dev]
    y=np.array([int(r["label"]) for r in dev])
    pilot=pd.Series([bool(r["pilot_exposed"]) for r in dev],index=ids)
    parts=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED).split(np.zeros(len(y)),y))
    results=[]; preds={}
    for arm in ("masked","unmasked","before_disclosure"):
        pre=add_transforms(pd.read_parquet(f'{OUT}/prefix_{arm}.parquet'))
        pre["user_id"]=pre["user_id"].astype(str)
        sets = SETS if arm=="masked" else {k:SETS[k] for k in MAIN}
        for day in DAYS:
            R=day_rows(pre,day,ids)
            for sname,feats in sets.items():
                X=R[feats].to_numpy(dtype=float)
                for kind,fitter in (("logistic",fit_logistic),("xgb",fit_xgb)):
                    rr=[]
                    for pos,(tr,va) in enumerate(parts):
                        m,_=fitter(X[tr],y[tr])
                        p=m.predict_proba(X[va])[:,1]
                        rr+=[{"repeat":pos//5,"fold":pos%5,"user_id":ids[i],"label":int(y[i]),"score":float(v)} for i,v in zip(va,p)]
                    sc=pd.DataFrame(rr); res=evaluate(sc,pilot)
                    key=f"{arm}|{day}|{sname}|{kind}"
                    preds[key]=sc
                    results.append(dict(arm=arm,day=day,feature_set=sname,model=kind,**res))
                    print("%-58s all %.3f±%.3f | NP %.3f±%.3f | AP %.3f/%.3f  [%.0fs]"%(
                        key,res["auroc_all"],res["sd_all"],res["auroc_np"],res["sd_np"],
                        res["auprc_all"],res["auprc_np"],time.time()-t0),flush=True)
    pd.DataFrame(results).to_csv(f'{OUT}/v6protocol_results.csv',index=False)
    pd.concat([v.assign(key=k) for k,v in preds.items()]).to_parquet(f'{OUT}/v6protocol_preds.parquet',index=False)
    print("done %.0fs"%(time.time()-t0))

main()
