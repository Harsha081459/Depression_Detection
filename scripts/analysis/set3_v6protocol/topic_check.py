"""Topic model (NMF k=20, label-free, fit on the 418 dev users, day 28), then
topic-adjusted and topic-matched AUROC for set-3 OOF scores, plus URL /
non-ASCII / writing-count strata. Mirrors v8_topic_confounding + v6_artefact_check."""
import json, sys, numpy as np, pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re as _re
BASE='/mnt/user-data/uploads/AIX PE'
SEED=42; DAY=28
CLINICAL_CANDIDATES=["depression","depressed","depressing","depress","depressive","antidepressant","anxiety","anxious","suicide","suicidal","therapy","therapist","psychiatrist","psychiatric","psychiatry","psychologist","psychology","medication","medicine","diagnosis","diagnosed","diagnose","disorder","mental","illness","bipolar","insomnia","counseling","counselor","prescription","pill","pills","hospital","doctor","symptom","symptoms","disease","sick","sickness"]
CLINICAL_STEMS=('depress','anxi','therap','psychiatr','psycholog','medicat','meds','antidepress','ssri','snri','zoloft','prozac','lexapro','sertraline','fluoxetine','escitalopram','citalopram','wellbutrin','bupropion','venlafaxine','effexor','diagnos','bipolar','ptsd','adhd','ocd','suicid','selfharm','counsel','mdd')
_CW='|'.join(sorted(_re.escape(w) for w in CLINICAL_CANDIDATES)); _CS='|'.join(_re.escape(s) for s in CLINICAL_STEMS)
CLINICAL_RE=_re.compile(r"\b(?:(?:%s)\b|(?:%s)\w*)"%(_CW,_CS),_re.IGNORECASE)
URL_RE=_re.compile(r'https?://\S+|www\.\S+|\S+\.(?:com|org|net|it|png|jpg|jpeg|gif|webp)\b|&amp;#x200b;|x200b|&amp;|&gt;|&lt;',_re.I)
MEDICAL_RE=_re.compile(r'\b(?:dose\w*|dr|mg|doctor\w*|pill\w*|prescri\w*|hospital\w*|clinic\w*|psych\w*|redd|preview|webp|png|jpg|gif|imgur|youtube|youtu)\b',_re.I)
def strict_clean(t): return MEDICAL_RE.sub(' ',URL_RE.sub(' ',CLINICAL_RE.sub(' ',str(t))))

def load():
    posts=pd.read_parquet(f'{BASE}/private/cohort_v1/eligible_posts.parquet')
    org=pd.read_parquet(f'{BASE}/private/cohort_v1/origins.parquet')
    posts['user_id']=posts.user_id.astype(str); org['user_id']=org.user_id.astype(str)
    posts['date']=pd.to_datetime(posts.date,utc=True)
    origin=pd.to_datetime(posts.user_id.map(org.set_index('user_id').origin),utc=True)
    posts['delta']=(posts.date-origin).dt.total_seconds()/86400
    rows=json.loads(open(f'{BASE}/private/cohort_v1/split.json').read())['rows']
    users=pd.DataFrame([{'user_id':str(r['user_id']),'label':int(r['label']),
                         'dev':r['split']=='train','pilot':bool(r['pilot_exposed'])} for r in rows])
    return posts,users

def user_docs(posts,ids,day):
    p=posts[posts.delta<=day]
    clean=p.text.fillna('').astype(str).map(strict_clean)
    return clean.groupby(p.user_id).apply(' '.join).reindex(ids).fillna('')

def topics(docs,dev_ids,k=20):
    vec=TfidfVectorizer(stop_words='english',min_df=5,max_df=0.5,max_features=20000,
                        token_pattern=r'(?u)\b[a-z][a-z]+\b',lowercase=True)
    Xdev=vec.fit_transform(docs.loc[dev_ids])
    nmf=NMF(n_components=k,init='nndsvda',random_state=SEED,max_iter=500).fit(Xdev)
    W=nmf.transform(vec.transform(docs)); s=W.sum(axis=1,keepdims=True)
    P=np.divide(W,s,out=np.zeros_like(W),where=s>0)
    names=np.array(vec.get_feature_names_out())
    top=[names[np.argsort(-c)[:8]].tolist() for c in nmf.components_]
    return pd.DataFrame(P,index=docs.index,columns=[f't{i:02d}' for i in range(k)]),top

def topic_only_oof(T,y):
    outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
    rows=[]
    for pos,(tr,va) in enumerate(outer.split(np.zeros(len(y)),y)):
        pipe=Pipeline([('s',StandardScaler()),('m',LogisticRegression(solver='liblinear',max_iter=2000,random_state=SEED))])
        g=GridSearchCV(pipe,{'m__C':[0.01,0.1,1,10]},scoring='roc_auc',
                       cv=StratifiedKFold(5,shuffle=True,random_state=SEED),n_jobs=2).fit(T.iloc[tr],y[tr])
        s=g.predict_proba(T.iloc[va])[:,1]
        rows+=[{'repeat':pos//5,'user_id':T.index[i],'label':int(y[i]),'score':float(v)} for i,v in zip(va,s)]
    return pd.DataFrame(rows)

def adjusted_auroc(score,label,strata):
    num=den=0.0; used=0
    for _,idx in strata.groupby(strata).groups.items():
        yy,ss=label.loc[idx],score.loc[idx]
        npos,nneg=int(yy.sum()),int((1-yy).sum())
        if npos and nneg:
            num+=roc_auc_score(yy,ss)*npos*nneg; den+=npos*nneg; used+=1
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

def main():
    posts,users=load()
    ids=users.user_id.tolist(); y=users.label.to_numpy(int)
    docs=user_docs(posts,ids,DAY)
    dev_ids=users.loc[users.dev,'user_id'].tolist()
    T,top=topics(docs,dev_ids)
    print('topics fitted',flush=True)
    topic_oof=topic_only_oof(T,y)
    prop=topic_oof.groupby('user_id').score.mean().reindex(ids)
    strata=pd.Series(pd.qcut(prop.rank(method='first'),5,labels=False),index=ids)
    T.to_parquet('topics_k20.parquet'); prop.to_frame('prop').to_parquet('topic_prop.parquet')
    strata.to_frame('stratum').to_parquet('topic_strata.parquet')
    out={'topic_top_words':{f't{i:02d}':w for i,w in enumerate(top)},
         'topic_mean_by_label':{c:{'positive':float(T.loc[users.loc[users.label==1,'user_id'],c].mean()),
                                   'control':float(T.loc[users.loc[users.label==0,'user_id'],c].mean())} for c in T.columns},
         'topic_only_mean_pooled_all523':float(np.mean([roc_auc_score(g.label,g.score) for _,g in topic_oof.groupby('repeat')]))}
    json.dump(out,open('topic_model_info.json','w'),indent=1)
    print('topic-only all523 %.3f'%out['topic_only_mean_pooled_all523'],flush=True)
main()
