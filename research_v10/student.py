"""Extend the five Qwen signals to all 155,284 writings with a text 'student' model.

Qwen scored only the 62k writings inside day 84 (GPU cost). A ridge model on hashed
word+char n-grams learns Qwen's logit(P(Yes)) per signal. Targets are Qwen outputs,
never depression labels, so this is label-free. Fitted on development users only.
Fidelity is checked on a user-disjoint 20% of development users first."""
import json, time, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import os as _os
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
_ROOT = _os.environ.get('AIXPE_ROOT') or (str(_here.parent) if (_here.parent / 'data').exists() else '/mnt/user-data/uploads/AIX PE')
B=_ROOT; SEED=42
SIG=['self_blame','helplessness','isolation','family_strain','emptiness']
t0=time.time()
rows=json.loads(open(f'{B}/private/cohort_v1/split.json').read())['rows']
dev={str(r['user_id']) for r in rows if r['split']=='train'}
w=pd.read_parquet('writings_PRIVATE.parquet')
txt=w.text.fillna('').str.slice(0,2000)          # Qwen saw the first 2,000 characters
hw=HashingVectorizer(ngram_range=(1,2),n_features=2**20,alternate_sign=False,norm=None,token_pattern=r"(?u)\b\w[\w']*\b")
hc=HashingVectorizer(analyzer='char_wb',ngram_range=(3,5),n_features=2**20,alternate_sign=False,norm=None)
Xw=hw.transform(txt); Xc=hc.transform(txt)
lab_mask=w.q_self_blame.notna().to_numpy() & w.user_id.isin(dev).to_numpy()
tw=TfidfTransformer(sublinear_tf=True).fit(Xw[lab_mask]); tc=TfidfTransformer(sublinear_tf=True).fit(Xc[lab_mask])
X=hstack([tw.transform(Xw),tc.transform(Xc)]).tocsr()
print('X',X.shape,'labelled dev writings',int(lab_mask.sum()),'%.0fs'%(time.time()-t0),flush=True)
logit=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
idx=np.where(lab_mask)[0]
gs=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED)
tr,te=next(gs.split(idx,groups=w.user_id.to_numpy()[idx])); tr,te=idx[tr],idx[te]
fid={}
for k in SIG:
    y=logit(w[f'q_{k}'].to_numpy())
    best=None
    for a in (3.0,10.0,30.0,100.0):
        m=Ridge(alpha=a,solver='sparse_cg',random_state=SEED).fit(X[tr],y[tr]); p=m.predict(X[te])
        r=roc_auc_score(w[f"q_{k}"].to_numpy()[te]>0.5,p)  # select on finding Qwen-yes writings
        if best is None or r>best[1]: best=(a,r,p)
    a,r,p=best; q=w[f'q_{k}'].to_numpy()[te]
    fid[k]={'alpha':a,'alpha_grid':[3.0,10.0,30.0,100.0],'spearman_all_writings':float(spearmanr(p,y[te]).correlation),'pearson_logit':float(pearsonr(p,y[te])[0]),
            'auroc_vs_qwen_yes':float(roc_auc_score(q>0.5,p)),'qwen_yes_rate':float((q>0.5).mean())}
    print(k,fid[k],'%.0fs'%(time.time()-t0),flush=True)
# refit on all labelled dev writings, predict every writing
for k in SIG:
    y=logit(w[f'q_{k}'].to_numpy())
    m=Ridge(alpha=fid[k]['alpha'],solver='sparse_cg',random_state=SEED).fit(X[lab_mask],y[lab_mask])
    w[f's_{k}']=1/(1+np.exp(-m.predict(X)))
out=w[['user_id','ts_sec']+[f's_{k}' for k in SIG]]
out.to_parquet('student_scores_PRIVATE.parquet',index=False)
json.dump(fid,open('student_fidelity.json','w'),indent=1)
print('done %.0fs'%(time.time()-t0))
