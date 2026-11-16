"""No-future test: recompute prefix means from writings with index < k, and check that
the k-th writing is at or before the decision day and the (k+1)-th is not before it."""
import json, numpy as np, pandas as pd, chunks as C
rows=json.loads(open(f'{C.B}/private/cohort_v1/split.json').read())['rows']
users={str(r['user_id']):int(r['label']) for r in rows if r['split']=='train'}
w=pd.read_parquet('writings_PRIVATE.parquet').drop(columns='text'); s=pd.read_parquet('student_scores_PRIVATE.parquet')
for k in C.SIG:
    p=np.clip(s[f's_{k}'].to_numpy(),1e-9,1-1e-9); w[f'lg_{k}']=np.log(p/(1-p))
w=w[w.user_id.isin(users)]; cuts=C.calibrated_cuts(w)
w=w[~w.disclosure].sort_values(['user_id','ts_sec'],kind='mergesort').reset_index(drop=True); w=C.per_writing(w,cuts)
rng=np.random.default_rng(0); bad=0; checked=0
for spec in C.SPECS:
    df=pd.read_parquet(f'chunksW_{spec}_PRIVATE.parquet')
    for uid in rng.choice(df.user_id.unique(),8,replace=False):
        x=w[w.user_id==uid]; ts=(x.ts_sec.to_numpy()-x.ts_sec.iloc[0])/86400
        for r in df[df.user_id==uid].itertuples():
            k=r.k; checked+=1
            if ts[k-1]>r.day+1e-6: bad+=1
            if k<len(ts) and ts[k]<r.day-1e-6: bad+=1
            for q in ('lg_self_blame','emo_sadness','distressed','polarity'):
                if not np.isclose(getattr(r,f'pre_{q}'),x[q].to_numpy()[:k].mean()): bad+=1
assert bad==0, f'{bad} violations'
print(f'PASS: {checked} decision rows across 6 specs, 0 future-information violations')
