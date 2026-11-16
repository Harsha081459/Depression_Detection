"""Writing-level base table for the full-history chunked study (all 155,284 writings).
Per-writing rule features, NRC emotions, disclosure flag, Qwen scores where they exist."""
import json, re, sys, time, numpy as np, pandas as pd
import os as _os
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
_ROOT = _os.environ.get('AIXPE_ROOT') or (str(_here.parent) if (_here.parent / 'data').exists() else '/mnt/user-data/uploads/AIX PE')
BASE=_ROOT
sys.path.insert(0,f'{BASE}/scripts/analysis')
import final_indirect_pipeline as fip
t0=time.time()
raw=pd.read_csv(f'{BASE}/data/processed/erisk/task2_target_text.csv',dtype={'user_id':str})
d=fip._prepare(raw)                     # rule flags, polarity, distress, disclosure, night (IST)
d['ts_sec']=d['ts'].to_numpy().astype('datetime64[s]').astype(np.int64)
d['is_comment']=(d['kind']=='comment').astype(int)
# NRC emotions (same lexicon file as feature set 3)
lex={}
for line in open(f'{BASE}/data/raw/lexicons/nrc/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',encoding='utf-8'):
    p=line.rstrip('\n').split('\t')
    if len(p)==3 and p[2]=='1': lex.setdefault(p[0],set()).add(p[1])
EMO=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","positive","negative"]
tok=re.compile(r"[a-z]+(?:'[a-z]+)?")
def emo(text):
    c=dict.fromkeys(EMO,0); toks=tok.findall(text)
    for w in toks:
        for e in lex.get(w,()): c[e]+=1
    n=max(len(toks),1); return [c[e]/n for e in EMO]
E=np.array([emo(t) for t in d['text'].to_numpy()])
for i,e in enumerate(EMO): d[f'emo_{e}']=E[:,i]
# Qwen P(Yes) for the 62,183 scored writings: match on user + second + normalised text.
# Timestamp alone is ambiguous (same-second writings). 79 score rows are exact duplicates.
s=pd.read_parquet(f'{BASE}/private/cohort_v1/scores/part-00000000.parquet')
e=pd.read_parquet(f'{BASE}/private/cohort_v1/eligible_posts.parquet')[['user_id','post_id','text']]
s=s.merge(e,on=['user_id','post_id'])
norm=lambda x: re.sub(r'\s+',' ',str(x).lower().replace('\u2019',"'").replace('\u2018',"'")).strip()[:200]
s['ts_sec']=pd.to_datetime(s['date'],utc=True).dt.tz_localize(None).to_numpy().astype('datetime64[s]').astype(np.int64)
s['key']=s.text.map(norm); d['key']=d.text.map(norm)
SIG=['self_blame','helplessness','isolation','family_strain','emptiness']
s=s.rename(columns={k:f'q_{k}' for k in SIG})[['user_id','ts_sec','key']+[f'q_{k}' for k in SIG]]
s=s.drop_duplicates(['user_id','ts_sec','key'])
d=d.merge(s,on=['user_id','ts_sec','key'],how='left').drop(columns='key')
print('qwen matched writings',int(d['q_self_blame'].notna().sum()),flush=True)
keep=['user_id','ts_sec','is_comment','night','n_tokens','self_tokens','absolutist','distress','polarity',
      'profanity','disclosure','text']+[f'emo_{e}' for e in EMO]+[f'q_{k}' for k in SIG]
d=d[keep].sort_values(['user_id','ts_sec'],kind='mergesort').reset_index(drop=True)
d.to_parquet('writings_PRIVATE.parquet',index=False)
print('rows',len(d),'users',d.user_id.nunique(),'disclosure',int(d.disclosure.sum()),'%.0fs'%(time.time()-t0))
