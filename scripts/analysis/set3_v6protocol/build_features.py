"""Rebuild set-3 prefix features for three disclosure arms, replicating
erisk_sequential_features.py exactly, plus per-writing artefact columns."""
import json, re, sys, time
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

BASE='/mnt/user-data/uploads/AIX PE'
OUT='/tmp/erisk_scratch/erisk'
sys.path.insert(0, f'{BASE}/scripts/analysis')
from erisk_disclosure_audit import STRICT, BROAD_EXTRA
from final_indirect_pipeline import CARE, ATTEMPT

TZ='Asia/Kolkata'; NIGHT_HOURS=set(range(0,6)); WEEKEND_DOW={5,6}
RATE_CHANGE_MIN_WRITINGS=6; RATE_CHANGE_MIN_SPAN_DAYS=1.0; RATE_SPAN_FLOOR_DAYS=1.0
TOKEN_PATTERN=r"(?u)\b[a-z][a-z'-]*[a-z]\b|\b[a-z]\b"
EMOTIONS=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","positive","negative"]
CLINICAL_CANDIDATES=["depression","depressed","depressing","depress","depressive","antidepressant","anxiety","anxious","suicide","suicidal","therapy","therapist","psychiatrist","psychiatric","psychiatry","psychologist","psychology","medication","medicine","diagnosis","diagnosed","diagnose","disorder","mental","illness","bipolar","insomnia","counseling","counselor","prescription","pill","pills","hospital","doctor","symptom","symptoms","disease","sick","sickness"]
URL_RE=re.compile(r'https?://|www\.')

def load_nrc(path):
    lex={}
    for line in open(path,encoding='utf-8'):
        p=line.rstrip('\n').split('\t')
        if len(p)!=3: continue
        w,e,f=p
        if f=='1' and e in EMOTIONS: lex.setdefault(w.lower(),set()).add(e)
    return lex

def count_emotions(texts, lex, prefix):
    vocab=sorted(lex)
    vec=CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=TOKEN_PATTERN)
    rows,cols=[],[]
    for j,w in enumerate(vocab):
        for e in lex[w]: rows.append(j); cols.append(EMOTIONS.index(e))
    ind=csr_matrix((np.ones(len(rows),dtype=np.int32),(rows,cols)),shape=(len(vocab),len(EMOTIONS)))
    x=vec.transform(texts.tolist())
    c=np.asarray((x@ind).todense(),dtype=np.int32)
    return pd.DataFrame(c,columns=[f'{prefix}_{e}_cnt' for e in EMOTIONS],index=texts.index)

def _entropy_bits_rows(cum):
    tot=cum.sum(axis=1,keepdims=True).astype(float)
    p=np.divide(cum,tot,out=np.zeros_like(cum,dtype=float),where=tot>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        lp=np.where(p>0,np.log2(p),0.0)
    return -(p*lp).sum(axis=1)

def user_prefix_clocks(g):
    n=len(g)
    ts=g['ts_sec'].to_numpy(dtype=np.int64); hour=g['hour'].to_numpy(dtype=np.int64)
    night=g['night'].to_numpy(dtype=np.int64); weekend=g['weekend'].to_numpy(dtype=np.int64)
    is_comment=g['is_comment'].to_numpy(dtype=np.int64)
    t=np.arange(1,n+1,dtype=np.int64)
    hours_since_first=(ts-ts[0])/3600.0; days_since_first=hours_since_first/24.0
    gap_hours=np.zeros(n,dtype=float); gap_hours[1:]=np.diff(ts)/3600.0
    span_floor=np.maximum(days_since_first,RATE_SPAN_FLOOR_DAYS)
    posts_per_week=t/span_floor*7.0
    k_gaps=np.maximum(t-1,1); gaps=gap_hours[1:]
    median_gap=np.zeros(n,dtype=float)
    if n>1: median_gap[1:]=pd.Series(gaps).expanding().median().to_numpy()
    cs=np.concatenate([[0.0],np.cumsum(gaps)]); css=np.concatenate([[0.0],np.cumsum(gaps**2)])
    mean_gap=cs/k_gaps; var_gap=np.maximum(css/k_gaps-mean_gap**2,0.0)
    gap_cv=np.where(t>=3,np.sqrt(var_gap)/np.maximum(mean_gap,1e-9),0.0)
    burst=np.concatenate([[0],(gaps<1.0).astype(np.int64)])
    burst_1h=np.where(t>=2,np.cumsum(burst)/k_gaps,0.0)
    onehot=np.zeros((n,24),dtype=np.int64); onehot[np.arange(n),hour]=1
    hour_entropy=_entropy_bits_rows(np.cumsum(onehot,axis=0))
    mid=ts[0]+(ts-ts[0])/2.0
    n1=np.searchsorted(ts,mid,side='right'); n2=t-n1
    ratio=(n2+0.5)/(n1+0.5)
    trusted=(t>=RATE_CHANGE_MIN_WRITINGS)&(days_since_first>=RATE_CHANGE_MIN_SPAN_DAYS)
    rate_change=np.where(trusted,ratio,1.0)
    n_last7d=t-np.searchsorted(ts,ts-7*86400,side='right')
    n_last28d=t-np.searchsorted(ts,ts-28*86400,side='right')
    recent_rate_ratio=n_last7d/np.maximum(posts_per_week,1e-9)
    return pd.DataFrame({'t':t,'days_since_first':days_since_first,'gap_hours':gap_hours,
        'posts_per_week_so_far':posts_per_week,'night_frac_so_far':np.cumsum(night)/t,
        'weekend_frac_so_far':np.cumsum(weekend)/t,'comment_frac_so_far':np.cumsum(is_comment)/t,
        'median_gap_so_far':median_gap,'gap_cv_so_far':gap_cv,'burst_1h_frac_so_far':burst_1h,
        'hour_entropy_so_far':hour_entropy,'rate_change_so_far':rate_change,
        'n_last7d':n_last7d,'n_last28d':n_last28d,'recent_rate_ratio':recent_rate_ratio},index=g.index)

def cumulative_emotions(df,prefix):
    grp=df.groupby('user_id',sort=False)
    cum_tokens=grp['n_tokens'].cumsum().to_numpy(dtype=float)
    out={}
    for e in EMOTIONS:
        c=grp[f'{prefix}_{e}_cnt'].cumsum().to_numpy(dtype=float)
        out[f'cum_{prefix}_{e}']=np.divide(c,cum_tokens,out=np.zeros_like(c),where=cum_tokens>0)
    return pd.DataFrame(out,index=df.index)

def build_arm(base, arm):
    if arm=='unmasked': d=base
    elif arm=='masked': d=base[~base['disclosure']]
    else:
        first=base[base['disclosure']].groupby('user_id')['ts_sec'].min()
        cut=base['user_id'].map(first)
        d=base[cut.isna()|(base['ts_sec']<cut)]
    d=d.sort_values(['user_id','ts_sec'],kind='mergesort').reset_index(drop=True)
    clocks=d.groupby('user_id',sort=False,group_keys=False)[['ts_sec','hour','night','weekend','is_comment']].apply(user_prefix_clocks)
    d=pd.concat([d,clocks],axis=1)
    d=pd.concat([d,cumulative_emotions(d,'emo'),cumulative_emotions(d,'emonc')],axis=1)
    grp=d.groupby('user_id',sort=False)
    d['mean_log_tokens_so_far']=grp['n_tokens'].transform(lambda s: np.log1p(s).cumsum())/d['t']
    # artefact cumulative shares
    d['url_frac_so_far']=grp['has_url'].cumsum()/d['t']
    d['nonascii_frac_so_far']=grp['nonascii_cnt'].cumsum()/np.maximum(grp['n_chars'].cumsum(),1)
    return d

def main():
    t0=time.time()
    text=pd.read_csv(f'{BASE}/data/processed/erisk/task2_target_text.csv')
    labels=pd.read_csv(f'{BASE}/data/processed/erisk/task2_user_l2_features.csv',usecols=['user_id','label'])
    print('writings',len(text),'users',text.user_id.nunique(),flush=True)
    dt=pd.to_datetime(text['date'],utc=True)
    text['ts_sec']=dt.dt.tz_convert(None).to_numpy().astype('datetime64[s]').astype(np.int64)
    loc=dt.dt.tz_convert(TZ)
    text['hour']=loc.dt.hour.astype(np.int64); text['dow']=loc.dt.dayofweek.astype(np.int64)
    text['night']=text['hour'].isin(NIGHT_HOURS).astype(np.int64)
    text['weekend']=text['dow'].isin(WEEKEND_DOW).astype(np.int64)
    text['is_comment']=(text['kind']=='comment').astype(np.int64)
    text=text.sort_values(['user_id','ts_sec'],kind='mergesort').reset_index(drop=True)

    raw=(text['title'].fillna('')+' '+text['body'].fillna('')).str.strip()
    low=raw.str.lower().str.replace('’',"'",regex=False).str.replace('‘',"'",regex=False)
    text['disclosure']=low.str.contains(STRICT)|low.str.contains(BROAD_EXTRA)|low.str.contains(CARE)|low.str.contains(ATTEMPT)
    text['has_url']=raw.str.contains(URL_RE).astype(np.int64)
    text['n_chars']=raw.str.len().astype(np.int64)
    text['nonascii_cnt']=raw.map(lambda s: sum(1 for ch in s if ord(ch)>127)).astype(np.int64)
    print('disclosure writings',int(text['disclosure'].sum()),flush=True)
    dz=text.merge(labels,on='user_id',how='left')
    print('  by label:',dz.groupby('label')['disclosure'].sum().to_dict(),flush=True)

    lex=load_nrc(f'{BASE}/data/raw/lexicons/nrc/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    clinical=sorted(w for w in CLINICAL_CANDIDATES if w in lex)
    lex_nc={w:s for w,s in lex.items() if w not in set(clinical)}
    print('lexicon',len(lex),'clinical removed',len(clinical),flush=True)
    emo=count_emotions(raw,lex,'emo'); emonc=count_emotions(raw,lex_nc,'emonc')
    analyzer=CountVectorizer(lowercase=True,token_pattern=TOKEN_PATTERN).build_analyzer()
    text['n_tokens']=np.fromiter((len(analyzer(d)) for d in raw.tolist()),dtype=np.int64,count=len(raw))
    text=text.drop(columns=['title','body','author'])
    text=pd.concat([text,emo,emonc],axis=1)
    text=text.merge(labels,on='user_id',how='left',validate='many_to_one')
    assert text['label'].notna().all()
    text['label']=text['label'].astype(np.int64)
    print('emotions done %.0fs'%(time.time()-t0),flush=True)

    for arm in ('unmasked','masked','before_disclosure'):
        d=build_arm(text,arm)
        d.to_parquet(f'{OUT}/prefix_{arm}.parquet',index=False)
        print(f'{arm}: rows {len(d)} users {d.user_id.nunique()}  %.0fs'%(time.time()-t0),flush=True)

main()
