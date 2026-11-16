"""Professor's 2-Nov data checks, development users only (418). Descriptive; no model."""
import json, numpy as np, pandas as pd
import os as _os
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
B = _os.environ.get('AIXPE_ROOT') or (str(_here.parent) if (_here.parent / 'data').exists() else '/mnt/user-data/uploads/AIX PE')
rows=json.loads(open(f'{B}/private/cohort_v1/split.json').read())['rows']
dev=pd.DataFrame([{'user_id':str(r['user_id']),'label':int(r['label']),'pilot':bool(r['pilot_exposed'])} for r in rows if r['split']=='train'])
w=pd.read_parquet('writings_PRIVATE.parquet',columns=['user_id','ts_sec','disclosure'])
w=w[w.user_id.isin(set(dev.user_id))]
w['day']=(w.ts_sec-w.groupby('user_id').ts_sec.transform('min'))/86400
g=w.groupby('user_id')
u=pd.DataFrame({'n':g.size(),'span':g.day.max()}).join(dev.set_index('user_id'))
u['rate_full']=u.n/np.maximum(u.span,7)*7
for D in (28,84):
    u[f'rate_{D}']=w[w.day<=D].groupby('user_id').size().reindex(u.index).fillna(0)/D*7
# frequency change between consecutive 28-day windows (only windows inside the user's span)
def jumps(x):
    span=x.day.max(); nwin=int(span//28)
    if nwin<2: return pd.Series({'max_up':np.nan,'max_down':np.nan,'n_win':nwin})
    c=np.bincount((x.day[x.day<nwin*28]//28).astype(int),minlength=nwin)[:nwin]
    r=np.log((c[1:]+1)/(c[:-1]+1))
    return pd.Series({'max_up':r.max(),'max_down':-r.min(),'n_win':nwin})
J=w.groupby('user_id')[['day']].apply(jumps); u=u.join(J)
fd=w[w.disclosure].groupby('user_id').day.min(); u['first_disc']=fd.reindex(u.index)
out={}
for lab,name in ((1,'positive'),(0,'control')):
    x=u[u.label==lab]
    out[name]={'users':int(len(x)),
      'median_writings':float(x.n.median()),'mean_writings':float(x.n.mean()),
      'median_span_days':float(x.span.median()),
      'share_span_under_9d':float((x.span<9).mean()),'share_span_under_28d':float((x.span<28).mean()),
      'share_span_over_365d':float((x.span>365).mean()),
      'median_per_week_full':float(x.rate_full.median()),
      'median_per_week_first28d':float(x.rate_28.median()),'median_per_week_first84d':float(x.rate_84.median()),
      'share_with_3x_rise_between_28d_windows':float((x.max_up>=np.log(3)).mean()),
      'share_with_3x_drop_between_28d_windows':float((x.max_down>=np.log(3)).mean()),
      'users_with_2plus_28d_windows':int((x.n_win>=2).sum()),
      'share_disclosing':float(x.first_disc.notna().mean()),
      'median_first_disclosure_day':float(x.first_disc.median()) if x.first_disc.notna().any() else None,
      'share_disclosing_within_28d':float((x.first_disc<=28).mean()),
      'share_disclosing_within_84d':float((x.first_disc<=84).mean())}
# Mann-Whitney for the frequency question
from scipy.stats import mannwhitneyu
for c in ('rate_full','rate_28','rate_84','n'):
    p=mannwhitneyu(u.loc[u.label==1,c],u.loc[u.label==0,c]).pvalue; out.setdefault('mannwhitney_p',{})[c]=float(p)
# writings beyond day 84: how much of each group's history is there
out['share_of_writings_after_day84']={n:float((w.day>84)[w.user_id.isin(u.index[u.label==l])].mean()) for l,n in ((1,'positive'),(0,'control'))}
json.dump(out,open('checks.json','w'),indent=1)
print(json.dumps(out,indent=1))
