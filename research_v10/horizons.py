"""Sequential horizons 84 / 128 / 256 days / all writings, same combined model.
View A: all 418 dev users. View B (length-bias control): a fixed panel of users whose
history runs past 256 days, so at 84/128/256 every panel user, positive or control,
is observed for the full horizon. Mean +- SD over 5 repeats."""
import json, numpy as np, pandas as pd, seq_eval as SE
from sklearn.metrics import roc_auc_score
u=SE.load_users(); npset=set(u.loc[~u.pilot,'user_id']); lab=u.set_index('user_id').label
w=pd.read_parquet('writings_PRIVATE.parquet',columns=['user_id','ts_sec','disclosure'])
w=w[w.user_id.isin(set(u.user_id))&~w.disclosure]
span=((w.groupby('user_id').ts_sec.max()-w.groupby('user_id').ts_sec.min())/86400).reindex(u.user_id)
panel=set(span[span>256].index); panel512=set(span[span>512].index)
H={'84 days':'cap84_B','128 days':'cap128_B','256 days':'cap256_B','512 days':'cap512_B','all writings':'B'}
def conf(d,ids):
    d=d[d.user_id.isin(ids)]; tp=((d.y==1)&(d.flagged==1)).sum(); fp=((d.y==0)&(d.flagged==1)).sum(); fn=((d.y==1)&(d.flagged==0)).sum()
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0; F=2*tp/(2*tp+fp+fn) if tp else 0.0
    return F,P,R
def auc(o,ids):
    s=o.sort_values(['user_id','k']).groupby('user_id').tail(1).set_index('user_id').p
    ids=[i for i in ids]; s=s.reindex(ids).fillna(s.min()-1); y=lab.reindex(ids)
    return roc_auc_score(y,s)
pops={'A all':set(u.user_id),'A np':npset,'B panel':panel,'B panel np':panel&npset,'C panel512':panel512,'C panel512 np':panel512&npset}
print('panel users',len(panel),'positives',int(lab[list(panel)].sum()),'| np panel',len(panel&npset),'positives',int(lab[list(panel&npset)].sum()))
out={}
for h,tag in H.items():
    row={}
    for pn,ids in pops.items():
        M=[]
        for r in range(5):
            d=pd.read_parquet(f'dec_{tag}_r{r}_PRIVATE.parquet'); d=d[d.objective=='f_latency']
            o=pd.read_parquet(f'oof_{tag}_r{r}_PRIVATE.parquet')
            M.append((*conf(d,ids),auc(o,ids)))
        M=np.array(M); row[pn]={'F1':M[:,0].mean(),'F1_sd':M[:,0].std(),'P':M[:,1].mean(),'R':M[:,2].mean(),'AUROC':M[:,3].mean(),'AUROC_sd':M[:,3].std()}
    rs=[json.load(open(f'res_{tag}_r{r}.json')) for r in range(5)]
    row['ERDE50 all']=float(np.mean([x['selected']['early_all']['erde50']['erde50'] for x in rs]))
    row['writings kept']=None
    out[h]=row
json.dump({'panel_n':len(panel),'panel_pos':int(lab[list(panel)].sum()),'panel_np_n':len(panel&npset),'panel_np_pos':int(lab[list(panel&npset)].sum()),'horizons':out},open('horizons.json','w'),indent=1,default=float)
for pn in pops:
    print(f'\n{pn}')
    for h in H:
        x=out[h][pn]; print(f"  {h:<13} F1 {x['F1']:.3f}±{x['F1_sd']:.3f}  P {x['P']:.3f}  R {x['R']:.3f}  AUROC {x['AUROC']:.3f}±{x['AUROC_sd']:.3f}")
print('\nERDE50 all:',{h:round(out[h]['ERDE50 all'],3) for h in H})
