"""Prespecified 5x5 nested user CV. Descriptive uncertainty, not 25 independent folds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from erisk_full_pipeline import SEED,SIGNALS,SelectSignals,folds,choose_feature


def repeated_cv(feature_banks, labels, output, population='all_users'):
    output=Path(output)
    if output.exists(): raise RuntimeError('Repeated CV output exists; do not adapt and rerun after inspection')
    output.mkdir(parents=True)
    y=labels.astype(int)
    if y.index.has_duplicates or (population=='all_users' and len(y)!=523):
        raise RuntimeError('Invalid user population')
    if population not in ('all_users','without_pilot'): raise RuntimeError('Unknown prespecified population')
    outer=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED)
    partitions=list(outer.split(np.zeros(len(y)),y))
    metrics=[]; predictions=[]
    for weeks,X in feature_banks.items():
        X=X.loc[y.index]
        for outer_index,(tr,va) in enumerate(partitions):
            repeat,fold=divmod(outer_index,5)
            xt,xv=X.iloc[tr],X.iloc[va]; yt,yv=y.iloc[tr],y.iloc[va]
            # Each signal's aggregation and cut selected in outer training only.
            for signal in SIGNALS:
                chosen,_=choose_feature(xt,yt,signal)
                pred=xv[chosen].to_numpy()
                metrics.append(dict(weeks=weeks,repeat=repeat,fold=fold,model='single_signal',
                    ablation=signal,auroc=roc_auc_score(yv,pred),auprc=average_precision_score(yv,pred),
                    aggregation=chosen))
            for mode in ('code','llm','both','count'):
                specs=[('logistic',LogisticRegression(solver='liblinear',max_iter=2000,random_state=SEED),
                        {'model__C':[.01,.1,1.,10.]}),
                       ('boosting',GradientBoostingClassifier(random_state=SEED),
                        {'model__n_estimators':[50,100],'model__max_depth':[1,2],'model__learning_rate':[.03,.1]})]
                for name,model,grid in specs:
                    # Aggregation/cut selection is fitted INSIDE every inner CV fold.
                    pipe=Pipeline([('select',SelectSignals(mode)),('scale',StandardScaler()),('model',model)])
                    fit=GridSearchCV(pipe,grid,cv=list(folds(yt)),scoring='roc_auc',n_jobs=1,
                                     error_score='raise',refit=True).fit(xt,yt)
                    pred=fit.predict_proba(xv)[:,1]
                    metrics.append(dict(weeks=weeks,repeat=repeat,fold=fold,model=name,ablation=mode,
                        auroc=roc_auc_score(yv,pred),auprc=average_precision_score(yv,pred),
                        parameters=json.dumps(fit.best_params_,sort_keys=True)))
                    predictions.extend(dict(user_id=uid,weeks=weeks,repeat=repeat,fold=fold,
                        model=name,ablation=mode,label=int(label),prediction=float(value))
                        for uid,label,value in zip(yv.index,yv,pred))
            # Never print test numbers during execution.
            pd.DataFrame(metrics).to_csv(output/'fold_metrics.csv',index=False)
    frame=pd.DataFrame(metrics)
    summary=frame.groupby(['weeks','model','ablation']).agg(
        mean_auroc=('auroc','mean'),fold_sd_auroc=('auroc','std'),
        mean_auprc=('auprc','mean'),fold_sd_auprc=('auprc','std'),folds=('auroc','size')).reset_index()
    summary['population']=population
    summary['users']=len(y)
    summary['positive_users']=int(y.sum())
    summary.to_csv(output/'headline_cv.csv',index=False)
    pd.DataFrame(predictions).to_parquet(output/'out_of_fold_predictions_PRIVATE.parquet',index=False)
    return summary


def truncation_audit(posts,origins,labels,max_chars=2000):
    rows=[]
    for weeks in (4,8,12):
        prefix=posts[posts.date<=posts.user_id.map(origins)+pd.Timedelta(weeks=weeks)].copy()
        prefix['label']=prefix.user_id.map(labels)
        prefix['chars']=prefix.text.str.len()
        for label,group in prefix.groupby('label'):
            size=group.chars.to_numpy()
            rows.append(dict(weeks=weeks,label=int(label),posts=len(size),
                truncated_posts=int((size>max_chars).sum()),truncated_fraction=float((size>max_chars).mean()),
                median_original_chars=float(np.median(size)),max_chars=max_chars,
                mean_fraction_retained=float(np.minimum(size,max_chars).sum()/max(1,size.sum()))))
    return pd.DataFrame(rows)
