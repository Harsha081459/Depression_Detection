"""Post-hoc robustness: text_doc with URLs, image/markup tokens and residual
medical words also removed. Not in the v5 prespec; reported as post hoc."""
import sys, re, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import run_v5, v5_features as v5

URL = re.compile(r'https?://\S+|www\.\S+|\S+\.(?:com|org|net|it|png|jpg|jpeg|gif|webp)\b|&amp;#x200b;|x200b|&amp;|\[|\]|\(|\)', re.I)
EXTRA = re.compile(r'\b(?:dose\w*|dr|mg|doctor\w*|pill\w*|prescri\w*|hospital\w*|clinic\w*|psych\w*|redd|preview|webp|png|jpg|gif|auto|format|width|imgur|youtube|youtu)\b', re.I)
def strict(t): return EXTRA.sub(' ', URL.sub(' ', t))

w, u, t = run_v5.load(Path(__file__).resolve().parents[2])
t['text_noclin'] = t.text_noclin.map(strict)
t['text_clin'] = t.text_noclin
ids = [str(x) for x in u.user_id]
y = u.set_index('user_id').label.astype(int).loc[ids]
pilot = set(u.loc[u.pilot_exposed, 'user_id'].astype(str))
rows = []
for day in (14, 28, 84):
    td = v5.TextDay(t, day, ids)
    X = td.doc['noclin']
    for pos, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(np.zeros(len(y)), y)):
        C = v5.tune_doc_C(X[tr], y.iloc[tr].to_numpy())
        m = v5.TextModel(3, C).fit(X[tr], y.iloc[tr].to_numpy())
        s = m.score(X[va])
        rows += [dict(repeat=pos // 5, decision_day=day, user_id=ids[i], label=int(y.iloc[i]), pilot=ids[i] in pilot, score=float(v))
                 for i, v in zip(va, s)]
    print('day', day, 'done', flush=True)
f = pd.DataFrame(rows)
f.to_parquet('' + str(Path(__file__).resolve().parents[2] / 'private' / 'cohort_v5') + '/posthoc_strict_text_PRIVATE.parquet', index=False)
out = []
for (rep, day), g in f.groupby(['repeat', 'decision_day']):
    for pop, sub in (('all_users', g), ('non_pilot', g[~g.pilot])):
        out.append(dict(repeat=rep, decision_day=day, population=pop, auroc=roc_auc_score(sub.label, sub.score),
                        auprc=average_precision_score(sub.label, sub.score)))
s = pd.DataFrame(out).groupby(['population', 'decision_day']).agg(mean_auroc=('auroc', 'mean'), sd_auroc=('auroc', 'std'), mean_auprc=('auprc', 'mean')).reset_index()
s.to_csv('' + str(Path(__file__).resolve().parents[2] / 'private' / 'cohort_v5') + '/posthoc_strict_text.csv', index=False)
print(s.round(3).to_string(index=False))
