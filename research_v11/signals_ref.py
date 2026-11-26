"""Reference distributions for the interpretable transformer signals (dev users only, label-free).

  writing level : mean / sd of each symptom similarity (bge, 21) and go_emotions probability (28)
                  over dev writings -> per-writing z-scores ("this writing reads unusually like 'hopelessness'")
  history level : 1001 quantiles of each signal's history-so-far mean over dev chunk-end rows (d14)
                  -> a user's percentile at a decision point
Writes demo/artifacts/signals_ref.json (aggregates only).
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'demo')); sys.path.insert(0, HERE)
from common import ART, RQ, load_split, load_writings, prepare   # noqa: E402
from views import Views                                           # noqa: E402
from v11 import load_matrix                                       # noqa: E402

users = load_split(); dev = set(users.user_id[users.split == 'train'])
meta = json.load(open(RQ / 'chunk_meta.json'))
w = load_writings(dev); w['orig'] = w.index.to_numpy(); w = prepare(w, meta['cuts'])
A = pd.read_parquet(RQ / 'chunks_d14_PRIVATE.parquet'); A['user_id'] = A.user_id.astype(str)
V = Views(w.user_id.to_numpy(), A); IA = V.index(A)
sym = list(json.load(open(os.path.join(HERE, 'symptoms.json')))['symptoms'])
emo = json.load(open(os.path.join(ROOT, 'private', 'encoder', 'goemo_meta.json')))['labels']
out = {'symptoms': sym, 'emotions': emo}
for name, labels in (('BDI', sym), ('EMO', emo)):
    M = load_matrix(name, w.orig.to_numpy())
    _, _, pre = Views.compute(M, IA)
    out[name] = {'writing_mean': M.mean(0).tolist(), 'writing_sd': M.std(0).tolist(),
                 'history_quantiles': np.quantile(pre, np.linspace(0, 1, 1001), axis=0).T.tolist()}
    # label direction on dev (descriptive): user-level AUROC of the full-history mean
    from sklearn.metrics import roc_auc_score
    last = pd.DataFrame(pre, columns=labels).assign(user_id=A.user_id.to_numpy(), k=A.k.to_numpy()).sort_values('k').groupby('user_id').last()
    yy = users.set_index('user_id').label.reindex(last.index)
    out[name]['dev_auroc'] = {l: float(roc_auc_score(yy, last[l])) for l in labels}
json.dump(out, open(ART / 'signals_ref.json', 'w'))
for name in ('BDI', 'EMO'):
    a = out[name]['dev_auroc']; top = sorted(a.items(), key=lambda t: -t[1])
    print(name, 'highest dev AUROC:', [(k, round(v, 3)) for k, v in top[:8]], '| lowest:', [(k, round(v, 3)) for k, v in top[-3:]])
