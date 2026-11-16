"""Artefact checks for the text ensemble (dev set, day 28). No new model fitting:
uses the cached cross-fitted fold scores of trial R2.23."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_research as R

root, out = Path(sys.argv[1]), Path(sys.argv[2])
ids, y, pilot = R.load_users(out, root)
parts = R.partitions(y)
text = json.loads((out / 'folds' / 'text_scores_day28.json').read_text())
blocks = pd.read_parquet(out / 'cache' / 'blocks_day28.parquet').loc[ids]
rows = []
for pos, (tr, va) in enumerate(parts):
    r = np.mean([pd.Series(text[pos][m]['valid']).rank(pct=True) for m in ('word', 'char', 'nb_word')], axis=0)
    r = pd.Series(r).rank(pct=True).to_numpy()
    rows += [dict(repeat=pos // 5, user_id=ids[i], label=int(y.iloc[i]), pilot=bool(pilot[i]), score=float(s)) for i, s in zip(va, r)]
f = pd.DataFrame(rows)
f = f.merge(blocks[['art__url_per_post', 'art__nonascii_share', 'v1__n_posts', 'psy__i_sing']].reset_index(), on='user_id')
res = {}
def auc(g): return float(roc_auc_score(g.label, g.score)) if g.label.nunique() == 2 else float('nan')
for name, col in (('url_rate', 'art__url_per_post'), ('nonascii', 'art__nonascii_share'), ('n_posts', 'v1__n_posts')):
    cut = f.groupby('repeat')[col].transform('median')
    lo, hi = f[f[col] <= cut], f[f[col] > cut]
    res[name] = {'auroc_low_half': np.mean([auc(g) for _, g in lo.groupby('repeat')]),
                 'auroc_high_half': np.mean([auc(g) for _, g in hi.groupby('repeat')]),
                 'spearman_score_vs_feature': float(f[['score', col]].corr('spearman').iloc[0, 1])}
res['users_with_zero_url'] = {'auroc': np.mean([auc(g) for _, g in f[f.art__url_per_post == 0].groupby('repeat')]),
                              'n': int((f[f.repeat == 0].art__url_per_post == 0).sum())}
print(json.dumps(res, indent=2))
(out / 'artefact_check_day28.json').write_text(json.dumps(res, indent=2))
