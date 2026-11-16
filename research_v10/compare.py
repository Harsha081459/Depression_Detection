"""Paired, class-stratified bootstrap (2,000) between variants on identical users.
AUROC at horizons from out-of-fold scores; F1 from the pooled sequential decisions
of the latency-F1 policy. Reported on all 418 and on the 175 non-pilot users."""
import json, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import seq_eval as SE

def f1(y, f):
    tp = ((y == 1) & (f == 1)).sum(); fp = ((y == 0) & (f == 1)).sum(); fn = ((y == 1) & (f == 0)).sum()
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

def boot(a_tag, b_tag, users, n=2000, seed=42):
    out = {}
    np_ids = set(users.loc[~users.pilot, 'user_id'])
    ha, _ = SE.horizon_scores(pd.read_parquet(f'oof_{a_tag}_PRIVATE.parquet'), users)
    hb, _ = SE.horizon_scores(pd.read_parquet(f'oof_{b_tag}_PRIVATE.parquet'), users)
    da = pd.read_parquet(f'dec_{a_tag}_PRIVATE.parquet'); db = pd.read_parquet(f'dec_{b_tag}_PRIVATE.parquet')
    da = da[da.objective == 'f_latency'].set_index('user_id'); db = db[db.objective == 'f_latency'].set_index('user_id')
    rng = np.random.default_rng(seed)
    for pop, ids in (('all', users.user_id.tolist()), ('np', [u for u in users.user_id if u in np_ids])):
        y = users.set_index('user_id').label.reindex(ids).to_numpy()
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        draws = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n)]
        for h in ('k25', 'k50', 'full'):
            sa, sb = ha[h].reindex(ids).to_numpy(), hb[h].reindex(ids).to_numpy()
            pt = roc_auc_score(y, sa) - roc_auc_score(y, sb)
            bs = [roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]) for i in draws]
            out[f'{pop}_auroc_{h}'] = {'diff': float(pt), 'ci': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        fa, fb = da.flagged.reindex(ids).to_numpy(), db.flagged.reindex(ids).to_numpy()
        pt = f1(y, fa) - f1(y, fb); bs = [f1(y[i], fa[i]) - f1(y[i], fb[i]) for i in draws]
        out[f'{pop}_f1_seq'] = {'diff': float(pt), 'ci': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    return out

if __name__ == '__main__':
    users = SE.load_users()
    pairs = [p.split(':') for p in sys.argv[1:]]
    res = {f'{a}_vs_{b}': boot(a, b, users) for a, b in pairs}
    json.dump(res, open('compare.json', 'w'), indent=1)
    for k, v in res.items():
        print(k)
        for m, r in v.items():
            sep = '*' if (r['ci'][0] > 0 or r['ci'][1] < 0) else ' '
            print(f"  {m:<18} {r['diff']:+.3f}  [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {sep}")
