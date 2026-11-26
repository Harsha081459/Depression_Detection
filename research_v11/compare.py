"""Compare v11 candidates (aggregates only). python research_v11/compare.py [policy=prev_adj]

Per candidate (5 repeats): mean F1 / precision / recall / ERDE (all, NP); AUROC and AP from repeat-averaged
end-of-history rank scores (as in research_v10 step 2); selection score = mean of (F1_all + F1_NP) / 2.
Paired user-level class-stratified bootstrap (2,000) against B for AUROC, AP (rank scores) and F1 (per-user
flags averaged over repeats is not an F1, so F1 differences are bootstrapped per repeat and averaged).
Writes research_v11/results/compare_<policy>.json.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'demo'))
from common import SE  # noqa: E402

EXPLORATORY = {'BDI', 'EMO+BDI'}          # added after results; never eligible for selection
OUT = os.path.join(ROOT, 'private', 'v11'); POL = sys.argv[1] if len(sys.argv) > 1 else 'prev_adj'
U = SE.load_users(); y = U.set_index('user_id').label; NP = set(U.user_id[~U.pilot])


def tags():
    t = {}
    for f in glob.glob(os.path.join(OUT, f'res_*_{POL}_r*.json')):
        r = json.load(open(f)); t.setdefault(r['candidate'], {})[r['rep']] = os.path.basename(f)[4:-5]
    return {c: v for c, v in t.items() if len(v) == 5}


def rank_score(tag):
    o = pd.read_parquet(os.path.join(OUT, f'oof_{tag}_PRIVATE.parquet'))
    s = o.sort_values(['user_id', 'k']).groupby('user_id').p.last().reindex(U.user_id)
    return s.fillna(s.min() - 1).rank(pct=True)


def f1(yy, ff):
    tp = int(((yy == 1) & (ff == 1)).sum()); fp = int(((yy == 0) & (ff == 1)).sum()); fn = int(((yy == 1) & (ff == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    global U, y, NP
    T = tags(); out = {'policy': POL, 'candidates': {}}
    if 'ALL' in POL:     # all 523 users
        from common import load_split
        U = load_split()[['user_id', 'label', 'pilot']]; y = U.set_index('user_id').label; NP = set(U.user_id[~U.pilot])
        out['n_users'] = len(U); out['n_pos'] = int(U.label.sum())
    if 'BM' in POL:      # matched pre-disclosure set: users with no writings left are excluded (MISSING=exclude)
        have = set(pd.read_parquet(os.path.join(OUT, f'oof_{next(iter(T.values()))[0]}_PRIVATE.parquet')).user_id)
        U = U[U.user_id.isin(have)].reset_index(drop=True); y = U.set_index('user_id').label; NP = set(U.user_id[~U.pilot])
        out['n_users'] = len(U); out['n_pos'] = int(U.label.sum())
    S, FL = {}, {}
    for c, reps in sorted(T.items()):
        R = [json.load(open(os.path.join(OUT, f'res_{reps[r]}.json'))) for r in range(5)]
        S[c] = pd.concat([rank_score(reps[r]) for r in range(5)], axis=1).mean(axis=1)
        FL[c] = [pd.read_parquet(os.path.join(OUT, f'dec_{reps[r]}_PRIVATE.parquet')).set_index('user_id').flagged.reindex(U.user_id).fillna(0).astype(int) for r in range(5)]
        d = {}
        for pop, key, ids in (('all', 'early_all', list(U.user_id)), ('np', 'early_np', sorted(NP))):
            for k in ('f1', 'precision', 'recall', 'erde5', 'erde50', 'f_latency'):
                d[f'{k}_{pop}'] = float(np.mean([x[key][k] for x in R]))
            d[f'f1_{pop}_reps'] = [round(x[key]['f1'], 3) for x in R]
            d[f'auroc_{pop}'] = float(roc_auc_score(y.reindex(ids), S[c].reindex(ids)))
            d[f'ap_{pop}'] = float(average_precision_score(y.reindex(ids), S[c].reindex(ids)))
        d['flagged_mean'] = float(np.mean([f.sum() for f in FL[c]]))
        d['selection_score'] = (d['f1_all'] + d['f1_np']) / 2
        d['eligible'] = c not in EXPLORATORY
        out['candidates'][c] = d
    if 'B' in S:
        rng = np.random.default_rng(42); out['vs_B'] = {}
        for pop, ids in (('all', list(U.user_id)), ('np', sorted(NP))):
            yy = y.reindex(ids).to_numpy(); p, q = np.where(yy == 1)[0], np.where(yy == 0)[0]
            D = [np.r_[rng.choice(p, len(p)), rng.choice(q, len(q))] for _ in range(2000)]
            for c in S:
                if c == 'B':
                    continue
                a, b = S[c].reindex(ids).to_numpy(), S['B'].reindex(ids).to_numpy()
                fa = [f.reindex(ids).to_numpy() for f in FL[c]]; fb = [f.reindex(ids).to_numpy() for f in FL['B']]
                da = [roc_auc_score(yy[i], a[i]) - roc_auc_score(yy[i], b[i]) for i in D]
                dp = [average_precision_score(yy[i], a[i]) - average_precision_score(yy[i], b[i]) for i in D]
                df = [np.mean([f1(yy[i], x[i]) - f1(yy[i], z[i]) for x, z in zip(fa, fb)]) for i in D[:1000]]
                ci = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                out['vs_B'][f'{c}|{pop}'] = {'auroc': ci(da), 'ap': ci(dp), 'f1': ci(df)}
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', f'compare_{POL}.json'), 'w'), indent=1)
    rows = [(c, d['selection_score'], d['f1_all'], d['f1_np'], d['auroc_all'], d['auroc_np'], d['ap_all'], d['ap_np'], d['flagged_mean'])
            for c, d in out['candidates'].items()]
    print(f'{"candidate":28s} score  F1all  F1np  AUCall AUCnp  APall  APnp  flagged')
    for r in sorted(rows, key=lambda r: -r[1]):
        print(f'{r[0] + (" (expl.)" if r[0] in EXPLORATORY else ""):28s} ' + '  '.join(f'{v:.3f}' for v in r[1:8]) + f'  {r[8]:.0f}')
    el = {c: d for c, d in out['candidates'].items() if d['eligible']}
    if 'B' in el:
        best = max(el, key=lambda c: el[c]['selection_score'])
        guard = {c: el[c]['auroc_np'] >= el['B']['auroc_np'] - 0.01 for c in el}
        ok = [c for c in el if guard[c]]; best = max(ok, key=lambda c: el[c]['selection_score'])
        near = [c for c in ok if el[c]['selection_score'] >= el[best]['selection_score'] - 0.01]
        out['selected'] = {'best_score': best, 'within_0.01': near, 'guard_failed': [c for c in el if not guard[c]]}
        print('SELECTION (pre-registered rule):', out['selected'])
        json.dump(out, open(os.path.join(HERE, 'results', f'compare_{POL}.json'), 'w'), indent=1)
    for k, v in out.get('vs_B', {}).items():
        print(k, {m: [round(x, 3) for x in v[m]] for m in v})


if __name__ == '__main__':
    main()
