"""The professor's robustness check: change one feature, does the flag change?

For each outer fold (repeat 0), refit the configuration that fold selected (spec, C)
and its latency-F1 threshold from the inner CV. Then, for each base quantity q, shift
q by +1 and -1 training-row SD in the test rows (current chunk, previous chunk and
prefix together, so deltas stay consistent) and recompute every test user's flag.
Report the share of test users whose final flag flips. Small = robust; a quantity
whose nudge flips many flags is one the model is sensitive to."""
import json, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import seq_eval as SE
from erisk_early_metrics import apply_policy, sweep_policy

def main(variant):
    res = json.load(open(f'res_{variant}_r0.json'))
    chosen = {s['fold']: (s['chosen_spec'], s['chosen_C']) for s in res['selection'] if 'chosen_spec' in s}
    users = SE.load_users(); yu = users.label.to_numpy()
    outer = StratifiedKFold(5, shuffle=True, random_state=SE.SEED)
    Q = [q for q in SE.CQ + SE.MAXLV]
    flips = {q: {'+1sd': 0, '-1sd': 0} for q in Q}; n_test = 0; base_flags = 0
    for fold, (tr_u, te_u) in enumerate(outer.split(np.zeros(len(yu)), yu), 1):
        spec, C = chosen[fold]
        df = pd.read_parquet(f'chunks_{spec}_PRIVATE.parquet')
        tr_ids, te_ids = set(users.user_id.iloc[tr_u]), set(users.user_id.iloc[te_u])
        tr = df[df.user_id.isin(tr_ids)].reset_index(drop=True); te = df[df.user_id.isin(te_ids)].reset_index(drop=True)
        enc = SE.Encoder(variant).fit(tr); Xtr = enc.transform(tr); ytr = tr.label.to_numpy()
        iu = users.iloc[tr_u].reset_index(drop=True)
        inner = StratifiedKFold(3, shuffle=True, random_state=SE.SEED + fold).split(np.zeros(len(iu)), iu.label)
        pin = np.zeros(len(tr))
        for itr, ite in inner:
            a = tr.user_id.isin(set(iu.user_id.iloc[itr])).to_numpy()
            pin[~a] = SE.fit_predict(Xtr[a], ytr[a], SE.row_weights(tr[a]), Xtr[~a], C)[0]
        pol, _ = sweep_policy(SE.stream(tr, pin), SE.tau_grid(pin), SE.T_MINS, ('f_latency',), c_fp=float(yu[tr_u].mean()))
        tau, tmin = pol['f_latency']
        _, m, sc = SE.fit_predict(Xtr, ytr, SE.row_weights(tr), Xtr[:5], C)
        def flags(frame):
            p = m.predict_proba(sc.transform(enc.transform(frame)))[:, 1]
            return apply_policy(SE.stream(frame, p), tau, tmin).flagged
        f0 = flags(te); base_flags += int(f0.sum()); n_test += len(f0)
        for q in Q:
            sd = float(tr[f'cur_{q}'].std())
            for sign, nm in ((1, '+1sd'), (-1, '-1sd')):
                t2 = te.copy()
                for c in ('cur', 'prv', 'pre'):
                    t2[f'{c}_{q}'] = t2[f'{c}_{q}'] + sign * sd
                    if q.startswith('maxlv_'):
                        t2[f'{c}_{q}'] = t2[f'{c}_{q}'].round().clip(0, 3)
                flips[q][nm] += int((flags(t2) != f0).sum())
        print(f'fold {fold} spec {spec} C {C} tau {tau:.4f}', flush=True)
    out = {'variant': variant, 'test_users': n_test, 'base_flags': base_flags,
           'flip_share': {q: {k: v / n_test for k, v in d.items()} for q, d in flips.items()}}
    json.dump(out, open(f'robust_{variant}.json', 'w'), indent=1)
    s = pd.DataFrame({q: {'flip_+1sd': d['+1sd'], 'flip_-1sd': d['-1sd']} for q, d in out['flip_share'].items()}).T
    s['max'] = s.max(axis=1); print(s.sort_values('max', ascending=False).round(3).to_string())

if __name__ == '__main__':
    main(sys.argv[1])
