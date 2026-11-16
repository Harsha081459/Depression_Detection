"""Sequential early detection over the full history, development users only (418).

Outer: StratifiedKFold(5, shuffle, seed 42) over the split.json user order. Same folds
as the earlier set-3 sequential run, so numbers are comparable.
Inside each outer training fold (3-fold user-disjoint inner CV):
  - chunk spec (d7/d14/d28/p10/p25/p50) and regularisation C are chosen by inner
    latency-weighted F1;
  - the flag threshold tau and minimum writings t_min are tuned per objective.
Nothing about the test fold, the non-pilot labels or the 105 held-out users is used.

Variants (same information sources, different encodings):
  C      continuous: current chunk, whole prefix, change from previous chunk
  O      professor's encoding: 0-3 levels and lower/same/higher changes, all categorical
  B      both
  count  writings seen only (artefact floor)
  set3   lexicon + rhythm prefix features (earlier best family) sampled at the same rows
Run:  python seq_eval.py <variant>
"""
import json, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import os as _os
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
B = _os.environ.get('AIXPE_ROOT') or (str(_here.parent) if (_here.parent / 'data').exists() else '/mnt/user-data/uploads/AIX PE')
sys.path.insert(0, f'{B}/scripts/analysis')
from erisk_early_metrics import Decisions, apply_policy, summarize, sweep_policy
import chunks as CH

SEED = 42
SPECS = list(CH.SPECS)
CS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0)
OBJ = ('f1', 'f_latency', 'erde5', 'erde50')
T_MINS = (1, 3, 5, 10)
SIG = CH.SIG
LLM_C = [f'lg_{k}' for k in SIG] + ['distressed']
RULE = ['absolutist', 'polarity', 'flip', 'abs_jump', 'first_person', 'profanity', 'polarity_sd']
EMOQ = [f'emo_{e}' for e in CH.EMO]
BEHAV = ['night', 'log_rate', 'log_med_gap', 'silence_after_distress']
EMBQ = [f'emb{i:02d}' for i in range(int(_os.environ.get('EMBDIM', '0')))]   # frozen encoder (optional)
CQ = LLM_C + RULE + EMOQ + BEHAV + EMBQ          # continuous quantities
GROUPS = {'llm': LLM_C + [f'maxlv_{k}' for k in SIG], 'rule': RULE, 'emotion': EMOQ, 'behaviour': BEHAV, 'embed': EMBQ}
OQ = ['distressed'] + RULE + EMOQ + BEHAV        # quantities given quartile levels in O
MAXLV = [f'maxlv_{k}' for k in SIG]              # LLM ordinal grades 0-3 (O uses these, not floats)
K_H = (10, 25, 50, 100, 200)
D_H = (14, 28, 56, 84, 182, 365)
DROP = set()
import os
MAXDAY = os.environ.get('CHUNKPREF')     # 'chunkscap84' = writings capped at day 84 before chunking                                     # quantities removed (sensitivity runs)


POLICY = os.environ.get('POLICY', 'standard')


def tune_policy(ist, pin, prev_tr, np_tr, prev_np_tr):
    """Threshold policy from the inner out-of-fold stream (training users only).
    standard  maximise latency-F1 on all inner users (original).
    np_users  tune only on the non-pilot users of the training folds.
    prev_adj  tune on all inner users, but score each (tau, t_min) by the F1 it would give at
              the non-pilot positive rate of the training folds (from its TPR and FPR)."""
    if POLICY == 'np_users':
        sub = ist[ist.user_id.isin(np_tr)]
        pol, table = sweep_policy(sub, tau_grid(sub.p.to_numpy()), T_MINS, OBJ, c_fp=prev_np_tr)
        return pol, table, float(table['f_latency'].max()), float(table['f1'].max())
    pol, table = sweep_policy(ist, tau_grid(pin), T_MINS, OBJ, c_fp=prev_tr)
    if POLICY == 'prev_adj':
        lab = ist.groupby('user_id').label.first(); npos = int(lab.sum()); nneg = int(len(lab) - npos)
        tp = table['recall'] * npos; fp = table['n_flagged'] - tp
        tpr, fpr, pi = table['recall'], fp / nneg, prev_np_tr
        f1a = (2 * pi * tpr / (pi * (1 + tpr) + (1 - pi) * fpr)).fillna(0.0)
        table = table.assign(f1=f1a, f_latency=f1a * table['speed'].fillna(0.0))
        for o in ('f1', 'f_latency'):
            r = table.sort_values([o, 'tau', 't_min'], ascending=[False, False, True]).iloc[0]
            pol[o] = (float(r['tau']), int(r['t_min']))
    return pol, table, float(table['f_latency'].max()), float(table['f1'].max())


def tau_grid(p):
    qs = np.concatenate([np.arange(0.50, 0.96, 0.025), [0.97, 0.98, 0.985, 0.99, 0.995]])
    return np.unique(np.round(np.quantile(np.asarray(p, float), qs), 6))


def row_weights(rows):
    n_per = rows.groupby('user_id')['k'].transform('size').to_numpy(float)
    u = rows.drop_duplicates('user_id'); nu = len(u); npos = int(u.label.sum())
    cf = np.where(rows.label.to_numpy() == 1, nu / (2 * npos), nu / (2 * (nu - npos)))
    w = cf / n_per
    return w * (len(w) / w.sum())


class Encoder:
    """Fitted on training rows only (label-free cut points)."""

    def __init__(self, variant):
        self.v = variant

    def fit(self, df):
        self.edges, self.tol = {}, {}
        for q in OQ:
            if q in DROP: continue
            e = np.unique(np.quantile(df[f'cur_{q}'], [0.25, 0.5, 0.75]))
            self.edges[q] = e
            dl = (df[f'cur_{q}'] - df[f'prv_{q}'])[df['first'] == 0]
            self.tol[q] = 0.5 * float(dl.std()) if len(dl) > 1 else 0.0
        return self

    def transform(self, df):
        cols = {}
        if self.v in ('C', 'B'):
            for q in CQ:
                if q in DROP: continue
                cols[f'c_{q}'] = df[f'cur_{q}'].to_numpy()
                cols[f'p_{q}'] = df[f'pre_{q}'].to_numpy()
                cols[f'd_{q}'] = (df[f'cur_{q}'] - df[f'prv_{q}']).to_numpy()
        if self.v in ('O', 'B'):
            for q in self.edges:
                e = self.edges[q]
                for name, src in (('lvc', 'cur'), ('lvp', 'pre')):
                    lv = np.searchsorted(e, df[f'{src}_{q}'].to_numpy(), side='right')
                    for j in range(1, len(e) + 1):
                        cols[f'{name}{j}_{q}'] = (lv == j).astype(float)
                dl = (df[f'cur_{q}'] - df[f'prv_{q}']).to_numpy(); t = self.tol[q]
                cols[f'up_{q}'] = (dl > t).astype(float); cols[f'down_{q}'] = (dl < -t).astype(float)
            for q in MAXLV:
                if q in DROP: continue
                for name, src in (('lvc', 'cur'), ('lvp', 'pre')):
                    lv = df[f'{src}_{q}'].to_numpy().astype(int)
                    for j in (1, 2, 3):
                        cols[f'{name}{j}_{q}'] = (lv == j).astype(float)
                dl = (df[f'cur_{q}'] - df[f'prv_{q}']).to_numpy()
                cols[f'up_{q}'] = (dl > 0).astype(float); cols[f'down_{q}'] = (dl < 0).astype(float)
        if self.v == 'count':
            cols['log_k'] = np.log(df['k'].to_numpy(float))
        if self.v == 'set3':
            for c in SET3:
                cols[c] = df[c].to_numpy(float)
        return pd.DataFrame(cols, index=df.index)


def fit_predict(Xtr, ytr, wtr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000, solver='lbfgs')
    m.fit(sc.transform(Xtr), ytr, sample_weight=wtr)
    return m.predict_proba(sc.transform(Xte))[:, 1], m, sc


def stream(df, p):
    return pd.DataFrame({'user_id': df.user_id.to_numpy(), 't': df.k.to_numpy(), 'label': df.label.to_numpy(), 'p': p})


def pool(ds):
    return Decisions(*[np.concatenate([getattr(d, a) for d in ds]) for a in ('user_id', 'y_true', 'flagged', 'delay', 'n_writings')])


def restrict(d, keep):
    m = np.isin(d.user_id, list(keep))
    return Decisions(d.user_id[m], d.y_true[m], d.flagged[m], d.delay[m], d.n_writings[m])


def horizon_scores(oof, users):
    """Latest decision score at each horizon. A user with no decision yet ranks at the bottom."""
    out = {}; floor = oof.p.min() - 1
    last = oof.sort_values(['user_id', 'k']).groupby('user_id').tail(1).set_index('user_id')
    span = last['day']
    for name, mask in [(f'k{k}', oof.k <= k) for k in K_H] + [(f'd{d}', oof.day <= d) for d in D_H] + [('full', oof.k > 0)]:
        s = oof[mask].sort_values(['user_id', 'k']).groupby('user_id').tail(1).set_index('user_id').p
        out[name] = s.reindex(users.user_id).fillna(floor)
    return out, span


def auroc_block(hs, span, users, np_ids):
    y = users.set_index('user_id').label
    res = {}
    for name, s in hs.items():
        npm = s.index.isin(np_ids)
        r = {'auroc_all': float(roc_auc_score(y, s)), 'auroc_np': float(roc_auc_score(y[npm], s[npm])),
             'auprc_all': float(average_precision_score(y, s)), 'auprc_np': float(average_precision_score(y[npm], s[npm]))}
        if name.startswith('d'):     # span-matched: only users still writing after the horizon
            D = int(name[1:]); act = span.reindex(s.index) > D
            for tag, mm in (('all', act), ('np', act & npm)):
                yy = y[mm]
                r[f'auroc_active_{tag}'] = float(roc_auc_score(yy, s[mm])) if yy.nunique() == 2 else None
                r[f'n_active_{tag}'] = int(mm.sum()); r[f'pos_active_{tag}'] = int(yy.sum())
        res[name] = r
    return res


def load_users():
    rows = json.loads(open(f'{B}/private/cohort_v1/split.json').read())['rows']
    dev = [r for r in rows if r['split'] == 'train']
    return pd.DataFrame({'user_id': [str(r['user_id']) for r in dev], 'label': [int(r['label']) for r in dev],
                         'pilot': [bool(r['pilot_exposed']) for r in dev]})


SET3 = []


def attach_set3(df):
    global SET3
    pre = pd.read_parquet(os.environ.get('SET3_PREFIX', str(_P(__file__).resolve().parent.parent / 'erisk' / 'prefix_masked.parquet')))
    pre['user_id'] = pre.user_id.astype(str)
    pre['log_posts_per_week_so_far'] = np.log1p(pre['posts_per_week_so_far'])
    pre['log_median_gap_so_far'] = np.log1p(pre['median_gap_so_far'])
    pre['log_rate_change_so_far'] = np.log(pre['rate_change_so_far'])
    pre['log_n_last7d'] = np.log1p(pre['n_last7d']); pre['log_n_last28d'] = np.log1p(pre['n_last28d'])
    pre['log_recent_rate_ratio'] = np.log(pre['recent_rate_ratio'].clip(lower=1e-3))
    RH = ["log_posts_per_week_so_far", "night_frac_so_far", "weekend_frac_so_far", "log_median_gap_so_far",
          "gap_cv_so_far", "burst_1h_frac_so_far", "hour_entropy_so_far", "log_rate_change_so_far",
          "log_n_last7d", "log_n_last28d", "log_recent_rate_ratio"]
    SET3 = RH + [f'cum_emo_{e}' for e in CH.EMO]
    m = df.merge(pre[['user_id', 't'] + SET3].rename(columns={'t': 'k'}), on=['user_id', 'k'], how='left')
    miss = m[SET3[0]].isna().mean()
    assert miss < 0.01, f'set3 join missing {miss:.3f}'
    m[SET3] = m[SET3].fillna(m[SET3].median())
    return m


def main(variant, tag=None, drop=(), rep=0):
    DROP.update(drop)
    tag = tag or variant
    oseed = SEED + 1000 * rep          # rep 0 = seed 42 = the earlier set-3 folds
    t0 = time.time()
    users = load_users(); ids = users.user_id.tolist(); yu = users.label.to_numpy()
    np_ids = set(users.loc[~users.pilot, 'user_id'])
    prev_all = float(yu.mean()); prev_np = float(users.loc[~users.pilot, 'label'].mean())
    data, dataW = {}, {}
    pref = os.environ.get('CHUNKPREF', 'chunks')
    # DECIDE=writing: fit on completed-chunk rows as before, but tune thresholds and decide
    # on a row per writing (current chunk = its writings so far). Lets a flag fire at writing 1.
    per_writing = os.environ.get('DECIDE') == 'writing'
    for s in SPECS:
        d = pd.read_parquet(f"{pref}_{s}_PRIVATE.parquet")
        data[s] = attach_set3(d) if variant == 'set3' else d
        if per_writing:
            dataW[s] = pd.read_parquet(f"{pref.replace('chunks', 'chunksW', 1)}_{s}_PRIVATE.parquet")
    outer = StratifiedKFold(5, shuffle=True, random_state=oseed)
    sel_rows, per_spec_dec, per_spec_oof = [], {s: {o: [] for o in OBJ} for s in SPECS}, {s: [] for s in SPECS}
    chosen_dec = {o: [] for o in OBJ}; chosen_oof = []
    for fold, (tr_u, te_u) in enumerate(outer.split(np.zeros(len(yu)), yu), 1):
        tr_ids, te_ids = set(users.user_id.iloc[tr_u]), set(users.user_id.iloc[te_u])
        prev_tr = float(yu[tr_u].mean())
        np_tr = set(users.user_id.iloc[tr_u][~users.pilot.iloc[tr_u].to_numpy()])      # training users only
        prev_np_tr = float(users.set_index('user_id').label.reindex(sorted(np_tr)).mean())
        iu = users.iloc[tr_u].reset_index(drop=True)
        inner = list(StratifiedKFold(3, shuffle=True, random_state=oseed + fold).split(np.zeros(len(iu)), iu.label))
        best = None; spec_best = {}
        for s in SPECS:
            df = data[s]; tr = df[df.user_id.isin(tr_ids)].reset_index(drop=True); te = df[df.user_id.isin(te_ids)].reset_index(drop=True)
            if per_writing:
                dw = dataW[s]; trD = dw[dw.user_id.isin(tr_ids)].reset_index(drop=True); te = dw[dw.user_id.isin(te_ids)].reset_index(drop=True)
            else:
                trD = tr
            enc = Encoder(variant).fit(tr); Xtr = enc.transform(tr); Xte = enc.transform(te)
            XtrD = enc.transform(trD) if per_writing else Xtr
            ytr = tr.label.to_numpy(); wtr = row_weights(tr)
            for C in (CS if variant != 'count' else (1.0,)):
                pin = np.zeros(len(trD))
                for itr, ite in inner:
                    keep = set(iu.user_id.iloc[itr])
                    a = tr.user_id.isin(keep).to_numpy(); aD = trD.user_id.isin(keep).to_numpy()
                    sub = tr[a]
                    pin[~aD] = fit_predict(Xtr[a], ytr[a], row_weights(sub), XtrD[~aD], C)[0]
                ist = stream(trD, pin)
                pol, table, score, f1 = tune_policy(ist, pin, prev_tr, np_tr, prev_np_tr)
                sel_rows.append({'fold': fold, 'spec': s, 'C': C, 'inner_f_latency': score, 'inner_f1': f1,
                                 'inner_erde50': float(table['erde50'].min())})
                if s not in spec_best or score > spec_best[s][0]:
                    spec_best[s] = (score, C, pol)
                if best is None or score > best[0]:
                    best = (score, s, C, pol)
            # refit this spec at its inner-best C on the whole training fold (sensitivity table)
            _, Cb, polb = spec_best[s]
            pte = fit_predict(Xtr, ytr, wtr, Xte, Cb)[0]
            tst = stream(te, pte)
            for o in OBJ:
                per_spec_dec[s][o].append(apply_policy(tst, *polb[o]))
            per_spec_oof[s].append(te.assign(p=pte)[['user_id', 'label', 'k', 'day', 'p']])
        _, sb, Cb, polb = best
        o_df = per_spec_oof[sb][-1]
        chosen_oof.append(o_df)
        for o in OBJ:
            chosen_dec[o].append(per_spec_dec[sb][o][-1])
        print(f'[{tag}] fold {fold}: chose {sb} C={Cb} inner f_lat {best[0]:.3f}  {time.time()-t0:.0f}s', flush=True)
        sel_rows.append({'fold': fold, 'chosen_spec': sb, 'chosen_C': Cb})

    def block(decs, oof):
        pooled = {o: pool(v) for o, v in decs.items()}
        oof = pd.concat(oof)
        hs, span = horizon_scores(oof, users)
        return {'early_all': {o: summarize(d, c_fp=prev_all) for o, d in pooled.items()},
                'early_np': {o: summarize(restrict(d, np_ids), c_fp=prev_np) for o, d in pooled.items()},
                'auroc': auroc_block(hs, span, users, np_ids)}, pooled, oof
    res = {'variant': variant, 'drop': sorted(DROP), 'maxday': MAXDAY, 'decide': 'writing' if per_writing else 'chunk_end', 'policy': POLICY, 'rep': rep, 'outer_seed': oseed, 'selection': sel_rows}
    res['selected'], pooled, oof = block(chosen_dec, chosen_oof)
    res['per_spec'] = {s: block(per_spec_dec[s], per_spec_oof[s])[0] for s in SPECS}
    oof.to_parquet(f'oof_{tag}_PRIVATE.parquet', index=False)
    pd.DataFrame([{'objective': o, 'user_id': u, 'y': y, 'flagged': f, 'delay': dl}
                  for o, d in pooled.items() for u, y, f, dl in zip(d.user_id, d.y_true, d.flagged, d.delay)]
                 ).to_parquet(f'dec_{tag}_PRIVATE.parquet', index=False)
    json.dump(res, open(f'res_{tag}.json', 'w'), indent=1, default=float)
    e = res['selected']
    print(f'[{tag}] DONE {time.time()-t0:.0f}s  f_lat-policy F1 all {e["early_all"]["f_latency"]["f1"]:.3f} '
          f'np {e["early_np"]["f_latency"]["f1"]:.3f} | AUROC full all {e["auroc"]["full"]["auroc_all"]:.3f} '
          f'np {e["auroc"]["full"]["auroc_np"]:.3f} | d28 np {e["auroc"]["d28"]["auroc_np"]:.3f}', flush=True)


if __name__ == '__main__':
    v = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    drop = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != '-' else ()
    rep = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    main(v, tag, drop, rep)
