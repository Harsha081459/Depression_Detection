"""Freeze the v11 winner: fit on the 418 dev users, choose spec / C / tau / t_min by 3-fold inner CV on dev
(same procedure as the v10 lockbox run), then score the 105 held-out users ONCE (labelled: second look).

  python research_v11/freeze_v11.py "<candidate>"        e.g.  MPN32   or   "ens:B|MPN32"

Writes demo/artifacts/frozen_v11.json (text-free: PCA basis, scaler, coefficients, thresholds) and
private/v11/final/ (held-out per-writing scores and decisions, all-user tables).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'demo')); sys.path.insert(0, HERE)
from common import ART, RQ, SE, CH, load_split, load_writings, prepare       # noqa: E402
from erisk_early_metrics import apply_policy, summarize                     # noqa: E402
from views import Views                                                       # noqa: E402
from v11 import load_matrix, parse, logit                                     # noqa: E402

SEED = 42; SPECS = list(CH.SPECS); CS = SE.CS; OBJ = 'f_latency'
FINAL = os.path.join(ROOT, 'private', 'v11', 'final')


def main(cand):
    t0 = time.time(); os.makedirs(FINAL, exist_ok=True)
    members = parse(cand); blocks = sorted({b for m in members for b in m})
    users = load_split(); dev = users.split.eq('train').to_numpy()
    lab = dict(zip(users.user_id, users.label))
    meta = json.load(open(RQ / 'chunk_meta.json'))
    w = load_writings(set(users.user_id)); w['orig'] = w.index.to_numpy()
    w = prepare(w, meta['cuts']); uid = w.user_id.to_numpy()
    A, W, IA, IW = {}, {}, {}, {}
    for s in SPECS:
        fa, fw = os.path.join(FINAL, f'A_{s}_PRIVATE.parquet'), os.path.join(FINAL, f'W_{s}_PRIVATE.parquet')
        if not os.path.exists(fw):
            CH.build(s, w, lab, False).to_parquet(fa, index=False); CH.build(s, w, lab, True).to_parquet(fw, index=False)
        A[s], W[s] = pd.read_parquet(fa), pd.read_parquet(fw)
        for T in (A[s], W[s]):
            T['user_id'] = T.user_id.astype(str)
        v = Views(uid, A[s]); IA[s], IW[s] = v.index(A[s]), v.index(W[s])
    print(f'tables ready {time.time()-t0:.0f}s', flush=True)

    dev_ids = set(users.user_id[dev]); ho_ids = set(users.user_id[~dev])
    devmask_w = np.isin(uid, list(dev_ids))
    views, pcas = {}, {}
    for b in blocks:
        name = b.rstrip('0123456789')
        if name in ('EMO', 'BDI'):
            M = load_matrix(name, w.orig.to_numpy())
        elif name in ('TXT', 'MPN'):
            E = load_matrix(name, w.orig.to_numpy()); k = int(b[len(name):])
            p = PCA(k, random_state=SEED, svd_solver='randomized').fit(E[devmask_w])     # dev writings only
            pcas[b] = p; M = p.transform(E).astype(np.float32)
        else:
            continue
        views[b] = {s: (Views.compute(M, IA[s]), Views.compute(M, IW[s])) for s in SPECS}

    def cont(v, keep):
        cur, prv, pre = v
        return np.hstack([cur[keep], pre[keep], (cur - prv)[keep]])

    def design(m, s, rows_a, rows_w_list):
        a = A[s]; ka = a.user_id.isin(rows_a).to_numpy()
        kws = [W[s].user_id.isin(ids).to_numpy() for ids in rows_w_list]
        tr, outs, encs = [], [[] for _ in kws], {}
        for b in m:
            if b == 'B':
                enc = SE.Encoder('B').fit(a[ka].reset_index(drop=True)); encs['B'] = enc
                tr.append(enc.transform(a[ka]).to_numpy(float))
                for o, kw in zip(outs, kws):
                    o.append(enc.transform(W[s][kw]).to_numpy(float))
            else:
                va, vw = views[b][s]; tr.append(cont(va, ka))
                for o, kw in zip(outs, kws):
                    o.append(cont(vw, kw))
        return (np.hstack(tr), [np.hstack(o) for o in outs], a[ka].reset_index(drop=True),
                [W[s][kw].reset_index(drop=True) for kw in kws], encs)

    U = users[dev].reset_index(drop=True); yu = U.label.to_numpy()
    prev_tr = float(yu.mean()); np_tr = set(U.user_id[~U.pilot]); prev_np_tr = float(U.set_index('user_id').label.reindex(sorted(np_tr)).mean())
    inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED + 1).split(np.zeros(len(U)), U.label))
    SE.POLICY = 'prev_adj'
    chosen, inner_streams, ho_streams, saved = [], [], [], []
    for m in members:
        best = None
        for s in SPECS:
            Xtr, (XtrD,), tr, (trD,), _ = design(m, s, dev_ids, [dev_ids])
            ytr = tr.label.to_numpy()
            for C in CS:
                pin = np.zeros(len(trD))
                for itr, _ in inner:
                    keep = set(U.user_id.iloc[itr])
                    aa = tr.user_id.isin(keep).to_numpy(); aD = trD.user_id.isin(keep).to_numpy()
                    pin[~aD] = SE.fit_predict(Xtr[aa], ytr[aa], SE.row_weights(tr[aa]), XtrD[~aD], C)[0]
                score = SE.tune_policy(SE.stream(trD, pin), pin, prev_tr, np_tr, prev_np_tr)[2]
                if best is None or score > best[0]:
                    best = (score, s, C, pin.copy(), trD[['user_id', 'k', 'label', 'day']].copy())
        score, s, C, pin, keyD = best
        Xtr, (Xho,), tr, (ho,), encs = design(m, s, dev_ids, [ho_ids])
        pho, lr, sc = SE.fit_predict(Xtr, tr.label.to_numpy(), SE.row_weights(tr), Xho, C)
        chosen.append({'member': '+'.join(m), 'spec': s, 'C': C, 'inner_f_latency': score})
        inner_streams.append(keyD.assign(z=logit(pin))); ho_streams.append(ho[['user_id', 'k', 'label', 'day']].assign(z=logit(pho)))
        rec = {'blocks': m, 'spec': s, 'C': C, 'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
               'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0])}
        if 'B' in encs:
            e = encs['B']; rec['encoder'] = {'edges': {q: [float(x) for x in v] for q, v in e.edges.items()},
                                             'tol': {q: float(t) for q, t in e.tol.items()}}
            rec['b_features'] = list(e.transform(A[s].head(2)).columns)
        for b in m:
            if b in pcas:
                rec.setdefault('pca', {})[b] = {'mean': pcas[b].mean_.tolist(), 'components': pcas[b].components_.tolist()}
        saved.append(rec)
        print(f'member {"+".join(m)}: spec {s} C {C} inner f_lat {score:.3f}  {time.time()-t0:.0f}s', flush=True)
    ins = inner_streams[0][['user_id', 'k', 'label', 'day']].copy(); ins['z'] = np.mean([d.z.to_numpy() for d in inner_streams], 0)
    hos = ho_streams[0][['user_id', 'k', 'label', 'day']].copy(); hos['z'] = np.mean([d.z.to_numpy() for d in ho_streams], 0)
    pin = 1 / (1 + np.exp(-ins.z.to_numpy())); pho = 1 / (1 + np.exp(-hos.z.to_numpy()))
    pols = {}
    for pol in ('prev_adj', 'standard'):
        SE.POLICY = pol; pols[pol] = SE.tune_policy(SE.stream(ins, pin), pin, prev_tr, np_tr, prev_np_tr)[0][OBJ]
    print('policies', pols, flush=True)

    # ---- held-out: scored once ----
    H = users[~dev].reset_index(drop=True); hnp = set(H.user_id[~H.pilot])
    out = {}
    for pol, (tau, tmin) in pols.items():
        d = apply_policy(SE.stream(hos, pho), tau, tmin)
        last = hos.assign(p=pho).sort_values(['user_id', 'k']).groupby('user_id').p.last().reindex(H.user_id)
        last = last.fillna(last.min() - 1)
        y = H.set_index('user_id').label.reindex(H.user_id)
        fl = pd.Series(d.flagged, index=d.user_id).reindex(H.user_id).fillna(0).astype(int)
        tp, fp, fn = int(((y == 1) & (fl == 1)).sum()), int(((y == 0) & (fl == 1)).sum()), int(((y == 1) & (fl == 0)).sum())
        out[pol] = {'tau': tau, 't_min': tmin, 'auroc': float(roc_auc_score(y, last)), 'ap': float(average_precision_score(y, last)),
                    'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                    'summary': summarize(d, c_fp=float(H.label.mean()))}
        pd.DataFrame({'user_id': d.user_id, 'y': d.y_true, 'flagged': d.flagged, 'delay': d.delay}).to_parquet(
            os.path.join(FINAL, f'dec_heldout_{pol}_PRIVATE.parquet'), index=False)
    hos.assign(p=pho).to_parquet(os.path.join(FINAL, 'scores_heldout_PRIVATE.parquet'), index=False)
    model = {'id': f'v11-{cand}', 'candidate': cand, 'members': saved, 'chunk_cuts': meta['cuts'],
             'tau': pols['prev_adj'][0], 't_min': pols['prev_adj'][1], 'policy': 'prev_adj',
             'tau_standard': pols['standard'][0], 't_min_standard': pols['standard'][1],
             'train': {'users': int(dev.sum()), 'positives': int(yu.sum())}, 'selection': chosen,
             'heldout_second_look': {p: {k: v for k, v in o.items() if k != 'summary'} | {'erde5': o['summary']['erde5'], 'erde50': o['summary']['erde50']}
                                     for p, o in out.items()},
             'note': 'Screening flag for a human counselling conversation; not a diagnosis. Held-out already scored the v10 model: this is a second look.'}
    json.dump(model, open(os.path.join(ART, 'frozen_v11.json'), 'w'))
    for p, o in out.items():
        print(f'held-out [{p}] tau {o["tau"]:.3f} t_min {o["t_min"]}: AUROC {o["auroc"]:.3f} AP {o["ap"]:.3f} '
              f'F1 {o["f1"]:.3f} TP {o["tp"]} FP {o["fp"]} FN {o["fn"]} ERDE5 {o["summary"]["erde5"]:.3f}', flush=True)
    print(f'done {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
