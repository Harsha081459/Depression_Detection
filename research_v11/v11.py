"""v11 candidate study: same nested CV as research_v10 (DECIDE=writing), more feature blocks, ensembles.

  python research_v11/v11.py <candidate> <rep> [policy=prev_adj]

Protocol = research_v10/seq_eval.py: outer StratifiedKFold(5, seed 42 + 1000*rep) over the 418 dev users;
inside each outer training fold a 3-fold inner CV picks chunk spec (6) and C (7) by inner latency-F1, then
tau and t_min by the policy. Train on chunk-end rows, decide after every writing. Held-out users never read.

Blocks (per decision row, continuous views cur / pre / cur-prv unless noted)
  B        research_v10 encoding B (277 inputs, ordinal cut points fitted on training rows)
  EMO      28 go_emotions probabilities (RoBERTa, per writing)            -> 84 inputs
  BDI      21 symptom similarities (bge-large vs symptom statements)       -> 63 inputs
  TXT<k>   bge-large embedding, PCA(k) fitted on the outer-training users' writings only -> 3k inputs
  MPN<k>   all-mpnet-base-v2 embedding, same PCA treatment
A candidate is one block set ('B+EMO') or an ensemble of block sets ('ens:B+EMO|TXT32'): each member picks its
own spec and C in the inner CV; the ensemble score = mean logit of the members' scores on the same writing;
tau / t_min are tuned on the ensemble's inner stream.
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'demo')); sys.path.insert(0, HERE)
from common import RQ, SE, CH, load_writings, prepare            # noqa: E402
from erisk_early_metrics import apply_policy, summarize           # noqa: E402
from views import Views                                            # noqa: E402

ENC = os.path.join(ROOT, 'private', 'encoder'); OUT = os.path.join(ROOT, 'private', 'v11')
SPECS = list(CH.SPECS); CS = SE.CS; OBJ = 'f_latency'; SEED = 42
logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))


def load_matrix(name, orig):
    f = {'EMO': 'goemo_PRIVATE.npy', 'BDI': 'bdi_PRIVATE.npy', 'TXT': 'embeddings_bge_PRIVATE.npy',
         'MPN': 'embeddings_PRIVATE.npy'}[name]
    return np.load(os.path.join(ENC, f), mmap_mode='r')[orig].astype(np.float32)


def parse(cand):
    if cand.startswith('ens:'):
        return [m.split('+') for m in cand[4:].split('|')]
    return [cand.split('+')]


def main(cand, rep, policy):
    t0 = time.time(); SE.POLICY = policy
    members = parse(cand)
    blocks = sorted({b for m in members for b in m})
    ALLU = os.environ.get('USERS') == 'all'      # all 523 users (held-out labels already used once): final check
    if ALLU:
        from common import load_split
        users = load_split()[['user_id', 'label', 'pilot']]
    else:
        users = SE.load_users()
    yu = users.label.to_numpy()
    np_ids = set(users.loc[~users.pilot, 'user_id'])
    prev_all = float(yu.mean()); prev_np = float(users.loc[~users.pilot, 'label'].mean())
    meta = json.load(open(RQ / 'chunk_meta.json'))
    w = load_writings(set(users.user_id)); w['orig'] = w.index.to_numpy()
    BM = os.environ.get('BEFOREDISC') == 'matched'
    if BM:     # leakage check, same truncation as research_v10 step 3 (chunks.py BEFOREDISC=matched)
        first = w[w.disclosure].groupby('user_id').ts_sec.min()
        w = w[w.ts_sec < w.user_id.map(first).fillna(np.inf)]
        w = w[~w.disclosure].sort_values(['user_id', 'ts_sec'], kind='mergesort').reset_index(drop=True)
        lab = dict(zip(users.user_id, users.label)); n = w.groupby('user_id').size()
        disc_pos = [u for u in first.index if lab.get(u) == 1 and u in n.index]
        pool = n.loc[disc_pos].to_numpy(); rng_ = np.random.default_rng(42); cap_n = {}
        for u in sorted(n.index):
            cap_n[u] = n[u] if u in set(disc_pos) else min(n[u], int(rng_.choice(pool)))
        kk = w.groupby('user_id').cumcount()
        w = w[kk.to_numpy() < w.user_id.map(cap_n).to_numpy()]
        w = w.assign(disclosure=False)
        users = users[users.user_id.isin(set(w.user_id))].reset_index(drop=True)   # MISSING=exclude
        yu = users.label.to_numpy(); np_ids = set(users.loc[~users.pilot, 'user_id'])
        prev_all = float(yu.mean()); prev_np = float(users.loc[~users.pilot, 'label'].mean())
    w = prepare(w, meta['cuts'])
    uid = w.user_id.to_numpy()
    A, W, V, IA, IW = {}, {}, {}, {}, {}
    bmdir = os.path.join(OUT, 'bm_tables'); os.makedirs(bmdir, exist_ok=True)
    for s in SPECS:
        if BM:
            fa, fw = os.path.join(bmdir, f'A_{s}_PRIVATE.parquet'), os.path.join(bmdir, f'W_{s}_PRIVATE.parquet')
            if not os.path.exists(fw):
                lab = dict(zip(users.user_id, users.label))
                CH.build(s, w, lab, False).to_parquet(fa, index=False); CH.build(s, w, lab, True).to_parquet(fw, index=False)
            A[s], W[s] = pd.read_parquet(fa), pd.read_parquet(fw)
        elif ALLU:      # all-user tables built by freeze_v11.py (same chunks.py code, dev-only cut points)
            fin = os.path.join(OUT, 'final')
            A[s] = pd.read_parquet(os.path.join(fin, f'A_{s}_PRIVATE.parquet')); W[s] = pd.read_parquet(os.path.join(fin, f'W_{s}_PRIVATE.parquet'))
        else:
            A[s] = pd.read_parquet(RQ / f'chunks_{s}_PRIVATE.parquet'); W[s] = pd.read_parquet(RQ / f'chunksW_{s}_PRIVATE.parquet')
        for T in (A[s], W[s]):
            T['user_id'] = T.user_id.astype(str)
        V[s] = Views(uid, A[s]); IA[s] = V[s].index(A[s]); IW[s] = V[s].index(W[s])
    print(f'[{cand} r{rep}] users {len(users)} pos {int(yu.sum())} writings {len(w)} BM={BM}', flush=True)
    fixed = {}                                       # block -> spec -> (A views, W views)
    raw = {}
    for b in blocks:
        name = b.rstrip('0123456789')
        if name in ('EMO', 'BDI'):
            M = load_matrix(name, w.orig.to_numpy())
            fixed[b] = {s: (Views.compute(M, IA[s]), Views.compute(M, IW[s])) for s in SPECS}
        elif name in ('TXT', 'MPN'):
            raw[b] = (load_matrix(name, w.orig.to_numpy()), int(b[len(name):]))
    print(f'[{cand} r{rep}] loaded {time.time()-t0:.0f}s blocks={blocks}', flush=True)

    def cont(views, keep):
        cur, prv, pre = views
        return np.hstack([cur[keep], pre[keep], (cur - prv)[keep]])

    outer = StratifiedKFold(5, shuffle=True, random_state=SEED + 1000 * rep)
    oof, decs, sel = [], [], []
    for fold, (tr_u, te_u) in enumerate(outer.split(np.zeros(len(yu)), yu), 1):
        tr_ids, te_ids = set(users.user_id.iloc[tr_u]), set(users.user_id.iloc[te_u])
        prev_tr = float(yu[tr_u].mean())
        np_tr = set(users.user_id.iloc[tr_u][~users.pilot.iloc[tr_u].to_numpy()])
        prev_np_tr = float(users.set_index('user_id').label.reindex(sorted(np_tr)).mean())
        iu = users.iloc[tr_u].reset_index(drop=True)
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED + 1000 * rep + fold).split(np.zeros(len(iu)), iu.label))
        # in-fold PCA for embedding blocks (training users' writings only, label-free)
        fold_views = {}
        for b, (M, k) in raw.items():
            trm = np.isin(uid, list(tr_ids))
            pca = PCA(k, random_state=SEED, svd_solver='randomized').fit(M[trm])
            Z = pca.transform(M).astype(np.float32)
            fold_views[b] = {s: (Views.compute(Z, IA[s]), Views.compute(Z, IW[s])) for s in SPECS}
        fv = lambda b: fixed[b] if b in fixed else fold_views[b]

        def design(m, s):
            a, wr = A[s], W[s]
            ka = a.user_id.isin(tr_ids).to_numpy(); kw = wr.user_id.isin(tr_ids).to_numpy(); kt = wr.user_id.isin(te_ids).to_numpy()
            parts_tr, parts_trD, parts_te = [], [], []
            for b in m:
                if b == 'B':
                    enc = SE.Encoder('B').fit(a[ka].reset_index(drop=True))
                    parts_tr.append(enc.transform(a[ka]).to_numpy(float)); parts_trD.append(enc.transform(wr[kw]).to_numpy(float))
                    parts_te.append(enc.transform(wr[kt]).to_numpy(float))
                else:
                    va, vw = fv(b)[s]
                    parts_tr.append(cont(va, ka)); parts_trD.append(cont(vw, kw)); parts_te.append(cont(vw, kt))
            return (np.hstack(parts_tr), np.hstack(parts_trD), np.hstack(parts_te),
                    a[ka].reset_index(drop=True), wr[kw].reset_index(drop=True), wr[kt].reset_index(drop=True))

        mem_inner, mem_test = [], []
        for mi, m in enumerate(members):
            best = None
            for s in SPECS:
                Xtr, XtrD, Xte, tr, trD, te = design(m, s)
                ytr = tr.label.to_numpy(); wtr = SE.row_weights(tr)
                for C in CS:
                    pin = np.zeros(len(trD))
                    for itr, _ in inner:
                        keep = set(iu.user_id.iloc[itr])
                        aa = tr.user_id.isin(keep).to_numpy(); aD = trD.user_id.isin(keep).to_numpy()
                        pin[~aD] = SE.fit_predict(Xtr[aa], ytr[aa], SE.row_weights(tr[aa]), XtrD[~aD], C)[0]
                    _, _, score, _ = SE.tune_policy(SE.stream(trD, pin), pin, prev_tr, np_tr, prev_np_tr)
                    if best is None or score > best[0]:
                        best = (score, s, C, pin.copy(), trD[['user_id', 'k', 'label', 'day']].copy())
            score, s, C, pin, keyD = best
            Xtr, XtrD, Xte, tr, trD, te = design(m, s)
            pte = SE.fit_predict(Xtr, tr.label.to_numpy(), SE.row_weights(tr), Xte, C)[0]
            mem_inner.append(keyD.assign(z=logit(pin)))
            mem_test.append(te[['user_id', 'k', 'label', 'day']].assign(z=logit(pte)))
            sel.append({'fold': fold, 'member': '+'.join(m), 'spec': s, 'C': C, 'inner_f_latency': score})
        ins = mem_inner[0][['user_id', 'k', 'label', 'day']].copy(); ins['z'] = np.mean([d.z.to_numpy() for d in mem_inner], 0)
        tes = mem_test[0][['user_id', 'k', 'label', 'day']].copy(); tes['z'] = np.mean([d.z.to_numpy() for d in mem_test], 0)
        for ref, group in ((ins, mem_inner), (tes, mem_test)):          # members score the same writings
            for d in group[1:]:
                assert (d.user_id.to_numpy() == ref.user_id.to_numpy()).all() and (d.k.to_numpy() == ref.k.to_numpy()).all()
        pin = 1 / (1 + np.exp(-ins.z.to_numpy())); pte = 1 / (1 + np.exp(-tes.z.to_numpy()))
        pol = SE.tune_policy(SE.stream(ins, pin), pin, prev_tr, np_tr, prev_np_tr)[0][OBJ]
        decs.append(apply_policy(SE.stream(tes, pte), *pol))
        oof.append(tes.assign(p=pte, fold=fold)[['user_id', 'label', 'k', 'day', 'p', 'fold']])
        sel.append({'fold': fold, 'tau': pol[0], 't_min': pol[1]})
        print(f'[{cand} r{rep}] fold {fold} {[(x["spec"], x["C"]) for x in sel if x.get("fold") == fold and "spec" in x]} '
              f'tau {pol[0]:.3f} t_min {pol[1]}  {time.time()-t0:.0f}s', flush=True)

    d = SE.pool(decs); o = pd.concat(oof)
    hs, span = SE.horizon_scores(o, users)
    res = {'candidate': cand, 'rep': rep, 'policy': policy, 'selection': sel,
           'early_all': summarize(d, c_fp=prev_all), 'early_np': summarize(SE.restrict(d, np_ids), c_fp=prev_np),
           'auroc': SE.auroc_block(hs, span, users, np_ids)}
    os.makedirs(OUT, exist_ok=True)
    tag = f"{cand.replace(':', '_').replace('|', '-').replace('+', '_')}_{policy}{'_BM' if BM else ''}{'_ALL' if ALLU else ''}_r{rep}"
    res['bm'] = BM
    o.to_parquet(os.path.join(OUT, f'oof_{tag}_PRIVATE.parquet'), index=False)
    pd.DataFrame({'user_id': d.user_id, 'y': d.y_true, 'flagged': d.flagged, 'delay': d.delay}).to_parquet(
        os.path.join(OUT, f'dec_{tag}_PRIVATE.parquet'), index=False)
    json.dump(res, open(os.path.join(OUT, f'res_{tag}.json'), 'w'), indent=1, default=float)
    ea, en, au = res['early_all'], res['early_np'], res['auroc']['full']
    print(f'[{cand} r{rep}] DONE {time.time()-t0:.0f}s  F1 all {ea["f1"]:.3f} np {en["f1"]:.3f} | AUROC all '
          f'{au["auroc_all"]:.3f} np {au["auroc_np"]:.3f} | AP all {au["auprc_all"]:.3f} np {au["auprc_np"]:.3f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'prev_adj')
