"""Freeze the held-out ("lockbox") model so the demo can run it on one user at a time.

Re-runs the lockbox fit from the step-3 seq_eval.py (USERS=lockbox, variant B, DECIDE=writing,
POLICY=standard) at its logged choice: spec d14, C = 1e-4 (res_LB_W_B.json). Training and threshold
tuning use the 418 dev users only. The 105 held-out users are then scored and compared with the
saved lockbox outputs; the script stops if they differ.

Writes
  demo/artifacts/frozen_model.json        text-free: encoder cut points, scaler, coefficients, thresholds
  private/demo/chunksA_d14_PRIVATE.parquet   chunk-end rows, all 523 users
  private/demo/chunksAW_d14_PRIVATE.parquet  one row per writing, all 523 users
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import ART, PRIV, RQ, SPEC, CH, SE, load_split, load_writings, prepare

SEED = 42
C_FROZEN = 0.0001
OBJECTIVE = 'f_latency'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    users = load_split()                                   # split.json order, as in lockbox load_users()
    dev = users.split.eq('train').to_numpy()
    lab = dict(zip(users.user_id, users.label))
    w = load_writings()
    w = w[w.user_id.isin(lab)]
    cuts = CH.calibrated_cuts(w[w.user_id.isin(set(users.user_id[dev]))])
    meta = json.loads((RQ / 'chunk_meta.json').read_text())
    dmax = max(abs(a - b) for k in CH.SIG for a, b in zip(cuts[k], meta['cuts'][k]))
    print(f'cuts vs chunk_meta.json max diff {dmax:.2e}', flush=True)
    w = prepare(w, cuts)

    PRIV.mkdir(parents=True, exist_ok=True)
    fA, fW = PRIV / f'chunksA_{SPEC}_PRIVATE.parquet', PRIV / f'chunksAW_{SPEC}_PRIVATE.parquet'
    if fA.exists() and fW.exists():
        A, AW = pd.read_parquet(fA), pd.read_parquet(fW)
    else:
        A = CH.build(SPEC, w, lab, every_writing=False); A.to_parquet(fA, index=False)
        AW = CH.build(SPEC, w, lab, every_writing=True); AW.to_parquet(fW, index=False)
    print(f'chunk rows {len(A)}  writing rows {len(AW)}  {time.time()-t0:.0f}s', flush=True)

    tr_ids = set(users.user_id[dev]); te_ids = set(users.user_id[~dev])
    tr = A[A.user_id.isin(tr_ids)].reset_index(drop=True)
    trD = AW[AW.user_id.isin(tr_ids)].reset_index(drop=True)
    te = AW[AW.user_id.isin(te_ids)].reset_index(drop=True)
    yu = users.label.to_numpy(); tr_u = np.where(dev)[0]
    prev_tr = float(yu[tr_u].mean())
    np_tr = set(users.user_id.iloc[tr_u][~users.pilot.iloc[tr_u].to_numpy()])
    prev_np_tr = float(users.set_index('user_id').label.reindex(sorted(np_tr)).mean())

    enc = SE.Encoder('B').fit(tr)
    Xtr, XtrD, Xte = enc.transform(tr), enc.transform(trD), enc.transform(te)
    ytr = tr.label.to_numpy(); wtr = SE.row_weights(tr)

    # inner 3-fold stream on dev users -> threshold tau and t_min (fold = 1 in the lockbox run)
    iu = users.iloc[tr_u].reset_index(drop=True)
    inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED + 1).split(np.zeros(len(iu)), iu.label))
    pin = np.zeros(len(trD))
    for itr, _ in inner:
        keep = set(iu.user_id.iloc[itr])
        a = tr.user_id.isin(keep).to_numpy(); aD = trD.user_id.isin(keep).to_numpy()
        pin[~aD] = SE.fit_predict(Xtr[a], ytr[a], SE.row_weights(tr[a]), XtrD[~aD], C_FROZEN)[0]
    ist = SE.stream(trD, pin)
    pols = {}
    for policy in ('standard', 'prev_adj'):
        SE.POLICY = policy
        pols[policy] = SE.tune_policy(ist, pin, prev_tr, np_tr, prev_np_tr)[0][OBJECTIVE]
    SE.POLICY = 'standard'
    print('policies', pols, flush=True)

    # final model on all dev chunk-end rows
    pte, m, sc = SE.fit_predict(Xtr, ytr, wtr, Xte, C_FROZEN)

    # ---- reproduction check against the saved lockbox outputs ----
    ref = pd.read_parquet(PRIV / 'ref' / 'oof_LB_W_B_PRIVATE.parquet')
    mine = te[['user_id', 'k']].assign(p=pte)
    j = ref.merge(mine, on=['user_id', 'k'], suffixes=('_ref', ''))
    assert len(j) == len(ref) == len(mine), (len(j), len(ref), len(mine))
    dp = float(np.abs(j.p - j.p_ref).max())
    print(f'held-out score max abs diff vs lockbox {dp:.2e}', flush=True)
    assert dp < 1e-6, 'scores do not reproduce the lockbox run'
    tau, tmin = pols['standard']
    dec = SE.apply_policy(SE.stream(te, pte), tau, tmin)
    refd = pd.read_parquet(PRIV / 'ref' / 'dec_LB_W_B_PRIVATE.parquet').query('objective == @OBJECTIVE')
    mined = pd.DataFrame({'user_id': dec.user_id, 'flagged': dec.flagged.astype(int), 'delay': dec.delay})
    jd = refd.merge(mined, on='user_id', suffixes=('_ref', ''))
    assert (jd.flagged == jd.flagged_ref).all(), 'flags differ from lockbox'
    hu = users[~dev]
    last = mine.sort_values(['user_id', 'k']).groupby('user_id').p.last().reindex(hu.user_id).fillna(mine.p.min() - 1)
    auroc = float(roc_auc_score(hu.label, last))
    y = jd.set_index('user_id').y.astype(int); f = jd.set_index('user_id').flagged
    tp, fp, fn = int(((y == 1) & (f == 1)).sum()), int(((y == 0) & (f == 1)).sum()), int(((y == 1) & (f == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f'held-out reproduced: AUROC {auroc:.4f}  F1 {f1:.4f}  TP {tp} FP {fp} FN {fn}', flush=True)
    assert abs(auroc - 0.9379) < 5e-4 and (tp, fp, fn) == (5, 2, 5)

    # prev_adj as a second look on held-out (reported, never used for selection)
    d2 = SE.apply_policy(SE.stream(te, pte), *pols['prev_adj'])
    second = {'n_flagged': int(d2.flagged.sum()), 'tp': int((d2.flagged & (d2.y_true == 1)).sum())}

    out = {
        'id': 'B-d14-lockbox', 'variant': 'B', 'spec': SPEC, 'C': C_FROZEN, 'objective': OBJECTIVE,
        'tau': float(tau), 't_min': int(tmin),
        'tau_prev_adj': float(pols['prev_adj'][0]), 't_min_prev_adj': int(pols['prev_adj'][1]),
        'cuts': cuts, 'Q': CH.Q,
        'encoder': {'edges': {q: [float(x) for x in e] for q, e in enc.edges.items()},
                    'tol': {q: float(t) for q, t in enc.tol.items()}},
        'features': list(Xtr.columns),
        'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
        'coef': m.coef_[0].tolist(), 'intercept': float(m.intercept_[0]),
        'train': {'users': int(dev.sum()), 'positives': int(yu[tr_u].sum()), 'rows': int(len(tr))},
        'heldout': {'auroc': auroc, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'n': int(len(hu)),
                    'policy': 'standard', 'prev_adj_second_look': second},
        'inputs_sha256': {n: sha(RQ / n) for n in ('writings_PRIVATE.parquet', 'student_scores_PRIVATE.parquet')},
        'note': 'Screening flag for a human counselling conversation; not a diagnosis.',
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / 'frozen_model.json').write_text(json.dumps(out, indent=1))
    print(f'wrote frozen_model.json ({len(out["features"])} inputs)  {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
