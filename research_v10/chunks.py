"""Full-history chunk features, development users only (418).

Each user's whole history is cut into chunks. A decision row is emitted at the end
of every non-empty chunk, and at the end of the history. A row uses only writings
at or before its decision time.

Chunk specs (the 'flexible bins'): time chunks of 7, 14, 28 days; post chunks of
10, 25, 50 writings. The spec is a hyperparameter, chosen inside training folds.

Per chunk, 27 base quantities q. Per row we store q for the current chunk, the
previous non-empty chunk, and the whole prefix. Continuous deltas and the
professor's lower/same/higher and level encodings are made later, inside the CV
loop, because their cut points must come from training users only.

Masked setting: the 610 disclosure writings are removed before anything else.
Excluded on purpose (known collection artefacts): comment share, writing length,
elapsed time, writings seen. Writings-seen is kept only as metadata and as the
count-only baseline.
"""
import json, os, sys, time
import numpy as np, pandas as pd

import os as _os
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
B = _os.environ.get('AIXPE_ROOT') or (str(_here.parent) if (_here.parent / 'data').exists() else '/mnt/user-data/uploads/AIX PE')
SIG = ['self_blame', 'helplessness', 'isolation', 'family_strain', 'emptiness']
EMO = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative"]
SPECS = {'d7': ('days', 7), 'd14': ('days', 14), 'd28': ('days', 28),
         'p10': ('posts', 10), 'p25': ('posts', 25), 'p50': ('posts', 50)}
DAY = 86400.0


def calibrated_cuts(w):
    """Student-logit cuts whose exceedance rates on the Qwen-scored dev writings match
    Qwen's P(Yes) > 0.25 / 0.5 / 0.75 rates. Gives a 0-3 level per writing per signal."""
    cuts = {}
    m = w[f'q_{SIG[0]}'].notna()
    for k in SIG:
        q = w.loc[m, f'q_{k}'].to_numpy(); s = w.loc[m, f'lg_{k}'].to_numpy()
        cuts[k] = [float(np.quantile(s, 1 - max((q > c).mean(), 1e-5))) for c in (0.25, 0.5, 0.75)]
    return cuts


def per_writing(w, cuts):
    for k in SIG:
        c = cuts[k]; s = w[f'lg_{k}'].to_numpy()
        w[f'lv_{k}'] = (s > c[0]).astype(int) + (s > c[1]) + (s > c[2])
    lv = w[[f'lv_{k}' for k in SIG]].to_numpy()
    w['lv_max'] = lv.max(1)
    w['distressed'] = (w['lv_max'] >= 2).astype(float)          # ~ Qwen P(Yes) > 0.5 on any signal
    w['lg_max'] = w[[f'lg_{k}' for k in SIG]].max(axis=1)
    w['first_person'] = w.self_tokens / np.maximum(w.n_tokens, 1)
    pol = w.polarity.to_numpy(); sg = np.sign(pol)
    prev = w.groupby('user_id').polarity.shift(1).to_numpy()
    w['flip'] = np.where(np.isnan(prev), 0.0, ((np.sign(prev) * sg) < 0).astype(float))
    w['abs_jump'] = np.where(np.isnan(prev), 0.0, np.abs(pol - np.nan_to_num(prev)))
    return w


# per-writing columns averaged within a chunk / prefix
MEAN_COLS = [f'lg_{k}' for k in SIG] + ['distressed', 'lv_max', 'absolutist', 'polarity', 'flip', 'abs_jump',
             'first_person', 'profanity'] + [f'emo_{e}' for e in EMO] + ['night']
# optional frozen sentence-encoder view: EMB=<embeddings .npy> EMBDIM=32 -> emb00..emb31 per writing
EMBD = int(os.environ.get('EMBDIM', '0')) if os.environ.get('EMB') else 0
MEAN_COLS = MEAN_COLS + [f'emb{i:02d}' for i in range(EMBD)]
# extra per-chunk quantities computed by hand: polarity_sd (mood swing), log_rate, log_med_gap, silence_after_distress
EXTRA = ['polarity_sd', 'log_rate', 'log_med_gap', 'silence_after_distress'] + [f'maxlv_{k}' for k in SIG]
Q = MEAN_COLS + EXTRA


def user_rows(x, spec, every_writing=False):
    kind, width = SPECS[spec]
    ts = x.ts_sec.to_numpy(np.int64); n = len(ts)
    day = (ts - ts[0]) / DAY
    if kind == 'days':
        cidx = (day // width).astype(int)
    else:
        cidx = np.arange(n) // width
    # boundaries: last writing index of each non-empty chunk
    ends = np.r_[np.where(np.diff(cidx) != 0)[0], n - 1]
    V = x[MEAN_COLS].to_numpy(float)
    cs = np.vstack([np.zeros(V.shape[1]), np.cumsum(V, 0)])
    pol = x.polarity.to_numpy(float)
    cp, cp2 = np.r_[0, np.cumsum(pol)], np.r_[0, np.cumsum(pol ** 2)]
    dist = x.distressed.to_numpy(bool)
    LV = x[[f'lv_{k}' for k in SIG]].to_numpy(int)
    gaps = np.diff(ts) / DAY

    def q_at(start, e, t_dec, dur):
        """Current chunk = writings start..e; prefix = 0..e; nothing after t_dec is used."""
        m = e - start + 1
        cur = (cs[e + 1] - cs[start]) / m
        pre = cs[e + 1] / (e + 1)
        s1, s2 = cp[e + 1] - cp[start], cp2[e + 1] - cp2[start]
        sd = np.sqrt(max(s2 / m - (s1 / m) ** 2, 0))
        psd = np.sqrt(max(cp2[e + 1] / (e + 1) - (cp[e + 1] / (e + 1)) ** 2, 0))
        g_chunk = gaps[start:e] if e > start else np.array([])
        g_pre = gaps[:e]
        base = np.median(g_pre[g_pre > 0]) if (g_pre > 0).any() else np.nan
        # silence after distress: gap after each distressed writing in the chunk, truncated at decision time
        di = np.where(dist[start:e + 1])[0] + start
        if len(di) and np.isfinite(base) and base > 0:
            nxt = np.where(di + 1 < n, ts[np.minimum(di + 1, n - 1)], np.iinfo(np.int64).max)
            g_after = (np.minimum(nxt, t_dec) - ts[di]) / DAY
            sil = float(np.log1p(g_after.max() / base))
        else:
            sil = 0.0
        di_p = np.where(dist[:e + 1])[0]
        if len(di_p) and np.isfinite(base) and base > 0:
            nxt = np.where(di_p + 1 < n, ts[np.minimum(di_p + 1, n - 1)], np.iinfo(np.int64).max)
            sil_p = float(np.log1p(((np.minimum(nxt, t_dec) - ts[di_p]) / DAY).max() / base))
        else:
            sil_p = 0.0
        el = max((t_dec - ts[0]) / DAY, 7.0)
        qc = np.r_[cur, sd, np.log1p(m / dur), np.log1p(np.median(g_chunk)) if len(g_chunk) else 0.0, sil,
                   LV[start:e + 1].max(0)]
        qp = np.r_[pre, psd, np.log1p((e + 1) / el), np.log1p(np.median(g_pre)) if len(g_pre) else 0.0, sil_p,
                   LV[:e + 1].max(0)]
        return qc, qp

    rows = []; prev_q = None; start = 0
    for ci, e in enumerate(ends):
        c = cidx[e]
        c0 = ts[0] + c * width * DAY if kind == 'days' else None
        if kind == 'days':
            t_dec = ts[0] + (c + 1) * width * DAY            # decision at chunk end (time has passed)
            if ci == len(ends) - 1:
                t_dec = min(t_dec, ts[-1])                   # history ends: decide at last writing
            dur = width if ci < len(ends) - 1 else max((ts[e] - c0) / DAY, 1.0)
        else:
            t_dec = ts[e]
            dur = max((ts[e] - ts[start]) / DAY, 1.0)
        if every_writing:
            # one decision per writing: current chunk = its writings so far, previous chunk = last complete one
            for i in range(start, e + 1):
                d_i = max(((ts[i] - c0) if kind == 'days' else (ts[i] - ts[start])) / DAY, 1.0)
                qc_i, qp_i = q_at(start, i, ts[i], d_i)
                rows.append({'k': i + 1, 'day': (ts[i] - ts[0]) / DAY, 'chunk': ci, 'cur': qc_i, 'pre': qp_i,
                             'prv': prev_q if prev_q is not None else qc_i, 'first': prev_q is None})
            qc, _ = q_at(start, e, t_dec, dur)
        else:
            qc, qp = q_at(start, e, t_dec, dur)
            rows.append({'k': e + 1, 'day': (t_dec - ts[0]) / DAY, 'chunk': ci, 'cur': qc, 'pre': qp,
                         'prv': prev_q if prev_q is not None else qc, 'first': prev_q is None})
        prev_q = qc; start = e + 1
    return rows


def build(spec, w, users, every_writing=False):
    out = []
    for uid, x in w.groupby('user_id', sort=False):
        lab = users[uid]
        for r in user_rows(x, spec, every_writing):
            d = {'user_id': uid, 'label': lab, 'k': r['k'], 'day': r['day'], 'chunk': r['chunk'],
                 'first': int(r['first'])}
            for j, q in enumerate(Q):
                d[f'cur_{q}'] = r['cur'][j]; d[f'prv_{q}'] = r['prv'][j]; d[f'pre_{q}'] = r['pre'][j]
            out.append(d)
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    rows = json.loads(open(f'{B}/private/cohort_v1/split.json').read())['rows']
    users = {str(r['user_id']): int(r['label']) for r in rows if r['split'] == 'train'}
    w = pd.read_parquet('writings_PRIVATE.parquet').drop(columns='text')
    s = pd.read_parquet('student_scores_PRIVATE.parquet')
    for k in SIG:
        p = np.clip(s[f's_{k}'].to_numpy(), 1e-9, 1 - 1e-9); w[f'lg_{k}'] = np.log(p / (1 - p))
    if EMBD:
        E = np.load(os.environ['EMB']).astype(np.float32)
        assert len(E) == len(w), f'embeddings rows {len(E)} != writings {len(w)}'
        from sklearn.decomposition import PCA
        fit = (w.user_id.isin(users) & ~w.disclosure).to_numpy()          # dev users, label-free
        Z = PCA(n_components=EMBD, random_state=42).fit(E[fit]).transform(E)
        for i in range(EMBD): w[f'emb{i:02d}'] = Z[:, i]
        del E
    w = w[w.user_id.isin(users)]
    cuts = calibrated_cuts(w)
    w = w[~w.disclosure].sort_values(['user_id', 'ts_sec'], kind='mergesort').reset_index(drop=True)
    import os
    cap = float(os.environ['CAPDAY']) if os.environ.get('CAPDAY') else None
    pref = 'chunks' if cap is None else f'chunkscap{int(cap)}'
    every = os.environ.get('WRITING') == '1'      # one decision row per writing
    if EMBD: pref = pref.replace('chunks', 'chunksE', 1)
    if every: pref = pref.replace('chunks', 'chunksW', 1)
    if cap is not None:          # ablation: the old 84-day cap, applied to writings before chunking
        w = w[(w.ts_sec - w.groupby('user_id').ts_sec.transform('min')) / DAY <= cap].reset_index(drop=True)
    w = per_writing(w, cuts)
    print('dev writings after masking', len(w), 'users', w.user_id.nunique(),
          'distressed share %.4f' % w.distressed.mean(), flush=True)
    if cap is None and not every: json.dump({'cuts': cuts, 'distressed_share': float(w.distressed.mean()), 'Q': Q}, open('chunk_meta.json', 'w'), indent=1)
    for spec in SPECS:
        df = build(spec, w, users, every)
        df.to_parquet(f'{pref}_{spec}_PRIVATE.parquet', index=False)
        g = df.groupby('user_id').size()
        print(f'{spec}: rows {len(df)}  median decisions/user {g.median():.0f}  max {g.max()}  {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
