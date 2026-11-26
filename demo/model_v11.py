"""Run the frozen v11 model on one user and explain it exactly.

Each member is a logistic regression on chunk views (cur, pre, cur - prv) of its blocks:
  B           research_v10 encoding (Qwen signals, word lists, timing; continuous + ordinal)
  MPN<k>/TXT<k>  sentence-embedding PCA (mpnet / bge-large)
  EMO / BDI   go_emotions probabilities / symptom-statement similarities
Ensemble score = mean of the members' logits. Refer at the first writing with score >= tau and k >= t_min.

Exact explanations (the model is linear in each writing's embedding, because chunk views are means):
  per-writing contribution of the text member: c_w = a_cur.z_w / n_cur [w in current chunk]
      + a_pre.z_w / k + a_d.z_w / n_cur [w in current chunk] - a_d.z_w / n_prv [w in previous chunk]
  with a = coef / scale (per view), z_w = PCA(e_w). Sum over writings + baseline = the member's logit.
  per-quantity contribution of B: coef x standardised input, summed per quantity (as in explain.py).
"""
import json

import numpy as np
import pandas as pd

from common import ART, CH, SE

SIG = CH.SIG


class Member:
    def __init__(self, rec):
        self.rec = rec; self.blocks = rec['blocks']; self.spec = rec['spec']
        self.mean = np.asarray(rec['scaler_mean']); self.scale = np.asarray(rec['scaler_scale'])
        self.coef = np.asarray(rec['coef']); self.b0 = rec['intercept']
        self.pca = {b: (np.asarray(p['mean'], np.float32), np.asarray(p['components'], np.float32)) for b, p in rec.get('pca', {}).items()}
        if 'B' in self.blocks:
            enc = SE.Encoder('B'); enc.edges = {q: np.asarray(e) for q, e in rec['encoder']['edges'].items()}
            enc.tol = dict(rec['encoder']['tol']); self.enc = enc
        # column layout of the design matrix: one slice per block
        self.slices, start = {}, 0
        for b in self.blocks:
            n = len(rec['b_features']) if b == 'B' else 3 * self.dim(b)
            self.slices[b] = slice(start, start + n); start += n
        assert start == len(self.coef), (start, len(self.coef))

    def dim(self, b):
        if b in self.pca:
            return self.pca[b][1].shape[0]
        return {'EMO': 28, 'BDI': 21}[b.rstrip('0123456789')]

    def project(self, b, E):
        mu, comp = self.pca[b]
        return (E.astype(np.float32) - mu) @ comp.T


class FrozenV11:
    def __init__(self, path=None):
        self.m = json.load(open(path or (ART / 'frozen_v11.json')))
        self.members = [Member(r) for r in self.m['members']]
        self.tau, self.t_min = self.m['tau'], self.m['t_min']

    # ---------------------------------------------------------------- views for one user
    @staticmethod
    def rows(x, spec, every):
        out = []
        for r in CH.user_rows(x, spec, every):
            d = {'k': r['k'], 'day': r['day'], 'chunk': r['chunk'], 'first': int(r['first'])}
            for j, q in enumerate(CH.Q):
                d[f'cur_{q}'] = r['cur'][j]; d[f'prv_{q}'] = r['prv'][j]; d[f'pre_{q}'] = r['pre'][j]
            out.append(d)
        return pd.DataFrame(out)

    @staticmethod
    def bounds(A, W):
        """start (0-based) of each W row's chunk and the previous chunk's [start, end)."""
        ends = dict(zip(A.chunk, A.k)); starts = {c: (ends.get(c - 1, 0) if c > 0 else 0) for c in A.chunk}
        s = W.chunk.map(starts).to_numpy(); ps = W.chunk.map(lambda c: starts.get(c - 1, 0)).to_numpy()
        pe = W.chunk.map(lambda c: ends.get(c - 1, 0)).to_numpy()
        return s, ps, pe

    @staticmethod
    def mean_views(Z, W, bnd):
        s, ps, pe = bnd; k = W.k.to_numpy(); first = W.chunk.to_numpy() == 0
        CS = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, 0, dtype=np.float64)])
        pre = (CS[k] - CS[0]) / k[:, None]; cur = (CS[k] - CS[s]) / (k - s)[:, None]
        prv = (CS[pe] - CS[ps]) / np.maximum(pe - ps, 1)[:, None]; prv[first] = cur[first]
        return cur, prv, pre

    def score_user(self, x, mats):
        """x: one user's prepared writings (chunks.per_writing applied, disclosure removed, time order).
        mats: {'MPN': (n, 768), 'TXT': (n, 1024), 'EMO': (n, 28), 'BDI': (n, 21)} rows aligned with x.
        Returns per-writing ensemble scores, member scores, and what the explanation needs."""
        x = x.reset_index(drop=True); out = {'members': []}; zs = []
        for mem in self.members:
            A = self.rows(x, mem.spec, False); W = self.rows(x, mem.spec, True); bnd = self.bounds(A, W)
            parts, keep = [], {}
            for b in mem.blocks:
                if b == 'B':
                    parts.append(mem.enc.transform(W)[mem.rec['b_features']].to_numpy(float))
                else:
                    name = b.rstrip('0123456789')
                    Z = mem.project(b, mats[name]) if b in mem.pca else np.asarray(mats[name], np.float32)
                    cur, prv, pre = self.mean_views(Z, W, bnd); keep[b] = Z
                    parts.append(np.hstack([cur, pre, cur - prv]))
            X = np.hstack(parts); Xs = (X - mem.mean) / mem.scale
            z = Xs @ mem.coef + mem.b0; zs.append(z)
            out['members'].append({'mem': mem, 'W': W, 'A': A, 'bnd': bnd, 'Xs': Xs, 'z': z, 'Z': keep})
        zbar = np.mean(zs, 0); p = 1 / (1 + np.exp(-zbar))
        k = out['members'][0]['W'].k.to_numpy()
        hit = np.where((p >= self.tau) & (k >= self.t_min))[0]
        out.update(p=p, z=zbar, k=k, day=out['members'][0]['W'].day.to_numpy(), flag=int(hit[0]) if len(hit) else None)
        return out

    # ---------------------------------------------------------------- explanations at one decision row
    def explain_row(self, res, i):
        """i = row index (0-based writing) of the decision. Returns per-member logits, B quantity contributions,
        per-writing contributions of each embedding block (sum + baseline = member logit, exactly)."""
        expl = []
        for r in res['members']:
            mem, W, Xs = r['mem'], r['W'], r['Xs']
            c = mem.coef * Xs[i]; e = {'member': '+'.join(mem.blocks), 'logit': float(r['z'][i]), 'intercept': float(mem.b0)}
            for b in mem.blocks:
                sl = mem.slices[b]; cb = c[sl]
                if b == 'B':
                    from explain import quantity, LABEL
                    q = {}
                    for f, v in zip(mem.rec['b_features'], cb):
                        q[quantity(f)] = q.get(quantity(f), 0.0) + float(v)
                    e['B'] = sorted(({'quantity': k, 'label': LABEL.get(k, k), 'contribution': v} for k, v in q.items()),
                                    key=lambda d: -d['contribution'])
                    e['B_total'] = float(cb.sum())
                else:
                    d = mem.dim(b); a = mem.coef[sl] / mem.scale[sl]; mu = mem.mean[sl]
                    a_cur, a_pre, a_d = a[:d], a[d:2 * d], a[2 * d:]
                    Z = r['Z'][b]; s, ps, pe = (v[i] for v in r['bnd']); k = int(W.k.iloc[i]); first = W.chunk.iloc[i] == 0
                    contrib = (Z[:k] @ a_pre) / k
                    n_cur = k - s
                    contrib[s:k] += (Z[s:k] @ (a_cur + (0 if first else a_d))) / n_cur
                    if not first and pe > ps:
                        contrib[ps:pe] -= (Z[ps:pe] @ a_d) / (pe - ps)
                    base = -float(mu @ a)                     # standardisation offsets
                    total = float(cb.sum())
                    assert abs(contrib.sum() + base - total) < 1e-3, (contrib.sum() + base, total)
                    e[b] = {'per_writing': contrib.tolist(), 'baseline': base, 'total': total}
            expl.append(e)
        return expl


def direction_vs_statements(mem, b, S):
    """Cosine between the text member's 'history' risk direction (back in embedding space) and statement vectors S."""
    d = mem.dim(b); sl = mem.slices[b]; a_pre = (mem.coef[sl] / mem.scale[sl])[d:2 * d]
    v = mem.pca[b][1].T @ a_pre; v = v / np.linalg.norm(v)
    return S @ v
