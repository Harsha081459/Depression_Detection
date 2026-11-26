"""Vectorised chunk views for any per-writing matrix, matching research_v10/chunks.py exactly.

For a decision row (user, k = writings seen, chunk c):
  pre = mean of writings 1..k
  cur = mean of the current chunk's writings up to k (start(c)+1 .. k)
  prv = mean of the previous non-empty chunk (whole chunk); first chunk: prv = cur
Chunk boundaries come from the chunk-end table A of the same spec (row per non-empty chunk, k = chunk end).
`M` rows must be the masked writings in chunks.py order (user, time), `user_ids` aligned with them.
"""
import numpy as np
import pandas as pd


class Views:
    def __init__(self, user_ids, A):
        u = np.asarray(user_ids)
        change = np.r_[True, u[1:] != u[:-1]]
        starts = np.where(change)[0]
        self.offset = dict(zip(u[starts], starts))          # user -> first row in M
        # chunk end k per (user, chunk) from A, and start = previous chunk's end
        a = A[['user_id', 'chunk', 'k']].sort_values(['user_id', 'chunk'])
        prev = a.groupby('user_id').k.shift(1).fillna(0).astype(int)
        self.cstart = dict(zip(zip(a.user_id, a.chunk), prev))
        self.cend = dict(zip(zip(a.user_id, a.chunk), a.k))

    def index(self, rows):
        """Integer index arrays for a rows table (user_id, k, chunk)."""
        o = rows.user_id.map(self.offset).to_numpy()
        k = rows.k.to_numpy()
        keys = list(zip(rows.user_id, rows.chunk))
        s = np.array([self.cstart[x] for x in keys])
        pk = [(u, c - 1) for u, c in keys]
        ps = np.array([self.cstart.get(x, 0) for x in pk]); pe = np.array([self.cend.get(x, 0) for x in pk])
        first = rows.chunk.to_numpy() == 0
        return o, k, s, ps, pe, first

    @staticmethod
    def compute(M, idx):
        """M: (N, d) float32 masked writing matrix. Returns cur, prv, pre arrays (n_rows, d)."""
        o, k, s, ps, pe, first = idx
        CS = np.vstack([np.zeros((1, M.shape[1]), np.float64), np.cumsum(M, 0, dtype=np.float64)])
        pre = (CS[o + k] - CS[o]) / k[:, None]
        cur = (CS[o + k] - CS[o + s]) / (k - s)[:, None]
        n = np.maximum(pe - ps, 1)
        prv = (CS[o + pe] - CS[o + ps]) / n[:, None]
        prv[first] = cur[first]
        return cur.astype(np.float32), prv.astype(np.float32), pre.astype(np.float32)
