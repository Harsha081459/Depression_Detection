"""v3 features for eRisk Task 2 (exploratory, designed after v2 results).

Implements `diagnostics_v3/prespec_v3.json`, locked 2026-11-10T05:38:05Z,
before any v3 metric. One question: does change over time add to v1's static
features? Nothing is rescored; every LLM number is the cached Ollama P(Yes).

The output is a flag for a person to review, not a diagnosis. Labels are eRisk
dataset task labels.

Label-free per-chunk statistics are computed once. The level cuts are fit on
the outer training users only, so the encoding is built per fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import v2_temporal_features as v2

SIGNALS = v2.SIGNALS
SPEC = 'd14'
WIDTH = 14
N_CHUNKS = v2.HORIZON_DAYS // WIDTH          # 6
DECISION_DAYS = (14, 28, 56, 84)
DECISION_CHUNKS = {day: day // WIDTH for day in DECISION_DAYS}   # 1, 2, 4, 6
PERCENTILES = (50, 75, 90)
CHANGES = ('lower', 'higher')                 # 'same' is the dropped reference

LEVEL_COLUMNS = ['lvl__%s__%d' % (s, k) for s in SIGNALS for k in (1, 2, 3)]
CHANGE_COLUMNS = ['chg__%s__%s' % (s, c) for s in SIGNALS for c in CHANGES]
CHANGE_INT_COLUMNS = ['chgint__%s' % s for s in SIGNALS]
V1_COLUMNS = list(v2.V1_STATIC_COLUMNS)

FEATURE_SETS = {
    'v1_static': V1_COLUMNS,
    'v1_plus_change': V1_COLUMNS + CHANGE_COLUMNS + ['empty_chunk'],
    'v1_plus_change_int': V1_COLUMNS + CHANGE_INT_COLUMNS + ['empty_chunk'],
    'v3_levels': LEVEL_COLUMNS + ['empty_chunk', 'n_posts'],
    'v3_all': V1_COLUMNS + LEVEL_COLUMNS + CHANGE_COLUMNS + ['empty_chunk'],
}

V1_LLM_MEANS = ['v1__%s__mean' % s for s in SIGNALS]
V1_STYLE = ['v1__polarity_flip', 'v1__polarity_jump', 'v1__distress_burst', 'v1__absolutist']
DROP_ONE_GROUPS = {
    'v1_llm_means': V1_LLM_MEANS,
    'v1_style': V1_STYLE,
    'llm_levels': LEVEL_COLUMNS,
    'llm_changes': CHANGE_COLUMNS,
    'empty_chunk': ['empty_chunk'],
}


# --------------------------------------------------------------------------- #
# Label-free statistics, computed once
# --------------------------------------------------------------------------- #
def chunk_means(history: v2.History) -> np.ndarray:
    """(N_CHUNKS, 5) mean P(Yes) per d14 chunk. NaN where the chunk is empty."""
    out = np.full((N_CHUNKS, len(SIGNALS)), np.nan)
    if history.size == 0:
        return out
    assignment = v2.chunk_assignment(history, SPEC, N_CHUNKS)
    for chunk in range(N_CHUNKS):
        members = assignment == chunk
        if members.any():
            out[chunk] = history.scores[members].mean(axis=0)
    return out


class Statistics:
    """Per-user chunk means (users x chunks x signals) and the v2 d14 bank,
    which supplies v1_static on the prefix and the chunk post count."""

    def __init__(self, histories: dict, user_ids):
        self.user_ids = [str(u) for u in user_ids]
        self.means = np.stack([chunk_means(histories[u]) for u in self.user_ids])
        self.position = {u: i for i, u in enumerate(self.user_ids)}
        bank = v2.raw_bank(histories, SPEC)
        self.v1 = {k: v2.rows_at(bank, k, self.user_ids) for k in range(1, N_CHUNKS + 1)}

    def rows(self, user_ids) -> np.ndarray:
        return np.array([self.position[str(u)] for u in user_ids], dtype=int)


# --------------------------------------------------------------------------- #
# Fold-specific rules
# --------------------------------------------------------------------------- #
def fit_cuts(means: np.ndarray) -> np.ndarray:
    """(5, 3) cuts per signal: p50, p75, p90 over non-empty chunks given.

    `means` must hold training users only. Empty chunks (NaN) are ignored.
    """
    cuts = np.zeros((len(SIGNALS), len(PERCENTILES)))
    for s in range(len(SIGNALS)):
        values = means[:, :, s].ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError('no non-empty chunk to fit cuts on')
        cuts[s] = np.percentile(values, PERCENTILES)
    return cuts


def levels(means: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Level 0-3 per chunk and signal; -1 for an empty chunk.

    0: mean <= p50, 1: <= p75, 2: <= p90, 3: above p90.
    """
    out = np.full(means.shape, -1, dtype=int)
    for s in range(len(SIGNALS)):
        values = means[..., s]
        finite = np.isfinite(values)
        level = np.searchsorted(cuts[s], values, side='left')
        out[..., s] = np.where(finite, level, -1)
    return out


def change_codes(level_matrix: np.ndarray, chunk: int) -> np.ndarray:
    """-1 lower, 0 same, +1 higher at a 1-based chunk, per user and signal.

    Compared with the user's most recent earlier non-empty chunk. Same when the
    current chunk is empty or no earlier non-empty chunk exists.
    `level_matrix` is (users, chunks, signals). Only chunks <= `chunk` are read.
    """
    users = level_matrix.shape[0]
    out = np.zeros((users, len(SIGNALS)), dtype=int)
    current = level_matrix[:, chunk - 1, :]
    for u in range(users):
        if (current[u] < 0).all():
            continue
        earlier = level_matrix[u, :chunk - 1, :]
        non_empty = np.flatnonzero((earlier >= 0).all(axis=1))
        if non_empty.size == 0:
            continue
        previous = earlier[non_empty[-1]]
        out[u] = np.sign(current[u] - previous)
    return out


def encode(stats: Statistics, user_ids, chunk: int, cuts: np.ndarray) -> pd.DataFrame:
    """Every v3 column for these users at a 1-based d14 chunk, with given cuts."""
    user_ids = [str(u) for u in user_ids]
    index = stats.rows(user_ids)
    means = stats.means[index][:, :chunk, :]           # nothing after the chunk
    level_matrix = levels(means, cuts)
    current = level_matrix[:, chunk - 1, :]
    change = change_codes(level_matrix, chunk)
    frame = pd.DataFrame(index=pd.Index(user_ids, name='user_id'))
    for s, signal in enumerate(SIGNALS):
        for k in (1, 2, 3):
            frame['lvl__%s__%d' % (signal, k)] = (current[:, s] == k).astype(float)
        frame['chg__%s__lower' % signal] = (change[:, s] < 0).astype(float)
        frame['chg__%s__higher' % signal] = (change[:, s] > 0).astype(float)
        frame['chgint__%s' % signal] = change[:, s].astype(float)
    frame['empty_chunk'] = (current < 0).all(axis=1).astype(float)
    v1 = stats.v1[chunk].loc[user_ids]
    frame['n_posts'] = v1['n_posts'].astype(float).to_numpy()
    for column in V1_COLUMNS:
        frame[column] = v1[column].astype(float).to_numpy()
    return frame


def fold_matrix(stats: Statistics, train_ids, valid_ids, chunk: int, feature_set: str):
    """Fit cuts on training users, then encode both sides at one chunk."""
    cuts = fit_cuts(stats.means[stats.rows(train_ids)])
    columns = FEATURE_SETS[feature_set]
    return (encode(stats, train_ids, chunk, cuts)[columns],
            encode(stats, valid_ids, chunk, cuts)[columns], cuts)


def feature_table_rows() -> list[dict]:
    return [
        {'feature': 'lvl__<signal>', 'type': 'ordinal 0-3, one-hot, level 0 dropped',
         'computed': 'mean cached P(Yes) of the writings in the 14-day chunk, cut at the training-fold '
                     '50th/75th/90th percentiles of non-empty chunk means', 'window': '14-day chunk',
         'source': 'LLM cache'},
        {'feature': 'chg__<signal>', 'type': 'lower/same/higher, one-hot, same dropped',
         'computed': 'level against the user\'s most recent earlier non-empty chunk; same when the chunk '
                     'is empty or nothing earlier exists', 'window': 'chunk vs last non-empty chunk',
         'source': 'LLM cache'},
        {'feature': 'chgint__<signal>', 'type': 'integer -1/0/+1 (sensitivity only)',
         'computed': 'the same change as one signed integer', 'window': 'chunk vs last non-empty chunk',
         'source': 'LLM cache'},
        {'feature': 'empty_chunk', 'type': 'integer 0/1', 'computed': '1 when the chunk has no eligible writing',
         'window': '14-day chunk', 'source': 'rule'},
        {'feature': 'n_posts', 'type': 'integer', 'computed': 'eligible writings in the chunk',
         'window': '14-day chunk', 'source': 'rule'},
        {'feature': 'v1__<signal>__mean', 'type': 'float', 'computed': 'mean cached P(Yes) over the prefix, as v1',
         'window': 'prefix to the decision', 'source': 'LLM cache'},
        {'feature': 'v1__polarity_flip / polarity_jump / distress_burst / absolutist', 'type': 'float',
         'computed': 'v1 style features on the prefix', 'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'v1__n_posts / v1__has_posts', 'type': 'integer', 'computed': 'v1 controls on the prefix',
         'window': 'prefix to the decision', 'source': 'rule'},
    ]
