"""v5 features: use the text and the cached LLM scores properly (exploratory).

Implements `diagnostics_v5/prespec_v5.json`, locked 2026-11-10T13:32:12Z,
before any v5 metric. No rescoring, no new LLM call. Output is a flag for a
person to review, not a diagnosis. Labels are eRisk dataset task labels.

Everything for a decision day uses writings with delta <= that day only.
Text models are fit on training users only and cross-fitted for stacking.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import v2_temporal_features as v2
import erisk_sequential_features as seq

SEED = 42
SIGNALS = v2.SIGNALS
DECISION_DAYS = (14, 28, 84)

CLINICAL_STEMS = ('depress', 'anxi', 'therap', 'psychiatr', 'psycholog', 'medicat', 'meds',
                  'antidepress', 'ssri', 'snri', 'zoloft', 'prozac', 'lexapro', 'sertraline',
                  'fluoxetine', 'escitalopram', 'citalopram', 'wellbutrin', 'bupropion',
                  'venlafaxine', 'effexor', 'diagnos', 'bipolar', 'ptsd', 'adhd', 'ocd', 'suicid',
                  'selfharm', 'counsel', 'mdd')
_CLIN_WORDS = '|'.join(sorted(re.escape(w) for w in seq.CLINICAL_CANDIDATES))
_CLIN_STEMS = '|'.join(re.escape(s) for s in CLINICAL_STEMS)
CLINICAL_RE = re.compile(r"\b(?:(?:%s)\b|(?:%s)\w*)" % (_CLIN_WORDS, _CLIN_STEMS), re.IGNORECASE)

STYLE = ['v1__polarity_flip', 'v1__polarity_jump', 'v1__distress_burst', 'v1__absolutist',
         'v1__n_posts', 'v1__has_posts']
TAIL = ([f'tail__{s}__{stat}' for s in SIGNALS for stat in ('mean', 'max', 'top3', 'gt25', 'gt50')]
        + ['tail__any_gt50', 'tail__mean_rowmax', 'tail__two_gt50'])
POST_AGG = ['post__mean', 'post__top5', 'post__p90']

FEATURE_SETS = {
    'v1_static': list(v2.V1_STATIC_COLUMNS),
    'llm_tail': TAIL + ['v1__n_posts', 'v1__has_posts'],
    'style_llm_tail': STYLE + TAIL,
    'text_doc_noclin': ['doc_noclin'],
    'text_doc_clin': ['doc_clin'],
    'text_post_noclin': POST_AGG,
    'v5_stack': STYLE + TAIL + ['doc_noclin'] + POST_AGG,
}
SCORE_ONLY = {'text_doc_noclin': 'doc_noclin', 'text_doc_clin': 'doc_clin'}


def strip_clinical(text: str) -> str:
    return CLINICAL_RE.sub(' ', text)


# --------------------------------------------------------------------------- #
# Label-free per-user tables
# --------------------------------------------------------------------------- #
def tail_rows(writings: pd.DataFrame, day: int, user_ids) -> pd.DataFrame:
    prefix = writings[writings.delta_days <= day]
    rows = {}
    for user_id, group in prefix.groupby('user_id', sort=True):
        scores = group[list(SIGNALS)].to_numpy(float)
        row = {}
        for k, s in enumerate(SIGNALS):
            col = np.sort(scores[:, k])[::-1]
            row[f'tail__{s}__mean'] = col.mean()
            row[f'tail__{s}__max'] = col[0]
            row[f'tail__{s}__top3'] = col[:3].mean()
            row[f'tail__{s}__gt25'] = (col > 0.25).mean()
            row[f'tail__{s}__gt50'] = (col > 0.5).mean()
        rowmax = scores.max(axis=1)
        row['tail__any_gt50'] = (rowmax > 0.5).mean()
        row['tail__mean_rowmax'] = rowmax.mean()
        row['tail__two_gt50'] = ((scores > 0.5).sum(axis=1) >= 2).mean()
        rows[user_id] = row
    frame = pd.DataFrame.from_dict(rows, orient='index').reindex([str(u) for u in user_ids])
    return frame.fillna(0.0)


def prefix_posts(texts: pd.DataFrame, day: int) -> pd.DataFrame:
    """user_id, text_clin, text_noclin for every writing with delta <= day."""
    return texts[texts.delta_days <= day][['user_id', 'text_clin', 'text_noclin']]


def prefix_docs(posts: pd.DataFrame, user_ids) -> pd.DataFrame:
    grouped = posts.groupby('user_id', sort=True)
    docs = pd.DataFrame({'clin': grouped.text_clin.apply(' '.join),
                         'noclin': grouped.text_noclin.apply(' '.join)})
    return docs.reindex([str(u) for u in user_ids]).fillna('')


# --------------------------------------------------------------------------- #
# Text models. Hashed counts are stateless (no vocabulary is learned), so they
# can be computed once. Everything learned (document-frequency filter, idf,
# weights) is fit on training users only.
# --------------------------------------------------------------------------- #
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

N_HASH = 2 ** 19


def hasher() -> HashingVectorizer:
    return HashingVectorizer(ngram_range=(1, 2), n_features=N_HASH, alternate_sign=False,
                             norm=None, lowercase=True, dtype=np.float32)


class TextModel:
    """min_df filter + sublinear TF-IDF + logistic, all fit on the rows given."""

    def __init__(self, min_df: int, C: float, balanced: bool = False):
        self.min_df, self.C, self.balanced = min_df, C, balanced

    def fit(self, X, y):
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        self.keep = np.flatnonzero(df >= self.min_df)
        Xk = X[:, self.keep]
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(Xk)
        self.model = LogisticRegression(solver='liblinear', C=self.C, max_iter=2000, random_state=SEED,
                                        class_weight='balanced' if self.balanced else None)
        self.model.fit(self.tfidf.transform(Xk), np.asarray(y, dtype=int))
        return self

    def score(self, X):
        if X.shape[0] == 0:
            return np.zeros(0)
        return self.model.predict_proba(self.tfidf.transform(X[:, self.keep]))[:, 1]


class TextDay:
    """Hashed counts for one decision day: user documents and single writings."""

    def __init__(self, texts: pd.DataFrame, day: int, user_ids):
        self.user_ids = [str(u) for u in user_ids]
        self.row = {u: i for i, u in enumerate(self.user_ids)}
        posts = texts[texts.delta_days <= day].reset_index(drop=True)
        h = hasher()
        self.post_users = posts.user_id.astype(str).to_numpy()
        self.post_X = h.transform(posts.text_noclin.tolist()).tocsr()
        self.doc = {}
        for v in ('clin', 'noclin'):
            Xp = self.post_X if v == 'noclin' else h.transform(posts.text_clin.tolist()).tocsr()
            # A user document = the sum of its writings' counts (same as joining the text,
            # except no bigram spans two writings).
            index = np.array([self.row[u] for u in self.post_users], dtype=int)
            agg = sparse.csr_matrix((np.ones(len(index), dtype=np.float32), (index, np.arange(len(index)))),
                                    shape=(len(self.user_ids), len(index)))
            self.doc[v] = (agg @ Xp).tocsr()

    def doc_rows(self, ids):
        return np.array([self.row[str(u)] for u in ids], dtype=int)

    def post_mask(self, ids):
        return np.isin(self.post_users, np.array([str(u) for u in ids]))


def tune_doc_C(Xdoc, y) -> float:
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    y = np.asarray(y, dtype=int)
    best, best_auc = 1.0, -1.0
    from sklearn.metrics import roc_auc_score
    for C in (0.1, 1.0, 10.0):
        aucs = []
        for a, b in inner.split(np.zeros(len(y)), y):
            m = TextModel(3, C).fit(Xdoc[a], y[a])
            aucs.append(roc_auc_score(y[b], m.score(Xdoc[b])))
        if np.mean(aucs) > best_auc:
            best, best_auc = C, float(np.mean(aucs))
    return best


def post_aggregates(users: np.ndarray, probs: np.ndarray, ids) -> pd.DataFrame:
    frame = pd.DataFrame({'user_id': users, 'p': probs})
    grouped = frame.groupby('user_id').p
    out = pd.DataFrame({'post__mean': grouped.mean(),
                        'post__top5': grouped.apply(lambda s: s.nlargest(5).mean()),
                        'post__p90': grouped.quantile(0.9)})
    return out.reindex([str(u) for u in ids]).fillna(0.0)


def crossfit_text(text: TextDay, train_ids, valid_ids, y_train: pd.Series):
    """Cross-fitted text features. Training users get inner out-of-fold values;
    test users get values from models refit on the whole training fold."""
    train_ids, valid_ids = [str(u) for u in train_ids], [str(u) for u in valid_ids]
    y = y_train.astype(int).to_numpy()
    label = dict(zip(train_ids, y))
    rows_tr = text.doc_rows(train_ids)
    C = {v: tune_doc_C(text.doc[v][rows_tr], y) for v in ('noclin', 'clin')}
    cols = ['doc_noclin', 'doc_clin'] + POST_AGG

    def post_fit(ids):
        mask = text.post_mask(ids)
        return TextModel(5, 1.0, balanced=True).fit(
            text.post_X[mask], np.array([label[u] for u in text.post_users[mask]]))

    def post_score(model, ids):
        mask = text.post_mask(ids)
        return post_aggregates(text.post_users[mask], model.score(text.post_X[mask]), ids)

    out_train = pd.DataFrame(index=train_ids, columns=cols, dtype=float)
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for a, b in inner.split(np.zeros(len(train_ids)), y):
        fit_ids, score_ids = [train_ids[i] for i in a], [train_ids[i] for i in b]
        for v in ('noclin', 'clin'):
            m = TextModel(3, C[v]).fit(text.doc[v][text.doc_rows(fit_ids)], y[a])
            out_train.loc[score_ids, 'doc_' + v] = m.score(text.doc[v][text.doc_rows(score_ids)])
        out_train.loc[score_ids, POST_AGG] = post_score(post_fit(fit_ids), score_ids).to_numpy()

    out_valid = pd.DataFrame(index=valid_ids, columns=cols, dtype=float)
    for v in ('noclin', 'clin'):
        m = TextModel(3, C[v]).fit(text.doc[v][rows_tr], y)
        out_valid['doc_' + v] = m.score(text.doc[v][text.doc_rows(valid_ids)])
    out_valid[POST_AGG] = post_score(post_fit(train_ids), valid_ids).to_numpy()
    return out_train.astype(float), out_valid.astype(float), C
