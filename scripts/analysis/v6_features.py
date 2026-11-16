"""v6 research features (exploratory, logged trials).

Every block here is either label-free (computed once from the text, a lexicon,
a public word-vector table, or a model trained on an external dataset that has
no eRisk labels), or fit on training users only inside each fold.

Text is the masked eligible writings (disclosure writings removed), cleaned
with the strict rule: clinical words, URLs, image markup and residual medical
words removed. Nothing past the decision day is read. Output of any model built
on these features is a flag for a person to review, not a diagnosis.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import v5_features as v5

SEED = 42

URL_RE = re.compile(r'https?://\S+|www\.\S+|\S+\.(?:com|org|net|it|png|jpg|jpeg|gif|webp)\b'
                    r'|&amp;#x200b;|x200b|&amp;|&gt;|&lt;', re.I)
MEDICAL_RE = re.compile(r'\b(?:dose\w*|dr|mg|doctor\w*|pill\w*|prescri\w*|hospital\w*|clinic\w*|'
                        r'psych\w*|redd|preview|webp|png|jpg|gif|imgur|youtube|youtu)\b', re.I)


def strict_clean(text: str) -> str:
    return MEDICAL_RE.sub(' ', URL_RE.sub(' ', v5.strip_clinical(str(text))))


def url_count(text: str) -> int:
    return len(re.findall(r'https?://|www\.', str(text)))


# --------------------------------------------------------------------------- #
# Psycholinguistic lexicons (published word lists, label-free)
# --------------------------------------------------------------------------- #
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
LEXICONS = {
    'i_sing': {'i', 'me', 'my', 'mine', 'myself', "i'm", 'im', "i've", 'ive', "i'd", "i'll"},
    'we': {'we', 'us', 'our', 'ours', 'ourselves', "we're"},
    'you': {'you', 'your', 'yours', 'yourself', "you're", 'u'},
    'they': {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their'},
    # Al-Mosaiwi & Johnstone (2018) absolutist dictionary
    'absolutist': {'absolutely', 'all', 'always', 'complete', 'completely', 'constant', 'constantly',
                   'definitely', 'entire', 'ever', 'every', 'everyone', 'everything', 'full', 'must',
                   'never', 'nothing', 'totally', 'whole'},
    'negation': {'no', 'not', 'never', "don't", 'dont', "can't", 'cant', "won't", 'nobody', 'none',
                 'nothing', 'nowhere', "isn't", "doesn't", "didn't", "wasn't"},
    'death': {'die', 'died', 'dying', 'death', 'dead', 'kill', 'killed', 'killing', 'funeral', 'grave',
              'end', 'disappear', 'gone'},
    'tired': {'tired', 'exhausted', 'sleep', 'sleeping', 'asleep', 'awake', 'insomnia', 'nap', 'energy',
              'fatigue', 'bed'},
    'lonely': {'alone', 'lonely', 'loneliness', 'isolated', 'nobody', 'friendless', 'ignored'},
    'worthless': {'worthless', 'useless', 'failure', 'pathetic', 'stupid', 'hate', 'ugly', 'burden',
                  'disappointment', 'loser'},
    'swear': {'fuck', 'fucking', 'fucked', 'shit', 'shitty', 'damn', 'hell', 'bitch', 'crap', 'ass'},
    'feel': {'feel', 'feels', 'feeling', 'felt', 'feelings'},
    'past_aux': {'was', 'were', 'had', 'did', 'used'},
}
PSY_COLUMNS = ([f'psy__{k}' for k in LEXICONS]
               + ['psy__log_tokens_per_post', 'psy__type_token', 'psy__question_per_post',
                  'psy__exclaim_per_post', 'psy__i_start_share'])
VADER_COLUMNS = ['vader__mean', 'vader__sd', 'vader__min', 'vader__neg_share', 'vader__pos_share',
                 'vader__neg_mean']
ARTEFACT_COLUMNS = ['art__url_per_post', 'art__nonascii_share']

EMPATH_SUBSET = ['sadness', 'negative_emotion', 'positive_emotion', 'death', 'pain', 'shame',
                 'nervousness', 'suffering', 'emotional', 'fear', 'sleep', 'weakness', 'hate',
                 'disappointment', 'rage', 'optimism', 'family', 'friends', 'love', 'work', 'school',
                 'money', 'achievement', 'leisure', 'sports', 'music', 'party', 'celebration',
                 'cheerfulness', 'irritability', 'exasperation', 'violence', 'injury', 'body',
                 'medical_emergency', 'help', 'trust', 'dispute', 'swearing_terms', 'internet']


def post_table(texts: pd.DataFrame) -> pd.DataFrame:
    """Per-writing label-free counts. `texts` has user_id, delta_days, raw text."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    rows = []
    for raw, clean in zip(texts.raw.tolist(), texts.clean.tolist()):
        tokens = TOKEN.findall(clean.lower())
        bag = Counter(tokens)
        row = {f'cnt__{k}': sum(bag[w] for w in words) for k, words in LEXICONS.items()}
        row['n_tokens'] = len(tokens)
        row['n_types'] = len(bag)
        row['questions'] = clean.count('?')
        row['exclaims'] = clean.count('!')
        row['i_start'] = int(bool(tokens) and tokens[0] in ('i', "i'm", 'im', "i've"))
        row['vader'] = vader.polarity_scores(clean[:5000])['compound'] if tokens else 0.0
        row['urls'] = url_count(raw)
        row['nonascii'] = (sum(ord(c) > 127 for c in raw) / max(1, len(raw))) > 0.05
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(0, 'user_id', texts.user_id.to_numpy())
    out.insert(1, 'delta_days', texts.delta_days.to_numpy())
    return out


def user_lexicon_rows(posts: pd.DataFrame, day: int, user_ids) -> pd.DataFrame:
    p = posts[posts.delta_days <= day]
    g = p.groupby('user_id')
    tok = g.n_tokens.sum().clip(lower=1)
    out = pd.DataFrame(index=tok.index)
    for k in LEXICONS:
        out[f'psy__{k}'] = g[f'cnt__{k}'].sum() / tok
    n = g.size()
    out['psy__log_tokens_per_post'] = np.log1p(g.n_tokens.mean())
    out['psy__type_token'] = (g.n_types.sum() / tok).clip(upper=1)
    out['psy__question_per_post'] = g.questions.sum() / n
    out['psy__exclaim_per_post'] = g.exclaims.sum() / n
    out['psy__i_start_share'] = g.i_start.mean()
    out['vader__mean'] = g.vader.mean()
    out['vader__sd'] = g.vader.std().fillna(0)
    out['vader__min'] = g.vader.min()
    out['vader__neg_share'] = g.vader.apply(lambda s: (s < -0.05).mean())
    out['vader__pos_share'] = g.vader.apply(lambda s: (s > 0.05).mean())
    out['vader__neg_mean'] = g.vader.apply(lambda s: s[s < -0.05].mean() if (s < -0.05).any() else 0.0)
    out['art__url_per_post'] = g.urls.sum() / n
    out['art__nonascii_share'] = g.nonascii.mean()
    return out.reindex([str(u) for u in user_ids]).fillna(0.0)


def user_empath_rows(texts: pd.DataFrame, day: int, user_ids, categories=None) -> pd.DataFrame:
    from empath import Empath
    lexicon = Empath()
    p = texts[texts.delta_days <= day]
    docs = p.groupby('user_id').clean.apply(' '.join)
    rows = {}
    for u, doc in docs.items():
        scores = lexicon.analyze(doc, categories=categories, normalize=True) or {}
        rows[u] = scores
    out = pd.DataFrame.from_dict(rows, orient='index').fillna(0.0)
    out.columns = [f'empath__{c}' for c in out.columns]
    return out.reindex([str(u) for u in user_ids]).fillna(0.0)


def user_vector_rows(texts: pd.DataFrame, day: int, user_ids, nlp) -> pd.DataFrame:
    """Mean spaCy en_core_web_lg word vector over the user's cleaned prefix tokens."""
    p = texts[texts.delta_days <= day]
    cache: dict[str, np.ndarray | None] = {}
    dim = nlp.vocab.vectors.shape[1]
    rows = {}
    for u, g in p.groupby('user_id'):
        total, count = np.zeros(dim, dtype=np.float64), 0
        for doc in g.clean.tolist():
            for t in TOKEN.findall(doc.lower()):
                if t not in cache:
                    lex = nlp.vocab[t]
                    cache[t] = lex.vector.astype(np.float64) if lex.has_vector else None
                v = cache[t]
                if v is not None:
                    total += v
                    count += 1
        rows[u] = total / max(1, count)
    out = pd.DataFrame.from_dict(rows, orient='index')
    out.columns = [f'vec__{i}' for i in range(dim)]
    return out.reindex([str(u) for u in user_ids]).fillna(0.0)


# --------------------------------------------------------------------------- #
# External weak supervision: r/depression vs four other mental-health
# subreddits (OCD, ADHD, PTSD, Asperger's), 2019-2021. No eRisk label is used.
# --------------------------------------------------------------------------- #
def train_external(path, exclude_hashes: set) -> tuple:
    frame = pd.read_csv(path, usecols=['title', 'body', 'subreddit'])
    frame['raw'] = (frame.title.fillna('') + ' ' + frame.body.fillna('')).str.strip()
    norm = frame.raw.map(lambda t: re.sub(r'\W+', ' ', t.lower()).strip()[:300])
    import hashlib
    keep = ~norm.map(lambda n: len(n) >= 60 and hashlib.md5(n.encode()).hexdigest() in exclude_hashes)
    frame = frame[keep & frame.raw.str.len().gt(20)]
    frame['clean'] = frame.raw.map(strict_clean)
    y = frame.subreddit.eq('depression').astype(int).to_numpy()
    h = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm=None,
                          dtype=np.float32)
    X = h.transform(frame.clean.tolist())
    rng = np.random.default_rng(SEED)
    test = rng.random(len(y)) < 0.1
    tfidf = TfidfTransformer(sublinear_tf=True).fit(X[~test])
    model = LogisticRegression(solver='liblinear', C=1.0, class_weight='balanced', max_iter=2000,
                               random_state=SEED).fit(tfidf.transform(X[~test]), y[~test])
    from sklearn.metrics import roc_auc_score
    check = roc_auc_score(y[test], model.predict_proba(tfidf.transform(X[test]))[:, 1])
    return h, tfidf, model, {'external_posts': int(len(y)), 'external_depression_posts': int(y.sum()),
                             'external_holdout_auroc': float(check),
                             'removed_as_duplicates_of_erisk_text': int((~keep).sum())}


def user_external_rows(post_scores: pd.DataFrame, day: int, user_ids) -> pd.DataFrame:
    p = post_scores[post_scores.delta_days <= day]
    g = p.groupby('user_id').ext
    out = pd.DataFrame({'ext__mean': g.mean(), 'ext__top5': g.apply(lambda s: s.nlargest(5).mean()),
                        'ext__p90': g.quantile(0.9), 'ext__gt50': g.apply(lambda s: (s > 0.5).mean())})
    return out.reindex([str(u) for u in user_ids]).fillna(0.0)


# --------------------------------------------------------------------------- #
# In-fold document models
# --------------------------------------------------------------------------- #
class NBLogistic:
    """NB-weighted logistic (Wang & Manning 2012): binarised features scaled by the
    log-count ratio estimated on the training rows only."""

    def __init__(self, min_df: int, C: float):
        self.min_df, self.C = min_df, C

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        B = (X > 0).astype(np.float32).tocsr()
        df = np.asarray(B.sum(axis=0)).ravel()
        self.keep = np.flatnonzero(df >= self.min_df)
        B = B[:, self.keep]
        p = np.asarray(B[y == 1].sum(axis=0)).ravel() + 1.0
        q = np.asarray(B[y == 0].sum(axis=0)).ravel() + 1.0
        self.r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
        self.model = LogisticRegression(solver='liblinear', C=self.C, max_iter=2000, random_state=SEED)
        self.model.fit(B.multiply(self.r).tocsr(), y)
        return self

    def score(self, X):
        B = (X[:, self.keep] > 0).astype(np.float32)
        return self.model.predict_proba(B.multiply(self.r).tocsr())[:, 1]


def make_model(kind: str, C: float):
    if kind == 'nb':
        return NBLogistic(3, C)
    return v5.TextModel(3, C)


def tune_C(kind, X, y, grid=(0.1, 1.0, 10.0)) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, best_auc = grid[0], -1.0
    for C in grid:
        aucs = [roc_auc_score(y[b], make_model(kind, C).fit(X[a], y[a]).score(X[b]))
                for a, b in inner.split(np.zeros(len(y)), y)]
        if np.mean(aucs) > best_auc:
            best, best_auc = C, float(np.mean(aucs))
    return best


def crossfit_doc(kind: str, X, train_rows, valid_rows, y_train):
    """Inner out-of-fold scores for training rows, refit scores for valid rows."""
    y = np.asarray(y_train, dtype=int)
    Xtr = X[train_rows]
    C = tune_C(kind, Xtr, y)
    oof = np.zeros(len(train_rows))
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for a, b in inner.split(np.zeros(len(y)), y):
        oof[b] = make_model(kind, C).fit(Xtr[a], y[a]).score(Xtr[b])
    valid = make_model(kind, C).fit(Xtr, y).score(X[valid_rows])
    return oof, valid, C


def doc_matrix(texts: pd.DataFrame, day: int, user_ids, analyzer: str) -> sparse.csr_matrix:
    """Hashed counts of each user's prefix document (sum over writings)."""
    if analyzer == 'word':
        h = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 19, alternate_sign=False, norm=None,
                              dtype=np.float32)
    else:
        h = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 5), n_features=2 ** 19,
                              alternate_sign=False, norm=None, dtype=np.float32)
    p = texts[texts.delta_days <= day]
    Xp = h.transform(p.clean.tolist()).tocsr()
    row = {str(u): i for i, u in enumerate(user_ids)}
    idx = np.array([row[str(u)] for u in p.user_id], dtype=int)
    agg = sparse.csr_matrix((np.ones(len(idx), dtype=np.float32), (idx, np.arange(len(idx)))),
                            shape=(len(row), len(idx)))
    return (agg @ Xp).tocsr()
