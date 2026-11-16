"""v4 final-pass features for eRisk Task 2 (exploratory).

Implements `diagnostics_v4/prespec_v4.json`, locked 2026-11-10T06:03:16Z,
before any v4 metric. Puts every feature family built so far side by side:
rhythm and comment share (from timestamps), NRC emotion (from masked text),
v1 static (LLM prefix means and style), and v3 change (fold-fitted).

Nothing is rescored. A flag from these features is for human review, not a
diagnosis. Labels are eRisk dataset task labels.

Every row for a decision day uses writings with delta <= that day only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import v2_temporal_features as v2
import v3_features as v3
import erisk_sequential_features as seq

DECISION_DAYS = (14, 28, 56, 84)
HORIZON = v2.HORIZON_DAYS
EMOTIONS = seq.EMOTIONS

RHYTHM = ['log_posts_per_week', 'night_frac', 'weekend_frac', 'log_median_gap', 'gap_cv',
          'burst_1h_frac', 'hour_entropy', 'log_rate_change', 'log_n_last7d', 'log_n_last28d',
          'log_recent_rate_ratio']
COMMENT = ['comment_frac']
EMOTION = ['emo__%s' % e for e in EMOTIONS] + ['mean_log_tokens']
EMOTION_CLIN = ['emoclin__%s' % e for e in EMOTIONS] + ['mean_log_tokens']
V1 = list(v2.V1_STATIC_COLUMNS)
CHANGE = list(v3.CHANGE_COLUMNS) + ['empty_chunk']

FEATURE_SETS = {
    'rhythm': RHYTHM,
    'clocks': RHYTHM + COMMENT,
    'emotion': EMOTION,
    'v1_static': V1,
    'v4_text': EMOTION + V1,
    'v4_behaviour': RHYTHM + EMOTION + V1,
    'v4_all': RHYTHM + COMMENT + EMOTION + V1 + CHANGE,
    'emotion_clinical_sens': EMOTION_CLIN,
}
DROP_ONE_GROUPS = {
    'rhythm': RHYTHM,
    'comment': COMMENT,
    'emotion': EMOTION,
    'v1_llm_means': ['v1__%s__mean' % s for s in v2.SIGNALS],
    'v1_style': ['v1__polarity_flip', 'v1__polarity_jump', 'v1__distress_burst', 'v1__absolutist'],
    'change': CHANGE,
}


# --------------------------------------------------------------------------- #
# Rhythm from timestamps (all target writings, text never read)
# --------------------------------------------------------------------------- #
def timestamp_table(text_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(text_csv, usecols=['user_id', 'date', 'kind'], dtype={'user_id': str})
    utc = pd.to_datetime(frame['date'], utc=True, errors='raise')
    frame['ts_sec'] = utc.dt.tz_convert(None).to_numpy().astype('datetime64[s]').astype(np.int64)
    frame['hour'] = utc.dt.hour.astype(np.int64)                     # UTC clock
    frame['night'] = frame.hour.isin(seq.NIGHT_HOURS).astype(np.int64)
    frame['weekend'] = utc.dt.dayofweek.isin(seq.WEEKEND_DOW).astype(np.int64)
    frame['is_comment'] = frame.kind.eq('comment').astype(np.int64)
    origin = frame.groupby('user_id').ts_sec.transform('min')
    frame['delta_days'] = (frame.ts_sec - origin) / 86400.0
    frame = frame[frame.delta_days <= HORIZON + 1e-9]
    return frame.sort_values(['user_id', 'ts_sec'], kind='mergesort').reset_index(drop=True)


def rhythm_rows(stamps: pd.DataFrame, day: int) -> pd.DataFrame:
    """Clock features at each user's last target writing with delta <= day."""
    prefix = stamps[stamps.delta_days <= day]
    rows = {}
    for user_id, group in prefix.groupby('user_id', sort=True):
        clocks = seq.user_prefix_clocks(group.reset_index(drop=True))
        last = clocks.iloc[-1]
        rows[user_id] = {
            'log_posts_per_week': np.log1p(last.posts_per_week_so_far),
            'night_frac': last.night_frac_so_far,
            'weekend_frac': last.weekend_frac_so_far,
            'comment_frac': last.comment_frac_so_far,
            'log_median_gap': np.log1p(last.median_gap_so_far),
            'gap_cv': last.gap_cv_so_far,
            'burst_1h_frac': last.burst_1h_frac_so_far,
            'hour_entropy': last.hour_entropy_so_far,
            'log_rate_change': np.log(last.rate_change_so_far),
            'log_n_last7d': np.log1p(last.n_last7d),
            'log_n_last28d': np.log1p(last.n_last28d),
            'log_recent_rate_ratio': np.log(max(last.recent_rate_ratio, 1e-3)),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


# --------------------------------------------------------------------------- #
# Emotion from masked text
# --------------------------------------------------------------------------- #
def writing_emotions(texts: pd.Series, lexicon_path: Path) -> pd.DataFrame:
    """Per-writing NRC counts with and without clinical words, plus token count."""
    full = seq.load_nrc(lexicon_path)
    clinical = set(w for w in seq.CLINICAL_CANDIDATES if w in full)
    no_clin = {w: s for w, s in full.items() if w not in clinical}
    with_clin, _ = seq.count_emotions(texts, full, 'emoclin')
    without, _ = seq.count_emotions(texts, no_clin, 'emo')
    from sklearn.feature_extraction.text import CountVectorizer
    analyzer = CountVectorizer(lowercase=True, token_pattern=seq.TOKEN_PATTERN).build_analyzer()
    tokens = np.fromiter((len(analyzer(t)) for t in texts.tolist()), dtype=np.int64, count=len(texts))
    out = pd.concat([without, with_clin], axis=1)
    out['n_tokens'] = tokens
    return out


def emotion_rows(per_writing: pd.DataFrame, day: int) -> pd.DataFrame:
    """Token-weighted category shares over the text prefix, plus mean log tokens."""
    prefix = per_writing[per_writing.delta_days <= day]
    grouped = prefix.groupby('user_id', sort=True)
    tokens = grouped.n_tokens.sum().astype(float)
    out = pd.DataFrame(index=tokens.index)
    for e in EMOTIONS:
        for stem in ('emo', 'emoclin'):
            count = grouped['%s_%s_cnt' % (stem, e)].sum().astype(float)
            out['%s__%s' % (stem, e)] = np.where(tokens > 0, count / tokens.where(tokens > 0, 1), 0.0)
    out['mean_log_tokens'] = grouped.n_tokens.apply(lambda s: float(np.log1p(s).mean()))
    return out


# --------------------------------------------------------------------------- #
class Bank:
    """Static (label-free, fold-independent) columns per decision day, plus the
    v3 statistics for the fold-fitted change columns."""

    def __init__(self, root: Path, writings: pd.DataFrame, histories: dict, users: pd.DataFrame):
        root = Path(root)
        self.user_ids = [str(u) for u in users.user_id]
        stamps = timestamp_table(root / 'data' / 'processed' / 'erisk' / 'task2_target_text.csv')
        posts = pd.read_parquet(root / 'private' / 'cohort_v1' / 'eligible_posts.parquet',
                                columns=['post_id', 'text'])
        text = writings[['user_id', 'post_id', 'delta_days']].merge(posts, on='post_id', how='left',
                                                                     validate='one_to_one')
        if text.text.isna().any():
            raise ValueError('an eligible writing has no text after the join')
        emotions = writing_emotions(text.text.fillna('').astype(str),
                                    root / 'data' / 'raw' / 'lexicons' / 'nrc' /
                                    'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
        per_writing = pd.concat([text[['user_id', 'delta_days']].reset_index(drop=True),
                                 emotions.reset_index(drop=True)], axis=1)
        del text, posts
        self.v3stats = v3.Statistics(histories, self.user_ids)
        v2bank = v2.raw_bank(histories, v3.SPEC)
        self.static = {}
        for day in DECISION_DAYS:
            rhythm = rhythm_rows(stamps, day)
            emotion = emotion_rows(per_writing, day)
            v1 = v2.rows_at(v2bank, day // v3.WIDTH, self.user_ids)[V1]
            frame = pd.concat([rhythm.reindex(self.user_ids), emotion.reindex(self.user_ids),
                               v1.reindex(self.user_ids)], axis=1)
            frame = frame.fillna(0.0).astype(float)
            frame.index.name = 'user_id'
            self.static[day] = frame

    def fold(self, train_ids, valid_ids, day: int, feature_set: str):
        columns = FEATURE_SETS[feature_set]
        static = self.static[day]
        if not set(columns) & set(CHANGE):
            return static.loc[list(train_ids), columns], static.loc[list(valid_ids), columns]
        chunk = v3.DECISION_CHUNKS[day]
        cuts = v3.fit_cuts(self.v3stats.means[self.v3stats.rows(train_ids)])
        parts = []
        for ids in (train_ids, valid_ids):
            change = v3.encode(self.v3stats, ids, chunk, cuts)[CHANGE]
            frame = pd.concat([static.loc[list(ids)], change], axis=1)
            parts.append(frame[columns])
        return parts[0], parts[1]


def feature_table_rows() -> list[dict]:
    rows = [{'feature': c, 'family': 'rhythm', 'source': 'timestamps (all target writings)',
             'window': 'prefix to the decision'} for c in RHYTHM]
    rows.append({'feature': 'comment_frac', 'family': 'comment', 'source': 'kind (all target writings)',
                 'window': 'prefix to the decision'})
    rows += [{'feature': c, 'family': 'emotion', 'source': 'NRC lexicon, clinical words removed, masked text',
              'window': 'prefix to the decision'} for c in EMOTION]
    rows += [{'feature': c, 'family': 'v1_static', 'source': 'LLM cache' if '__mean' in c else 'rule',
              'window': 'prefix to the decision'} for c in V1]
    rows += [{'feature': c, 'family': 'change', 'source': 'LLM cache' if c != 'empty_chunk' else 'rule',
              'window': '14-day chunk vs last non-empty chunk'} for c in CHANGE]
    return rows
