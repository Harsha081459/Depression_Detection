"""Thread-context features for eRisk Task 2 (v5, exploratory).

Built from private/cohort_v5/context/, written by v5_context_extract.py.
A row for decision day D uses the user's own writings with delta <= D and only
the other-user writings that existed by then: a reply counts only if it was
posted by origin + D. Disclosure writings, and the replies to and parents of
them, are left out. NRC counts in context exclude clinical words; clinical
counts are kept only for a sensitivity set.

A flag from these features is for human review, not a diagnosis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import erisk_sequential_features as seq
from v5_context_extract import (REL_COMMENT_OTHER, REL_MISSING, REL_ROOT_OTHER, REL_SELF,
                                REL_SUBMISSION)

EMOTIONS = seq.EMOTIONS
DAY = 86400

STRUCTURE = ['ctx_share_submission', 'ctx_share_root_other', 'ctx_share_comment_other',
             'ctx_share_self', 'ctx_share_missing', 'ctx_log_prior_others', 'ctx_log_prior_authors']
RECEPTION = ['ctx_log_replies_per_writing', 'ctx_share_answered', 'ctx_share_submissions_answered',
             'ctx_log_first_reply_hours', 'ctx_log_repliers_per_writing', 'ctx_recurring_reply_share',
             'ctx_reciprocity']
REPLY_CONTENT = (['rep_emo__%s' % e for e in EMOTIONS] +
                 ['rep_support_rate', 'rep_second_person_rate', 'rep_question_share', 'rep_mean_log_tokens'])
PARENT_CONTENT = (['par_emo__%s' % e for e in EMOTIONS] +
                  ['par_question_share', 'par_mean_log_tokens'])
CLINICAL_SENS = ['rep_clinical_rate', 'par_clinical_rate']
CONTEXT = STRUCTURE + RECEPTION + REPLY_CONTENT + PARENT_CONTENT
ALL_COLUMNS = CONTEXT + CLINICAL_SENS

GROUPS = {'ctx_structure': STRUCTURE, 'ctx_reception': RECEPTION,
          'ctx_reply_content': REPLY_CONTENT, 'ctx_parent_content': PARENT_CONTENT}


def load(context_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.read_parquet(Path(context_dir) / 'targets.parquet')
    context = pd.read_parquet(Path(context_dir) / 'context.parquet')
    targets['user_id'] = targets.user_id.astype(str)
    return targets, context


def origin_seconds(origins: pd.Series) -> pd.Series:
    """origins: user_id -> tz-aware UTC timestamp (v2.load_writings)."""
    utc = pd.to_datetime(origins, utc=True)
    return pd.Series(utc.dt.tz_convert(None).to_numpy().astype('datetime64[s]').astype(np.int64),
                     index=origins.index.astype(str))


def _shares(frame: pd.DataFrame, prefix: str, by: str) -> pd.DataFrame:
    grouped = frame.groupby(by, sort=True)
    tokens = grouped.n_tokens.sum().astype(float)
    safe = tokens.where(tokens > 0, 1.0)
    out = pd.DataFrame(index=tokens.index)
    for e in EMOTIONS:
        out['%s_emo__%s' % (prefix, e)] = np.where(tokens > 0, grouped['emo_%s' % e].sum() / safe, 0.0)
    out['%s_clinical_rate' % prefix] = np.where(tokens > 0, grouped.clinical.sum() / safe, 0.0)
    out['%s_question_share' % prefix] = grouped.question.mean()
    out['%s_mean_log_tokens' % prefix] = grouped.n_tokens.apply(lambda s: float(np.log1p(s).mean()))
    return out


def context_rows(targets: pd.DataFrame, context: pd.DataFrame, origins_sec: pd.Series,
                 user_ids, day: float | None) -> pd.DataFrame:
    """One row per user at decision day `day` (None = whole history)."""
    origin = targets.user_id.map(origins_sec)
    if origin.isna().any():
        raise ValueError('a target writing has no origin')
    cutoff = origin + (day * DAY if day is not None else np.inf)
    own = targets[(targets.ts <= cutoff) & ~targets.disclosure].copy()
    own['cutoff'] = cutoff[own.index]

    out = pd.DataFrame(index=pd.Index(sorted(set(map(str, user_ids))), name='user_id'))
    grouped = own.groupby('user_id', sort=True)
    for code, column in ((REL_SUBMISSION, 'ctx_share_submission'), (REL_ROOT_OTHER, 'ctx_share_root_other'),
                         (REL_COMMENT_OTHER, 'ctx_share_comment_other'), (REL_SELF, 'ctx_share_self'),
                         (REL_MISSING, 'ctx_share_missing')):
        out[column] = own.relation.eq(code).groupby(own.user_id).mean()
    comments = own[own.relation != REL_SUBMISSION]
    out['ctx_log_prior_others'] = np.log1p(comments.prior_others_in_thread).groupby(comments.user_id).mean()
    out['ctx_log_prior_authors'] = np.log1p(comments.prior_other_authors_in_thread).groupby(
        comments.user_id).mean()

    links = context.merge(own[['source_row', 'user_id', 'ts', 'cutoff', 'relation']],
                          left_on='target_row', right_on='source_row', suffixes=('', '_target'))
    replies = links[(links.role == 'reply') & (links.ts <= links.cutoff)]
    parents = links[links.role == 'parent']

    n_own = grouped.size().astype(float)
    per_writing = replies.groupby('target_row').agg(n=('ts', 'size'), first=('ts', 'min'))
    answered = own.source_row.isin(per_writing.index)
    out['ctx_log_replies_per_writing'] = np.log1p(replies.groupby('user_id').size() / n_own)
    out['ctx_share_answered'] = answered.groupby(own.user_id).mean()
    subs = own.relation == REL_SUBMISSION
    out['ctx_share_submissions_answered'] = answered[subs].groupby(own.user_id[subs]).mean()
    latency = (per_writing['first'] - own.set_index('source_row').ts.reindex(per_writing.index)) / 3600.0
    user_of_row = own.set_index('source_row').user_id
    out['ctx_log_first_reply_hours'] = np.log1p(latency.clip(lower=0)).groupby(
        user_of_row.reindex(per_writing.index)).median()
    out['ctx_log_repliers_per_writing'] = np.log1p(replies.groupby('user_id').author.nunique() / n_own)
    by_author = replies.groupby(['user_id', 'author']).size()
    recurring = by_author[by_author >= 2].groupby(level='user_id').sum()
    out['ctx_recurring_reply_share'] = recurring / replies.groupby('user_id').size()

    first_reply = replies.groupby(['user_id', 'author']).ts.min().rename('first_reply_ts').reset_index()
    answering = parents.merge(first_reply, on=['user_id', 'author'], how='left')
    answering['returned'] = answering.first_reply_ts < answering.ts_target
    out['ctx_reciprocity'] = answering.groupby('user_id').returned.mean()

    reply_content = _shares(replies, 'rep', 'user_id')
    rgroup = replies.groupby('user_id')
    reply_content['rep_support_rate'] = rgroup.support.mean()
    reply_content['rep_second_person_rate'] = rgroup.second_person.mean()
    parent_content = _shares(parents, 'par', 'user_id')
    out = out.join(reply_content, how='left').join(parent_content, how='left')
    return out.reindex(columns=ALL_COLUMNS).fillna(0.0).astype(float)
