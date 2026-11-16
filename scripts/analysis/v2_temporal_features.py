"""v2 temporal features for eRisk Task 2 (exploratory).

Implements `diagnostics_v2/prespec_v2.json`, locked 2026-11-10T04:41:27Z, before
any v2 metric. Nothing here is rescored: the five LLM numbers per writing are the
cached Ollama P(Yes) values. No prompt is read or edited.

The output of the models built on these features is a flag for a person to
review. It is not a diagnosis. Labels are eRisk dataset task labels.

Nothing in this module looks at a label, and nothing in a chunk's row uses a
writing that falls after that chunk's end. `test_v2_temporal_features.py`
enforces both.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SIGNALS = ('self_blame', 'helplessness', 'isolation', 'family_strain', 'emptiness')

HORIZON_DAYS = 84
HIGH_SCORE = 0.5                      # P(Yes) > 0.5 counts as a high writing
SHARE_CUTS = (0.10, 0.30)             # headline ordinal, share of high writings
MAX_CUTS = (0.25, 0.50, 0.75)         # secondary ordinal, unit-tested only
JUMP_CUTS = (0.25, 0.50)
DISTRESS_PERCENTILE = 75
SILENCE_RATIO = 1.5

LEVELS = (0, 1, 2, 3)
CHANGES = ('lower', 'same', 'higher')
TRENDS = ('falling', 'flat', 'rising')
SILENCE_CLASSES = ('shorter', 'similar', 'longer')

# Reference categories dropped from the one-hot encoding.
REFERENCE_LEVEL = 0
REFERENCE_CHANGE = 'same'
REFERENCE_TREND = 'flat'
REFERENCE_SILENCE = 'similar'

CHUNK_SPECS = {
    'd7': ('days', 7),
    'd14': ('days', 14),
    'd28': ('days', 28),
    'p10': ('posts', 10),
    'p20': ('posts', 20),
}
DAY_SPECS = tuple(k for k, (kind, _) in CHUNK_SPECS.items() if kind == 'days')
POST_SPECS = tuple(k for k, (kind, _) in CHUNK_SPECS.items() if kind == 'posts')
# A post-count spec needs an upper bound on chunk indices. The day-28 and day-84
# equivalents sit far below this cap; run_v2 asserts the chosen index is interior.
POST_CHUNK_CAP = 30

SHARE_FEATURES = ('absolutist', 'night', 'reply')   # share -> level, same cuts as the LLM share


# --------------------------------------------------------------------------- #
# Scalar rules
# --------------------------------------------------------------------------- #
def share_level(share: float) -> int:
    """Headline ordinal. 0 when nothing is high, then cuts at 0.10 and 0.30."""
    if not np.isfinite(share) or share <= 0:
        return 0
    if share <= SHARE_CUTS[0]:
        return 1
    if share <= SHARE_CUTS[1]:
        return 2
    return 3


def max_level(value: float) -> int:
    """Secondary ordinal from the chunk maximum. Unit-tested only, never fit."""
    if not np.isfinite(value):
        return 0
    for index, cut in enumerate(MAX_CUTS):
        if value <= cut:
            return index
    return 3


def frequency_level(n_posts: int) -> int:
    if n_posts <= 0:
        return 0
    if n_posts <= 2:
        return 1
    if n_posts <= 7:
        return 2
    return 3


def mood_level(flips: int) -> int:
    if flips <= 0:
        return 0
    if flips == 1:
        return 1
    if flips <= 3:
        return 2
    return 3


def jump_level(mean_absolute_jump: float) -> int:
    if not np.isfinite(mean_absolute_jump) or mean_absolute_jump <= 0:
        return 0
    if mean_absolute_jump <= JUMP_CUTS[0]:
        return 1
    if mean_absolute_jump <= JUMP_CUTS[1]:
        return 2
    return 3


def change_label(current: int, previous: int | None) -> str:
    """lower / same / higher against the previous chunk. First chunk is same."""
    if previous is None:
        return REFERENCE_CHANGE
    if current < previous:
        return 'lower'
    if current > previous:
        return 'higher'
    return 'same'


def trend_label(current: int, two_back: int | None) -> str:
    """falling / flat / rising against the level two chunks earlier."""
    if two_back is None:
        return REFERENCE_TREND
    if current < two_back:
        return 'falling'
    if current > two_back:
        return 'rising'
    return 'flat'


def silence_class(gap_days: float, baseline_days: float | None) -> str:
    """Compare one gap with 1.5 times the median earlier closed gap."""
    if baseline_days is None or not np.isfinite(baseline_days) or baseline_days <= 0:
        return REFERENCE_SILENCE
    if gap_days > SILENCE_RATIO * baseline_days:
        return 'longer'
    if gap_days < baseline_days / SILENCE_RATIO:
        return 'shorter'
    return REFERENCE_SILENCE


# --------------------------------------------------------------------------- #
# One user's history inside the 84-day horizon
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class History:
    """Eligible writings of one user, ascending, measured from the user's origin.

    `delta` is days since the user's first writing, which is taken before
    disclosure writings are removed. All arrays share one order.
    """
    user_id: str
    delta: np.ndarray
    scores: np.ndarray        # (n, 5) cached P(Yes), column order = SIGNALS
    polarity: np.ndarray
    distress: np.ndarray
    absolutist: np.ndarray
    night: np.ndarray
    reply: np.ndarray

    @property
    def size(self) -> int:
        return int(self.delta.size)


def empty_history(user_id: str) -> History:
    zero = np.zeros(0, dtype=float)
    return History(user_id, zero, np.zeros((0, len(SIGNALS)), dtype=float),
                   zero, zero, zero, zero, zero)


def chunk_count(history: History, spec_id: str) -> int:
    """Number of chunk indices this spec offers for this user.

    Day chunks are fixed by the horizon, so a quiet user still has all of them
    and going quiet stays visible. Post-count chunks exist only where the user
    has the writings to fill them, capped by POST_CHUNK_CAP.
    """
    kind, width = CHUNK_SPECS[spec_id]
    if kind == 'days':
        return HORIZON_DAYS // width
    return int(min(POST_CHUNK_CAP, max(1, history.size // width)))


def chunk_assignment(history: History, spec_id: str, n_chunks: int) -> np.ndarray:
    """0-based chunk of every writing, or -1 when it falls outside the chunks."""
    kind, width = CHUNK_SPECS[spec_id]
    if kind == 'days':
        index = np.floor(history.delta / width).astype(int)
        # The instant at the horizon stays in the last chunk.
        index = np.where(history.delta >= HORIZON_DAYS, n_chunks - 1, index)
        return np.where((index >= 0) & (index < n_chunks), index, -1)
    index = np.arange(history.size) // width
    return np.where(index < n_chunks, index, -1)


def chunk_window(history: History, spec_id: str, chunk: int, assignment: np.ndarray):
    """(start_day, end_day, reached) for a 1-based chunk index.

    For a day spec the window is fixed. For a post-count spec the end is the day
    of the chunk's last writing; a user who never filled the chunk keeps their
    last writing as the end, and `reached` is False.
    """
    kind, width = CHUNK_SPECS[spec_id]
    if kind == 'days':
        start, end = float((chunk - 1) * width), float(chunk * width)
        return start, end, bool(history.delta.size and history.delta.max() >= start)
    reached = history.size >= chunk * width
    members = np.flatnonzero(assignment == chunk - 1)
    if members.size:
        start = float(history.delta[members[0]])
        end = float(history.delta[members[-1]])
    elif history.size:
        start = end = float(history.delta[-1])
    else:
        start = end = 0.0
    return start, end, reached


def prefix_mask(history: History, spec_id: str, chunk: int, end_day: float) -> np.ndarray:
    """Writings from the origin through the chunk end. Nothing later."""
    kind, width = CHUNK_SPECS[spec_id]
    if kind == 'days':
        return history.delta <= end_day
    limit = chunk * width
    return np.arange(history.size) < limit


def _silence(history: History, prefix: np.ndarray, in_chunk: np.ndarray,
             chunk_end_day: float, allow_open_gap: bool) -> tuple[str, int]:
    """Silence class and the longest gap ending in the chunk, in whole days."""
    index = np.flatnonzero(prefix)
    if index.size == 0:
        # Before the user's first writing.
        return REFERENCE_SILENCE, 0
    delta = history.delta[index]
    distress = history.distress[index]
    threshold = float(np.percentile(distress, DISTRESS_PERCENTILE))
    closed = [float(delta[j] - delta[j - 1]) for j in range(1, delta.size)]

    gaps = []  # (length, start position within the prefix, is_closed)
    for j in range(1, delta.size):
        if in_chunk[index[j]]:
            gaps.append((closed[j - 1], j - 1, True))
    chunk_has_writing = bool(in_chunk[index].any()) if index.size else False
    if allow_open_gap and not chunk_has_writing and chunk_end_day > delta[-1]:
        # An empty chunk after the first writing: the gap is still open at the
        # chunk end, so a user who goes quiet is visible.
        gaps.append((float(chunk_end_day - delta[-1]), delta.size - 1, False))

    if not gaps:
        return REFERENCE_SILENCE, 0
    max_silence = int(math.floor(max(length for length, _, _ in gaps)))

    candidates = [g for g in gaps if distress[g[1]] >= threshold]
    if not candidates:
        return REFERENCE_SILENCE, max_silence
    length, start, _ = max(candidates, key=lambda g: g[0])
    earlier = [value for value in closed[:start] if value > 0]
    baseline = float(np.median(earlier)) if earlier else None
    return silence_class(length, baseline), max_silence


def _chunk_block(history: History, members: np.ndarray) -> dict:
    """Label-free summaries of the writings inside one chunk."""
    n = int(members.size)
    block = {'n_posts': n}
    if n:
        scores = history.scores[members]
        polarity = history.polarity[members]
        for position, signal in enumerate(SIGNALS):
            column = scores[:, position]
            block['share__' + signal] = float((column > HIGH_SCORE).mean())
            block['max__' + signal] = float(column.max())
        flips = int(((polarity[1:] * polarity[:-1]) < 0).sum()) if n > 1 else 0
        block['mood_flips'] = flips
        block['mean_absolute_jump'] = float(np.abs(np.diff(polarity)).mean()) if n > 1 else 0.0
        block['share__absolutist'] = float((history.absolutist[members] > 0).mean())
        block['share__night'] = float((history.night[members] > 0).mean())
        block['share__reply'] = float((history.reply[members] > 0).mean())
    else:
        for signal in SIGNALS:
            block['share__' + signal] = 0.0
            block['max__' + signal] = 0.0
        block['mood_flips'] = 0
        block['mean_absolute_jump'] = 0.0
        for name in SHARE_FEATURES:
            block['share__' + name] = 0.0
    return block


def _v1_static(history: History, prefix: np.ndarray) -> dict:
    """v1's own summaries on the prefix up to the decision. Floats, as in v1."""
    index = np.flatnonzero(prefix)
    row = {'v1__n_posts': int(index.size), 'v1__has_posts': 1.0 if index.size else 0.0}
    if index.size == 0:
        for signal in SIGNALS:
            row['v1__' + signal + '__mean'] = 0.0
        row.update({'v1__polarity_flip': 0.0, 'v1__polarity_jump': 0.0,
                    'v1__distress_burst': 0.0, 'v1__absolutist': 0.0})
        return row
    scores = history.scores[index]
    for position, signal in enumerate(SIGNALS):
        row['v1__' + signal + '__mean'] = float(scores[:, position].mean())
    polarity = history.polarity[index]
    distress = history.distress[index]
    row['v1__polarity_flip'] = (float(((polarity[1:] * polarity[:-1]) < 0).mean())
                                if index.size > 1 else 0.0)
    row['v1__polarity_jump'] = (float(np.abs(np.diff(polarity)).mean() / 2)
                                if index.size > 1 else 0.0)
    row['v1__distress_burst'] = float(max(0.0, np.sort(distress)[-3:].mean() - np.median(distress)))
    row['v1__absolutist'] = float(history.absolutist[index].mean())
    return row


def user_chunk_rows(history: History, spec_id: str) -> list[dict]:
    """Raw (pre-one-hot) rows for every chunk index this spec offers.

    Row `k` uses writings up to the end of chunk `k` and nothing after it.
    """
    n_chunks = chunk_count(history, spec_id)
    assignment = chunk_assignment(history, spec_id, n_chunks)
    rows: list[dict] = []
    level_history: dict[str, list[int]] = {}

    def remember(name: str, value: int) -> tuple[str, str]:
        past = level_history.setdefault(name, [])
        previous = past[-1] if past else None
        two_back = past[-2] if len(past) > 1 else None
        past.append(value)
        return change_label(value, previous), trend_label(value, two_back)

    for chunk in range(1, n_chunks + 1):
        start_day, end_day, reached = chunk_window(history, spec_id, chunk, assignment)
        members = np.flatnonzero(assignment == chunk - 1)
        prefix = prefix_mask(history, spec_id, chunk, end_day)
        in_chunk = assignment == chunk - 1
        block = _chunk_block(history, members)
        row = {'user_id': history.user_id, 'spec': spec_id, 'chunk_index': chunk,
               'chunk_start_day': start_day, 'chunk_end_day': end_day,
               'reached': bool(reached), 'n_posts': block['n_posts']}

        for signal in SIGNALS:
            level = share_level(block['share__' + signal])
            change, trend = remember('llm__' + signal, level)
            row['llm__%s__level' % signal] = level
            row['llm__%s__change' % signal] = change
            row['llm__%s__trend' % signal] = trend
            # Secondary ordinal: carried for the unit tests, never encoded.
            row['max_ordinal__%s__level' % signal] = max_level(block['max__' + signal])

        frequency = frequency_level(block['n_posts'])
        row['freq__level'] = frequency
        row['freq__change'] = remember('freq', frequency)[0]

        mood = mood_level(block['mood_flips'])
        row['mood__level'] = mood
        row['mood__change'] = remember('mood', mood)[0]
        row['mood_flips'] = block['mood_flips']
        row['jump__level'] = jump_level(block['mean_absolute_jump'])

        for name in SHARE_FEATURES:
            level = share_level(block['share__' + name])
            row[name + '__level'] = level
            if name in ('night', 'reply'):
                row[name + '__change'] = remember(name, level)[0]

        allow_open_gap = CHUNK_SPECS[spec_id][0] == 'days'
        klass, max_silence = _silence(history, prefix, in_chunk, end_day, allow_open_gap)
        row['silence__class'] = klass
        row['max_silence_days'] = max_silence
        row.update(_v1_static(history, prefix))
        rows.append(row)
    return rows


def no_data_row(user_id: str, spec_id: str, chunk: int) -> dict:
    """The row of a user with no eligible writing, or no chunk at this index."""
    history = empty_history(user_id)
    rows = user_chunk_rows(history, spec_id)
    row = dict(rows[0])
    row.update({'user_id': user_id, 'chunk_index': chunk, 'reached': False,
                'chunk_start_day': float('nan'), 'chunk_end_day': float('nan')})
    return row


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def _one_hot_columns() -> list[str]:
    columns: list[str] = ['n_posts', 'max_silence_days']
    for signal in SIGNALS:
        columns += ['llm__%s__level_%d' % (signal, level) for level in LEVELS
                    if level != REFERENCE_LEVEL]
        columns += ['llm__%s__change_%s' % (signal, value) for value in CHANGES
                    if value != REFERENCE_CHANGE]
        columns += ['llm__%s__trend_%s' % (signal, value) for value in TRENDS
                    if value != REFERENCE_TREND]
    for name in ('freq', 'mood'):
        columns += ['%s__level_%d' % (name, level) for level in LEVELS if level != REFERENCE_LEVEL]
        columns += ['%s__change_%s' % (name, value) for value in CHANGES if value != REFERENCE_CHANGE]
    columns += ['jump__level_%d' % level for level in LEVELS if level != REFERENCE_LEVEL]
    for name in SHARE_FEATURES:
        columns += ['%s__level_%d' % (name, level) for level in LEVELS if level != REFERENCE_LEVEL]
        if name in ('night', 'reply'):
            columns += ['%s__change_%s' % (name, value) for value in CHANGES
                        if value != REFERENCE_CHANGE]
    columns += ['silence__class_%s' % value for value in SILENCE_CLASSES
                if value != REFERENCE_SILENCE]
    return columns


V1_STATIC_COLUMNS = (['v1__%s__mean' % signal for signal in SIGNALS]
                     + ['v1__polarity_flip', 'v1__polarity_jump', 'v1__distress_burst',
                        'v1__absolutist', 'v1__n_posts', 'v1__has_posts'])
ENCODED_COLUMNS = _one_hot_columns()
ALL_COLUMNS = ENCODED_COLUMNS + V1_STATIC_COLUMNS


def encode(raw: pd.DataFrame) -> pd.DataFrame:
    """One-hot the ordinal and categorical columns, dropping the reference level.

    `n_posts` and `max_silence_days` stay integers. Columns are fixed, so every
    fold and every spec sees the same matrix shape.
    """
    out = pd.DataFrame(index=raw.index)
    out['n_posts'] = raw['n_posts'].astype(float)
    out['max_silence_days'] = raw['max_silence_days'].astype(float)
    for signal in SIGNALS:
        for level in LEVELS:
            if level != REFERENCE_LEVEL:
                out['llm__%s__level_%d' % (signal, level)] = (
                    raw['llm__%s__level' % signal] == level).astype(float)
        for value in CHANGES:
            if value != REFERENCE_CHANGE:
                out['llm__%s__change_%s' % (signal, value)] = (
                    raw['llm__%s__change' % signal] == value).astype(float)
        for value in TRENDS:
            if value != REFERENCE_TREND:
                out['llm__%s__trend_%s' % (signal, value)] = (
                    raw['llm__%s__trend' % signal] == value).astype(float)
    for name in ('freq', 'mood'):
        for level in LEVELS:
            if level != REFERENCE_LEVEL:
                out['%s__level_%d' % (name, level)] = (raw[name + '__level'] == level).astype(float)
        for value in CHANGES:
            if value != REFERENCE_CHANGE:
                out['%s__change_%s' % (name, value)] = (raw[name + '__change'] == value).astype(float)
    for level in LEVELS:
        if level != REFERENCE_LEVEL:
            out['jump__level_%d' % level] = (raw['jump__level'] == level).astype(float)
    for name in SHARE_FEATURES:
        for level in LEVELS:
            if level != REFERENCE_LEVEL:
                out['%s__level_%d' % (name, level)] = (raw[name + '__level'] == level).astype(float)
        if name in ('night', 'reply'):
            for value in CHANGES:
                if value != REFERENCE_CHANGE:
                    out['%s__change_%s' % (name, value)] = (
                        raw[name + '__change'] == value).astype(float)
    for value in SILENCE_CLASSES:
        if value != REFERENCE_SILENCE:
            out['silence__class_%s' % value] = (raw['silence__class'] == value).astype(float)
    for column in V1_STATIC_COLUMNS:
        out[column] = raw[column].astype(float)
    return out[ALL_COLUMNS]


# --------------------------------------------------------------------------- #
# Feature sets and sensitivity groups
# --------------------------------------------------------------------------- #
def _starts(prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in ENCODED_COLUMNS if c.startswith(prefixes)]


FEATURE_SETS = {
    'v1_static': list(V1_STATIC_COLUMNS),
    'v2_llm': ['n_posts'] + _starts(('llm__',)),
    'v2_code': (['n_posts', 'max_silence_days']
                + _starts(('freq__', 'silence__', 'mood__', 'jump__', 'absolutist__',
                           'night__', 'reply__'))),
}
FEATURE_SETS['v2_all'] = (FEATURE_SETS['v2_llm']
                          + [c for c in FEATURE_SETS['v2_code'] if c not in FEATURE_SETS['v2_llm']])
FEATURE_SETS['frequency'] = ['n_posts'] + _starts(('freq__',))
FEATURE_SETS['count'] = ['n_posts']

DROP_ONE_GROUPS = {
    'llm_levels': [c for c in ENCODED_COLUMNS if c.startswith('llm__') and '__level_' in c],
    'llm_changes': [c for c in ENCODED_COLUMNS
                    if c.startswith('llm__') and ('__change_' in c or '__trend_' in c)],
    'frequency': _starts(('freq__',)),
    'silence': _starts(('silence__',)) + ['max_silence_days'],
    'mood': _starts(('mood__',)),
    'polarity': _starts(('jump__',)),
    'absolutist': _starts(('absolutist__',)),
    'night': _starts(('night__',)),
    'reply': _starts(('reply__',)),
}


def feature_matrix(raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return encode(raw)[FEATURE_SETS[feature_set]]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _flag_functions():
    """The frozen v1 word rules. Imported, never re-implemented here."""
    from final_indirect_pipeline import flags
    return flags


def load_writings(root: Path, text_csv: Path | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Join the cached scores, the eligible writings, and the clock columns.

    Returns (writings, origins, users). `writings` has one row per eligible
    writing with the five cached P(Yes) values, the v1 word-rule numbers, the
    UTC night flag and the reply proxy. Writing text never leaves this function.
    """
    root = Path(root)
    private = root / 'private' / 'cohort_v1'
    posts = pd.read_parquet(private / 'eligible_posts.parquet')
    scores = pd.read_parquet(private / 'scores' / 'part-00000000.parquet')
    labels = pd.read_parquet(private / 'labels.parquet')
    origins = pd.read_parquet(private / 'origins.parquet')
    pilot = pd.read_parquet(private / 'pilot_users.parquet')

    for frame in (posts, scores, labels, origins, pilot):
        frame['user_id'] = frame.user_id.astype(str)
    if scores.duplicated(['user_id', 'post_id']).any():
        raise ValueError('score cache is not one row per writing')
    if not scores[list(SIGNALS)].notna().all().all():
        raise ValueError('a scored writing is missing one of the five signals')
    if not (scores[list(SIGNALS)].ge(0).all().all() and scores[list(SIGNALS)].le(1).all().all()):
        raise ValueError('a cached score is outside [0, 1]')

    data = posts.merge(scores.drop(columns='date'), on=['user_id', 'post_id'],
                       how='inner', validate='one_to_one')
    if len(data) != len(scores):
        raise ValueError('eligible writings and the score cache do not line up')

    clock = _clock_columns(root, text_csv)
    data = data.merge(clock, on='post_id', how='left', validate='one_to_one')
    if data[['hour', 'kind']].isna().any().any():
        raise ValueError('a writing has no hour or kind after the join')

    data['date'] = pd.to_datetime(data.date, utc=True, errors='raise')
    origins = origins.set_index('user_id').origin
    origins = pd.to_datetime(origins, utc=True, errors='raise')
    data['delta_days'] = ((data.date - data.user_id.map(origins)).dt.total_seconds()
                          / 86400.0)
    if (data.delta_days < -1e-9).any() or (data.delta_days > HORIZON_DAYS + 1e-9).any():
        raise ValueError('a writing sits outside the 84-day horizon')

    flags = _flag_functions()
    parsed = data.text.map(flags)
    data['polarity'] = parsed.map(lambda item: item['polarity']).astype(float)
    data['distress'] = parsed.map(lambda item: item['distress']).astype(float)
    data['absolutist'] = parsed.map(lambda item: item['absolutist']).astype(float)
    data['night'] = data.hour.astype(int).between(0, 5).astype(float)   # UTC clock, not local
    data['reply'] = data.kind.eq('comment').astype(float)               # no parent id exists

    users = labels.merge(pilot.assign(pilot_exposed=True), on='user_id', how='left')
    users['pilot_exposed'] = users.pilot_exposed.fillna(False).astype(bool)
    users = users.sort_values('user_id').reset_index(drop=True)

    keep = ['user_id', 'post_id', 'delta_days', 'hour', 'kind', 'night', 'reply',
            'polarity', 'distress', 'absolutist'] + list(SIGNALS)
    data = data.sort_values(['user_id', 'date', 'post_id'], kind='mergesort')
    return data[keep].reset_index(drop=True), origins, users


def _clock_columns(root: Path, text_csv: Path | None = None) -> pd.DataFrame:
    """Recover `hour` and `kind` for each eligible writing.

    The score cache stores a hashed post id. This rebuilds the same id from the
    source CSV with the same rule the export used, so the join is exact.
    """
    from llm_signal_scorer import canonical_hash
    from prepare_llm_pilot import empty_post
    from final_indirect_pipeline import disclosure

    path = Path(text_csv) if text_csv is not None else (
        Path(root) / 'data' / 'processed' / 'erisk' / 'task2_target_text.csv')
    writings = pd.read_csv(path, usecols=['user_id', 'date', 'hour', 'kind', 'title', 'body'],
                           dtype={'user_id': str})
    writings['text'] = (writings.title.fillna('') + ' ' + writings.body.fillna('')).str.strip()
    writings['date'] = pd.to_datetime(writings['date'], utc=True, errors='raise')
    within = writings.date <= writings.groupby('user_id').date.transform('min') + pd.Timedelta(days=HORIZON_DAYS)
    eligible = writings.loc[within & ~writings.text.map(disclosure)
                            & ~writings.text.map(empty_post)].copy()
    eligible['post_id'] = [canonical_hash([uid, date.isoformat(), int(i), text])
                           for uid, date, i, text in zip(eligible.user_id, eligible.date,
                                                         eligible.index, eligible.text)]
    out = eligible[['post_id', 'hour', 'kind']].copy()
    if out.post_id.duplicated().any():
        raise ValueError('rebuilt post ids are not unique')
    return out


def histories(writings: pd.DataFrame, users: pd.DataFrame) -> dict[str, History]:
    """One History per user, including users with no eligible writing."""
    built: dict[str, History] = {}
    for user_id, group in writings.groupby('user_id', sort=True):
        group = group.sort_values(['delta_days', 'post_id'], kind='mergesort')
        built[str(user_id)] = History(
            user_id=str(user_id),
            delta=group.delta_days.to_numpy(float),
            scores=group[list(SIGNALS)].to_numpy(float),
            polarity=group.polarity.to_numpy(float),
            distress=group.distress.to_numpy(float),
            absolutist=group.absolutist.to_numpy(float),
            night=group.night.to_numpy(float),
            reply=group.reply.to_numpy(float))
    for user_id in users.user_id.astype(str):
        built.setdefault(user_id, empty_history(user_id))
    return built


def raw_bank(built: dict[str, History], spec_id: str) -> pd.DataFrame:
    """Raw rows for every user and every chunk index of one spec."""
    rows: list[dict] = []
    for user_id in sorted(built):
        rows.extend(user_chunk_rows(built[user_id], spec_id))
    frame = pd.DataFrame(rows)
    return frame.set_index(['user_id', 'chunk_index']).sort_index()


def rows_at(bank: pd.DataFrame, chunk: int, user_ids) -> pd.DataFrame:
    """The raw row of every user at one chunk index, in the given user order."""
    user_ids = [str(u) for u in user_ids]
    spec_id = str(bank['spec'].iloc[0])
    present = bank.xs(chunk, level='chunk_index') if chunk in bank.index.get_level_values(1) \
        else bank.iloc[0:0].droplevel('chunk_index')
    missing = [u for u in user_ids if u not in present.index]
    if missing:
        filler = pd.DataFrame([no_data_row(u, spec_id, chunk) for u in missing]).set_index('user_id')
        present = pd.concat([present, filler.drop(columns=['chunk_index'], errors='ignore')])
    return present.reindex(user_ids)


def chunk_for_day(spec_id: str, day: int, bank: pd.DataFrame | None = None,
                  training_users: list[str] | None = None) -> int:
    """The chunk index a decision at `day` lands on.

    A day spec lands on the chunk that ends on that day. A post-count spec lands
    on the complete chunk whose median end day, over the training users who
    reached it, is closest to that day. That median is label-free, but it is
    still computed on training users only.
    """
    kind, width = CHUNK_SPECS[spec_id]
    if kind == 'days':
        if day % width:
            raise ValueError('decision day %d is not a chunk end of %s' % (day, spec_id))
        return day // width
    if bank is None or training_users is None:
        raise ValueError('a post-count spec needs the bank and the training users')
    subset = bank.loc[bank.index.get_level_values('user_id').isin([str(u) for u in training_users])]
    reached = subset[subset.reached]
    if reached.empty:
        return 1
    medians = reached.groupby(level='chunk_index').chunk_end_day.median()
    return int(medians.sub(day).abs().idxmin())


def post_chunk_medians(bank: pd.DataFrame, training_users: list[str]) -> pd.Series:
    subset = bank.loc[bank.index.get_level_values('user_id').isin([str(u) for u in training_users])]
    reached = subset[subset.reached]
    return reached.groupby(level='chunk_index').chunk_end_day.median()


def decision_grid(spec_id: str, bank: pd.DataFrame | None = None,
                  training_users: list[str] | None = None) -> list[int]:
    """Chunk indices a flag could be raised at, earliest first."""
    kind, width = CHUNK_SPECS[spec_id]
    if kind == 'days':
        return list(range(1, HORIZON_DAYS // width + 1))
    medians = post_chunk_medians(bank, training_users)
    return [int(k) for k in medians.index]


def feature_table_rows() -> list[dict]:
    """The feature table the professor asked for: name, type, rule, window, source."""
    rows = []
    for signal in SIGNALS:
        rows.append({'feature': 'llm__%s__level' % signal, 'type': 'ordinal (0-3, one-hot, level 0 dropped)',
                     'computed': 'share of writings in the chunk with cached P(Yes) > 0.5, cut at 0, 0.10, 0.30',
                     'window': 'chunk', 'source': 'LLM cache'})
        rows.append({'feature': 'llm__%s__change' % signal, 'type': 'categorical (lower/same/higher, same dropped)',
                     'computed': 'level against the previous chunk; first chunk is same',
                     'window': 'chunk vs previous chunk', 'source': 'LLM cache'})
        rows.append({'feature': 'llm__%s__trend' % signal, 'type': 'categorical (falling/flat/rising, flat dropped)',
                     'computed': 'level against the level two chunks earlier; flat with fewer than three chunks',
                     'window': 'three chunks', 'source': 'LLM cache'})
    rows += [
        {'feature': 'n_posts', 'type': 'integer', 'computed': 'eligible writings in the chunk',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'freq__level', 'type': 'ordinal (0-3, one-hot)',
         'computed': '0, 1-2, 3-7, 8 or more writings in the chunk', 'window': 'chunk', 'source': 'rule'},
        {'feature': 'freq__change', 'type': 'categorical (lower/same/higher)',
         'computed': 'frequency ordinal against the previous chunk, not the raw count',
         'window': 'chunk vs previous chunk', 'source': 'rule'},
        {'feature': 'silence__class', 'type': 'categorical (shorter/similar/longer, similar dropped)',
         'computed': 'longest gap ending in the chunk that starts on a writing at or above the user\'s '
                     '75th distress percentile, against 1.5 times the median earlier closed gap',
         'window': 'chunk, baseline from the origin', 'source': 'rule'},
        {'feature': 'max_silence_days', 'type': 'integer',
         'computed': 'longest gap ending in the chunk, floored to whole days',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'mood__level', 'type': 'ordinal (0-3, one-hot)',
         'computed': 'polarity sign flips in the chunk, binned 0, 1, 2-3, 4 or more',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'mood__change', 'type': 'categorical (lower/same/higher)',
         'computed': 'mood ordinal against the previous chunk', 'window': 'chunk vs previous chunk',
         'source': 'rule'},
        {'feature': 'jump__level', 'type': 'ordinal (0-3, one-hot)',
         'computed': 'mean absolute polarity change inside the chunk, cut at 0, 0.25, 0.50',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'absolutist__level', 'type': 'ordinal (0-3, one-hot)',
         'computed': 'share of writings with an absolutist word and a negative word, cut at 0, 0.10, 0.30',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'night__level', 'type': 'ordinal (0-3, one-hot)',
         'computed': 'share of writings with UTC hour 0-5, cut at 0, 0.10, 0.30. UTC, not local night',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'night__change', 'type': 'categorical (lower/same/higher)',
         'computed': 'night ordinal against the previous chunk', 'window': 'chunk vs previous chunk',
         'source': 'rule'},
        {'feature': 'reply__level', 'type': 'ordinal (0-3, one-hot)',
         'computed': 'share of writings with kind == comment, cut at 0, 0.10, 0.30. '
                     'There is no parent id, so a comment is only a reply proxy',
         'window': 'chunk', 'source': 'rule'},
        {'feature': 'reply__change', 'type': 'categorical (lower/same/higher)',
         'computed': 'reply ordinal against the previous chunk', 'window': 'chunk vs previous chunk',
         'source': 'rule'},
    ]
    for signal in SIGNALS:
        rows.append({'feature': 'v1__%s__mean' % signal, 'type': 'float',
                     'computed': 'mean cached P(Yes) over the prefix, as in v1',
                     'window': 'prefix to the decision', 'source': 'LLM cache'})
    rows += [
        {'feature': 'v1__polarity_flip', 'type': 'float',
         'computed': 'share of successive writings whose polarity changes sign',
         'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'v1__polarity_jump', 'type': 'float',
         'computed': 'mean absolute polarity change divided by 2, as in v1',
         'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'v1__distress_burst', 'type': 'float',
         'computed': 'mean of the three highest distress values minus the prefix median, floored at 0',
         'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'v1__absolutist', 'type': 'float', 'computed': 'mean absolutist flag over the prefix',
         'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'v1__n_posts', 'type': 'integer', 'computed': 'eligible writings in the prefix',
         'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'v1__has_posts', 'type': 'integer', 'computed': '1 when the prefix has a writing',
         'window': 'prefix to the decision', 'source': 'rule'},
        {'feature': 'max_ordinal__<signal>__level', 'type': 'ordinal (0-3), NOT FIT',
         'computed': 'chunk maximum P(Yes) cut at 0.25, 0.50, 0.75. Unit-tested only, never encoded '
                     'and never fit to labels',
         'window': 'chunk', 'source': 'LLM cache'},
    ]
    return rows


def prespec(root: Path) -> dict:
    return json.loads((Path(root) / 'diagnostics_v2' / 'prespec_v2.json').read_text(encoding='utf-8'))
