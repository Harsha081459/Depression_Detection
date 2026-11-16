"""Graded closed-gap silence; no text input or observation-length feature.

Call with the window's full timestamp sequence. In the masked arm retain
disclosure timestamps with e=0 and polarity=0. In the before-disclosure arm
pass only timestamps strictly before the first disclosure and end=None.
Open silence is returned separately and must never enter a classifier.
"""
import math
import statistics

CLOSED_COLUMNS = ('silence_distress_sudden', 'silence_distress_fade', 'silence_after_ordinary')
OPEN_COLUMN = 'open_silence_distress'


def graded_silence(rows, end=None):
    values = dict.fromkeys(CLOSED_COLUMNS + (OPEN_COLUMN,), 0.0)
    if any(rows[i]['ts'] > rows[i + 1]['ts'] for i in range(len(rows) - 1)):
        raise ValueError('timestamps_must_be_sorted')
    if end is not None:
        rows = [row for row in rows if row['ts'] <= end]
    gaps = [(rows[i + 1]['ts'] - rows[i]['ts']).total_seconds() for i in range(len(rows) - 1)]
    for i in range(len(rows)):
        # g_i is never included in its own baseline.
        past = [g for g in gaps[:i] if g > 0]
        if len(past) < 4:
            continue
        baseline = statistics.median(past)
        closed = i < len(gaps)
        if not closed and end is None:
            continue
        gap = gaps[i] if closed else (end - rows[i]['ts']).total_seconds()
        if gap < 86400 or gap < 3 * baseline:
            continue
        strength = min(1.0, math.log2(gap / baseline) / math.log2(30))
        context = rows[max(0, i - 2):i + 1]
        distress = statistics.mean(0.0 if row.get('e') is None else row['e'] for row in context)
        polarity = statistics.mean(row.get('polarity', 0.0) for row in context)
        if not 0 <= distress <= 1:
            raise ValueError('distress_outside_unit_interval')
        if not closed:
            values[OPEN_COLUMN] = strength * distress
            continue
        sudden = i < 3 or all(g <= 1.5 * baseline for g in gaps[i - 3:i])
        name = 'silence_distress_sudden' if sudden else 'silence_distress_fade'
        values[name] = max(values[name], strength * distress)
        if polarity >= 0:
            values['silence_after_ordinary'] = max(values['silence_after_ordinary'], strength * (1 - distress))
    return values
