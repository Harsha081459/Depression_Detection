"""Unit tests for v4 features. Run before the v4 cross-validation.

    python -m unittest discover -s scripts/analysis -p 'test_v4_features.py' -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import v4_features as v4


def stamps(deltas, user='u', comments=None, hours=None):
    n = len(deltas)
    ts = (np.asarray(deltas, dtype=float) * 86400).astype(np.int64) + 1_700_000_000
    hours = hours if hours is not None else [12] * n
    return pd.DataFrame({'user_id': user, 'ts_sec': ts, 'hour': hours,
                         'night': [int(h < 6) for h in hours], 'weekend': 0,
                         'is_comment': comments if comments is not None else [1] * n,
                         'delta_days': np.asarray(deltas, dtype=float)})


def emotion_frame(deltas, user='u', sad=1, tokens=10):
    n = len(deltas)
    frame = pd.DataFrame({'user_id': user, 'delta_days': deltas, 'n_tokens': [tokens] * n})
    for e in v4.EMOTIONS:
        for stem in ('emo', 'emoclin'):
            frame['%s_%s_cnt' % (stem, e)] = sad if e == 'sadness' else 0
    return frame


class TestSets(unittest.TestCase):
    def test_sizes(self):
        sizes = {k: len(v) for k, v in v4.FEATURE_SETS.items()}
        self.assertEqual(sizes, {'rhythm': 11, 'clocks': 12, 'emotion': 11, 'v1_static': 11,
                                 'v4_text': 22, 'v4_behaviour': 33, 'v4_all': 45,
                                 'emotion_clinical_sens': 11})
        for cols in v4.FEATURE_SETS.values():
            self.assertEqual(len(cols), len(set(cols)))

    def test_headline_set_has_no_comment_share(self):
        self.assertNotIn('comment_frac', v4.FEATURE_SETS['v4_behaviour'])
        self.assertNotIn('comment_frac', v4.FEATURE_SETS['rhythm'])

    def test_drop_one_groups_inside_v4_all(self):
        for cols in v4.DROP_ONE_GROUPS.values():
            self.assertTrue(set(cols) <= set(v4.FEATURE_SETS['v4_all']))


class TestRhythm(unittest.TestCase):
    def test_rate_and_night(self):
        frame = v4.rhythm_rows(stamps([0, 1, 2, 3, 6.0], hours=[1, 1, 12, 12, 12]), 28)
        row = frame.loc['u']
        self.assertAlmostEqual(row.night_frac, 0.4)
        self.assertAlmostEqual(row.log_posts_per_week, np.log1p(5 / 6 * 7))

    def test_rhythm_ignores_later_writings(self):
        base = stamps([0, 1, 2, 10, 20])
        later = stamps([0, 1, 2, 10, 20, 30, 31, 32, 80], comments=[1] * 5 + [0] * 4)
        a = v4.rhythm_rows(base, 28).loc['u']
        b = v4.rhythm_rows(later, 28).loc['u']
        pd.testing.assert_series_equal(a, b)


class TestEmotion(unittest.TestCase):
    def test_token_weighted_share(self):
        row = v4.emotion_rows(emotion_frame([0, 1], sad=2, tokens=10), 28).loc['u']
        self.assertAlmostEqual(row['emo__sadness'], 0.2)
        self.assertAlmostEqual(row['mean_log_tokens'], np.log1p(10))

    def test_emotion_ignores_later_writings(self):
        base = emotion_frame([0, 5, 20])
        later = pd.concat([base, emotion_frame([40, 70], sad=9, tokens=12)], ignore_index=True)
        a = v4.emotion_rows(base, 28).loc['u']
        b = v4.emotion_rows(later, 28).loc['u']
        pd.testing.assert_series_equal(a, b)


if __name__ == '__main__':
    unittest.main()
