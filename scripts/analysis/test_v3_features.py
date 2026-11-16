"""Unit tests for v3 features. Run before any v3 cross-validation.

    python -m unittest discover -s scripts/analysis -p 'test_v3_features.py' -v

No real writing, user id or label appears here.
"""
from __future__ import annotations

import unittest

import numpy as np

import v2_temporal_features as v2
import v3_features as v3
from test_v2_temporal_features import make_history


def stats_for(histories):
    return v3.Statistics(histories, sorted(histories))


class TestRules(unittest.TestCase):
    def test_levels_follow_the_three_cuts(self):
        cuts = np.tile([0.2, 0.4, 0.6], (5, 1))
        means = np.array([[[0.1] * 5, [0.2] * 5, [0.3] * 5, [0.5] * 5, [0.7] * 5, [np.nan] * 5]])
        got = v3.levels(means, cuts)[0, :, 0].tolist()
        self.assertEqual(got, [0, 0, 1, 2, 3, -1])

    def test_cuts_ignore_empty_chunks(self):
        means = np.full((2, 6, 5), np.nan)
        means[0, 0, :] = 0.1
        means[1, 0, :] = 0.3
        cuts = v3.fit_cuts(means)
        self.assertAlmostEqual(cuts[0, 0], 0.2)

    def test_change_uses_the_last_non_empty_chunk(self):
        m = np.full((1, 6, 5), -1)
        m[0, 0, :] = 2       # chunk 1 level 2
        # chunk 2 and 3 empty
        m[0, 3, :] = 1       # chunk 4 level 1
        self.assertEqual(v3.change_codes(m, 4)[0, 0], -1)   # against chunk 1, not empty chunk 3
        self.assertEqual(v3.change_codes(m, 3)[0, 0], 0)    # empty chunk -> same
        self.assertEqual(v3.change_codes(m, 1)[0, 0], 0)    # nothing earlier -> same

    def test_feature_sets_match_the_prespec_sizes(self):
        sizes = {k: len(v) for k, v in v3.FEATURE_SETS.items()}
        self.assertEqual(sizes, {'v1_static': 11, 'v1_plus_change': 22, 'v1_plus_change_int': 17,
                                 'v3_levels': 17, 'v3_all': 37})
        for columns in v3.FEATURE_SETS.values():
            self.assertEqual(len(columns), len(set(columns)))

    def test_drop_one_groups_sit_inside_v3_all(self):
        for group in v3.DROP_ONE_GROUPS.values():
            self.assertTrue(set(group) <= set(v3.FEATURE_SETS['v3_all']))


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.histories = {
            'a': make_history([0.0, 1.0, 15.0, 16.0, 50.0], score=[[0.9] * 5, [0.8] * 5, [0.1] * 5,
                                                                   [0.1] * 5, [0.9] * 5], user_id='a'),
            'b': make_history([0.0, 30.0], score=0.2, user_id='b'),
            'c': v2.empty_history('c'),
        }
        self.stats = stats_for(self.histories)
        self.cuts = np.tile([0.15, 0.5, 0.85], (5, 1))

    def test_columns_and_values(self):
        frame = v3.encode(self.stats, ['a', 'b', 'c'], 2, self.cuts)
        self.assertEqual(frame.loc['a', 'lvl__self_blame__3'], 0.0)
        self.assertEqual(frame.loc['a', 'chg__self_blame__lower'], 1.0)  # chunk 1 level 3 -> chunk 2 level 0
        self.assertEqual(frame.loc['a', 'chgint__self_blame'], -1.0)
        self.assertEqual(frame.loc['b', 'empty_chunk'], 1.0)            # b writes on day 30, chunk 3
        self.assertEqual(frame.loc['c', 'empty_chunk'], 1.0)
        self.assertEqual(float(frame.loc['c', v3.LEVEL_COLUMNS + v3.CHANGE_COLUMNS].sum()), 0.0)
        self.assertTrue(np.isfinite(frame.to_numpy(float)).all())

    def test_v1_prefix_columns_come_from_the_same_chunk(self):
        frame = v3.encode(self.stats, ['a'], 2, self.cuts)
        self.assertEqual(frame.loc['a', 'v1__n_posts'], 4.0)    # writings on days 0, 1, 15, 16
        self.assertEqual(frame.loc['a', 'n_posts'], 2.0)

    def test_chunk_row_ignores_later_writings(self):
        """Leakage test: with fixed cuts, chunk k must not see a later writing."""
        base = self.histories['a']
        extended = make_history(list(base.delta) + [60.0, 70.0, 83.0],
                                score=np.vstack([base.scores, [[0.01] * 5] * 3]), user_id='a')
        first = v3.encode(stats_for({'a': base}), ['a'], 2, self.cuts)
        second = v3.encode(stats_for({'a': extended}), ['a'], 2, self.cuts)
        self.assertTrue(first.equals(second))

    def test_cuts_come_from_training_users_only(self):
        _, _, cuts = v3.fold_matrix(self.stats, ['a', 'c'], ['b'], 2, 'v3_all')
        alone = v3.fit_cuts(self.stats.means[self.stats.rows(['a', 'c'])])
        np.testing.assert_allclose(cuts, alone)


if __name__ == '__main__':
    unittest.main()
