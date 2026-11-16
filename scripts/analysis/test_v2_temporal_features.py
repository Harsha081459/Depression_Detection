"""Unit tests for the v2 temporal features. Run these before any cross-validation.

    python -m unittest discover -s scripts/analysis -p 'test_v2_temporal_features.py' -v

No real writing, no user id, and no label appears in this file. The leakage test
is `test_chunk_row_ignores_later_writings`: it fails if a feature for chunk k
uses a writing that falls after chunk k.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import v2_temporal_features as v2


def make_history(deltas, score=0.1, polarity=None, distress=None,
                 absolutist=None, night=None, reply=None, user_id='u'):
    """A synthetic user. `score` is the cached P(Yes) used for all five signals."""
    n = len(deltas)
    scores = np.full((n, len(v2.SIGNALS)), float(score)) if np.isscalar(score) \
        else np.asarray(score, dtype=float).reshape(n, len(v2.SIGNALS))
    def fill(values, default=0.0):
        return np.asarray(values if values is not None else [default] * n, dtype=float)
    return v2.History(user_id=user_id, delta=np.asarray(deltas, dtype=float), scores=scores,
                      polarity=fill(polarity), distress=fill(distress),
                      absolutist=fill(absolutist), night=fill(night), reply=fill(reply))


def row_for(history, spec_id, chunk):
    rows = {row['chunk_index']: row for row in v2.user_chunk_rows(history, spec_id)}
    return rows[chunk]


class TestScalarRules(unittest.TestCase):
    def test_share_level_cuts(self):
        self.assertEqual(v2.share_level(0.0), 0)
        self.assertEqual(v2.share_level(0.01), 1)
        self.assertEqual(v2.share_level(0.10), 1)
        self.assertEqual(v2.share_level(0.1001), 2)
        self.assertEqual(v2.share_level(0.30), 2)
        self.assertEqual(v2.share_level(0.3001), 3)
        self.assertEqual(v2.share_level(1.0), 3)

    def test_max_level_is_the_secondary_rule(self):
        self.assertEqual(v2.max_level(0.0), 0)
        self.assertEqual(v2.max_level(0.25), 0)
        self.assertEqual(v2.max_level(0.26), 1)
        self.assertEqual(v2.max_level(0.50), 1)
        self.assertEqual(v2.max_level(0.51), 2)
        self.assertEqual(v2.max_level(0.75), 2)
        self.assertEqual(v2.max_level(0.76), 3)

    def test_max_ordinal_is_never_encoded(self):
        """The max-P(Yes) binning is unit-tested only. It is not fit to labels."""
        self.assertFalse([c for c in v2.ALL_COLUMNS if 'max_ordinal' in c])
        for columns in v2.FEATURE_SETS.values():
            self.assertFalse([c for c in columns if 'max_ordinal' in c])

    def test_frequency_bins(self):
        self.assertEqual([v2.frequency_level(n) for n in (0, 1, 2, 3, 7, 8, 40)],
                         [0, 1, 1, 2, 2, 3, 3])

    def test_mood_bins(self):
        self.assertEqual([v2.mood_level(k) for k in (0, 1, 2, 3, 4, 9)], [0, 1, 2, 2, 3, 3])

    def test_jump_bins(self):
        self.assertEqual([v2.jump_level(x) for x in (0.0, 0.1, 0.25, 0.26, 0.5, 0.51, 2.0)],
                         [0, 1, 1, 2, 2, 3, 3])

    def test_change_and_trend_defaults(self):
        self.assertEqual(v2.change_label(2, None), 'same')
        self.assertEqual(v2.change_label(1, 2), 'lower')
        self.assertEqual(v2.change_label(3, 2), 'higher')
        self.assertEqual(v2.trend_label(2, None), 'flat')
        self.assertEqual(v2.trend_label(0, 2), 'falling')
        self.assertEqual(v2.trend_label(3, 1), 'rising')

    def test_silence_class_ratio(self):
        self.assertEqual(v2.silence_class(10.0, None), 'similar')
        self.assertEqual(v2.silence_class(10.0, 0.0), 'similar')
        self.assertEqual(v2.silence_class(10.0, 2.0), 'longer')
        self.assertEqual(v2.silence_class(3.0, 2.0), 'similar')
        self.assertEqual(v2.silence_class(1.0, 2.0), 'shorter')


class TestChunking(unittest.TestCase):
    def test_day_chunks_cover_the_horizon(self):
        history = make_history([0.0, 1.0])
        self.assertEqual(v2.chunk_count(history, 'd7'), 12)
        self.assertEqual(v2.chunk_count(history, 'd14'), 6)
        self.assertEqual(v2.chunk_count(history, 'd28'), 3)

    def test_horizon_instant_stays_in_the_last_chunk(self):
        history = make_history([0.0, 84.0])
        assignment = v2.chunk_assignment(history, 'd7', 12)
        self.assertEqual(int(assignment[1]), 11)
        self.assertEqual(row_for(history, 'd7', 12)['n_posts'], 1)

    def test_day_chunk_boundary_is_half_open(self):
        history = make_history([0.0, 6.999, 7.0])
        self.assertEqual(row_for(history, 'd7', 1)['n_posts'], 2)
        self.assertEqual(row_for(history, 'd7', 2)['n_posts'], 1)

    def test_empty_day_chunks_remain(self):
        history = make_history([0.0, 1.0, 2.0])
        rows = v2.user_chunk_rows(history, 'd7')
        self.assertEqual(len(rows), 12)
        quiet = rows[4]
        self.assertEqual(quiet['n_posts'], 0)
        self.assertEqual(quiet['freq__level'], 0)
        self.assertEqual(quiet['llm__self_blame__level'], 0)
        # A user who goes quiet still shows a growing open gap.
        self.assertGreater(quiet['max_silence_days'], 0)
        self.assertGreater(rows[8]['max_silence_days'], quiet['max_silence_days'])

    def test_post_chunks_need_complete_writings(self):
        history = make_history(list(np.arange(25, dtype=float)))
        self.assertEqual(v2.chunk_count(history, 'p10'), 2)
        rows = {row['chunk_index']: row for row in v2.user_chunk_rows(history, 'p10')}
        self.assertEqual(rows[1]['n_posts'], 10)
        self.assertEqual(rows[2]['n_posts'], 10)
        self.assertTrue(rows[2]['reached'])
        self.assertEqual(rows[2]['v1__n_posts'], 20)

    def test_prefix_only_grows(self):
        history = make_history([0.0, 3.0, 9.0, 30.0, 80.0])
        counts = [row['v1__n_posts'] for row in v2.user_chunk_rows(history, 'd7')]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 5)


class TestLevels(unittest.TestCase):
    def test_share_level_from_cached_scores(self):
        # Four writings in chunk 1, one of them above 0.5 -> share 0.25 -> level 2.
        scores = np.array([[0.9] * 5, [0.1] * 5, [0.2] * 5, [0.3] * 5])
        history = make_history([0.0, 1.0, 2.0, 3.0], score=scores)
        row = row_for(history, 'd7', 1)
        self.assertEqual(row['llm__self_blame__level'], 2)
        self.assertEqual(row['max_ordinal__self_blame__level'], 3)

    def test_change_and_trend_across_chunks(self):
        # Chunk 1 all high (level 3), chunk 2 none high (level 0), chunk 3 all high.
        deltas = [0.0, 1.0, 8.0, 9.0, 15.0, 16.0]
        scores = np.array([[0.9] * 5, [0.9] * 5, [0.1] * 5, [0.1] * 5, [0.9] * 5, [0.9] * 5])
        history = make_history(deltas, score=scores)
        rows = {row['chunk_index']: row for row in v2.user_chunk_rows(history, 'd7')}
        self.assertEqual(rows[1]['llm__isolation__change'], 'same')
        self.assertEqual(rows[1]['llm__isolation__trend'], 'flat')
        self.assertEqual(rows[2]['llm__isolation__change'], 'lower')
        self.assertEqual(rows[2]['llm__isolation__trend'], 'flat')
        self.assertEqual(rows[3]['llm__isolation__change'], 'higher')
        self.assertEqual(rows[3]['llm__isolation__trend'], 'flat')
        self.assertEqual(rows[4]['llm__isolation__change'], 'lower')
        # Chunk 4 is level 0 and chunk 2 was level 0, so the three-chunk trend is flat.
        self.assertEqual(rows[4]['llm__isolation__trend'], 'flat')

    def test_trend_falls_and_rises_over_three_chunks(self):
        # Chunk 1 all high (level 3), chunks 2 and 3 none high (level 0), chunk 4 high.
        deltas = [0.0, 1.0, 8.0, 9.0, 15.5, 16.0, 22.0, 23.0]
        scores = np.array([[0.9] * 5] * 2 + [[0.1] * 5] * 4 + [[0.9] * 5] * 2)
        history = make_history(deltas, score=scores)
        rows = {row['chunk_index']: row for row in v2.user_chunk_rows(history, 'd7')}
        self.assertEqual(rows[3]['llm__isolation__trend'], 'falling')   # 0 against chunk 1's 3
        self.assertEqual(rows[4]['llm__isolation__trend'], 'rising')    # 3 against chunk 2's 0
        self.assertEqual(rows[5]['llm__isolation__trend'], 'flat')      # 0 against chunk 3's 0
        self.assertEqual(rows[6]['llm__isolation__trend'], 'falling')   # 0 against chunk 4's 3

    def test_frequency_change_uses_the_ordinal(self):
        # Chunk 1 has 3 writings (level 2), chunk 2 has 2 (level 1).
        history = make_history([0.0, 1.0, 2.0, 7.0, 8.0])
        rows = {row['chunk_index']: row for row in v2.user_chunk_rows(history, 'd7')}
        self.assertEqual(rows[1]['freq__level'], 2)
        self.assertEqual(rows[2]['freq__level'], 1)
        self.assertEqual(rows[2]['freq__change'], 'lower')

    def test_mood_counts_sign_flips_inside_the_chunk(self):
        history = make_history([0.0, 1.0, 2.0, 3.0], polarity=[0.5, -0.5, 0.5, -0.5])
        row = row_for(history, 'd7', 1)
        self.assertEqual(row['mood_flips'], 3)
        self.assertEqual(row['mood__level'], 2)

    def test_night_and_reply_shares(self):
        history = make_history([0.0, 1.0, 2.0, 3.0], night=[1, 1, 0, 0], reply=[1, 1, 1, 0])
        row = row_for(history, 'd7', 1)
        self.assertEqual(row['night__level'], 3)   # share 0.50
        self.assertEqual(row['reply__level'], 3)   # share 0.75


class TestSilence(unittest.TestCase):
    def test_long_gap_after_high_distress_is_longer(self):
        deltas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 25.0]
        distress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]
        history = make_history(deltas, distress=distress)
        row = row_for(history, 'd28', 1)
        self.assertEqual(row['silence__class'], 'longer')
        self.assertEqual(row['max_silence_days'], 20)

    def test_gap_after_calm_writing_is_not_classified(self):
        deltas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 25.0]
        distress = [0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0]
        history = make_history(deltas, distress=distress)
        row = row_for(history, 'd28', 1)
        self.assertEqual(row['silence__class'], 'similar')
        # The longest gap still counts, whether or not it started on distress.
        self.assertEqual(row['max_silence_days'], 20)

    def test_short_gap_after_high_distress_is_shorter(self):
        # Writings every four days, then a distressed writing followed at once by
        # another writing. The only gap that starts at or above the user's 75th
        # distress percentile is that short one.
        deltas = [0.0, 4.0, 8.0, 12.0, 16.0, 16.1]
        distress = [0.0, 0.0, 0.0, 0.1, 0.9, 0.5]
        history = make_history(deltas, distress=distress)
        row = row_for(history, 'd28', 1)
        self.assertEqual(row['silence__class'], 'shorter')

    def test_class_ignores_a_long_gap_that_starts_calm(self):
        """The class only uses gaps that start on a high-distress writing."""
        deltas = [0.0, 2.0, 4.0, 6.0, 8.0, 40.0]
        distress = [0.0, 0.0, 0.9, 0.8, 0.0, 0.0]
        history = make_history(deltas, distress=distress)
        row = row_for(history, 'd28', 2)   # the 32-day gap ends in the second chunk
        self.assertNotEqual(row['silence__class'], 'longer')
        self.assertEqual(row['max_silence_days'], 32)

    def test_before_the_first_writing_is_similar_with_no_silence(self):
        history = make_history([30.0, 31.0])
        row = row_for(history, 'd7', 1)
        self.assertEqual(row['silence__class'], 'similar')
        self.assertEqual(row['max_silence_days'], 0)
        self.assertEqual(row['n_posts'], 0)

    def test_empty_chunk_opens_a_gap_from_the_last_writing(self):
        history = make_history([0.0, 1.0, 2.0, 3.0, 4.0], distress=[0, 0, 0, 0, 0.9])
        row = row_for(history, 'd7', 3)   # days 14-21, no writing
        self.assertEqual(row['n_posts'], 0)
        self.assertEqual(row['max_silence_days'], 17)
        self.assertEqual(row['silence__class'], 'longer')


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.history = make_history([0.0, 1.0, 8.0], score=0.9, polarity=[0.4, -0.4, 0.4],
                                    distress=[0.1, 0.2, 0.3], absolutist=[1, 0, 1],
                                    night=[1, 0, 1], reply=[1, 1, 0])
        self.raw = pd.DataFrame(v2.user_chunk_rows(self.history, 'd7')).set_index('chunk_index')

    def test_reference_levels_are_dropped(self):
        for column in v2.ENCODED_COLUMNS:
            self.assertFalse(column.endswith('__level_0'), column)
            self.assertFalse(column.endswith('__change_same'), column)
            self.assertFalse(column.endswith('__trend_flat'), column)
            self.assertFalse(column.endswith('class_similar'), column)

    def test_encoded_columns_are_fixed_and_numeric(self):
        encoded = v2.encode(self.raw)
        self.assertEqual(list(encoded.columns), v2.ALL_COLUMNS)
        self.assertTrue(np.isfinite(encoded.to_numpy(float)).all())
        for column in encoded.columns:
            if column in ('n_posts', 'max_silence_days') or column.startswith('v1__'):
                continue
            self.assertTrue(set(np.unique(encoded[column])) <= {0.0, 1.0}, column)

    def test_integers_stay_integers(self):
        encoded = v2.encode(self.raw)
        self.assertEqual(encoded.loc[1, 'n_posts'], 2.0)
        self.assertEqual(encoded.loc[2, 'n_posts'], 1.0)

    def test_a_user_with_no_writing_encodes_to_zeros(self):
        raw = pd.DataFrame([v2.no_data_row('u2', 'd7', 4)]).set_index('user_id')
        encoded = v2.encode(raw)
        self.assertEqual(float(encoded.to_numpy(float).sum()), 0.0)

    def test_feature_sets_are_disjoint_where_they_should_be(self):
        sets = v2.FEATURE_SETS
        self.assertEqual(sets['count'], ['n_posts'])
        self.assertEqual(set(sets['v2_all']), set(sets['v2_llm']) | set(sets['v2_code']))
        self.assertEqual(len(sets['v2_all']), len(set(sets['v2_all'])))
        self.assertTrue(set(sets['frequency']) <= set(sets['v2_code']))
        self.assertTrue(all(c.startswith('v1__') for c in sets['v1_static']))
        self.assertFalse(set(sets['v1_static']) & set(sets['v2_all']))
        for name, columns in sets.items():
            self.assertTrue(set(columns) <= set(v2.ALL_COLUMNS), name)

    def test_drop_one_groups_cover_v2_all_once(self):
        covered = [c for group in v2.DROP_ONE_GROUPS.values() for c in group]
        self.assertEqual(len(covered), len(set(covered)))
        missing = set(v2.FEATURE_SETS['v2_all']) - set(covered)
        self.assertEqual(missing, {'n_posts'})   # the control stays in every arm


class TestNoFutureWritings(unittest.TestCase):
    """The leakage test the handoff asks for."""

    def test_chunk_row_ignores_later_writings(self):
        base = [0.0, 1.0, 6.0, 9.0, 13.0, 20.0]
        history = make_history(base, score=0.9, polarity=[0.3, -0.3, 0.3, -0.3, 0.3, -0.3],
                               distress=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                               absolutist=[1, 0, 1, 0, 1, 0], night=[1, 0, 1, 0, 1, 0],
                               reply=[0, 1, 0, 1, 0, 1])
        later = base + [22.0, 23.0, 40.0, 83.0]
        extended = make_history(
            later, score=0.1,
            polarity=[0.3, -0.3, 0.3, -0.3, 0.3, -0.3, -0.9, 0.9, -0.9, 0.9],
            distress=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.99, 0.99, 0.99, 0.99],
            absolutist=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            night=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            reply=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1])
        # The first six writings keep their scores; the added ones differ in
        # every feature input. Chunks 1 to 3 end on days 7, 14 and 21.
        extended.scores[:6] = history.scores[:6]
        for spec_id in ('d7', 'd14', 'd28'):
            first = {row['chunk_index']: row for row in v2.user_chunk_rows(history, spec_id)}
            second = {row['chunk_index']: row for row in v2.user_chunk_rows(extended, spec_id)}
            last_safe = int(np.floor(21.0 / v2.CHUNK_SPECS[spec_id][1]))
            for chunk in range(1, last_safe + 1):
                self.assertEqual(first[chunk], second[chunk],
                                 'spec %s chunk %d used a later writing' % (spec_id, chunk))

    def test_post_chunk_row_ignores_later_writings(self):
        base = list(np.arange(20, dtype=float))
        history = make_history(base, score=0.9, distress=[0.1] * 20,
                               polarity=[0.2, -0.2] * 10)
        later = base + [21.0, 22.0, 23.0, 24.0, 25.0]
        extended = make_history(later, score=0.1, distress=[0.1] * 20 + [0.99] * 5,
                                polarity=[0.2, -0.2] * 10 + [0.9, -0.9, 0.9, -0.9, 0.9])
        extended.scores[:20] = history.scores[:20]
        first = {row['chunk_index']: row for row in v2.user_chunk_rows(history, 'p10')}
        second = {row['chunk_index']: row for row in v2.user_chunk_rows(extended, 'p10')}
        for chunk in (1, 2):
            self.assertEqual(first[chunk], second[chunk],
                             'p10 chunk %d used a later writing' % chunk)

    def test_changing_a_writing_inside_the_chunk_does_change_the_row(self):
        """Guard against a row that ignores everything, which would pass the test above."""
        history = make_history([0.0, 1.0], score=0.1)
        louder = make_history([0.0, 1.0], score=0.9)
        self.assertNotEqual(row_for(history, 'd7', 1)['llm__self_blame__level'],
                            row_for(louder, 'd7', 1)['llm__self_blame__level'])


class TestBank(unittest.TestCase):
    def setUp(self):
        self.built = {
            'a': make_history([0.0, 1.0, 2.0, 30.0], score=0.9, user_id='a'),
            'b': make_history([0.0, 10.0], score=0.1, user_id='b'),
            'c': v2.empty_history('c'),
        }

    def test_bank_has_every_user_and_chunk(self):
        bank = v2.raw_bank(self.built, 'd28')
        self.assertEqual(len(bank), 3 * 3)
        rows = v2.rows_at(bank, 2, ['a', 'b', 'c'])
        self.assertEqual(list(rows.index), ['a', 'b', 'c'])
        self.assertEqual(int(rows.loc['c', 'n_posts']), 0)

    def test_rows_at_fills_a_missing_post_chunk(self):
        bank = v2.raw_bank(self.built, 'p10')
        rows = v2.rows_at(bank, 3, ['a', 'b', 'c'])
        self.assertEqual(list(rows.index), ['a', 'b', 'c'])
        self.assertEqual(float(v2.encode(rows).to_numpy(float).sum()), 0.0)

    def test_chunk_for_day_on_day_specs(self):
        self.assertEqual(v2.chunk_for_day('d7', 28), 4)
        self.assertEqual(v2.chunk_for_day('d14', 28), 2)
        self.assertEqual(v2.chunk_for_day('d28', 84), 3)
        with self.assertRaises(ValueError):
            v2.chunk_for_day('d28', 30)

    def test_chunk_for_day_on_post_specs_uses_training_users(self):
        built = {str(i): make_history(list(np.arange(0.0, 60.0, 2.0)), user_id=str(i))
                 for i in range(6)}
        bank = v2.raw_bank(built, 'p10')
        # Writings every two days, so a 10-writing chunk ends near day 18, 38, 58.
        chosen = v2.chunk_for_day('p10', 28, bank, [str(i) for i in range(4)])
        self.assertIn(chosen, (1, 2))
        medians = v2.post_chunk_medians(bank, [str(i) for i in range(4)])
        self.assertEqual(int(medians.index.min()), 1)


class TestFeatureTable(unittest.TestCase):
    def test_every_encoded_feature_appears_in_the_table(self):
        rows = v2.feature_table_rows()
        named = {row['feature'] for row in rows}
        for column in v2.ALL_COLUMNS:
            stem = column
            for suffix in ('_0', '_1', '_2', '_3'):
                if stem.endswith('__level' + suffix):
                    stem = stem[: -len(suffix)]
            for value in v2.CHANGES + v2.TRENDS:
                if stem.endswith('__change_' + value):
                    stem = stem[: -len('_' + value)]
                if stem.endswith('__trend_' + value):
                    stem = stem[: -len('_' + value)]
            if stem.startswith('silence__class'):
                stem = 'silence__class'
            if stem.startswith('llm__'):
                parts = stem.split('__')
                stem = 'llm__%s__%s' % (parts[1], parts[2])
            self.assertIn(stem, named, '%s is not described in the feature table' % column)

    def test_table_says_whether_a_feature_is_llm_or_rule(self):
        for row in v2.feature_table_rows():
            self.assertIn(row['source'], ('LLM cache', 'rule'))
            self.assertTrue(row['type'])
            self.assertTrue(row['window'])


if __name__ == '__main__':
    unittest.main()
