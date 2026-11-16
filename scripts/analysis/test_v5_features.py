"""Unit tests for v5. Run before the v5 cross-validation."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import v5_features as v5


def writings(deltas, scores, user='u'):
    frame = pd.DataFrame({'user_id': user, 'delta_days': deltas})
    for k, s in enumerate(v5.SIGNALS):
        frame[s] = [row[k] if isinstance(row, (list, tuple)) else row for row in scores]
    return frame


class TestClinical(unittest.TestCase):
    def test_strips_words_and_stems(self):
        text = 'My therapist said the Depression and my meds and Zoloft dose, ok'
        out = v5.strip_clinical(text).lower()
        for word in ('therapist', 'depression', 'meds', 'zoloft'):
            self.assertNotIn(word, out)
        self.assertIn('dose', out)
        self.assertIn('said', out)

    def test_keeps_ordinary_words(self):
        self.assertEqual(v5.strip_clinical('the game was hard'), 'the game was hard')


class TestTail(unittest.TestCase):
    def test_values(self):
        row = v5.tail_rows(writings([0, 1, 2, 3], [0.9, 0.6, 0.3, 0.1]), 28, ['u']).loc['u']
        self.assertAlmostEqual(row['tail__self_blame__max'], 0.9)
        self.assertAlmostEqual(row['tail__self_blame__top3'], 0.6)
        self.assertAlmostEqual(row['tail__self_blame__gt50'], 0.5)
        self.assertAlmostEqual(row['tail__self_blame__gt25'], 0.75)
        self.assertAlmostEqual(row['tail__two_gt50'], 0.5)

    def test_ignores_later_writings(self):
        base = writings([0, 5, 20], [0.2, 0.4, 0.1])
        later = pd.concat([base, writings([30, 60], [0.99, 0.99])], ignore_index=True)
        pd.testing.assert_series_equal(v5.tail_rows(base, 28, ['u']).loc['u'],
                                       v5.tail_rows(later, 28, ['u']).loc['u'])

    def test_empty_user_is_zero(self):
        row = v5.tail_rows(writings([0], [0.5]), 28, ['u', 'v']).loc['v']
        self.assertEqual(float(row.sum()), 0.0)


class TestText(unittest.TestCase):
    def setUp(self):
        self.texts = pd.DataFrame({
            'user_id': ['a', 'a', 'b', 'b', 'c'],
            'delta_days': [0, 40, 1, 2, 3],
            'text_clin': ['sad day', 'later words', 'happy game', 'fun game', 'x'],
        })
        self.texts['text_noclin'] = self.texts.text_clin

    def test_document_is_the_sum_of_prefix_writings(self):
        day = v5.TextDay(self.texts, 28, ['a', 'b', 'c'])
        h = v5.hasher()
        expected = h.transform(['happy game']) + h.transform(['fun game'])
        diff = day.doc['noclin'][day.row['b']] - expected
        self.assertEqual(abs(diff).sum(), 0)
        later = h.transform(['later words'])
        overlap = day.doc['noclin'][day.row['a']].multiply(later)
        self.assertEqual(overlap.sum(), 0)      # day-40 writing not in the day-28 document

    def test_text_model_learns_only_from_given_rows(self):
        X = v5.hasher().transform(['good good', 'bad bad', 'good', 'bad', 'good bad']).tocsr()
        m = v5.TextModel(1, 1.0).fit(X[:4], [1, 0, 1, 0])
        self.assertGreater(m.score(X[[0]])[0], m.score(X[[1]])[0])


if __name__ == '__main__':
    unittest.main()
