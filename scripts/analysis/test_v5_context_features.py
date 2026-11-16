import unittest

import numpy as np
import pandas as pd

import v5_context_features as v5
from v5_context_extract import REL_COMMENT_OTHER, REL_SUBMISSION

DAY = 86400
EMO = {'emo_%s' % e: 0 for e in v5.EMOTIONS}


def target(row, user, day, relation, disclosure=False):
    return {'source_row': row, 'user_id': user, 'ts': int(day * DAY), 'kind': 'comment',
            'thread_no': 0, 'relation': relation, 'parent_row': -1, 'prior_others_in_thread': 3,
            'prior_other_authors_in_thread': 2, 'disclosure': disclosure}


def link(role, row, day, author, sadness=0, tokens=10):
    return dict(EMO, role=role, target_row=row, ts=int(day * DAY), author=author, emo_sadness=sadness,
                n_tokens=tokens, clinical=0, support=1, second_person=1, question=0)


class ContextFeatures(unittest.TestCase):
    def setUp(self):
        self.targets = pd.DataFrame([
            target(0, 'a', 0, REL_SUBMISSION),
            target(1, 'a', 20, REL_COMMENT_OTHER),
            target(2, 'a', 40, REL_SUBMISSION, disclosure=True),
            target(3, 'b', 0, REL_SUBMISSION),
        ])
        self.context = pd.DataFrame([
            link('reply', 0, 1, author=7),
            link('reply', 0, 30, author=8, sadness=5),
            link('parent', 1, 19, author=7),
            link('reply', 2, 41, author=9, sadness=10),
        ])
        self.origins = pd.Series({'a': 0, 'b': 0})

    def rows(self, day):
        return v5.context_rows(self.targets, self.context, self.origins, ['a', 'b'], day)

    def test_reply_after_decision_is_not_used(self):
        early = self.rows(28)
        self.assertAlmostEqual(early.loc['a', 'ctx_share_answered'], 0.5)
        self.assertEqual(early.loc['a', 'rep_emo__sadness'], 0.0)
        later = self.rows(35)
        self.assertGreater(later.loc['a', 'rep_emo__sadness'], 0.0)

    def test_disclosure_writing_and_its_replies_are_dropped(self):
        full = self.rows(None)
        self.assertAlmostEqual(full.loc['a', 'ctx_share_submission'], 0.5)
        self.assertAlmostEqual(full.loc['a', 'rep_emo__sadness'], 5 / 20)

    def test_reciprocity_needs_an_earlier_reply_from_the_same_author(self):
        self.assertEqual(self.rows(28).loc['a', 'ctx_reciprocity'], 1.0)
        context = self.context.copy()
        context.loc[0, 'author'] = 99
        rows = v5.context_rows(self.targets, context, self.origins, ['a', 'b'], 28)
        self.assertEqual(rows.loc['a', 'ctx_reciprocity'], 0.0)

    def test_user_without_context_gets_zeros(self):
        row = self.rows(28).loc['b']
        self.assertEqual(row['ctx_share_answered'], 0.0)
        self.assertTrue(np.isfinite(self.rows(28).to_numpy()).all())

    def test_column_order_is_fixed(self):
        self.assertEqual(list(self.rows(28).columns), v5.ALL_COLUMNS)


if __name__ == '__main__':
    unittest.main()
