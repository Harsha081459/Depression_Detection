import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erisk_candidate_signals import user_windows, writing_signs


class CandidateSignsTests(unittest.TestCase):
    def setUp(self):
        self.posts = pd.DataFrame([
            {"user_id": "a", "date": "2024-01-01T00:00:00Z", "kind": "comment", "title": "", "body": "I feel lonely with my family"},
            {"user_id": "a", "date": "2024-01-10T00:00:00Z", "kind": "comment", "title": "", "body": "We like games"},
            {"user_id": "a", "date": "2024-04-01T00:00:00Z", "kind": "submission", "title": "", "body": "I feel lonely with my family"},
            {"user_id": "b", "date": "2024-02-01T00:00:00Z", "kind": "comment", "title": "", "body": "They discuss depression"},
        ])

    def test_prefix_rows_do_not_depend_on_future_writings(self):
        full = writing_signs(self.posts.iloc[:3])
        partial = writing_signs(self.posts.iloc[:2])
        pd.testing.assert_frame_equal(full.iloc[:2], partial)
        self.assertEqual(full.iloc[0]["candidate_negative_family_post"], 1)
        self.assertEqual(full.iloc[1]["candidate_negative_family_post_so_far"], 0.5)
        self.assertGreater(full.iloc[2]["candidate_bigram_overlap_frac"], 0)

    def test_clinical_word_is_not_a_disclosure_or_negative_self_reference(self):
        rows = writing_signs(self.posts)
        other = rows.loc[rows["user_id"] == "b"].iloc[0]
        self.assertEqual(other["candidate_clinical_term_cnt"], 1)
        self.assertEqual(other["candidate_self_cnt"], 0)
        self.assertEqual(other["candidate_negative_self_post"], 0)
        self.assertEqual(other["candidate_bigram_overlap_frac"], 0)

    def test_input_is_sorted_and_history_isolated_by_user(self):
        rows = writing_signs(self.posts.iloc[::-1].reset_index(drop=True))
        self.assertEqual(rows.loc[rows["user_id"] == "a", "t"].tolist(), [1, 2, 3])
        self.assertEqual(rows.loc[rows["user_id"] == "b", "t"].tolist(), [1])
        self.assertNotIn("body", rows.columns)
        self.assertNotIn("title", rows.columns)

    def test_windows_are_user_level_prefixes(self):
        rows = writing_signs(self.posts)
        rows["label"] = rows["user_id"].map({"a": 1, "b": 0})
        windows = user_windows(rows)
        self.assertTrue((windows.groupby("window").size() == 2).all())
        self.assertEqual(windows.loc[(windows["user_id"] == "a") & (windows["window"] == "4w"), "t"].item(), 2)
        self.assertEqual(windows.loc[(windows["user_id"] == "a") & (windows["window"] == "full"), "t"].item(), 3)
        self.assertFalse({"body", "title", "span_days", "n_writings"} & set(windows.columns))


if __name__ == "__main__":
    unittest.main()
