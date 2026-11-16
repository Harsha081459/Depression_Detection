import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meeting_windows as mw


def history(items):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    return mw.timed_events(SimpleNamespace(date=start + days * mw.DAY, text=text,
                                          disclosure=mw.lexical.disclosure(text))
                           for days, text in items)


class MeetingWindowTests(unittest.TestCase):
    def test_warmup_boundary_and_lookahead(self):
        items = [(0, "Hello"), (4, "I am lonely"), (5, "I am lonely"), (8, "I feel good")]
        prefix = mw.windows(history(items), 4, "masked")
        future = mw.windows(history(items + [(9, "I feel depressed")]), 4, "masked")
        self.assertEqual(prefix, future)
        self.assertEqual([x["day"] for x in prefix], [8])
        self.assertEqual(prefix[0]["frequency"], 2)
        self.assertEqual(prefix[0]["change_frequency"], "unchanged")
        self.assertEqual(prefix[0]["change_isolation"], "unchanged")
        self.assertEqual(prefix[0]["polarity_flip"], "1")
        self.assertEqual(mw.windows(history(items), 28, "masked"), [])

    def test_cut_before_disclosure_including_exact_decision_boundary(self):
        items = [(0, "Hi"), (5, "I am lonely"), (8, "I was diagnosed with depression")]
        self.assertEqual(mw.windows(history(items), 4, "before_disclosure"), [])
        masked = mw.windows(history(items), 4, "masked")[0]
        self.assertEqual(masked["frequency"], 2)
        self.assertEqual(masked["isolation"], "2")
        later = history([(0, "Hi"), (5, "I am lonely"), (9, "I feel depressed")])
        self.assertEqual(len(mw.windows(later, 4, "before_disclosure")), 1)

    def test_empty_bins_and_unavailable_evidence(self):
        records = mw.windows(history([(0, "Hi"), (6, "[deleted]"), (17, "Hello")]), 4, "masked")
        self.assertEqual([r["day"] for r in records], [20])
        self.assertEqual(records[0]["change_frequency"], "higher")
        self.assertEqual(records[0]["change_helpless"], "unknown")
        self.assertEqual(records[0]["distress_burst"], "unknown")
        self.assertEqual(records[0]["closed_silence"], "unknown")

    def test_only_closed_gaps_and_mask_preserves_timestamps(self):
        items = [(0, "Hi"), (0.1, "Hi"), (0.2, "Hi"), (0.3, "Hi"), (0.4, "I feel hopeless")]
        self.assertIsNone(history(items)[-1]["closed_silence"])
        self.assertEqual(history(items + [(6, "Hello")])[-1]["closed_silence"], 1)
        split = history(items + [(0.5, "I feel depressed"), (6, "Hello")])
        self.assertEqual(split[-2]["closed_silence"], 0)
        self.assertIsNone(split[-1]["closed_silence"])

    def test_simultaneous_posts_have_no_artificial_order(self):
        items = [(0, "Hi"), (6, "I feel great"), (6, "I feel hopeless")]
        swapped = [items[0], items[2], items[1]]
        self.assertEqual(mw.windows(history(items), 4, "masked"),
                         mw.windows(history(swapped), 4, "masked"))

    def test_all_cohort_metrics_keep_unseen_users_and_first_flag(self):
        labels = pd.Series([1, 1, 0, 0], index=["a", "b", "c", "d"])
        pred = pd.DataFrame([("a", 8, .7), ("a", 12, .1), ("c", 8, .9), ("d", 12, .3)],
                            columns=["user_id", "day", "score"])
        m = mw.metrics(pred, labels, 12)
        self.assertEqual([m[k] for k in ("true_positive", "false_positive", "true_negative", "false_negative")],
                         [1, 1, 1, 1])
        self.assertEqual(m["eligible_positive"], 1)
        self.assertEqual(m["median_first_flag_day_positive_among_flagged"], 8)


if __name__ == "__main__":
    unittest.main()
