import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_indirect_pipeline as pipe


def row(ts, text, **overrides):
    parsed = pipe.flags(text)
    parsed.update({
        "ts": ts,
        "night": 0,
        "disclosure": pipe.disclosure(text),
        "text": text,
    })
    parsed.update(overrides)
    return parsed


class SignalTests(unittest.TestCase):
    def test_game_vent_is_not_self_blame(self):
        parsed = pipe.flags("I hate this boss and this weapon build is worthless")
        self.assertEqual(parsed["self_blame"], 0)
        self.assertEqual(parsed["distress"], 0.0)

    def test_family_mention_without_distress_is_zero(self):
        parsed = pipe.flags("I visited my family yesterday and we had dinner")
        self.assertEqual(parsed["family_strain"], 0)

    def test_personal_blame_and_family_strain(self):
        self.assertEqual(pipe.flags("I am a failure and I hate myself")["self_blame"], 1)
        self.assertEqual(pipe.flags("My mother hurt me and I feel awful")["family_strain"], 1)

    def test_disclosure_is_detected_and_not_a_feature_word(self):
        self.assertTrue(pipe.disclosure("I was diagnosed with depression last year"))
        self.assertTrue(pipe.disclosure("I feel depressed"))
        self.assertFalse(pipe.disclosure("I feel hopeless and trapped"))
        self.assertNotIn("depressed", pipe.INDIRECT_NEG)
        self.assertNotIn("suicidal", pipe.INDIRECT_NEG)

    def test_window_does_not_see_the_future(self):
        start = datetime(2024, 1, 1)
        frame = pd.DataFrame([
            {"user_id": "a", "date": start, "title": "", "body": "I visited my family yesterday"},
            {"user_id": "a", "date": start + timedelta(days=40), "title": "", "body": "I am a failure"},
        ])
        labels = pd.DataFrame([{"user_id": "a", "label": 1}])
        table = pipe.build_table(frame, labels)
        early = table.loc[table.window.eq("4w") & table.arm.eq("unmasked")].iloc[0]
        late = table.loc[table.window.eq("full") & table.arm.eq("unmasked")].iloc[0]
        self.assertEqual(early["n_writings"], 1)
        self.assertEqual(early["self_blame"], 0)
        self.assertEqual(late["n_writings"], 2)
        self.assertGreater(late["self_blame"], 0)

    def test_masked_arm_drops_the_disclosure_writing(self):
        start = datetime(2024, 1, 1)
        frame = pd.DataFrame([
            {"user_id": "a", "date": start, "title": "", "body": "I was diagnosed with depression"},
            {"user_id": "a", "date": start + timedelta(days=1), "title": "", "body": "I am a failure"},
        ])
        labels = pd.DataFrame([{"user_id": "a", "label": 1}])
        table = pipe.build_table(frame, labels)
        masked = table.loc[table.window.eq("full") & table.arm.eq("masked")].iloc[0]
        before = table.loc[table.window.eq("full") & table.arm.eq("before_disclosure")].iloc[0]
        self.assertEqual(masked["n_writings"], 1)
        self.assertEqual(masked["self_blame"], 1)
        self.assertEqual(before["n_writings"], 0)
        self.assertEqual(before["self_blame"], 0)

    def test_polarity_flips_and_silence_gap(self):
        start = datetime(2024, 1, 1)
        rows = [
            row(start + timedelta(hours=i), "I feel good and happy", distress=0.0, polarity=0.5)
            for i in range(4)
        ]
        rows.append(row(
            start + timedelta(hours=4), "I feel hopeless and trapped",
            distress=0.2, polarity=-0.5, helpless=1,
        ))
        rows.append(row(
            start + timedelta(hours=80), "I feel good and happy",
            distress=0.0, polarity=0.5,
        ))
        scored = pipe.aggregate(rows, None)
        self.assertGreater(scored["polarity_flip"], 0)
        self.assertGreater(scored["silence"], 0)
        self.assertGreater(scored["helpless"], 0)

    def test_removing_a_middle_disclosure_does_not_join_gaps(self):
        start = datetime(2024, 1, 1)
        times = [start + timedelta(hours=i) for i in range(5)]
        times.append(start + timedelta(hours=5, minutes=30))
        times.append(start + timedelta(hours=7))
        bodies = ["I feel hopeless and trapped"] * 5
        bodies.append("I was diagnosed with depression")
        bodies.append("I went to class")
        frame = pd.DataFrame({
            "user_id": ["a"] * 7,
            "date": times,
            "title": [""] * 7,
            "body": bodies,
        })
        labels = pd.DataFrame({"user_id": ["a"], "label": [1]})
        table = pipe.build_table(frame, labels)
        full = table[table.window.eq("full")]
        self.assertEqual(float(full.loc[full.arm.eq("unmasked"), "silence"].iloc[0]), 0.0)
        self.assertEqual(float(full.loc[full.arm.eq("masked"), "silence"].iloc[0]), 0.0)
        self.assertEqual(int(full.loc[full.arm.eq("masked"), "n_disclosure_removed"].iloc[0]), 1)

    def test_full_history_does_not_invent_silence_after_the_file(self):
        start = datetime(2024, 1, 1)
        rows = [row(start + timedelta(hours=i), "I feel hopeless", distress=0.2, polarity=-0.4) for i in range(6)]
        self.assertEqual(pipe.aggregate(rows, None)["silence"], 0.0)


if __name__ == "__main__":
    unittest.main()
