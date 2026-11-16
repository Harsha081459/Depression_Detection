from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_signal_silence import CLOSED_COLUMNS, graded_silence

START = datetime(2024, 1, 1)


def timeline(gaps, e=1.0, polarity=0.0):
    hours = [0]
    for gap in gaps:
        hours.append(hours[-1] + gap)
    return [{'ts': START + timedelta(hours=h), 'e': e, 'polarity': polarity} for h in hours]


class SilenceTests(unittest.TestCase):
    def test_steady_posting(self):
        self.assertTrue(all(v == 0 for v in graded_silence(timeline([1] * 12)).values()))

    def test_sudden_distressed_gap(self):
        values = graded_silence(timeline([1] * 5 + [240]))
        self.assertEqual(values['silence_distress_sudden'], 1)
        self.assertEqual(values['silence_distress_fade'], 0)

    def test_fading_distressed_gap(self):
        values = graded_silence(timeline([1, 3, 9, 24, 240]))
        self.assertGreater(values['silence_distress_fade'], .9)
        self.assertEqual(values['silence_distress_sudden'], 0)

    def test_positive_tone_does_not_override_distress(self):
        values = graded_silence(timeline([1] * 5 + [240], e=.95, polarity=.8))
        self.assertGreater(values['silence_distress_sudden'], .9)
        self.assertLess(values['silence_after_ordinary'], .1)

    def test_ordinary_positive_gap(self):
        values = graded_silence(timeline([1] * 5 + [240], e=.05, polarity=.8))
        self.assertGreater(values['silence_after_ordinary'], .9)
        self.assertLess(values['silence_distress_sudden'], .1)

    def test_masked_timestamp_is_retained(self):
        rows = timeline([6] * 4 + [12, 12])
        rows[-2].update(e=0, polarity=0)
        self.assertTrue(all(v == 0 for v in graded_silence(rows).values()))
        # Deleting it would manufacture a 24-hour, 4x gap.
        deleted = rows[:-2] + rows[-1:]
        self.assertGreater(graded_silence(deleted)['silence_distress_sudden'], 0)

    def test_full_history_has_no_open_silence(self):
        self.assertTrue(all(v == 0 for v in graded_silence(timeline([1] * 8)).values()))

    def test_post_after_window_is_not_used(self):
        rows = timeline([1] * 5 + [28 * 24])
        values = graded_silence(rows, START + timedelta(days=28))
        self.assertTrue(all(values[name] == 0 for name in CLOSED_COLUMNS))
        self.assertGreater(values['open_silence_distress'], .9)

    def test_requires_four_earlier_positive_gaps(self):
        self.assertEqual(graded_silence(timeline([1, 1, 1, 240]))['silence_distress_sudden'], 0)
        self.assertEqual(graded_silence(timeline([0, 0, 1, 1, 1, 240]))['silence_distress_sudden'], 0)

    def test_baseline_excludes_current_gap_and_floor_is_inclusive(self):
        value = graded_silence(timeline([8, 8, 8, 8, 24]))['silence_distress_sudden']
        self.assertAlmostEqual(value, __import__('math').log2(3) / __import__('math').log2(30))
        self.assertEqual(graded_silence(timeline([1] * 4 + [23]))['silence_distress_sudden'], 0)


if __name__ == '__main__':
    unittest.main()
