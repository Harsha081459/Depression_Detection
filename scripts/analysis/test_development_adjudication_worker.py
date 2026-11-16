"""Synthetic labels only: partial or extra adjudication must never be accepted."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wait_for_development_adjudication import completed_adjudication


class HumanAdjudicationGateTests(unittest.TestCase):
    def test_accepts_only_complete_conflict_only_answers(self):
        cases = [
            ('item_id,answer\nsynthetic-a,Yes\nsynthetic-b,No\n', True),
            ('item_id,answer\nsynthetic-a,Yes\n', False),
            ('item_id,answer\nsynthetic-a,Yes\nsynthetic-b,No\nsynthetic-c,Yes\n', False),
            ('item_id,answer\nsynthetic-a,Yes\nsynthetic-b,\n', False),
            ('item_id,answer\nsynthetic-a,Yes\nsynthetic-a,No\n', False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic_answers.csv'
            for content, expected in cases:
                with self.subTest(expected=expected, content=content):
                    path.write_text(content, encoding='utf-8')
                    self.assertEqual(completed_adjudication(path, {'synthetic-a', 'synthetic-b'}), expected)


if __name__ == '__main__':
    unittest.main()
