import json
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_llm_signals import auroc, kappa
from llm_pilot_review import choose_tasks, review_page
from llm_signal_scorer import HERE, SIGNALS, canonical_hash, load_prompts, validate_gate, SafeError


class PilotTests(unittest.TestCase):
    def test_auc_ties_and_direction(self):
        self.assertEqual(auroc([0, 1], [1, 1]), .5)
        self.assertEqual(auroc([0, 1], [.1, .9]), 1)
        self.assertEqual(auroc([0, 1], [.9, .1]), 0)
        self.assertIsNone(auroc([0, 0], [.1, .9]))

    def test_kappa(self):
        self.assertEqual(kappa([0, 1, 0, 1], [0, 1, 0, 1]), 1)
        self.assertEqual(kappa([0, 1, 0, 1], [0, 1, 1, 0]), 0)

    def test_sampler_caps_users_and_balances_score_cells(self):
        config = load_prompts(HERE / 'prompts.json')
        records, scores = [], {}
        for split in ('development', 'heldout'):
            for signal in SIGNALS:
                for i in range(120):
                    pid = '%s-%s-%d' % (split, signal, i)
                    records.append({'post_id': pid, 'user_id': '%s-user-%d' % (split, i), 'split': split, 'label': i % 2,
                                    'domains': ['game', 'sport', 'news'], 'lexical': {s: 0 for s in SIGNALS},
                                    'text': 'Synthetic content.'})
                    scores[pid, signal] = .9 if i < 60 else .1
        tasks, shortfalls = choose_tasks(records, scores, config)
        self.assertFalse(shortfalls)
        self.assertEqual(len(tasks), 600)
        self.assertEqual(len({t['post_id'] for t in tasks}), 600)
        self.assertLessEqual(max(Counter(t['user_id'] for t in tasks).values()), 3)
        for split in ('development', 'heldout'):
            for signal in SIGNALS:
                subset = [t for t in tasks if t['split'] == split and t['signal'] == signal]
                self.assertEqual(len(subset), 60)
                self.assertEqual(sum(t['score'] >= .5 for t in subset), 30)

    def test_review_hides_labels_scores_user_ids_and_escapes_html(self):
        config = load_prompts(HERE / 'prompts.json')
        task = {'item_id': 'blind', 'split': 'development', 'signal': 'isolation', 'text': '</script><img src=x>',
                'user_id': 'SECRET_USER', 'label': 'SECRET_LABEL', 'score': 'SECRET_SCORE'}
        page = review_page([task], config, 'A', 'development')
        for hidden in ('SECRET_USER', 'SECRET_LABEL', 'SECRET_SCORE', '<img src=x>'):
            self.assertNotIn(hidden, page)
        self.assertIn('connect-src', page)

    def test_a_post_can_be_judged_for_two_different_signals(self):
        config = load_prompts(HERE / 'prompts.json')
        records, scores = {}, {}
        for split in ('development', 'heldout'):
            for j, signal in enumerate(SIGNALS):
                for high in (True, False):
                    for i in range(30):
                        name = 'shared' if j < 2 and i < 10 else signal
                        pid = '%s-%s-%s-%d' % (split, name, high, i)
                        records[pid] = {'post_id': pid, 'user_id': pid, 'split': split, 'label': i % 2,
                                        'domains': ['game', 'sport', 'news'], 'lexical': dict.fromkeys(SIGNALS, 0),
                                        'text': 'Synthetic multi-signal post.'}
                        scores[pid, signal] = .9 if high else .1
        tasks, shortfalls = choose_tasks(list(records.values()), scores, config)
        self.assertFalse(shortfalls)
        self.assertEqual(len(tasks), 600)
        self.assertEqual(len({t['post_id'] for t in tasks}), 560)
        self.assertEqual(len({(t['post_id'], t['signal']) for t in tasks}), 600)
        self.assertLessEqual(max(Counter(t['user_id'] for t in tasks).values()), 3)

    def test_reusing_signals_never_adds_a_fourth_post_for_one_user(self):
        config = load_prompts(HERE / 'prompts.json')
        records = [{'post_id': str(i), 'user_id': 'one-user', 'split': 'development', 'label': 0,
                    'domains': [], 'lexical': dict.fromkeys(SIGNALS, 0), 'text': 'Synthetic text.'} for i in range(5)]
        scores = {(r['post_id'], signal): .9 for r in records for signal in SIGNALS}
        tasks, _ = choose_tasks(records, scores, config)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len({t['post_id'] for t in tasks}), 3)
        self.assertLessEqual(max(Counter(t['post_id'] for t in tasks).values()), 2)

    def test_gate_rejects_untested_workers_and_changed_prompts(self):
        config = load_prompts(HERE / 'prompts.json')
        report = {'phase1_passed': True, 'status': 'complete', 'prompt_hash': canonical_hash(config),
                  'model_digest': config['expected_digest'], 'benchmarks': [{'workers': 1, 'identical_within_tolerance': True}]}
        validate_gate(config, report, 1)
        with self.assertRaises(SafeError):
            validate_gate(config, report, 4)
        changed = dict(config, prompt_version=99)
        with self.assertRaises(SafeError):
            validate_gate(changed, report, 1)


if __name__ == '__main__':
    unittest.main()
