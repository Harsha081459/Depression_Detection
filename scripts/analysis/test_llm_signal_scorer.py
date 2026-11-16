import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_signal_scorer as scorer


def response(*tokens):
    return {'logprobs': [{'top_logprobs': [{'token': token, 'logprob': lp} for token, lp in tokens]}]}


class LogprobTests(unittest.TestCase):
    def test_cache_is_erased_before_inference(self):
        config = scorer.load_prompts(scorer.HERE / 'prompts.json')
        events = []
        def fake_request(*args):
            events.append('inference')
            return response(('Yes', -1), ('No', -2))
        with patch.object(scorer, 'erase_prompt_cache', side_effect=lambda: events.append('erase')):
            with patch.object(scorer, 'request', side_effect=fake_request):
                scorer.score_one('Synthetic text.', 'isolation', config)
        self.assertEqual(events, ['erase', 'inference'])

    def test_combines_case_and_space_variants(self):
        value = scorer.parse_logprobs(response(('Yes', math.log(.2)), (' YES', math.log(.1)), ('No', math.log(.7))))
        self.assertAlmostEqual(value['score'], .3)
        self.assertEqual(value['missing_flag'], 0)

    def test_missing_side_uses_lowest_minus_one(self):
        value = scorer.parse_logprobs(response(('No', -1), ('other', -4)))
        self.assertEqual(value['l_yes'], -5)
        self.assertEqual(value['missing_flag'], 1)
        self.assertAlmostEqual(value['score'], 1 / (1 + math.exp(4)))

    def test_log_space_does_not_underflow(self):
        value = scorer.parse_logprobs(response(('Yes', -1001), ('No', -1000)))
        self.assertAlmostEqual(value['score'], 1 / (1 + math.e))

    def test_bad_responses_fail_closed(self):
        for value in ({}, {'logprobs': []}, response(('maybe', -1)), response(('Yes', float('nan')))):
            with self.subTest(value=value), self.assertRaises(scorer.SafeError):
                scorer.parse_logprobs(value)

    def test_exact_prompt_shape_and_truncation(self):
        config = scorer.load_prompts(scorer.HERE / 'prompts.json')
        text = 'x' * 2000 + 'PRIVATE_TAIL'
        payload = scorer.payload_for(text, 'self_blame', config)
        self.assertEqual([m['role'] for m in payload['messages']], ['system', 'user'])
        self.assertNotIn('PRIVATE_TAIL', payload['messages'][1]['content'])
        self.assertLess(payload['messages'][1]['content'].index('Post:'), payload['messages'][1]['content'].index('Question:'))
        self.assertEqual(payload['options']['num_predict'], 1)
        self.assertEqual(payload['top_logprobs'], 20)

    def test_prompt_has_no_prohibited_task_terms(self):
        config = scorer.load_prompts(scorer.HERE / 'prompts.json')
        for signal in scorer.SIGNALS:
            messages = json.dumps(scorer.payload_for('Synthetic post.', signal, config)['messages']).lower()
            for word in ('depression', 'diagnosis', 'suicide', 'erisk', 'control'):
                self.assertNotIn(word, messages)

    def test_cache_survives_restart_without_text(self):
        config = scorer.load_prompts(scorer.HERE / 'prompts.json')
        info = {'model_digest': config['expected_digest'], 'ollama_version': 'test'}
        result = {'score': .2, 'l_yes': -2, 'l_no': -.5, 'missing_flag': 0, 'truncated_flag': 0, 'latency_seconds': .1}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cache.sqlite'
            cache = scorer.ScoreCache(path)
            cache.put({'post_id': 'synthetic', 'text': 'SECRET_SYNTHETIC_BODY'}, 'self_blame', config, info, result)
            cache.close()
            cache = scorer.ScoreCache(path)
            self.assertTrue(cache.has('synthetic', 'self_blame', scorer.canonical_hash(config), info['model_digest']))
            self.assertFalse(cache.has('synthetic', 'self_blame', 'changed_prompt', info['model_digest']))
            self.assertFalse(cache.has('synthetic', 'self_blame', scorer.canonical_hash(config), 'changed_model'))
            cache.close()
            self.assertNotIn(b'SECRET_SYNTHETIC_BODY', path.read_bytes())


if __name__ == '__main__':
    unittest.main()
