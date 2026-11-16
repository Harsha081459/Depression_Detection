import json
import math
import tempfile
import unittest
from pathlib import Path

import diagnose_backend as d
import llm_signal_scorer as ols

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'private' / 'cohort_v1' / 'diagnostics' / 'reference'
PROMPTS = Path(__file__).resolve().with_name('prompts.json')


def passing():
    return {'T1': {'all_identical': True}, 'T2': {'passed': True},
            'T3': {'variants': [{'variant': 'Yes', 'multi_token': False}, {'variant': 'No', 'multi_token': False}],
                   'local_scorer_ids_match_rule': True},
            'T4': {'prompt_broken': False},
            'T5': {'passed': True, 'probes': [{'within_tolerance': True}], 'left_padding': {'passed': True}},
            'T6': {'reachable': True, 'logit_based': True, 'chat_render_equals_hf_render': True}}


class DiagnoseBackendChecks(unittest.TestCase):
    def test_ollama_renderer_layout(self):
        text = d.render_ollama([{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'}])
        self.assertEqual(text, '<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n')

    def test_ollama_renderer_rejects_turns_it_does_not_cover(self):
        with self.assertRaises(ValueError):
            d.render_ollama([{'role': 'assistant', 'content': 'x'}, {'role': 'user', 'content': 'y'}])

    @unittest.skipUnless((REFERENCE / 'hf' / 'tokenizer_config.json').exists(), 'pinned HF template not downloaded')
    def test_hf_and_ollama_prompts_match_for_every_probe(self):
        template = json.loads((REFERENCE / 'hf' / 'tokenizer_config.json').read_text(encoding='utf-8'))['chat_template']
        config = json.loads(PROMPTS.read_text(encoding='utf-8'))
        for _, text, signal, _ in ols.PROBES:
            messages = d.messages_for(text, signal, config)
            self.assertEqual(d.render_hf_jinja(template, messages), d.render_ollama(messages))

    @unittest.skipUnless((REFERENCE / 'ollama_template.txt').exists(), 'Ollama template not downloaded')
    def test_ollama_template_is_the_pilot_build(self):
        self.assertEqual(d.sha256_bytes((REFERENCE / 'ollama_template.txt').read_bytes()), d.OLLAMA_TEMPLATE_DIGEST)

    def test_two_token_score_uses_exact_yes_no_only(self):
        response = {'logprobs': [{'top_logprobs': [{'token': 'No', 'logprob': math.log(.6)},
                                                   {'token': ' Yes', 'logprob': math.log(.3)},
                                                   {'token': 'Yes', 'logprob': math.log(.1)}]}]}
        self.assertAlmostEqual(d._two_token(response), .1 / .7)
        self.assertAlmostEqual(ols.parse_logprobs(response)['score'], .4)
        self.assertIsNone(d._two_token({'logprobs': [{'top_logprobs': [{'token': 'No', 'logprob': 0.}]}]}))

    def test_probe_stages(self):
        self.assertEqual([p[0] for p in d.probes('failed')], list(d.FAILED))
        self.assertEqual(len(d.probes('all')), 10)

    def test_verdict_branches(self):
        self.assertIsNone(d.verdict({}))
        self.assertEqual(d.verdict(passing()), 'H5_quantization_or_backend')
        cases = [('T1', {'all_identical': False}, 'H1_prompt_string_differs'),
                 ('T2', {'passed': False}, 'H2_chat_template'),
                 ('T4', {'prompt_broken': True}, 'hf_low_yes_no_mass'),
                 ('T5', {'passed': False, 'probes': [{'within_tolerance': False}]}, 'H4_logit_path'),
                 ('T6', {'reachable': True, 'logit_based': False, 'chat_render_equals_hf_render': True},
                  'H6_ollama_not_logit_based'),
                 ('T6', {'reachable': True, 'logit_based': True, 'chat_render_equals_hf_render': False},
                  'H1_prompt_string_differs')]
        for key, value, expected in cases:
            results = passing()
            results[key] = value
            self.assertEqual(d.verdict(results), expected)
        results = passing()
        results['T3']['variants'][0]['multi_token'] = True
        self.assertEqual(d.verdict(results), 'H3_token_ids')
        results = passing()
        results['T6'] = {'reachable': False}
        self.assertIsNone(d.verdict(results))

    def test_padding_drift_alone_is_not_a_logit_path_bug(self):
        results = passing()
        results['T5'] = {'passed': False, 'probes': [{'within_tolerance': True}], 'left_padding': {'passed': False}}
        self.assertEqual(d.verdict(results), 'H5_quantization_or_backend')

    @unittest.skipUnless((REFERENCE / 'hf' / 'tokenizer.json').exists(), 'pinned HF tokenizer not downloaded')
    def test_finished_tests_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ['--prompts', str(PROMPTS), '--tokenizer-dir', str(REFERENCE / 'hf'), '--out', tmp, '--only', 'T3']
            d.main(args)
            path = Path(tmp) / 'T3.json'
            path.write_text(json.dumps({'sentinel': True, 'variants': [], 'local_scorer_ids_match_rule': None}))
            d.main(args)
            self.assertTrue(json.loads(path.read_text())['sentinel'])
            d.main(args + ['--force', 'T3'])
            self.assertNotIn('sentinel', json.loads(path.read_text()))

    def test_cache_erasure_loads_an_unloaded_model_first(self):
        calls, state = [], {'loaded': False}

        def erase():
            calls.append('erase')
            if not state['loaded']:
                raise ols.SafeError('isolated_one_slot_runner_not_found')

        def request(endpoint, payload=None):
            calls.append(endpoint)
            if endpoint == '/api/generate' and 'prompt' not in payload:
                state['loaded'] = True
            return {'ok': True}

        saved = ols.erase_prompt_cache, ols.request
        ols.erase_prompt_cache, ols.request = erase, request
        try:
            d._ollama('/api/chat', {'model': 'm', 'keep_alive': '5m'}, erase=True)
            d._ollama('/api/chat', {'model': 'm'}, erase=True)
        finally:
            ols.erase_prompt_cache, ols.request = saved
        self.assertEqual(calls, ['erase', '/api/generate', 'erase', '/api/chat', 'erase', '/api/chat'])

    def test_other_erasure_failures_are_not_hidden(self):
        def erase():
            raise ols.SafeError('prompt_cache_erasure_failed')

        saved = ols.erase_prompt_cache
        ols.erase_prompt_cache = erase
        try:
            with self.assertRaises(ols.SafeError):
                d._ollama('/api/chat', {'model': 'm'}, erase=True)
        finally:
            ols.erase_prompt_cache = saved


if __name__ == '__main__':
    unittest.main()
