"""Gate 1 backend diagnosis: HF BF16 vs the validated Ollama scorer.

Synthetic probes only. Reads no pilot posts, cohort data, or human labels.
Each test writes its own JSON under --out and is skipped when that file
exists (use --force to redo one test). Prompts are used exactly as frozen.

Lab (GPU + Ollama):
  python3 scripts/analysis/diagnose_backend.py --prompts scripts/prompts.json \
      --model-dir /path/to/Qwen2.5-7B-Instruct --port 12466

Laptop (tokenizer files only, T1-T3):
  python scripts/analysis/diagnose_backend.py --prompts scripts/analysis/prompts.json \
      --tokenizer-dir private/cohort_v1/diagnostics/reference/hf \
      --ollama-template private/cohort_v1/diagnostics/reference/ollama_template.txt --only T1,T2,T3
"""
import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_signal_scorer as ols  # noqa: E402  standard library only

SEED = 42
FAILED = ('game_blame', 'game_helpless')
VARIANTS = ('Yes', 'No', ' Yes', ' No', 'yes', 'no', 'YES', 'NO')
OLLAMA_TEMPLATE_DIGEST = 'eb4402837c7829a690fa845de4d7f3fd842c2adee476d5341da8a46ea9255175'
PILOT_MODEL_DIGEST = '845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e'
GENERATION_PROMPT = '<|im_start|>assistant\n'
PATH_TOLERANCE = 1e-4   # T5 manual vs plain logits, as specified
BATCH_TOLERANCE = 1e-3  # existing LocalScorer.check_backend batch/repeat tolerance
LOW_MASS = 0.5          # T4: below this, the prompt is not eliciting a Yes/No answer


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)
    return value


def probes(stage):
    rows = [p for p in ols.PROBES if stage == 'all' or p[0] in FAILED]
    assert len(rows) == (10 if stage == 'all' else 2)
    return rows


def messages_for(text, signal, config):
    return ols.payload_for(text, signal, config)['messages']


# ---------------------------------------------------------------- rendering
def render_ollama(messages):
    """Ollama Go template for qwen2.5:7b-instruct (template blob eb44028...),
    evaluated for one optional system message and user turns, no tools.

    The `{{-` markers trim the newlines around actions, so this yields
    <|im_start|>system\\n{S}<|im_end|>\\n ... <|im_start|>assistant\\n.
    """
    roles = [m['role'] for m in messages]
    if any(r not in ('system', 'user') for r in roles) or roles[-1] != 'user':
        raise ValueError('renderer covers system/user turns without tools only')
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
    out = '<|im_start|>system\n%s<|im_end|>\n' % system if system else ''
    for m in messages:
        if m['role'] == 'user':
            out += '<|im_start|>user\n%s<|im_end|>\n' % m['content']
    return out + GENERATION_PROMPT


def render_hf_jinja(template, messages, add_generation_prompt=True):
    """Same Jinja settings transformers uses for apply_chat_template."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)
    env.globals['raise_exception'] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                            tools=None, documents=None)


class Tokenizer:
    """transformers tokenizer on the lab; tokenizers + Jinja fallback on a laptop."""

    def __init__(self, directory):
        directory = Path(directory)
        try:
            import transformers
            from transformers import AutoTokenizer
            self.hf = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            self.kind = 'transformers'
            self.version = {'transformers': transformers.__version__}
            self.template = self.hf.chat_template
        except ImportError:
            import tokenizers
            self.hf = None
            self.fast = tokenizers.Tokenizer.from_file(str(directory / 'tokenizer.json'))
            self.template = json.loads((directory / 'tokenizer_config.json').read_text(encoding='utf-8'))['chat_template']
            self.kind = 'tokenizers+jinja2 (transformers not installed)'
            self.version = {'tokenizers': tokenizers.__version__}
        self.template_sha256 = sha256_bytes(self.template.encode('utf-8'))

    def render(self, messages):
        if self.hf is not None:
            return self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return render_hf_jinja(self.template, messages)

    def encode(self, text):
        if self.hf is not None:
            return self.hf.encode(text, add_special_tokens=False)
        return self.fast.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        if self.hf is not None:
            return self.hf.decode(ids)
        return self.fast.decode(ids, skip_special_tokens=False)


# --------------------------------------------------------------- T1 - T3
def char_diff(a, b):
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [{'op': op, 'hf': a[i1:i2], 'ollama': b[j1:j2], 'hf_at': i1, 'ollama_at': j1}
            for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != 'equal']


def t1_prompt_diff(tok, config, stage, template_path=None):
    template_ok = None
    if template_path:
        template_ok = sha256_bytes(Path(template_path).read_bytes()) == OLLAMA_TEMPLATE_DIGEST
    rows = []
    for name, text, signal, _ in probes(stage):
        messages = messages_for(text, signal, config)
        hf, ollama = tok.render(messages), render_ollama(messages)
        diff = char_diff(hf, ollama)
        unified = ''.join(difflib.unified_diff(hf.splitlines(True), ollama.splitlines(True), 'hf', 'ollama'))
        if unified:
            print('T1 %s\n%s' % (name, unified))
        rows.append({
            'probe': name, 'signal': signal, 'identical': hf == ollama, 'char_differences': diff,
            'unified_diff': unified, 'hf_prompt': hf, 'ollama_prompt': ollama,
            'system_message_in_both': config['system'] in hf and config['system'] in ollama,
            'whitespace_only_difference': bool(diff) and hf.split() == ollama.split(),
            'input_truncated': len(text) > config['max_chars'],
            'special_tokens': {k: [hf.count(k), ollama.count(k)] for k in ('<|im_start|>', '<|im_end|>')}})
    return {'test': 'T1_prompt_diff', 'stage': stage, 'all_identical': all(r['identical'] for r in rows),
            'ollama_renderer': 'python evaluation of the published Go template, system+user, no tools',
            'ollama_template_file_matches_pilot_digest': template_ok, 'hf_template_sha256': tok.template_sha256,
            'tokenizer': tok.kind, 'probes': rows}


def t2_chat_template(tok, config, stage):
    rows = []
    for name, text, signal, _ in probes(stage):
        rendered = tok.render(messages_for(text, signal, config))
        ids = tok.encode(rendered)
        tail = ids[-10:]
        rows.append({'probe': name, 'n_tokens': len(ids), 'ends_with_generation_prompt': rendered.endswith(GENERATION_PROMPT),
                     'last_10_ids': tail, 'last_10_decoded': [tok.decode([i]) for i in tail],
                     'assistant_turn_open': [tok.decode([i]) for i in ids[-3:]] == ['<|im_start|>', 'assistant', '\n']})
    passed = all(r['ends_with_generation_prompt'] and r['assistant_turn_open'] for r in rows)
    if passed:
        print('T2 assistant turn opens correctly on %d probes' % len(rows))
    return {'test': 'T2_chat_template', 'stage': stage, 'passed': passed, 'tokenizer': tok.kind, 'probes': rows}


def t3_token_ids(tok, scorer_ids=None):
    rows = []
    for variant in VARIANTS:
        ids = tok.encode(variant)
        rows.append({'variant': variant, 'ids': ids, 'n_tokens': len(ids),
                     'decoded': tok.decode(ids), 'multi_token': len(ids) != 1})
    chosen = [tok.encode(x)[0] for x in ('Yes', 'No')]
    ollama_counted = [r['variant'] for r in rows if r['variant'].strip().lower() in ('yes', 'no') and not r['multi_token']]
    return {'test': 'T3_token_ids', 'variants': rows, 'any_multi_token': any(r['multi_token'] for r in rows),
            'local_scorer_rule_ids': chosen, 'local_scorer_loaded_ids': scorer_ids,
            'local_scorer_ids_match_rule': None if scorer_ids is None else list(scorer_ids) == chosen,
            'hf_score_uses': ['Yes', 'No'],
            'ollama_score_uses': 'every top-20 first token whose stripped lowercase text is yes/no '
                                 '(log-sum-exp), e.g. %s' % ollama_counted}


# --------------------------------------------------------------- GPU tests
def _inputs(scorer, texts, signal):
    messages = [messages_for(t, signal, scorer.config) for t in texts]
    rendered = [scorer.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    enc = scorer.tokenizer(rendered, padding=True, truncation=False, add_special_tokens=False, return_tensors='pt')
    return {k: v.to('cuda:0') for k, v in enc.items()}


def _yes_no_vocab(tokenizer):
    groups = {'yes': [], 'no': []}
    for token, idx in tokenizer.get_vocab().items():
        word = tokenizer.convert_tokens_to_string([token]).strip().lower()
        if word in groups:
            groups[word].append(idx)
    return groups


def ollama_style_score(logprobs, tokenizer, k=20):
    """Apply the Ollama scorer's rule to a full HF distribution (top-k variants)."""
    import torch
    top = torch.topk(logprobs, k)
    entries = [{'token': tokenizer.decode([int(i)]), 'logprob': float(v)} for v, i in zip(top.values, top.indices)]
    try:
        return ols.parse_logprobs({'logprobs': [{'top_logprobs': entries}]})
    except ols.SafeError as error:
        return {'error': str(error)}


def t4_topk(scorer, stage):
    torch = scorer.torch
    groups = _yes_no_vocab(scorer.tokenizer)
    rows = []
    for name, text, signal, _ in probes(stage):
        with torch.no_grad():
            logits = scorer.model(**_inputs(scorer, [text], signal), use_cache=False).logits[0, -1, :].float()
        logprobs = logits.log_softmax(-1)
        probs = logprobs.exp()
        top = torch.topk(probs, 15)
        yes, no = (float(probs[scorer.ids[i]]) for i in (0, 1))
        rows.append({'probe': name, 'signal': signal,
                     'top15': [{'id': int(i), 'token': scorer.tokenizer.decode([int(i)]), 'p': float(v)}
                               for v, i in zip(top.values, top.indices)],
                     'mass_chosen_yes_no': yes + no, 'p_yes_two_token': yes / (yes + no),
                     'mass_all_yes_no_variants': float(probs[groups['yes'] + groups['no']].sum()),
                     'ollama_rule_on_hf_distribution': ollama_style_score(logprobs, scorer.tokenizer)})
    low = [r['probe'] for r in rows if r['mass_chosen_yes_no'] < LOW_MASS]
    return {'test': 'T4_full_vocab_topk', 'stage': stage, 'low_mass_threshold': LOW_MASS,
            'low_mass_probes': low, 'prompt_broken': bool(low), 'probes': rows}


def t5_logit_path(scorer, stage):
    torch = scorer.torch
    rows = []
    for name, text, signal, _ in probes(stage):
        manual = float(scorer.predict([text], signal)[0])
        with torch.no_grad():
            logits = scorer.model(**_inputs(scorer, [text], signal), use_cache=False).logits[0, -1, :].float()
        plain = float(logits[scorer.ids].softmax(-1)[0])
        rows.append({'probe': name, 'manual_lm_head_slice': manual, 'plain_logits': plain,
                     'abs_diff': abs(manual - plain), 'within_tolerance': abs(manual - plain) <= PATH_TOLERANCE})
    name, text, signal, _ = probes('failed')[0]
    longer = 'Spent the weekend alone hiking, loved it. ' * 6
    alone = float(scorer.predict([text], signal)[0])
    first = float(scorer.predict([text, longer], signal)[0])
    second = float(scorer.predict([longer, text], signal)[1])
    # The probe is the shorter item, so it is left-padded in both batch orders.
    padding = {'probe': name, 'batch1': alone, 'batch2_position0': first, 'batch2_position1': second,
               'max_abs_diff': max(abs(alone - first), abs(alone - second)), 'tolerance': BATCH_TOLERANCE}
    padding['passed'] = padding['max_abs_diff'] <= BATCH_TOLERANCE
    return {'test': 'T5_logit_path', 'stage': stage, 'tolerance': PATH_TOLERANCE,
            'passed': all(r['within_tolerance'] for r in rows) and padding['passed'],
            'probes': rows, 'left_padding': padding}


def _ollama(endpoint, payload, erase):
    if erase:
        try:
            ols.erase_prompt_cache()
        except ols.SafeError as error:
            # The one-slot runner only exists once the model is resident.
            if str(error) != 'isolated_one_slot_runner_not_found':
                raise
            ols.request('/api/generate', {'model': payload['model'], 'keep_alive': payload.get('keep_alive', '30m')})
            ols.erase_prompt_cache()
    return ols.request(endpoint, payload)


def _two_token(response):
    top = response['logprobs'][0]['top_logprobs']
    exact = {t['token']: t['logprob'] for t in top if t['token'] in ('Yes', 'No')}
    if len(exact) < 2:
        return None
    return 1 / (1 + math.exp(exact['No'] - exact['Yes']))


def t6_ollama(config, stage, port, hf_prompts):
    ols.BASE = 'http://127.0.0.1:%d' % port
    try:
        models = ols.request('/api/tags').get('models', [])
        version = ols.request('/api/version')['version']
    except ols.SafeError:
        return {'test': 'T6_ollama', 'stage': stage, 'reachable': False, 'port': port}
    digest = next((m.get('digest') for m in models if m.get('name') == config['model']), None)
    erase = config.get('cache_policy') == 'erase_before_each_call' and port != 11434
    rows = []
    for name, text, signal, _ in probes(stage):
        payload = ols.payload_for(text, signal, config)
        chat = _ollama('/api/chat', payload, erase)
        no_penalty = dict(payload, options=dict(payload['options'], repeat_penalty=1.0))
        chat_np = _ollama('/api/chat', no_penalty, erase)
        raw = {k: payload[k] for k in ('model', 'stream', 'logprobs', 'top_logprobs', 'keep_alive', 'options')}
        raw.update(prompt=hf_prompts[name], raw=True)
        gen = _ollama('/api/generate', raw, erase)
        row = {'probe': name, 'signal': signal, 'prompt_eval_count': chat.get('prompt_eval_count'),
               'raw_response_keys': sorted(chat)}
        try:
            scored = ols.parse_logprobs(chat)
        except ols.SafeError as error:
            row.update(logprobs_returned=False, error=str(error))
            rows.append(row)
            continue
        row.update(logprobs_returned=True, chat_score=scored['score'], l_yes=scored['l_yes'], l_no=scored['l_no'],
                   missing_flag=scored['missing_flag'], yes_no_mass=math.exp(scored['l_yes']) + math.exp(scored['l_no']),
                   two_token_score=_two_token(chat), repeat_penalty_1_score=ols.parse_logprobs(chat_np)['score'],
                   chat_top_logprobs=chat['logprobs'][0]['top_logprobs'])
        try:
            row['raw_hf_prompt_score'] = ols.parse_logprobs(gen)['score']
            row['raw_prompt_eval_count'] = gen.get('prompt_eval_count')
        except (ols.SafeError, KeyError, IndexError, TypeError):
            row['raw_hf_prompt_score'] = None
        rows.append(row)
    scored_rows = [r for r in rows if r['logprobs_returned']]
    same = [r['raw_hf_prompt_score'] is not None and abs(r['raw_hf_prompt_score'] - r['chat_score']) <= 1e-6
            for r in scored_rows]
    loaded = [{k: m.get(k) for k in ('name', 'digest', 'size', 'size_vram')}
              for m in ols.request('/api/ps').get('models', [])]
    return {'test': 'T6_ollama', 'stage': stage, 'reachable': True, 'port': port, 'ollama_version': version,
            'loaded_models': loaded,
            'model_digest': digest, 'digest_matches_pilot': digest == PILOT_MODEL_DIGEST, 'prefix_cache_erased': erase,
            'method': 'first generated token, top-20 log probabilities; log-sum-exp over yes/no variants; '
                      'score = exp(l_yes)/(exp(l_yes)+exp(l_no)); missing side = lowest top-20 logprob - 1',
            'logit_based': len(scored_rows) == len(rows),
            'chat_render_equals_hf_render': bool(scored_rows) and all(same), 'probes': rows}


def t7_quantization(model_dir, config, stage, t4, alt_tag, port):
    rows = {p: {'bf16_two_token': r['p_yes_two_token']} for p, r in ((r['probe'], r) for r in t4['probes'])}
    notes = []
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, quantization_config=quant,
                                                     device_map={'': 'cuda:0'}).eval()
        ids = [tok.encode(x, add_special_tokens=False)[0] for x in ('Yes', 'No')]
        for name, text, signal, _ in probes(stage):
            rendered = tok.apply_chat_template(messages_for(text, signal, config), tokenize=False, add_generation_prompt=True)
            enc = tok(rendered, add_special_tokens=False, return_tensors='pt').to('cuda:0')
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[0, -1, :].float()
            rows[name]['nf4_two_token'] = float(logits[ids].softmax(-1)[0])
        del model
        torch.cuda.empty_cache()
    except ImportError:
        notes.append('bitsandbytes not installed: 4-bit HF skipped')
    if alt_tag:
        ols.BASE = 'http://127.0.0.1:%d' % port
        alt = dict(config, model=alt_tag)
        erase = config.get('cache_policy') == 'erase_before_each_call' and port != 11434
        for name, text, signal, _ in probes(stage):
            response = _ollama('/api/chat', ols.payload_for(text, signal, alt), erase)
            rows[name]['ollama_' + alt_tag] = ols.parse_logprobs(response)['score']
    return {'test': 'T7_quantization', 'stage': stage, 'notes': notes, 'alt_ollama_tag': alt_tag,
            'scores': rows, 'scope': 'Probes only; prompts unchanged.'}


# ---------------------------------------------------------------- verdict
def verdict(results):
    """Map saved evidence to the Phase 3 branch. None means not yet decidable."""
    t1, t2, t3 = results.get('T1'), results.get('T2'), results.get('T3')
    t4, t5, t6 = results.get('T4'), results.get('T5'), results.get('T6')
    if t1 and not t1['all_identical']:
        return 'H1_prompt_string_differs'
    if t2 and not t2['passed']:
        return 'H2_chat_template'
    if t3 and (any(v['variant'] in ('Yes', 'No') and v['multi_token'] for v in t3['variants'])
               or t3.get('local_scorer_ids_match_rule') is False):
        return 'H3_token_ids'
    if t4 and t4['prompt_broken']:
        return 'hf_low_yes_no_mass'
    # Padding drift is a batch-size effect (Gate 2), not a wrong batch-1 score.
    if t5 and not all(r['within_tolerance'] for r in t5['probes']):
        return 'H4_logit_path'
    if t6 and t6.get('reachable') and not t6['logit_based']:
        return 'H6_ollama_not_logit_based'
    if t6 and t6.get('reachable') and t6['chat_render_equals_hf_render'] is False:
        return 'H1_prompt_string_differs'
    if all(results.get(k) for k in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')) and t6.get('reachable'):
        return 'H5_quantization_or_backend'
    return None


def meta(args, tok, scorer=None):
    setup = Path('results/hf_backend_setup.json')
    value = {'seed': SEED, 'python': platform.python_version(), 'prompt_hash': ols.canonical_hash(args.config),
             'prompt_version': args.config['prompt_version'], 'model_dir': str(args.model_dir or ''),
             'tokenizer_dir': str(args.tokenizer_dir or args.model_dir), 'tokenizer': tok.kind,
             'versions': tok.version, 'hf_revision': json.loads(setup.read_text()).get('revision') if setup.exists() else None,
             'chat_template_sha256': tok.template_sha256}
    if scorer is not None:
        value.update(torch=scorer.meta['torch'], transformers=scorer.meta['transformers'],
                     yes_no_ids=scorer.ids, weights=scorer.meta['weights'])
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--tokenizer-dir', type=Path)
    ap.add_argument('--ollama-template', type=Path)
    ap.add_argument('--out', type=Path, default=Path('private/cohort_v1/diagnostics'))
    ap.add_argument('--port', type=int, default=12466)
    ap.add_argument('--only', default='T1,T2,T3,T4,T5,T6,T7')
    ap.add_argument('--force', default='')
    ap.add_argument('--ollama-alt-tag', default='', help='Already-pulled tag, e.g. qwen2.5:7b-instruct-q8_0')
    args = ap.parse_args(argv)
    args.config = json.loads(args.prompts.read_text(encoding='utf-8'))
    wanted, force = set(args.only.split(',')), set(filter(None, args.force.split(',')))
    tok = Tokenizer(args.tokenizer_dir or args.model_dir)
    out = args.out
    results = {}

    def run(key, stage, fn):
        path = out / ('%s_%s.json' % (key, stage)) if stage else out / ('%s.json' % key)
        if key not in wanted:
            return None
        if path.exists() and key not in force:
            value = json.loads(path.read_text(encoding='utf-8'))
        else:
            value = write(path, fn())
        results[key] = value
        return value

    # HF BF16 fills most of a 16 GB card, so it is loaded only when an HF
    # result is missing and released before Ollama runs on the same GPU.
    def pending(key):
        stages = ('failed', 'all') if key in ('T4', 'T5') else (None,)
        return key in wanted and (key in force or any(
            not (out / ('%s_%s.json' % (key, s) if s else '%s.json' % key)).exists() for s in stages))

    scorer = None
    if pending('T4') or pending('T5'):
        from erisk_full_pipeline import LocalScorer
        scorer = LocalScorer(args.model_dir, args.prompts, batch_size=1)
        write(out / '00_meta.json', meta(args, tok, scorer))
    elif not (out / '00_meta.json').exists():
        write(out / '00_meta.json', meta(args, tok))
    run('T3', None, lambda: t3_token_ids(tok, None if scorer is None else scorer.ids))
    for stage in ('failed', 'all'):
        run('T1', stage, lambda: t1_prompt_diff(tok, args.config, stage, args.ollama_template))
        run('T2', stage, lambda: t2_chat_template(tok, args.config, stage))
        run('T4', stage, lambda: t4_topk(scorer, stage))
        run('T5', stage, lambda: t5_logit_path(scorer, stage))
    if scorer is not None:
        del scorer.model
        scorer.torch.cuda.empty_cache()
        scorer = None
    for stage in ('failed', 'all'):
        if 'T6' in wanted:
            hf_prompts = {p['probe']: p['hf_prompt'] for p in results['T1']['probes']} if 'T1' in results else {}
            if len(hf_prompts) != len(probes(stage)):
                hf_prompts = {n: tok.render(messages_for(t, s, args.config)) for n, t, s, _ in probes(stage)}
            t6 = run('T6', stage, lambda: t6_ollama(args.config, stage, args.port, hf_prompts))
            if t6 and not t6['reachable']:
                print('STOP: Ollama is not reachable on port %d' % args.port)
                break
    branch = verdict(results)
    if 'T7' in wanted and branch == 'H5_quantization_or_backend' and 'T4' in results:
        run('T7', 'all', lambda: t7_quantization(args.model_dir, args.config, 'all', results['T4'],
                                                  args.ollama_alt_tag, args.port))
    padding = (results.get('T5') or {}).get('left_padding')
    summary = write(out / 'summary.json', {'branch': branch, 'tests_present': sorted(results),
                                           'decidable': branch is not None,
                                           'left_padding_within_tolerance': padding['passed'] if padding else None})
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
