"""Private lab-only Qwen scorer. Python 3.6+, standard library only.

Preflight uses made-up posts. Score input is a private JSONL export, never
printed. Cache contains IDs/hashes/numbers only, not text or model responses.
All HTTP requests stay on loopback; deploy this script onto the lab machine.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
SIGNALS = ('self_blame', 'helplessness', 'isolation', 'family_strain', 'emptiness')
BASE = 'http://127.0.0.1:11434'
CACHE_LOCK = threading.Lock()
PROBES = (
    ('game_blame', 'I keep failing this boss, I hate this level', 'self_blame', False),
    ('game_helpless', 'I keep failing this boss, I hate this level', 'helplessness', False),
    ('personal_blame', 'I am a failure and I ruin everything', 'self_blame', True),
    ('personal_helpless', "I can't do this anymore, nothing I try changes anything", 'helplessness', True),
    ('happy_alone', 'Spent the weekend alone hiking, loved it', 'isolation', False),
    ('lonely', 'Nobody ever texts me back, I feel invisible', 'isolation', True),
    ('family_dinner', 'Visited my parents for dinner', 'family_strain', False),
    ('family_conflict', "My dad screamed at me again, I can't stand being home", 'family_strain', True),
    ('numb', "Haven't felt anything in weeks, I just lie in bed", 'emptiness', True),
    ('ordinary_tired', 'Long day, going to sleep', 'emptiness', False),
)


class SafeError(Exception):
    """Only constant, text-free error descriptions reach stdout."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def load_prompts(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(config['signals']) != set(SIGNALS) or config['max_chars'] != 2000:
        raise SafeError('invalid_prompt_configuration')
    if config['model'] != 'qwen2.5:7b-instruct':
        raise SafeError('unexpected_model')
    return config


def request(endpoint, payload=None):
    raw = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE + endpoint, data=raw, headers={'Content-Type': 'application/json'})
    # Do not inherit proxy settings: private text must stay on this machine.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=180) as response:
            return json.load(response)
    except Exception:
        raise SafeError('ollama_request_failed') from None


def model_info(config):
    models = request('/api/tags').get('models', [])
    match = next((x for x in models if x.get('name') == config['model']), None)
    if not match or match.get('digest') != config['expected_digest']:
        raise SafeError('model_digest_mismatch')
    if config.get('cache_policy') == 'erase_before_each_call':
        # An empty request preloads exactly these options without scoring text.
        request('/api/generate', {'model': config['model'], 'stream': False,
                                 'keep_alive': '10m', 'options': config['options']})
    return {'model': config['model'], 'model_digest': match['digest'],
            'ollama_version': request('/api/version')['version']}


def payload_for(text, signal, config):
    question = config['signals'][signal]
    content = 'Post:\n"""\n%s\n"""\n\nQuestion: Does the writer %s?\nAnswer No if %s.\nAnswer Yes or No.' % (
        text[:config['max_chars']], question['yes'], question['no'])
    return {'model': config['model'], 'messages': [
        {'role': 'system', 'content': config['system']},
        {'role': 'user', 'content': content}],
        'stream': False, 'logprobs': True, 'top_logprobs': 20,
        'keep_alive': '10m',
        'options': config.get('options', {'temperature': 0, 'num_predict': 1, 'num_ctx': 2048, 'seed': 42})}


def logsumexp(values):
    largest = max(values)
    return largest + math.log(sum(math.exp(x - largest) for x in values))


def parse_logprobs(response):
    entries = response.get('logprobs')
    if not entries or not isinstance(entries, list):
        raise SafeError('missing_first_token_logprobs')
    top = entries[0].get('top_logprobs', [])
    if not top:
        raise SafeError('missing_top_logprobs')
    values = {'yes': [], 'no': []}
    finite = []
    for item in top:
        lp = float(item['logprob'])
        if not math.isfinite(lp):
            raise SafeError('nonfinite_logprob')
        finite.append(lp)
        token = item['token'].strip().lower()
        if token in values:
            values[token].append(lp)
    # Neither alternative present is not a meaningful binary answer.
    if not values['yes'] and not values['no']:
        raise SafeError('neither_yes_nor_no_present')
    missing = int(not values['yes'] or not values['no'])
    low = min(finite) - 1
    yes = logsumexp(values['yes']) if values['yes'] else low
    no = logsumexp(values['no']) if values['no'] else low
    score = math.exp(yes - logsumexp([yes, no]))
    return {'score': score, 'l_yes': yes, 'l_no': no, 'missing_flag': missing}


def erase_prompt_cache():
    """Only our owned, isolated one-slot service can be cleared."""
    if os.name != 'posix' or BASE.endswith(':11434'):
        raise SafeError('cache_erasure_requires_isolated_lab_service')
    expected_host = BASE.replace('http://', '').encode('ascii')
    matches = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            proc = Path('/proc') / pid
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().decode('utf-8').split('\0')
            if not args[0].endswith('/llama-server') or '--port' not in args:
                continue
            env = (proc / 'environ').read_bytes().split(b'\0')
            host_matches = b'OLLAMA_HOST=' + expected_host in env or b'OLLAMA_HOST=http://' + expected_host in env
            if host_matches and b'LLAMA_ARG_CACHE_RAM=0' in env and b'OLLAMA_NUM_PARALLEL=1' in env:
                matches.append(int(args[args.index('--port') + 1]))
        except (OSError, ValueError, UnicodeError):
            continue
    if len(matches) != 1:
        raise SafeError('isolated_one_slot_runner_not_found')
    req = urllib.request.Request('http://127.0.0.1:%d/slots/0?action=erase' % matches[0],
                                 data=b'{}', headers={'Content-Type': 'application/json'})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=15) as response:
            value = json.load(response)
        if value.get('id_slot') != 0 or 'n_erased' not in value:
            raise SafeError('cache_erasure_not_confirmed')
    except Exception:
        raise SafeError('prompt_cache_erasure_failed') from None


def _score_one(text, signal, config):
    start = time.monotonic()
    if config.get('cache_policy') == 'erase_before_each_call':
        erase_prompt_cache()
    response = request('/api/chat', payload_for(text, signal, config))
    result = parse_logprobs(response)
    result.update({'truncated_flag': int(len(text) > config['max_chars']),
                   'latency_seconds': time.monotonic() - start})
    return result


def score_one(text, signal, config):
    if config.get('cache_policy') == 'erase_before_each_call':
        # Erasure and inference form one operation even with queued clients.
        with CACHE_LOCK:
            return _score_one(text, signal, config)
    return _score_one(text, signal, config)


def parallel_calls(calls, workers, config):
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_one, text, signal, config) for text, signal in calls]
        scores = [future.result() for future in futures]
    return scores, time.monotonic() - start


def max_difference(a, b):
    return max(abs(x['score'] - y['score']) for x, y in zip(a, b))


def preflight(config, output):
    info = model_info(config)
    probes = []
    for name, text, signal, expected in PROBES:
        result = score_one(text, signal, config)
        probes.append(dict(result, probe=name, signal=signal, expected_yes=expected,
                           passed=result['score'] > .5 if expected else result['score'] < .5))
    report = dict(info, prompt_hash=canonical_hash(config), prompt_version=config['prompt_version'],
                  probes=probes, probes_passed=all(x['passed'] for x in probes), status='running')
    write_json(output, report)
    print(json.dumps({'event': 'probes', 'passed': sum(x['passed'] for x in probes), 'total': len(probes)}), flush=True)
    # Made-up, varied contexts only. Benchmark is never evidence of human validity.
    # Each block shares a post prefix across its five signals, like cohort scoring.
    calls = [(text + '\nSynthetic context number %d.' % i, signal)
             for i in range(20) for _, text, _, _ in PROBES[:1] for signal in SIGNALS]
    first, _ = parallel_calls(calls, 1, config)
    second, _ = parallel_calls(calls, 1, config)
    delta = max_difference(first, second)
    report['determinism'] = {'calls_each_run': 100, 'max_absolute_difference': delta, 'passed': delta <= .001}
    write_json(output, report)
    print(json.dumps({'event': 'determinism', **report['determinism']}), flush=True)
    benchmark_calls = [(PROBES[i % len(PROBES)][1] + '\nSynthetic benchmark context %d.' % (i // 5), SIGNALS[i % 5])
                       for i in range(200)]
    reference = None
    benchmarks = []
    for workers in (1, 4, 8):
        results, elapsed = parallel_calls(benchmark_calls, workers, config)
        if reference is None:
            reference = results
        diff = max_difference(reference, results)
        item = {'workers': workers, 'calls': 200, 'seconds': elapsed,
                'calls_per_second': 200 / elapsed, 'max_absolute_difference': diff,
                'identical_within_tolerance': diff <= .001,
                'missing_flag_rate': sum(x['missing_flag'] for x in results) / 200}
        benchmarks.append(item)
        report['benchmarks'] = benchmarks
        write_json(output, report)
        print(json.dumps({'event': 'benchmark', **item}), flush=True)
    valid = [x for x in benchmarks if x['identical_within_tolerance']]
    report['selected_workers'] = max(valid, key=lambda x: x['calls_per_second'])['workers']
    placement = next((x for x in request('/api/ps').get('models', []) if x.get('digest') == info['model_digest']), {})
    report['gpu_loaded'] = placement.get('size_vram', 0) > 0
    report['phase1_passed'] = report['probes_passed'] and report['determinism']['passed'] and report['gpu_loaded']
    report['status'] = 'complete'
    report['benchmark_scope'] = 'synthetic short posts; cohort throughput may be lower'
    write_json(output, report)
    return report


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    if os.name == 'posix':
        os.chmod(str(temp), 0o600)
    os.replace(str(temp), str(path))


class ScoreCache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS scores (
            post_id TEXT, signal TEXT, prompt_hash TEXT, prompt_version INTEGER,
            model_digest TEXT, ollama_version TEXT, text_hash TEXT,
            score REAL, l_yes REAL, l_no REAL, missing_flag INTEGER,
            truncated_flag INTEGER, latency_seconds REAL,
            PRIMARY KEY(post_id, signal, prompt_hash, model_digest, ollama_version))''')
        self.db.commit()

    def has(self, post_id, signal, prompt_hash, digest, version=None, text_hash=None):
        row = self.db.execute('SELECT ollama_version,text_hash FROM scores WHERE post_id=? AND signal=? AND prompt_hash=? AND model_digest=?',
                              (post_id, signal, prompt_hash, digest)).fetchall()
        return any((version is None or v == version) and (text_hash is None or t == text_hash) for v, t in row)

    def put(self, post, signal, config, info, result):
        self.db.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (
            post['post_id'], signal, canonical_hash(config), config['prompt_version'],
            info['model_digest'], info['ollama_version'], canonical_hash(post['text']),
            result['score'], result['l_yes'], result['l_no'], result['missing_flag'],
            result['truncated_flag'], result['latency_seconds']))
        self.db.commit()  # A completed call survives a crash.

    def close(self):
        self.db.close()


def validate_gate(config, report, workers):
    if (not report.get('phase1_passed') or report.get('status') != 'complete'
            or report.get('prompt_hash') != canonical_hash(config)
            or report.get('model_digest') != config['expected_digest']):
        raise SafeError('matching_successful_phase1_report_required')
    stable = [x['workers'] for x in report.get('benchmarks', []) if x['identical_within_tolerance']]
    if workers not in stable:
        raise SafeError('worker_count_failed_repeatability')


def validated_cohort_signals(config, gate):
    if config.get('prompt_version') != 1 or not isinstance(gate, dict) or gate.get('prompt_hash') != canonical_hash(config):
        raise SafeError('cohort_requires_matching_frozen_prompts_and_human_gate')
    signals = gate.get('validated_signals')
    if (not isinstance(signals, list) or not signals or
            any(s not in SIGNALS for s in signals) or len(set(signals)) != len(signals)):
        raise SafeError('invalid_validated_signal_list')
    return tuple(s for s in SIGNALS if s in signals)


def score_file(path, cache_path, workers, config, phase1_report, purpose='pilot', pilot_gate=None,
               heldout_report=None):
    if purpose not in ('pilot', 'cohort'):
        raise SafeError('invalid_scoring_purpose')
    cohort_signals = validated_cohort_signals(config, pilot_gate) if purpose == 'cohort' else ()
    if purpose == 'cohort':
        if (not isinstance(heldout_report, dict) or heldout_report.get('split') != 'heldout'
                or heldout_report.get('prompt_version') != 1
                or heldout_report.get('prompt_hash') != canonical_hash(config)):
            raise SafeError('matching_independent_heldout_report_required')
        for signal in cohort_signals:
            metrics = heldout_report.get('signals', {}).get(signal, {})
            if (metrics.get('n') != 60 or metrics.get('numeric_gate_passed') is not True
                    or metrics.get('probes_passed') is not True
                    or not isinstance(metrics.get('model_auroc'), (float, int))
                    or not isinstance(metrics.get('word_list_auroc'), (float, int))
                    or not 0.80 <= metrics['model_auroc'] <= 1
                    or not 0 <= metrics['word_list_auroc'] < metrics['model_auroc']):
                raise SafeError('admitted_signal_failed_heldout_gate')
    if os.name != 'posix':
        raise SafeError('private_scoring_must_run_on_lab_linux')
    info = model_info(config)
    validate_gate(config, phase1_report, workers)
    if info['ollama_version'] != phase1_report['ollama_version']:
        raise SafeError('ollama_version_changed_repeat_preflight')
    cache = ScoreCache(cache_path)
    completed, skipped, failed = 0, 0, 0
    def work(post, signal):
        return post, signal, score_one(post['text'], signal, config)
    try:
        # Bounded queue, ordered by post so five calls reuse the prefix.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()
            with open(str(path), encoding='utf-8') as source:
                for line in source:
                    post = json.loads(line)
                    signals = post.get('requested_signals', list(SIGNALS)) if purpose == 'pilot' else cohort_signals
                    if not signals or len(signals) != len(set(signals)) or not set(signals).issubset(SIGNALS):
                        raise SafeError('invalid_requested_signals')
                    if not post['text'].strip() or post['text'].strip().lower() in ('[removed]', '[deleted]'):
                        skipped += len(signals)
                        continue
                    for signal in signals:
                        if cache.has(post['post_id'], signal, canonical_hash(config), info['model_digest'],
                                     info['ollama_version'], canonical_hash(post['text'])):
                            skipped += 1
                            continue
                        pending.add(pool.submit(work, post, signal))
                        if len(pending) >= workers * 2:
                            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                            for future in done:
                                try:
                                    p, s, result = future.result()
                                    cache.put(p, s, config, info, result)
                                    completed += 1
                                except SafeError:
                                    failed += 1
                    if completed and completed % 100 == 0:
                        print(json.dumps({'completed': completed, 'skipped': skipped, 'failed': failed}), flush=True)
                for future in concurrent.futures.as_completed(pending):
                    try:
                        p, s, result = future.result()
                        cache.put(p, s, config, info, result)
                        completed += 1
                    except SafeError:
                        failed += 1
    finally:
        cache.close()
    return dict(info, completed=completed, skipped=skipped, failed=failed)


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('preflight', 'score'))
    ap.add_argument('--prompts', type=Path, default=HERE / 'prompts.json')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--workers', type=int, choices=(1, 4, 8), default=1)
    ap.add_argument('--port', type=int, default=11434, help='Loopback Ollama port on this lab machine')
    ap.add_argument('--phase1-report', type=Path)
    ap.add_argument('--purpose', choices=('pilot', 'cohort'), default='pilot')
    ap.add_argument('--pilot-gate', type=Path)
    ap.add_argument('--heldout-report', type=Path)
    args = ap.parse_args()
    global BASE
    if not 1024 <= args.port <= 65535:
        ap.error('port must be between 1024 and 65535')
    BASE = 'http://127.0.0.1:%d' % args.port
    try:
        config = load_prompts(args.prompts)
        if args.command == 'preflight':
            result = preflight(config, args.output)
            return 0 if result['phase1_passed'] else 2
        if args.input is None or args.cache is None:
            raise SafeError('score_requires_input_and_cache')
        if args.phase1_report is None:
            raise SafeError('score_requires_phase1_report')
        gate = json.loads(args.pilot_gate.read_text(encoding='utf-8')) if args.pilot_gate else None
        heldout = json.loads(args.heldout_report.read_text(encoding='utf-8')) if args.heldout_report else None
        report = json.loads(args.phase1_report.read_text(encoding='utf-8'))
        result = score_file(args.input, args.cache, args.workers, config, report, args.purpose, gate, heldout)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)
        return 0 if not result['failed'] else 2
    except SafeError as error:
        print(json.dumps({'error': str(error)}), flush=True)
        return 2
    except Exception:
        print(json.dumps({'error': 'unexpected_failure_details_withheld_to_protect_text'}), flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
