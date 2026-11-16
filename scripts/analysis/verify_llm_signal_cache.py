"""Repeat 100 cached pilot questions privately; report numbers only."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

import llm_signal_scorer as scorer


def verify(pool_path, cache_path, config):
    records = {r['post_id']: r for r in (json.loads(line) for line in Path(pool_path).read_text(encoding='utf-8').splitlines())}
    db = sqlite3.connect(str(cache_path))
    info = scorer.model_info(config)
    rows = db.execute('SELECT post_id,signal,score FROM scores WHERE prompt_hash=? AND model_digest=? AND ollama_version=?',
                      (scorer.canonical_hash(config), info['model_digest'], info['ollama_version'])).fetchall()
    db.close()
    calls = []
    for signal in scorer.SIGNALS:
        candidates = sorted((r for r in rows if r[1] == signal and r[0] in records), key=lambda r: scorer.canonical_hash(['verify', r[0]]))
        if len(candidates) < 20:
            raise scorer.SafeError('not_enough_cached_questions_for_verification')
        calls.extend(candidates[:20])
    request_calls = [(records[pid]['text'], signal) for pid, signal, _ in calls]
    first, elapsed_a = scorer.parallel_calls(request_calls, 1, config)
    second, elapsed_b = scorer.parallel_calls(request_calls, 1, config)
    delta = scorer.max_difference(first, second)
    vs_cache = max(abs(result['score'] - old) for result, (_, _, old) in zip(second, calls))
    return dict(info, status='complete', calls_each_run=100, prompt_hash=scorer.canonical_hash(config),
                max_absolute_difference=delta, max_difference_from_original_cache=vs_cache,
                passed=delta <= .001 and vs_cache <= .001,
                seconds=elapsed_a + elapsed_b, scope='100 real private pilot questions, 20 per signal; no text exported')


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'cache', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompts.json'))
    ap.add_argument('--port', type=int, default=12466)
    args = ap.parse_args()
    try:
        if os.name != 'posix':
            raise scorer.SafeError('private_verification_must_run_on_lab_linux')
        import fcntl
        scorer.BASE = 'http://127.0.0.1:%d' % args.port
        scorer.write_json(args.output, {'status': 'waiting_for_pilot_runner'})
        with (args.cache.parent / 'pilot_runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            scorer.write_json(args.output, {'status': 'checking_private_repeatability'})
            result = verify(args.pool, args.cache, scorer.load_prompts(args.prompts))
            scorer.write_json(args.output, result)
        print(json.dumps(result), flush=True)
        return 0 if result['passed'] else 2
    except scorer.SafeError as error:
        result = {'status': 'failed', 'error': str(error)}
    except Exception:
        result = {'status': 'failed', 'error': 'private_verification_failed_details_withheld'}
    scorer.write_json(args.output, result)
    print(json.dumps(result), flush=True)
    return 2


if __name__ == '__main__':
    sys.exit(main())
