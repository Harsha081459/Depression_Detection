"""Resumable lab pilot runner. Stops at human review; never scores the cohort.

Only reuses the approved pool. Enrichment asks additional questions about
those same posts; it never adds/transfers a post or changes a prompt.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys

import llm_signal_scorer as scorer
from llm_pilot_review import build, expand_requests


def run(root, port, pool_path=None, preflight_path=None, max_rounds=10):
    config = scorer.load_prompts(root / 'scripts/prompts.json')
    preflight = json.loads((preflight_path or root / 'results/phase1_cache_control.json').read_text(encoding='utf-8'))
    scorer.BASE = 'http://127.0.0.1:%d' % port
    private, output = root / 'private', root / 'results'
    pool = pool_path if pool_path is not None else private / 'pilot_pool.jsonl'
    cache = private / 'pilot_scores.sqlite'
    state = {'status': 'scoring', 'prompt_hash': scorer.canonical_hash(config), 'rounds': []}
    for round_number in range(max_rounds):
        scorer.write_json(output / 'pilot_status.json', state)
        result = scorer.score_file(pool, cache, preflight['selected_workers'], config, preflight, 'pilot')
        state['rounds'].append(dict(result, round=round_number))
        print(json.dumps({'event': 'scoring_round', **state['rounds'][-1]}), flush=True)
        scorer.write_json(output / 'pilot_scoring.json', result)
        summary = build(pool, cache, private / 'review', config)
        state['sampling'] = summary
        if summary['ready_for_review']:
            state['status'] = 'awaiting_two_independent_human_raters'
            break
        if round_number == max_rounds - 1:
            state['status'] = 'pilot_sample_shortfall'
            break
        next_pool = private / ('pilot_pool_round%d.jsonl' % (round_number + 1))
        expansion = expand_requests(pool, cache, private / 'review/sampling_summary.json', next_pool, config)
        print(json.dumps({'event': 'existing_pool_enrichment', **expansion}), flush=True)
        if expansion['additional_questions'] == 0:
            state['status'] = 'pilot_sample_shortfall'
            break
        pool = next_pool
    db = sqlite3.connect(str(cache))
    row = db.execute('SELECT count(*),sum(missing_flag),min(score),max(score),sum(latency_seconds) FROM scores WHERE prompt_hash=?',
                     (scorer.canonical_hash(config),)).fetchone()
    state['score_quality'] = {'cached_calls': row[0], 'missing_flag_rate': row[1] / row[0] if row[0] else None,
                              'min_score': row[2], 'max_score': row[3], 'sum_latency_seconds': row[4]}
    db.close()
    state['final_pool'] = str(pool)
    scorer.write_json(output / 'pilot_status.json', state)
    print(json.dumps({'event': 'pilot_stopped', 'status': state['status'], 'selected_tasks': state['sampling']['selected_tasks']}), flush=True)
    return 0 if state['status'] == 'awaiting_two_independent_human_raters' else 2


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--port', type=int, default=12466)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--wait-for-lock', action='store_true')
    ap.add_argument('--require-private-check', action='store_true')
    ap.add_argument('--verify-first', action='store_true')
    ap.add_argument('--phase1-report', type=Path)
    ap.add_argument('--max-rounds', type=int, choices=range(1, 21), default=10)
    args = ap.parse_args()
    try:
        # Prevent duplicate pilot workers from changing request order.
        with (args.root / 'private/pilot_runner.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if args.wait_for_lock else fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.verify_first:
                from verify_llm_signal_cache import verify
                config = scorer.load_prompts(args.root / 'scripts/prompts.json')
                phase1_path = args.phase1_report or args.root / 'results/phase1_cache_control.json'
                phase1 = json.loads(phase1_path.read_text(encoding='utf-8'))
                pool_path = args.pool or args.root / 'private/pilot_pool.jsonl'
                records = [json.loads(line) for line in pool_path.read_text(encoding='utf-8').splitlines()][:20]
                mini_pool = args.root / 'private/repeatability_pool.jsonl'
                with mini_pool.open('w', encoding='utf-8') as f:
                    for record in records:
                        record['requested_signals'] = list(scorer.SIGNALS)
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                scorer.BASE = 'http://127.0.0.1:%d' % args.port
                scorer.score_file(mini_pool, args.root / 'private/pilot_scores.sqlite', 1, config, phase1)
                check = verify(mini_pool, args.root / 'private/pilot_scores.sqlite', config)
                scorer.write_json(args.root / 'results/private_repeatability_current.json', check)
                print(json.dumps({'event': 'private_repeatability', **check}), flush=True)
                if not check['passed']:
                    raise scorer.SafeError('private_repeatability_check_failed')
            if args.require_private_check:
                check = json.loads((args.root / 'results/private_repeatability_current.json').read_text(encoding='utf-8'))
                config = scorer.load_prompts(args.root / 'scripts/prompts.json')
                if not check.get('passed') or check.get('prompt_hash') != scorer.canonical_hash(config):
                    raise scorer.SafeError('private_repeatability_check_must_pass_before_continuing')
            return run(args.root, args.port, args.pool, args.phase1_report, args.max_rounds)
    except BlockingIOError:
        print(json.dumps({'error': 'pilot_runner_already_active'}))
        return 2
    except scorer.SafeError as e:
        print(json.dumps({'error': str(e)}))
        return 2
    except Exception:
        print(json.dumps({'error': 'pilot_runner_failed_details_withheld'}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
