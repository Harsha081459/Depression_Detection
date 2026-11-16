"""One-shot lab worker: wait for human adjudication, evaluate development only.

No model calls, prompt changes, held-out release, or dataset text output.
The original rater files, prompt configuration and task manifest must stay
unchanged while the two humans settle their disagreements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate_llm_signals import read_answers
from llm_signal_scorer import canonical_hash, load_prompts, SafeError, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_adjudication(path, conflicts):
    try:
        answers = read_answers(path)
        return set(answers) == conflicts
    except Exception:
        return False


def finish(root, timeout_minutes):
    config_path = root / 'scripts/prompts.json'
    config = load_prompts(config_path)
    if config['prompt_version'] != 0:
        raise SafeError('worker_requires_development_prompt_version_zero')
    prompt_hash = canonical_hash(config)
    manifest = root / 'private/review/private_manifest.json'
    a_path = root / 'private/review/rater_A.csv'
    b_path = root / 'private/review/rater_B.csv'
    tasks = [t for t in json.loads(manifest.read_text(encoding='utf-8')) if t['split'] == 'development']
    expected = {t['item_id'] for t in tasks}
    a, b = read_answers(a_path), read_answers(b_path)
    if len(tasks) != 300 or set(a) != expected or set(b) != expected:
        raise SafeError('development_rater_files_incomplete_or_mismatched')
    conflicts = {i for i in expected if a[i] != b[i]}
    baseline = {p: digest(p) for p in [config_path, manifest, a_path, b_path]}
    report_path = root / 'results/phase1_cache_control.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('phase1_passed') or report.get('prompt_hash') != prompt_hash:
        raise SafeError('matching_successful_phase1_report_required')
    baseline[report_path] = digest(report_path)
    output = root / 'results/pilot_development.json'
    if output.exists():
        raise SafeError('development_output_already_exists')
    adjudication = root / 'private/review/adjudication.csv'
    state_path = root / 'results/development_completion_status.json'
    deadline = time.monotonic() + 60 * timeout_minutes
    state = {'status': 'awaiting_development_adjudication', 'development_items': 300,
             'disagreements': len(conflicts), 'timeout_minutes': timeout_minutes}
    write_json(state_path, state)
    print(json.dumps(state), flush=True)
    while True:
        if any(digest(path) != saved for path, saved in baseline.items()):
            raise SafeError('development_inputs_changed_stop_and_revalidate')
        ready = not conflicts or (adjudication.is_file() and completed_adjudication(adjudication, conflicts))
        if not ready and not adjudication.exists():
            # The review page's Save button uses this deterministic filename.
            # Accept only a complete conflict-only export for this prompt hash.
            name = 'erisk_review_development_adjudication_%s*.csv' % prompt_hash[:12]
            candidates = []
            for directory in [root / 'private/review', Path.home() / 'Downloads', Path.home() / 'Desktop']:
                if directory.is_dir():
                    candidates.extend(p for p in directory.glob(name) if completed_adjudication(p, conflicts))
            if candidates:
                source = max(candidates, key=lambda path: path.stat().st_mtime)
                shutil.copy2(str(source), str(adjudication))
                os.chmod(str(adjudication), 0o600)
                ready = True
        if ready:
            break
        if time.monotonic() >= deadline:
            state['status'] = 'timed_out_waiting_for_human_adjudication'
            write_json(state_path, state)
            print(json.dumps(state), flush=True)
            return 2
        time.sleep(15)
    if output.exists():
        raise SafeError('development_output_created_by_another_process')
    command = [sys.executable, str(root / 'scripts/evaluate_llm_signals.py'),
               '--manifest', str(manifest), '--cache', str(root / 'private/pilot_scores.sqlite'),
               '--rater-a', str(a_path), '--rater-b', str(b_path),
               '--phase1-report', str(report_path), '--split', 'development', '--output', str(output)]
    if conflicts:
        command.extend(['--adjudication', str(adjudication)])
    result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode != 0:
        raise SafeError('development_evaluation_failed_no_private_details_printed')
    metrics = json.loads(output.read_text(encoding='utf-8'))
    if metrics.get('split') != 'development' or metrics.get('prompt_hash') != prompt_hash:
        raise SafeError('unexpected_development_result_metadata')
    state['status'] = 'development_evaluated'
    write_json(state_path, state)
    pilot_path = root / 'results/pilot_status.json'
    pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
    pilot['status'] = 'development_evaluated'
    pilot['development_review'] = {'rater_A_answers': len(a), 'rater_B_answers': len(b),
                                    'disagreements': len(conflicts), 'adjudication_ready': True,
                                    'evaluation_completed': True}
    write_json(pilot_path, pilot)
    # Only aggregate metrics leave this process; no task references or text.
    print(json.dumps({'development_metrics': {signal: {key: value for key, value in data.items()
          if key in ('n', 'cohens_kappa', 'adjudicated', 'model_auroc', 'word_list_auroc')}
          for signal, data in metrics['signals'].items()}}), flush=True)
    return 0


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--timeout-minutes', type=int, default=120)
    args = ap.parse_args()
    try:
        if os.name != 'posix' or not 1 <= args.timeout_minutes <= 1440:
            raise SafeError('worker_requires_lab_linux_and_bounded_timeout')
        import fcntl
        with (args.root / 'private/development_finish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return finish(args.root, args.timeout_minutes)
    except SafeError as error:
        state = {'status': 'stopped', 'error': str(error)}
    except Exception:
        state = {'status': 'stopped', 'error': 'development_worker_failed_private_details_withheld'}
    write_json(args.root / 'results/development_completion_status.json', state)
    print(json.dumps(state), flush=True)
    return 2


if __name__ == '__main__':
    sys.exit(main())
