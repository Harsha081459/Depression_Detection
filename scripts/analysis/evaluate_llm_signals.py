"""Evaluate the human pilot locally on the lab; never prints post text.

This implements Phase 2 only. Cohort AUROC/classifier evaluation must wait
for the independent human gate. Inputs are two blinded rater CSV exports
and an optional adjudication CSV (item_id,answer). Held-out evaluation is
allowed once per output path and only with frozen prompts/version 1.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys

from llm_signal_scorer import SIGNALS, SafeError, canonical_hash, load_prompts, write_json


def auroc(labels, scores):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def kappa(a, b):
    if not a or len(a) != len(b):
        raise SafeError('invalid_annotation_lengths')
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def read_answers(path):
    answers = {}
    if path is None:
        return answers
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['item_id', 'answer']:
            raise SafeError('answer_csv_requires_only_item_id_and_answer')
        for row in reader:
            key = row['item_id']
            answer = row['answer'].strip().lower()
            if key in answers or answer not in ('yes', 'no'):
                raise SafeError('duplicate_or_incomplete_annotation')
            answers[key] = int(answer == 'yes')
    return answers


def evaluate(tasks, a, b, adjudication, scores, split, config, preflight):
    selected = [t for t in tasks if t['split'] == split]
    required = {t['item_id'] for t in selected}
    if set(a) != required or set(b) != required or not set(adjudication).issubset(required):
        raise SafeError('annotation_ids_do_not_match_selected_split')
    if split == 'heldout' and config['prompt_version'] != 1:
        raise SafeError('heldout_requires_frozen_version1')
    if preflight.get('prompt_hash') != canonical_hash(config) or not preflight.get('phase1_passed'):
        raise SafeError('matching_phase1_report_required')
    results = {}
    for signal in SIGNALS:
        subset = [t for t in selected if t['signal'] == signal]
        if len(subset) != 60:
            raise SafeError('pilot_requires_sixty_items_per_signal_and_split')
        truth, model, lexical, rater_a, rater_b = [], [], [], [], []
        for task in subset:
            key = task['item_id']
            if a[key] != b[key] and key not in adjudication:
                raise SafeError('adjudication_required_for_disagreements')
            if a[key] == b[key] and key in adjudication and adjudication[key] != a[key]:
                raise SafeError('adjudication_conflicts_with_agreed_answer')
            label = adjudication[key] if a[key] != b[key] else a[key]
            pair = (task['post_id'], signal)
            if pair not in scores:
                raise SafeError('missing_matching_prompt_model_score')
            truth.append(label)
            model.append(scores[pair])
            lexical.append(task['word_list'])
            rater_a.append(a[key])
            rater_b.append(b[key])
        model_auc, lexical_auc = auroc(truth, model), auroc(truth, lexical)
        probes_passed = all(p['passed'] for p in preflight['probes'] if p['signal'] == signal)
        results[signal] = {'n': 60, 'human_yes': sum(truth), 'cohens_kappa': kappa(rater_a, rater_b),
                           'agreement': sum(x == y for x, y in zip(rater_a, rater_b)) / 60,
                           'adjudicated': sum(x != y for x, y in zip(rater_a, rater_b)),
                           'model_auroc': model_auc, 'word_list_auroc': lexical_auc,
                           'model_minus_word_list': None if model_auc is None else model_auc - lexical_auc,
                           'probes_passed': probes_passed,
                           'numeric_gate_passed': split == 'heldout' and model_auc is not None and model_auc >= .80 and probes_passed}
    return {'split': split, 'prompt_version': config['prompt_version'], 'prompt_hash': canonical_hash(config),
            'signals': results, 'note': 'Enriched pilot sample, not population prevalence. Researcher must judge word-list improvement before admitting a signal.'}


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache', 'rater-a', 'rater-b', 'phase1-report', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--adjudication', type=Path)
    ap.add_argument('--split', choices=('development', 'heldout'), default='development')
    ap.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompts.json'))
    args = ap.parse_args()
    try:
        if os.name != 'posix':
            raise SafeError('private_pilot_evaluation_must_run_on_lab_linux')
        if args.split == 'heldout' and args.output.exists():
            raise SafeError('heldout_output_exists_do_not_evaluate_repeatedly')
        config = load_prompts(args.prompts)
        if args.split == 'heldout' and config['prompt_version'] != 1:
            raise SafeError('freeze_version1_before_reading_heldout_annotations')
        tasks = json.loads(args.manifest.read_text(encoding='utf-8'))
        db = sqlite3.connect(str(args.cache))
        scores = {(pid, signal): score for pid, signal, score in db.execute(
            'SELECT post_id,signal,score FROM scores WHERE prompt_hash=? AND model_digest=?',
            (canonical_hash(config), config['expected_digest']))}
        db.close()
        result = evaluate(tasks, read_answers(args.rater_a), read_answers(args.rater_b),
                          read_answers(args.adjudication), scores, args.split, config,
                          json.loads(args.phase1_report.read_text(encoding='utf-8')))
        write_json(args.output, result)
        print(json.dumps(result, allow_nan=False))
        return 0
    except SafeError as e:
        print(json.dumps({'error': str(e)}))
        return 2
    except Exception:
        print(json.dumps({'error': 'pilot_evaluation_failed_details_withheld'}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
