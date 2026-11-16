"""Thread context for eRisk Task 2 (v5, exploratory): one pass over the raw JSON.

The raw files hold each subject's threads: the submission and every comment,
including other people's, linked by `parent_id`. v1-v4 used only the subject's
own (`target == true`) writings. This pass keeps, for every target writing,
how it sits in its thread and what other people wrote directly around it:

    targets.parquet   one row per target writing, in the same order as
                      data/processed/erisk/task2_target_text.csv (checked row
                      by row): position in the thread and the disclosure flag.
    context.parquet   one row per other-user writing that is a direct reply to
                      a target writing (role 'reply') or the writing a target
                      comment answers (role 'parent'): timestamp, hashed
                      author, NRC counts with clinical words removed, and cue
                      counts.

No text and no raw author id is written. Output goes to private/ only.
A flag built from these features is for human review, not a diagnosis.

    python scripts/analysis/v5_context_extract.py
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'archive'))

import erisk_sequential_features as seq
from erisk_task2_features import _to_dt
from final_indirect_pipeline import disclosure

ROOT = HERE.parents[1]
RAW = ROOT / 'data' / 'raw' / 'erisk26'
TEXT_CSV = ROOT / 'data' / 'processed' / 'erisk' / 'task2_target_text.csv'
LEXICON = ROOT / 'data' / 'raw' / 'lexicons' / 'nrc' / 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

# Relation of a target writing to what it answers.
REL_SUBMISSION, REL_ROOT_OTHER, REL_COMMENT_OTHER, REL_SELF, REL_MISSING = range(5)

SUPPORT = re.compile(
    r"\b(?:sorry|hugs?|hope you|here for you|feel better|you'?re not alone|not alone|"
    r"take care|proud of you|you got this|reach out|talk to someone|are you ok(?:ay)?|"
    r"sending (?:love|hugs)|i hear you|stay strong)\b", re.I)
SECOND_PERSON = re.compile(r"\b(?:you|your|you're|youre|yours|yourself)\b", re.I)


def author_hash(name) -> int:
    digest = hashlib.blake2b(str(name).encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little', signed=True)


def to_seconds(dt) -> int:
    return int(dt.timestamp())


def nodes(item):
    """Same traversal as erisk_task2_features.writings_from_user_file."""
    out = []
    if isinstance(item.get('submission'), dict):
        out.append(item['submission'])
    comments = item.get('comments')
    if isinstance(comments, list):
        out.extend(c for c in comments if isinstance(c, dict))
    if 'created_utc' in item or 'date' in item:
        out.append(item)
    return out


def node_id(node):
    return node.get('comment_id') or node.get('submission_id')


def node_text(node) -> str:
    return (str(node.get('title') or '') + ' ' + str(node.get('body') or '')).strip()


class Counter:
    def __init__(self, lexicon_path: Path):
        full = seq.load_nrc(lexicon_path)
        clinical = set(seq.CLINICAL_CANDIDATES)
        self.lexicon = {w: s for w, s in full.items() if w not in clinical}
        self.vec, self.ind, _ = seq.emotion_counter(self.lexicon)
        self.tokens = CountVectorizer(lowercase=True, token_pattern=seq.TOKEN_PATTERN).build_analyzer()
        self.clinical = CountVectorizer(vocabulary=sorted(clinical), lowercase=True,
                                        token_pattern=seq.TOKEN_PATTERN)

    def count(self, texts: list[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=['emo_%s' % e for e in seq.EMOTIONS] +
                                ['n_tokens', 'clinical', 'support', 'second_person', 'question'])
        emo = np.asarray((self.vec.transform(texts) @ self.ind).todense(), dtype=np.int32)
        frame = pd.DataFrame(emo, columns=['emo_%s' % e for e in seq.EMOTIONS])
        frame['n_tokens'] = np.fromiter((len(self.tokens(t)) for t in texts), np.int32, len(texts))
        frame['clinical'] = np.asarray(self.clinical.transform(texts).sum(axis=1)).ravel().astype(np.int32)
        frame['support'] = np.fromiter((len(SUPPORT.findall(t)) for t in texts), np.int32, len(texts))
        frame['second_person'] = np.fromiter((len(SECOND_PERSON.findall(t)) for t in texts), np.int32,
                                             len(texts))
        frame['question'] = np.fromiter(('?' in t for t in texts), np.int8, len(texts))
        return frame


def extract_user(path: Path, first_row: int, counter: Counter):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        payload = [payload]
    subject = path.stem
    targets, links, texts = [], [], []
    row = first_row
    for thread_no, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        items = nodes(item)
        by_id = {}
        for node in items:
            key = node_id(node)
            if key is not None and key not in by_id:
                by_id[key] = node
        stamps, rows_of = {}, {}
        for node in items:
            dt = _to_dt(node.get('created_utc') or node.get('date'))
            stamps[id(node)] = dt
            if node.get('target') is True and dt is not None:
                rows_of[id(node)] = row
                row += 1
        others = sorted(((stamps[id(n)], n) for n in items
                         if n.get('target') is not True and stamps[id(n)] is not None),
                        key=lambda pair: pair[0])
        other_times = [t for t, _ in others]
        seen, distinct_so_far = set(), []
        for _, n in others:
            seen.add(n.get('user_id'))
            distinct_so_far.append(len(seen))
        for node in items:
            if id(node) not in rows_of:
                continue
            dt = stamps[id(node)]
            parent = by_id.get(node.get('parent_id')) if node.get('parent_id') else None
            if node.get('type') == 'submission' or node.get('parent_id') is None:
                relation = REL_SUBMISSION
            elif parent is None:
                relation = REL_MISSING
            elif parent.get('target') is True:
                relation = REL_SELF
            elif parent.get('type') == 'submission' or 'comment_id' not in parent:
                relation = REL_ROOT_OTHER
            else:
                relation = REL_COMMENT_OTHER
            n_prior = bisect.bisect_right(other_times, dt)
            targets.append({
                'source_row': rows_of[id(node)], 'user_id': subject, 'date': dt.isoformat(),
                'ts': to_seconds(dt), 'kind': node.get('type') or 'unknown', 'thread_no': thread_no,
                'relation': relation,
                'parent_row': rows_of.get(id(parent), -1) if parent is not None else -1,
                'prior_others_in_thread': n_prior,
                'prior_other_authors_in_thread': distinct_so_far[n_prior - 1] if n_prior else 0,
                'disclosure': bool(disclosure(node_text(node))),
            })
            if relation in (REL_ROOT_OTHER, REL_COMMENT_OTHER) and stamps[id(parent)] is not None:
                links.append({'role': 'parent', 'target_row': rows_of[id(node)],
                              'ts': to_seconds(stamps[id(parent)]),
                              'author': author_hash(parent.get('user_id'))})
                texts.append(node_text(parent))
        for t, node in others:
            parent = by_id.get(node.get('parent_id')) if node.get('parent_id') else None
            if parent is not None and id(parent) in rows_of:
                links.append({'role': 'reply', 'target_row': rows_of[id(parent)], 'ts': to_seconds(t),
                              'author': author_hash(node.get('user_id'))})
                texts.append(node_text(node))
    counts = counter.count(texts)
    context = pd.concat([pd.DataFrame(links), counts], axis=1) if links else counts
    return targets, context, row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / 'private' / 'cohort_v5' / 'context'))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.rglob('subject_*.json') if p.is_file())
    if len(files) != 523:
        raise SystemExit('expected 523 subject files, found %d' % len(files))
    counter = Counter(LEXICON)
    started = time.time()
    all_targets, all_context, row = [], [], 0
    for i, path in enumerate(files):
        targets, context, row = extract_user(path, row, counter)
        all_targets.extend(targets)
        all_context.append(context)
        if (i + 1) % 25 == 0:
            print('%d/%d files, %d target writings, %.0fs' % (i + 1, len(files), row, time.time() - started),
                  flush=True)
    targets = pd.DataFrame(all_targets)
    context = pd.concat(all_context, ignore_index=True)

    reference = pd.read_csv(TEXT_CSV, usecols=['user_id', 'date', 'kind'], dtype={'user_id': str})
    if len(reference) != len(targets):
        raise SystemExit('row count differs from task2_target_text.csv: %d vs %d' % (len(targets), len(reference)))
    for column in ('user_id', 'date', 'kind'):
        mismatched = int((reference[column].to_numpy() != targets[column].to_numpy()).sum())
        if mismatched:
            raise SystemExit('%d rows differ from task2_target_text.csv in %s' % (mismatched, column))
    if not (targets.source_row.to_numpy() == np.arange(len(targets))).all():
        raise SystemExit('source_row is not the csv row order')
    if int(targets.disclosure.sum()) != 610:
        raise SystemExit('disclosure writings %d, the audited count is 610' % int(targets.disclosure.sum()))

    for column in ('target_row', 'ts', 'author'):
        context[column] = context[column].astype(np.int64)
    targets.drop(columns=['date']).to_parquet(out / 'targets.parquet', index=False)
    context.to_parquet(out / 'context.parquet', index=False)
    summary = {
        'ran_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'subject_files': len(files), 'target_writings': int(len(targets)),
        'rows_match_task2_target_text_csv': True,
        'disclosure_writings': int(targets.disclosure.sum()),
        'relation_counts': {name: int((targets.relation == code).sum()) for code, name in enumerate(
            ['submission', 'top_level_on_other_thread', 'reply_to_other_comment', 'reply_to_self',
             'parent_missing'])},
        'context_rows': {role: int((context.role == role).sum()) for role in ('reply', 'parent')},
        'target_writings_with_a_reply': int(context.loc[context.role == 'reply', 'target_row'].nunique()),
        'text_written': False, 'raw_author_ids_written': False,
        'seconds': round(time.time() - started, 1),
    }
    (out / 'extract_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
