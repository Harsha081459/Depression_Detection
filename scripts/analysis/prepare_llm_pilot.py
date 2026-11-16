"""Build a private pilot pool locally; stdout contains aggregate counts only.

Reuses the existing disclosure/lexical code. Text is exported only to a
gitignored private directory and may be uploaded only to the lab machines.
This is candidate sampling, not full-cohort model scoring or evaluation.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
import sys

import pandas as pd

import final_indirect_pipeline as old
from llm_signal_scorer import SIGNALS, canonical_hash, write_json


def empty_post(text):
    return not text.strip() or text.strip().lower() in ('[removed]', '[deleted]', 'none')


def audit(writings, labels):
    if labels.user_id.duplicated().any() or set(labels.label) != {0, 1}:
        raise ValueError('invalid_labels')
    if (len(labels), int(labels.label.sum()), len(writings), writings.user_id.nunique()) != (523, 91, 155284, 523):
        raise ValueError('cohort_count_mismatch')
    if set(writings.user_id) != set(labels.user_id):
        raise ValueError('cohort_user_mismatch')
    data = writings.copy()
    data['text'] = (data.title.fillna('') + ' ' + data.body.fillna('')).str.strip()
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='raise')
    data['disclosure'] = data.text.map(old.disclosure)
    merged = data[['user_id', 'disclosure']].merge(labels, validate='many_to_one')
    by_label = merged.groupby('label').disclosure.sum().astype(int).to_dict()
    if by_label != {0: 39, 1: 571}:
        raise ValueError('disclosure_count_mismatch')
    report = {'users': 523, 'positive': 91, 'control': 432, 'target_writings': 155284,
              'disclosure_writings': int(data.disclosure.sum()),
              'disclosure_positive': by_label[1], 'disclosure_control': by_label[0],
              'empty_removed_deleted': int(data.text.map(empty_post).sum())}
    data['within_12w'] = data.date <= data.groupby('user_id').date.transform('min') + pd.Timedelta(days=84)
    report['prefix_12w_writings'] = int(data.within_12w.sum())
    return data, report


def split_users(labels):
    # Lock user-disjoint pilot halves before any model scores or human labels.
    splits = {}
    for label, group in labels.groupby('label'):
        users = sorted(group.user_id, key=lambda x: canonical_hash(['pilot_split_seed42', x]))
        splits.update({uid: 'development' if i % 2 == 0 else 'heldout' for i, uid in enumerate(users)})
    return splits


def pool(data, labels):
    data = data.loc[data.within_12w & ~data.disclosure & ~data.text.map(empty_post)].copy()
    data['source_row'] = data.index
    data = data.merge(labels[['user_id', 'label']], validate='many_to_one')
    data['split'] = data.user_id.map(split_users(labels))
    data['post_id'] = [canonical_hash([uid, date.isoformat(), int(i), text])
                       for uid, date, i, text in zip(data.user_id, data.date, data.source_row, data.text)]
    data['parsed'] = data.text.map(old.flags)
    data['domains'] = data.text.map(lambda x: [domain for domain, words in {
        'game': old.HOBBY, 'sport': {'football', 'soccer', 'basketball', 'cricket', 'baseball', 'tennis', 'hockey'},
        'news': {'election', 'president', 'government', 'parliament', 'headline', 'news', 'politics'},
    }.items() if set(old.TOKEN_RE.findall(old._clean(x))) & words])
    chosen = {}
    def take(indices, signals):
        for index in indices:
            chosen.setdefault(index, set()).update(signals)
    # Oversample lexical candidates solely to find enough model Yes examples.
    # Final blind human sample uses model >=.5 vs <.5, not these flags.
    for split in ('development', 'heldout'):
        for label in (0, 1):
            subset = data.loc[data.split.eq(split) & data.label.eq(label)]
            for signal in SIGNALS:
                lexical = 'helpless' if signal == 'helplessness' else signal
                hit = subset.parsed.map(lambda item: bool(item[lexical]))
                for polarity, limit in ((True, 45), (False, 15)):
                    candidates = subset.loc[hit.eq(polarity)].copy()
                    candidates['rank'] = candidates.post_id.map(lambda x: canonical_hash([signal, x]))
                    candidates = candidates.sort_values('rank').groupby('user_id', sort=False).head(2)
                    take(candidates.head(limit).index, [signal])
            for domain in ('game', 'sport', 'news'):
                candidates = subset.loc[subset.domains.map(lambda x: domain in x)]
                candidates = candidates.sort_values('post_id').groupby('user_id', sort=False).head(2)
                take(candidates.head(8).index, SIGNALS)
    selected = data.loc[sorted(chosen)].sort_values(['user_id', 'date', 'post_id'])
    rows = []
    for index, item in zip(selected.index, selected.to_dict('records')):
        rows.append({'post_id': item['post_id'], 'user_id': item['user_id'], 'date': item['date'].isoformat(),
                     'text': item['text'], 'split': item['split'], 'label': int(item['label']),
                     'domains': item['domains'], 'requested_signals': [s for s in SIGNALS if s in chosen[index]],
                     'lexical': {signal: int(item['parsed']['helpless' if signal == 'helplessness' else signal]) for signal in SIGNALS}})
    return rows


def expand_pool(data, labels, existing):
    """Broaden user coverage for scarce Yes strata; never loosen annotation caps."""
    data = data.loc[data.within_12w & ~data.disclosure & ~data.text.map(empty_post)].copy()
    data['source_row'] = data.index
    data = data.merge(labels[['user_id', 'label']], validate='many_to_one')
    data['split'] = data.user_id.map(split_users(labels))
    data['post_id'] = [canonical_hash([uid, date.isoformat(), int(i), text])
                       for uid, date, i, text in zip(data.user_id, data.date, data.source_row, data.text)]
    old_posts = {r['post_id'] for r in existing}
    old_counts = Counter(r['user_id'] for r in existing)
    data = data.loc[~data.post_id.isin(old_posts)].copy()
    patterns = {
        'self_blame': r'\b(worthless|useless|burden|guilty|guilt|ashamed|shame|stupid|failure)\b|not good enough|hate myself',
        'helplessness': r'\b(hopeless|helpless|trapped|powerless|pointless)\b|cannot cope|can.t cope|need help|do this anymore|nothing.*chang',
        'isolation': r'\b(lonely|isolated|unwanted|rejected|abandoned)\b|no one|nobody|no friends|feel alone',
        'family_strain': r'\b(mother|father|parents|family|partner|wife|husband|friend|friends)\b',
        'emptiness': r'\b(numb|empty|exhausted|crying|insomnia|drained)\b|feel anything|no energy|can.t sleep|cannot sleep',
    }
    data['parsed'] = data.text.map(old.flags)
    added, user_counts = {}, Counter()
    for split in ('development', 'heldout'):
        for signal in SIGNALS:
            lexical = 'helpless' if signal == 'helplessness' else signal
            candidates = data.loc[data.split.eq(split)].copy()
            candidates['strong'] = candidates.text.map(lambda text: bool(re.search(patterns[signal], text, flags=re.I)))
            candidates['lexical_hit'] = candidates.parsed.map(lambda x: bool(x[lexical]))
            candidates = candidates.loc[candidates.strong | candidates.lexical_hit]
            # Extend coverage to people underrepresented in the initial pool.
            candidates['prior_posts'] = candidates.user_id.map(old_counts)
            candidates['rank'] = candidates.post_id.map(lambda x: canonical_hash([signal, x]))
            candidates = candidates.sort_values(['strong', 'prior_posts', 'rank'], ascending=[False, True, True])
            used_users = set()
            taken = 0
            for item in candidates.to_dict('records'):
                uid, pid = item['user_id'], item['post_id']
                if uid in used_users or (pid not in added and user_counts[uid] >= 3):
                    continue
                if pid not in added:
                    parsed = item['parsed']
                    added[pid] = {'post_id': pid, 'user_id': uid, 'date': item['date'].isoformat(),
                                  'text': item['text'], 'split': split, 'label': int(item['label']),
                                  'domains': [], 'requested_signals': [],
                                  'lexical': {s: int(parsed['helpless' if s == 'helplessness' else s]) for s in SIGNALS}}
                    user_counts[uid] += 1
                added[pid]['requested_signals'].append(signal)
                used_users.add(uid)
                taken += 1
                if taken >= 100:
                    break
    return sorted(existing + list(added.values()), key=lambda r: (r['user_id'], r['date'], r['post_id']))


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=old.ROOT / 'data/intake/llm_signals')
    ap.add_argument('--expand-existing', type=Path)
    args = ap.parse_args()
    try:
        writings = pd.read_csv(old.TEXT_PATH, usecols=['user_id', 'date', 'title', 'body'])
        labels = pd.read_csv(old.LABEL_PATH, usecols=['user_id', 'label'])
        data, report = audit(writings, labels)
        rows = pool(data, labels)
        if args.expand_existing:
            existing = [json.loads(line) for line in args.expand_existing.read_text(encoding='utf-8').splitlines()]
            rows = expand_pool(data, labels, existing)
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / 'pilot_pool.jsonl'
        with destination.open('w', encoding='utf-8') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        report['pilot_pool_posts'] = len(rows)
        report['pilot_pool_users'] = len({x['user_id'] for x in rows})
        report['pilot_pool_calls'] = sum(len(x['requested_signals']) for x in rows)
        report['pilot_pool_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        report['pilot_sampling'] = '12-week non-disclosure posts; lexical/domain enrichment; user-disjoint development and heldout'
        write_json(args.output / 'data_audit.json', report)
        print(json.dumps(report))
        return 0
    except Exception:
        print(json.dumps({'error': 'pilot_preparation_failed_no_text_printed'}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
