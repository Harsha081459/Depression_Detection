"""Lab-only: score user-level LLM signals from a digest of each user's writings.

Run on the lab (redhatacademy5), against the isolated Ollama on 127.0.0.1:12466.
Python 3.6+, needs pandas + pyarrow. Never prints writing text.

    /home/user5/bio/bin/python scripts/lab_user_digest_scorer.py \
        --posts  private/cohort_v1/eligible_posts.parquet \
        --scores private/cohort_v1/scores/part-00000000.parquet \
        --origins private/cohort_v1/origins.parquet \
        --prompts scripts/prompts_user_digest_v1.json \
        --day 28 --out private/cohort_v6/user_digest_day28.sqlite

Why: v1-v5 asked the LLM about one post at a time and averaged. A person reading
for low mood reads a history. This asks the same frozen model (qwen2.5:7b-instruct,
digest 845dbda0...) about a short digest of each user's own writings up to the
decision day. The digest is chosen without labels: the K writings with the
highest cached post-level signal, in time order, each cut to M characters, with
clinical words, URLs and medical words removed (same strict rule as v6).

Score = P(Yes) on the first answer token, Yes/yes/YES combined, exactly as v1.
One request at a time. Resumable: finished (user, signal, day) rows are skipped.
"""
import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

SIGNALS_V1 = ('self_blame', 'helplessness', 'isolation', 'family_strain', 'emptiness')

CLIN_WORDS = ("depression depressed depressing depress depressive antidepressant anxiety anxious suicide "
              "suicidal therapy therapist psychiatrist psychiatric psychiatry psychologist psychology "
              "medication medicine diagnosis diagnosed diagnose disorder mental illness bipolar insomnia "
              "counseling counselor prescription pill pills hospital doctor symptom symptoms disease sick "
              "sickness").split()
CLIN_STEMS = ('depress', 'anxi', 'therap', 'psychiatr', 'psycholog', 'medicat', 'meds', 'antidepress',
              'ssri', 'snri', 'zoloft', 'prozac', 'lexapro', 'sertraline', 'fluoxetine', 'escitalopram',
              'citalopram', 'wellbutrin', 'bupropion', 'venlafaxine', 'effexor', 'diagnos', 'bipolar',
              'ptsd', 'adhd', 'ocd', 'suicid', 'selfharm', 'counsel', 'mdd')
CLIN_RE = re.compile(r"\b(?:(?:%s)\b|(?:%s)\w*)" % ('|'.join(map(re.escape, sorted(CLIN_WORDS))),
                                                   '|'.join(map(re.escape, CLIN_STEMS))), re.I)
URL_RE = re.compile(r'https?://\S+|www\.\S+|\S+\.(?:com|org|net|it|png|jpg|jpeg|gif|webp)\b'
                    r'|&amp;#x200b;|x200b|&amp;|&gt;|&lt;', re.I)
MED_RE = re.compile(r'\b(?:dose\w*|dr|mg|doctor\w*|pill\w*|prescri\w*|hospital\w*|clinic\w*|'
                    r'psych\w*|redd|preview|webp|png|jpg|gif|imgur|youtube|youtu)\b', re.I)


def strict_clean(text):
    return re.sub(r'\s+', ' ', MED_RE.sub(' ', URL_RE.sub(' ', CLIN_RE.sub(' ', str(text))))).strip()


def digest_for(group, k, m):
    """Top-k writings by highest cached post-level signal, then time order."""
    group = group.assign(rank_score=group[list(SIGNALS_V1)].max(axis=1))
    top = group.nlargest(k, 'rank_score').sort_values('date')
    parts = []
    for i, text in enumerate(top.text.tolist(), 1):
        clean = strict_clean(text)[:m]
        if clean:
            parts.append('[%d] %s' % (i, clean))
    return '\n\n'.join(parts)


def request(base, payload):
    req = urllib.request.Request(base + '/api/chat', data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=600) as r:
        return json.load(r)


def logsumexp(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def parse(response):
    top = response['logprobs'][0]['top_logprobs']
    vals = {'yes': [], 'no': []}
    finite = []
    for item in top:
        lp = float(item['logprob'])
        finite.append(lp)
        t = item['token'].strip().lower()
        if t in vals:
            vals[t].append(lp)
    if not vals['yes'] and not vals['no']:
        raise ValueError('neither_yes_nor_no')
    low = min(finite) - 1
    yes = logsumexp(vals['yes']) if vals['yes'] else low
    no = logsumexp(vals['no']) if vals['no'] else low
    return math.exp(yes - logsumexp([yes, no])), int(not vals['yes'] or not vals['no'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posts', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--origins', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--day', type=int, default=28)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='http://127.0.0.1:12466')
    ap.add_argument('--limit-users', type=int, default=0, help='smoke test on the first N users')
    a = ap.parse_args()

    config = json.loads(Path(a.prompts).read_text(encoding='utf-8'))
    prompt_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    posts = pd.read_parquet(a.posts)
    scores = pd.read_parquet(a.scores)
    origins = pd.read_parquet(a.origins).set_index('user_id').origin
    data = posts.merge(scores.drop(columns='date'), on=['user_id', 'post_id'])
    data['date'] = pd.to_datetime(data.date, utc=True)
    origin = pd.to_datetime(data.user_id.map(origins), utc=True)
    data = data[(data.date - origin).dt.total_seconds() <= a.day * 86400]

    db = sqlite3.connect(a.out)
    db.execute('CREATE TABLE IF NOT EXISTS s (user_id TEXT, signal TEXT, day INT, prompt_hash TEXT, '
               'digest_hash TEXT, score REAL, missing INT, n_writings INT, PRIMARY KEY (user_id, signal, day, prompt_hash))')
    done = {(r[0], r[1]) for r in db.execute('SELECT user_id, signal FROM s WHERE day=? AND prompt_hash=?',
                                            (a.day, prompt_hash))}
    users = sorted(data.user_id.unique())
    if a.limit_users:
        users = users[:a.limit_users]
    started, calls = time.time(), 0
    for n, (user, group) in enumerate(data[data.user_id.isin(users)].groupby('user_id'), 1):
        digest = digest_for(group, config['k_writings'], config['max_chars_per_writing'])
        dhash = hashlib.sha256(digest.encode()).hexdigest()
        for signal, q in config['signals'].items():
            if (user, signal) in done:
                continue
            content = ('Writings by one person, in time order:\n"""\n%s\n"""\n\nQuestion: Across these writings, '
                       'does the writer %s?\nAnswer No if %s.\nAnswer Yes or No.') % (digest, q['yes'], q['no'])
            payload = {'model': config['model'], 'stream': False, 'logprobs': True, 'top_logprobs': 20,
                       'keep_alive': '30m', 'options': config['options'],
                       'messages': [{'role': 'system', 'content': config['system']},
                                    {'role': 'user', 'content': content}]}
            score, missing = parse(request(a.base, payload))
            db.execute('INSERT OR REPLACE INTO s VALUES (?,?,?,?,?,?,?,?)',
                       (user, signal, a.day, prompt_hash, dhash, score, missing, int(len(group))))
            calls += 1
        db.commit()
        if n % 25 == 0:
            print('users %d/%d, calls %d, %.1f s/call' % (n, len(users), calls,
                                                          (time.time() - started) / max(1, calls)), flush=True)
    print('done: %d calls, prompt hash %s' % (calls, prompt_hash))


if __name__ == '__main__':
    sys.exit(main())
