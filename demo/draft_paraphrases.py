"""Draft paraphrases of the walkthrough cases' key writings with the project's own Qwen (local / lab Ollama).

Raw text is read here and sent only to the Ollama server on 127.0.0.1:11434 (this laptop, or the lab
through demo/lab_tunnel.sh) - the same path the cohort scoring used. Nothing is printed except counts.

For each case in private/demo/walkthrough_v11.json: the writings that moved the score most (and the
referral writing) get one paraphrase: third person, <= 20 words, no names, places, ages, dates, numbers,
usernames, subreddits, no quotes. Automatic checks (no text printed) flag:
  digits / handles / links, capitalised words that may be names, any 5-word run copied from the original,
  more than 30 words.
Writes private/demo/paraphrases_DRAFT.json and private/demo/paraphrase_checks.json (flags only).
A team member reviews the drafts, then copies them to private/demo/paraphrases.json.
"""
import json
import re
import urllib.request

import pandas as pd

from common import PRIV, RQ

URL = 'http://127.0.0.1:11434/api/chat'
MODEL = 'qwen2.5:7b-instruct'
SYSTEM = ('You write short, neutral paraphrases of social-media writings for a research report on '
          'mental-health screening. Rules: one sentence, at most 20 words, third person starting with '
          '"The writer". Describe the gist and the feeling. Never quote the writing. Never include names of '
          'people, places, organisations, games or products, usernames, subreddit names, ages, dates or any '
          'numbers. If the writing is not about the writer\'s own life, say what it is about in general terms.')
OK_CAPS = {'The', 'I', 'A', 'An', 'They', 'He', 'She', 'It', 'This', 'Their', 'His', 'Her'}


def ask(text):
    body = json.dumps({'model': MODEL, 'stream': False, 'options': {'temperature': 0, 'seed': 42, 'num_predict': 70},
                       'messages': [{'role': 'system', 'content': SYSTEM},
                                    {'role': 'user', 'content': 'Writing:\n"""\n' + text[:1500] + '\n"""\nParaphrase:'}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as r:
        out = json.load(r)['message']['content'].strip().strip('"').strip()
    return re.sub(r'\s+', ' ', out.split('\n')[0])


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def checks(para, original):
    f = []
    if re.search(r'\d|@|https?:|www\.|\br/|\bu/', para):
        f.append('digits/handle/link')
    caps = [w for i, w in enumerate(re.findall(r"[A-Za-z']+", para)) if i > 0 and w[0].isupper() and w not in OK_CAPS]
    if caps:
        f.append(f'{len(caps)} capitalised word(s): check for names')
    pw, ow = words(para), ' '.join(words(original))
    if any(' '.join(pw[i:i + 5]) in ow for i in range(max(len(pw) - 4, 0))):
        f.append('5+ words copied from the original')
    if len(pw) > 30:
        f.append('longer than 30 words')
    return f


def main():
    wt = json.loads((PRIV / 'walkthrough_v11.json').read_text())
    uid = {v: k for k, v in json.loads((PRIV / 'case_ids_PRIVATE.json').read_text()).items()}
    w = pd.read_parquet(RQ / 'writings_PRIVATE.parquet', columns=['user_id', 'ts_sec', 'disclosure', 'text'])
    w['user_id'] = w.user_id.astype(str)
    drafts, flags, n = {}, {}, 0
    for name, d in wt.items():
        cid = d['case_id']; case = json.loads((PRIV / 'cases' / f'{cid}.json').read_text()); v = case['v11']
        keys = [t['i'] for t in v['top_writings'][:4]]
        if v['first_flag']['i'] and v['first_flag']['i'] not in keys:
            keys.append(v['first_flag']['i'])
        x = w[(w.user_id == uid[cid]) & ~w.disclosure].sort_values('ts_sec', kind='mergesort').reset_index(drop=True)
        assert len(x) == case['n_writings'], (cid, len(x), case['n_writings'])
        drafts[cid], flags[cid] = {}, {}
        for i in sorted(keys):
            original = x.text.iloc[i - 1] or ''
            p = ask(original); f = checks(p, original); tries = 1
            while f and tries < 3:      # stricter retry: generic words only
                p = ask(original + '\n\n(Rewrite in generic words only: no proper nouns, no capitalised names, '
                        'no numbers of any kind.)'); f = checks(p, original); tries += 1
            if f:
                p = None                # still flagged: dropped, the demo shows "[text hidden]"
            drafts[cid][str(i)] = p; flags[cid][str(i)] = f; n += 1
    (PRIV / 'paraphrases_DRAFT.json').write_text(json.dumps(drafts, indent=1, ensure_ascii=False), encoding='utf-8')
    (PRIV / 'paraphrase_checks.json').write_text(json.dumps(flags, indent=1))
    kept = sum(p is not None for d in drafts.values() for p in d.values())
    dropped = {c: [i for i, p in d.items() if p is None] for c, d in drafts.items()}
    print(json.dumps({'writings': n, 'paraphrases_kept': kept, 'dropped_after_3_tries': {c: v for c, v in dropped.items() if v}}, indent=1))


if __name__ == '__main__':
    main()
