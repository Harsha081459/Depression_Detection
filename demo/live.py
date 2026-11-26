"""Score new writings live: rule features + NRC here, the five Qwen signals through a local Ollama.

Input: [{'ts': '2026-01-03T22:15:00Z', 'text': '...', 'is_comment': 0}, ...]
Same prompts (frozen v1, hash 238ac1a3...) and model digest as the cohort run. Differences:
  - the lab-only prompt-cache erase step is skipped (it needs the isolated Linux runner);
  - scores are cached in private/demo/live_cache.json by text hash only (no text stored).
Text never leaves this machine: Ollama is called on 127.0.0.1 with proxies disabled.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

from common import EMO, PRIV, ROOT, SIG, prepare

PROMPTS = ROOT / 'private' / 'cohort_v1' / 'prompts.v1.json'
LEX = ROOT / 'data' / 'raw' / 'lexicons' / 'nrc' / 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
CACHE = PRIV / 'live_cache.json'
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")
_lex = None


def nrc_available():
    return LEX.exists()


def nrc(text):
    """NRC word-emotion shares. The lexicon has its own licence and is not in the repository: without it
    these features are 0 (they are shown for explanation only; the final text model does not use them)."""
    global _lex
    if _lex is None:
        _lex = {}
        if LEX.exists():
            for line in open(LEX, encoding='utf-8'):
                p = line.rstrip('\n').split('\t')
                if len(p) == 3 and p[2] == '1':
                    _lex.setdefault(p[0], set()).add(p[1])
    c = dict.fromkeys(EMO, 0); toks = TOK.findall(text)
    for w in toks:
        for e in _lex.get(w, ()):
            c[e] += 1
    n = max(len(toks), 1)
    return {f'emo_{e}': c[e] / n for e in EMO}


def rule_features(writings):
    """Same code path as research_v10/real_qwen/base.py (fip._prepare + NRC). Returns frame incl. text."""
    import final_indirect_pipeline as fip
    raw = pd.DataFrame({'user_id': 'live', 'date': [w['ts'] for w in writings], 'title': '',
                        'body': [w['text'] for w in writings],
                        'kind': ['comment' if w.get('is_comment') else 'post' for w in writings]})
    d = fip._prepare(raw)
    d['ts_sec'] = d['ts'].to_numpy().astype('datetime64[s]').astype(np.int64)
    d['is_comment'] = (d['kind'] == 'comment').astype(int)
    E = pd.DataFrame([nrc(t) for t in d['text']], index=d.index)
    return pd.concat([d, E], axis=1)


def ollama_status():
    import llm_signal_scorer as L
    try:
        cfg = L.load_prompts(PROMPTS)
        info = L.model_info({**cfg, 'cache_policy': None})
        return True, {**info, 'prompt_hash': L.canonical_hash(cfg)}
    except Exception as e:          # SafeError carries no text
        return False, {'error': str(e) or type(e).__name__}


def qwen_scores(texts, progress=None):
    import llm_signal_scorer as L
    cfg = {**L.load_prompts(PROMPTS), 'cache_policy': None}
    ph = L.canonical_hash(L.load_prompts(PROMPTS))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out = []; n = len(texts) * len(SIG); done = 0
    for t in texts:
        row = {}
        for k in SIG:
            key = f"{ph[:16]}:{k}:{hashlib.sha256(t[:cfg['max_chars']].encode('utf-8')).hexdigest()}"
            if key not in cache:
                cache[key] = L.score_one(t, k, cfg)['score']
            row[f'q_{k}'] = cache[key]; done += 1
            if progress:
                progress(done / n)
        out.append(row)
    PRIV.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return pd.DataFrame(out)


def prepared(writings, cuts, use_qwen=True, progress=None):
    d = rule_features(writings)
    if use_qwen:
        q = qwen_scores(d['text'].tolist(), progress)
        q.index = d.index
        d = pd.concat([d, q], axis=1)
        for k in SIG:
            p = np.clip(d[f'q_{k}'].to_numpy(float), 1e-9, 1 - 1e-9)
            d[f'p_{k}'] = p; d[f'lg_{k}'] = np.log(p / (1 - p))
    d = d.drop(columns=['text', 'title', 'body'], errors='ignore')
    return prepare(d, cuts) if use_qwen else d.sort_values('ts_sec').reset_index(drop=True)
