"""Shared paths and loaders for the demo. Never loads the `text` column of the private writing table."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V10 = ROOT / 'research_v10'
RQ = V10 / 'real_qwen'                       # real-Qwen writing table + scores
PRIV = ROOT / 'private' / 'demo'             # user-level demo files (git-ignored)
ART = ROOT / 'demo' / 'artifacts'            # aggregate, text-free model files
SPEC = 'd14'

os.environ.setdefault('AIXPE_ROOT', str(ROOT))
for p in (str(V10), str(ROOT / 'scripts' / 'analysis')):
    if p not in sys.path:
        sys.path.insert(0, p)

import chunks as CH          # noqa: E402  research_v10/chunks.py
import seq_eval as SE        # noqa: E402  research_v10/seq_eval.py

SIG = CH.SIG
EMO = CH.EMO
NOTEXT = ['user_id', 'ts_sec', 'is_comment', 'night', 'n_tokens', 'self_tokens', 'absolutist', 'distress',
          'polarity', 'profanity', 'disclosure'] + [f'emo_{e}' for e in EMO] + [f'q_{k}' for k in SIG]


def load_split():
    rows = json.loads((ROOT / 'private' / 'cohort_v1' / 'split.json').read_text())['rows']
    return pd.DataFrame({'user_id': [str(r['user_id']) for r in rows], 'label': [int(r['label']) for r in rows],
                         'pilot': [bool(r['pilot_exposed']) for r in rows], 'split': [r['split'] for r in rows]})


def load_writings(user_ids=None):
    """Per-writing features + real Qwen P(Yes) and logits, no text. Rows keep file order."""
    w = pd.read_parquet(RQ / 'writings_PRIVATE.parquet', columns=NOTEXT)
    s = pd.read_parquet(RQ / 'student_scores_PRIVATE.parquet')
    assert len(w) == len(s) and (w.user_id.to_numpy() == s.user_id.to_numpy()).all() \
        and (w.ts_sec.to_numpy() == s.ts_sec.to_numpy()).all(), 'score table not aligned with writings'
    for k in SIG:
        p = np.clip(s[f's_{k}'].to_numpy(), 1e-9, 1 - 1e-9)
        w[f'p_{k}'] = p
        w[f'lg_{k}'] = np.log(p / (1 - p))
    w['user_id'] = w.user_id.astype(str)
    if user_ids is not None:
        w = w[w.user_id.isin(set(user_ids))]
    return w


def prepare(w, cuts):
    """Masking + per-writing derived features exactly as chunks.main does it."""
    w = w[~w.disclosure].sort_values(['user_id', 'ts_sec'], kind='mergesort').reset_index(drop=True)
    return CH.per_writing(w, cuts)


def load_model():
    return json.loads((ART / 'frozen_model.json').read_text())
