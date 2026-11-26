"""Checks that run from a fresh clone (no private data, no GPU, no Qwen). Run: pytest demo/tests -q

- the frozen v11 model loads and scores the fictional example after every writing
- the referral rule is applied as stated (first writing with score >= tau and k >= t_min)
- the per-writing explanation is exact: shares + baseline = the model's log-odds at the decision writing
Embeddings here are random stand-ins (the real ones need the sentence transformer); the arithmetic is the same.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CH, SIG, load_model                                   # noqa: E402
from model_v11 import FrozenV11                                           # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def fictional_user():
    import live
    w = json.loads((HERE / 'examples' / 'synthetic_rising.json').read_text())['writings']
    x = live.prepared(w, load_model()['cuts'], use_qwen=False)
    for k in SIG:
        x[f'lg_{k}'] = np.nan; x[f'p_{k}'] = np.nan
    return CH.per_writing(x.assign(user_id='demo'), load_model()['cuts'])


def test_frozen_v11_scores_and_explains():
    fz = FrozenV11(); x = fictional_user(); n = len(x)
    rng = np.random.default_rng(0)
    E = rng.normal(size=(n, 768)).astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    mats = {'MPN': E, 'TXT': np.zeros((n, 1024), np.float32), 'EMO': rng.random((n, 28)), 'BDI': rng.random((n, 21))}
    res = fz.score_user(x, mats)
    assert len(res['p']) == n and np.all((res['p'] > 0) & (res['p'] < 1))
    hit = np.where((res['p'] >= fz.tau) & (res['k'] >= fz.t_min))[0]
    assert res['flag'] == (int(hit[0]) if len(hit) else None)
    for i in (fz.t_min - 1, n - 1):
        for e in fz.explain_row(res, i):
            for b, v in e.items():
                if isinstance(v, dict) and 'per_writing' in v:
                    assert abs(sum(v['per_writing']) + v['baseline'] - v['total']) < 1e-6
                    assert abs(v['total'] - e['logit'] + e['intercept']) < 1e-6 or 'B' in e


def test_artifacts_hold_no_text():
    for f in (HERE / 'artifacts').glob('*.json'):
        assert '"text"' not in f.read_text(), f
