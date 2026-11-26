"""v11 demo checks. Run: pytest demo/tests/test_v11.py -q
- the demo scorer (model_v11.FrozenV11) gives the same held-out scores as the frozen fit (freeze_v11.py, lab run)
- per-writing contributions + baseline = the member's log-odds, exactly
- referral decisions in the case files match the frozen fit's held-out decisions
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not (Path(__file__).resolve().parents[2] / 'private' / 'cohort_v1' / 'split.json').exists():
    pytest.skip('needs the private eRisk data, which is not in the repository', allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRIV, ROOT, load_split, load_writings, prepare        # noqa: E402
from model_v11 import FrozenV11                                         # noqa: E402

sys.path.insert(0, str(ROOT / 'research_v11'))
from v11 import load_matrix                                             # noqa: E402

LAB = PRIV.parent / 'v11' / 'final_from_lab'


@pytest.fixture(scope='module')
def ctx():
    fz = FrozenV11(); users = load_split(); ho = users[users.split != 'train']
    pick = list(np.random.default_rng(5).choice(ho.user_id.to_numpy(), 8, replace=False))
    cuts = json.load(open(ROOT / 'research_v10' / 'real_qwen' / 'chunk_meta.json'))['cuts']
    w = load_writings(set(pick)); w['orig'] = w.index.to_numpy(); w = prepare(w, cuts)
    return fz, pick, w, {n: load_matrix(n, w.orig.to_numpy()) for n in ('MPN', 'EMO', 'BDI')}


def test_scores_match_frozen_fit(ctx):
    fz, pick, w, mats = ctx
    ref = pd.read_parquet(LAB / 'scores_heldout_PRIVATE.parquet'); ref['user_id'] = ref.user_id.astype(str)
    for u in pick:
        m = (w.user_id == u).to_numpy()
        if not m.any():
            continue
        res = fz.score_user(w[m], {n: M[m] for n, M in mats.items()})
        r = ref[ref.user_id == u].sort_values('k').p.to_numpy()
        assert len(r) == len(res['p']) and np.abs(res['p'] - r).max() < 1e-6


def test_contributions_add_up(ctx):
    fz, pick, w, mats = ctx
    u = pick[0]; m = (w.user_id == u).to_numpy()
    res = fz.score_user(w[m], {n: M[m] for n, M in mats.items()})
    for i in (4, len(res['p']) // 2, len(res['p']) - 1):
        for e, r in zip(fz.explain_row(res, i), res['members']):
            for b, v in e.items():
                if isinstance(v, dict) and 'per_writing' in v:
                    assert abs(sum(v['per_writing']) + v['baseline'] - v['total']) < 1e-6


def test_case_decisions_match_frozen_fit():
    dec = pd.read_parquet(LAB / 'dec_heldout_prev_adj_PRIVATE.parquet'); dec['user_id'] = dec.user_id.astype(str)
    ids = json.loads((PRIV / 'case_ids_PRIVATE.json').read_text())
    for _, r in dec.iterrows():
        v = json.loads((PRIV / 'cases' / f"{ids[r.user_id]}.json").read_text())['v11']
        assert (v['status'] == 'refer') == bool(r.flagged)
        assert v['status'] == 'refer' or v['max_score'] is None or v['max_score'] < v['tau']
