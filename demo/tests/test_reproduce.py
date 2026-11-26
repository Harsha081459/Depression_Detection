"""Checks that the demo pipeline is the frozen research pipeline. Run: pytest demo/tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not (Path(__file__).resolve().parents[2] / 'private' / 'cohort_v1' / 'split.json').exists():
    pytest.skip('needs the private eRisk data, which is not in the repository', allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ART, PRIV, RQ, load_model, load_split, load_writings, prepare   # noqa: E402
import pipeline as PL                                                              # noqa: E402

RNG = np.random.default_rng(7)


@pytest.fixture(scope='module')
def ctx():
    m = load_model(); users = load_split()
    return {'m': m, 'fz': PL.Frozen(m), 'users': users}


def _prepared(uids, m):
    return prepare(load_writings(uids), m['cuts'])


def test_dev_rows_equal_research_tables(ctx):
    ref = pd.read_parquet(RQ / 'chunksW_d14_PRIVATE.parquet')
    dev = ctx['users'].query("split == 'train'").user_id.to_numpy()
    pick = list(RNG.choice(dev, 5, replace=False))
    w = _prepared(pick, ctx['m'])
    for u in pick:
        mine = PL.rows_frame(w[w.user_id == u], every_writing=True)
        r = ref[ref.user_id == u].reset_index(drop=True)
        cols = [c for c in mine.columns if c.startswith(('cur_', 'prv_', 'pre_'))]
        assert len(mine) == len(r)
        assert np.nanmax(np.abs(mine[cols].to_numpy(float) - r[cols].to_numpy(float))) < 1e-9


def test_heldout_scores_equal_lockbox(ctx):
    ref = pd.read_parquet(PRIV / 'ref' / 'oof_LB_W_B_PRIVATE.parquet')
    ho = ctx['users'].query("split != 'train'").user_id.to_numpy()
    pick = list(RNG.choice(ho, 8, replace=False))
    w = _prepared(pick, ctx['m'])
    for u in pick:
        x = w[w.user_id == u]
        if len(x) == 0:
            continue
        o = PL.run_user(x, ctx['fz'])
        mine = np.array([s['p'] for s in o['scores']])
        r = ref[ref.user_id == u].sort_values('k').p.to_numpy()
        assert np.abs(mine - r).max() < 1e-5          # output rounded to 5 decimals


def test_heldout_flags_equal_lockbox(ctx):
    dec = pd.read_parquet(PRIV / 'ref' / 'dec_LB_W_B_PRIVATE.parquet').query("objective == 'f_latency'")
    cases = PRIV / 'cases'
    if not cases.exists():
        pytest.skip('run build_cases.py first')
    idmap = json.loads((PRIV / 'case_ids_PRIVATE.json').read_text())
    for _, r in dec.iterrows():
        o = json.loads((cases / f"{idmap[r.user_id]}.json").read_text())
        assert (o['status'] == 'refer') == bool(r.flagged)


def test_no_text_in_exports():
    for f in list(ART.glob('*.json')) + list((PRIV / 'cases').glob('*.json')):
        s = f.read_text()
        assert '"text"' not in s, f
