"""Precompute the demo JSON for the 105 held-out users (frozen model, scored once already).

Writes, all under private/demo/ (git-ignored):
  cases/HO-xxx.json            per-user demo record (no text; paraphrases only if present)
  case_ids_PRIVATE.json        real user_id -> HO-xxx (never shown in the demo)
  heldout_index.json           one line per case: outcome, flag writing, colour (used by the app's picker)
and demo/artifacts/heldout_scorecard_agreement.json (aggregates only).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import ART, PRIV, load_model, load_split, load_writings, prepare
import pipeline as PL
from explain import explain
from scorecard import Scorecard, agreement_note


def paraphrases():
    f = PRIV / 'paraphrases.json'
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}


def build_one(x, fz, sc, case_id, source, label=None, para=None):
    o = PL.run_user(x, fz, case_id, source, label, para)
    if 'error' in o:
        return o
    _, row = o['_decision_inputs']
    card = sc.card(row)
    card['agreement'] = agreement_note(card['colour'], o['status'] == 'refer')
    o['scorecard'] = card
    o['explanation'] = explain(fz, o, card)
    return PL.strip(o)


def outcome(label, refer):
    return {(1, True): 'TP', (0, True): 'FP', (1, False): 'FN', (0, False): 'TN'}[(int(label), bool(refer))]


def main():
    m = load_model(); fz = PL.Frozen(m); sc = Scorecard()
    users = load_split(); ho = users[users.split != 'train'].reset_index(drop=True)
    w = prepare(load_writings(ho.user_id), m['cuts'])
    rng = np.random.default_rng(2026)
    ids = {u: f'HO-{i + 1:03d}' for i, u in enumerate(rng.permutation(ho.user_id.to_numpy()))}
    para = paraphrases()
    (PRIV / 'cases').mkdir(parents=True, exist_ok=True)
    idx = []
    for _, u in ho.iterrows():
        cid = ids[u.user_id]
        o = build_one(w[w.user_id == u.user_id], fz, sc, cid, 'heldout', int(u.label), para.get(cid))
        (PRIV / 'cases' / f'{cid}.json').write_text(json.dumps(o))
        if 'error' in o:
            idx.append({'case_id': cid, 'outcome': 'no data', 'label': int(u.label)}); continue
        idx.append({'case_id': cid, 'label': int(u.label), 'outcome': outcome(u.label, o['status'] == 'refer'),
                    'n_writings': o['n_writings'], 'span_days': o['span_days'], 'flag_i': o['first_flag']['i'],
                    'max_p': o['decision_row']['max_p'], 'last_p': o['scores'][-1]['p'],
                    'colour': o['scorecard']['colour'], 'composite': o['scorecard']['composite'],
                    'pilot': bool(u.pilot)})
    (PRIV / 'case_ids_PRIVATE.json').write_text(json.dumps(ids, indent=1))
    I = pd.DataFrame(sorted(idx, key=lambda d: d['case_id']))
    (PRIV / 'heldout_index.json').write_text(I.to_json(orient='records', indent=1))

    J = I[I.outcome != 'no data']
    refer = J.outcome.isin(['TP', 'FP']).astype(int)
    agg = {'n': int(len(I)), 'outcomes': I.outcome.value_counts().to_dict(),
           'colour_rate': J.colour.value_counts().to_dict(),
           'crosstab_refer_x_colour': pd.crosstab(refer, J.colour).to_dict('index'),
           'positive_rate_by_colour': J.groupby('colour').label.mean().round(3).to_dict(),
           'kappa_red_vs_refer': float(cohen_kappa_score(refer, (J.colour == 'red').astype(int))),
           'kappa_redamber_vs_refer': float(cohen_kappa_score(refer, (J.colour != 'green').astype(int))),
           'note': 'held-out was used once for the frozen model; this agreement table is a descriptive second look'}
    (ART / 'heldout_scorecard_agreement.json').write_text(json.dumps(agg, indent=1, default=str))
    print(json.dumps(agg, indent=1, default=str))


if __name__ == '__main__':
    main()
