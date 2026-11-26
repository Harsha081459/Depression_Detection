"""Choose the walkthrough cases from the held-out users by a fixed rule (no text is read).

Rule, written before any text is looked at:
  TP  largest margin max_score - tau among true positives that were referred
  FP  largest margin max_score - tau among referred controls
  FN  largest margin tau - max_score among positives that were not referred
  TP-late (optional)  referred positive with the latest first flag
Ties: history length closest to the held-out median.
Key writings to paraphrase (3-6 per case): the writings with the highest Qwen level (sum over signals),
plus the flag writing. Writes private/demo/walkthrough.json and private/demo/paraphrases_TEMPLATE.json.
"""
import json

import pandas as pd

from common import PRIV, load_model


def key_writings(o, n=5):
    W = pd.DataFrame([{'i': w['i'], 'lv': sum(q['level'] for q in w['qwen'].values()),
                       'pmax': max(q['p'] for q in w['qwen'].values())} for w in o['writings']])
    upto = o['decision_row']['i']
    W = W[W.i <= upto]
    pick = W.sort_values(['lv', 'pmax'], ascending=False).head(n).i.tolist()
    if o['first_flag']['i'] and o['first_flag']['i'] not in pick:
        pick.append(o['first_flag']['i'])
    return sorted(int(i) for i in pick)


def main():
    tau = load_model()['tau']
    I = pd.DataFrame(json.loads((PRIV / 'heldout_index.json').read_text()))
    med = I.n_writings.median()
    I['margin'] = (I.max_p - tau).where(I.outcome.isin(['TP', 'FP']), tau - I.max_p)
    I['tie'] = (I.n_writings - med).abs()
    rank = lambda d: d.sort_values(['margin', 'tie'], ascending=[False, True]).iloc[0]
    tp = rank(I[I.outcome == 'TP']); fp = rank(I[I.outcome == 'FP']); fn = rank(I[I.outcome == 'FN'])
    late = I[(I.outcome == 'TP') & (I.case_id != tp.case_id)].sort_values('flag_i', ascending=False).iloc[0]
    why = {'TP': 'referred positive, clearest margin', 'FP': 'referred control (false alarm)',
           'FN': 'missed positive, largest miss', 'TP-late': 'referred positive, latest flag'}
    wt, tmpl = {}, {}
    for name, r in (('TP', tp), ('FP', fp), ('FN', fn), ('TP-late', late)):
        o = json.loads((PRIV / 'cases' / f'{r.case_id}.json').read_text())
        wt[name] = {'case_id': r.case_id, 'why': why[name], 'margin': round(float(r.margin), 3),
                    'n_writings': int(r.n_writings), 'flag_i': None if pd.isna(r.flag_i) else int(r.flag_i),
                    'colour': r.colour, 'key_writings': key_writings(o)}
        tmpl[r.case_id] = {str(i): '' for i in wt[name]['key_writings']}
    (PRIV / 'walkthrough.json').write_text(json.dumps(wt, indent=1))
    t = PRIV / 'paraphrases_TEMPLATE.json'
    t.write_text(json.dumps({'_how': 'Copy to paraphrases.json. One paraphrase (<= 25 words) per key writing: '
                             'no names, places, usernames, subreddits, dates or direct quotes. A second team '
                             'member checks each one. Then re-run build_cases.py.', **tmpl}, indent=1))
    print(json.dumps(wt, indent=1))


if __name__ == '__main__':
    main()
