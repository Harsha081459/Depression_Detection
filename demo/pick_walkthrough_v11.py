"""Walkthrough cases for the v11 model, by the same fixed rule as v10 (no text is read):
  TP  largest margin max_score - tau among referred positives
  FP  largest margin among referred controls (if any)
  FN  largest margin tau - max_score among missed positives (if any)
  TP-late  referred positive with the latest referral
  CHANGED  a held-out user whose outcome changed from v10 to v11 (first by case id), if any
Writes private/demo/walkthrough_v11.json and a paraphrase template for any new case.
"""
import json

import pandas as pd

from common import PRIV


def main():
    I = pd.DataFrame(json.loads((PRIV / 'heldout_index_v11.json').read_text()))
    tau = json.loads((PRIV / 'cases' / f'{I.case_id.iloc[0]}.json').read_text())['v11']['tau']
    I['margin'] = (I.max_p - tau).where(I.outcome.isin(['TP', 'FP']), tau - I.max_p)
    I['tie'] = (I.n_writings - I.n_writings.median()).abs()
    pick = lambda d: d.sort_values(['margin', 'tie'], ascending=[False, True]).iloc[0] if len(d) else None
    out = {}
    for name, sub, why in (('TP', I[I.outcome == 'TP'], 'referred positive, clearest margin'),
                           ('FP', I[I.outcome == 'FP'], 'referred control (false alarm)'),
                           ('FN', I[I.outcome == 'FN'], 'missed positive, largest miss')):
        r = pick(sub)
        if r is not None:
            out[name] = {'case_id': r.case_id, 'why': why, 'margin': round(float(r.margin), 3)}
    late = I[(I.outcome == 'TP') & (~I.case_id.isin([d['case_id'] for d in out.values()]))].sort_values('flag_i', ascending=False)
    if len(late):
        out['TP-late'] = {'case_id': late.case_id.iloc[0], 'why': 'referred positive, latest referral'}
    ch = I[I.outcome != I.v10_outcome].sort_values('case_id')
    if len(ch):
        r = ch.iloc[0]; out['CHANGED'] = {'case_id': r.case_id, 'why': f'outcome changed: {r.v10_outcome} (v10) -> {r.outcome} (v11)'}
    (PRIV / 'walkthrough_v11.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print('all outcome changes v10 -> v11:', ch[['case_id', 'v10_outcome', 'outcome']].to_dict('records'))


if __name__ == '__main__':
    main()
