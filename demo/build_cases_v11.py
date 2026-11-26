"""Demo records for the 105 held-out users under the frozen v11 model (adds to the v10 case files).

For each user (same pseudonymous HO-xxx ids):
  v11.scores          per-writing final score; member scores (text model, signals model if any)
  v11.first_flag      first writing with score >= tau and k >= t_min (prev_adj policy)
  v11.top_writings    writings that moved the text model's score most (exact attribution), with their
                      symptom matches, emotions and Qwen levels; paraphrase if the team wrote one
  v11.symptoms        history-so-far symptom-similarity profile at the decision point, as dev percentiles
  v11.emotions        same for go_emotions
  v11.summary         template sentence (signs, not diagnosis)
Writes private/demo/cases/HO-xxx.json (updated in place) and demo/artifacts/heldout_v11_index.json-like
private/demo/heldout_index_v11.json.
"""
import json
import sys

import numpy as np
import pandas as pd

from common import ART, PRIV, ROOT, load_split, load_writings, prepare, SIG
from model_v11 import FrozenV11

sys.path.insert(0, str(ROOT / 'research_v11'))
from v11 import load_matrix                                                     # noqa: E402

REF = json.load(open(ART / 'signals_ref.json'))
SYM, EMO = REF['symptoms'], REF['emotions']
SYM_TEXT = json.load(open(ROOT / 'research_v11' / 'symptoms.json'))['symptoms']
NICE = {k: k.replace('_', ' ') for k in SYM}
NICE.update({'self_harm': 'self-harm thoughts', 'anhedonia': 'loss of enjoyment', 'self_hate': 'self-hatred',
             'no_energy': 'no energy', 'deserve_punishment': 'feeling deserving of punishment'})


def pct(q, v):
    q = np.asarray(q); return float(100 * np.searchsorted(q, v, 'right') / (len(q) - 1)) if np.isfinite(v) else None


def profile(M, upto, name):
    """History mean up to writing `upto` (1-based) -> percentile vs dev history means."""
    labels = SYM if name == 'BDI' else EMO
    m = M[:upto].mean(0); qs = REF[name]['history_quantiles']
    return sorted(({'key': l, 'label': NICE.get(l, l) if name == 'BDI' else l, 'value': round(float(v), 4),
                    'percentile': round(pct(qs[j], v), 1), 'dev_auroc': round(REF[name]['dev_auroc'][l], 3)}
                   for j, (l, v) in enumerate(zip(labels, m))), key=lambda d: -d['percentile'])


def writing_tags(Mb, Me, i):
    zs = (Mb[i] - np.asarray(REF['BDI']['writing_mean'])) / np.asarray(REF['BDI']['writing_sd'])
    ze = (Me[i] - np.asarray(REF['EMO']['writing_mean'])) / np.asarray(REF['EMO']['writing_sd'])
    sym = [NICE[SYM[j]] for j in np.argsort(-zs)[:3] if zs[j] > 1.5]
    emo = [EMO[j] for j in np.argsort(-ze) if EMO[j] != 'neutral' and ze[j] > 1.5][:3]
    return sym, emo, float(zs.max())


def summary(o, top_sym, top_writings):
    signs = ', '.join(d['label'] for d in top_sym[:3])
    if o['status'] == 'refer':
        s = (f"Suggested: refer for a human counselling conversation. The score reached {o['score_at_decision']:.2f} "
             f"(threshold {o['tau']:.2f}) at writing {o['first_flag']['i']} (day {o['first_flag']['day']:.0f}). ")
    else:
        if o['max_score'] is None:
            s = f"No referral at this point: fewer than {o['t_min']} writings, too few to decide. "
        else:
            s = (f"No referral at this point: from writing {o['t_min']} on, the highest score was {o['max_score']:.2f}, "
                 f"below the threshold {o['tau']:.2f}. ")
    s += f"Strongest signs in the writing history so far: {signs}."
    if top_writings:
        s += f" {len(top_writings)} writings moved the score most (shown below)."
    return s + ' This is a screening flag, not a diagnosis.'


def build(x, mats, fz, para=None):
    """x: prepared writings of one user; mats: {'MPN','TXT','EMO','BDI'} rows aligned with x."""
    res = fz.score_user(x, mats)
    i = res['flag'] if res['flag'] is not None else len(res['p']) - 1
    ex = fz.explain_row(res, i)
    k = int(res['k'][i])
    o = {'model_id': fz.m['id'], 'tau': fz.tau, 't_min': fz.t_min,
         'scores': [{'i': int(a), 'day': round(float(d), 2), 'p': round(float(v), 5)} for a, d, v in zip(res['k'], res['day'], res['p'])],
         'member_scores': {e['member']: round(float(1 / (1 + np.exp(-r['z'][i]))), 4) for e, r in zip(ex, res['members'])},
         'member_series': {'+'.join(r['mem'].blocks): [round(float(1 / (1 + np.exp(-z))), 4) for z in r['z']] for r in res['members']},
         'first_flag': {'i': k if res['flag'] is not None else None, 'day': round(float(res['day'][i]), 2) if res['flag'] is not None else None},
         'status': 'refer' if res['flag'] is not None else 'no_referral', 'decision_i': k,
         'score_at_decision': float(res['p'][i]),
         # only writings the rule can act on (k >= t_min); earlier scores rest on too few writings
         'max_score': float(res['p'][res['k'] >= fz.t_min].max()) if (res['k'] >= fz.t_min).any() else None,
         'max_score_before_t_min': float(res['p'][res['k'] < fz.t_min].max()) if (res['k'] < fz.t_min).any() else None}
    Mb, Me = mats['BDI'], mats['EMO']
    tops = []
    for e in ex:
        for b, v in e.items():
            if isinstance(v, dict) and 'per_writing' in v:
                pw = np.asarray(v['per_writing'])
                for j in np.argsort(-pw)[:5]:
                    if pw[j] <= 0:
                        continue
                    sym, emo, _ = writing_tags(Mb, Me, j)
                    r = x.iloc[j]
                    tops.append({'i': int(j + 1), 'day': round(float((r.ts_sec - x.ts_sec.iloc[0]) / 86400), 1),
                                 'contribution': round(float(pw[j]), 4), 'block': b,
                                 'symptoms': sym, 'emotions': emo,
                                 'qwen_levels': {s: int(r[f'lv_{s}']) for s in SIG},
                                 'paraphrase': (para or {}).get(str(j + 1))})
                o['per_writing_contribution'] = [round(float(c), 5) for c in pw]
                o['text_attribution'] = {'block': b, 'baseline': round(v['baseline'], 4), 'total': round(v['total'], 4),
                                         'writings_up': int((pw > 0).sum()), 'writings_down': int((pw < 0).sum()),
                                         'sum_up': round(float(pw[pw > 0].sum()), 4), 'sum_down': round(float(pw[pw < 0].sum()), 4)}
        if 'B' in e:
            o['signals_contributions'] = [{'label': d['label'], 'contribution': round(d['contribution'], 4)} for d in e['B'][:5]]
    o['top_writings'] = sorted(tops, key=lambda d: -d['contribution'])[:5]
    o['symptoms'] = profile(Mb, k, 'BDI'); o['emotions'] = [d for d in profile(Me, k, 'EMO') if d['key'] != 'neutral']
    o['per_writing_signs'] = []
    for j in range(len(x)):
        sym, emo, zmax = writing_tags(Mb, Me, j)
        o['per_writing_signs'].append({'i': j + 1, 'symptoms': sym, 'emotions': emo, 'max_z': round(zmax, 2)})
    o['summary'] = summary(o, o['symptoms'], o['top_writings'])
    return o


def main():
    fz = FrozenV11(); users = load_split(); ho = users[users.split != 'train']
    ids = json.loads((PRIV / 'case_ids_PRIVATE.json').read_text())
    para_all = json.loads((PRIV / 'paraphrases.json').read_text(encoding='utf-8')) if (PRIV / 'paraphrases.json').exists() else {}
    cuts = json.load(open(ROOT / 'research_v10' / 'real_qwen' / 'chunk_meta.json'))['cuts']
    w = load_writings(set(ho.user_id)); w['orig'] = w.index.to_numpy(); w = prepare(w, cuts)
    need = {b.rstrip('0123456789') for mem in fz.members for b in mem.blocks if b != 'B'} | {'EMO', 'BDI'}
    mats_all = {n: load_matrix(n, w.orig.to_numpy()) for n in need}
    rows = []
    for _, u in ho.iterrows():
        cid = ids[u.user_id]; m = (w.user_id == u.user_id).to_numpy()
        f = PRIV / 'cases' / f'{cid}.json'; case = json.loads(f.read_text())
        if m.sum() == 0:
            continue
        x = w[m]; mats = {n: M[m] for n, M in mats_all.items()}
        case['v11'] = build(x, mats, fz, para_all.get(cid)); f.write_text(json.dumps(case))
        v = case['v11']; refer = v['status'] == 'refer'
        rows.append({'case_id': cid, 'label': int(u.label), 'outcome': {(1, True): 'TP', (0, True): 'FP', (1, False): 'FN', (0, False): 'TN'}[(int(u.label), refer)],
                     'flag_i': v['first_flag']['i'], 'max_p': round(v['max_score'], 4), 'n_writings': int(m.sum()),
                     'v10_outcome': None})
    I = pd.DataFrame(rows)
    old = pd.DataFrame(json.loads((PRIV / 'heldout_index.json').read_text())).set_index('case_id')
    I['v10_outcome'] = I.case_id.map(old.outcome)
    (PRIV / 'heldout_index_v11.json').write_text(I.to_json(orient='records', indent=1))
    print(I.outcome.value_counts().to_dict(), '| v10 ->', pd.crosstab(I.v10_outcome, I.outcome).to_dict())


if __name__ == '__main__':
    main()
