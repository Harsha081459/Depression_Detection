"""Why the model refers (or not): exact contributions of the frozen logistic regression.

score logit = intercept + sum_j coef_j * z_j, with z_j the input standardised against the dev training rows.
Each input belongs to one quantity q (c_/p_/d_ continuous; lvc/lvp quartile levels; up_/down_ changes),
so the logit splits exactly into one contribution per quantity. Positive = pushes towards referral,
relative to an average development chunk.
"""
import re

import numpy as np

LABEL = {
    'lg_self_blame': 'self-blame (Qwen)', 'lg_helplessness': 'helplessness (Qwen)',
    'lg_isolation': 'isolation (Qwen)', 'lg_family_strain': 'family strain (Qwen)', 'lg_emptiness': 'emptiness (Qwen)',
    'maxlv_self_blame': 'self-blame level 0-3', 'maxlv_helplessness': 'helplessness level 0-3',
    'maxlv_isolation': 'isolation level 0-3', 'maxlv_family_strain': 'family strain level 0-3',
    'maxlv_emptiness': 'emptiness level 0-3',
    'distressed': 'share of distressed writings', 'lv_max': 'highest Qwen level',
    'absolutist': 'absolutist + negative words', 'polarity': 'polarity (positive - negative words)',
    'flip': 'mood flips', 'abs_jump': 'mood jumps', 'polarity_sd': 'mood swings', 'first_person': 'first-person words',
    'profanity': 'profanity', 'night': 'night-time posting', 'log_rate': 'posting frequency',
    'log_med_gap': 'gap between posts', 'silence_after_distress': 'silence after distress',
}
LABEL.update({f'emo_{e}': f'{e} words (NRC)' for e in
              ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'positive', 'negative']})
PREFIX = re.compile(r'^(c|p|d|lvc\d|lvp\d|up|down)_')
VIEW = {'c': 'last 14 days', 'p': 'history so far', 'd': 'change vs previous chunk',
        'up': 'rose vs previous chunk', 'down': 'fell vs previous chunk'}


def quantity(feature):
    return PREFIX.sub('', feature)


def view(feature):
    m = PREFIX.match(feature).group(1)
    if m.startswith('lvc'):
        return f'last-14-day level {m[3:]}'
    if m.startswith('lvp'):
        return f'history level {m[3:]}'
    return VIEW[m]


def contributions(fz, X1):
    """X1: one-row DataFrame of model inputs. Returns per-quantity contributions, sorted (desc)."""
    z = fz.z(X1)[0]; c = fz.coef * z
    out = {}
    for f, v in zip(fz.features, c):
        q = quantity(f)
        d = out.setdefault(q, {'quantity': q, 'label': LABEL.get(q, q), 'contribution': 0.0, 'parts': []})
        d['contribution'] += float(v)
        if abs(v) > 1e-3:
            d['parts'].append((view(f), float(v)))
    rows = sorted(out.values(), key=lambda d: -d['contribution'])
    for d in rows:
        d['contribution'] = round(d['contribution'], 4)
        d['parts'] = sorted(d['parts'], key=lambda t: -abs(t[1]))[:2]
        d['note'] = '; '.join(f'{a} {b:+.3f}' for a, b in d['parts'])
        del d['parts']
    return rows, float(fz.b0)


def summary(o, top, card):
    up = [d['label'] for d in top if d['contribution'] > 0][:3]
    ups = ', '.join(up[:-1]) + (' and ' if len(up) > 1 else '') + up[-1] if up else 'no single group'
    m = o['model']; dr = o['decision_row']
    if o['status'] == 'refer':
        s = (f"Suggested: refer for a human counselling conversation. The model score reached {dr['p']:.2f} "
             f"(threshold {m['tau']:.2f}) at writing {o['first_flag']['i']} (day {o['first_flag']['day']:.0f}). "
             f"Main reasons: {ups}.")
    else:
        s = (f"No referral at this point. The highest score was {dr['max_p']:.2f}, below the threshold "
             f"{m['tau']:.2f}. Signals that pushed up most at the last writing: {ups}.")
    if card:
        s += f" Scorecard: {card['colour'].upper()}"
        s += f" ({'; '.join(card['rules_fired'])})." if card['rules_fired'] else '.'
    s += ' This is a screening flag, not a diagnosis.'
    return s


def explain(fz, o, card=None, k=5):
    X1, _ = o['_decision_inputs']
    rows, b0 = contributions(fz, X1)
    top = rows[:k]
    neg = [d for d in rows[::-1] if d['contribution'] < 0][:2]
    return {'at_writing': o['decision_row']['i'], 'intercept': round(b0, 4),
            'top': top, 'against': neg, 'summary': summary(o, top, card),
            'logit_check': round(b0 + sum(d['contribution'] for d in rows), 4)}
