"""Green / amber / red scorecard (the professor's 'scorecarding').

Every item is oriented so that a higher value is worse. Item value -> percentile among development
chunk-end rows (label-free) -> points 0-10. Rules (constants in artifacts/scorecard.json):
  red    any single item >= P_single_red, or >= 2 items >= P_pair_red, or composite >= P_comp_red
  amber  any single item >= P_single_amber, or >= 2 items >= P_pair_amber, or composite >= P_comp_amber
  green  otherwise
The logistic-regression flag is the decision; the scorecard explains it in plain terms.
"""
import json

import numpy as np

from common import ART

# item -> (label, source quantities averaged, sign). Values come from one window of a decision row:
# 'cur' = current 14-day chunk, 'pre' = whole history so far.
ITEMS = {
    'self_blame':    ('Self-blame (Qwen)', ['lg_self_blame'], 1),
    'helplessness':  ('Helplessness (Qwen)', ['lg_helplessness'], 1),
    'isolation':     ('Isolation (Qwen)', ['lg_isolation'], 1),
    'family_strain': ('Family strain (Qwen)', ['lg_family_strain'], 1),
    'emptiness':     ('Emptiness (Qwen)', ['lg_emptiness'], 1),
    'distressed':    ('Share of distressed writings', ['distressed'], 1),
    'neg_affect':    ('Negative emotion words', ['emo_sadness', 'emo_fear', 'emo_negative'], 1),
    'low_positivity': ('Low positivity (-polarity)', ['polarity'], -1),
    'mood_swing':    ('Mood swings (polarity SD)', ['polarity_sd'], 1),
    'absolutist':    ('Absolutist + negative words', ['absolutist'], 1),
    'night':         ('Night-time posting', ['night'], 1),
    'silence':       ('Silence after distress', ['silence_after_distress'], 1),
}


def item_value(row, k, win):
    _, qs, sign = ITEMS[k]
    return sign * sum(row[f'{win}_{q}'] for q in qs) / len(qs)


COLOURS = ('green', 'amber', 'red')


def item_values(rows, win):
    """rows: DataFrame of decision rows -> DataFrame of raw item values for window `win`."""
    import pandas as pd
    return pd.DataFrame({k: item_value(rows, k, win) for k in ITEMS}, index=rows.index)


class Scorecard:
    def __init__(self, cfg=None):
        self.cfg = cfg or json.loads((ART / 'scorecard.json').read_text())
        self.q = {k: np.asarray(v) for k, v in self.cfg['quantiles'].items()}   # 1001 quantiles per item
        self.groups = self.cfg['groups']                                         # merged items
        self.win = self.cfg['window']

    def item_points(self, vals):
        """vals: dict item -> raw value. Returns item -> points (0-10)."""
        pct = {}
        for k, v in vals.items():
            if k not in self.q or v is None or not np.isfinite(v):
                continue
            q = self.q[k]
            lo, hi = np.searchsorted(q, v, 'left'), np.searchsorted(q, v, 'right')
            pct[k] = (lo + hi) / 2 / (len(q) - 1) * 10                     # mid-rank for ties
        return pct

    def points(self, vals):
        """Group points = mean of member item points."""
        pct = self.item_points(vals); out = {}
        for g, members in self.groups.items():
            ps = [pct[m] for m in members if m in pct]
            if ps:
                out[g] = min(10.0, float(np.mean(ps)))
        return out

    def colour(self, pts):
        c = self.cfg['rules']; v = np.array(list(pts.values())); comp = float(v.mean()) if len(v) else 0.0
        fired = {'red': [], 'amber': []}
        for col in ('red', 'amber'):
            r = c[col]
            lab = self.cfg['labels']
            single = [lab.get(g, g) for g, p in pts.items() if p >= r['single'] / 10]
            pair = [lab.get(g, g) for g, p in pts.items() if p >= r['pair'] / 10]
            if single:
                fired[col].append(f"single item >= P{r['single']:g}: " + ', '.join(single))
            if len(pair) >= 2:
                fired[col].append(f"{len(pair)} items >= P{r['pair']:g}: " + ', '.join(pair))
            if comp >= c['composite_cut'][col]:
                fired[col].append(f"composite {comp:.1f} >= {c['composite_cut'][col]:.1f}")
        colour = 'red' if fired['red'] else 'amber' if fired['amber'] else 'green'
        return colour, comp, fired['red'] if colour == 'red' else fired['amber'] if colour == 'amber' else []

    def card(self, row):
        """row: a Series / dict with cur_* values (one decision row)."""
        vals = {k: float(item_value(row, k, self.win)) for k in ITEMS}
        ip = self.item_points(vals); pts = self.points(vals)
        colour, comp, fired = self.colour(pts)
        return {'window': self.win,
                'items': {g: {'label': self.cfg['labels'][g], 'points': round(p, 2),
                              'members': {m: {'label': ITEMS[m][0], 'value': round(vals[m], 4),
                                              'points': round(ip.get(m, float('nan')), 2)} for m in self.groups[g]}}
                          for g, p in pts.items()},
                'composite': round(comp, 2), 'colour': colour, 'rules_fired': fired}


def agreement_note(colour, refer):
    if refer and colour == 'green':
        return 'Model refers, card is green: the referral comes from the pattern over time, not from one extreme signal.'
    if not refer and colour == 'red':
        return 'Card is red but the model does not refer: an isolated high signal. Worth a manual look.'
    if refer:
        return f'Model refers and the card is {colour}: they agree.'
    return f'No referral and the card is {colour}.'
