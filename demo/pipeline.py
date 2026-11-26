"""One user, every step: per-writing features -> d14 chunk rows -> B encoding -> frozen LR -> flag.

Two entry points
  run_user(w_user, ...)       w_user = rows from the private writing table (no text), Qwen already scored
  run_raw(writings, ...)      new input: [{'ts': ISO time, 'text': str, 'is_comment': 0/1}], scored live
                              (rule features + NRC here, Qwen through local Ollama; see live.py)
The output is a plain dict that follows the demo JSON schema (see demo/README.md).
"""
import numpy as np
import pandas as pd

from common import CH, SE, SIG, EMO, SPEC, load_model

KEY_Q = [f'lg_{k}' for k in SIG] + [f'maxlv_{k}' for k in SIG] + [
    'distressed', 'polarity', 'polarity_sd', 'absolutist', 'first_person', 'profanity', 'night',
    'emo_sadness', 'emo_fear', 'emo_negative', 'emo_joy', 'log_rate', 'log_med_gap', 'silence_after_distress']
RULES = ['polarity', 'flip', 'abs_jump', 'absolutist', 'first_person', 'profanity', 'night']
STATUS = {'refer': 'Refer for a human counselling conversation',
          'no_referral': 'No referral at this point'}


class Frozen:
    """The frozen model as plain numpy, plus the original Encoder for the B inputs."""

    def __init__(self, m=None):
        self.m = m or load_model()
        enc = SE.Encoder('B')
        enc.edges = {q: np.asarray(e) for q, e in self.m['encoder']['edges'].items()}
        enc.tol = dict(self.m['encoder']['tol'])
        self.enc = enc
        self.features = self.m['features']
        self.mean = np.asarray(self.m['scaler_mean']); self.scale = np.asarray(self.m['scaler_scale'])
        self.coef = np.asarray(self.m['coef']); self.b0 = self.m['intercept']
        self.tau, self.t_min = self.m['tau'], self.m['t_min']

    def inputs(self, rows):
        X = self.enc.transform(rows)
        return X[self.features]

    def z(self, X):
        return (X.to_numpy(float) - self.mean) / self.scale

    def score(self, X):
        return 1 / (1 + np.exp(-(self.z(X) @ self.coef + self.b0)))


def rows_frame(x, every_writing):
    out = []
    for r in CH.user_rows(x, SPEC, every_writing):
        d = {'k': r['k'], 'day': r['day'], 'chunk': r['chunk'], 'first': int(r['first'])}
        for j, q in enumerate(CH.Q):
            d[f'cur_{q}'] = r['cur'][j]; d[f'prv_{q}'] = r['prv'][j]; d[f'pre_{q}'] = r['pre'][j]
        out.append(d)
    return pd.DataFrame(out)


def first_flag(p, k, tau, t_min):
    hit = np.where((p >= tau) & (k >= t_min))[0]
    return int(hit[0]) if len(hit) else None


def _r(v, n=4):
    v = float(v)
    return None if not np.isfinite(v) else round(v, n)


def run_user(x, fz=None, case_id='user', source='heldout', label=None, paraphrases=None):
    """x: one user's prepared writings (chunks.per_writing already applied, disclosure removed, sorted)."""
    fz = fz or Frozen()
    x = x.reset_index(drop=True)
    if len(x) == 0:
        return {'case_id': case_id, 'source': source, 'error': 'no writings left after masking'}
    W = rows_frame(x, every_writing=True)          # one decision row per writing
    E = rows_frame(x, every_writing=False)         # chunk-end rows (what the model was trained on)
    XW = fz.inputs(W)
    p = fz.score(XW)
    ff = first_flag(p, W.k.to_numpy(), fz.tau, fz.t_min)
    ff2 = first_flag(p, W.k.to_numpy(), fz.m['tau_prev_adj'], fz.m['t_min_prev_adj'])
    para = paraphrases or {}

    writings = []
    for i, r in x.iterrows():
        writings.append({
            'i': i + 1, 'day': _r((r.ts_sec - x.ts_sec.iloc[0]) / 86400, 2), 'is_comment': int(r.is_comment),
            'paraphrase': para.get(str(i + 1)),
            'qwen': {k: {'p': _r(1 / (1 + np.exp(-r[f'lg_{k}']))), 'logit': _r(r[f'lg_{k}'], 3),
                         'level': int(r[f'lv_{k}'])} for k in SIG},
            'distressed': int(r.distressed),
            'rules': {q: _r(r[q]) for q in RULES},
            'emo': {e: _r(r[f'emo_{e}']) for e in EMO},
        })

    chunks = []
    for _, r in E.iterrows():
        lvl, chg = {}, {}
        for q in KEY_Q:
            cur, prv = r[f'cur_{q}'], r[f'prv_{q}']
            if q in fz.enc.edges:
                lvl[q] = int(np.searchsorted(fz.enc.edges[q], cur, side='right'))
                t = fz.enc.tol[q]
            else:                                      # maxlv_*: already a 0-3 grade
                lvl[q] = int(cur) if q.startswith('maxlv_') else None
                t = 0.0 if q.startswith('maxlv_') else None
            dl = cur - prv
            chg[q] = None if (t is None or r['first']) else ('up' if dl > t else 'down' if dl < -t else 'same')
        chunks.append({'chunk': int(r.chunk), 'end_i': int(r.k), 'day': _r(r.day, 2),
                       'cur': {q: _r(r[f'cur_{q}']) for q in KEY_Q},
                       'prv': {q: _r(r[f'prv_{q}']) for q in KEY_Q},
                       'pre': {q: _r(r[f'pre_{q}']) for q in KEY_Q},
                       'level': lvl, 'change': chg})

    dec_i = ff if ff is not None else len(W) - 1     # explain the decision point (flag, or end of history)
    out = {
        'case_id': case_id, 'source': source,
        'model': {'id': fz.m['id'], 'spec': fz.m['spec'], 'C': fz.m['C'], 'tau': fz.tau, 't_min': fz.t_min,
                  'tau_prev_adj': fz.m['tau_prev_adj'], 't_min_prev_adj': fz.m['t_min_prev_adj'],
                  'inputs_sha256': fz.m['inputs_sha256']},
        'n_writings': len(x), 'span_days': _r((x.ts_sec.iloc[-1] - x.ts_sec.iloc[0]) / 86400, 1),
        'writings': writings, 'chunks': chunks,
        'scores': [{'i': int(k), 'day': _r(d, 2), 'p': _r(v, 5)} for k, d, v in zip(W.k, W.day, p)],
        'first_flag': {'i': int(W.k.iloc[ff]) if ff is not None else None,
                       'day': _r(W.day.iloc[ff], 2) if ff is not None else None},
        'first_flag_prev_adj': {'i': int(W.k.iloc[ff2]) if ff2 is not None else None},
        'status': 'refer' if ff is not None else 'no_referral',
        'decision_row': {'i': int(W.k.iloc[dec_i]), 'p': _r(p[dec_i], 5), 'max_p': _r(p.max(), 5)},
        'label': label,
    }
    out['status_text'] = STATUS[out['status']]
    out['_decision_inputs'] = (XW.iloc[[dec_i]], W.iloc[dec_i])     # internal, stripped before export
    return out


def strip(o):
    return {k: v for k, v in o.items() if not k.startswith('_')}
