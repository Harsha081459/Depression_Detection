"""Early depression-risk screening demo (eRisk 2026 Task 2, Team 6). Runs locally only:
    streamlit run demo/app.py --server.address localhost
One user, every step: writings -> transformer + Qwen signals -> chunks -> final score -> referral -> reasons.
Output wording: a suggestion to refer for a human counselling conversation, never a diagnosis.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ART, PRIV, ROOT, SIG, EMO, load_model           # noqa: E402

st.set_page_config(page_title='Risk screening demo', layout='wide')
SIGLAB = {'self_blame': 'Self-blame', 'helplessness': 'Helplessness', 'isolation': 'Isolation',
          'family_strain': 'Family strain', 'emptiness': 'Emptiness'}
COL = {'green': '#2e7d32', 'amber': '#ef8f00', 'red': '#c62828', 'blue': '#1f4e79', 'grey': '#90a4ae'}


@st.cache_resource
def engines():
    import pipeline as PL
    from scorecard import Scorecard
    from model_v11 import FrozenV11
    return PL.Frozen(), Scorecard(), FrozenV11()


@st.cache_data
def index():
    f = PRIV / 'heldout_index_v11.json'
    return pd.DataFrame(json.loads(f.read_text())) if f.exists() else pd.DataFrame()


@st.cache_data
def walkthrough():
    f = PRIV / 'walkthrough_v11.json'
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data
def looks_for():
    """What the text model's history direction is closest to, among the 21 symptom statements (global)."""
    from model_v11 import FrozenV11, direction_vs_statements
    fz = FrozenV11(); sym = json.load(open(ROOT / 'research_v11' / 'symptoms.json'))['symptoms']
    S = np.load(ART / 'symptoms_mpnet.npy')
    for mem in fz.members:
        for b in mem.blocks:
            if b.startswith('MPN'):
                c = direction_vs_statements(mem, b, S)
                return sorted(zip(sym.values(), c.tolist()), key=lambda t: -t[1])
    return []


def load_case(cid):
    return json.loads((PRIV / 'cases' / f'{cid}.json').read_text())


m10 = load_model(); m11 = json.load(open(ART / 'frozen_v11.json'))
ho = m11['heldout_second_look']['prev_adj']

# ---------------------------------------------------------------- sidebar: model card
with st.sidebar:
    st.header('Final model (v11)')
    mem = m11['members']
    st.markdown(f"""
- **Decision model**: {' + '.join('+'.join(x['blocks']) for x in mem)}, logistic regression
  on chunk views ({mem[0]['spec'].replace('p', '')}-writing chunks: current, previous, history) of each writing's
  sentence embedding (all-mpnet-base-v2, PCA {mem[0]['blocks'][0][3:]} fitted on dev writings)
- **Refer** at the first writing with score ≥ **{m11['tau']:.3f}** after ≥ {m11['t_min']} writings
- **Dev (418 users, nested CV, 5×5)**: see tab 7
- **Held-out (105 users, second look)**: AUROC {ho['auroc']:.3f}, F1 {ho['f1']:.3f} (TP {ho['tp']}, FP {ho['fp']}, FN {ho['fn']})
- **Signals shown for explanation**: Qwen2.5-7B (5 signals), go_emotions (RoBERTa), 21 symptom statements (bge-large)
""")
    st.caption('Output is a suggestion to refer for a human counselling conversation. It is not a diagnosis.')

st.title('Early risk screening: one user, step by step')
tabs = st.tabs(['0 Input', '1 Writings & signals', '2 Word features', '3 Chunks', '4 Score & referral',
                '5 Signs & scorecard', '6 Why', '7 Results'])


def run_live(writings, use_qwen):
    import live
    from build_cases import build_one
    from build_cases_v11 import build
    fz10, sc, fz11 = engines()
    tmp = PRIV / 'live_tmp'; tmp.mkdir(parents=True, exist_ok=True)
    rf = live.rule_features(writings)                              # same order as prepared rows (time order)
    texts = rf[~rf.disclosure].sort_values('ts_sec', kind='mergesort').text.tolist()   # = prepared row order
    (tmp / 'in.json').write_text(json.dumps(texts), encoding='utf-8')
    py = ROOT / '.venv' / 'Scripts' / 'python.exe'
    import os
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')    # cached models only: no network calls
    r = subprocess.run([str(py), str(HERE / 'embed_live.py'), '--all', str(tmp / 'in.json'), str(tmp / 'x')],
                       capture_output=True, text=True, env=env, timeout=300)
    (tmp / 'in.json').unlink(missing_ok=True)
    if r.returncode != 0:
        return {'error': 'embedding step failed (GPU environment .venv)'}
    mats = {'MPN': np.load(tmp / 'x_mpn.npy'), 'EMO': np.load(tmp / 'x_emo.npy'), 'BDI': np.load(tmp / 'x_bdi.npy')}
    mats['TXT'] = np.zeros((len(texts), 1024), np.float32)
    if use_qwen:
        x = live.prepared(writings, m10['cuts'], True)
        o = build_one(x, fz10, sc, 'LIVE', 'upload')
    else:                                                          # text model needs no Qwen: show signals as n/a
        import chunks as CH
        x = live.prepared(writings, m10['cuts'], False)
        for k in SIG:
            x[f'lg_{k}'] = np.nan; x[f'p_{k}'] = np.nan
        x = CH.per_writing(x.assign(user_id='live'), m10['cuts'])
        o = {'case_id': 'LIVE', 'n_writings': len(x), 'span_days': float((x.ts_sec.iloc[-1] - x.ts_sec.iloc[0]) / 86400),
             'no_qwen': True, 'label': None}
    o['v11'] = build(x, mats, fz11)
    return o


# ---------------------------------------------------------------- tab 0: input
with tabs[0]:
    src = st.radio('Source', ['Walkthrough case', 'Any held-out user', 'Upload JSON', 'Type writings'], horizontal=True)
    o = None; I = index()
    if src == 'Walkthrough case':
        wt = walkthrough()
        if not wt:
            st.info('No walkthrough cases chosen yet (run demo/pick_walkthrough.py).')
        else:
            name = st.selectbox('Case', list(wt), format_func=lambda k: f"{k}: {wt[k]['why']}")
            o = load_case(wt[name]['case_id'])
    elif src == 'Any held-out user':
        if I.empty:
            st.warning('Run demo/build_cases_v11.py first.')
        else:
            show_lab = st.checkbox('Show true label and outcome (walkthrough only)', value=False)
            lab = I.apply(lambda r: f"{r.case_id}  ·  {r.n_writings} writings" + (f"  ·  {r.outcome}" if show_lab else ''), axis=1)
            cid = st.selectbox('Held-out user (pseudonymous)', I.case_id, format_func=dict(zip(I.case_id, lab)).get)
            o = load_case(cid)
            if not show_lab:
                o['label'] = None
    else:
        import live
        writings = None
        if src == 'Upload JSON':
            up = st.file_uploader('JSON: {"writings": [{"ts": ISO time, "text": str, "is_comment": 0/1}]}', type='json')
            ex = st.selectbox('…or a fictional example', ['', 'synthetic_rising.json', 'synthetic_ordinary.json'])
            if up is not None:
                writings = json.loads(up.read())['writings']
            elif ex:
                writings = json.loads((HERE / 'examples' / ex).read_text())['writings']
        else:
            txt = st.text_area('One writing per line: `YYYY-MM-DD HH:MM | text` (5–40 lines, times in IST)', height=200)
            rows = [l.split('|', 1) for l in txt.splitlines() if '|' in l]
            if rows:
                writings = [{'ts': a.strip() + '+05:30', 'text': b.strip(), 'is_comment': 0} for a, b in rows]
        if writings:
            wkey = hashlib.sha256(json.dumps(writings, sort_keys=True).encode('utf-8')).hexdigest()
            st.caption(f'{len(writings)} writings received. The text stays on this machine (and the lab, if Qwen runs '
                       'there through the SSH tunnel). Use fictional writings.')
            ok, info = live.ollama_status()
            if ok:
                st.success(f"Qwen ready: {info['model']} digest {info['model_digest'][:8]}…, prompt hash {info['prompt_hash'][:8]}…")
            else:
                st.info('Qwen not reachable: the final score still runs (it needs only the sentence embeddings); the '
                        'five Qwen signals will show as not scored. To add them, run `bash demo/lab_tunnel.sh` and reload.')
            if st.button('Run pipeline', type='primary'):
                with st.spinner('Embedding writings on the GPU and scoring…'):
                    st.session_state['live'] = (wkey, run_live(writings, ok))
            res = st.session_state.get('live')
            o = res[1] if res and res[0] == wkey else None
            if o is None:
                st.caption('Press Run pipeline to score this input.')

    if o is not None and 'error' not in o:
        v = o['v11']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Writings (after masking)', o['n_writings'])
        c2.metric('History span', f"{o['span_days']:.0f} days")
        c3.metric('Decision', 'Refer' if v['status'] == 'refer' else 'No referral')
        c4.metric('Scorecard (signals)', o['scorecard']['colour'].upper() if 'scorecard' in o else 'n/a')
        if v['status'] == 'refer':
            st.error(f"**Refer for a human counselling conversation**: first flagged at writing {v['first_flag']['i']} "
                     f"(day {v['first_flag']['day']:.0f}).")
        else:
            st.success('**No referral at this point**')
        st.markdown(f"> {v['summary']}")
        if o.get('label') is not None:
            st.caption(f"True label (walkthrough only): {'positive' if o['label'] else 'control'}")
        st.download_button('Export JSON', json.dumps(o, indent=1), file_name=f"{o['case_id']}.json")
    elif o is not None:
        st.warning(o['error'])

if o is None or 'error' in o:
    for t in tabs[1:7]:
        with t:
            st.info('Pick or run a user in tab 0.')
else:
    v = o['v11']; nq = o.get('no_qwen', False)
    signs = {d['i']: d for d in v['per_writing_signs']}
    push = v.get('per_writing_contribution', [])
    # ------------------------------------------------------------ tab 1: writings & signals
    with tabs[1]:
        st.markdown('One row per writing. **Text push** = how much this writing moved the final model\'s score at the '
                    'decision point (exact share of the log-odds; + towards referral). **Symptoms** = symptom statements '
                    'this writing reads unusually like (bge-large similarity, > 1.5 SD above dev writings). '
                    '**Emotions** = go_emotions (RoBERTa) labels unusually high for this writing. '
                    '**Qwen** = 0–3 level of each of the five yes/no signals (P(Yes) cut at 0.25 / 0.5 / 0.75).')
        rows = []
        for j in range(o['n_writings']):
            i = j + 1; s = signs.get(i, {})
            w = o['writings'][j] if not nq else None
            rows.append({'#': i, 'day': w['day'] if w else None,
                         'writing': (w['paraphrase'] if w and w['paraphrase'] else '[text hidden]'),
                         'text push': push[j] if j < len(push) else None,
                         'symptoms': ', '.join(s.get('symptoms', [])), 'emotions': ', '.join(s.get('emotions', [])),
                         **({f'Q {SIGLAB[k]}': w['qwen'][k]['level'] for k in SIG} if w else {})})
        df = pd.DataFrame(rows)
        if push:
            fig = go.Figure(go.Bar(x=df['#'], y=df['text push'], marker_color=[COL['red'] if (p or 0) > 0 else COL['blue'] for p in df['text push']]))
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title=f"Text push per writing (decision at writing {v['decision_i']})",
                              xaxis_title='writing #', yaxis_title='log-odds')
            st.plotly_chart(fig, use_container_width=True)
        if not nq:
            lv = np.array([[w['qwen'][k]['level'] for w in o['writings']] for k in SIG])
            fig = go.Figure(go.Heatmap(z=lv, x=list(range(1, o['n_writings'] + 1)), y=[SIGLAB[k] for k in SIG], zmin=0, zmax=3,
                                       colorscale=[[0, '#f1f3f4'], [0.34, '#fde0a0'], [0.67, '#f39c4a'], [1, '#c62828']],
                                       colorbar=dict(title='level', tickvals=[0, 1, 2, 3])))
            fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), title='Qwen signal level per writing', xaxis_title='writing #')
            st.plotly_chart(fig, use_container_width=True)
        only = st.checkbox('Only writings with a sign (symptom, emotion or Qwen level ≥ 2) or a paraphrase', value=True)
        if only:
            qcols = [c for c in df.columns if c.startswith('Q ')]
            keep = (df.symptoms != '') | (df.emotions != '') | (df.writing != '[text hidden]')
            if qcols:
                keep |= (df[qcols] >= 2).any(axis=1)
            df = df[keep]
        st.dataframe(df, hide_index=True, use_container_width=True, height=440)

    # ------------------------------------------------------------ tab 2: word features
    with tabs[2]:
        if nq:
            st.info('Word features of uploaded writings are shown in the signals view when Qwen is connected.')
        else:
            st.markdown('Word-list features per writing (part of the signals model). Polarity = (pos − neg)/(pos + neg + 1). '
                        'Absolutist = absolutist word together with a negative word. Night = posted 00:00–05:59 IST.')
            W = o['writings']
            rf = pd.DataFrame([{'#': w['i'], 'day': w['day'], **w['rules'], **{f'emo_{e}': w['emo'][e] for e in EMO}} for w in W])
            st.dataframe(rf, hide_index=True, use_container_width=True, height=380)

    # ------------------------------------------------------------ tab 3: chunks
    with tabs[3]:
        if nq:
            st.info('Chunk view needs the Qwen signals.')
        else:
            st.markdown('The history is cut into chunks. For each chunk the models see the **current chunk** (*cur*), the '
                        '**previous chunk** (*prv*) and the **history so far** (*pre*), plus **up / same / down** changes. '
                        'The final text model uses the same three views of the sentence embeddings (10-writing chunks).')
            C = o['chunks']
            q = st.selectbox('Signal', list(C[0]['cur']), index=list(C[0]['cur']).index('maxlv_helplessness'))
            arrow = {'up': '↑', 'down': '↓', 'same': '=', None: ''}
            ct = pd.DataFrame([{'chunk': c['chunk'], 'ends at #': c['end_i'], 'day': c['day'], 'cur': c['cur'][q],
                                'prv': c['prv'][q], 'pre': c['pre'][q],
                                'level (0-3)': '–' if c['level'][q] is None else c['level'][q], 'change': arrow[c['change'][q]]} for c in C])
            fig = go.Figure([go.Scatter(x=ct['day'], y=ct['cur'], mode='lines+markers', name='cur (chunk)'),
                             go.Scatter(x=ct['day'], y=ct['pre'], mode='lines', name='pre (history)')])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title='day')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ct, hide_index=True, use_container_width=True, height=320)

    # ------------------------------------------------------------ tab 4: score & referral
    with tabs[4]:
        S = pd.DataFrame(v['scores'])
        fig = go.Figure(go.Scatter(x=S['i'], y=S['p'], mode='lines', name='final model (text)', line=dict(color=COL['blue'], width=2)))
        if not nq and 'scores' in o:
            S10 = pd.DataFrame(o['scores'])
            fig.add_trace(go.Scatter(x=S10['i'], y=S10['p'], mode='lines', name='signals model (Qwen + words, v10)',
                                     line=dict(color=COL['grey'], width=1, dash='dot')))
        fig.add_hline(y=v['tau'], line_dash='dash', line_color=COL['red'], annotation_text=f"threshold {v['tau']:.3f}",
                      annotation_position='bottom right')
        fig.add_vrect(x0=0.5, x1=v['t_min'] - 0.5, fillcolor='#ccc', opacity=0.3, line_width=0)
        if v['first_flag']['i']:
            fig.add_vline(x=v['first_flag']['i'], line_color=COL['red'])
            fig.add_annotation(x=v['first_flag']['i'], y=1, text='referral', showarrow=False, yshift=10)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title='writing #', yaxis_title='score',
                          yaxis_range=[0, 1], legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"Rule: refer at the **first writing with score ≥ {v['tau']:.3f} after at least {v['t_min']} writings**. "
                    f"Here: **{'refer at writing ' + str(v['first_flag']['i']) if v['status'] == 'refer' else 'no referral'}**. "
                    'The grey dotted line is the earlier signals-only model, for comparison.')

    # ------------------------------------------------------------ tab 5: signs & scorecard
    with tabs[5]:
        st.markdown(f"**Signs in the writing history so far** (at writing {v['decision_i']}), as percentiles of development "
                    'users\' histories. Each bar = how much the writing history reads like one plain-language symptom '
                    'statement (bge-large similarity). 95 = higher than 95% of development histories.')
        sy = pd.DataFrame(v['symptoms'][:10])
        fig = go.Figure(go.Bar(x=sy['percentile'], y=sy['label'], orientation='h',
                               marker_color=[COL['red'] if p >= 95 else COL['amber'] if p >= 85 else COL['grey'] for p in sy['percentile']]))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 100], yaxis=dict(autorange='reversed'),
                          xaxis_title='percentile vs development histories')
        st.plotly_chart(fig, use_container_width=True)
        em = pd.DataFrame(v['emotions'][:6])
        st.markdown('**Emotions (go_emotions)**: ' + ', '.join(f"{r.key} (P{r.percentile:.0f})" for r in em.itertuples()))
        if 'scorecard' in o:
            card = o['scorecard']
            st.markdown('---')
            st.markdown(f"**Green / amber / red scorecard (signals)**: <span style='padding:4px 10px;border-radius:6px;"
                        f"background:{COL[card['colour']]};color:white'>{card['colour'].upper()} · {card['composite']:.1f}/10</span>",
                        unsafe_allow_html=True)
            st.markdown('Rules fired: ' + ('; '.join(card['rules_fired']) or 'none'))
            bars = pd.DataFrame([{'item': x['label'], 'points': x['points']} for x in card['items'].values()])
            fig = go.Figure(go.Bar(x=bars['points'], y=bars['item'], orientation='h',
                                   marker_color=[COL['red'] if p >= 9.75 else COL['amber'] if p >= 9 else COL['grey'] for p in bars['points']]))
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 10], xaxis_title='points (dev percentile ÷ 10)')
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Red: any item ≥ P97.5, or ≥ 2 items ≥ P90, or composite ≥ P90. Amber: any ≥ P90, ≥ 2 ≥ P75, or composite ≥ P75.')

    # ------------------------------------------------------------ tab 6: why
    with tabs[6]:
        st.markdown(f"> {v['summary']}")
        st.markdown(f"**Writings that moved the score most** (at writing {v['decision_i']}). The final model averages the "
                    'meaning of the writings, so its log-odds split exactly into one share per writing.')
        tw = pd.DataFrame([{'#': t['i'], 'day': t['day'], 'push (log-odds)': t['contribution'],
                            'reads like': ', '.join(t['symptoms']) or '—', 'emotions': ', '.join(t['emotions']) or '—',
                            'Qwen ≥ 2': ', '.join(SIGLAB[k] for k, l in t['qwen_levels'].items() if l >= 2) or '—',
                            'writing': t['paraphrase'] or '[text hidden]'} for t in v['top_writings']])
        st.dataframe(tw, hide_index=True, use_container_width=True)
        if 'text_attribution' in v:
            ta = v['text_attribution']
            st.caption(f"{ta['writings_up']} writings pushed up (total {ta['sum_up']:+.2f}), {ta['writings_down']} pushed down "
                       f"(total {ta['sum_down']:+.2f}); baseline {ta['baseline']:+.2f}; log-odds {ta['total']:+.2f}.")
        lf = looks_for()
        if lf:
            st.markdown('**What the text model looks for (whole model, not this user).** Closeness of the model\'s risk '
                        'direction to each plain-language statement (cosine):')
            st.dataframe(pd.DataFrame([{'statement': s, 'cosine': round(c, 3)} for s, c in lf[:6]]), hide_index=True)
        if not nq and 'explanation' in o:
            st.markdown('**Signals model reasons (earlier model, for comparison)**')
            ex = o['explanation']
            st.dataframe(pd.DataFrame([{'quantity': d['label'], 'contribution': d['contribution']} for d in ex['top']]), hide_index=True)

# ---------------------------------------------------------------- tab 7: results
with tabs[7]:
    cmp = json.loads((ROOT / 'research_v11' / 'results' / 'compare_prev_adj.json').read_text())
    rows = []
    for c, d in cmp['candidates'].items():
        rows.append({'model': c + ('' if d.get('eligible', True) else ' (exploratory)'), 'F1 all': d['f1_all'], 'F1 NP': d['f1_np'],
                     'AUROC all': d['auroc_all'], 'AUROC NP': d['auroc_np'], 'AP all': d['ap_all'], 'AP NP': d['ap_np'],
                     'precision all': d['precision_all'], 'recall all': d['recall_all'], 'ERDE50': d['erde50_all']})
    st.markdown('**Development set: 418 users, nested CV (5-fold × 5 repeats), decision after every writing, NP-adjusted threshold**')
    st.dataframe(pd.DataFrame(rows).sort_values('F1 all', ascending=False).round(3), hide_index=True, use_container_width=True)
    st.caption('B = signals model (Qwen + word lists + timing). MPN32 / TXT32 = sentence embeddings (mpnet / bge-large, '
               'PCA 32 fitted inside each fold). EMO = go_emotions. BDI = 21 symptom-statement similarities. ens = average of models.')
    for fname, title in (('compare_prev_adj_ALL.json', '**All 523 users (91 positive), same nested CV, 5 repeats**: the largest test available'),
                         ('compare_prev_adj_BM.json', '**Robustness: only writings before any disclosure, controls cut to the same length** (408 users)')):
        f = ROOT / 'research_v11' / 'results' / fname
        if f.exists():
            cc = json.loads(f.read_text()); st.markdown(title)
            st.dataframe(pd.DataFrame([{'model': c, 'F1 all': d['f1_all'], 'F1 NP': d['f1_np'], 'AUROC all': d['auroc_all'],
                                        'AUROC NP': d['auroc_np'], 'AP all': d['ap_all']} for c, d in cc['candidates'].items()]).round(3),
                         hide_index=True)
            vb = cc.get('vs_B', {}).get('MPN32|all')
            if vb:
                st.caption(f"MPN32 − B, paired bootstrap 95% CI: F1 {vb['f1'][0]:+.3f} [{vb['f1'][1]:+.3f}, {vb['f1'][2]:+.3f}], "
                           f"AP {vb['ap'][0]:+.3f} [{vb['ap'][1]:+.3f}, {vb['ap'][2]:+.3f}]")
    st.markdown(f"**Held-out (105 users, 10 positive), final model, second look**: AUROC {ho['auroc']:.3f}, AP {ho['ap']:.3f}, "
                f"F1 {ho['f1']:.3f} (TP {ho['tp']}, FP {ho['fp']}, FN {ho['fn']}). Earlier signals model on the same users: "
                f"AUROC {m10['heldout']['auroc']:.3f}, F1 {m10['heldout']['f1']:.3f}. With only 10 positives one user moves F1 by ~0.1. "
                'The held-out controls the text model over-scores have short histories (median 50 writings vs 162 for other '
                'controls): averaged meaning is noisy when there are few writings.')
