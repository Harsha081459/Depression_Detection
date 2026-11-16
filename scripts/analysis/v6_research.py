"""v6 research harness: build label-free blocks once, cross-fit text models per
fold once, then run logged trials. Every trial is appended to
private/cohort_v6/trials.csv with its settings, so the paper can report how many
variants were tried.

    python scripts/analysis/v6_research.py build   --root .
    python scripts/analysis/v6_research.py textfit --root . --day 28
    python scripts/analysis/v6_research.py trials  --root . --day 28

Protocol: 5x5 RepeatedStratifiedKFold over all 523 users, seed 42. Training
uses every training-fold user. Scores are pooled per repeat; AUROC is reported
on all users and on the 280 users never in the pilot. The 105-user holdout is
NOT used here. No rescoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_temporal_features as v2
import v5_features as v5
import v6_features as v6
from run_v2 import inner_folds, log, logistic_search

SEED = 42
DAYS = (14, 28, 84)
TEXT_MODELS = ('word', 'char', 'nb_word')


def paths(root: Path, out: Path):
    return {'cache': out / 'cache', 'folds': out / 'folds', 'trials': out / 'trials.csv'}


# --------------------------------------------------------------------------- #
def build(root: Path, out: Path):
    P = paths(root, out)
    P['cache'].mkdir(parents=True, exist_ok=True)
    writings, _, users = v2.load_writings(root)
    ids = [str(u) for u in users.user_id]
    users[['user_id', 'label', 'pilot_exposed']].to_parquet(P['cache'] / 'users.parquet', index=False)
    posts = pd.read_parquet(root / 'private' / 'cohort_v1' / 'eligible_posts.parquet', columns=['post_id', 'text'])
    texts = writings[['user_id', 'post_id', 'delta_days']].merge(posts, on='post_id', validate='one_to_one')
    texts['raw'] = texts.text.fillna('').astype(str)
    texts['clean'] = texts.raw.map(v6.strict_clean)
    texts = texts.drop(columns='text')
    log('texts cleaned: %d writings' % len(texts))

    table = v6.post_table(texts)
    table.to_parquet(P['cache'] / 'post_table.parquet', index=False)
    log('post lexicon table done')

    # external weak supervision
    norm = texts.raw.map(lambda t: re.sub(r'\W+', ' ', t.lower()).strip()[:300])
    erisk_hashes = {hashlib.md5(n.encode()).hexdigest() for n in norm if len(n) >= 60}
    ext_path = root / 'data' / 'raw' / 'hf_solomonk' / 'solomonk_posts.csv'
    h, tfidf, model, ext_info = v6.train_external(ext_path, erisk_hashes)
    texts['ext'] = model.predict_proba(tfidf.transform(h.transform(texts.clean.tolist())))[:, 1]
    (P['cache'] / 'external_info.json').write_text(json.dumps(ext_info, indent=2))
    log('external model: %s' % ext_info)

    try:
        import spacy
        nlp = spacy.load('en_core_web_lg', exclude=['tagger', 'parser', 'ner', 'lemmatizer',
                                                    'attribute_ruler', 'senter'])
    except Exception as exc:  # recorded, not fatal
        nlp = None
        log('spaCy vectors unavailable: %s' % exc)

    v2bank = v2.raw_bank(v2.histories(writings, users), 'd14')
    for day in DAYS:
        blocks = [v2.rows_at(v2bank, day // 14, ids)[list(v2.V1_STATIC_COLUMNS)].astype(float),
                  v5.tail_rows(writings, day, ids),
                  v6.user_lexicon_rows(table, day, ids),
                  v6.user_external_rows(texts[['user_id', 'delta_days', 'ext']], day, ids),
                  v6.user_empath_rows(texts, day, ids, v6.EMPATH_SUBSET)]
        frame = pd.concat(blocks, axis=1)
        frame.index.name = 'user_id'
        frame.to_parquet(P['cache'] / f'blocks_day{day}.parquet')
        full = v6.user_empath_rows(texts, day, ids, None)
        full.columns = [c.replace('empath__', 'empathall__') for c in full.columns]
        full.to_parquet(P['cache'] / f'empath_full_day{day}.parquet')
        if nlp is not None:
            v6.user_vector_rows(texts, day, ids, nlp).to_parquet(P['cache'] / f'vectors_day{day}.parquet')
        for analyzer in ('word', 'char'):
            sparse.save_npz(P['cache'] / f'doc_{analyzer}_day{day}.npz',
                            v6.doc_matrix(texts, day, ids, analyzer))
        log('day %d blocks done' % day)


# --------------------------------------------------------------------------- #
DEV_ONLY = True   # trials select on the 418 split.json training users; the 105 stay locked


def dev_ids(root: Path) -> set:
    rows = json.loads((root / 'private' / 'cohort_v1' / 'split.json').read_text())['rows']
    return {str(r['user_id']) for r in rows if r['split'] == 'train'}


def load_users(out: Path, root: Path | None = None, dev_only: bool = DEV_ONLY):
    users = pd.read_parquet(paths(None, out)['cache'] / 'users.parquet')
    if dev_only:
        keep = dev_ids(root or Path(__file__).resolve().parents[2])
        users = users[users.user_id.astype(str).isin(keep)]
    ids = [str(u) for u in users.user_id]
    y = users.set_index('user_id').label.astype(int).loc[ids]
    pilot = users.set_index('user_id').pilot_exposed.loc[ids].to_numpy(bool)
    return ids, y, pilot


def partitions(y):
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
                .split(np.zeros(len(y)), y))


def textfit(out: Path, day: int, root: Path):
    """Cross-fitted document-model scores for every outer fold, cached."""
    P = paths(None, out)
    P['folds'].mkdir(parents=True, exist_ok=True)
    ids, y, _ = load_users(out, root)
    all_ids = [str(u) for u in pd.read_parquet(P['cache'] / 'users.parquet').user_id]
    rows_all = {u: i for i, u in enumerate(all_ids)}
    pick = np.array([rows_all[u] for u in ids])
    mats = {'word': sparse.load_npz(P['cache'] / f'doc_word_day{day}.npz')[pick],
            'char': sparse.load_npz(P['cache'] / f'doc_char_day{day}.npz')[pick]}
    mats['nb_word'] = mats['word']
    started = time.time()
    records = []
    for pos, (tr, va) in enumerate(partitions(y)):
        row = {'fold_index': pos}
        for name in TEXT_MODELS:
            kind = 'nb' if name.startswith('nb') else 'tfidf'
            oof, valid, C = v6.crossfit_doc(kind, mats[name], tr, va, y.iloc[tr])
            row[name] = {'train': oof.tolist(), 'valid': valid.tolist(), 'C': C}
        records.append(row)
        log('textfit day %d fold %d/25 (%.0fs)' % (day, pos + 1, time.time() - started))
    (P['folds'] / f'text_scores_day{day}.json').write_text(json.dumps(records))


# --------------------------------------------------------------------------- #
def evaluate(scores: pd.DataFrame, pilot_by_user: pd.Series, fold_rank: bool = True) -> dict:
    """scores: repeat, fold, user_id, label, score. Pooled per repeat.

    fold_rank=True (primary since round 2): each fold's test scores are turned into
    percentile ranks before pooling, so a model is not penalised or rewarded for
    score-scale differences between the five refits. Label-free."""
    scores = scores.copy()
    per_fold = [roc_auc_score(g.label, g.score) for _, g in scores.groupby(['repeat', 'fold'])]
    if fold_rank:
        scores['score'] = scores.groupby(['repeat', 'fold']).score.rank(pct=True)
    res = {'all': [], 'np': [], 'all_ap': [], 'np_ap': []}
    for _, g in scores.groupby('repeat'):
        npm = ~g.user_id.map(pilot_by_user).to_numpy(bool)
        res['all'].append(roc_auc_score(g.label, g.score))
        res['np'].append(roc_auc_score(g.label[npm], g.score[npm]))
        res['all_ap'].append(average_precision_score(g.label, g.score))
        res['np_ap'].append(average_precision_score(g.label[npm], g.score[npm]))
    return {'auroc_all': float(np.mean(res['all'])), 'sd_all': float(np.std(res['all'], ddof=1)),
            'auroc_np': float(np.mean(res['np'])), 'sd_np': float(np.std(res['np'], ddof=1)),
            'auprc_all': float(np.mean(res['all_ap'])), 'auprc_np': float(np.mean(res['np_ap'])),
            'np_by_repeat': [round(v, 4) for v in res['np']],
            'per_fold_auroc_all': float(np.mean(per_fold))}


class Trials:
    def __init__(self, out: Path, day: int, root: Path):
        self.out, self.day = out, day
        P = paths(None, out)
        self.ids, self.y, pilot = load_users(out, root)
        self.pilot_by_user = pd.Series(pilot, index=self.ids)
        self.parts = partitions(self.y)
        self.blocks = pd.read_parquet(P['cache'] / f'blocks_day{day}.parquet').loc[self.ids]
        extra = [P['cache'] / f'empath_full_day{day}.parquet', P['cache'] / f'vectors_day{day}.parquet']
        for e in extra:
            if e.exists():
                self.blocks = pd.concat([self.blocks, pd.read_parquet(e).loc[self.ids]], axis=1)
        tpath = P['folds'] / f'text_scores_day{day}.json'
        self.text = json.loads(tpath.read_text()) if tpath.exists() else None
        self.log_path = P['trials']
        self.preds = {}

    def columns(self, prefixes):
        return [c for c in self.blocks.columns if c.startswith(tuple(prefixes))]

    def run(self, name: str, block_prefixes=(), text_models=(), combiner='logistic', note='',
            save=False, block_groups=()):
        cols = self.columns(block_prefixes) if block_prefixes else []
        if combiner == 'rankmean_models':
            cols = []
        rows = []
        for pos, (tr, va) in enumerate(self.parts):
            Xtr = self.blocks.iloc[tr][cols].copy() if cols else pd.DataFrame(index=[self.ids[i] for i in tr])
            Xva = self.blocks.iloc[va][cols].copy() if cols else pd.DataFrame(index=[self.ids[i] for i in va])
            for m in text_models:
                # Rank within each side: inner out-of-fold scores (train) and refit
                # scores (test) come from different models and differ in scale.
                Xtr[f'text__{m}'] = pd.Series(self.text[pos][m]['train']).rank(pct=True).to_numpy()
                Xva[f'text__{m}'] = pd.Series(self.text[pos][m]['valid']).rank(pct=True).to_numpy()
            ytr = self.y.iloc[tr]
            if combiner == 'rankmean_models':
                # each block group is its own in-fold logistic; then mean of percentile ranks
                ranks = [Xva[f'text__{m}'].to_numpy() for m in text_models]
                for group in block_groups:
                    gcols = self.columns(group)
                    fit = logistic_search(inner_folds(ytr)).fit(self.blocks.iloc[tr][gcols], ytr)
                    ranks.append(pd.Series(fit.predict_proba(self.blocks.iloc[va][gcols])[:, 1]).rank(pct=True).to_numpy())
                s = np.mean(ranks, axis=0)
            elif combiner == 'score' and Xtr.shape[1] == 1:
                s = Xva.iloc[:, 0].to_numpy()
            elif combiner == 'rankmean':
                s = Xva.rank(pct=True).mean(axis=1).to_numpy()
            else:
                s = logistic_search(inner_folds(ytr)).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            rows += [{'repeat': pos // 5, 'fold': pos % 5, 'user_id': self.ids[i], 'label': int(self.y.iloc[i]),
                      'score': float(v)} for i, v in zip(va, s)]
        scores = pd.DataFrame(rows)
        result = evaluate(scores, self.pilot_by_user, fold_rank=True)
        raw = evaluate(scores, self.pilot_by_user, fold_rank=False)
        result['raw_pooled_auroc_all'], result['raw_pooled_auroc_np'] = raw['auroc_all'], raw['auroc_np']
        record = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'day': self.day,
                  'trial': name, 'blocks': '|'.join(block_prefixes) or ';'.join('|'.join(g) for g in block_groups), 'text_models': '|'.join(text_models),
                  'combiner': combiner, 'n_features': len(cols) + len(text_models) + sum(len(self.columns(g)) for g in block_groups), 'note': note, **result}
        frame = pd.DataFrame([record])
        frame.to_csv(self.log_path, mode='a', header=not self.log_path.exists(), index=False)
        if save:
            self.preds[name] = scores
        print('%-44s all %.3f±%.3f | NP %.3f±%.3f | AP %.3f/%.3f | per-fold %.3f | raw %.3f/%.3f' % (
            name, result['auroc_all'], result['sd_all'], result['auroc_np'], result['sd_np'],
            result['auprc_all'], result['auprc_np'], result['per_fold_auroc_all'],
            result['raw_pooled_auroc_all'], result['raw_pooled_auroc_np']), flush=True)
        return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'textfit'])
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--out', default=None)
    ap.add_argument('--day', type=int, default=28)
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / 'private' / 'cohort_v6'
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'build':
        build(root, out)
    else:
        textfit(out, a.day, root)


if __name__ == '__main__':
    main()
