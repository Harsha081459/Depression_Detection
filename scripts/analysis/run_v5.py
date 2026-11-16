"""Run v5 (diagnostics_v5/prespec_v5.json). Exploratory.

    python scripts/analysis/run_v5.py

No GPU, no rescoring, no holdout, private/cohort_v1/evaluation/ not opened.
Outputs under private/cohort_v5/. Flags are for human review, not diagnosis.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_temporal_features as v2
import v5_features as v5
from erisk_full_pipeline import bootstrap, sha_file
from run_v2 import inner_folds, log, logistic_search

SEED = 42


def load(root: Path):
    writings, _, users = v2.load_writings(root)
    posts = pd.read_parquet(root / 'private' / 'cohort_v1' / 'eligible_posts.parquet', columns=['post_id', 'text'])
    texts = writings[['user_id', 'post_id', 'delta_days']].merge(posts, on='post_id', validate='one_to_one')
    texts['text_clin'] = texts.text.fillna('').astype(str)
    texts['text_noclin'] = texts.text_clin.map(v5.strip_clinical)
    texts = texts.drop(columns='text')
    return writings, users, texts


def threshold(train_scores, y_train):
    return float(np.quantile(np.asarray(train_scores, float), 1 - float(np.mean(y_train))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--out', default=None)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'private' / 'cohort_v5'
    out.mkdir(parents=True, exist_ok=True)
    spec_path = root / 'diagnostics_v5' / 'prespec_v5.json'
    spec = json.loads(spec_path.read_text())
    if spec.get('status') != 'locked':
        raise SystemExit('prespec_v5.json is not locked')

    log('loading')
    writings, users, texts = load(root)
    ids = [str(u) for u in users.user_id]
    y = users.set_index('user_id').label.astype(int).loc[ids]
    pilot = set(users.loc[users.pilot_exposed, 'user_id'].astype(str))
    histories = v2.histories(writings, users)
    v2bank = v2.raw_bank(histories, 'd14')

    static, textdays = {}, {}
    for day in v5.DECISION_DAYS:
        v1 = v2.rows_at(v2bank, day // 14, ids)[list(v2.V1_STATIC_COLUMNS)].astype(float)
        tail = v5.tail_rows(writings, day, ids)
        static[day] = pd.concat([v1, tail], axis=1)
        textdays[day] = v5.TextDay(texts, day, ids)
        log('day %d features ready' % day)

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    parts = list(outer.split(np.zeros(len(y)), y))[:5 * args.repeats]
    preds, chosen = [], []
    started = time.time()
    for position, (train, valid) in enumerate(parts):
        repeat, fold = divmod(position, 5)
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        for day in v5.DECISION_DAYS:
            t_tr, t_va, C = v5.crossfit_text(textdays[day], tr, va, ytr)
            chosen.append({'repeat': repeat, 'fold': fold, 'decision_day': day,
                           'doc_C_noclin': C['noclin'], 'doc_C_clin': C['clin']})
            full_tr = pd.concat([static[day].loc[tr], t_tr], axis=1)
            full_va = pd.concat([static[day].loc[va], t_va], axis=1)
            for name, cols in v5.FEATURE_SETS.items():
                if name in v5.SCORE_ONLY:
                    s_tr = full_tr[v5.SCORE_ONLY[name]].to_numpy()
                    s_va = full_va[v5.SCORE_ONLY[name]].to_numpy()
                else:
                    search = logistic_search(inner_folds(ytr)).fit(full_tr[cols], ytr)
                    s_tr = search.predict_proba(full_tr[cols])[:, 1]
                    s_va = search.predict_proba(full_va[cols])[:, 1]
                cut = threshold(s_tr, ytr)
                preds.extend({'repeat': repeat, 'fold': fold, 'decision_day': day, 'feature_set': name,
                              'user_id': u, 'label': int(l), 'pilot': u in pilot, 'score': float(s),
                              'flagged': bool(s >= cut)} for u, l, s in zip(va, yva, s_va))
        log('outer %d/%d done (%.0fs)' % (position + 1, len(parts), time.time() - started))

    frame = pd.DataFrame(preds)
    frame.to_parquet(out / 'oof_predictions_PRIVATE.parquet', index=False)
    pd.DataFrame(chosen).to_csv(out / 'text_C_chosen.csv', index=False)

    rows = []
    for (repeat, day, name), g in frame.groupby(['repeat', 'decision_day', 'feature_set']):
        for population, sub in (('all_users', g), ('non_pilot', g[~g.pilot])):
            flagged = sub[sub.flagged]
            rows.append({'repeat': repeat, 'decision_day': day, 'feature_set': name, 'population': population,
                         'users': len(sub), 'positives': int(sub.label.sum()),
                         'auroc': roc_auc_score(sub.label, sub.score),
                         'auprc': average_precision_score(sub.label, sub.score),
                         'recall': float(sub[sub.label == 1].flagged.mean()),
                         'precision': float(flagged.label.mean()) if len(flagged) else float('nan'),
                         'flags_per_100': 100 * float(sub.flagged.mean())})
    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(out / 'metrics_per_repeat.csv', index=False)
    headline = per_repeat.groupby(['population', 'decision_day', 'feature_set']).agg(
        mean_auroc=('auroc', 'mean'), sd_auroc=('auroc', 'std'), mean_auprc=('auprc', 'mean'),
        sd_auprc=('auprc', 'std'), mean_recall=('recall', 'mean'), mean_precision=('precision', 'mean'),
        flags_per_100=('flags_per_100', 'mean'), repeats=('auroc', 'size')).reset_index()
    headline.to_csv(out / 'headline_v5.csv', index=False)

    def paired(population, day, a='v5_stack', b='v1_static'):
        g = frame[(frame.repeat == 0) & (frame.decision_day == day)]
        if population == 'non_pilot':
            g = g[~g.pilot]
        w = g.pivot_table(index=['user_id', 'label'], columns='feature_set', values='score').reset_index()
        r = bootstrap(w.label.to_numpy(int), w[a].to_numpy(float), other=w[b].to_numpy(float), n=2000)
        r.update({'comparison': f'{a} minus {b}', 'population': population, 'decision_day': day,
                  'users': int(len(w)), 'positives': int(w.label.sum()),
                  f'auroc_{a}': float(roc_auc_score(w.label, w[a])),
                  f'auroc_{b}': float(roc_auc_score(w.label, w[b]))})
        return r

    primary = paired('non_pilot', 28)
    primary['success'] = bool(primary['estimate'] > 0.02 and primary['ci_low'] > 0)
    secondary = [paired(p, d) for p, d in (('all_users', 28), ('non_pilot', 84), ('all_users', 84),
                                            ('non_pilot', 14), ('all_users', 14))]
    secondary += [paired('non_pilot', 28, 'v5_stack', 'style_llm_tail'),
                  paired('non_pilot', 28, 'style_llm_tail', 'v1_static'),
                  paired('all_users', 28, 'style_llm_tail', 'v1_static')]
    (out / 'bootstrap_v5.json').write_text(json.dumps({'primary': primary, 'secondary': secondary}, indent=2))
    (out / 'run_v5.json').write_text(json.dumps({
        'seed': SEED, 'ran_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'exploratory': True,
        'repeats': args.repeats, 'output_is': 'a flag for a person to review, not a diagnosis',
        'rescored': False, 'holdout_105_used': False, 'cohort_v1_evaluation_opened': False,
        'prespec_sha256': sha_file(spec_path), 'features_sha256': sha_file(Path(__file__).with_name('v5_features.py')),
        'runner_sha256': sha_file(Path(__file__)), 'primary_endpoint': primary,
        'packages': {n: version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'scipy')},
        'python': platform.python_version()}, indent=2))
    log('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
