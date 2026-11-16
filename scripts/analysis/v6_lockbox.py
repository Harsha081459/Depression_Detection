"""v6 lockbox: train the selected model on the 418 development users, score the
105 split.json test users ONCE. Run only after the selection is frozen in
diagnostics_v6/final_model_v6.json. Refuses to run twice.

    python scripts/analysis/v6_lockbox.py --root . --out private/cohort_v6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_temporal_features as v2
import v6_features as v6
from erisk_full_pipeline import bootstrap
from run_v2 import inner_folds, logistic_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / 'private' / 'cohort_v6'
    final = json.loads((root / 'diagnostics_v6' / 'final_model_v6.json').read_text())
    result_path = out / 'lockbox_result.json'
    if result_path.exists():
        raise SystemExit('lockbox already opened; result is in %s' % result_path)

    rows = json.loads((root / 'private' / 'cohort_v1' / 'split.json').read_text())['rows']
    split = {str(r['user_id']): r['split'] for r in rows}
    users = pd.read_parquet(out / 'cache' / 'users.parquet')
    all_ids = [str(u) for u in users.user_id]
    labels = users.set_index('user_id').label.astype(int)
    dev = [u for u in all_ids if split[u] == 'train']
    test = [u for u in all_ids if split[u] == 'test']
    pos = {u: i for i, u in enumerate(all_ids)}
    idev, itest = np.array([pos[u] for u in dev]), np.array([pos[u] for u in test])
    ydev, ytest = labels.loc[dev].to_numpy(), labels.loc[test].to_numpy()

    report = {'opened_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'final_model': final, 'dev_users': len(dev), 'dev_positive': int(ydev.sum()),
              'test_users': len(test), 'test_positive': int(ytest.sum()), 'days': {}}
    for day in final['days']:
        mats = {'word': sparse.load_npz(out / 'cache' / f'doc_word_day{day}.npz'),
                'char': sparse.load_npz(out / 'cache' / f'doc_char_day{day}.npz')}
        mats['nb_word'] = mats['word']
        ranks = []
        for name in final['text_models']:
            kind = 'nb' if name.startswith('nb') else 'tfidf'
            X = mats[name]
            C = v6.tune_C(kind, X[idev], ydev)
            s = v6.make_model(kind, C).fit(X[idev], ydev).score(X[itest])
            ranks.append(pd.Series(s).rank(pct=True).to_numpy())
        score = np.mean(ranks, axis=0)
        blocks = pd.read_parquet(out / 'cache' / f'blocks_day{day}.parquet')
        v1cols = [c for c in blocks.columns if c.startswith('v1__')]
        ref = logistic_search(inner_folds(pd.Series(ydev))).fit(blocks.loc[dev, v1cols], ydev)
        ref_score = ref.predict_proba(blocks.loc[test, v1cols])[:, 1]
        boot = bootstrap(ytest, score, other=ref_score, n=2000)
        single = bootstrap(ytest, score, n=2000)
        k = int(round(0.2 * len(test)))
        top = np.argsort(-score)[:k]
        top_ref = np.argsort(-ref_score)[:k]
        report['days'][str(day)] = {
            'auroc_final': float(roc_auc_score(ytest, score)),
            'auroc_final_ci': [single['ci_low'], single['ci_high']],
            'auprc_final': float(average_precision_score(ytest, score)),
            'auroc_v1_static': float(roc_auc_score(ytest, ref_score)),
            'auprc_v1_static': float(average_precision_score(ytest, ref_score)),
            'paired_diff_final_minus_v1': boot,
            'flag_top20pct': {'flags': k, 'recall_final': float(ytest[top].sum() / ytest.sum()),
                              'precision_final': float(ytest[top].mean()),
                              'recall_v1': float(ytest[top_ref].sum() / ytest.sum()),
                              'precision_v1': float(ytest[top_ref].mean())}}
    result_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['days'], indent=2))


if __name__ == '__main__':
    main()
