"""Run the v4 final pass described in diagnostics_v4/prespec_v4.json.

    python scripts/analysis/run_v4.py

Exploratory. No GPU, no rescoring, no holdout, private/cohort_v1/evaluation/
not opened. Outputs under private/cohort_v4/. A flag is for human review, not a
diagnosis; labels are eRisk dataset task labels.

Run the unit tests first:
    python -m unittest discover -s scripts/analysis -p 'test_v4_features.py'
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_temporal_features as v2
import v4_features as v4
from erisk_full_pipeline import bootstrap, sha_file
from run_v2 import flag_metrics, inner_folds, log, logistic_search

SEED = 42
POPULATIONS = ('all_users', 'without_pilot')
MODELS = ('logistic', 'xgboost')
PRIMARY = ('without_pilot', 28, 'logistic')
SECONDARY = [('all_users', 28, 'logistic'), ('without_pilot', 84, 'logistic'),
             ('all_users', 84, 'logistic'), ('without_pilot', 28, 'xgboost'),
             ('all_users', 28, 'xgboost'), ('without_pilot', 84, 'xgboost'),
             ('all_users', 84, 'xgboost')]


def xgboost_model():
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=20,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, tree_method='hist',
                         eval_metric='logloss', n_jobs=2, random_state=SEED)


def fit(model, Xtr, ytr):
    if model == 'logistic':
        return logistic_search(inner_folds(ytr)).fit(Xtr, ytr), None
    return xgboost_model().fit(Xtr.to_numpy(float), ytr.to_numpy(int)), 'array'


def score(fitted, kind, X):
    data = X.to_numpy(float) if kind == 'array' else X
    return fitted.predict_proba(data)[:, 1]


def partitions(y):
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
                .split(np.zeros(len(y)), y))


def evaluate_population(bank, labels, population, out):
    y = labels.astype(int)
    ids = list(y.index)
    metrics, predictions, flags = [], [], []
    started = time.time()
    for position, (train, valid) in enumerate(partitions(y)):
        repeat, fold = divmod(position, 5)
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        for day in v4.DECISION_DAYS:
            for feature_set in v4.FEATURE_SETS:
                Xtr, Xva = bank.fold(tr, va, day, feature_set)
                for model in MODELS:
                    fitted, kind = fit(model, Xtr, ytr)
                    s_va, s_tr = score(fitted, kind, Xva), score(fitted, kind, Xtr)
                    base = {'population': population, 'repeat': repeat, 'fold': fold,
                            'decision_day': day, 'feature_set': feature_set, 'model': model}
                    params = json.dumps(fitted.best_params_) if model == 'logistic' else 'fixed'
                    metrics.append(dict(base, n_features=Xtr.shape[1], parameters=params,
                                        auroc=float(roc_auc_score(yva, s_va)),
                                        auprc=float(average_precision_score(yva, s_va))))
                    flags.append(dict(base, **flag_metrics(s_tr, ytr, s_va, yva)))
                    predictions.extend(dict(base, user_id=u, label=int(l), score=float(s))
                                       for u, l, s in zip(va, yva, s_va))
        log('%s outer %d/25 done (%.0fs)' % (population, position + 1, time.time() - started))
    frame = pd.DataFrame(metrics)
    frame.to_csv(out / ('fold_metrics_%s.csv' % population), index=False)
    pd.DataFrame(flags).to_csv(out / ('flag_metrics_folds_%s.csv' % population), index=False)
    keys = ['population', 'decision_day', 'feature_set', 'model']
    summary = frame.groupby(keys).agg(mean_auroc=('auroc', 'mean'), fold_sd_auroc=('auroc', 'std'),
                                      mean_auprc=('auprc', 'mean'), fold_sd_auprc=('auprc', 'std'),
                                      folds=('auroc', 'size')).reset_index()
    summary['users'], summary['positive_users'] = len(y), int(y.sum())
    summary.to_csv(out / ('headline_%s.csv' % population), index=False)
    pd.DataFrame(flags).groupby(keys).agg(
        mean_recall=('recall', 'mean'), mean_precision=('precision', 'mean'),
        mean_flags_per_100_users=('flags_per_100_users', 'mean'),
        mean_flag_rate=('flag_rate', 'mean')).reset_index().to_csv(
        out / ('flag_summary_%s.csv' % population), index=False)
    pd.DataFrame(predictions).to_parquet(out / ('oof_predictions_%s_PRIVATE.parquet' % population),
                                         index=False)


def time_to_flag(bank, labels, population, out):
    y = labels.astype(int)
    ids = list(y.index)
    records = []
    for position, (train, valid) in enumerate(partitions(y)[:5]):
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        for feature_set in ('v1_static', 'v4_behaviour'):
            for day in v4.DECISION_DAYS:
                Xtr, Xva = bank.fold(tr, va, day, feature_set)
                fitted, kind = fit('logistic', Xtr, ytr)
                s_va = score(fitted, kind, Xva)
                threshold = flag_metrics(score(fitted, kind, Xtr), ytr, s_va, yva)['threshold']
                records.extend({'population': population, 'fold': position, 'feature_set': feature_set,
                                'decision_day': day, 'user_id': u, 'label': int(l),
                                'flagged': bool(s >= threshold)} for u, l, s in zip(va, yva, s_va))
    frame = pd.DataFrame(records)
    frame.to_parquet(out / ('time_to_flag_folds_%s_PRIVATE.parquet' % population), index=False)
    rows = []
    for feature_set, group in frame.groupby('feature_set'):
        pos = group[group.label == 1]
        first = pos[pos.flagged].groupby('user_id').decision_day.min()
        n = pos.user_id.nunique()
        rows.append({'population': population, 'feature_set': feature_set, 'positive_users': n,
                     'ever_flagged': int(first.size), 'never_flagged_share': 1 - first.size / max(1, n),
                     'flagged_by_day_14': int((first <= 14).sum()),
                     'flagged_by_day_28': int((first <= 28).sum()),
                     'median_days_to_first_flag': float(first.median()) if first.size else float('nan')})
    pd.DataFrame(rows).to_csv(out / ('time_to_flag_%s.csv' % population), index=False)


def paired(out, population, day, model):
    frame = pd.read_parquet(out / ('oof_predictions_%s_PRIVATE.parquet' % population))
    frame = frame[(frame.repeat == 0) & (frame.decision_day == day) & (frame.model == model)]
    wide = frame.pivot_table(index=['user_id', 'label'], columns='feature_set', values='score').reset_index()
    result = bootstrap(wide.label.to_numpy(int), wide.v4_behaviour.to_numpy(float),
                       other=wide.v1_static.to_numpy(float), n=2000)
    result.update({'comparison': 'v4_behaviour minus v1_static', 'population': population,
                   'decision_day': day, 'model': model, 'repeat': 0, 'resamples': 2000, 'seed': SEED,
                   'users': int(len(wide)), 'positive_users': int(wide.label.sum()),
                   'auroc_v4_behaviour': float(roc_auc_score(wide.label, wide.v4_behaviour)),
                   'auroc_v1_static': float(roc_auc_score(wide.label, wide.v1_static))})
    return result


def drop_one(bank, labels, out):
    y = labels.astype(int)
    ids = list(y.index)
    base_cols = v4.FEATURE_SETS['v4_all']
    arms = {'v4_all': base_cols}
    arms.update({'without_' + g: [c for c in base_cols if c not in set(cols)]
                 for g, cols in v4.DROP_ONE_GROUPS.items()})
    rows = []
    for position, (train, valid) in enumerate(partitions(y)):
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        Xtr_all, Xva_all = bank.fold(tr, va, 28, 'v4_all')
        for arm, cols in arms.items():
            fitted = logistic_search(inner_folds(ytr)).fit(Xtr_all[cols], ytr)
            rows.append({'arm': arm, 'repeat': position // 5, 'fold': position % 5, 'n_features': len(cols),
                         'auroc': float(roc_auc_score(yva, fitted.predict_proba(Xva_all[cols])[:, 1]))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'sensitivity_drop_one_folds.csv', index=False)
    summary = frame.groupby('arm').agg(mean_auroc=('auroc', 'mean'), fold_sd_auroc=('auroc', 'std'),
                                       n_features=('n_features', 'first')).reset_index()
    base = float(summary.loc[summary.arm == 'v4_all', 'mean_auroc'].iloc[0])
    summary['auroc_drop_from_v4_all'] = base - summary.mean_auroc
    summary['above_noise_0.02'] = summary.auroc_drop_from_v4_all.abs() > 0.02
    summary.to_csv(out / 'sensitivity_drop_one.csv', index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--out', default=None, help='default: <root>/private/cohort_v4')
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'private' / 'cohort_v4'
    out.mkdir(parents=True, exist_ok=True)
    prespec_path = root / 'diagnostics_v4' / 'prespec_v4.json'
    prespec = json.loads(prespec_path.read_text(encoding='utf-8'))
    if prespec.get('status') != 'locked' or not prespec.get('written_before_any_v4_metric'):
        raise SystemExit('prespec_v4.json is not the locked file')

    log('loading')
    writings, _, users = v2.load_writings(root)
    histories = v2.histories(writings, users)
    labels = users.set_index('user_id').label.astype(int)
    log('building feature bank')
    bank = v4.Bank(root, writings, histories, users)
    pd.DataFrame(v4.feature_table_rows()).to_csv(out / 'feature_table.csv', index=False)

    for population in POPULATIONS:
        subset = labels if population == 'all_users' else labels.loc[
            list(users.loc[~users.pilot_exposed, 'user_id'])]
        log('population %s: %d users, %d positive' % (population, len(subset), int(subset.sum())))
        evaluate_population(bank, subset, population, out)
        time_to_flag(bank, subset, population, out)

    log('paired bootstraps')
    primary = paired(out, *PRIMARY)
    primary['success_rule'] = 'estimate > 0.02 and ci_low > 0'
    primary['success'] = bool(primary['estimate'] > 0.02 and primary['ci_low'] > 0)
    boot = {'primary': primary, 'secondary': [paired(out, *s) for s in SECONDARY]}
    (out / 'bootstrap_v4_behaviour_minus_v1_static.json').write_text(json.dumps(boot, indent=2))
    log('drop-one sensitivity')
    drop_one(bank, labels, out)

    headline = pd.concat([pd.read_csv(out / ('headline_%s.csv' % p)) for p in POPULATIONS])
    headline.to_csv(out / 'headline_v4.csv', index=False)
    (out / 'run_v4.json').write_text(json.dumps({
        'seed': SEED, 'ran_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'exploratory': True, 'output_is': 'a flag for a person to review, not a diagnosis',
        'labels_are': 'eRisk dataset task labels, not a clinical diagnosis',
        'rescored': False, 'holdout_105_used': False, 'cohort_v1_evaluation_opened': False,
        'prespec_written_at_utc': prespec['written_at_utc'], 'prespec_sha256': sha_file(prespec_path),
        'features_sha256': sha_file(Path(__file__).with_name('v4_features.py')),
        'runner_sha256': sha_file(Path(__file__)),
        'writings_scored': int(len(writings)), 'users': int(len(users)),
        'positive_users': int(users.label.sum()), 'pilot_users': int(users.pilot_exposed.sum()),
        'primary_endpoint': primary, 'headline_rows': int(len(headline)),
        'packages': {n: version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'pyarrow', 'xgboost')},
        'python': platform.python_version(),
    }, indent=2))
    log('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
