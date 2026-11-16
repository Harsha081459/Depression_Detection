"""Run the v3 cross-validation described in diagnostics_v3/prespec_v3.json.

    python scripts/analysis/run_v3.py

Exploratory: v3 was designed after the v2 results were seen. No GPU, no
rescoring, no holdout, and private/cohort_v1/evaluation/ is not opened.
Outputs go under private/cohort_v3/. A flag is for human review, not a
diagnosis; labels are eRisk dataset task labels.

Run the unit tests first:
    python -m unittest discover -s scripts/analysis -p 'test_v3_features.py'
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
import v3_features as v3
from erisk_full_pipeline import bootstrap, sha_file
from run_v2 import boosting_search, flag_metrics, inner_folds, log, logistic_search

SEED = 42
BOOSTING_SETS = ('v1_static', 'v1_plus_change')
BOOSTING_DAYS = (28, 84)
POPULATIONS = ('all_users', 'without_pilot')
PRIMARY = ('without_pilot', 28)
SECONDARY = (('all_users', 28), ('without_pilot', 84), ('all_users', 84))


def outer_partitions(y):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    return list(outer.split(np.zeros(len(y)), y))


def evaluate_population(stats, labels, population, out):
    y = labels.astype(int)
    ids = list(y.index)
    metrics, predictions, flags = [], [], []
    started = time.time()
    for position, (train, valid) in enumerate(outer_partitions(y)):
        repeat, fold = divmod(position, 5)
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        for day, chunk in v3.DECISION_CHUNKS.items():
            for feature_set in v3.FEATURE_SETS:
                Xtr, Xva, _ = v3.fold_matrix(stats, tr, va, chunk, feature_set)
                models = [('logistic', logistic_search)]
                if feature_set in BOOSTING_SETS and day in BOOSTING_DAYS:
                    models.append(('boosting', boosting_search))
                for name, builder in models:
                    search = builder(inner_folds(ytr)).fit(Xtr, ytr)
                    s_va = search.predict_proba(Xva)[:, 1]
                    s_tr = search.predict_proba(Xtr)[:, 1]
                    base = {'population': population, 'repeat': repeat, 'fold': fold,
                            'decision_day': day, 'feature_set': feature_set, 'model': name}
                    metrics.append(dict(base, n_features=Xtr.shape[1],
                                        auroc=float(roc_auc_score(yva, s_va)),
                                        auprc=float(average_precision_score(yva, s_va)),
                                        parameters=json.dumps(search.best_params_, sort_keys=True)))
                    flags.append(dict(base, test_users=len(va), test_positives=int(yva.sum()),
                                      **flag_metrics(s_tr, ytr, s_va, yva)))
                    if name == 'logistic':
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
    return summary


def time_to_flag(stats, labels, population, out):
    y = labels.astype(int)
    ids = list(y.index)
    records = []
    for position, (train, valid) in enumerate(outer_partitions(y)[:5]):
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        for feature_set in BOOSTING_SETS:
            for day, chunk in v3.DECISION_CHUNKS.items():
                Xtr, Xva, _ = v3.fold_matrix(stats, tr, va, chunk, feature_set)
                search = logistic_search(inner_folds(ytr)).fit(Xtr, ytr)
                s_va = search.predict_proba(Xva)[:, 1]
                threshold = flag_metrics(search.predict_proba(Xtr)[:, 1], ytr, s_va, yva)['threshold']
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
        rows.append({'population': population, 'feature_set': feature_set, 'model': 'logistic',
                     'positive_users': n, 'ever_flagged': int(first.size),
                     'never_flagged_share': 1 - first.size / max(1, n),
                     'median_days_to_first_flag': float(first.median()) if first.size else float('nan'),
                     'flagged_by_day_14': int((first <= 14).sum()),
                     'flagged_by_day_28': int((first <= 28).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / ('time_to_flag_%s.csv' % population), index=False)
    return summary


def paired(out, population, day):
    frame = pd.read_parquet(out / ('oof_predictions_%s_PRIVATE.parquet' % population))
    frame = frame[(frame.repeat == 0) & (frame.decision_day == day)]
    wide = frame.pivot_table(index=['user_id', 'label'], columns='feature_set', values='score').reset_index()
    result = bootstrap(wide.label.to_numpy(int), wide.v1_plus_change.to_numpy(float),
                       other=wide.v1_static.to_numpy(float), n=2000)
    result.update({'comparison': 'v1_plus_change minus v1_static, logistic AUROC',
                   'population': population, 'decision_day': day, 'repeat': 0, 'resamples': 2000,
                   'seed': SEED, 'users': int(len(wide)), 'positive_users': int(wide.label.sum()),
                   'auroc_v1_plus_change': float(roc_auc_score(wide.label, wide.v1_plus_change)),
                   'auroc_v1_static': float(roc_auc_score(wide.label, wide.v1_static))})
    return result


def drop_one(stats, labels, out):
    y = labels.astype(int)
    ids = list(y.index)
    chunk = v3.DECISION_CHUNKS[28]
    base_cols = v3.FEATURE_SETS['v3_all']
    arms = {'v3_all': base_cols}
    arms.update({'without_' + g: [c for c in base_cols if c not in set(cols)]
                 for g, cols in v3.DROP_ONE_GROUPS.items()})
    rows = []
    for position, (train, valid) in enumerate(outer_partitions(y)):
        tr, va = [ids[i] for i in train], [ids[i] for i in valid]
        ytr, yva = y.iloc[train], y.iloc[valid]
        Xtr_all, Xva_all, _ = v3.fold_matrix(stats, tr, va, chunk, 'v3_all')
        for arm, cols in arms.items():
            search = logistic_search(inner_folds(ytr)).fit(Xtr_all[cols], ytr)
            rows.append({'arm': arm, 'repeat': position // 5, 'fold': position % 5,
                         'n_features': len(cols),
                         'auroc': float(roc_auc_score(yva, search.predict_proba(Xva_all[cols])[:, 1]))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'sensitivity_drop_one_folds.csv', index=False)
    summary = frame.groupby('arm').agg(mean_auroc=('auroc', 'mean'), fold_sd_auroc=('auroc', 'std'),
                                       n_features=('n_features', 'first')).reset_index()
    base = float(summary.loc[summary.arm == 'v3_all', 'mean_auroc'].iloc[0])
    summary['auroc_drop_from_v3_all'] = base - summary.mean_auroc
    summary['above_noise_0.02'] = summary.auroc_drop_from_v3_all.abs() > 0.02
    summary.to_csv(out / 'sensitivity_drop_one.csv', index=False)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--out', default=None, help='default: <root>/private/cohort_v3')
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'private' / 'cohort_v3'
    out.mkdir(parents=True, exist_ok=True)
    prespec_path = root / 'diagnostics_v3' / 'prespec_v3.json'
    prespec = json.loads(prespec_path.read_text(encoding='utf-8'))
    if prespec.get('status') != 'locked' or not prespec.get('written_before_any_v3_metric'):
        raise SystemExit('prespec_v3.json is not the locked file')

    log('loading cached scores and writings')
    writings, _, users = v2.load_writings(root)
    histories = v2.histories(writings, users)
    labels = users.set_index('user_id').label.astype(int)
    stats = v3.Statistics(histories, users.user_id)
    pd.DataFrame(v3.feature_table_rows()).to_csv(out / 'feature_table.csv', index=False)
    coverage = pd.DataFrame({'chunk_index': range(1, v3.N_CHUNKS + 1),
                             'users_with_a_writing': [int(np.isfinite(stats.means[:, k, 0]).sum())
                                                      for k in range(v3.N_CHUNKS)]})
    coverage.to_csv(out / 'chunk_coverage.csv', index=False)

    for population in POPULATIONS:
        subset = labels if population == 'all_users' else labels.loc[
            list(users.loc[~users.pilot_exposed, 'user_id'])]
        log('population %s: %d users, %d positive' % (population, len(subset), int(subset.sum())))
        evaluate_population(stats, subset, population, out)
        time_to_flag(stats, subset, population, out)

    log('paired bootstraps')
    primary = paired(out, *PRIMARY)
    primary['success_rule'] = 'estimate > 0.02 and ci_low > 0'
    primary['success'] = bool(primary['estimate'] > 0.02 and primary['ci_low'] > 0)
    boot = {'primary': primary, 'secondary': [paired(out, p, d) for p, d in SECONDARY]}
    (out / 'bootstrap_v1_plus_change_minus_v1_static.json').write_text(json.dumps(boot, indent=2))
    log('drop-one sensitivity')
    drop_one(stats, labels, out)

    headline = pd.concat([pd.read_csv(out / ('headline_%s.csv' % p)) for p in POPULATIONS])
    headline.to_csv(out / 'headline_v3.csv', index=False)
    (out / 'run_v3.json').write_text(json.dumps({
        'seed': SEED, 'ran_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'exploratory': True, 'designed_after_seeing_v2_results': True,
        'output_is': 'a flag for a person to review, not a diagnosis',
        'labels_are': 'eRisk dataset task labels, not a clinical diagnosis',
        'rescored': False, 'holdout_105_used': False, 'cohort_v1_evaluation_opened': False,
        'prespec_written_at_utc': prespec['written_at_utc'],
        'prespec_sha256': sha_file(prespec_path),
        'features_sha256': sha_file(Path(__file__).with_name('v3_features.py')),
        'runner_sha256': sha_file(Path(__file__)),
        'writings': int(len(writings)), 'users': int(len(users)),
        'positive_users': int(users.label.sum()), 'pilot_users': int(users.pilot_exposed.sum()),
        'primary_endpoint': primary,
        'headline_rows': int(len(headline)),
        'packages': {n: version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'pyarrow')},
        'python': platform.python_version(),
        'warning': 'A gap under about 0.02 AUROC is noise. The non-pilot population has 26 positives.',
    }, indent=2))
    log('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
