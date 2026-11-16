"""Run the v2 nested cross-validation described in diagnostics_v2/prespec_v2.json.

    python scripts/analysis/run_v2.py                       # project defaults
    python scripts/analysis/run_v2.py --root . --out private/cohort_v2

No GPU. Nothing is rescored: every LLM number is the cached Ollama P(Yes).
The 105-user holdout in split.json is not read. private/cohort_v1/evaluation/ is
not opened. Aggregates and user-level rows are written under private/cohort_v2/.

The models score a user so a person can review the top of the list. A flag is
not a diagnosis, and the labels are eRisk dataset task labels.

Run the unit tests first:
    python -m unittest discover -s scripts/analysis -p 'test_v2_temporal_features.py'
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_temporal_features as v2
from erisk_full_pipeline import bootstrap, sha_file

SEED = v2.SEED
DECISION_DAYS = (28, 84)
SELECTION_DAY = 28
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
BOOSTING_GRID = {'model__n_estimators': [50, 100], 'model__max_depth': [1, 2],
                 'model__learning_rate': [0.1]}
BOOSTING_SETS = ('v2_all', 'v1_static')
POPULATIONS = ('all_users', 'without_pilot')
FIXED_SENSITIVITY_SPEC = 'd28'
MIN_INNER_POSITIVES = 5

INTERPRETATIONS = [
    'Prefix: a decision at day D uses writings with delta <= D, as in v1. Chunk '
    'membership stays half-open, [(c-1)w, cw), with the horizon instant in the last chunk.',
    'Silence class: the longest gap that ends in the chunk and starts on a writing at or '
    'above the user\'s 75th distress percentile. The baseline is the median of the strictly '
    'earlier closed gaps with positive length.',
    'Silence degeneracy: when a user\'s 75th distress percentile is 0, every writing is at or '
    'above it, so the class falls back to the longest gap ending in the chunk. This is the '
    'literal prespec rule and was not changed.',
    'One-hot reference for the silence class is "similar", matching the prespec default for '
    'no baseline.',
    'Per the prespec feature list, absolutist and polarity jump contribute a level only. '
    'Frequency, mood, night and reply contribute a level and a change. Trend is computed for '
    'the five LLM levels only.',
    'The chunk polarity-jump level uses the mean absolute polarity change undivided. v1\'s own '
    'polarity_jump keeps its division by 2 inside v1_static.',
    'Post-count specs: a user who has not filled chunk k gets an empty chunk and their whole '
    'available prefix, with the chunk end set to their last writing. Median end day uses only '
    'training users who filled the chunk. Chunk indices are capped at %d.' % v2.POST_CHUNK_CAP,
    'The day-84 decision of a post-count spec is the complete chunk whose training median end '
    'day is closest to 84, by the same rule the prespec gives for day 28. Ties go to the '
    'smaller chunk index.',
    'Known consequence of the post-count rule, not a change to it: the median end day is taken '
    'over the users who filled the chunk, which keeps selecting heavy posters, so the median '
    'end day plateaus near day 20 instead of reaching day 28. The chunk the rule picks is '
    'therefore one that many users never filled, and their rows are empty. The selection table '
    'records how many training users reached each candidate chunk.',
    'Flag threshold: the (1 - training prevalence) quantile of the refit model\'s scores on the '
    'training fold, applied to the test fold.',
    'Time to flag: decisions run from chunk 1 to the chunk used for the day-84 decision. The day '
    'recorded is the chunk end day, which for a post-count spec is the user\'s own chunk end.',
    'A user with no eligible writing, and any chunk a user never reached, encode to an all-zero '
    'row.',
]


# --------------------------------------------------------------------------- #
def log(message: str) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def logistic_search(inner):
    pipeline = Pipeline([('scale', StandardScaler()),
                         ('model', LogisticRegression(solver='liblinear',
                                                      max_iter=2000, random_state=SEED))])
    return GridSearchCV(pipeline, {'model__C': list(LOGISTIC_C)}, scoring='roc_auc',
                        cv=inner, n_jobs=-1, refit=True, error_score='raise')


def boosting_search(inner):
    pipeline = Pipeline([('scale', StandardScaler()),
                         ('model', GradientBoostingClassifier(random_state=SEED))])
    return GridSearchCV(pipeline, BOOSTING_GRID, scoring='roc_auc', cv=inner,
                        n_jobs=-1, refit=True, error_score='raise')


def inner_folds(y) -> list:
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def eligible_inner(y) -> bool:
    y = np.asarray(y, dtype=int)
    if y.sum() < MIN_INNER_POSITIVES:
        return False
    return all(int(y[train].sum()) >= MIN_INNER_POSITIVES for train, _ in inner_folds(y))


class Bank:
    """Raw and encoded feature frames, built once per (spec, chunk) and cached."""

    def __init__(self, histories: dict, users: pd.DataFrame):
        self.users = [str(u) for u in users.user_id]
        self.raw_banks = {spec: v2.raw_bank(histories, spec) for spec in v2.CHUNK_SPECS}
        self._cache: dict[tuple[str, int], pd.DataFrame] = {}

    def encoded(self, spec: str, chunk: int) -> pd.DataFrame:
        key = (spec, int(chunk))
        if key not in self._cache:
            raw = v2.rows_at(self.raw_banks[spec], int(chunk), self.users)
            self._cache[key] = v2.encode(raw)
        return self._cache[key]

    def raw(self, spec: str, chunk: int) -> pd.DataFrame:
        return v2.rows_at(self.raw_banks[spec], int(chunk), self.users)

    def chunk_for_day(self, spec: str, day: int, training_users) -> int:
        return v2.chunk_for_day(spec, day, self.raw_banks[spec], list(training_users))

    def chunk_end_days(self, spec: str, chunk: int) -> pd.Series:
        return self.raw(spec, chunk).chunk_end_day


def features(bank: Bank, spec: str, chunk: int, feature_set: str, user_ids) -> pd.DataFrame:
    return bank.encoded(spec, chunk).loc[list(user_ids), v2.FEATURE_SETS[feature_set]]


# --------------------------------------------------------------------------- #
def reached_count(bank: Bank, spec: str, chunk: int, user_ids) -> int:
    """How many of these users actually filled the chunk. Day chunks always exist."""
    raw = bank.raw(spec, int(chunk)).loc[list(user_ids)]
    return int(raw.reached.fillna(False).astype(bool).sum())


def select_spec(bank: Bank, train_ids, y_train) -> tuple[str, list[dict]]:
    """Choose the chunk spec inside one outer training fold. Labels of the test
    fold are never touched; the inner score is v2_all logistic with C = 1."""
    rows = []
    inner = inner_folds(y_train)
    pipeline = Pipeline([('scale', StandardScaler()),
                         ('model', LogisticRegression(solver='liblinear', C=1.0,
                                                      max_iter=2000, random_state=SEED))])
    for spec in v2.CHUNK_SPECS:
        chunk = bank.chunk_for_day(spec, SELECTION_DAY, train_ids)
        if not eligible_inner(y_train):
            rows.append({'spec': spec, 'chunk_index': chunk, 'inner_auroc': float('nan'),
                         'eligible': False})
            continue
        X = features(bank, spec, chunk, 'v2_all', train_ids).to_numpy(float)
        y = np.asarray(y_train, dtype=int)
        scores = []
        for train, valid in inner:
            fit = pipeline.fit(X[train], y[train])
            scores.append(roc_auc_score(y[valid], fit.predict_proba(X[valid])[:, 1]))
        rows.append({'spec': spec, 'chunk_index': chunk,
                     'inner_auroc': float(np.mean(scores)), 'eligible': True,
                     'training_users_who_reached_the_chunk': reached_count(bank, spec, chunk, train_ids),
                     'at_post_chunk_cap': bool(v2.CHUNK_SPECS[spec][0] == 'posts'
                                               and chunk == v2.POST_CHUNK_CAP)})
    usable = [row for row in rows if row['eligible']]
    if not usable:
        return FIXED_SENSITIVITY_SPEC, rows
    best = max(usable, key=lambda row: (row['inner_auroc'], -list(v2.CHUNK_SPECS).index(row['spec'])))
    return best['spec'], rows


def flag_metrics(train_scores, train_y, test_scores, test_y) -> dict:
    """Flag the top fraction of the training fold equal to its positive prevalence."""
    train_y = np.asarray(train_y, dtype=int)
    test_y = np.asarray(test_y, dtype=int)
    rate = float(train_y.mean())
    threshold = float(np.quantile(np.asarray(train_scores, dtype=float), 1.0 - rate))
    flagged = np.asarray(test_scores, dtype=float) >= threshold
    positives = int(test_y.sum())
    return {'flag_rate': rate, 'threshold': threshold,
            'flagged': int(flagged.sum()),
            'flags_per_100_users': 100.0 * float(flagged.mean()),
            'recall': float(flagged[test_y == 1].mean()) if positives else float('nan'),
            'precision': float(test_y[flagged].mean()) if flagged.any() else float('nan')}


def evaluate_population(bank: Bank, labels: pd.Series, population: str, out: Path) -> dict:
    y = labels.astype(int)
    user_ids = list(y.index)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    partitions = list(outer.split(np.zeros(len(y)), y))
    metrics, predictions, selection, flags = [], [], [], []
    started = time.time()
    for position, (train, valid) in enumerate(partitions):
        repeat, fold = divmod(position, 5)
        train_ids = [user_ids[i] for i in train]
        valid_ids = [user_ids[i] for i in valid]
        y_train, y_valid = y.iloc[train], y.iloc[valid]
        spec, rows = select_spec(bank, train_ids, y_train)
        for row in rows:
            selection.append(dict(row, population=population, repeat=repeat, fold=fold,
                                  selected=row['spec'] == spec))
        for day in DECISION_DAYS:
            chunk = bank.chunk_for_day(spec, day, train_ids)
            for feature_set in v2.FEATURE_SETS:
                models = [('logistic', logistic_search)]
                if feature_set in BOOSTING_SETS:
                    models.append(('boosting', boosting_search))
                Xtr = features(bank, spec, chunk, feature_set, train_ids)
                Xva = features(bank, spec, chunk, feature_set, valid_ids)
                for name, builder in models:
                    search = builder(inner_folds(y_train))
                    search.fit(Xtr, y_train)
                    score_valid = search.predict_proba(Xva)[:, 1]
                    score_train = search.predict_proba(Xtr)[:, 1]
                    metrics.append({
                        'population': population, 'repeat': repeat, 'fold': fold,
                        'decision_day': day, 'spec': spec, 'chunk_index': chunk,
                        'feature_set': feature_set, 'model': name,
                        'n_features': Xtr.shape[1],
                        'training_users_who_reached_the_chunk': reached_count(
                            bank, spec, chunk, train_ids),
                        'auroc': float(roc_auc_score(y_valid, score_valid)),
                        'auprc': float(average_precision_score(y_valid, score_valid)),
                        'parameters': json.dumps(search.best_params_, sort_keys=True)})
                    flags.append(dict(
                        {'population': population, 'repeat': repeat, 'fold': fold,
                         'decision_day': day, 'spec': spec, 'feature_set': feature_set,
                         'model': name, 'test_users': len(valid_ids),
                         'test_positives': int(y_valid.sum())},
                        **flag_metrics(score_train, y_train, score_valid, y_valid)))
                    if name == 'logistic':
                        predictions.extend(
                            {'user_id': uid, 'population': population, 'repeat': repeat,
                             'fold': fold, 'decision_day': day, 'spec': spec,
                             'feature_set': feature_set, 'model': name,
                             'label': int(label), 'score': float(value)}
                            for uid, label, value in zip(valid_ids, y_valid, score_valid))
        log('%s outer %d/%d done (%.0fs elapsed)' % (population, position + 1,
                                                     len(partitions), time.time() - started))
    frame = pd.DataFrame(metrics)
    frame.to_csv(out / ('fold_metrics_%s.csv' % population), index=False)
    pd.DataFrame(selection).to_csv(out / ('spec_selection_%s.csv' % population), index=False)
    pd.DataFrame(flags).to_csv(out / ('flag_metrics_folds_%s.csv' % population), index=False)
    summary = frame.groupby(['population', 'decision_day', 'feature_set', 'model']).agg(
        mean_auroc=('auroc', 'mean'), fold_sd_auroc=('auroc', 'std'),
        mean_auprc=('auprc', 'mean'), fold_sd_auprc=('auprc', 'std'),
        folds=('auroc', 'size')).reset_index()
    summary['users'] = len(y)
    summary['positive_users'] = int(y.sum())
    summary.to_csv(out / ('headline_%s.csv' % population), index=False)
    flag_summary = pd.DataFrame(flags).groupby(
        ['population', 'decision_day', 'feature_set', 'model']).agg(
        mean_recall=('recall', 'mean'), mean_precision=('precision', 'mean'),
        mean_flags_per_100_users=('flags_per_100_users', 'mean'),
        mean_flag_rate=('flag_rate', 'mean')).reset_index()
    flag_summary.to_csv(out / ('flag_summary_%s.csv' % population), index=False)
    pd.DataFrame(predictions).to_parquet(
        out / ('oof_predictions_%s_PRIVATE.parquet' % population), index=False)
    return {'folds': len(partitions),
            'selected_specs': pd.DataFrame(selection).query('selected').spec.value_counts().to_dict()}


# --------------------------------------------------------------------------- #
def time_to_flag(bank: Bank, labels: pd.Series, population: str, out: Path) -> pd.DataFrame:
    """Repeat 0 only, so each user is tested once. Earliest chunk that flags them."""
    y = labels.astype(int)
    user_ids = list(y.index)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    partitions = list(outer.split(np.zeros(len(y)), y))[:5]
    records = []
    for position, (train, valid) in enumerate(partitions):
        train_ids = [user_ids[i] for i in train]
        valid_ids = [user_ids[i] for i in valid]
        y_train, y_valid = y.iloc[train], y.iloc[valid]
        spec, _ = select_spec(bank, train_ids, y_train)
        last = bank.chunk_for_day(spec, 84, train_ids)
        for feature_set in BOOSTING_SETS:
            for chunk in range(1, last + 1):
                Xtr = features(bank, spec, chunk, feature_set, train_ids)
                Xva = features(bank, spec, chunk, feature_set, valid_ids)
                search = logistic_search(inner_folds(y_train))
                search.fit(Xtr, y_train)
                decision = flag_metrics(search.predict_proba(Xtr)[:, 1], y_train,
                                        search.predict_proba(Xva)[:, 1], y_valid)
                score = search.predict_proba(Xva)[:, 1]
                ends = bank.chunk_end_days(spec, chunk).loc[valid_ids]
                records.extend(
                    {'population': population, 'fold': position, 'spec': spec,
                     'feature_set': feature_set, 'chunk_index': chunk,
                     'chunk_end_day': float(end), 'user_id': uid, 'label': int(label),
                     'flagged': bool(value >= decision['threshold']) and np.isfinite(end)}
                    for uid, label, value, end in zip(valid_ids, y_valid, score, ends))
        log('time-to-flag %s fold %d/5 done' % (population, position + 1))
    frame = pd.DataFrame(records)
    frame.to_parquet(out / ('time_to_flag_folds_%s_PRIVATE.parquet' % population), index=False)
    rows = []
    for feature_set, group in frame.groupby('feature_set'):
        positives = group[group.label == 1]
        first = (positives[positives.flagged].groupby('user_id').chunk_end_day.min())
        n_positive = positives.user_id.nunique()
        rows.append({'population': population, 'feature_set': feature_set, 'model': 'logistic',
                     'positive_users': n_positive, 'ever_flagged': int(first.size),
                     'never_flagged_share': 1.0 - first.size / max(1, n_positive),
                     'median_days_to_first_flag': float(first.median()) if first.size else float('nan'),
                     'decisions_per_user': int(group.chunk_index.nunique())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / ('time_to_flag_%s.csv' % population), index=False)
    return summary


# --------------------------------------------------------------------------- #
def paired_bootstrap(out: Path, population: str = 'all_users') -> dict:
    """v2_all logistic minus v1_static logistic at day 28, repeat 0, 2000 resamples."""
    frame = pd.read_parquet(out / ('oof_predictions_%s_PRIVATE.parquet' % population))
    frame = frame[(frame.repeat == 0) & (frame.decision_day == 28) & (frame.model == 'logistic')]
    wide = frame.pivot_table(index=['user_id', 'label'], columns='feature_set', values='score')
    wide = wide.reset_index()
    result = bootstrap(wide.label.to_numpy(int), wide.v2_all.to_numpy(float),
                       other=wide.v1_static.to_numpy(float), n=2000)
    result.update({'population': population, 'decision_day': 28, 'repeat': 0,
                   'resamples': 2000, 'seed': SEED, 'users': int(len(wide)),
                   'positive_users': int(wide.label.sum()),
                   'auroc_v2_all': float(roc_auc_score(wide.label, wide.v2_all)),
                   'auroc_v1_static': float(roc_auc_score(wide.label, wide.v1_static)),
                   'note': 'Paired class-stratified user bootstrap on repeat-0 out-of-fold '
                           'scores. Conditional on the fitted models.'})
    (out / 'bootstrap_v2_all_minus_v1_static.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    return result


def drop_one(bank: Bank, labels: pd.Series, out: Path) -> pd.DataFrame:
    """Drop-one group sensitivity at the fixed d28 spec, day 28, all users, 5x5."""
    y = labels.astype(int)
    user_ids = list(y.index)
    chunk = v2.chunk_for_day(FIXED_SENSITIVITY_SPEC, 28)
    base_columns = v2.FEATURE_SETS['v2_all']
    arms = {'v2_all': base_columns}
    for group, columns in v2.DROP_ONE_GROUPS.items():
        arms['without_' + group] = [c for c in base_columns if c not in set(columns)]
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    X_all = bank.encoded(FIXED_SENSITIVITY_SPEC, chunk).loc[user_ids]
    rows = []
    for position, (train, valid) in enumerate(outer.split(np.zeros(len(y)), y)):
        repeat, fold = divmod(position, 5)
        y_train, y_valid = y.iloc[train], y.iloc[valid]
        for arm, columns in arms.items():
            Xtr, Xva = X_all.iloc[train][columns], X_all.iloc[valid][columns]
            search = logistic_search(inner_folds(y_train))
            search.fit(Xtr, y_train)
            rows.append({'arm': arm, 'repeat': repeat, 'fold': fold,
                         'n_features': Xtr.shape[1],
                         'auroc': float(roc_auc_score(y_valid, search.predict_proba(Xva)[:, 1]))})
    frame = pd.DataFrame(rows)
    summary = frame.groupby('arm').agg(mean_auroc=('auroc', 'mean'),
                                       fold_sd_auroc=('auroc', 'std'),
                                       n_features=('n_features', 'first')).reset_index()
    base = float(summary.loc[summary.arm == 'v2_all', 'mean_auroc'].iloc[0])
    summary['auroc_drop_from_v2_all'] = base - summary.mean_auroc
    summary['above_noise_0.02'] = summary.auroc_drop_from_v2_all.abs() > 0.02
    summary['spec'] = FIXED_SENSITIVITY_SPEC
    summary['decision_day'] = 28
    summary.to_csv(out / 'sensitivity_drop_one.csv', index=False)
    frame.to_csv(out / 'sensitivity_drop_one_folds.csv', index=False)
    return summary


def permutation_importance(bank: Bank, labels: pd.Series, out: Path, repeats: int = 5) -> pd.DataFrame:
    """Permute each group on a depth-1 boosting model, inside the five folds of repeat 0."""
    y = labels.astype(int)
    user_ids = list(y.index)
    chunk = v2.chunk_for_day(FIXED_SENSITIVITY_SPEC, 28)
    X_all = bank.encoded(FIXED_SENSITIVITY_SPEC, chunk).loc[user_ids, v2.FEATURE_SETS['v2_all']]
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    partitions = list(outer.split(np.zeros(len(y)), y))[:5]
    rows = []
    for position, (train, valid) in enumerate(partitions):
        y_train, y_valid = y.iloc[train], y.iloc[valid]
        model = Pipeline([('scale', StandardScaler()),
                          ('model', GradientBoostingClassifier(
                              n_estimators=50, max_depth=1, learning_rate=0.1,
                              random_state=SEED))]).fit(X_all.iloc[train], y_train)
        Xva = X_all.iloc[valid]
        base = float(roc_auc_score(y_valid, model.predict_proba(Xva)[:, 1]))
        rng = np.random.default_rng(SEED + position)
        for group, columns in v2.DROP_ONE_GROUPS.items():
            columns = [c for c in columns if c in Xva.columns]
            for repeat in range(repeats):
                shuffled = Xva.copy()
                order = rng.permutation(len(shuffled))
                shuffled[columns] = shuffled[columns].to_numpy()[order]
                score = float(roc_auc_score(y_valid, model.predict_proba(shuffled)[:, 1]))
                rows.append({'fold': position, 'group': group, 'permutation': repeat,
                             'base_auroc': base, 'permuted_auroc': score,
                             'auroc_drop': base - score})
    frame = pd.DataFrame(rows)
    summary = frame.groupby('group').agg(mean_base_auroc=('base_auroc', 'mean'),
                                         mean_permuted_auroc=('permuted_auroc', 'mean'),
                                         mean_auroc_drop=('auroc_drop', 'mean'),
                                         sd_auroc_drop=('auroc_drop', 'std')).reset_index()
    summary['above_noise_0.02'] = summary.mean_auroc_drop.abs() > 0.02
    summary['spec'] = FIXED_SENSITIVITY_SPEC
    summary['decision_day'] = 28
    summary['model'] = 'boosting depth 1, 50 trees, lr 0.1'
    summary.to_csv(out / 'sensitivity_permutation.csv', index=False)
    frame.to_csv(out / 'sensitivity_permutation_folds.csv', index=False)
    return summary


# --------------------------------------------------------------------------- #
def describe_features(bank: Bank, out: Path) -> None:
    pd.DataFrame(v2.feature_table_rows()).to_csv(out / 'feature_table.csv', index=False)
    rows = []
    for spec in v2.CHUNK_SPECS:
        raw_bank = bank.raw_banks[spec]
        for chunk in sorted(set(raw_bank.index.get_level_values('chunk_index'))):
            raw = v2.rows_at(raw_bank, int(chunk), bank.users)
            rows.append({'spec': spec, 'chunk_index': int(chunk),
                         'users_with_a_writing_in_the_chunk': int((raw.n_posts > 0).sum()),
                         'median_n_posts': float(raw.n_posts.median()),
                         'median_chunk_end_day': float(raw.chunk_end_day.median()),
                         'users_reached': int(raw.reached.fillna(False).sum()),
                         'median_max_silence_days': float(raw.max_silence_days.median())})
    pd.DataFrame(rows).to_csv(out / 'chunk_coverage.csv', index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[2]
    parser.add_argument('--root', default=str(root_default))
    parser.add_argument('--out', default=None, help='default: <root>/private/cohort_v2')
    parser.add_argument('--text-csv', default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / 'private' / 'cohort_v2'
    out.mkdir(parents=True, exist_ok=True)

    prespec = v2.prespec(root)
    if prespec.get('status') != 'locked' or not prespec.get('written_before_any_v2_metric'):
        raise SystemExit('prespec_v2.json is not the locked file')
    (out / 'interpretations.json').write_text(json.dumps({
        'written_before_any_v2_metric_in_this_run': True,
        'prespec_written_at_utc': prespec['written_at_utc'],
        'note': 'Places the prespec did not pin down, resolved once, before this run produced '
                'a number. No cut, bin, chunk length or model grid was changed.',
        'decisions': INTERPRETATIONS}, indent=2), encoding='utf-8')

    log('loading cached scores and writings')
    writings, origins, users = v2.load_writings(
        root, Path(args.text_csv) if args.text_csv else None)
    log('writings %d, users with a writing %d, labelled users %d, positive %d'
        % (len(writings), writings.user_id.nunique(), len(users), int(users.label.sum())))
    histories = v2.histories(writings, users)
    labels = users.set_index('user_id').label.astype(int)
    log('building chunk banks')
    bank = Bank(histories, users)
    describe_features(bank, out)

    summaries = {}
    for population in POPULATIONS:
        if population == 'all_users':
            subset = labels
        else:
            keep = users.loc[~users.pilot_exposed, 'user_id']
            subset = labels.loc[list(keep)]
        log('population %s: %d users, %d positive' % (population, len(subset), int(subset.sum())))
        summaries[population] = evaluate_population(bank, subset, population, out)
        summaries[population]['time_to_flag'] = time_to_flag(
            bank, subset, population, out).to_dict('records')

    log('paired bootstrap')
    boot = paired_bootstrap(out, 'all_users')
    log('drop-one sensitivity')
    drops = drop_one(bank, labels, out)
    log('permutation importance')
    permutation_importance(bank, labels, out)

    headline = pd.concat([pd.read_csv(out / ('headline_%s.csv' % p)) for p in POPULATIONS])
    headline.to_csv(out / 'headline_v2.csv', index=False)

    variants = {
        'chunk_specs': len(v2.CHUNK_SPECS),
        'feature_sets': len(v2.FEATURE_SETS),
        'populations': len(POPULATIONS),
        'decision_days': len(DECISION_DAYS),
        'model_families': 2,
        'logistic_fits_per_fold_per_day': len(v2.FEATURE_SETS),
        'boosting_feature_sets': len(BOOSTING_SETS),
        'drop_one_groups': len(v2.DROP_ONE_GROUPS),
        'ordinal_definitions': 2,
        'ordinal_definitions_fit': 1,
        'headline_rows': int(len(headline)),
    }
    payload = {
        'seed': SEED,
        'ran_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'exploratory': True,
        'output_is': 'a flag for a person to review, not a diagnosis',
        'labels_are': 'eRisk dataset task labels, not a clinical diagnosis',
        'rescored': False,
        'holdout_105_used': False,
        'cohort_v1_evaluation_opened': False,
        'prespec_written_at_utc': prespec['written_at_utc'],
        'prespec_sha256': sha_file(root / 'diagnostics_v2' / 'prespec_v2.json'),
        'features_sha256': sha_file(Path(__file__).with_name('v2_temporal_features.py')),
        'runner_sha256': sha_file(Path(__file__)),
        'rule_source_sha256': sha_file(Path(__file__).with_name('final_indirect_pipeline.py')),
        'writings': int(len(writings)),
        'users': int(len(users)),
        'positive_users': int(users.label.sum()),
        'pilot_users': int(users.pilot_exposed.sum()),
        'encoded_columns': len(v2.ALL_COLUMNS),
        'variants_counted': variants,
        'selected_specs': {p: summaries[p]['selected_specs'] for p in POPULATIONS},
        'time_to_flag': {p: summaries[p]['time_to_flag'] for p in POPULATIONS},
        'paired_bootstrap': boot,
        'drop_one_above_noise': drops.loc[drops['above_noise_0.02'], 'arm'].tolist(),
        'packages': {name: version(name) for name in
                     ('numpy', 'pandas', 'scikit-learn', 'pyarrow')},
        'python': platform.python_version(),
        'warning': 'Small AUROC gaps on 523 users, 91 of them eRisk-positive, are noise. '
                   'A gap under about 0.02 AUROC means nothing here.',
    }
    (out / 'run_v2.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    log('wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
