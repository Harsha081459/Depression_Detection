"""Replay the November meeting's screening experiment on private target writings.

Run from any directory with the project's Python environment. Only the aggregate
JSON leaves memory. All fitted models use word rules for the five semantic signs.
"""
from __future__ import annotations

import json
from bisect import insort
from itertools import groupby
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import final_indirect_pipeline as lexical
from prepare_llm_pilot import audit, empty_post


SEMANTIC = ("self_blame", "helpless", "isolation", "family_strain", "emptiness")
RATES = SEMANTIC + ("absolutist",)
SIGNS = RATES + ("distress_burst", "closed_silence", "polarity_flip")
FEATURES = SIGNS + ("frequency",)
DAY = pd.Timedelta(days=1)
HORIZON = 84
OUTPUT = lexical.ROOT / "results" / "meeting_windows" / "summary.json"


def direction(current, previous):
    if previous is None or current is None:
        return "unknown"
    return "higher" if current > previous else "lower" if current < previous else "unchanged"


def cohort_stats(data, labels):
    users = data.groupby("user_id").agg(
        writings=("date", "size"), start=("date", "min"), end=("date", "max"),
        submissions=("kind", lambda kinds: (kinds == "submission").sum()),
    ).join(labels.set_index("user_id").label)
    users["comments"] = users.writings - users.submissions
    users["days"] = (users.end - users.start) / DAY
    users["per_week"] = 7 * users.writings / users.days.clip(lower=1)
    groups = {}
    for key, group in (("all", users), ("positive", users[users.label == 1]),
                       ("control", users[users.label == 0])):
        groups[key] = {
            "users": len(group), "writings": int(group.writings.sum()),
            "submissions": int(group.submissions.sum()), "comments": int(group.comments.sum()),
            "mean_writings": group.writings.mean(), "median_writings": group.writings.median(),
            "median_span_days": group.days.median(), "mean_per_week": group.per_week.mean(),
            "median_per_week": group.per_week.median(),
            "span_under_days": {str(d): int((group.days < d).sum()) for d in (1, 4, 8, 9, 28, 56)},
        }
    return groups


def thread_context(raw_root, user_ids, expected_comments):
    """Count target comments by the author of the linked root submission."""
    paths = list(raw_root.rglob("subject_*.json"))
    if not paths:
        return None
    if len(paths) != len(user_ids) or {p.stem for p in paths} != set(user_ids):
        raise ValueError("Raw thread audit and classifier cohort differ")
    counts = dict(own_thread=0, other_thread=0, unlinked=0)
    for path in paths:
        for thread in json.loads(path.read_text(encoding="utf-8")):
            root = thread.get("submission") or {}
            for comment in thread.get("comments", []):
                if comment.get("target") is not True:
                    continue
                linked = (root.get("submission_id") and
                          comment.get("submission_id") == root["submission_id"] and
                          isinstance(root.get("target"), bool))
                kind = ("own_thread" if root["target"] else "other_thread") if linked else "unlinked"
                counts[kind] += 1
    if sum(counts.values()) != expected_comments:
        raise ValueError("Raw comment total disagrees with the target-writing file")
    return counts


def timed_events(data, post_flags=lexical.flags):
    """Score content once. Treat simultaneous writings as one instant for gaps/flips."""
    events = []
    past_gaps = []
    previous_ts = previous_distress = previous_polarity = None
    for ts, same_time in groupby(data, key=lambda row: row.date):
        rows = list(same_time)
        eligible = [post_flags(row.text) for row in rows
                    if not row.disclosure and not empty_post(row.text)]
        polarity = sum(row["polarity"] for row in eligible) if eligible else None
        silence = flip = None
        if previous_ts is not None:
            gap = (ts - previous_ts) / DAY
            if len(past_gaps) >= 4 and previous_distress is not None:
                silence = int(previous_distress and gap >= 1 and gap >= 3 * median(past_gaps))
            if previous_polarity is not None and polarity is not None:
                flip = int(previous_polarity * polarity < 0)
            insort(past_gaps, gap)
        events.append({"ts": ts, "n": len(rows), "flags": eligible,
                       "disclosure": any(row.disclosure for row in rows),
                       "closed_silence": silence, "polarity_flip": flip})
        previous_ts, previous_polarity = ts, polarity
        previous_distress = any(row["distress"] > 0 for row in eligible) if eligible else None
    return events


def window_levels(events):
    posts = [post for event in events for post in event["flags"]]
    levels = dict.fromkeys(FEATURES)
    levels["frequency"] = sum(event["n"] for event in events)
    if posts:
        for name in RATES:
            n = sum(post[name] for post in posts)
            levels[name] = 0 if n == 0 else 1 if 2 * n < len(posts) else 2
    if len(posts) >= 4:
        distress = sorted(post["distress"] for post in posts)
        levels["distress_burst"] = int(np.mean(distress[-3:]) > median(distress))
    for name in ("closed_silence", "polarity_flip"):
        values = [event[name] for event in events if event[name] is not None]
        levels[name] = max(values, default=None)
    return levels, len(posts)


def window_states(events, width, arm, until_day=HORIZON):
    """Yield closed windows, including those that cannot produce a decision."""
    if width not in (4, 28) or arm not in ("masked", "before_disclosure"):
        raise ValueError("Unknown window width or disclosure policy")
    if not 0 <= until_day <= HORIZON:
        raise ValueError("Cutoff must be within the 84-day research horizon")
    if not events:
        return
    origin = events[0]["ts"]
    pos = 0
    previous = None
    for day in range(width, until_day + 1, width):
        start = pos
        while pos < len(events) and events[pos]["ts"] <= origin + day * DAY:
            pos += 1
        block = events[start:pos]
        if arm == "before_disclosure" and any(event["disclosure"] for event in block):
            yield {"day": day, "status": "stopped_at_disclosure"}
            return
        levels, n_content = window_levels(block)
        status = "baseline" if previous is None else "eligible" if n_content else "no_new_evidence"
        yield {"day": day, "status": status, "levels": levels, "previous": previous,
               "n_content": n_content}
        previous = levels


def encode_levels(levels, previous):
    row = {}
    for name, value in levels.items():
        row[name] = value if name == "frequency" else "unknown" if value is None else str(value)
        row["change_" + name] = direction(value, previous[name])
    return row


def windows(events, width, arm):
    """Feature rows consumed by the original aggregate experiment."""
    return [dict(day=s["day"], **encode_levels(s["levels"], s["previous"]))
            for s in window_states(events, width, arm) if s["status"] == "eligible"]


def feature_sets(ablate):
    all_columns = [col for name in FEATURES for col in (name, "change_" + name)]
    sets = {"frequency_only": ["frequency", "change_frequency"],
            "current_only": list(FEATURES), "current_and_change": all_columns}
    if ablate:
        for name in FEATURES:
            sets["without_" + name] = [col for col in all_columns if col not in (name, "change_" + name)]
    return sets


def fit_classifier(rows, labels, columns):
    train_users = rows.user_id
    y = train_users.map(labels)
    user_counts = train_users.value_counts()
    classes = labels.loc[user_counts.index].value_counts()
    if y.isna().any() or len(classes) != 2:
        raise ValueError("Training data must have a label for every user and contain both classes")
    weights = len(user_counts) / (2 * y.map(classes) * train_users.map(user_counts))
    transforms = [("categories", OneHotEncoder(handle_unknown="ignore"),
                   [col for col in columns if col != "frequency"])]
    if "frequency" in columns:
        transforms.append(("count", StandardScaler(), ["frequency"]))
    model = make_pipeline(ColumnTransformer(transforms),
                          LogisticRegression(solver="liblinear", C=1, max_iter=500, random_state=42))
    return model.fit(rows[columns], y, logisticregression__sample_weight=weights)


def predict_oof(rows, labels, folds, columns):
    scores = np.full(len(rows), np.nan)
    for train_ids, test_ids in folds:
        train, test = rows.user_id.isin(train_ids), rows.user_id.isin(test_ids)
        if not test.any():
            continue
        model = fit_classifier(rows.loc[train], labels, columns)
        scores[test] = model.predict_proba(rows.loc[test, columns])[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Missing out-of-fold prediction")
    return rows[["user_id", "day"]].assign(score=scores)


def metrics(predictions, labels, day):
    seen = predictions.loc[predictions.day <= day]
    peak = seen.groupby("user_id").score.max().reindex(labels.index)
    covered = peak.notna()
    flagged = peak.ge(0.5)
    first_flags = seen[seen.score >= 0.5].groupby("user_id").day.min()
    tp = int((flagged & labels.eq(1)).sum())
    fp = int((flagged & labels.eq(0)).sum())
    early = first_flags.reindex(labels[labels == 1].index).dropna()
    ranked = peak.fillna(-1)  # All users without a decision tie at the bottom.
    return {
        "eligible_positive": int((covered & labels.eq(1)).sum()),
        "eligible_control": int((covered & labels.eq(0)).sum()),
        "true_positive": tp, "false_positive": fp,
        "false_negative": int(labels.sum()) - tp, "true_negative": int((labels == 0).sum()) - fp,
        "auroc": roc_auc_score(labels, ranked), "average_precision": average_precision_score(labels, ranked),
        "eligible_auroc": roc_auc_score(labels[covered], peak[covered]) if labels[covered].nunique() == 2 else None,
        "eligible_average_precision": average_precision_score(labels[covered], peak[covered]) if labels[covered].nunique() == 2 else None,
        "median_first_flag_day_positive_among_flagged": float(early.median()) if len(early) else None,
    }


def activity_changes(rows, labels):
    """Count the direction of changes, separately from the classifier."""
    result = {}
    for label, name in ((0, "control"), (1, "positive")):
        sample = rows[rows.user_id.map(labels) == label]
        result[name] = {
            d: {"windows": int((sample.change_frequency == d).sum()),
                "users": int(sample.loc[sample.change_frequency == d, "user_id"].nunique())}
            for d in ("higher", "lower", "unchanged")
        }
    return result


def load_cohort():
    writings = pd.read_csv(lexical.TEXT_PATH, usecols=["user_id", "date", "kind", "title", "body"])
    users = pd.read_csv(lexical.LABEL_PATH, usecols=["user_id", "label"]).sort_values("user_id")
    data, checks = audit(writings, users)
    if data.date.isna().any() or not data.kind.isin(["submission", "comment"]).all():
        raise ValueError("Missing date or unexpected writing type")
    return data, users, checks


def prefix_histories(data, post_flags=lexical.flags):
    prefix = data[data.within_12w].sort_values(["user_id", "date"], kind="stable")
    return {uid: timed_events(group.itertuples(index=False), post_flags)
            for uid, group in prefix.groupby("user_id")}


def run():
    data, users, checks = load_cohort()
    result = {"audit": checks, "cohort": cohort_stats(data, users), "experiments": {}}
    result["target_comment_context"] = thread_context(
        lexical.ROOT / "data" / "raw" / "erisk26", users.user_id, result["cohort"]["all"]["comments"])
    labels = users.set_index("user_id").label
    folds = [(users.user_id.iloc[a], users.user_id.iloc[b]) for a, b in
             StratifiedKFold(5, shuffle=True, random_state=42).split(users, users.label)]
    histories = prefix_histories(data)
    for width in (4, 28):
        for arm in ("masked", "before_disclosure"):
            rows = pd.DataFrame([dict(record, user_id=uid) for uid, events in histories.items()
                                 for record in windows(events, width, arm)])
            experiment = {"decision_windows": len(rows), "activity_change": activity_changes(rows, labels),
                          "models": {}}
            for name, columns in feature_sets(ablate=arm == "masked").items():
                predictions = predict_oof(rows, labels, folds, columns)
                experiment["models"][name] = {
                    str(day): metrics(predictions, labels, day)
                    for day in (8, 28, 56, 84) if day >= 2 * width
                }
            result["experiments"][f"{width}d_{arm}"] = experiment
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(OUTPUT)
    print("Wrote aggregate results for 523 users and both schedules.")


if __name__ == "__main__":
    run()
