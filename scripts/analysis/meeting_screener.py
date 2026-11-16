"""Train, check and replay the meeting's early-warning research classifier."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

import meeting_windows as mw
from llm_signal_scorer import canonical_hash, load_prompts, SafeError
from meeting_qwen_features import CachedPostFlags, verified_source

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive"))
from erisk_task2_features import find_user_jsons, writings_from_user_file


MODEL_PATH = mw.lexical.ROOT / "data" / "intake" / "meeting_screener.joblib"
QWEN_PATH = MODEL_PATH.with_name("meeting_screener_qwen.joblib")
COLUMNS = mw.feature_sets(False)["current_and_change"]
CONTROL_BUDGET = 0.10
VERSION = 6


def calibration_threshold(control_peaks):
    """At most floor(10% of calibration controls) may ever alert by day 84.

    An unscored control has a NaN peak and still counts in the denominator.
    An entire tied boundary is excluded. Never lower the original 0.5 cutoff.
    """
    peaks = np.asarray(control_peaks, dtype=float)
    seen = np.sort(peaks[~np.isnan(peaks)])[::-1]
    if not len(peaks) or not np.all((seen >= 0) & (seen <= 1)):
        raise ValueError("Expected calibration controls with valid model scores")
    limit = math.floor(CONTROL_BUDGET * len(peaks))
    if len(seen) <= limit:
        return 0.5
    return max(0.5, float(np.nextafter(seen[limit], np.inf)))


def scores_for(model, rows):
    if rows.empty:
        return np.array([], dtype=float)
    return model.predict_proba(rows[COLUMNS])[:, 1]


def alternatives(state):
    """Change one measured feature level one step and recompute its trend.

    This tests the classifier at the feature-vector level. Unknowns stay
    unknown. A one-writing window cannot have the minority share grade.
    """
    levels, previous = state["levels"], state["previous"]
    for name, value in levels.items():
        if value is None:
            continue
        if name == "frequency":
            candidates = [n for n in (value - 1, value + 1) if n >= state["n_content"]]
        else:
            grades = [0, 1]
            if name in mw.RATES:
                grades = [0, 2] if state["n_content"] < 3 else [0, 1, 2]
            idx = grades.index(value)
            candidates = [grade for i, grade in enumerate(grades) if abs(i - idx) == 1]
        for alternate in candidates:
            yield name, alternate, mw.encode_levels(dict(levels, **{name: alternate}), previous)


def screening_scores(model, threshold, states):
    """Score windows and test every single-feature alternative for each candidate alert."""
    rows = [mw.encode_levels(s["levels"], s["previous"]) for s in states]
    scores = scores_for(model, pd.DataFrame(rows))
    flipped = [[] for _ in states]
    alternatives_rows, keys = [], []
    for i, (state, score) in enumerate(zip(states, scores)):
        if score >= threshold:
            for name, alternate, row in alternatives(state):
                alternatives_rows.append(row)
                keys.append((i, name, alternate))
    for (i, name, alternate), score in zip(keys, scores_for(model, pd.DataFrame(alternatives_rows))):
        if score < threshold:
            flipped[i].append({"feature": name, "alternative_level": alternate, "score": float(score)})
    return scores, flipped


def decision_counts(rows, labels, policy, day):
    relevant = rows[rows.day <= day]
    covered = labels.index.isin(relevant.user_id)
    firsts = relevant[relevant[policy]].groupby("user_id").day.min()
    flags = labels.index.isin(firsts.index)
    positive_delays = firsts.reindex(labels[labels == 1].index).dropna()
    replay_reference = sorted((uid, bool(seen), int(firsts.get(uid, 0)))
                              for uid, seen in zip(labels.index, covered))
    return {
        "replay_digest": canonical_hash(replay_reference),
        "positive_flagged": int((flags & (labels == 1)).sum()),
        "control_flagged": int((flags & (labels == 0)).sum()),
        "positive_users": int((labels == 1).sum()), "control_users": int((labels == 0).sum()),
        "positive_assessed": int((covered & (labels == 1)).sum()),
        "control_assessed": int((covered & (labels == 0)).sum()),
        "median_first_flag_day_positive": float(positive_delays.median()) if len(positive_delays) else None,
    }


def paired_window_counts(short, long, labels, policy):
    """Compare both schedules on the same people through day 84."""
    short_assessed, long_assessed = set(short.user_id), set(long.user_id)
    short_flagged = set(short.loc[short[policy], "user_id"])
    long_flagged = set(long.loc[long[policy], "user_id"])
    result = {}
    for label, name in ((1, "positive"), (0, "control")):
        users = set(labels[labels == label].index)
        common = users & short_assessed & long_assessed
        only_short = users & short_assessed - long_assessed
        only_long = users & long_assessed - short_assessed
        both_flagged = common & short_flagged & long_flagged
        result[name] = {
            "both_assessed": len(common), "only_4d_assessed": len(only_short),
            "only_28d_assessed": len(only_long),
            "neither_assessed": len(users - short_assessed - long_assessed),
            "both_flagged": len(both_flagged),
            "only_4d_flagged": len((common & short_flagged) - long_flagged),
            "only_28d_flagged": len((common & long_flagged) - short_flagged),
            "neither_flagged_among_both_assessed": len(common - short_flagged - long_flagged),
        }
    return result


def train_schedule(histories, labels, fold_ids, width, arm):
    states = {(uid, s["day"]): s for uid, events in histories.items()
              for s in mw.window_states(events, width, arm) if s["status"] == "eligible"}
    rows = pd.DataFrame([dict(user_id=uid, day=day, **mw.encode_levels(s["levels"], s["previous"]))
                         for (uid, day), s in states.items()])
    models, predictions = [], []
    reversible = Counter()
    n_sensitive = 0
    for fold in range(5):
        calibration_fold = (fold + 1) % 5
        row_folds = rows.user_id.map(fold_ids)
        fit = rows[~row_folds.isin([fold, calibration_fold])]
        model = mw.fit_classifier(fit, labels, COLUMNS)
        controls = labels[(fold_ids == calibration_fold) & (labels == 0)].index
        cal = rows[rows.user_id.isin(controls)]
        peak = cal.assign(score=scores_for(model, cal)).groupby("user_id").score.max().reindex(controls)
        threshold = calibration_threshold(peak)
        models.append({"model": model, "threshold": threshold,
                       "calibration_controls": len(controls), "calibration_flags": int(peak.ge(threshold).sum())})
        test = rows[row_folds == fold]
        test_states = [states[key] for key in test[["user_id", "day"]].itertuples(index=False, name=None)]
        scores, flipped = screening_scores(model, threshold, test_states)
        calibrated = scores >= threshold
        stable = calibrated & np.array([not flips for flips in flipped], dtype=bool)
        outcome = test[["user_id", "day"]].assign(score=scores, fixed=scores >= 0.5,
                                                  calibrated=calibrated, stable=stable)
        predictions.append(outcome)
        firsts = outcome.assign(flips=flipped)[outcome.calibrated].sort_values("day").drop_duplicates("user_id")
        for flips in firsts.flips:
            changed = {x["feature"] for x in flips}
            n_sensitive += int(bool(changed))
            reversible.update(changed)
    all_scores = pd.concat(predictions, ignore_index=True)
    if len(all_scores) != len(states) or all_scores.duplicated(["user_id", "day"]).any():
        raise ValueError("A held-out decision was missed or scored twice")
    comparison = {policy: {str(day): decision_counts(all_scores, labels, policy, day)
                           for day in (8, 28, 56, 84) if day >= 2 * width}
                  for policy in ("fixed", "calibrated", "stable")}
    comparison["replay_digests"] = {
        policy: {str(day): decision_counts(all_scores, labels, policy, day)["replay_digest"]
                 for day in range(0, mw.HORIZON + 1, width)}
        for policy in ("calibrated", "stable")
    }
    comparison["first_flags_reversed_by_one_feature"] = n_sensitive
    comparison["reversals_by_feature"] = {name: reversible[name] for name in mw.FEATURES}
    return models, comparison, all_scores


def build_bundle(post_flags=mw.lexical.flags, qwen_source=None):
    data, users, _ = mw.load_cohort()
    histories = mw.prefix_histories(data, post_flags)
    labels = users.set_index("user_id").label
    fold_ids = pd.Series(-1, index=labels.index, dtype=int)
    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(users, labels)):
        fold_ids.iloc[test] = fold
    qwen_metadata = dict(qwen_source, quality_counts=dict(post_flags.quality_counts)) if qwen_source else None
    bundle = {"version": VERSION, "sklearn": sklearn.__version__,
              "source": "validated_qwen_and_rules" if qwen_source else "word_rules", "qwen": qwen_metadata,
              "user_folds": fold_ids.to_dict(), "models": {}, "evaluation": {}}
    scored = {}
    for width in (4, 28):
        for arm in ("masked", "before_disclosure"):
            key = f"{width}d_{arm}"
            bundle["models"][key], bundle["evaluation"][key], scored[key] = train_schedule(
                histories, labels, fold_ids, width, arm)
    bundle["evaluation"]["paired_window_comparison"] = {
        arm: {policy: paired_window_counts(scored[f"4d_{arm}"], scored[f"28d_{arm}"], labels, policy)
              for policy in ("calibrated", "stable")}
        for arm in ("masked", "before_disclosure")
    }
    return bundle


def save_bundle(bundle, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        joblib.dump(bundle, temp_path, compress=3)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_bundle(path):
    # Only load the trusted model produced by this project's train command.
    bundle = joblib.load(path)
    if bundle.get("version") != VERSION or bundle.get("sklearn") != sklearn.__version__:
        raise ValueError("Model format or scikit-learn version changed. Run train again.")
    return bundle


def write_qwen_inputs(data, source, path):
    """Build the lab scorer input using the exact post IDs from pilot sampling."""
    subset = data.loc[data.within_12w & ~data.disclosure & ~data.text.map(mw.empty_post)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("w", encoding="utf-8") as output:
            for item in subset[["user_id", "date", "text"]].itertuples():
                post_id = canonical_hash([item.user_id, item.date.isoformat(), int(item.Index), item.text])
                output.write(json.dumps({"post_id": post_id, "text": item.text,
                                         "requested_signals": source["signals"]}, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return len(subset)


def screen(bundle, uid, events, until_day, width, arm, policy="calibrated"):
    """Score only completed windows, retaining the first alert for this history."""
    fold = bundle["user_folds"].get(uid, 0)
    fitted = bundle["models"][f"{width}d_{arm}"][fold]
    model, threshold = fitted["model"], fitted["threshold"]
    states = list(mw.window_states(events, width, arm, until_day))
    eligible = [s for s in states if s["status"] == "eligible"]
    if policy not in ("calibrated", "stable"):
        raise ValueError("Unknown screening policy")
    scored = iter(zip(*screening_scores(model, threshold, eligible)))
    timeline, first = [], None
    for state in states:
        item = {"day": state["day"], "status": state["status"]}
        if state["status"] == "eligible":
            score, flips = next(scored)
            score = float(score)
            item.update(score=score, status="below_threshold" if score < threshold else "above_threshold")
            if score >= threshold:
                item["sensitive_features"] = sorted({x["feature"] for x in flips})
            if score >= threshold and policy == "stable" and flips:
                item["status"] = "waiting_for_stable_evidence"
            elif score >= threshold and first is None:
                features = mw.encode_levels(state["levels"], state["previous"])
                first = {"day": state["day"], "score": score, "features": features,
                         "single_feature_reversals": flips}
                item["status"] = "first_review_flag"
        timeline.append(item)
    return {"decision": "flag_for_review" if first is not None else "no_flag",
            "as_of_day": until_day, "window_days": width, "disclosure_policy": arm,
            "source": bundle["source"], "qwen_signals": (bundle.get("qwen") or {}).get("signals", []),
            "policy": policy, "threshold": threshold, "first_flag": first,
            "observation": timeline[-1]["status"] if timeline else
                           "waiting_for_first_window" if events else "no_target_writings",
            "model_partition": "held_out_user" if uid in bundle["user_folds"] else "new_user",
            "timeline": timeline}


def read_history(path, day, post_flags=mw.lexical.flags, arm="masked"):
    rows, _ = writings_from_user_file(path)
    if not rows:
        return path.stem, []
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data.date, utc=True, format="mixed")
    data = data[data.date <= data.date.min() + day * mw.DAY].sort_values("date", kind="stable").copy()
    owners = data.author.unique()
    if len(owners) != 1:
        raise ValueError("One input history must contain exactly one target writer")
    data["text"] = (data.title.fillna("") + " " + data.body.fillna("")).str.strip()
    data["disclosure"] = data.text.map(mw.lexical.disclosure)
    if arm == "before_disclosure" and data.disclosure.any():
        first = data.loc[data.disclosure, "date"].min()
        data = data.loc[data.date <= first].copy()
        # No content from the disclosure instant can lead to a scheduled decision.
        data.loc[data.date == first, "disclosure"] = True
    return owners[0], mw.timed_events(data.itertuples(index=False), post_flags)


def replay_cohort(bundle, flags, args):
    paths = find_user_jsons(mw.lexical.ROOT / "data" / "raw" / "erisk26")
    if len(paths) != len(bundle["user_folds"]) or {p.stem for p in paths} != set(bundle["user_folds"]):
        raise ValueError("Replay requires the original cohort files with their original filenames")
    destination = MODEL_PATH.parent / "meeting_decisions" / (
        f"{bundle['source']}_{args.width}d_{args.arm}_{args.policy}_day{args.day}.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flagged = assessed = 0
    replay_reference = []
    with NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("w", encoding="utf-8") as output:
            for path in paths:
                uid, events = read_history(path, args.day, flags, args.arm)
                if uid != path.stem:
                    raise ValueError("Original cohort filename and target writer disagree")
                result = screen(bundle, uid, events, args.day, args.width, args.arm, args.policy)
                output.write(json.dumps(dict(user_id=path.stem, **result), allow_nan=False) + "\n")
                flagged += result["first_flag"] is not None
                seen = any("score" in item for item in result["timeline"])
                assessed += seen
                replay_reference.append((uid, seen, result["first_flag"]["day"] if result["first_flag"] else 0))
        closed_day = args.day // args.width * args.width
        expected = bundle["evaluation"][f"{args.width}d_{args.arm}"]["replay_digests"][args.policy].get(str(closed_day))
        if expected and canonical_hash(sorted(replay_reference)) != expected:
            raise ValueError("Raw-file replay differs from the saved held-out evaluation")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return {"users": len(paths), "assessed": assessed, "flagged": flagged, "private_output": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Fit, independently calibrate, verify and save the local research models")
    train.add_argument("--qwen-cache", type=Path)
    prepare = sub.add_parser("prepare-qwen", help="Prepare private scorer input after independent signal validation")
    for command in (train, prepare):
        for option in ("prompts", "pilot-gate", "heldout-report", "phase1-report"):
            command.add_argument("--" + option, type=Path, required=command is prepare)
    check = sub.add_parser("check", help="Print the saved out-of-fold evaluation without reading private posts")
    check.add_argument("--qwen", action="store_true", help="Inspect the separately fitted Qwen-backed model")
    single = sub.add_parser("screen", help="Screen one original eRisk user JSON through a completed day")
    single.add_argument("history", type=Path)
    batch = sub.add_parser("replay", help="Write private out-of-fold decisions for the whole cohort")
    for replay in (single, batch):
        replay.add_argument("--day", type=int, required=True, choices=range(mw.HORIZON + 1), metavar="0..84")
        replay.add_argument("--width", type=int, choices=(4, 28), default=4)
        replay.add_argument("--arm", choices=("before_disclosure", "masked"), default="before_disclosure")
        replay.add_argument("--policy", choices=("calibrated", "stable"), default="calibrated")
        replay.add_argument("--qwen-cache", type=Path, help="Use the Qwen-backed model with its private frozen scores")
    args = parser.parse_args()
    if args.command in ("train", "prepare-qwen"):
        reports = [args.pilot_gate, args.heldout_report, args.phase1_report]
        use_qwen = args.command == "prepare-qwen" or bool(args.qwen_cache)
        if use_qwen != bool(args.prompts) or (use_qwen and not all(reports)) or (not use_qwen and any(reports)):
            parser.error("Qwen training requires --qwen-cache, --prompts, --pilot-gate, --heldout-report and --phase1-report")
        source = None
        flags = mw.lexical.flags
        if use_qwen:
            source = verified_source(load_prompts(args.prompts),
                                     *(json.loads(p.read_text(encoding="utf-8-sig")) for p in reports))
            if args.command == "prepare-qwen":
                data, _, _ = mw.load_cohort()
                path = MODEL_PATH.parent / "meeting_qwen_input.jsonl"
                count = write_qwen_inputs(data, source, path)
                print(f"Prepared {count} eligible private writings at {path}")
                return
            flags = CachedPostFlags(args.qwen_cache, source)
        bundle = build_bundle(flags, source)
        save_bundle(bundle, QWEN_PATH if source else MODEL_PATH)
        print("Saved 20 classifiers with user-disjoint calibration. Feature source: " + bundle["source"])
        return
    path = QWEN_PATH if getattr(args, "qwen_cache", None) or getattr(args, "qwen", False) else MODEL_PATH
    if not path.exists():
        parser.error("No trained screener found. Run the train command first.")
    bundle = load_bundle(path)
    if args.command == "check":
        result = {key: ({name: value for name, value in comparison.items()
                         if name != "replay_digests"} if key != "paired_window_comparison" else comparison)
                  for key, comparison in bundle["evaluation"].items()}
    else:
        flags = CachedPostFlags(args.qwen_cache, bundle["qwen"]) if args.qwen_cache else mw.lexical.flags
        if args.command == "replay":
            result = replay_cohort(bundle, flags, args)
        else:
            uid, events = read_history(args.history, args.day, flags, args.arm)
            if args.history.stem in bundle["user_folds"] and uid != args.history.stem:
                parser.error("Known cohort filename and target writer disagree")
            result = screen(bundle, uid, events, args.day, args.width, args.arm, args.policy)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, SafeError) as exc:
        sys.exit(str(exc))
