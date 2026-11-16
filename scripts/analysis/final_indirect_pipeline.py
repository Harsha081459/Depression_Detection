"""Indirect eRisk signals and a user-level classifier.

Explicit disclosure is never a feature. It is only used to build two
sensitivity arms: writings removed, and the timeline cut before the first
disclosure. File length and total writing count are not features.

The named model uses eight signals chosen from the earlier full-cohort
screen. Because that screen saw every user, the script also refits an
XGBoost model whose features are chosen inside each training fold.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from erisk_disclosure_audit import BROAD_EXTRA, STRICT

ROOT = Path(__file__).resolve().parents[2]
TEXT_PATH = ROOT / "data" / "processed" / "erisk" / "task2_target_text.csv"
LABEL_PATH = ROOT / "results" / "erisk_understand" / "user_duration.csv"
OUT_DIR = ROOT / "results" / "final_indirect"

WINDOWS = (("4w", 28), ("8w", 56), ("12w", 84), ("full", None))
ARMS = ("unmasked", "masked", "before_disclosure")

# Eight signals. Polarity is stored as two numbers so the trees can weight
# flips and jumps themselves; they are not hand-averaged.
FIXED = (
    "self_blame",
    "helpless",
    "emptiness",
    "isolation",
    "family_strain",
    "absolutist",
    "polarity_flip",
    "polarity_jump",
    "distress_burst",
)
# Candidates the training fold may use. Weak ones are here so the fold can
# reject them instead of us hiding the comparison.
CANDIDATES = FIXED + ("first_person", "night", "silence", "profanity")

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
CARE = re.compile(
    r"\bmy\s+(?:therapist|psychiatrist|psychologist)\b|"
    r"\bi\s+(?:see|saw|am\s+seeing|went\s+to)\s+(?:a|my)\s+"
    r"(?:therapist|psychiatrist|psychologist)\b",
    re.I,
)
ATTEMPT = re.compile(
    r"\b(?:my\s+suicide\s+attempt|i\s+(?:attempted|tried)\s+suicide|"
    r"i\s+tried\s+to\s+kill\s+myself|i(?:'m|\s+am|\s+feel|\s+felt)\s+suicidal)\b",
    re.I,
)

SELF = {"i", "me", "my", "myself", "mine", "i'm", "i've", "i'd", "i'll"}
ABSOLUTIST = {
    "always", "never", "nothing", "completely", "nobody", "every", "everyone",
    "everything", "entirely", "totally", "constantly", "forever", "impossible",
}
BLAME = {
    "worthless", "useless", "failure", "failed", "loser", "pathetic", "burden",
    "ashamed", "shame", "guilty", "guilt", "fault", "hate", "ruined", "broken",
}
HELPLESS = {
    "helpless", "hopeless", "pointless", "meaningless", "lost", "trapped",
    "stuck", "overwhelmed", "afraid", "scared", "worry", "worried",
}
EMPTY = {
    "empty", "numb", "exhausted", "drained", "tired", "fatigue", "insomnia",
    "crying", "cry", "tears", "hurt", "hurts",
}
ISOLATION = {
    "alone", "lonely", "isolated", "ignored", "abandoned", "rejected",
    "unwanted", "avoid",
}
FAMILY = {
    "mom", "mother", "dad", "father", "parent", "parents", "brother", "sister",
    "family", "husband", "wife", "partner", "boyfriend", "girlfriend",
    "roommate", "friend", "friends",
}
# No diagnosis, medication, or suicide words. Those are disclosure, not features.
INDIRECT_NEG = BLAME | HELPLESS | EMPTY | ISOLATION | {
    "sad", "miserable", "terrible", "awful", "horrible", "worst", "suffer",
    "suffering", "upset",
}
POSITIVE = {
    "happy", "happiness", "joy", "love", "loved", "great", "amazing",
    "wonderful", "fantastic", "excellent", "beautiful", "grateful", "thankful",
    "excited", "fun", "laugh", "smile", "hope", "hopeful", "proud", "good",
    "better", "best", "awesome",
}
PROFANITY = {"fuck", "fucking", "shit", "damn", "asshole", "bitch", "bullshit"}
HOBBY = {
    "game", "gaming", "boss", "quest", "raid", "rpg", "weapon", "loot",
    "level", "build", "artifact", "server", "guild", "minecraft", "character",
    "player", "players", "match", "damage", "rank", "patch",
}
REAL_LIFE = FAMILY | {
    "life", "myself", "school", "college", "job", "work", "feeling", "feelings",
}


def _clean(text: str) -> str:
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")


def disclosure(text: str) -> bool:
    text = _clean(text)
    return bool(STRICT.search(text) or BROAD_EXTRA.search(text) or CARE.search(text) or ATTEMPT.search(text))


def flags(text: str) -> dict:
    tokens = TOKEN_RE.findall(_clean(text))
    n = len(tokens)
    empty = {
        "n_tokens": n, "self_tokens": 0, "self_blame": 0, "helpless": 0,
        "emptiness": 0, "isolation": 0, "family_strain": 0, "absolutist": 0,
        "distress": 0.0, "polarity": 0.0, "profanity": 0,
    }
    if n == 0:
        return empty
    bag = set(tokens)
    self_tokens = sum(token in SELF for token in tokens)
    profanity = sum(token in PROFANITY for token in tokens)
    game_only = bool(bag & HOBBY) and not bool(bag & REAL_LIFE)
    if game_only or self_tokens == 0:
        empty["self_tokens"] = self_tokens
        empty["profanity"] = profanity
        return empty
    neg = sum(token in INDIRECT_NEG for token in tokens)
    pos = sum(token in POSITIVE for token in tokens)
    return {
        "n_tokens": n,
        "self_tokens": self_tokens,
        "self_blame": int(bool(bag & BLAME)),
        "helpless": int(bool(bag & HELPLESS)),
        "emptiness": int(bool(bag & EMPTY)),
        "isolation": int(bool(bag & ISOLATION)),
        "family_strain": int(bool(bag & FAMILY) and neg > 0),
        "absolutist": int(bool(bag & ABSOLUTIST) and neg > 0),
        "distress": neg / n,
        "polarity": (pos - neg) / (pos + neg + 1),
        "profanity": profanity,
    }


def _clip(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _silence(rows: list[dict], end: datetime | None) -> float:
    """Largest distress-then-gap score. Gaps are between the supplied rows.

    ``end`` is set only for a fixed window. Full history passes None, so time
    after the last writing is not treated as silence.
    """
    if len(rows) < 5:
        return 0.0
    gaps = [(rows[i]["ts"] - rows[i - 1]["ts"]).total_seconds() for i in range(1, len(rows))]
    best = 0.0
    for i in range(4, len(rows)):
        past = [gap for gap in gaps[:i] if gap > 0]
        if len(past) < 4:
            continue
        baseline = float(np.median(past))
        if baseline <= 0:
            continue
        if i + 1 < len(rows):
            gap = (rows[i + 1]["ts"] - rows[i]["ts"]).total_seconds()
        elif end is not None:
            gap = (end - rows[i]["ts"]).total_seconds()
        else:
            continue
        if gap <= 2 * baseline:
            continue
        distress = float(np.mean([rows[j]["distress"] for j in range(max(0, i - 2), i + 1)]))
        if distress <= 0.02:
            continue
        best = max(best, distress * np.log2(gap / baseline))
    return _clip(1 - np.exp(-best))


def aggregate(rows: list[dict], end: datetime | None) -> dict:
    if not rows:
        return {name: 0.0 for name in CANDIDATES}
    n = len(rows)
    tokens = sum(row["n_tokens"] for row in rows)
    polarity = np.array([row["polarity"] for row in rows], dtype=float)
    distress = np.array([row["distress"] for row in rows], dtype=float)
    flips = 0.0
    jump = 0.0
    if n > 1:
        flips = float(np.mean((polarity[1:] * polarity[:-1]) < 0))
        jump = float(np.mean(np.abs(np.diff(polarity))) / 2)
    top = np.sort(distress)[-min(3, n):]
    burst = max(0.0, float(top.mean() - np.median(distress)))
    nights = [row["night"] for row in rows]
    return {
        "self_blame": _clip(float(np.mean([row["self_blame"] for row in rows]))),
        "helpless": _clip(float(np.mean([row["helpless"] for row in rows]))),
        "emptiness": _clip(float(np.mean([row["emptiness"] for row in rows]))),
        "isolation": _clip(float(np.mean([row["isolation"] for row in rows]))),
        "family_strain": _clip(float(np.mean([row["family_strain"] for row in rows]))),
        "absolutist": _clip(float(np.mean([row["absolutist"] for row in rows]))),
        "polarity_flip": _clip(flips),
        "polarity_jump": _clip(jump),
        "distress_burst": _clip(burst),
        "first_person": _clip(sum(row["self_tokens"] for row in rows) / tokens) if tokens else 0.0,
        "night": _clip(float(np.mean(nights))),
        "silence": _silence(rows, end),
        "profanity": _clip(sum(row["profanity"] for row in rows) / tokens) if tokens else 0.0,
    }


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ts"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None)
    text = (data["title"].fillna("") + " " + data["body"].fillna(""))
    text = text.str.lower().str.replace("\u2019", "'", regex=False).str.replace("\u2018", "'", regex=False)
    data["text"] = text
    data["disclosure"] = text.map(disclosure)
    parsed = text.map(flags)
    for name in ("n_tokens", "self_tokens", "self_blame", "helpless", "emptiness",
                 "isolation", "family_strain", "absolutist", "distress", "polarity", "profanity"):
        data[name] = parsed.map(lambda item, key=name: item[key])
    clock = data["ts"] + pd.Timedelta(hours=5, minutes=30)
    data["night"] = clock.dt.hour.between(0, 5).astype(int)
    return data.sort_values(["user_id", "ts"], kind="mergesort")


def build_table(writings: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    data = _prepare(writings)
    labels = labels[["user_id", "label"]].drop_duplicates("user_id").sort_values("user_id")
    rows = []
    grouped = {uid: group.to_dict("records") for uid, group in data.groupby("user_id", sort=False)}
    for user_id, label in labels.itertuples(index=False):
        posts = grouped.get(user_id, [])
        origin = posts[0]["ts"].to_pydatetime() if posts else None
        first_disclosure = next((post["ts"].to_pydatetime() for post in posts if post["disclosure"]), None)
        for window, days in WINDOWS:
            end = None if origin is None or days is None else origin + timedelta(days=days)
            prefix = posts if end is None else [post for post in posts if post["ts"].to_pydatetime() <= end]
            selections = {
                "unmasked": prefix,
                "masked": [post for post in prefix if not post["disclosure"]],
                "before_disclosure": [
                    post for post in prefix
                    if first_disclosure is None or post["ts"].to_pydatetime() < first_disclosure
                ],
            }
            silence_end = end
            # Keep every timestamp when a disclosure writing is removed.
            # Deleting it would join the gaps on either side into a false silence.
            silence_timeline = []
            for post in prefix:
                if post["disclosure"]:
                    kept = dict(post)
                    kept["distress"] = 0.0
                    silence_timeline.append(kept)
                else:
                    silence_timeline.append(post)
            for arm, selected in selections.items():
                # A fixed window may score the gap from the last writing to the
                # window end. Full history has no end beyond the file. Cutting
                # the timeline at the first disclosure is not itself silence.
                feature_end = silence_end if arm != "before_disclosure" else None
                values = aggregate(selected, feature_end)
                if arm == "masked":
                    values["silence"] = _silence(silence_timeline, silence_end)
                rows.append({
                    "user_id": user_id,
                    "label": int(label),
                    "window": window,
                    "arm": arm,
                    "n_writings": len(selected),
                    "n_disclosure_removed": len(prefix) - len(selections["masked"]) if arm == "masked" else 0,
                    **values,
                })
    table = pd.DataFrame(rows)
    if not np.isfinite(table[list(CANDIDATES)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature")
    if table[list(CANDIDATES)].lt(0).any().any() or table[list(CANDIDATES)].gt(1).any().any():
        raise ValueError("Feature outside [0, 1]")
    return table


def _models(y_train: np.ndarray, seed: int) -> dict:
    weight = float((y_train == 0).sum()) / max(1, int((y_train == 1).sum()))
    return {
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=weight,
            random_state=seed, eval_metric="logloss",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=seed,
        ),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        ),
    }


def _score(y: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
    }


def _choose(train: pd.DataFrame) -> list[str]:
    y = train["label"].to_numpy()
    ranked = []
    for name in CANDIDATES:
        score = roc_auc_score(y, train[name])
        if score >= 0.60:
            ranked.append((score, name))
    ranked.sort(reverse=True)
    chosen = [name for _, name in ranked[:8]]
    if len(chosen) < 4:
        fallback = sorted(
            ((roc_auc_score(y, train[name]), name) for name in CANDIDATES),
            reverse=True,
        )
        chosen = [name for _, name in fallback[:4]]
    return chosen


def evaluate(table: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    users = table.loc[table.window.eq("4w") & table.arm.eq("unmasked"), ["user_id", "label"]]
    users = users.sort_values("user_id").reset_index(drop=True)
    if len(users) != 523 or int(users.label.sum()) != 91:
        raise ValueError(f"Expected 523 users and 91 positives, got {len(users)} and {int(users.label.sum())}")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(users, users.label))
    user_ids = users.user_id.to_numpy()
    results = {"fixed": {}, "nested": {}}
    predictions = []
    selections = []
    for window in ("4w", "8w", "12w", "full"):
        for arm in ARMS:
            subset = table.loc[table.window.eq(window) & table.arm.eq(arm)].set_index("user_id").loc[user_ids]
            y = subset.label.to_numpy(dtype=int)
            fixed_x = subset[list(FIXED)].to_numpy(dtype=float)
            fixed_prob = {name: np.zeros(len(y)) for name in ("xgboost", "random_forest", "logistic")}
            nested_prob = np.zeros(len(y))
            for fold, (train_idx, valid_idx) in enumerate(folds):
                for name, model in _models(y[train_idx], 42 + fold).items():
                    model.fit(fixed_x[train_idx], y[train_idx])
                    fixed_prob[name][valid_idx] = model.predict_proba(fixed_x[valid_idx])[:, 1]
                chosen = _choose(subset.iloc[train_idx])
                selections.append({"window": window, "arm": arm, "fold": fold, "features": chosen})
                nested = _models(y[train_idx], 42 + fold)["xgboost"]
                nested.fit(subset.iloc[train_idx][chosen], y[train_idx])
                nested_prob[valid_idx] = nested.predict_proba(subset.iloc[valid_idx][chosen])[:, 1]
            results["fixed"].setdefault(window, {})[arm] = {
                name: _score(y, probability) for name, probability in fixed_prob.items()
            }
            results["nested"].setdefault(window, {})[arm] = {"xgboost": _score(y, nested_prob)}
            for name, probability in fixed_prob.items():
                predictions.append(pd.DataFrame({
                    "user_id": user_ids, "label": y, "window": window, "arm": arm,
                    "model": name, "selection": "fixed", "probability": probability,
                }))
            predictions.append(pd.DataFrame({
                "user_id": user_ids, "label": y, "window": window, "arm": arm,
                "model": "xgboost", "selection": "nested", "probability": nested_prob,
            }))
    return results, pd.concat(predictions, ignore_index=True), pd.DataFrame(selections)


def univariate(table: pd.DataFrame) -> dict:
    out = {}
    for window in ("4w", "12w", "full"):
        out[window] = {}
        for arm in ARMS:
            subset = table.loc[table.window.eq(window) & table.arm.eq(arm)]
            y = subset.label.to_numpy()
            out[window][arm] = {}
            for name in CANDIDATES:
                out[window][arm][name] = _score(y, subset[name].to_numpy(dtype=float))
    return out


def main() -> int:
    writings = pd.read_csv(TEXT_PATH, usecols=["user_id", "date", "title", "body"])
    labels = pd.read_csv(LABEL_PATH, usecols=["user_id", "label"])
    table = build_table(writings, labels)
    results, predictions, selections = evaluate(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "features.csv", index=False)
    predictions.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    selections.to_csv(OUT_DIR / "feature_selection.csv", index=False)
    disclosed = int(table.loc[table.window.eq("full") & table.arm.eq("masked"), "n_disclosure_removed"].sum())
    payload = {
        "users": 523,
        "positives": 91,
        "fixed_features": list(FIXED),
        "candidates": list(CANDIDATES),
        "disclosure_writings": disclosed,
        "note": (
            "Fixed features were screened on the full cohort, so that CV is optimistic. "
            "Nested selection is fit inside each training fold. Neither number is a clinical result."
        ),
        "univariate": univariate(table),
        "models": results,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(results["fixed"]["4w"]["masked"], indent=2))
    print("nested 4w masked", json.dumps(results["nested"]["4w"]["masked"]))
    print(f"Wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
