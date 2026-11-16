"""Provisional prefix-only lexical signs for eRisk Task 2; not a classifier."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "erisk"
TOKEN_RE = re.compile(r"(?u)\b[a-z][a-z'-]*[a-z]\b|\b[a-z]\b")
SELF = frozenset({"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself"})
OTHER = frozenset({"we", "us", "our", "ours", "you", "your", "yours", "they", "them", "their"})
FAMILY_FRIEND = frozenset({
    "mom", "mother", "dad", "father", "parent", "parents", "brother", "sister",
    "family", "husband", "wife", "son", "daughter", "friend", "friends",
    "roommate", "coworker", "colleague", "classmate", "partner", "boyfriend", "girlfriend",
})
PROFANITY = frozenset({"fuck", "fucking", "fucked", "shit", "shitty", "damn", "asshole", "bitch", "bullshit"})
NEGATIVE = frozenset({
    "alone", "lonely", "hopeless", "worthless", "useless", "failure", "failed",
    "ashamed", "shame", "guilty", "miserable", "sad", "crying", "hurt", "empty",
})
CLINICAL = frozenset({
    "depression", "depressed", "antidepressant", "therapy", "therapist",
    "psychiatrist", "diagnosed", "diagnosis", "medication", "suicidal",
})
WORD_COUNTS = {
    "candidate_self_cnt": SELF,
    "candidate_other_cnt": OTHER,
    "candidate_family_friend_cnt": FAMILY_FRIEND,
    "candidate_profanity_cnt": PROFANITY,
    "candidate_clinical_term_cnt": CLINICAL,
}
POST_FLAGS = ("candidate_negative_self_post", "candidate_negative_family_post")


def writing_signs(writings: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "date", "kind", "title", "body"}
    if not required <= set(writings.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(writings.columns))}")
    data = writings[list(required)].copy()
    data["dt"] = pd.to_datetime(data["date"], utc=True)
    data = data.sort_values(["user_id", "dt"], kind="mergesort").reset_index(drop=True)
    data["t"] = data.groupby("user_id").cumcount() + 1

    rows = []
    previous_user = None
    seen_bigrams = set()
    for uid, title, body in data[["user_id", "title", "body"]].itertuples(index=False, name=None):
        if uid != previous_user:
            seen_bigrams = set()
            previous_user = uid
        tokens = TOKEN_RE.findall(f"{title if pd.notna(title) else ''} {body if pd.notna(body) else ''}".lower())
        token_set = set(tokens)
        bigrams = set(zip(tokens, tokens[1:]))
        row = {"n_tokens": len(tokens)}
        for name, words in WORD_COUNTS.items():
            row[name] = sum(token in words for token in tokens)
        row["candidate_negative_self_post"] = int(bool(token_set & SELF) and bool(token_set & NEGATIVE))
        row["candidate_negative_family_post"] = int(bool(token_set & SELF) and bool(token_set & NEGATIVE) and bool(token_set & FAMILY_FRIEND))
        row["candidate_bigram_overlap_frac"] = len(bigrams & seen_bigrams) / max(len(bigrams), 1)
        seen_bigrams.update(bigrams)
        rows.append(row)

    counts = pd.DataFrame(rows, index=data.index)
    data = pd.concat([data, counts], axis=1)
    groups = data.groupby("user_id", sort=False)
    total_tokens = groups["n_tokens"].cumsum().to_numpy(dtype=float)
    for name in WORD_COUNTS:
        total = groups[name].cumsum().to_numpy(dtype=float)
        data[name.replace("_cnt", "_rate_so_far")] = np.divide(
            total, total_tokens, out=np.zeros(len(data), dtype=float), where=total_tokens > 0
        )
    for name in POST_FLAGS + ("candidate_bigram_overlap_frac",):
        data[f"{name}_so_far"] = groups[name].cumsum() / data["t"]

    keep = ["user_id", "t", "date", "kind", "dt", "n_tokens", *WORD_COUNTS, *POST_FLAGS,
            "candidate_bigram_overlap_frac"]
    keep += [name.replace("_cnt", "_rate_so_far") for name in WORD_COUNTS]
    keep += [f"{name}_so_far" for name in POST_FLAGS + ("candidate_bigram_overlap_frac",)]
    return data[keep]


def user_windows(vectors: pd.DataFrame) -> pd.DataFrame:
    elapsed = (vectors["dt"] - vectors.groupby("user_id")["dt"].transform("first")).dt.total_seconds() / 86400
    columns = ["user_id", "label", "t"] + [c for c in vectors if c.endswith("_so_far")]
    windows = []
    for name, days in (("4w", 28), ("8w", 56), ("12w", 84), ("full", float("inf"))):
        selected = vectors.loc[elapsed <= days].groupby("user_id", sort=False).tail(1)[columns].copy()
        selected.insert(1, "window", name)
        windows.append(selected)
    return pd.concat(windows, ignore_index=True)


def main() -> None:
    text = pd.read_csv(PROC / "task2_target_text.csv")
    labels = pd.read_csv(PROC / "task2_user_l2_features.csv", usecols=["user_id", "label"])
    vectors = writing_signs(text).merge(labels, on="user_id", validate="many_to_one")
    if vectors["label"].isna().any() or len(vectors) != len(text):
        raise ValueError("Missing labels or writings")
    windows = user_windows(vectors)
    vectors.drop(columns="dt").to_csv(PROC / "task2_candidate_writing_vectors.csv", index=False)
    windows.to_csv(PROC / "task2_candidate_user_windows.csv", index=False)
    report = {
        "status": "provisional named proxies; no classifier or validated depression signs",
        "users": int(vectors["user_id"].nunique()),
        "writings": int(len(vectors)),
        "positive_users": int(labels["label"].sum()),
        "window_rows": {w: int((windows["window"] == w).sum()) for w in ("4w", "8w", "12w", "full")},
        "caveats": [
            "Clinical-term hits are for ablation, not evidence of first-person disclosure.",
            "Negative co-occurrence does not establish who or what a writing is about.",
            "Bigram overlap is lexical repetition, not verified rumination.",
            "No future writings or full-history observation length appear in feature columns.",
        ],
    }
    (PROC / "task2_candidate_signs_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidate vectors: {len(vectors):,} writings; {report['users']} users; no post text exported")


if __name__ == "__main__":
    main()
